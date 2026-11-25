"""
ZoneScore — ranking engine for search_zones.

Computes a weighted match score per zone and returns the top K, with per-criterion
scores and strongest/weakest criteria so the LLM can explain results without
inventing anything.

Importance levels (low / medium / high / must_have) are mapped to numeric weights
in criteria_config.IMPORTANCE_WEIGHTS — the LLM never emits raw floats.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .criteria_config import (
    CRITERIA_CONFIG, CRITERION_IDS, IMPORTANCE_WEIGHTS,
    DEFAULT_WEIGHT, MUST_HAVE_DEFAULT_MIN,
)
from .zone_data import ZONE_BY_ID, Zone


@dataclass
class MinScore:
    criterion_id: str
    min_value: float   # 0-100


@dataclass
class CriterionWeight:
    criterion_id: str
    level: str         # "low" | "medium" | "high" | "must_have"
    weight: float      # resolved numeric weight
    min_value: Optional[float] = None   # hard floor if must_have or user specified


@dataclass
class RankedZone:
    zone: Zone
    match_score: float            # 0-100 weighted average
    matched_criteria: Dict[str, float]   # criterion_id -> score used in computation
    strongest: Optional[str]
    weakest: Optional[str]
    reasons: List[str]            # human-readable explanation fragments
    excluded_reason: Optional[str] = None   # if filtered out by a hard constraint

    def to_dict(self) -> dict:
        return {
            "zone": self.zone.to_dict(include_scores=True),
            "match_score": round(self.match_score, 1),
            "matched_criteria": {k: round(v, 1) for k, v in self.matched_criteria.items()},
            "strongest": self.strongest,
            "weakest": self.weakest,
            "reasons": self.reasons,
        }


def _level_to_weight(level: str) -> float:
    """Map an importance level to a numeric weight; default to 'medium'."""
    return IMPORTANCE_WEIGHTS.get(level, IMPORTANCE_WEIGHTS["medium"])


def _resolve_weights(
    weights: Dict[str, str],
    mentioned_criteria: List[str],
) -> Dict[str, CriterionWeight]:
    """
    Build a {criterion_id: CriterionWeight} for all criteria.
    Criteria the user mentioned get their level-based weight; unmentioned criteria
    get DEFAULT_WEIGHT (so avg_score still factors in slightly).
    """
    resolved: Dict[str, CriterionWeight] = {}
    for cid in CRITERION_IDS:
        if cid in weights:
            level = weights[cid]
            w = _level_to_weight(level)
            min_v = MUST_HAVE_DEFAULT_MIN if level == "must_have" else None
            resolved[cid] = CriterionWeight(cid, level, w, min_v)
        else:
            # Unmentioned: small default weight so it still contributes weakly.
            resolved[cid] = CriterionWeight(cid, "default", DEFAULT_WEIGHT, None)
    return resolved


def search_zones(
    location_filter: Optional[str] = None,
    min_scores: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, str]] = None,
    top_k: int = 10,
    mentioned_criteria: Optional[List[str]] = None,
) -> Tuple[List[RankedZone], Dict[str, object]]:
    """
    Rank Tunisia zones by weighted criterion scores.

    Args:
        location_filter: Governorate or delegation name (free text, resolved via resolve_location).
        min_scores: Hard minimums {criterion_id: min_value (0-100)}. Zones below are excluded.
        weights: {criterion_id: importance_level} where level in {low, medium, high, must_have}.
        top_k: Maximum number of results to return.
        mentioned_criteria: Criterion ids the user actually mentioned (for scoring defaults).

    Returns:
        (ranked_zones, debug_info) where ranked_zones is a list of RankedZone
        (sorted by match_score desc) and debug_info contains filter stats.
    """
    min_scores = min_scores or {}
    weights = weights or {}
    mentioned_criteria = mentioned_criteria or []

    # ─── Resolve location filter ────────────────────────────────────────────
    matched_zone_ids: Optional[set[str]] = None
    location_debug: Optional[dict] = None
    if location_filter:
        from .location_resolver import resolve_location
        candidates = resolve_location(location_filter, limit=50, min_score=0.4)
        matched_zone_ids = {c["zone_id"] for c in candidates}
        location_debug = {
            "query": location_filter,
            "candidates": candidates[:10],
            "matched_zone_count": len(matched_zone_ids),
        }
        if not matched_zone_ids:
            return [], {"location": location_debug, "total_zones": len(ZONE_BY_ID), "filtered": 0}

    # ─── Ensure real API + GNN scores are loaded for target zones ───────────
    from .live_zones import ensure_zones_scored
    if matched_zone_ids is not None:
        candidate_zones = [ZONE_BY_ID[zid] for zid in matched_zone_ids if zid in ZONE_BY_ID]
    else:
        candidate_zones = list(ZONE_BY_ID.values())
    ensure_zones_scored(candidate_zones)

    # ─── Build criterion weights ────────────────────────────────────────────
    cw_map = _resolve_weights(weights, mentioned_criteria)
    total_weight = sum(cw.weight for cw in cw_map.values()) or 1.0

    # ─── Score and filter each zone ─────────────────────────────────────────
    ranked: List[RankedZone] = []
    excluded_count = 0

    for zone in candidate_zones:
        # Hard minimums: exclude if any criterion below its floor.
        excluded_reason = None
        for cid, min_val in min_scores.items():
            if cid in zone.scores and zone.scores[cid] < min_val:
                excluded_reason = f"{cid} below {min_val}"
                break
        # must_have criteria without explicit min: use default floor.
        if excluded_reason is None:
            for cid, cw in cw_map.items():
                if cw.level == "must_have" and cw.min_value is not None:
                    if cid in zone.scores and zone.scores[cid] < cw.min_value:
                        excluded_reason = f"{cid} must be >= {cw.min_value}"
                        break
        if excluded_reason is not None:
            excluded_count += 1
            ranked.append(RankedZone(
                zone=zone, match_score=0.0, matched_criteria={},
                strongest=None, weakest=None, reasons=[],
                excluded_reason=excluded_reason,
            ))
            continue

        # Weighted match score = Σ weight * score / Σ weights.
        weighted_sum = 0.0
        used_criteria: Dict[str, float] = {}
        for cid, cw in cw_map.items():
            score = zone.scores.get(cid)
            if score is None:
                continue
            weighted_sum += cw.weight * score
            used_criteria[cid] = score
        match_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        strongest = zone.strongest_criterion()
        weakest = zone.weakest_criterion()

        # Build explanation fragments.
        reasons: List[str] = []
        if strongest and strongest in used_criteria:
            reasons.append(f"Fort sur {CRITERIA_CONFIG[strongest].name} ({used_criteria[strongest]:.0f}/100)")
        if weakest and weakest in used_criteria and weakest != strongest:
            reasons.append(f"Plus faible sur {CRITERIA_CONFIG[weakest].name} ({used_criteria[weakest]:.0f}/100)")
        # Highlight must_have passes.
        for cid, cw in cw_map.items():
            if cw.level == "must_have" and cid in used_criteria:
                reasons.append(f"Répond au critère obligatoire {CRITERIA_CONFIG[cid].name} (≥{cw.min_value})")

        ranked.append(RankedZone(
            zone=zone, match_score=round(match_score, 1),
            matched_criteria=used_criteria, strongest=strongest, weakest=weakest,
            reasons=reasons,
        ))

    # Sort by match_score desc, then by avg_score desc as tiebreaker.
    ranked.sort(key=lambda rz: (rz.match_score, rz.zone.avg_score), reverse=True)

    top = ranked[:top_k]
    debug = {
        "location": location_debug,
        "total_zones": len(ZONE_BY_ID),
        "excluded": excluded_count,
        "returned": len(top),
        "weights_used": {cid: cw.weight for cid, cw in cw_map.items() if cw.level != "default"},
        "total_weight": round(total_weight, 2),
    }
    return top, debug
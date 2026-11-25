"""
Location resolver for Tunisian place names.

Tunisian place names have many spelling variants (Ariana / أريانة / Aryana,
La Marsa / المرسى).  This module uses fuzzy matching against zone names,
governorates and delegations to return candidate matches, avoiding silent
failures when the user's spelling doesn't exactly match our data.
"""

from typing import List, Optional, Tuple
from difflib import SequenceMatcher

from .zone_data import (
    ZONES, ZONE_BY_ID, ZONES_BY_GOVERNORATE, ZONES_BY_DELEGATION,
    GOVERNORATES, DELEGATIONS, _LOCATION_ALIASES,
)


def _norm(text: str) -> str:
    """Lowercase and strip accents/whitespace for comparison."""
    import unicodedata
    text = text.lower().strip()
    # Decompose unicode (e.g. é → e + combining accent) and drop combining marks.
    text = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    return text


def _has_arabic(text: str) -> bool:
    """Return True if the text contains Arabic-script characters."""
    import unicodedata
    for ch in text:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if "ARABIC" in name:
            return True
    return False


def _similarity(a: str, b: str) -> float:
    """Return a 0-1 similarity score between two strings."""
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _alias_in_text(alias: str, text_n: str) -> float:
    """Return a high score if any word of the alias matches any word of the text (either direction)."""
    alias_n = _norm(alias)
    if not alias_n:
        return 0.0
    text_words = [w for w in text_n.split() if w]
    alias_words = [w for w in alias_n.split() if w]
    best = 0.0
    for aw in alias_words:
        for tw in text_words:
            if aw in tw or tw in aw:
                shorter = min(len(aw), len(tw))
                longer = max(len(aw), len(tw))
                score = min(1.0, 0.8 + 0.2 * (shorter / longer))
                best = max(best, score)
    return best


def resolve_location(
    text: str,
    limit: int = 5,
    min_score: float = 0.45,
) -> List[dict]:
    """
    Fuzzy-match a free-text location string against Tunisian zones,
    governorates and delegations.

    Args:
        text: User's location query (any language / spelling).
        limit: Maximum number of candidates to return.
        min_score: Minimum similarity score (0-1) to include a candidate.

    Returns:
        List of dicts: {zone_id, name, governorate, delegation, score, match_type}
        sorted by score descending.  Empty list if nothing matches.
    """
    if not text or not text.strip():
        return []

    # Be more lenient with Arabic script (fuzzy matching is weaker across scripts).
    effective_min = min_score
    if _has_arabic(text):
        effective_min = min(0.35, min_score)

    query = _norm(text)
    candidates: List[Tuple[float, str, str]] = []  # (score, zone_id, match_type)

    # 1) Substring match: alias appears inside the user's text (handles extra words).
    for alias, zone_id in _LOCATION_ALIASES:
        score = _alias_in_text(alias, query)
        if score >= effective_min:
            match_type = "exact" if score >= 0.95 else "substring"
            candidates.append((score, zone_id, match_type))

    # 2) Fuzzy similarity fallback (whole-text comparison).
    if not candidates:
        for alias, zone_id in _LOCATION_ALIASES:
            score = _similarity(query, alias)
            if score >= effective_min:
                match_type = "exact" if score >= 0.9 else "fuzzy"
                candidates.append((score, zone_id, match_type))

    # 3) Governorate-level match: if user names a governorate, include all its zones.
    for gov in GOVERNORATES:
        if _similarity(query, gov) >= 0.75 or _alias_in_text(gov, query) > 0:
            for z in ZONES_BY_GOVERNORATE[gov]:
                if z.id not in [c[1] for c in candidates]:
                    candidates.append((0.8, z.id, "governorate"))

    # 4) Delegation-level match.
    for delg in DELEGATIONS:
        if _similarity(query, delg) >= 0.75 or _alias_in_text(delg, query) > 0:
            for z in ZONES_BY_DELEGATION[delg]:
                if z.id not in [c[1] for c in candidates]:
                    candidates.append((0.85, z.id, "delegation"))

    # Deduplicate by zone_id, keeping the highest score.
    best: dict[str, Tuple[float, str]] = {}
    for score, zone_id, match_type in candidates:
        if zone_id not in best or score > best[zone_id][0]:
            best[zone_id] = (score, match_type)

    results: List[dict] = []
    for zone_id, (score, match_type) in sorted(
        best.items(), key=lambda kv: kv[1][0], reverse=True
    ):
        z = ZONE_BY_ID[zone_id]
        results.append({
            "zone_id": z.id,
            "name": z.name,
            "name_ar": z.name_ar,
            "governorate": z.governorate,
            "governorate_ar": z.governorate_ar,
            "delegation": z.delegation,
            "delegation_ar": z.delegation_ar,
            "lat": z.lat,
            "lon": z.lon,
            "score": round(score, 2),
            "match_type": match_type,
        })
        if len(results) >= limit:
            break

    return results
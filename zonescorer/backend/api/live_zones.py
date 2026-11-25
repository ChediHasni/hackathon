"""
Live Zone Scorer for the ZoneScore Chat Agent.

Replaces synthetic zone scores in zone_data.py with real 7-criterion scores
computed from the live preprocessing modules (GEE, CDS, Overture, OSM) and
the ZoneGAT Graph Attention Network, persisted to zone_scores_cache.json.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import h3 as h3lib
import numpy as np
import pandas as pd

from .views import _import_pipeline, _load_criterion_frame
from .zone_data import ZONES, Zone

logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).resolve().parent / "zone_scores_cache.json"
CACHE_TTL_SECONDS = 24 * 3600  # 24 hours
_CACHE_LOCK = threading.Lock()


def _cell_bbox(cell: str, pad: float = 0.005) -> List[float]:
    """Return [minLon, minLat, maxLon, maxLat] tightly wrapping an H3 cell."""
    boundary = h3lib.cell_to_boundary(cell)
    lats = [lat for lat, _ in boundary]
    lons = [lon for _, lon in boundary]
    return [min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad]


def _overall_bbox(cells: List[str]) -> List[float]:
    """Return [minLon, minLat, maxLon, maxLat] covering all provided H3 cells."""
    all_lats, all_lons = [], []
    for cell in cells:
        for lat, lon in h3lib.cell_to_boundary(cell):
            all_lats.append(lat)
            all_lons.append(lon)
    return [min(all_lons) - 0.05, min(all_lats) - 0.05, max(all_lons) + 0.05, max(all_lats) + 0.05]


def _load_cache() -> Dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read zone cache: %s", exc)
        return {}


def _save_cache(cache: Dict[str, dict]) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def _apply_cached_entry(zone: Zone, entry: dict) -> None:
    zone.scores = {k: float(v) for k, v in entry.get("scores", {}).items()}
    zone.avg_score = float(entry.get("avg_score", 0.0))
    zone.data_sources = dict(entry.get("data_sources", {}))
    if entry.get("h3_cells"):
        zone.h3_cells = list(entry["h3_cells"])


def _score_per_zone_local(label: str, fn, zones: List[Zone]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Run a bbox-scoped fetcher (buildings, accessibility, transit) per zone's local cell."""
    frames = []
    sources: Dict[str, str] = {}
    for zone in zones:
        cell = zone.h3_cells[0]
        local_bbox = _cell_bbox(cell)
        df, src, _ = _load_criterion_frame(
            f"{label} ({zone.name})",
            fn,
            lambda f=fn, c=[cell], b=local_bbox: f(c, b, use_mock=True),
            [cell],
            local_bbox,
        )
        frames.append(df)
        sources[zone.id] = src
    combined = pd.concat(frames, ignore_index=True).set_index("h3_index")
    return combined, sources


def score_zones_live(zones: List[Zone], save_to_cache: bool = True) -> None:
    """Compute live 7-criterion + ZoneGAT scores for the given Tunisian zones."""
    if not zones:
        return

    (
        _,
        get_greenness, get_weather, get_buildings,
        get_air_quality, get_heat, get_accessibility, get_transit,
        run_inference, build_edge_index,
    ) = _import_pipeline()

    cells = [z.h3_cells[0] for z in zones]
    full_bbox = _overall_bbox(cells)

    # Batch GEE (sequential inside one worker for httplib2 safety) + CDS + local Overture/OSM
    def _run_gee():
        out = {}
        for key, label, fn in (
            ("greenness", "Greenness", get_greenness),
            ("air_quality", "Air quality", get_air_quality),
            ("heat", "Heat", get_heat),
        ):
            df, src, _ = _load_criterion_frame(
                label, fn, lambda f=fn: f(cells, full_bbox, use_mock=True), cells, full_bbox
            )
            out[key] = (df.set_index("h3_index"), {z.id: src for z in zones})
        return out

    with ThreadPoolExecutor(max_workers=5) as executor:
        gee_fut = executor.submit(_run_gee)
        clim_fut = executor.submit(
            lambda: _load_criterion_frame(
                "Climate", get_weather, lambda: get_weather(cells, full_bbox, use_mock=True), cells, full_bbox
            )
        )
        build_fut = executor.submit(_score_per_zone_local, "Buildings", get_buildings, zones)
        access_fut = executor.submit(_score_per_zone_local, "Accessibility", get_accessibility, zones)
        trans_fut = executor.submit(_score_per_zone_local, "Transit", get_transit, zones)

        gee_res = gee_fut.result()
        df_clim, clim_src, _ = clim_fut.result()
        df_clim = df_clim.set_index("h3_index")
        df_build, build_sources = build_fut.result()
        df_access, access_sources = access_fut.result()
        df_trans, trans_sources = trans_fut.result()

    df_green, green_sources = gee_res["greenness"]
    df_air, air_sources = gee_res["air_quality"]
    df_heat, heat_sources = gee_res["heat"]

    feature_matrix = np.column_stack([
        df_green.reindex(cells)["criterion_value_normalized"].fillna(0.5).values,
        df_clim.reindex(cells)["criterion_value_normalized"].fillna(0.5).values,
        df_build.reindex(cells)["criterion_value_normalized"].fillna(0.5).values,
        df_air.reindex(cells)["criterion_value_normalized"].fillna(0.5).values,
        df_heat.reindex(cells)["criterion_value_normalized"].fillna(0.5).values,
        df_access.reindex(cells)["criterion_value_normalized"].fillna(0.5).values,
        df_trans.reindex(cells)["criterion_value_normalized"].fillna(0.5).values,
    ]).astype(np.float32)

    edge_index = build_edge_index(cells)
    gat_scores = run_inference(feature_matrix, edge_index, cells)

    now = time.time()
    cache = _load_cache() if save_to_cache else {}

    for z in zones:
        cell = z.h3_cells[0]
        z.scores = {
            "greenness": round(float(df_green.at[cell, "criterion_value_normalized"] * 100), 1),
            "climate": round(float(df_clim.at[cell, "criterion_value_normalized"] * 100), 1),
            "building_svf": round(float(df_build.at[cell, "criterion_value_normalized"] * 100), 1),
            "air_quality": round(float(df_air.at[cell, "criterion_value_normalized"] * 100), 1),
            "heat": round(float(df_heat.at[cell, "criterion_value_normalized"] * 100), 1),
            "accessibility": round(float(df_access.at[cell, "criterion_value_normalized"] * 100), 1),
            "transit": round(float(df_trans.at[cell, "criterion_value_normalized"] * 100), 1),
        }
        z.avg_score = round(float(gat_scores.get(cell, 50.0)), 1)
        z.data_sources = {
            "greenness": green_sources[z.id],
            "climate": clim_src,
            "building_svf": build_sources[z.id],
            "air_quality": air_sources[z.id],
            "heat": heat_sources[z.id],
            "accessibility": access_sources[z.id],
            "transit": trans_sources[z.id],
        }
        cache[z.id] = {
            "scores": z.scores,
            "avg_score": z.avg_score,
            "data_sources": z.data_sources,
            "h3_cells": z.h3_cells,
            "updated_at": now,
        }

    if save_to_cache:
        _save_cache(cache)


def ensure_zones_scored(zones: List[Zone] | None = None, force_refresh: bool = False) -> None:
    """Populate Zone.scores & avg_score from live cache, scoring missing/stale zones live."""
    target_zones = list(zones) if zones is not None else list(ZONES)
    if not target_zones:
        return

    with _CACHE_LOCK:
        cache = _load_cache()
        now = time.time()
        to_score: List[Zone] = []

        for z in target_zones:
            entry = cache.get(z.id)
            if (
                not force_refresh
                and entry
                and entry.get("scores")
                and (now - float(entry.get("updated_at", 0))) < CACHE_TTL_SECONDS
            ):
                _apply_cached_entry(z, entry)
            else:
                to_score.append(z)

        if to_score:
            # Score all uncached zones together (and if most are uncached, score all ZONES
            # together so relative normalization like NDVI/SVF compares across Tunisia).
            batch = list(ZONES) if len(to_score) >= len(ZONES) // 2 else to_score
            score_zones_live(batch, save_to_cache=True)

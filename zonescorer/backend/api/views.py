"""
ZoneScore API Views
====================
Endpoints:
  GET  /api/health/   → {"status": "ok"}
  GET  /api/criteria/ → list of criteria names + weights
  POST /api/score/    → GeoJSON FeatureCollection of scored H3 hexagons
"""

import sys
import os
import logging
import numpy as np
from decouple import config

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

logger = logging.getLogger(__name__)


def _config_bool(name, default=False):
    raw = config(name, default=None)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {'1', 'true', 'yes', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'off'}:
        return False
    return default


USE_MOCK_DATA = _config_bool("USE_MOCK_DATA", default=False)
USE_LIVE_DATA = _config_bool("USE_LIVE_DATA", default=True)

# ─── Criteria Metadata ────────────────────────────────────────────────────────
CRITERIA = [
    {"id": "greenness",     "name": "Greenness",          "weight": 0.20, "unit": "NDVI",       "description": "Vegetation coverage from Sentinel-2"},
    {"id": "climate",       "name": "Climate Comfort",    "weight": 0.15, "unit": "Score",      "description": "Temperature & precipitation comfort (ERA5-Land)"},
    {"id": "building_svf",  "name": "Sky View Factor",    "weight": 0.10, "unit": "SVF",        "description": "Open sky availability derived from building heights"},
    {"id": "air_quality",   "name": "Air Quality",        "weight": 0.15, "unit": "Inverted",   "description": "PM2.5 & NO2 levels — lower pollution = higher score"},
    {"id": "heat",          "name": "Heat / LST",         "weight": 0.15, "unit": "Inverted°C", "description": "Land Surface Temperature — cooler = higher score"},
    {"id": "accessibility", "name": "Accessibility",      "weight": 0.15, "unit": "POI count",  "description": "Schools, hospitals, pharmacies, supermarkets (OSM)"},
    {"id": "transit",       "name": "Transit Access",     "weight": 0.10, "unit": "Stops",      "description": "Public transport stops per hexagon (Transitland)"},
]

CRITERION_IDS = [c["id"] for c in CRITERIA]
MAX_CELLS = 5000  # safety cap to prevent overloading the model
H3_RESOLUTION = 7
H3_FALLBACK_RESOLUTIONS = (7, 8, 9, 10)


# ─── Lazy imports to avoid heavy startup cost ─────────────────────────────────
def _import_pipeline():
    """Import all preprocessing modules + GNN inference."""
    # Add backend root to path so sibling packages resolve
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    import h3 as h3lib
    from preprocessing.greenness    import get_greenness
    from preprocessing.weather      import get_weather
    from preprocessing.buildings    import get_buildings
    from preprocessing.air_quality  import get_air_quality
    from preprocessing.heat         import get_heat
    from preprocessing.accessibility import get_accessibility
    from preprocessing.transit      import get_transit
    from gnn.inference              import run_inference, build_edge_index

    return (
        h3lib,
        get_greenness, get_weather, get_buildings,
        get_air_quality, get_heat, get_accessibility, get_transit,
        run_inference, build_edge_index,
    )


def _build_h3_cells(h3lib, bbox):
    """Return H3 cells for a bbox, trying finer resolutions if needed."""
    min_lon, min_lat, max_lon, max_lat = bbox
    bbox_outline = [
        (min_lat, min_lon),
        (min_lat, max_lon),
        (max_lat, max_lon),
        (max_lat, min_lon),
    ]
    h3_poly = h3lib.LatLngPoly(bbox_outline)

    for resolution in H3_FALLBACK_RESOLUTIONS:
        cells = list(h3lib.h3shape_to_cells(h3_poly, resolution))
        if cells:
            return cells, resolution

    return [], H3_RESOLUTION


def _load_criterion_frame(
    label: str,
    live_fn,
    mock_fn,
    h3_cells,
    bbox,
):
    """
    Try live data first when enabled, then fall back to synthetic data.

    The returned DataFrame is always indexed by h3_index in the caller.
    """
    if USE_MOCK_DATA or not USE_LIVE_DATA:
        return mock_fn()

    try:
        return live_fn(h3_cells, bbox, use_mock=False)
    except Exception as exc:
        logger.warning("%s live data failed, falling back to mock: %s", label, exc)
        return mock_fn()


# ─── Views ────────────────────────────────────────────────────────────────────

class HealthView(APIView):
    """GET /api/health/ — liveness probe."""

    def get(self, request):
        return Response({"status": "ok"})


class CriteriaView(APIView):
    """GET /api/criteria/ — returns scoring criteria metadata."""

    def get(self, request):
        return Response({"criteria": CRITERIA})


class ScoreView(APIView):
    """
    POST /api/score/
    Body: {"bbox": [minLon, minLat, maxLon, maxLat]}
    Returns: GeoJSON FeatureCollection of scored H3 hexagons.
    """

    def post(self, request):
        # ── Validate input ─────────────────────────────────────────────────────
        bbox = request.data.get("bbox")
        if not bbox or len(bbox) != 4:
            return Response(
                {"error": "bbox must be [minLon, minLat, maxLon, maxLat]"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            min_lon, min_lat, max_lon, max_lat = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return Response(
                {"error": "bbox values must be numeric"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if min_lon >= max_lon or min_lat >= max_lat:
            return Response(
                {"error": "Invalid bbox: min values must be less than max values"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # ── Load modules ───────────────────────────────────────────────────────
        try:
            (
                h3lib,
                get_greenness, get_weather, get_buildings,
                get_air_quality, get_heat, get_accessibility, get_transit,
                run_inference, build_edge_index,
            ) = _import_pipeline()
        except ImportError as e:
            logger.error("Pipeline import error: %s", e)
            return Response(
                {"error": f"Server dependency missing: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # ── Generate H3 cells covering the bbox (h3 v4 API) ────────────────────
        # LatLngPoly expects (lat, lng) tuples — note the order
        try:
            bbox_outline = [
                (min_lat, min_lon),
                (min_lat, max_lon),
                (max_lat, max_lon),
                (max_lat, min_lon),
            ]
            h3_poly = h3lib.LatLngPoly(bbox_outline)
            h3_cells = []
            h3_resolution = H3_RESOLUTION
            for resolution in H3_FALLBACK_RESOLUTIONS:
                h3_cells = list(h3lib.h3shape_to_cells(h3_poly, resolution))
                if h3_cells:
                    h3_resolution = resolution
                    break
            if not h3_cells:
                center_lat = (min_lat + max_lat) / 2.0
                center_lon = (min_lon + max_lon) / 2.0
                h3_cells = [h3lib.latlng_to_cell(center_lat, center_lon, 10)]
                h3_resolution = 10
        except Exception as e:
            return Response(
                {"error": f"H3 polyfill failed: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if not h3_cells:
            return Response(
                {"error": "No H3 cells found for this bounding box. Try a larger area."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if len(h3_cells) > MAX_CELLS:
            return Response(
                {"error": f"Area too large: {len(h3_cells)} cells (max {MAX_CELLS}). Please zoom in."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        logger.info(
            "Scoring %d H3 cells at H3 resolution %d for bbox %s",
            len(h3_cells),
            h3_resolution,
            bbox,
        )

        # Try live data first when enabled, then fall back to synthetic data per criterion.
        try:
            df_green = _load_criterion_frame(
                "Greenness",
                get_greenness,
                lambda: get_greenness(h3_cells, bbox, use_mock=True),
                h3_cells,
                bbox,
            ).set_index('h3_index')
            df_clim = _load_criterion_frame(
                "Climate",
                get_weather,
                lambda: get_weather(h3_cells, bbox, use_mock=True),
                h3_cells,
                bbox,
            ).set_index('h3_index')
            df_build = _load_criterion_frame(
                "Buildings",
                get_buildings,
                lambda: get_buildings(h3_cells, bbox, use_mock=True),
                h3_cells,
                bbox,
            ).set_index('h3_index')
            df_air = _load_criterion_frame(
                "Air quality",
                get_air_quality,
                lambda: get_air_quality(h3_cells, bbox, use_mock=True),
                h3_cells,
                bbox,
            ).set_index('h3_index')
            df_heat = _load_criterion_frame(
                "Heat",
                get_heat,
                lambda: get_heat(h3_cells, bbox, use_mock=True),
                h3_cells,
                bbox,
            ).set_index('h3_index')
            df_access = _load_criterion_frame(
                "Accessibility",
                get_accessibility,
                lambda: get_accessibility(h3_cells, bbox, use_mock=True),
                h3_cells,
                bbox,
            ).set_index('h3_index')
            df_trans = _load_criterion_frame(
                "Transit",
                get_transit,
                lambda: get_transit(h3_cells, bbox, use_mock=True),
                h3_cells,
                bbox,
            ).set_index('h3_index')
        except Exception as e:
            logger.exception("Preprocessing error")
            return Response(
                {"error": f"Preprocessing failed: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # ── Build feature matrix [N, 7] ────────────────────────────────────────
        feature_matrix = np.column_stack([
            df_green.reindex(h3_cells)['criterion_value_normalized'].fillna(0.5).values,
            df_clim.reindex(h3_cells)['criterion_value_normalized'].fillna(0.5).values,
            df_build.reindex(h3_cells)['criterion_value_normalized'].fillna(0.5).values,
            df_air.reindex(h3_cells)['criterion_value_normalized'].fillna(0.5).values,
            df_heat.reindex(h3_cells)['criterion_value_normalized'].fillna(0.5).values,
            df_access.reindex(h3_cells)['criterion_value_normalized'].fillna(0.5).values,
            df_trans.reindex(h3_cells)['criterion_value_normalized'].fillna(0.5).values,
        ]).astype(np.float32)

        # ── Build edge index ───────────────────────────────────────────────────
        try:
            edge_index = build_edge_index(h3_cells)
        except Exception as e:
            logger.exception("Edge index build error")
            return Response(
                {"error": f"Graph construction failed: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # ── Run GAT inference ──────────────────────────────────────────────────
        try:
            scores = run_inference(feature_matrix, edge_index, h3_cells)
        except Exception as e:
            logger.exception("Inference error")
            return Response(
                {"error": f"Model inference failed: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # ── Build GeoJSON FeatureCollection ────────────────────────────────────
        features = []
        for cell in h3_cells:
            score = scores.get(cell, 50.0)
            boundary = h3lib.cell_to_boundary(cell)  # list of (lat, lng)
            # GeoJSON uses [lng, lat]
            coords = [[lng, lat] for lat, lng in boundary]
            coords.append(coords[0])  # close ring

            # Per-criterion raw values for the sidebar radar chart
            cell_criteria = {
                "greenness":     round(float(df_green.at[cell,  'criterion_value_normalized'] * 100), 1),
                "climate":       round(float(df_clim.at[cell,   'criterion_value_normalized'] * 100), 1),
                "building_svf":  round(float(df_build.at[cell,  'criterion_value_normalized'] * 100), 1),
                "air_quality":   round(float(df_air.at[cell,    'criterion_value_normalized'] * 100), 1),
                "heat":          round(float(df_heat.at[cell,   'criterion_value_normalized'] * 100), 1),
                "accessibility": round(float(df_access.at[cell, 'criterion_value_normalized'] * 100), 1),
                "transit":       round(float(df_trans.at[cell,  'criterion_value_normalized'] * 100), 1),
            }

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
                "properties": {
                    "h3_index": cell,
                    "score": round(score, 1),
                    "criteria": cell_criteria,
                },
            })

        geojson = {
            "type": "FeatureCollection",
            "features": features,
        }
        return Response(geojson)

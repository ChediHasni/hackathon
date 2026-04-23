"""
Transit Access Preprocessing Module
======================================
Criterion: Number of transit stops per H3 cell
Live source: Transitland API (GTFS aggregator)
Mock: Realistic random stop counts per cell

Output DataFrame columns: [h3_index, criterion_value_normalized]
"""

import numpy as np
import pandas as pd
import requests
from decouple import config


TRANSITLAND_BASE_URL = "https://transit.land/api/v2"
MAX_STOP_COUNT = 15  # 95th percentile for normalization


def get_transit(h3_cells: list[str], bbox: list[float], use_mock: bool = True) -> pd.DataFrame:
    """
    Compute transit accessibility score per H3 cell.

    Args:
        h3_cells: List of H3 cell indices.
        bbox: [minLon, minLat, maxLon, maxLat]
        use_mock: If True, return realistic random data.

    Returns:
        DataFrame with columns [h3_index, criterion_value_normalized]
    """
    if use_mock:
        return _mock_transit(h3_cells)
    else:
        return _live_transit(h3_cells, bbox)


def _mock_transit(h3_cells: list[str]) -> pd.DataFrame:
    """
    Mock transit stop counts.
    Transit hubs: 10–15 stops; average urban: 3–8; suburban: 0–3.
    """
    rng = np.random.default_rng(seed=37)
    n = len(h3_cells)

    zone_type = rng.choice(['hub', 'urban', 'suburban', 'rural'], size=n, p=[0.05, 0.35, 0.40, 0.20])
    counts = np.where(
        zone_type == 'hub', rng.integers(10, MAX_STOP_COUNT + 1, n),
        np.where(zone_type == 'urban', rng.integers(3, 9, n),
                 np.where(zone_type == 'suburban', rng.integers(0, 4, n),
                          np.zeros(n)))
    ).astype(float)

    normalized = np.log1p(counts) / np.log1p(MAX_STOP_COUNT)
    normalized = np.clip(normalized, 0.0, 1.0)

    return pd.DataFrame({
        'h3_index': h3_cells,
        'criterion_value_normalized': normalized,
    })


def _live_transit(h3_cells: list[str], bbox: list[float]) -> pd.DataFrame:
    """
    Fetch transit stops from Transitland API v2.
    Requires TRANSITLAND_API_KEY in .env.
    """
    api_key = config('TRANSITLAND_API_KEY', default='')
    min_lon, min_lat, max_lon, max_lat = bbox

    try:
        import h3 as h3lib
        from shapely.geometry import Polygon, Point
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}")

    headers = {'apikey': api_key}
    params = {
        'bbox': f"{min_lon},{min_lat},{max_lon},{max_lat}",
        'per_page': 1000,
        'format': 'geojson',
    }

    try:
        response = requests.get(
            f"{TRANSITLAND_BASE_URL}/stops",
            params=params,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        stops = data.get('stops', [])
    except Exception:
        return _mock_transit(h3_cells)

    # Map each stop to its H3 cell
    stop_counts: dict[str, int] = {cell: 0 for cell in h3_cells}
    for stop in stops:
        try:
            lon = stop['geometry']['coordinates'][0]
            lat = stop['geometry']['coordinates'][1]
            cell = h3lib.latlng_to_cell(lat, lon, 7)
            if cell in stop_counts:
                stop_counts[cell] += 1
        except Exception:
            continue

    counts = np.array([float(stop_counts[cell]) for cell in h3_cells])
    normalized = np.log1p(counts) / np.log1p(MAX_STOP_COUNT)
    normalized = np.clip(normalized, 0.0, 1.0)

    return pd.DataFrame({
        'h3_index': h3_cells,
        'criterion_value_normalized': normalized,
    })

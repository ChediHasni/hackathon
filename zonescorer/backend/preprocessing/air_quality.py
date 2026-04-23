"""
Air Quality Preprocessing Module
===================================
Criterion: PM2.5 + NO2 concentrations (lower = better)
Live source: CAMS (Copernicus Atmosphere) via Google Earth Engine
Mock: Realistic random PM2.5 [2–80 µg/m³] and NO2 [5–200 µg/m³]

Output DataFrame columns: [h3_index, criterion_value_normalized]
Note: Values are INVERTED — lower pollution = higher score.
"""

import numpy as np
import pandas as pd
from decouple import config


# WHO guideline thresholds
PM25_MIN, PM25_MAX = 2.0, 80.0    # µg/m³
NO2_MIN, NO2_MAX = 5.0, 200.0     # µg/m³


def get_air_quality(h3_cells: list[str], bbox: list[float], use_mock: bool = True) -> pd.DataFrame:
    """
    Compute air quality score per H3 cell (inverted: lower pollution = higher score).

    Args:
        h3_cells: List of H3 cell indices.
        bbox: [minLon, minLat, maxLon, maxLat]
        use_mock: If True, return realistic random data.

    Returns:
        DataFrame with columns [h3_index, criterion_value_normalized]
    """
    if use_mock:
        return _mock_air_quality(h3_cells)
    else:
        return _live_air_quality(h3_cells, bbox)


def _pollution_to_score(pm25: float, no2: float) -> float:
    """Convert pollution levels to a quality score [0, 1] (inverted)."""
    pm25_norm = (pm25 - PM25_MIN) / (PM25_MAX - PM25_MIN)
    no2_norm = (no2 - NO2_MIN) / (NO2_MAX - NO2_MIN)
    # Combined pollution index → invert for quality score
    pollution_index = 0.6 * pm25_norm + 0.4 * no2_norm
    return float(np.clip(1.0 - pollution_index, 0.0, 1.0))


def _mock_air_quality(h3_cells: list[str]) -> pd.DataFrame:
    """
    Mock realistic air quality data.
    Urban hot spots get higher pollution; rural areas lower.
    """
    rng = np.random.default_rng(seed=99)
    n = len(h3_cells)

    # Bimodal: clean areas + polluted urban zones
    urban_mask = rng.random(n) < 0.35
    pm25 = np.where(
        urban_mask,
        rng.uniform(30.0, 80.0, n),    # urban
        rng.uniform(2.0, 20.0, n),     # rural/suburban
    )
    no2 = np.where(
        urban_mask,
        rng.uniform(80.0, 200.0, n),
        rng.uniform(5.0, 40.0, n),
    )

    scores = np.array([_pollution_to_score(p, n2) for p, n2 in zip(pm25, no2)])

    return pd.DataFrame({
        'h3_index': h3_cells,
        'criterion_value_normalized': scores,
    })


def _live_air_quality(h3_cells: list[str], bbox: list[float]) -> pd.DataFrame:
    """
    Fetch PM2.5 and NO2 from CAMS via Google Earth Engine.
    """
    try:
        import ee
        import h3 as h3lib
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}")

    gee_service_account = config('GEE_SERVICE_ACCOUNT', default='')
    gee_key_file = config('GEE_KEY_FILE', default='')
    credentials = ee.ServiceAccountCredentials(gee_service_account, gee_key_file)
    ee.Initialize(credentials)

    min_lon, min_lat, max_lon, max_lat = bbox
    region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    # CAMS Global Reanalysis (EAC4) — PM2.5 surrogate: total aerosol optical depth
    # Using Sentinel-5P for NO2
    s5p_no2 = (
        ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2')
        .filterBounds(region)
        .filterDate(ee.Date.now().advance(-3, 'month'), ee.Date.now())
        .select('NO2_column_number_density')
        .median()
        .multiply(1e6)  # mol/m² → µmol/m² proxy
    )

    # PM2.5 from Sentinel-5P aerosol
    s5p_aer = (
        ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_AER_AI')
        .filterBounds(region)
        .filterDate(ee.Date.now().advance(-3, 'month'), ee.Date.now())
        .select('absorbing_aerosol_index')
        .median()
    )

    records = []
    for cell in h3_cells:
        boundary = h3lib.cell_to_boundary(cell)
        coords = [[lng, lat] for lat, lng in boundary]
        coords.append(coords[0])
        poly = ee.Geometry.Polygon(coords)

        no2_val = s5p_no2.reduceRegion(ee.Reducer.mean(), poly, 1000).get('NO2_column_number_density').getInfo() or 50
        pm25_proxy = s5p_aer.reduceRegion(ee.Reducer.mean(), poly, 1000).get('absorbing_aerosol_index').getInfo() or 10
        pm25_val = max(2.0, min(80.0, float(pm25_proxy) * 10))
        score = _pollution_to_score(float(pm25_val), float(no2_val))
        records.append({'h3_index': cell, 'criterion_value_normalized': score})

    return pd.DataFrame(records)

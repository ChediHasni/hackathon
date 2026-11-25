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
    from datetime import datetime, timedelta, timezone
    import h3 as h3lib
    from preprocessing.gee_client import build_h3_feature_collection, initialize_gee

    ee = initialize_gee()

    fc = build_h3_feature_collection(ee, h3lib, h3_cells)

    end_dt = datetime.now(timezone.utc)
    start_str = (end_dt - timedelta(days=90)).strftime('%Y-%m-%d')
    end_str = end_dt.strftime('%Y-%m-%d')

    # Using Sentinel-5P for NO2
    s5p_no2 = (
        ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2')
        .filterBounds(fc)
        .filterDate(start_str, end_str)
        .select('NO2_column_number_density')
        .median()
        .multiply(1e6)  # mol/m² → µmol/m² proxy
    )

    # PM2.5 proxy from Sentinel-5P aerosol
    s5p_aer = (
        ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_AER_AI')
        .filterBounds(fc)
        .filterDate(start_str, end_str)
        .select('absorbing_aerosol_index')
        .median()
    )

    no2_res = s5p_no2.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=1000).getInfo()
    aer_res = s5p_aer.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=1000).getInfo()

    no2_by_cell = {f['properties']['h3_index']: f['properties'].get('mean') for f in no2_res.get('features', [])}
    aer_by_cell = {f['properties']['h3_index']: f['properties'].get('mean') for f in aer_res.get('features', [])}

    records = []
    for cell in h3_cells:
        no2_val = no2_by_cell.get(cell)
        no2_val = float(no2_val) if no2_val is not None else 50.0
        pm25_proxy = aer_by_cell.get(cell)
        pm25_proxy = float(pm25_proxy) if pm25_proxy is not None else 1.0
        pm25_val = max(2.0, min(80.0, pm25_proxy * 10.0))
        score = _pollution_to_score(pm25_val, no2_val)
        records.append({'h3_index': cell, 'criterion_value_normalized': score})

    return pd.DataFrame(records)

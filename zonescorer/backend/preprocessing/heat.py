"""
Heat / Land Surface Temperature Preprocessing Module
======================================================
Criterion: LST (Land Surface Temperature) from Landsat 8/9 Band 10
Live source: Landsat 8/9 via Google Earth Engine
Mock: Realistic random LST in [15°C, 45°C]
Note: Values are INVERTED — lower LST = cooler = higher score.

Output DataFrame columns: [h3_index, criterion_value_normalized]
"""

import numpy as np
import pandas as pd
from decouple import config


LST_MIN = 15.0   # °C — comfortable
LST_MAX = 45.0   # °C — extreme heat


def get_heat(h3_cells: list[str], bbox: list[float], use_mock: bool = True) -> pd.DataFrame:
    """
    Compute Land Surface Temperature score per H3 cell (inverted: lower = better).

    Args:
        h3_cells: List of H3 cell indices.
        bbox: [minLon, minLat, maxLon, maxLat]
        use_mock: If True, return realistic random data.

    Returns:
        DataFrame with columns [h3_index, criterion_value_normalized]
    """
    if use_mock:
        return _mock_heat(h3_cells)
    else:
        return _live_heat(h3_cells, bbox)


def _lst_to_score(lst_celsius: float) -> float:
    """Invert LST: cooler areas score higher."""
    normalized = (lst_celsius - LST_MIN) / (LST_MAX - LST_MIN)
    return float(np.clip(1.0 - normalized, 0.0, 1.0))


def _mock_heat(h3_cells: list[str]) -> pd.DataFrame:
    """
    Mock LST values.
    Urban heat islands: 35–45°C; suburban: 25–35°C; rural/water: 15–25°C.
    """
    rng = np.random.default_rng(seed=55)
    n = len(h3_cells)

    zone_type = rng.choice(['urban', 'suburban', 'rural'], size=n, p=[0.25, 0.45, 0.30])
    lst = np.where(
        zone_type == 'urban', rng.uniform(35.0, 45.0, n),
        np.where(zone_type == 'suburban', rng.uniform(25.0, 35.0, n),
                 rng.uniform(15.0, 28.0, n))
    )

    scores = np.array([_lst_to_score(t) for t in lst])

    return pd.DataFrame({
        'h3_index': h3_cells,
        'criterion_value_normalized': scores,
    })


def _live_heat(h3_cells: list[str], bbox: list[float]) -> pd.DataFrame:
    """
    Fetch LST from Landsat 8/9 Band 10 via Google Earth Engine.
    Converts thermal infrared (Band 10, K) to Celsius.
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

    # Landsat 8/9 Collection 2 Level 2
    landsat = (
        ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
        .filterBounds(region)
        .filterDate(ee.Date.now().advance(-12, 'month'), ee.Date.now())
        .filter(ee.Filter.lt('CLOUD_COVER', 20))
        .select('ST_B10')
        .median()
    )

    # Convert to Celsius: DN → K → °C
    # ST_B10 scale factor: 0.00341802, offset: 149.0 (Kelvin)
    lst_celsius = landsat.multiply(0.00341802).add(149.0).subtract(273.15)

    records = []
    for cell in h3_cells:
        boundary = h3lib.cell_to_boundary(cell)
        coords = [[lng, lat] for lat, lng in boundary]
        coords.append(coords[0])
        poly = ee.Geometry.Polygon(coords)

        val = lst_celsius.reduceRegion(ee.Reducer.mean(), poly, 30).get('ST_B10').getInfo()
        lst_val = float(val) if val is not None else 25.0
        score = _lst_to_score(lst_val)
        records.append({'h3_index': cell, 'criterion_value_normalized': score})

    return pd.DataFrame(records)

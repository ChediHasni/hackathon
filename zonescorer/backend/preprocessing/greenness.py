"""
Greenness Preprocessing Module
================================
Criterion: NDVI (Normalized Difference Vegetation Index)
Live source: Sentinel-2 via Google Earth Engine Python API (ee)
Mock: Realistic random NDVI values in [0.1, 0.8]

Output DataFrame columns: [h3_index, criterion_value_normalized]
"""

import numpy as np
import pandas as pd
from decouple import config


def get_greenness(h3_cells: list[str], bbox: list[float], use_mock: bool = True) -> pd.DataFrame:
    """
    Compute mean NDVI per H3 cell.

    Args:
        h3_cells: List of H3 cell indices (resolution 7).
        bbox: [minLon, minLat, maxLon, maxLat]
        use_mock: If True, return realistic random data without API calls.

    Returns:
        DataFrame with columns [h3_index, criterion_value_normalized]
    """
    if use_mock:
        return _mock_greenness(h3_cells)
    else:
        return _live_greenness(h3_cells, bbox)


def _mock_greenness(h3_cells: list[str]) -> pd.DataFrame:
    """
    Return realistic mock NDVI values.
    NDVI range: [-1, 1] in nature; vegetation typically [0.1, 0.8].
    We simulate a spatially-coherent distribution using random seeds per cell.
    """
    rng = np.random.default_rng(seed=42)
    n = len(h3_cells)

    # Realistic NDVI: urban areas ~0.1–0.3, suburban ~0.3–0.5, rural/forest ~0.5–0.8
    raw_ndvi = rng.uniform(0.1, 0.8, size=n)

    # Add spatial autocorrelation-like smoothing
    if n > 1:
        window = min(5, n)
        smoothed = np.convolve(raw_ndvi, np.ones(window) / window, mode='same')
    else:
        smoothed = raw_ndvi

    # Min-max normalize to [0, 1]
    ndvi_min, ndvi_max = 0.1, 0.8
    normalized = (smoothed - ndvi_min) / (ndvi_max - ndvi_min)
    normalized = np.clip(normalized, 0.0, 1.0)

    return pd.DataFrame({
        'h3_index': h3_cells,
        'criterion_value_normalized': normalized,
    })


def _live_greenness(h3_cells: list[str], bbox: list[float]) -> pd.DataFrame:
    """
    Fetch real NDVI data from Google Earth Engine (Sentinel-2).
    Requires GEE_SERVICE_ACCOUNT and GEE_KEY_FILE in .env.
    """
    try:
        import ee
    except ImportError:
        raise ImportError("earthengine-api is required for live greenness data.")

    import h3 as h3lib

    gee_service_account = config('GEE_SERVICE_ACCOUNT', default='')
    gee_key_file = config('GEE_KEY_FILE', default='')

    credentials = ee.ServiceAccountCredentials(gee_service_account, gee_key_file)
    ee.Initialize(credentials)

    min_lon, min_lat, max_lon, max_lat = bbox
    region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    # Sentinel-2 Surface Reflectance, cloud-masked, last 6 months
    s2 = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(region)
        .filterDate(ee.Date.now().advance(-6, 'month'), ee.Date.now())
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .median()
    )

    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
    records = []

    for cell in h3_cells:
        boundary = h3lib.cell_to_boundary(cell)
        coords = [[lng, lat] for lat, lng in boundary]
        coords.append(coords[0])
        poly = ee.Geometry.Polygon(coords)
        mean_val = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=poly,
            scale=10
        ).get('NDVI').getInfo()

        if mean_val is None:
            mean_val = 0.3  # fallback
        records.append({'h3_index': cell, 'raw': float(mean_val)})

    df = pd.DataFrame(records)
    ndvi_min, ndvi_max = df['raw'].min(), df['raw'].max()
    if ndvi_max > ndvi_min:
        df['criterion_value_normalized'] = (df['raw'] - ndvi_min) / (ndvi_max - ndvi_min)
    else:
        df['criterion_value_normalized'] = 0.5

    return df[['h3_index', 'criterion_value_normalized']]

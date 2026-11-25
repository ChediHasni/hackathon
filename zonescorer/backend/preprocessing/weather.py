"""
Weather / Climate Preprocessing Module
========================================
Criterion: Mean Temperature + Precipitation
Live source: ERA5-Land via cdsapi (Copernicus Climate Data Store)
Mock: Realistic random temperature [5°C–35°C] and precipitation [0–200 mm/month]

Output DataFrame columns: [h3_index, criterion_value_normalized]
"""

import numpy as np
import pandas as pd
from zonescorer.env import config


# ─── Comfortable climate targets ─────────────────────────────────────────────
# Temperature sweet spot: 15–25°C → score 1.0; extremes → 0
# Precipitation sweet spot: 30–100 mm/month → score 1.0

TEMP_OPTIMAL_LOW = 15.0   # °C
TEMP_OPTIMAL_HIGH = 25.0  # °C
TEMP_MIN = -20.0
TEMP_MAX = 50.0

PRECIP_OPTIMAL_LOW = 30.0   # mm/month
PRECIP_OPTIMAL_HIGH = 100.0
PRECIP_MIN = 0.0
PRECIP_MAX = 300.0


def get_weather(h3_cells: list[str], bbox: list[float], use_mock: bool = True) -> pd.DataFrame:
    """
    Compute climate comfort score per H3 cell.

    Args:
        h3_cells: List of H3 cell indices.
        bbox: [minLon, minLat, maxLon, maxLat]
        use_mock: If True, return realistic random data.

    Returns:
        DataFrame with columns [h3_index, criterion_value_normalized]
    """
    if use_mock:
        return _mock_weather(h3_cells)
    else:
        return _live_weather(h3_cells, bbox)


def _climate_score(temp: float, precip: float) -> float:
    """Score climate comfort: 1.0 = ideal, 0.0 = extreme."""
    # Temperature score: triangular / trapezoidal
    if TEMP_OPTIMAL_LOW <= temp <= TEMP_OPTIMAL_HIGH:
        t_score = 1.0
    elif temp < TEMP_OPTIMAL_LOW:
        t_score = max(0.0, (temp - TEMP_MIN) / (TEMP_OPTIMAL_LOW - TEMP_MIN))
    else:
        t_score = max(0.0, (TEMP_MAX - temp) / (TEMP_MAX - TEMP_OPTIMAL_HIGH))

    # Precipitation score
    if PRECIP_OPTIMAL_LOW <= precip <= PRECIP_OPTIMAL_HIGH:
        p_score = 1.0
    elif precip < PRECIP_OPTIMAL_LOW:
        p_score = precip / PRECIP_OPTIMAL_LOW
    else:
        p_score = max(0.0, (PRECIP_MAX - precip) / (PRECIP_MAX - PRECIP_OPTIMAL_HIGH))

    return 0.6 * t_score + 0.4 * p_score


def _mock_weather(h3_cells: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(seed=7)
    n = len(h3_cells)

    temperatures = rng.uniform(5.0, 35.0, size=n)
    precipitations = rng.uniform(0.0, 200.0, size=n)

    scores = np.array([
        _climate_score(t, p) for t, p in zip(temperatures, precipitations)
    ])
    scores = np.clip(scores, 0.0, 1.0)

    return pd.DataFrame({
        'h3_index': h3_cells,
        'criterion_value_normalized': scores,
    })


def _live_weather(h3_cells: list[str], bbox: list[float]) -> pd.DataFrame:
    """
    Fetch ERA5-Land monthly data via cdsapi.
    Rasterize temperature + precipitation to H3 cells using XArray.
    """
    try:
        import cdsapi
        import xarray as xr
        import h3 as h3lib
    except ImportError as e:
        raise ImportError(f"Missing dependency for live weather data: {e}")

    cds_key = config('CDS_API_KEY', default='').strip()
    if not cds_key:
        raise ValueError("CDS_API_KEY is not configured in .env")
    c = cdsapi.Client(key=cds_key, url='https://cds.climate.copernicus.eu/api', retry_max=3, timeout=60)

    import os
    import tempfile
    import zipfile

    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
        tmp_path = tmp.name

    min_lon, min_lat, max_lon, max_lat = bbox
    # Pad small bounding boxes so at least one ERA5-Land 0.1° grid point is included
    if (max_lat - min_lat) < 0.25:
        mid_lat = (min_lat + max_lat) / 2.0
        min_lat, max_lat = mid_lat - 0.125, mid_lat + 0.125
    if (max_lon - min_lon) < 0.25:
        mid_lon = (min_lon + max_lon) / 2.0
        min_lon, max_lon = mid_lon - 0.125, mid_lon + 0.125

    extracted_path = None
    try:
        c.retrieve(
            'reanalysis-era5-land-monthly-means',
            {
                'product_type': 'monthly_averaged_reanalysis',
                'variable': ['2m_temperature', 'total_precipitation'],
                'year': '2023',
                'month': [str(m).zfill(2) for m in range(1, 13)],
                'time': '00:00',
                'area': [max_lat, min_lon, min_lat, max_lon],
                'data_format': 'netcdf',
                'download_format': 'unarchived',
            },
            tmp_path,
        )

        nc_path = tmp_path
        if zipfile.is_zipfile(tmp_path):
            with zipfile.ZipFile(tmp_path, 'r') as zf:
                nc_members = [m for m in zf.namelist() if m.endswith('.nc')]
                if not nc_members:
                    raise ValueError("CDS archive did not contain a .nc file")
                with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as nc_tmp:
                    nc_tmp.write(zf.read(nc_members[0]))
                    extracted_path = nc_tmp.name
                    nc_path = extracted_path

        with xr.open_dataset(nc_path, engine='netcdf4') as ds:
            scores = []
            for cell in h3_cells:
                lat, lon = h3lib.cell_to_latlng(cell)
                try:
                    t_sub = ds['t2m'].sel(latitude=lat, longitude=lon, method='nearest')
                    p_sub = ds['tp'].sel(latitude=lat, longitude=lon, method='nearest')
                    t_val = float(t_sub.mean().values)
                    p_val = float(p_sub.mean().values)
                    if np.isnan(t_val) or np.isnan(p_val):
                        raise ValueError("NaN at nearest grid point")
                except Exception:
                    t_val = float(ds['t2m'].mean().values)
                    p_val = float(ds['tp'].mean().values)
                temp_c = t_val - 273.15
                precip_mm = p_val * 1000 * 30  # m/day → mm/month approx
                scores.append(_climate_score(temp_c, precip_mm))
    finally:
        for path in (tmp_path, extracted_path):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    return pd.DataFrame({
        'h3_index': h3_cells,
        'criterion_value_normalized': np.clip(scores, 0.0, 1.0),
    })

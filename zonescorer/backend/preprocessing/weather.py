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
from decouple import config


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

    cds_key = config('CDS_API_KEY', default='')
    c = cdsapi.Client(key=cds_key, url='https://cds.climate.copernicus.eu/api/v2')

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
        tmp_path = tmp.name

    min_lon, min_lat, max_lon, max_lat = bbox
    c.retrieve(
        'reanalysis-era5-land-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': ['2m_temperature', 'total_precipitation'],
            'year': '2023',
            'month': [str(m).zfill(2) for m in range(1, 13)],
            'time': '00:00',
            'area': [max_lat, min_lon, min_lat, max_lon],
            'format': 'netcdf',
        },
        tmp_path,
    )

    ds = xr.open_dataset(tmp_path)
    temp_k = float(ds['t2m'].mean().values)
    temp_c = temp_k - 273.15
    precip_m = float(ds['tp'].mean().values)
    precip_mm = precip_m * 1000 * 30  # m/day → mm/month approx

    os.unlink(tmp_path)

    scores = []
    for cell in h3_cells:
        score = _climate_score(temp_c, precip_mm)
        scores.append(score)

    return pd.DataFrame({
        'h3_index': h3_cells,
        'criterion_value_normalized': np.clip(scores, 0.0, 1.0),
    })

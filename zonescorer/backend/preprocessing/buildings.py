"""
Building Heights / Sky View Factor Preprocessing Module
==========================================================
Criterion: Sky View Factor (SVF) derived from building heights
Live source: Overture Maps Buildings theme (GeoParquet)
SVF formula: SVF = 1 / (1 + mean_height / 20)
Mock: Realistic building heights in [0, 80] metres

Output DataFrame columns: [h3_index, criterion_value_normalized]
"""

import numpy as np
import pandas as pd


def get_buildings(h3_cells: list[str], bbox: list[float], use_mock: bool = True) -> pd.DataFrame:
    """
    Compute mean building height and Sky View Factor per H3 cell.

    Args:
        h3_cells: List of H3 cell indices.
        bbox: [minLon, minLat, maxLon, maxLat]
        use_mock: If True, return realistic random data.

    Returns:
        DataFrame with columns [h3_index, criterion_value_normalized]
        (higher SVF = more open sky = better score)
    """
    if use_mock:
        return _mock_buildings(h3_cells)
    else:
        return _live_buildings(h3_cells, bbox)


def _svf_from_height(mean_height: float) -> float:
    """Compute Sky View Factor from mean building height."""
    return 1.0 / (1.0 + mean_height / 20.0)


def _mock_buildings(h3_cells: list[str]) -> pd.DataFrame:
    """
    Mock building heights.
    Distribution: 60% low-rise (0–15m), 30% mid-rise (15–50m), 10% high-rise (50–80m).
    """
    rng = np.random.default_rng(seed=13)
    n = len(h3_cells)

    categories = rng.choice([0, 1, 2], size=n, p=[0.60, 0.30, 0.10])
    heights = np.where(
        categories == 0, rng.uniform(0, 15, size=n),
        np.where(categories == 1, rng.uniform(15, 50, size=n),
                 rng.uniform(50, 80, size=n))
    )

    svf_values = np.array([_svf_from_height(h) for h in heights])
    # SVF range: [1/(1+80/20), 1/(1+0/20)] = [0.2, 1.0] → min-max normalize
    svf_min, svf_max = 1.0 / (1.0 + 80.0 / 20.0), 1.0
    normalized = (svf_values - svf_min) / (svf_max - svf_min)
    normalized = np.clip(normalized, 0.0, 1.0)

    return pd.DataFrame({
        'h3_index': h3_cells,
        'criterion_value_normalized': normalized,
    })


def _live_buildings(h3_cells: list[str], bbox: list[float]) -> pd.DataFrame:
    """
    Download Overture Maps Buildings theme GeoParquet and compute mean height.
    Requires geopandas and pyarrow.
    """
    try:
        import geopandas as gpd
        import h3 as h3lib
        from shapely.geometry import box as shapely_box
    except ImportError as e:
        raise ImportError(f"Missing dependency for live buildings data: {e}")

    min_lon, min_lat, max_lon, max_lat = bbox

    # Overture Maps S3 endpoint (public, no auth required for parquet)
    overture_url = (
        "s3://overturemaps-us-west-2/release/2024-07-22.0/theme=buildings/type=building/"
    )

    try:
        import pyarrow.dataset as ds
        import pyarrow.compute as pc

        dataset = ds.dataset(overture_url, format='parquet', filesystem=None)
        # Spatial filter by bbox
        filt = (
            (pc.field('bbox', 'minx') < max_lon) &
            (pc.field('bbox', 'maxx') > min_lon) &
            (pc.field('bbox', 'miny') < max_lat) &
            (pc.field('bbox', 'maxy') > min_lat)
        )
        table = dataset.to_table(filter=filt, columns=['geometry', 'height'])
        gdf = gpd.GeoDataFrame.from_arrow(table)
    except Exception:
        # Fallback to mock if Overture is unavailable
        return _mock_buildings(h3_cells)

    records = []
    for cell in h3_cells:
        boundary = h3lib.cell_to_boundary(cell)
        coords = [[lng, lat] for lat, lng in boundary]
        from shapely.geometry import Polygon
        poly = Polygon(coords)
        clipped = gdf[gdf.geometry.intersects(poly)]
        mean_h = clipped['height'].dropna().mean() if len(clipped) > 0 else 5.0
        if np.isnan(mean_h):
            mean_h = 5.0
        svf = _svf_from_height(mean_h)
        records.append({'h3_index': cell, 'svf': svf})

    df = pd.DataFrame(records)
    svf_min, svf_max = df['svf'].min(), df['svf'].max()
    if svf_max > svf_min:
        df['criterion_value_normalized'] = (df['svf'] - svf_min) / (svf_max - svf_min)
    else:
        df['criterion_value_normalized'] = 0.5

    return df[['h3_index', 'criterion_value_normalized']]

"""
Accessibility (POI) Preprocessing Module
==========================================
Criterion: Count of key amenities near each H3 cell
Live source: OpenStreetMap via osmnx
POI types: schools, hospitals, pharmacies, supermarkets
Mock: Realistic random POI counts per cell

Output DataFrame columns: [h3_index, criterion_value_normalized]
"""

import numpy as np
import pandas as pd


# Tags to query from OSM
POI_TAGS = {
    'amenity': ['school', 'hospital', 'pharmacy'],
    'shop': ['supermarket'],
}

# Expected max POI count for normalization (95th percentile typical urban density)
MAX_POI_COUNT = 20


def get_accessibility(h3_cells: list[str], bbox: list[float], use_mock: bool = True) -> pd.DataFrame:
    """
    Compute accessibility score (POI density) per H3 cell.

    Args:
        h3_cells: List of H3 cell indices.
        bbox: [minLon, minLat, maxLon, maxLat]
        use_mock: If True, return realistic random data.

    Returns:
        DataFrame with columns [h3_index, criterion_value_normalized]
    """
    if use_mock:
        return _mock_accessibility(h3_cells)
    else:
        return _live_accessibility(h3_cells, bbox)


def _mock_accessibility(h3_cells: list[str]) -> pd.DataFrame:
    """
    Mock POI counts.
    Dense urban: 10–20, suburban: 3–10, rural: 0–3.
    """
    rng = np.random.default_rng(seed=21)
    n = len(h3_cells)

    zone_type = rng.choice(['urban', 'suburban', 'rural'], size=n, p=[0.30, 0.45, 0.25])
    counts = np.where(
        zone_type == 'urban', rng.integers(8, MAX_POI_COUNT + 1, n),
        np.where(zone_type == 'suburban', rng.integers(2, 10, n),
                 rng.integers(0, 4, n))
    ).astype(float)

    # Log-normalize: gives better dynamic range for sparse areas
    normalized = np.log1p(counts) / np.log1p(MAX_POI_COUNT)
    normalized = np.clip(normalized, 0.0, 1.0)

    return pd.DataFrame({
        'h3_index': h3_cells,
        'criterion_value_normalized': normalized,
    })


def _live_accessibility(h3_cells: list[str], bbox: list[float]) -> pd.DataFrame:
    """
    Query OpenStreetMap POIs via osmnx for each H3 cell.
    """
    try:
        import osmnx as ox
        import h3 as h3lib
        from shapely.geometry import Polygon, box as shapely_box
        import geopandas as gpd
    except ImportError as e:
        raise ImportError(f"Missing dependency for live accessibility data: {e}")

    min_lon, min_lat, max_lon, max_lat = bbox
    overall_bbox = (max_lat, min_lat, max_lon, min_lon)  # osmnx format: N,S,E,W

    # Download all POIs within bounding box at once
    try:
        pois_gdf = ox.features_from_bbox(
            north=max_lat, south=min_lat, east=max_lon, west=min_lon,
            tags={'amenity': ['school', 'hospital', 'pharmacy'],
                  'shop': ['supermarket']}
        )
    except Exception:
        return _mock_accessibility(h3_cells)

    pois_gdf = pois_gdf.to_crs(epsg=4326)

    records = []
    for cell in h3_cells:
        boundary = h3lib.cell_to_boundary(cell)
        coords = [[lng, lat] for lat, lng in boundary]
        poly = Polygon(coords)
        clipped = pois_gdf[pois_gdf.geometry.intersects(poly)]
        count = len(clipped)
        records.append({'h3_index': cell, 'count': float(count)})

    df = pd.DataFrame(records)
    df['criterion_value_normalized'] = np.clip(
        np.log1p(df['count']) / np.log1p(MAX_POI_COUNT), 0.0, 1.0
    )
    return df[['h3_index', 'criterion_value_normalized']]

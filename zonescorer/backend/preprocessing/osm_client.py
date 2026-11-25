"""Shared thread-safe OpenStreetMap (Overpass) fetcher via osmnx."""

import threading

_OSM_LOCK = threading.Lock()
_OVERPASS_ENDPOINTS = (
    'https://overpass-api.de/api',
    'https://overpass.private.coffee/api',
)


def fetch_osm_features(bbox: list[float], tags: dict):
    """
    Fetch OSM features within bbox=(min_lon, min_lat, max_lon, max_lat).
    Returns a GeoDataFrame in EPSG:4326, or None when OSM legitimately has 0 matches.
    Raises on transport/Overpass failures.
    """
    import osmnx as ox

    min_lon, min_lat, max_lon, max_lat = bbox
    with _OSM_LOCK:
        ox.settings.overpass_rate_limit = False
        ox.settings.requests_timeout = 10
        last_exc = None
        for endpoint in _OVERPASS_ENDPOINTS:
            ox.settings.overpass_url = endpoint
            try:
                gdf = ox.features_from_bbox(
                    bbox=(min_lon, min_lat, max_lon, max_lat),
                    tags=tags,
                )
                return gdf.to_crs(epsg=4326)
            except ox._errors.InsufficientResponseError:
                return None
            except Exception as exc:
                last_exc = exc
        raise last_exc

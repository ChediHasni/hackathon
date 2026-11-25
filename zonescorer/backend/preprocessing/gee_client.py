"""Shared Google Earth Engine initialization helper."""

from pathlib import Path
import threading
from zonescorer.env import ENV_ROOT, config

_GEE_LOCK = threading.Lock()


def initialize_gee():
    """Initialize Earth Engine with service account credentials from .env."""
    try:
        import ee
    except ImportError as exc:
        raise ImportError("earthengine-api is required for live Earth Engine data.") from exc

    with _GEE_LOCK:
        gee_service_account = config("GEE_SERVICE_ACCOUNT", default="").strip()
        gee_key_file = config("GEE_KEY_FILE", default="").strip()

        if not gee_service_account or not gee_key_file:
            raise ValueError("GEE_SERVICE_ACCOUNT and GEE_KEY_FILE must be set in .env")

        key_path = Path(gee_key_file)
        if not key_path.is_absolute() and not key_path.exists():
            candidate = ENV_ROOT / key_path
            if candidate.exists():
                key_path = candidate

        if not key_path.exists():
            raise FileNotFoundError(f"GEE key file not found: {key_path}")

        credentials = ee.ServiceAccountCredentials(gee_service_account, str(key_path))
        ee.Initialize(credentials)
        return ee


def build_h3_feature_collection(ee, h3lib, h3_cells: list[str]):
    """Build an Earth Engine FeatureCollection from H3 cell boundaries."""
    features = []
    for cell in h3_cells:
        boundary = h3lib.cell_to_boundary(cell)
        coords = [[lng, lat] for lat, lng in boundary]
        coords.append(coords[0])
        features.append(ee.Feature(ee.Geometry.Polygon(coords), {'h3_index': cell}))
    return ee.FeatureCollection(features)

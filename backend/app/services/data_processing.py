import numpy as np
import rasterio
from typing import Tuple

def load_geotiff(file_path: str) -> Tuple[np.ndarray, dict]:
    """
    Load a GeoTIFF image and return the image data and its geospatial profile.
    """
    with rasterio.open(file_path) as src:
        image = src.read()
        profile = src.profile
    return image, profile

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize and resize the image before model inference.
    Assumes standard Sentinel-2 MSI band setup.
    """
    # Dummy preprocessing logic for hackathon readiness
    processed = image.astype(np.float32) / 10000.0
    return processed

def get_lat_lon_from_pixel(transform, row: int, col: int) -> Tuple[float, float]:
    """
    Convert row, col to latitude and longitude using rasterio affine transform.
    """
    lon, lat = transform * (col, row)
    return lat, lon

def extract_patches(image: np.ndarray, patch_size: int = 256) -> list:
    """
    Extract patches from the large satellite image for model processing.
    """
    patches = []
    # Logic to divide image into patches goes here
    return patches

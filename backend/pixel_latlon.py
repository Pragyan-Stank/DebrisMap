"""
pixel_latlon.py
===============
Utility: Map any Sentinel-2 MARIDA patch pixel (row, col) to (latitude, longitude).

Usage:
    python pixel_latlon.py
    OR import as a module in your pipeline.
"""

import sys
import numpy as np
import rasterio
import pyproj
from rasterio.transform import xy as rasterio_xy
from pathlib import Path


def pixel_to_latlon(tif_path: str, row: int, col: int) -> tuple:
    """
    Convert a single pixel (row, col) in a GeoTIFF to (lat, lon).

    Parameters
    ----------
    tif_path : str  - Path to the .tif image
    row      : int  - Pixel row    (0 = top)
    col      : int  - Pixel column (0 = left)

    Returns
    -------
    (lat, lon) : tuple of float  [WGS-84 decimal degrees]
    """
    with rasterio.open(tif_path) as src:
        transformer = pyproj.Transformer.from_crs(
            src.crs, "EPSG:4326", always_xy=True
        )
        x, y = rasterio_xy(src.transform, row, col)
        lon, lat = transformer.transform(x, y)
    return float(lat), float(lon)


def full_pixel_grid(tif_path: str) -> tuple:
    """
    Compute the lat/lon of EVERY pixel in the patch.

    Returns
    -------
    lats : np.ndarray  shape (H, W)
    lons : np.ndarray  shape (H, W)
    """
    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        transformer = pyproj.Transformer.from_crs(
            src.crs, "EPSG:4326", always_xy=True
        )
        rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        xs, ys = rasterio_xy(src.transform, rows.ravel(), cols.ravel())
        lons, lats = transformer.transform(xs, ys)

    return (
        np.array(lats).reshape(H, W),
        np.array(lons).reshape(H, W),
    )


def patch_corners(tif_path: str) -> dict:
    """
    Return lat/lon of the four corners of the patch.
    """
    with rasterio.open(tif_path) as src:
        H, W = src.height - 1, src.width - 1
        transformer = pyproj.Transformer.from_crs(
            src.crs, "EPSG:4326", always_xy=True
        )
        corners = {}
        for name, r, c in [("top_left", 0, 0), ("top_right", 0, W),
                            ("bot_left", H, 0), ("bot_right", H, W)]:
            x, y   = rasterio_xy(src.transform, r, c)
            lo, la = transformer.transform(x, y)
            corners[name] = (round(la, 6), round(lo, 6))
    return corners


def pixel_coords_from_prediction(pred_mask: np.ndarray,
                                  tif_path: str,
                                  target_class: int = 1) -> list:
    """
    Given a U-Net prediction mask and target class, return a list of
    (lat, lon) pairs for every pixel predicted as that class.

    Parameters
    ----------
    pred_mask    : np.ndarray (H, W) - predicted class IDs from U-Net
    tif_path     : str               - matching source GeoTIFF for georef
    target_class : int               - MARIDA class ID (1 = Marine Debris)

    Returns
    -------
    List of (lat, lon) tuples
    """
    lats, lons = full_pixel_grid(tif_path)
    rows, cols = np.where(pred_mask == target_class)
    return [(float(lats[r, c]), float(lons[r, c])) for r, c in zip(rows, cols)]


# -------------------------------------------------------
#  MARIDA class legend
# -------------------------------------------------------
CLASS_NAMES = {
    0: "Unlabeled",       1: "Marine Debris",
    2: "Dense Sargassum", 3: "Sparse Sargassum",
    4: "Natural Organic", 5: "Ship",
    6: "Clouds",          7: "Marine Water",
    8: "Sediment Water",  9: "Foam",
   10: "Turbid Water",   11: "Shallow Water",
   12: "Waves",          13: "Cloud Shadows",
   14: "Wakes",          15: "Mixed Water",
}


# -------------------------------------------------------
#  Demo / CLI
# -------------------------------------------------------
if __name__ == "__main__":
    # Default demo patch
    DEMO_TIF = Path(r"c:\Users\omtil\Downloads\MARIDA\patches"
                    r"\S2_1-12-19_48MYU\S2_1-12-19_48MYU_0.tif")

    if len(sys.argv) == 4:
        tif_path = sys.argv[1]
        row      = int(sys.argv[2])
        col      = int(sys.argv[3])
    else:
        tif_path = str(DEMO_TIF)
        row, col = 128, 128

    print(f"\nPatch : {Path(tif_path).name}")
    lat, lon = pixel_to_latlon(tif_path, row, col)
    print(f"Pixel ({row}, {col})  ->  lat={lat:.6f}, lon={lon:.6f}")

    print("\nPatch corners:")
    for k, v in patch_corners(tif_path).items():
        print(f"  {k:12s}: lat={v[0]:.6f}, lon={v[1]:.6f}")

    # Build full grid and print stats
    lats, lons = full_pixel_grid(tif_path)
    print(f"\nFull 256x256 pixel grid:")
    print(f"  Lat range: {lats.min():.6f}  to  {lats.max():.6f}")
    print(f"  Lon range: {lons.min():.6f}  to  {lons.max():.6f}")
    print(f"  Pixel size (approx): "
          f"{abs(lats[0,0]-lats[1,0])*111000:.1f} m   x   "
          f"{abs(lons[0,0]-lons[0,1])*111000:.1f} m")

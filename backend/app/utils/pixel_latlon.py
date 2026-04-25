import rasterio
import numpy as np
from typing import Tuple, List

def pixel_to_latlon(tif_path: str, row: int, col: int) -> Tuple[float, float]:
    """
    Transforms pixel coordinates (row, col) to Geographic (lat, lon) coordinates
    using the bounding box / CRS embedded in the GeoTIFF.
    """
    with rasterio.open(tif_path) as src:
        lon, lat = src.transform * (col, row)
        crs = src.crs
        
        # If the CRS is UTM (meter-based, like EPSG:32748), map back to Global Lat/Lon
        if crs and crs.to_epsg() != 4326:
            from pyproj import Transformer
            # always_xy=True ensures output is (lon, lat)
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(lon, lat)
            
        return lat, lon

def pixel_coords_from_prediction(pred_mask: np.ndarray, tif_path: str, target_class: int = 1) -> List[Tuple[float, float]]:
    """
    Extracts all latitude and longitude pairs for pixels matching a specific class.
    target_class=1 corresponds to Marine Debris in the MARIDA dataset.
    """
    y_indices, x_indices = np.where(pred_mask == target_class)
    
    debris_coords = []
    with rasterio.open(tif_path) as src:
        transform = src.transform
        crs = src.crs
        
        # Setup PyProj transformer if the image projection is NOT standard WGS84
        transformer = None
        if crs and crs.to_epsg() != 4326:
            from pyproj import Transformer
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            
        for y, x in zip(y_indices, x_indices):
            x_coord, y_coord = transform * (x, y)
            
            if transformer:
                lon, lat = transformer.transform(x_coord, y_coord)
            else:
                lon, lat = x_coord, y_coord
                
            debris_coords.append((float(lat), float(lon)))
            
    return debris_coords

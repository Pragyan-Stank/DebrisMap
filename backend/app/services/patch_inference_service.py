import numpy as np
from app.models.inference import get_inferencer
from app.services.fdi import calculate_fdi

# Lower threshold for live satellite data (real reflectance varies from training data)
LIVE_THRESHOLD = 0.04

def _compute_water_mask(img_data: np.ndarray) -> np.ndarray:
    """
    Computes a binary water mask using NDWI (Normalized Difference Water Index).
    
    NDWI = (Green - NIR) / (Green + NIR)
    Water pixels → NDWI > 0 (positive), Land pixels → NDWI < 0 (negative).
    
    Band mapping (MARIDA order):
        Index 1: B3  (Green, 560nm)
        Index 6: B8  (NIR,   842nm)
    """
    green = img_data[1, :, :].astype(np.float32)  # B3
    nir   = img_data[6, :, :].astype(np.float32)  # B8

    denominator = green + nir
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    ndwi = (green - nir) / denominator
    
    # Water pixels have positive NDWI; use a slight negative threshold (-0.1)
    # to include shallow coastal waters that might have mixed pixels
    water_mask = ndwi > -0.1
    
    water_pct = np.sum(water_mask) / water_mask.size * 100
    print(f"[WATER MASK] NDWI range=[{ndwi.min():.3f}, {ndwi.max():.3f}], water={water_pct:.1f}%")
    
    return water_mask


def process_live_patch(img_data: np.ndarray, bbox: list) -> list:
    """
    1. Computes a water mask (NDWI) to exclude land pixels.
    2. Runs the U-Net Model on the live in-memory satellite patch.
    3. Computes FDI from spectral bands as a physics-based secondary signal.
    4. Combines both signals, masks out land, and uses an adaptive threshold.
    5. Interpolates detected pixel grids back to the geographic bbox.
    """
    # 0. Generate water mask — only ocean/coastal pixels will be considered
    water_mask = _compute_water_mask(img_data)

    # 1. U-Net deep learning execution
    inferencer = get_inferencer()
    predict_res = inferencer.predict(img_data)
    if isinstance(predict_res, dict):
        unet_probs = predict_res["probs"]
        class_map = predict_res["class_map"]
    else:
        unet_probs = predict_res
        class_map = None

    # 2. Combine with FDI spectral physics rules
    if img_data.shape[0] >= 11:
        nir = img_data[7, :, :]      # B8A
        red_edge = img_data[5, :, :] # B7
        swir = img_data[8, :, :]     # B11

        fdi_map = calculate_fdi(nir, red_edge, swir)
        fdi_norm = np.clip(fdi_map, 0, 1)

        # Weight: 60% model, 40% FDI — boosted FDI for live data
        final_probs = 0.6 * unet_probs + 0.4 * fdi_norm
    else:
        final_probs = unet_probs

    # 3. Apply water mask — zero out all land pixels
    final_probs = final_probs * water_mask

    # Log diagnostics
    max_prob = float(final_probs.max())
    above_threshold = int(np.sum(final_probs > LIVE_THRESHOLD))
    print(f"[PATCH INFERENCE] max_prob={max_prob:.4f}, pixels>{LIVE_THRESHOLD}: {above_threshold} (land filtered)")

    # 4. Extract geographic points using adaptive threshold
    results = []
    y_indices, x_indices = np.where(final_probs > LIVE_THRESHOLD)

    # Increase limit to 8000 for high-res pixel-level rendering
    if len(y_indices) > 8000:
        top_idx = np.argsort(final_probs[y_indices, x_indices])[-8000:]
        y_indices = y_indices[top_idx]
        x_indices = x_indices[top_idx]

    # Interpolate pixel coordinates -> real-time bbox
    lon_min, lat_min, lon_max, lat_max = bbox
    height, width = final_probs.shape
    
    from app.models.inference import CLASS_NAMES

    for y, x in zip(y_indices, x_indices):
        lon = float(lon_min + (x / width) * (lon_max - lon_min))
        lat = float(lat_max - (y / height) * (lat_max - lat_min))
        prob = float(final_probs[y, x])
        
        c_id = int(class_map[y, x]) if class_map is not None else 1
        
        # Override to Debris if probability is super high but class map mismatched
        if prob > 0.4:
            c_id = 1
            
        c_name = CLASS_NAMES.get(c_id, "Unknown")

        results.append({
            "lat": lat,
            "lon": lon,
            "probability": prob,
            "class_name": c_name
        })

    return results

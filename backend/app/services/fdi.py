import numpy as np

def calculate_fdi(nir: np.ndarray, redEdge: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """
    Compute Floating Debris Index (FDI).
    FDI = NIR - (RedEdge + (SWIR - RedEdge) * 10 * ((lambda_NIR - lambda_RedEdge) / 10 * (lambda_SWIR - lambda_RedEdge)))
    Simplified formula often used in literature for Sentinel-2.
    """
    # Assuming valid arrays and avoiding division by zero
    lambda_nir = 832.8
    lambda_redEdge = 740.5
    lambda_swir = 1613.7
    
    # Simple FDI approximation
    baseline = redEdge + (swir - redEdge) * ((lambda_nir - lambda_redEdge) / (lambda_swir - lambda_redEdge))
    fdi = nir - baseline
    
    return fdi

def combine_fdi_and_predictions(fdi_map: np.ndarray, unet_preds: np.ndarray, fdi_threshold: float = 0.05) -> np.ndarray:
    """
    Enhance U-Net predictions by combining with thresholded FDI.
    Returns enhanced probability map.
    """
    # This is a sample combination logic: 
    # E.g., boost probability if FDI indicates high likelihood of debris
    enhanced_preds = np.copy(unet_preds)
    enhanced_preds[fdi_map > fdi_threshold] = np.clip(enhanced_preds[fdi_map > fdi_threshold] + 0.2, 0, 1.0)
    
    return enhanced_preds

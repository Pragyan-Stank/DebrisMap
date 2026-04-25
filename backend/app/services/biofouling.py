import numpy as np

def compute_biofouling_correction(t_days: float, k: float = 0.05) -> float:
    """
    Computes the Signal Decay correction factor for biofouling.
    
    R_observed(t) = R_plastic * exp(-k * t) + R_algae * (1 - exp(-k * t))
    Correction factor: CF(t) = 1 / exp(-k * t) -> boosts dampened signal back to baseline
    
    Args:
        t_days: Days since first detection
        k: Biofouling rate (~0.03-0.07 per day in tropical waters)
        
    Returns:
        Correction factor to scale the FDI/NIR signal.
    """
    if t_days <= 0:
        return 1.0
    
    # Cap the maximum days to avoid extreme amplification of noise
    t_days = min(t_days, 60.0) 
    
    cf = 1.0 / np.exp(-k * t_days)
    return cf

def apply_biofouling_correction(fdi_map: np.ndarray, t_days: float) -> np.ndarray:
    """
    Applies the biofouling correction multiplier to a generated FDI map.
    """
    cf = compute_biofouling_correction(t_days)
    return fdi_map * cf

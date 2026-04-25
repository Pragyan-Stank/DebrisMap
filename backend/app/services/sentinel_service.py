"""
sentinel_service.py
====================
Fetches real Sentinel-2 L2A multi-spectral patches from Sentinel Hub Process API.
Uses the official sentinelhub-py SDK with OAuth credentials from .env.

Band mapping (MARIDA 11-band order):
  Index 0: B2  (Blue,       490nm)
  Index 1: B3  (Green,      560nm)
  Index 2: B4  (Red,        665nm)
  Index 3: B5  (Red Edge 1, 705nm)
  Index 4: B6  (Red Edge 2, 740nm)
  Index 5: B7  (Red Edge 3, 783nm)
  Index 6: B8  (NIR,        842nm)
  Index 7: B8A (NIR narrow, 865nm)
  Index 8: B11 (SWIR 1,    1610nm)
  Index 9: B12 (SWIR 2,    2190nm)
  Index 10: B1 (Coastal,    443nm)
"""

import os
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=env_path)

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

# Evalscript that returns all 11 bands the U-Net expects, as float32 reflectance
EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","B01"],
      units: "REFLECTANCE"
    }],
    output: {
      bands: 11,
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(s) {
  return [s.B02, s.B03, s.B04, s.B05, s.B06, s.B07, s.B08, s.B8A, s.B11, s.B12, s.B01];
}
"""

DATE_RANGE_MAP = {
    "last_1_day": 5,
    "last_3_days": 15,
    "last_5_days": 30,
}


def _generate_fallback_patch(bbox: list, max_size: int = 256) -> np.ndarray:
    """
    Fallback simulation if credentials are missing or the API fails.
    Returns a synthetic 11-band patch with injected spectral anomalies.
    """
    print(f"[FALLBACK] Generating synthetic Sentinel-2 patch for bbox: {bbox}")
    sz = max(max_size, 128)
    img = np.random.uniform(0.01, 0.25, size=(11, sz, sz)).astype(np.float32)

    margin = min(30, sz // 4)
    cy = np.random.randint(margin, sz - margin)
    cx = np.random.randint(margin, sz - margin)
    s = min(4, margin)
    img[7, cy:cy+s, cx:cx+s] = 0.55  # B8A
    img[8, cy:cy+s, cx:cx+s] = 0.48  # B11

    return img


def fetch_sentinel2_patch(bbox: list, size: tuple | int = 256, date_range: str = "last_3_days") -> np.ndarray:
    """
    Fetches raw Sentinel-2 L2A optical bands (11 channels matching MARIDA) from Copernicus Data Space.
    Returns:
        np.ndarray of shape (11, H, W) with reflectance values.
    """
    if not CLIENT_ID or not CLIENT_SECRET:
        print("[WARN] Sentinel Hub credentials not found in .env – using fallback simulation.")
        return _generate_fallback_patch(bbox, size if isinstance(size, int) else max(size))

    try:
        from sentinelhub import (
            SHConfig, CRS, BBox as SHBBox,
            DataCollection, SentinelHubRequest, MimeType
        )

        config = SHConfig()
        config.sh_client_id = CLIENT_ID
        config.sh_client_secret = CLIENT_SECRET
        config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        config.sh_base_url = "https://sh.dataspace.copernicus.eu"

        sh_bbox = SHBBox(bbox=[bbox[0], bbox[1], bbox[2], bbox[3]], crs=CRS.WGS84)

        days = DATE_RANGE_MAP.get(date_range, 3)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        time_interval = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

        print(f"[SENTINEL HUB] Requesting patch: bbox={bbox}, time={time_interval}, size={size}")

        request = SentinelHubRequest(
            evalscript=EVALSCRIPT,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A.define_from(
                        "s2l2a", service_url=config.sh_base_url
                    ),
                    time_interval=time_interval,
                    maxcc=0.3
                )
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.TIFF)
            ],
            bbox=sh_bbox,
            size=size if isinstance(size, tuple) else (size, size),
            config=config
        )

        data = request.get_data()

        if data and len(data) > 0:
            patch = data[0].astype(np.float32)
            if patch.ndim == 3 and patch.shape[2] == 11:
                patch = np.transpose(patch, (2, 0, 1))  # (H,W,C) -> (C,H,W)

            if np.mean(patch) < 0.001:
                print("[WARN] Received empty/no-data patch — likely no imagery for this date range.")
                return _generate_fallback_patch(bbox, size if isinstance(size, int) else max(size))

            print(f"[SENTINEL HUB] Successfully received patch: shape={patch.shape}, mean={np.mean(patch):.4f}")
            return patch
        else:
            print("[WARN] No data returned from Sentinel Hub — using fallback.")
            return _generate_fallback_patch(bbox, max_size)

    except Exception as e:
        print(f"[ERROR] Sentinel Hub request failed: {e}")
        return _generate_fallback_patch(bbox, max_size)

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import math
import numpy as np
import time
from typing import Any
from pathlib import Path

app = FastAPI(title="DebrisTrack Coastal Command API")
DATA_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = DATA_DIR / "uploaded_tifs"
UPLOAD_DIR.mkdir(exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScanRequest(BaseModel):
    lat: float
    lon: float
    lookbackDays: int = 45
    maxCloudCover: float = 70.0
    allowMock: bool = True

# Physics constants
EARTH_R    = 6_371_000.0   # metres
ALPHA      = 0.03          # wind leeway factor (3%)
DT_HOURS   = 1             # integration time step (hours)
FORECAST_H = 72            # total forecast window (hours)

BASE_U_CURRENT = 0.12   # m/s eastward
BASE_V_CURRENT = 0.04   # m/s northward
NOISE_STD = 0.05

def fetch_wind_forecast(lat: float, lon: float, hours: int = FORECAST_H):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude"       : lat,
        "longitude"      : lon,
        "hourly"         : "windspeed_10m,winddirection_10m",
        "wind_speed_unit": "ms",
        "forecast_days"  : math.ceil(hours / 24) + 1,
        "timezone"       : "auto",
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data  = resp.json()["hourly"]
        spds  = data["windspeed_10m"][:hours]
        dirs  = data["winddirection_10m"][:hours]
        
        uv = []
        for spd, d in zip(spds, dirs):
            dr  = math.radians(d)
            uv.append((-spd * math.sin(dr), -spd * math.cos(dr)))
        return uv, np.mean(spds), np.mean(dirs)
    except Exception as exc:
        print(f"Warning: Open-Meteo failed ({exc}). Using fallback wind.")
        fallback_uv = [(1.5, -0.5)] * hours
        return fallback_uv, 1.5, 110


def simulate_median_trajectory(start_lat, start_lon, wind_uv):
    """
    Simulates the median 72h path taking local wind and baseline current into account.
    We return an array of {lat, lon} for each hour.
    """
    trajectory = [{"lat": start_lat, "lon": start_lon}]
    
    lat, lon = start_lat, start_lon
    
    for h in range(FORECAST_H):
        u_w, v_w = wind_uv[h]
        
        # Deterministic single-image drift estimate for the UI.
        u_total = BASE_U_CURRENT + ALPHA * u_w
        v_total = BASE_V_CURRENT + ALPHA * v_w
        
        dt_sec = DT_HOURS * 3600
        dlat   = (v_total * dt_sec) / EARTH_R * (180 / math.pi)
        dlon   = (u_total * dt_sec) / (EARTH_R * math.cos(math.radians(lat))) * (180 / math.pi)
        
        lat += dlat
        lon += dlon
        trajectory.append({"lat": lat, "lon": lon})
        
    return trajectory


def build_mock_scan(center_lat: float, center_lon: float) -> dict[str, Any]:
    """Fallback used when Sentinel Hub credentials/network are unavailable."""
    time.sleep(1.5)
    wind_uv, avg_spd, avg_dir = fetch_wind_forecast(center_lat, center_lon)
    hotspots = []

    # The clicked coordinate is the patch top-left; mock detections stay inside
    # the 2.56 km x 2.56 km area by moving south/east from that corner.
    offsets = [
        (-0.004, 0.006),
        (-0.011, 0.014),
        (-0.017, 0.007),
        (-0.021, 0.020),
    ]

    vessels = ["CG Vessel Delta-4", "CG Cutter Osprey-2", "CG Patrol Foxtrot-7", "CG Vessel Lima-11"]
    threats = ["Coral Reef Zone", "Marine Protected Area", "Shipping Lane", "Tourist Beach"]
    pixels = [520, 310, 440, 180]
    confidence = [0.91, 0.82, 0.87, 0.69]
    threat_hours = [12, 26, 18, 44]

    for i, (lat_off, lon_off) in enumerate(offsets):
        h_lat = center_lat + lat_off
        h_lon = center_lon + lon_off
        trajectory = simulate_median_trajectory(h_lat, h_lon, wind_uv)

        hotspot = {
            "id": f"MOCK-{i + 1:02d}",
            "lat": h_lat,
            "lon": h_lon,
            "pixels": pixels[i],
            "mass": f"~{pixels[i] * 0.03:.1f} t",
            "risk": ["CRITICAL", "HIGH", "CRITICAL", "MEDIUM"][i],
            "confidence": confidence[i],
            "windSpeed": round(avg_spd, 2),
            "windDir": int(avg_dir),
            "currentDrift": "0.12 m/s",
            "threat": threats[i],
            "threatHours": threat_hours[i],
            "vessel": vessels[i],
            "interceptTime": f"{1.5 + i * 1.4:.1f} h",
            "trajectory": trajectory  # list of 73 {lat, lon} objects
        }
        hotspots.append(hotspot)

    return {
        "status": "mock",
        "mode": "mock_wind_drift",
        "hotspots": hotspots,
        "patch": {
            "top_left_lat": center_lat,
            "top_left_lon": center_lon,
            "width": 256,
            "height": 256,
            "resolution_m": 10.0,
        },
    }


@app.post("/api/scan")
def scan_sector(req: ScanRequest):
    top_left_lat = req.lat
    top_left_lon = req.lon

    try:
        from live_two_state_scan import run_live_two_state_scan

        result = run_live_two_state_scan(
            top_left_lat=top_left_lat,
            top_left_lon=top_left_lon,
            lookback_days=req.lookbackDays,
            max_cloud_cover=req.maxCloudCover,
        )
        for obs in result.get("observations", []):
            diag = obs.get("diagnostics", {})
            print(
                "S2 inference diagnostics "
                f"acquired={obs.get('acquired_at')} "
                f"bands={diag.get('band_count')} "
                f"shape={diag.get('width')}x{diag.get('height')} "
                f"dtype={diag.get('dtype')} "
                f"range=[{diag.get('image_min')}, {diag.get('image_max')}] "
                f"pred_counts={diag.get('pred_class_counts')} "
                f"debris_prob_max={diag.get('debris_prob_max')}"
            )
        return {"status": "success", **result}
    except Exception as exc:
        print(f"Live Sentinel Hub scan failed: {exc}")
        if not req.allowMock:
            return {
                "status": "error",
                "mode": "sentinelhub_two_state",
                "error": str(exc),
                "hotspots": [],
            }

        fallback = build_mock_scan(top_left_lat, top_left_lon)
        fallback["error"] = str(exc)
        return fallback


def risk_from_pixels_confidence(pixels: int, confidence: float) -> str:
    if pixels >= 400 or confidence >= 0.88:
        return "CRITICAL"
    if pixels >= 150 or confidence >= 0.75:
        return "HIGH"
    if pixels >= 40 or confidence >= 0.60:
        return "MEDIUM"
    return "LOW"


def hotspots_from_uploaded_clusters(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hotspots = []
    for i, cluster in enumerate(clusters):
        lat = float(cluster["lat"])
        lon = float(cluster["lon"])
        pixels = int(cluster["n_pixels"])
        confidence = float(cluster["mean_conf"])
        wind_uv, avg_spd, avg_dir = fetch_wind_forecast(lat, lon)
        trajectory = simulate_median_trajectory(lat, lon, wind_uv)

        hotspots.append({
            "id": f"TIF-{i + 1:03d}",
            "lat": lat,
            "lon": lon,
            "pixels": pixels,
            "mass": f"~{max(0.1, pixels * 0.03):.1f} t",
            "risk": risk_from_pixels_confidence(pixels, confidence),
            "confidence": round(confidence, 3),
            "windSpeed": round(float(avg_spd), 2),
            "windDir": int(round(float(avg_dir))),
            "currentDrift": "0.12 m/s + 3% wind",
            "threat": "Uploaded GeoTIFF debris detection",
            "threatHours": 72,
            "vessel": "Cleanup asset assignment pending",
            "interceptTime": "TBD",
            "trajectory": trajectory,
            "observations": {"newer": cluster},
        })

    return hotspots


@app.post("/api/import-tif")
async def import_tif(request: Request):
    filename = Path(request.headers.get("x-filename", "uploaded.tif")).name
    if not filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Upload a GeoTIFF file with .tif or .tiff extension.")

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Uploaded GeoTIFF is empty.")

    out_path = UPLOAD_DIR / f"{int(time.time() * 1000)}_{filename}"
    out_path.write_bytes(body)

    try:
        from live_two_state_scan import infer_geotiff

        result = infer_geotiff(out_path)
        hotspots = hotspots_from_uploaded_clusters(result["clusters"])
        obs = result["observation"]
        diag = obs.get("diagnostics", {})
        print(
            "Uploaded TIF inference diagnostics "
            f"file={filename} "
            f"bands={diag.get('band_count')} "
            f"shape={diag.get('width')}x{diag.get('height')} "
            f"dtype={diag.get('dtype')} "
            f"range=[{diag.get('image_min')}, {diag.get('image_max')}] "
            f"pred_counts={diag.get('pred_class_counts')} "
            f"debris_prob_max={diag.get('debris_prob_max')}"
        )
        return {
            "status": "success",
            "mode": "uploaded_geotiff",
            "patch": result["patch"],
            "hotspots": hotspots,
            "observations": [obs],
        }
    except Exception as exc:
        print(f"Uploaded TIF inference failed: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/api/drop")
def drop_tracker(req: ScanRequest):
    lat = req.lat
    lon = req.lon
    
    # Fast simulation (no delay for manual drops)
    wind_uv, avg_spd, avg_dir = fetch_wind_forecast(lat, lon)
    
    trajectory = simulate_median_trajectory(lat, lon, wind_uv)
    
    hotspot = {
        "id": "MANUAL-001",
        "lat": lat,
        "lon": lon,
        "pixels": 100,
        "mass": "~3.0 t",
        "risk": "MEDIUM",
        "confidence": 1.0,
        "windSpeed": round(avg_spd, 2),
        "windDir": int(avg_dir),
        "currentDrift": "0.12 m/s",
        "threat": "Manually Tracked Anomaly",
        "threatHours": 72,
        "vessel": "Standby Air Asset",
        "interceptTime": "2.0 h",
        "trajectory": trajectory
    }
    
    return {"status": "success", "hotspot": hotspot}

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)

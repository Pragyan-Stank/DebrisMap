from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
import numpy as np
import os
import rasterio

from app.schemas.predict import PredictionResponse, PatchInferenceRequest, TrajectoryRequest, TrajectoryFromClustersRequest
from app.models.inference import run_marine_debris_pipeline
from app.services.sentinel_service import fetch_sentinel2_patch
from app.services.patch_inference_service import process_live_patch
from app.services.clustering_service import compute_clusters
from app.services.trajectory_service import predict_trajectory_for_point, predict_trajectories_for_clusters
from app.services.detection_store import record_detections, get_history_summary

router = APIRouter()

# Global memory to act as a mock database for the dashboard's live-polling feature
live_hotspots = []

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Handle generic image upload metadata.
    """
    return {"status": "success", "message": f"File {file.filename} uploaded successfully"}

@router.post("/predict")
async def predict_debris(file: UploadFile = File(...)):
    """
    Takes an actual uploaded MARIDA `.tif` image, runs:
    1. U-Net + FDI debris detection (returns points above threshold)
    2. Full pixel-level segmentation (class_map for all pixels)
    Returns both for visualization.
    """
    global live_hotspots
    
    if not file.filename.endswith(('.tif', '.tiff')):
        raise HTTPException(status_code=400, detail="Only GeoTIFF files (.tif) are currently supported.")
    
    os.makedirs("data/uploads", exist_ok=True)
    temp_path = f"data/uploads/{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
        
    try:
        from app.models.inference import get_inferencer, CLASS_NAMES
        import torch
        import torch.nn.functional as TF
        
        with rasterio.open(temp_path) as src:
            image_data = src.read().astype(np.float32)
            orig_H, orig_W = src.height, src.width
            transform = src.transform
            crs = src.crs
            if image_data.shape[0] > 11:
                image_data = image_data[:11, :, :]
        
        # ── 1. Debris detection pipeline ──────────────────────
        results = run_marine_debris_pipeline(image_data, tif_path=temp_path)
        live_hotspots = results
        clusters = compute_clusters(results)
        record_detections(results, clusters, source="upload", metadata={"filename": file.filename})
        
        # ── 2. Full pixel segmentation ────────────────────────
        inferencer = get_inferencer()
        TARGET = 256
        if orig_H != TARGET or orig_W != TARGET:
            t = torch.from_numpy(image_data).unsqueeze(0)
            t = TF.interpolate(t, size=(TARGET, TARGET), mode='bilinear', align_corners=False)
            img_resized = t.squeeze(0).numpy()
        else:
            img_resized = image_data
            
        pred = inferencer.predict(img_resized)
        class_map = pred["class_map"] if isinstance(pred, dict) else np.zeros((TARGET, TARGET), dtype=np.int32)
        
        H, W = class_map.shape
        
        # Build geo-coordinate transformer
        transformer = None
        if crs and crs.to_epsg() != 4326:
            from pyproj import Transformer as ProjTransformer
            transformer = ProjTransformer.from_crs(crs, "EPSG:4326", always_xy=True)
        
        # Sample pixels uniformly via 2D grid (cap at ~15000)
        import math
        target_pts = 15000
        total_pts = H * W
        stride = max(1, int(math.sqrt(total_pts / target_pts)))
        
        seg_pixels = []
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                c_id = int(class_map[y, x])
                # Scale to original raster size for geo lookup
                orig_x = int(x / W * orig_W)
                orig_y = int(y / H * orig_H)
                xc, yc = transform * (orig_x, orig_y)
                if transformer:
                    lon, lat = transformer.transform(xc, yc)
                else:
                    lon, lat = xc, yc
                seg_pixels.append({
                    "lat": float(lat), "lon": float(lon),
                    "class_id": c_id,
                    "class_name": CLASS_NAMES.get(c_id, "Unknown")
                })
        
        # Class statistics
        unique, counts = np.unique(class_map, return_counts=True)
        class_stats = sorted([
            {"class_id": int(u), "class_name": CLASS_NAMES.get(int(u), "Unknown"),
             "pixel_count": int(c), "pct": round(float(c) / (H * W) * 100, 2)}
            for u, c in zip(unique, counts)
        ], key=lambda x: -x["pixel_count"])
        
        return {
            "status": "success",
            "message": f"Inference complete. {len(results)} debris detections.",
            "points": results,
            "clusters": clusters,
            "seg_pixels": seg_pixels,
            "class_stats": class_stats,
            "metadata": {"filename": file.filename, "found_hotspots": len(results)}
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patch-inference", response_model=PredictionResponse)
async def live_patch_inference(request: PatchInferenceRequest):
    """
    Simulates real-time tracking by fetching a live satellite grid via Sentinel Hub
    and passing the optical swath directly into the active U-Net architecture.
    Groups dense fields into clusters across the dashboard via DBSCAN.
    """
    global live_hotspots
    try:
        import math
        lon_min, lat_min, lon_max, lat_max = request.bbox
        width_km = abs(lon_max - lon_min) * 111.32 * math.cos(abs(lat_min + lat_max) / 2 * math.pi / 180)
        height_km = abs(lat_max - lat_min) * 111.32
        
        # Enforce exact 10m resolution mapping (100 pixels per km)
        width_px = int(width_km * 100)
        height_px = int(height_km * 100)
        
        # Clamp to multiple of 16 (for LightUNet poolings) between 32 and 1024 to avoid crashes on huge/tiny areas
        width_px = max(64, min(1024, (width_px // 16) * 16))
        height_px = max(64, min(1024, (height_px // 16) * 16))
        
        # 1. Fetch satellite swath dynamically across chosen Bounds
        img_data = fetch_sentinel2_patch(request.bbox, size=(width_px, height_px), date_range=request.date_range)
        
        # 2. Execute inference and remap to geo-coordinates perfectly
        points = process_live_patch(img_data, request.bbox)
        
        # 3. Group the individual pixels into macro clusters utilizing DBSCAN
        clusters = compute_clusters(points)
        live_hotspots = points
        
        # Record in detection history for Clean-Up Programme
        record_detections(points, clusters, source="patch_inference", metadata={"bounds": request.bbox, "date_range": request.date_range})
        
        return PredictionResponse(
            status="success",
            message="Live Bounding-Box Inference & Clustering completed.",
            points=points,
            clusters=clusters,
            metadata={"source": "Sentinel-2 Live", "bounds": request.bbox}
        )
    except Exception as e:
        print(f"Patch inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segmentation-map")
async def get_segmentation_map(request: PatchInferenceRequest):
    """
    Runs full U-Net pixel-level segmentation over a bbox.
    Returns EVERY pixel with its predicted class (not just Marine Debris above a threshold).
    Used for the pixel-level segmentation overlay in the Analyst tab.
    """
    try:
        import math
        from app.models.inference import get_inferencer, CLASS_NAMES
        
        lon_min, lat_min, lon_max, lat_max = request.bbox
        width_km = abs(lon_max - lon_min) * 111.32 * math.cos(abs(lat_min + lat_max) / 2 * math.pi / 180)
        height_km = abs(lat_max - lat_min) * 111.32
        
        width_px = max(64, min(512, (int(width_km * 100) // 16) * 16))
        height_px = max(64, min(512, (int(height_km * 100) // 16) * 16))
        
        print(f"[SEG-MAP] bbox={request.bbox}, size=({width_px},{height_px})")
        
        img_data = fetch_sentinel2_patch(request.bbox, size=(width_px, height_px), date_range=request.date_range)
        
        inferencer = get_inferencer()
        result = inferencer.predict(img_data)
        
        if isinstance(result, dict):
            class_map = result["class_map"]   # (H, W) int array of class IDs 0-15
        else:
            # Fallback: all debris
            class_map = np.ones((img_data.shape[1], img_data.shape[2]), dtype=np.int32)
        
        H, W = class_map.shape
        pixels = []
        
        # Subsample uniformly via 2D grid to avoid vertical line artifacts
        import math
        target_pts = 20000
        total_pts = H * W
        stride = max(1, int(math.sqrt(total_pts / target_pts)))
        
        pixels = []
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                c_id = int(class_map[y, x])
                c_name = CLASS_NAMES.get(c_id, "Unknown")
                
                lon = float(lon_min + (x / W) * (lon_max - lon_min))
                lat = float(lat_max - (y / H) * (lat_max - lat_min))
                
                pixels.append({
                    "lat": lat,
                    "lon": lon,
                    "class_id": c_id,
                    "class_name": c_name
                })
        
        # Build class statistics
        unique, counts = np.unique(class_map, return_counts=True)
        class_stats = [
            {
                "class_id": int(uid),
                "class_name": CLASS_NAMES.get(int(uid), "Unknown"),
                "pixel_count": int(cnt),
                "pct": round(float(cnt) / (H * W) * 100, 2)
            }
            for uid, cnt in zip(unique, counts)
        ]
        class_stats.sort(key=lambda x: x["pixel_count"], reverse=True)
        
        return {
            "status": "success",
            "width": W,
            "height": H,
            "pixels": pixels,
            "class_stats": class_stats,
            "bbox": request.bbox
        }
    except Exception as e:
        import traceback
        print(f"[SEG-MAP] Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/visualization-data", response_model=PredictionResponse)
async def get_visualization_data():
    """
    Returns the real aggregated live-hotspots cached from recent user `.tif` predictions.
    """
    return PredictionResponse(
        status="success",
        message="Live visualization data fetched.",
        points=live_hotspots
    )


# ─── Trajectory Prediction Endpoints ──────────────────────────────────

@router.post("/trajectory/predict")
async def predict_single_trajectory(request: TrajectoryRequest):
    """
    Predict 72h drift trajectory for a single coordinate.
    Uses Leeway physics model + live Open-Meteo wind + Monte Carlo uncertainty.
    """
    try:
        result = predict_trajectory_for_point(
            lat=request.lat,
            lon=request.lon,
            label=request.label,
            n_pixels=request.n_pixels,
            confidence=request.confidence,
        )
        return {"status": "success", "hotspot": result}
    except Exception as e:
        print(f"Trajectory prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trajectory/from-clusters")
async def predict_cluster_trajectories(request: TrajectoryFromClustersRequest):
    """
    Predict 72h trajectories for multiple debris clusters (from scan or upload).
    """
    try:
        hotspots = predict_trajectories_for_clusters(request.clusters, source=request.source)
        return {
            "status": "success",
            "hotspots": hotspots,
            "summary": {
                "total_clusters": len(hotspots),
                "critical": sum(1 for h in hotspots if h["risk"] == "CRITICAL"),
                "high": sum(1 for h in hotspots if h["risk"] == "HIGH"),
                "medium": sum(1 for h in hotspots if h["risk"] == "MEDIUM"),
                "low": sum(1 for h in hotspots if h["risk"] == "LOW"),
            }
        }
    except Exception as e:
        print(f"Cluster trajectory error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weather")
async def get_weather(lat: float, lon: float):
    """Fetch real-time marine weather for a coordinate."""
    from app.services.weather_service import fetch_marine_weather
    try:
        return fetch_marine_weather(lat, lon)
    except Exception as e:
        print(f"Weather fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Clean-Up Programme Endpoints ──────────────────────────────────

@router.get("/cleanup-hotspots")
async def get_cleanup_hotspots(hours: float = 72):
    """Returns aggregated cleanup intelligence from stored detection history."""
    from app.services.cleanup_service import build_cleanup_intelligence
    try:
        return build_cleanup_intelligence(max_age_hours=hours)
    except Exception as e:
        print(f"Cleanup hotspots error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/detection-history")
async def get_detection_history():
    """Returns summary of the detection store."""
    return get_history_summary()


# ─── Coast Guard Operations Endpoints ─────────────────────────────────

@router.post("/threat-assessment")
async def get_threat_assessment(lat: float, lon: float, density: int = 10, confidence: float = 0.5):
    """Assess threat level for a debris location, checking MPA proximity."""
    from app.services.coastguard_service import assess_threat
    return assess_threat(lat, lon, density, confidence)


@router.post("/intercept-plan")
async def plan_intercept(
    debris_lat: float, debris_lon: float,
    vessel_lat: float, vessel_lon: float,
    vessel_speed_knots: float = 22
):
    """
    Compute optimal intercept point where a vessel can meet drifting debris.
    First predicts the debris trajectory, then finds the earliest interception.
    """
    from app.services.coastguard_service import compute_intercept
    hotspot = predict_trajectory_for_point(debris_lat, debris_lon, label="INTERCEPT")
    intercept = compute_intercept(
        hotspot["trajectory"], vessel_lat, vessel_lon, vessel_speed_knots
    )
    return {
        "status": "success",
        "debris_origin": {"lat": debris_lat, "lon": debris_lon},
        "vessel_origin": {"lat": vessel_lat, "lon": vessel_lon},
        "intercept": intercept,
        "trajectory": hotspot["trajectory"],
    }


@router.post("/dispatch-plan")
async def generate_dispatch(hours: float = 72):
    """
    Generate a full dispatch plan from current cleanup zones.
    Returns vessel assignments, threat assessments, and ETAs for each zone.
    """
    from app.services.cleanup_service import build_cleanup_intelligence
    from app.services.coastguard_service import generate_dispatch_plan
    intel = build_cleanup_intelligence(max_age_hours=hours)
    if not intel.get("clusters"):
        return {"status": "empty", "dispatches": [], "message": "No zones to dispatch."}
    dispatches = generate_dispatch_plan(intel["clusters"])
    return {"status": "success", "dispatches": dispatches, "summary": intel["summary"]}


@router.get("/persistent-zones")
async def get_persistent_zones(hours: float = 168):
    """Detect chronic debris accumulation zones over the selected time window."""
    from app.services.coastguard_service import detect_persistent_zones
    zones = detect_persistent_zones(max_age_hours=hours)
    return {"status": "success", "zones": zones, "total": len(zones)}


@router.post("/optimal-route")
async def get_optimal_route(vessel_lat: float, vessel_lon: float, hours: float = 72):
    """
    Computes optimal greedy route to visit all active cleanup zones.
    """
    from app.services.cleanup_service import build_cleanup_intelligence
    from app.services.coastguard_service import compute_optimal_route
    intel = build_cleanup_intelligence(max_age_hours=hours)
    
    if not intel.get("clusters"):
        return {"status": "empty", "route": [], "total_distance_km": 0}
        
    result = compute_optimal_route(intel["clusters"], vessel_lat, vessel_lon)
    return {
        "status": "success",
        "route": result["route"],
        "total_distance_km": result["total_distance_km"],
        "vessel_origin": {"lat": vessel_lat, "lon": vessel_lon}
    }

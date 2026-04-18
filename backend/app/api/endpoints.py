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

router = APIRouter()

# Global memory to act as a mock database for the dashboard's live-polling feature
live_hotspots = []

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Handle generic image upload metadata.
    """
    return {"status": "success", "message": f"File {file.filename} uploaded successfully"}

@router.post("/predict", response_model=PredictionResponse)
async def predict_debris(file: UploadFile = File(...)):
    """
    Takes an actual uploaded MARIDA `.tif` image, runs physical U-Net inference,
    extracts the geographical CRS, and isolates specific debris latitude and longitudes.
    """
    global live_hotspots
    
    if not file.filename.endswith(('.tif', '.tiff')):
        raise HTTPException(status_code=400, detail="Only GeoTIFF files (.tif) are currently supported for geographic extraction.")
    
    # Safely save the file locally for parsing
    os.makedirs("data/uploads", exist_ok=True)
    temp_path = f"data/uploads/{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
        
    try:
        # Load the actual image array natively
        with rasterio.open(temp_path) as src:
            image_data = src.read().astype(np.float32)
            
            # The model requires strictly 11 bands!
            if image_data.shape[0] > 11:
                image_data = image_data[:11, :, :]
        
        # Execute the true mathematical pipeline over the physical file properties
        results = run_marine_debris_pipeline(image_data, tif_path=temp_path)
        live_hotspots = results  # Update the dashboard polling state!
        
        # Compute clusters for the response
        clusters = compute_clusters(results)
        
        return PredictionResponse(
            status="success",
            message="Real Inference completed successfully.",
            points=results,
            clusters=clusters,
            metadata={"filename": file.filename, "found_hotspots": len(results)}
        )
    except Exception as e:
        print(f"Inference error: {e}")
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
        # Scale the resolution dynamically (10km is ~512 optical grid constraint)
        size_mapping = {2: 128, 5: 256, 10: 512}  
        resolution = size_mapping.get(request.resolution, 256)
        
        # 1. Fetch satellite swath dynamically across chosen Bounds
        img_data = fetch_sentinel2_patch(request.bbox, max_size=resolution, date_range=request.date_range)
        
        # 2. Execute inference and remap to geo-coordinates perfectly
        points = process_live_patch(img_data, request.bbox)
        
        # 3. Group the individual pixels into macro clusters utilizing DBSCAN
        clusters = compute_clusters(points)
        live_hotspots = points
        
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

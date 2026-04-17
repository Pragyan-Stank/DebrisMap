from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
import numpy as np

from app.schemas.predict import PredictionResponse
from app.models.inference import run_marine_debris_pipeline
from app.services.data_processing import preprocess_image

router = APIRouter()

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Handle simple image upload, e.g. saving to disk or returning metadata.
    """
    if not file.filename.endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    return {"status": "success", "message": f"File {file.filename} uploaded successfully"}

@router.post("/predict", response_model=PredictionResponse)
async def predict_debris(file: UploadFile = File(...)):
    """
    Upload a satellite imagery slice and run U-Net + FDI inference.
    """
    if not file.filename.endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    # Simulating data loading
    image_data = np.random.uniform(0, 10000, size=(12, 256, 256))
    
    processed_img = preprocess_image(image_data)
    results = run_marine_debris_pipeline(processed_img, transform=None)
    
    return PredictionResponse(
        status="success",
        message="Inference completed successfully.",
        points=results,
        metadata={"filename": file.filename}
    )

@router.get("/visualization-data", response_model=PredictionResponse)
async def get_visualization_data():
    """
    Returns sample test data or recent inference points for frontend maps.
    """
    import random
    
    base_lat = 37.7749  # San Francisco coast example
    base_lon = -122.4194
    
    points = []
    for _ in range(500):
        lat = base_lat + random.uniform(-0.1, 0.1)
        lon = base_lon + random.uniform(-0.1, 0.1)
        prob = random.uniform(0.5, 0.99)
        points.append({"lat": lat, "lon": lon, "probability": prob})
        
    return PredictionResponse(
        status="success",
        message="Sample visualization data fetched.",
        points=points
    )

from pydantic import BaseModel
from typing import List, Optional

class GeoPoint(BaseModel):
    lat: float
    lon: float
    probability: float

class PredictionResponse(BaseModel):
    status: str
    message: str
    points: List[GeoPoint] = []
    metadata: Optional[dict] = None

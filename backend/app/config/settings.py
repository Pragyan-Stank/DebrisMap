from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Marine Plastic Detection System API"
    API_V1_STR: str = "/api/v1"
    MODEL_PATH: str = os.getenv("MODEL_PATH", "trained_models/unet_model.pth")
    # For dev origins, standard setup
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost",
        "http://localhost:5173", 
        "http://localhost:3000",
    ]

settings = Settings()

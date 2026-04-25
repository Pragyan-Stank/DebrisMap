"""
weather_service.py
==================
Fetches real-time marine weather data from WeatherAPI.com
and current/forecast conditions for the Drift Forecast tab.
"""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=env_path)

WEATHER_API_KEY = os.getenv("WEATHER_API")


def fetch_marine_weather(lat: float, lon: float) -> dict:
    """
    Fetch current weather + marine forecast from WeatherAPI.com.
    Returns wind, waves, visibility, sea temperature etc.
    """
    if not WEATHER_API_KEY:
        print("[WARN] WEATHER_API key missing. Returning fallback.")
        return _fallback_weather(lat, lon)

    try:
        # Current conditions
        current_resp = requests.get(
            "https://api.weatherapi.com/v1/forecast.json",
            params={
                "key": WEATHER_API_KEY,
                "q": f"{lat},{lon}",
                "days": 3,
                "aqi": "no",
            },
            timeout=10,
        )
        current_resp.raise_for_status()
        data = current_resp.json()

        current = data.get("current", {})
        location = data.get("location", {})
        forecast_days = data.get("forecast", {}).get("forecastday", [])

        # Extract hourly wind data for animated particles
        hourly_wind = []
        for day in forecast_days:
            for hour in day.get("hour", []):
                hourly_wind.append({
                    "time": hour.get("time"),
                    "wind_kph": hour.get("wind_kph", 0),
                    "wind_dir": hour.get("wind_dir", "N"),
                    "wind_degree": hour.get("wind_degree", 0),
                    "temp_c": hour.get("temp_c", 20),
                    "humidity": hour.get("humidity", 70),
                    "vis_km": hour.get("vis_km", 10),
                    "chance_of_rain": hour.get("chance_of_rain", "0"),
                })

        result = {
            "location": {
                "name": location.get("name", "Ocean"),
                "region": location.get("region", ""),
                "country": location.get("country", ""),
                "lat": location.get("lat", lat),
                "lon": location.get("lon", lon),
                "localtime": location.get("localtime", ""),
            },
            "current": {
                "temp_c": current.get("temp_c", 22),
                "wind_kph": current.get("wind_kph", 15),
                "wind_mph": current.get("wind_mph", 9),
                "wind_degree": current.get("wind_degree", 180),
                "wind_dir": current.get("wind_dir", "S"),
                "pressure_mb": current.get("pressure_mb", 1013),
                "humidity": current.get("humidity", 75),
                "cloud": current.get("cloud", 30),
                "feelslike_c": current.get("feelslike_c", 22),
                "vis_km": current.get("vis_km", 10),
                "uv": current.get("uv", 5),
                "gust_kph": current.get("gust_kph", 20),
                "condition": current.get("condition", {}).get("text", "Clear"),
                "condition_icon": current.get("condition", {}).get("icon", ""),
            },
            "hourly_wind": hourly_wind[:72],  # 3 days of hourly data
        }

        print(f"[WEATHER] OK: {result['current']['wind_kph']} kph {result['current']['wind_dir']}, "
              f"{result['current']['temp_c']}°C, {len(hourly_wind)} hourly records")
        return result

    except Exception as e:
        print(f"[WEATHER] API failed: {e}")
        return _fallback_weather(lat, lon)


def _fallback_weather(lat: float, lon: float) -> dict:
    return {
        "location": {"name": "Ocean", "region": "", "country": "", "lat": lat, "lon": lon, "localtime": ""},
        "current": {
            "temp_c": 24, "wind_kph": 18, "wind_mph": 11, "wind_degree": 135,
            "wind_dir": "SE", "pressure_mb": 1012, "humidity": 78, "cloud": 40,
            "feelslike_c": 26, "vis_km": 15, "uv": 6, "gust_kph": 25,
            "condition": "Partly Cloudy", "condition_icon": "",
        },
        "hourly_wind": [],
    }

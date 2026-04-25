"""
trajectory_service.py
=====================
72-hour debris drift forecasting using the Leeway physics model.
- Fetches live wind data from Open-Meteo API (free, no key)
- Simulates drift with ocean current + wind forcing
- Monte Carlo ensemble for uncertainty estimation
"""

import math
import numpy as np
import requests
from typing import Any

EARTH_R = 6_371_000.0
ALPHA = 0.03          # wind leeway factor (3%)
DT_HOURS = 1
FORECAST_H = 72
N_PARTICLES = 50      # Monte Carlo ensemble per cluster
NOISE_STD = 0.10

BASE_U_CURRENT = 0.12  # m/s eastward
BASE_V_CURRENT = 0.04  # m/s northward


def fetch_wind_forecast(lat: float, lon: float, hours: int = FORECAST_H) -> tuple:
    """
    Fetches hourly wind forecast from Open-Meteo.
    Returns (uv_list, avg_speed, avg_direction).
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "windspeed_10m,winddirection_10m",
        "wind_speed_unit": "ms",
        "forecast_days": math.ceil(hours / 24) + 1,
        "timezone": "auto",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()["hourly"]
        spds = data["windspeed_10m"][:hours]
        dirs = data["winddirection_10m"][:hours]

        uv = []
        for spd, d in zip(spds, dirs):
            dr = math.radians(d)
            uv.append((-spd * math.sin(dr), -spd * math.cos(dr)))

        print(f"[WIND] Open-Meteo OK: avg_speed={np.mean(spds):.2f} m/s, avg_dir={np.mean(dirs):.0f}°")
        return uv, float(np.mean(spds)), float(np.mean(dirs))
    except Exception as exc:
        print(f"[WIND] Open-Meteo failed ({exc}). Using fallback.")
        fallback = [(1.5, -0.5)] * hours
        return fallback, 1.5, 110.0


def simulate_median_trajectory(start_lat: float, start_lon: float, wind_uv: list) -> list:
    """Deterministic median trajectory for UI display."""
    trajectory = [{"lat": start_lat, "lon": start_lon}]
    lat, lon = start_lat, start_lon

    for h in range(FORECAST_H):
        u_w, v_w = wind_uv[h]
        u_total = BASE_U_CURRENT + ALPHA * u_w
        v_total = BASE_V_CURRENT + ALPHA * v_w

        dt_sec = DT_HOURS * 3600
        dlat = (v_total * dt_sec) / EARTH_R * (180 / math.pi)
        dlon = (u_total * dt_sec) / (EARTH_R * math.cos(math.radians(lat))) * (180 / math.pi)

        lat += dlat
        lon += dlon
        trajectory.append({"lat": lat, "lon": lon})

    return trajectory


def simulate_monte_carlo(start_lat: float, start_lon: float, wind_uv: list, seed: int = 0) -> dict:
    """
    Monte Carlo ensemble simulation.
    Returns median trajectory + uncertainty cone (5th/95th percentile).
    """
    rng = np.random.default_rng(seed=seed)

    all_lats = []
    all_lons = []

    for _ in range(N_PARTICLES):
        lats = [start_lat]
        lons = [start_lon]
        lat, lon = start_lat, start_lon

        for h in range(FORECAST_H):
            u_w, v_w = wind_uv[h]
            noise = rng.normal(1.0, NOISE_STD, size=4)

            u_total = BASE_U_CURRENT * noise[2] + ALPHA * u_w * noise[0]
            v_total = BASE_V_CURRENT * noise[3] + ALPHA * v_w * noise[1]

            dt_sec = DT_HOURS * 3600
            dlat = (v_total * dt_sec) / EARTH_R * (180 / math.pi)
            dlon = (u_total * dt_sec) / (EARTH_R * math.cos(math.radians(lat))) * (180 / math.pi)

            lat += dlat
            lon += dlon
            lats.append(lat)
            lons.append(lon)

        all_lats.append(lats)
        all_lons.append(lons)

    lat_matrix = np.array(all_lats)
    lon_matrix = np.array(all_lons)

    median_lat = np.median(lat_matrix, axis=0).tolist()
    median_lon = np.median(lon_matrix, axis=0).tolist()
    p5_lat = np.percentile(lat_matrix, 5, axis=0).tolist()
    p95_lat = np.percentile(lat_matrix, 95, axis=0).tolist()
    p5_lon = np.percentile(lon_matrix, 5, axis=0).tolist()
    p95_lon = np.percentile(lon_matrix, 95, axis=0).tolist()

    return {
        "median": [{"lat": la, "lon": lo} for la, lo in zip(median_lat, median_lon)],
        "cone_upper": [{"lat": la, "lon": lo} for la, lo in zip(p95_lat, p95_lon)],
        "cone_lower": [{"lat": la, "lon": lo} for la, lo in zip(p5_lat, p5_lon)],
    }


def risk_from_cluster(n_pixels: int, confidence: float, speed_m_s: float = 0) -> str:
    if n_pixels >= 400 or confidence >= 0.88 or speed_m_s >= 0.35:
        return "CRITICAL"
    if n_pixels >= 150 or confidence >= 0.75 or speed_m_s >= 0.20:
        return "HIGH"
    if n_pixels >= 40 or confidence >= 0.60:
        return "MEDIUM"
    return "LOW"


def predict_trajectory_for_point(lat: float, lon: float, label: str = "MANUAL", 
                                  n_pixels: int = 100, confidence: float = 0.8) -> dict:
    """
    Predict 72h trajectory for a single point.
    Used for manual drops and individual cluster tracking.
    """
    wind_uv, avg_spd, avg_dir = fetch_wind_forecast(lat, lon)
    trajectory = simulate_median_trajectory(lat, lon, wind_uv)
    monte_carlo = simulate_monte_carlo(lat, lon, wind_uv, seed=hash(f"{lat}{lon}") % 10000)
    risk = risk_from_cluster(n_pixels, confidence)

    return {
        "id": label,
        "lat": lat,
        "lon": lon,
        "pixels": n_pixels,
        "mass": f"~{max(0.1, n_pixels * 0.03):.1f} t",
        "risk": risk,
        "confidence": round(confidence, 3),
        "windSpeed": round(avg_spd, 2),
        "windDir": int(avg_dir),
        "currentDrift": f"{BASE_U_CURRENT:.2f} m/s + {ALPHA*100:.0f}% wind",
        "forecastHours": FORECAST_H,
        "trajectory": trajectory,
        "monteCarlo": monte_carlo,
        "milestones": {
            "t24": trajectory[24] if len(trajectory) > 24 else trajectory[-1],
            "t48": trajectory[48] if len(trajectory) > 48 else trajectory[-1],
            "t72": trajectory[72] if len(trajectory) > 72 else trajectory[-1],
        },
    }


def predict_trajectories_for_clusters(clusters: list, source: str = "detection") -> list:
    """
    Predict trajectories for all detected debris clusters.
    Each cluster dict needs: lat, lon, and optionally n_pixels, confidence/probability.
    """
    hotspots = []
    for i, cluster in enumerate(clusters):
        lat = cluster.get("lat", 0)
        lon = cluster.get("lon", 0)
        n_pixels = cluster.get("n_pixels", cluster.get("density", 10))
        conf = cluster.get("mean_conf", cluster.get("confidence", cluster.get("probability", 0.5)))

        label = f"{source.upper()}-{i+1:03d}"
        hotspot = predict_trajectory_for_point(lat, lon, label=label, n_pixels=n_pixels, confidence=conf)
        hotspots.append(hotspot)

    return hotspots

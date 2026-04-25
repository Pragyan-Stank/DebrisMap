"""
coastguard_service.py
=====================
Operational Coast Guard intelligence services:
1. Threat Assessment — risk scoring with MPA/coastline proximity
2. Vessel Intercept Planning — optimum intercept point along debris trajectory
3. Dispatch Recommendations — asset assignment with ETA
4. Persistent Zone Detection — chronic debris accumulation areas
"""

import math
import time
from typing import Any

EARTH_R = 6_371_000.0

# Known Marine Protected Areas (simplified bounding boxes)
# A real system would use WDPA shapefiles
PROTECTED_AREAS = [
    {"name": "Mesoamerican Reef", "lat_min": 15.8, "lat_max": 18.5, "lon_min": -88.5, "lon_max": -86.0, "type": "Coral Reef"},
    {"name": "Bay Islands MPA", "lat_min": 16.2, "lat_max": 16.6, "lon_min": -86.8, "lon_max": -85.8, "type": "National Marine Park"},
    {"name": "Sian Ka'an Reserve", "lat_min": 19.2, "lat_max": 20.0, "lon_min": -88.0, "lon_max": -87.2, "type": "UNESCO Biosphere"},
    {"name": "Great Barrier Reef", "lat_min": -24.5, "lat_max": -10.0, "lon_min": 142.5, "lon_max": 154.0, "type": "UNESCO Heritage"},
    {"name": "Galápagos Marine Reserve", "lat_min": -1.8, "lat_max": 1.5, "lon_min": -92.5, "lon_max": -88.5, "type": "Marine Reserve"},
    {"name": "Papahānaumokuākea", "lat_min": 22.0, "lat_max": 30.0, "lon_min": -180.0, "lon_max": -161.0, "type": "Marine Monument"},
    {"name": "Mediterranean MPA Network", "lat_min": 30.0, "lat_max": 46.0, "lon_min": -6.0, "lon_max": 36.0, "type": "Regional MPA"},
    {"name": "Chagos Marine Reserve", "lat_min": -8.0, "lat_max": -4.5, "lon_min": 70.0, "lon_max": 73.0, "type": "No-Take Zone"},
]

# Coast Guard vessel types with speeds
VESSEL_TYPES = [
    {"type": "Fast Response Cutter", "speed_knots": 28, "crew": 24, "range_nm": 2500},
    {"type": "Patrol Boat", "speed_knots": 22, "crew": 10, "range_nm": 600},
    {"type": "Offshore Patrol Vessel", "speed_knots": 18, "crew": 35, "range_nm": 4000},
    {"type": "Cleanup Barge", "speed_knots": 8, "crew": 15, "range_nm": 300},
]


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a)) / 1000


def assess_threat(lat: float, lon: float, density: int = 10,
                  confidence: float = 0.5, trajectory: list = None) -> dict:
    """
    Full threat assessment for a debris cluster.
    Checks MPA proximity, coastal proximity, and trajectory intersection.
    """
    threats = []
    mpa_risk = "NONE"
    nearest_mpa = None
    nearest_mpa_dist = float('inf')

    # Check current position against MPAs
    for mpa in PROTECTED_AREAS:
        if (mpa["lat_min"] <= lat <= mpa["lat_max"] and
            mpa["lon_min"] <= lon <= mpa["lon_max"]):
            threats.append({
                "type": "INSIDE_MPA",
                "area": mpa["name"],
                "area_type": mpa["type"],
                "severity": "CRITICAL",
                "message": f"Debris INSIDE {mpa['name']} ({mpa['type']})"
            })
            mpa_risk = "CRITICAL"
            nearest_mpa = mpa["name"]
            nearest_mpa_dist = 0
        else:
            center_lat = (mpa["lat_min"] + mpa["lat_max"]) / 2
            center_lon = (mpa["lon_min"] + mpa["lon_max"]) / 2
            dist = _haversine_km(lat, lon, center_lat, center_lon)
            if dist < nearest_mpa_dist:
                nearest_mpa_dist = dist
                nearest_mpa = mpa["name"]

            if dist < 50:
                threats.append({
                    "type": "NEAR_MPA",
                    "area": mpa["name"],
                    "area_type": mpa["type"],
                    "distance_km": round(dist, 1),
                    "severity": "HIGH",
                    "message": f"Debris {dist:.0f}km from {mpa['name']}"
                })
                if mpa_risk != "CRITICAL":
                    mpa_risk = "HIGH"

    # Check if trajectory intersects any MPA
    if trajectory:
        for mpa in PROTECTED_AREAS:
            for pt in trajectory[::6]:  # check every 6h
                if (mpa["lat_min"] <= pt["lat"] <= mpa["lat_max"] and
                    mpa["lon_min"] <= pt["lon"] <= mpa["lon_max"]):
                    threats.append({
                        "type": "TRAJECTORY_INTERSECT",
                        "area": mpa["name"],
                        "area_type": mpa["type"],
                        "severity": "HIGH",
                        "message": f"Drift path will enter {mpa['name']} within 72h"
                    })
                    break

    # Aggregate threat level
    if density >= 400 or any(t["severity"] == "CRITICAL" for t in threats):
        level = "CRITICAL"
    elif density >= 100 or confidence >= 0.75 or any(t["severity"] == "HIGH" for t in threats):
        level = "HIGH"
    elif density >= 30 or confidence >= 0.5:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "level": level,
        "mpa_risk": mpa_risk,
        "nearest_mpa": nearest_mpa,
        "nearest_mpa_km": round(nearest_mpa_dist, 1) if nearest_mpa_dist != float('inf') else None,
        "threats": threats,
        "factors": {
            "density": density,
            "confidence": round(confidence, 3),
            "mpa_proximity": mpa_risk != "NONE",
            "trajectory_risk": any(t["type"] == "TRAJECTORY_INTERSECT" for t in threats),
        }
    }


def compute_intercept(trajectory: list, vessel_lat: float, vessel_lon: float,
                      vessel_speed_knots: float = 22) -> dict:
    """
    Given a debris trajectory and vessel position, find the optimal intercept point.
    The debris moves along the trajectory at 1 point per hour.
    The vessel moves at vessel_speed_knots.
    """
    vessel_speed_kmh = vessel_speed_knots * 1.852
    best = None

    for hour, point in enumerate(trajectory):
        dist_km = _haversine_km(vessel_lat, vessel_lon, point["lat"], point["lon"])
        vessel_time_h = dist_km / vessel_speed_kmh if vessel_speed_kmh > 0 else float('inf')

        # Can the vessel arrive before or at the same time as the debris?
        if vessel_time_h <= hour + 1:
            if best is None or hour < best["intercept_hour"]:
                best = {
                    "intercept_hour": hour,
                    "intercept_lat": point["lat"],
                    "intercept_lon": point["lon"],
                    "vessel_travel_km": round(dist_km, 1),
                    "vessel_travel_hours": round(vessel_time_h, 1),
                    "fuel_estimate_liters": round(dist_km * 3.5, 0),  # ~3.5 L/km for patrol
                }

    if best is None:
        # Vessel can't catch up — find closest approach
        min_dist = float('inf')
        min_hour = 0
        for hour, point in enumerate(trajectory):
            dist_km = _haversine_km(vessel_lat, vessel_lon, point["lat"], point["lon"])
            if dist_km < min_dist:
                min_dist = dist_km
                min_hour = hour
        best = {
            "intercept_hour": min_hour,
            "intercept_lat": trajectory[min_hour]["lat"],
            "intercept_lon": trajectory[min_hour]["lon"],
            "vessel_travel_km": round(min_dist, 1),
            "vessel_travel_hours": round(min_dist / vessel_speed_kmh, 1),
            "fuel_estimate_liters": round(min_dist * 3.5, 0),
            "warning": "Vessel cannot overtake debris — closest approach shown",
        }

    return best


def generate_dispatch_plan(clusters: list) -> list:
    """
    Generate dispatch recommendations for a set of cleanup zones.
    Assigns vessel types based on distance and priority.
    """
    dispatches = []

    for i, cluster in enumerate(clusters):
        priority = cluster.get("priority", "LOW")
        density = cluster.get("density", 0)
        lat = cluster.get("lat", 0)
        lon = cluster.get("lon", 0)

        # Assign vessel type based on priority
        if priority == "HIGH":
            vessel = VESSEL_TYPES[0]  # Fast Response Cutter
            urgency = "IMMEDIATE"
        elif priority == "MEDIUM" and cluster.get("persistence", False):
            vessel = VESSEL_TYPES[2]  # Offshore Patrol
            urgency = "PRIORITY"
        elif priority == "MEDIUM":
            vessel = VESSEL_TYPES[1]  # Patrol Boat
            urgency = "SCHEDULED"
        else:
            vessel = VESSEL_TYPES[3]  # Cleanup Barge
            urgency = "ROUTINE"

        # Threat assessment
        threat = assess_threat(lat, lon, density, cluster.get("avg_confidence", 0.5))

        # Escalate urgency based on MPA threat
        if threat["level"] == "CRITICAL":
            vessel = VESSEL_TYPES[0]  # Fast Response Cutter
            urgency = "IMMEDIATE"
        elif threat["level"] == "HIGH" and urgency == "ROUTINE":
            vessel = VESSEL_TYPES[1]  # Patrol Boat
            urgency = "SCHEDULED"

        dispatches.append({
            "zone_id": cluster.get("id", i),
            "lat": lat,
            "lon": lon,
            "priority": priority,
            "urgency": urgency,
            "assigned_vessel": vessel["type"],
            "vessel_speed_knots": vessel["speed_knots"],
            "vessel_crew": vessel["crew"],
            "threat_level": threat["level"],
            "mpa_risk": threat["mpa_risk"],
            "nearest_mpa": threat["nearest_mpa"],
            "nearest_mpa_km": threat["nearest_mpa_km"],
            "threats": threat["threats"],
            "recommended_action": _action_for_urgency(urgency, threat),
            "estimated_cleanup_hours": max(2, density // 20),
        })

    dispatches.sort(key=lambda d: {"IMMEDIATE": 0, "PRIORITY": 1, "SCHEDULED": 2, "ROUTINE": 3}[d["urgency"]])
    return dispatches


def _action_for_urgency(urgency: str, threat: dict) -> str:
    if threat["level"] == "CRITICAL":
        return "ALERT: Deploy Fast Response Cutter immediately. MPA threat detected."
    if urgency == "IMMEDIATE":
        return "Deploy assets within 2 hours. High-density debris zone."
    if urgency == "PRIORITY":
        return "Schedule patrol within 12 hours. Persistent debris accumulation."
    if urgency == "SCHEDULED":
        return "Add to next scheduled patrol route."
    return "Monitor via satellite. No immediate action required."


def detect_persistent_zones(max_age_hours: float = 168) -> list:
    """
    Identify zones where debris consistently accumulates over time.
    Uses the detection store across multiple time intervals.
    """
    from app.services.detection_store import get_all_detections

    entries = get_all_detections(max_age_hours=max_age_hours)
    if not entries:
        return []

    # Divide into 24h intervals and track which grid cells appear in multiple intervals
    now = time.time()
    grid_res = 0.03  # ~3.3km cells
    interval_grids = {}  # interval_idx -> set of grid keys

    for entry in entries:
        age_h = (now - entry["timestamp"]) / 3600
        interval_idx = int(age_h / 24)

        if interval_idx not in interval_grids:
            interval_grids[interval_idx] = {}

        for pt in entry.get("points", []):
            gk = f"{round(pt['lat']/grid_res)*grid_res:.3f},{round(pt['lon']/grid_res)*grid_res:.3f}"
            if gk not in interval_grids[interval_idx]:
                interval_grids[interval_idx][gk] = 0
            interval_grids[interval_idx][gk] += 1

    # Find grid cells that appear in multiple intervals
    all_grids = {}
    for idx, grids in interval_grids.items():
        for gk, count in grids.items():
            if gk not in all_grids:
                all_grids[gk] = {"intervals": set(), "total_detections": 0}
            all_grids[gk]["intervals"].add(idx)
            all_grids[gk]["total_detections"] += count

    persistent = []
    for gk, info in all_grids.items():
        if len(info["intervals"]) >= 2:  # appears in at least 2 separate 24h windows
            parts = gk.split(",")
            lat, lon = float(parts[0]), float(parts[1])
            threat = assess_threat(lat, lon, info["total_detections"])
            persistent.append({
                "lat": lat,
                "lon": lon,
                "intervals_active": len(info["intervals"]),
                "total_detections": info["total_detections"],
                "persistence_score": len(info["intervals"]) * info["total_detections"],
                "threat": threat,
            })

    persistent.sort(key=lambda z: z["persistence_score"], reverse=True)
    return persistent[:20]


def compute_optimal_route(clusters: list, vessel_lat: float, vessel_lon: float, vessel_speed_knots: float = 22) -> dict:
    """
    Computes a greedy intercept-based route starting from the vessel's location.
    Dynamically accounts for debris drift trajectories, calculating true optimum interception
    distances as time elapses, rather than hardcoded static distances.
    """
    if not clusters:
        return {"route": [], "total_distance_km": 0.0}

    from app.services.trajectory_service import fetch_wind_forecast, simulate_median_trajectory
    
    # Use a single local wind forecast for all nearby clusters to avoid API spam
    wind_uv, _, _ = fetch_wind_forecast(vessel_lat, vessel_lon, hours=168)

    unvisited = []
    for i, c in enumerate(clusters):
        lat, lon = c.get("lat", 0), c.get("lon", 0)
        # Pre-compute 168h trajectory so we can track drifting over long sweeps
        traj = simulate_median_trajectory(lat, lon, wind_uv)
        unvisited.append({
            "id": c.get("id", i),
            "priority": c.get("priority", "LOW"),
            "full_trajectory": traj,
        })

    route = []
    total_dist = 0.0
    current_lat, current_lon = vessel_lat, vessel_lon
    current_time_hr = 0.0

    while unvisited:
        best_idx = -1
        min_travel_h = float('inf')
        best_intercept = None

        for idx, node in enumerate(unvisited):
            # The trajectory point corresponding to the current elapsed time
            start_idx = min(int(current_time_hr), len(node["full_trajectory"]) - 1)
            future_drift = node["full_trajectory"][start_idx:]
            
            # Predict intercept from the current vessel location to the drifting debris
            ic = compute_intercept(future_drift, current_lat, current_lon, vessel_speed_knots)
            
            if ic and ic["vessel_travel_hours"] < min_travel_h:
                min_travel_h = ic["vessel_travel_hours"]
                best_idx = idx
                best_intercept = ic

        if best_idx == -1:
            break

        next_node = unvisited.pop(best_idx)
        
        # Move vessel to intercept point
        travel_km = best_intercept["vessel_travel_km"]
        travel_h = best_intercept["vessel_travel_hours"]
        
        route.append({
            "id": next_node["id"],
            "priority": next_node["priority"],
            "intercept_hour": round(current_time_hr + travel_h, 1),
            "lat": best_intercept["intercept_lat"],  # Show actual intercept coordinates
            "lon": best_intercept["intercept_lon"],
            "distance_from_prev_km": round(travel_km, 1)
        })
        
        total_dist += travel_km
        current_time_hr += travel_h + 2.0  # Add 2 hours for the cleanup operation itself
        current_lat = best_intercept["intercept_lat"]
        current_lon = best_intercept["intercept_lon"]

    return {
        "route": route,
        "total_distance_km": round(total_dist, 1)
    }


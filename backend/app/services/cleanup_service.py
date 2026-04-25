"""
cleanup_service.py
==================
Aggregates stored debris detections into actionable cleanup intelligence.
- DBSCAN clustering of all historical points
- Priority scoring: density × frequency × recency
- Grid-based frequency analysis for "most affected regions"
- Coastal proximity scoring
- Recommended actions per cluster
"""

import time
import math
import numpy as np
from sklearn.cluster import DBSCAN
from app.services.detection_store import get_all_points

EARTH_R = 6_371_000.0

# Coarse grid resolution for frequency analysis (in degrees)
GRID_RESOLUTION = 0.05  # ~5.5km cells


def _haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def _classify_priority(score: float) -> str:
    if score >= 5.0:
        return "HIGH"
    if score >= 1.5:
        return "MEDIUM"
    return "LOW"


def _recommend_action(priority: str, persistence: bool) -> str:
    if priority == "HIGH":
        return "Immediate Attention — Deploy cleanup assets"
    if priority == "MEDIUM" and persistence:
        return "Deploy Cleanup — Persistent debris zone"
    if priority == "MEDIUM":
        return "Deploy Cleanup — Elevated risk"
    return "Monitor — Low density area"


def _coastal_risk_boost(lat: float, lon: float) -> float:
    """
    Basic coastal proximity heuristic.
    Points within typical coastal bands get a risk multiplier.
    This is a rough approximation — real systems use coastline shapefiles.
    """
    # Very basic: check if coordinates are in ranges typical of coastal regions
    # (shallow water tends to be near integer lat/lon values...)
    # For now, use absolute latitude as a proxy: higher latitudes often have
    # more coastline. Also, points closer to 0° have more open ocean.
    # This is a placeholder — a real system would use GIS coastline distance.
    abs_lat = abs(lat)
    if abs_lat > 50 or abs_lat < 5:
        return 1.0

    # Points that are frequently detected are more likely coastal (debris accumulates)
    return 1.2


def build_cleanup_intelligence(max_age_hours: float = 72) -> dict:
    """
    Main pipeline:
    1. Get all historical detection points within time window
    2. DBSCAN cluster them into cleanup zones
    3. Score and prioritize each zone
    4. Grid frequency analysis for top regions
    5. Return structured intelligence
    """
    all_points = get_all_points(max_age_hours=max_age_hours)

    if not all_points:
        return {
            "status": "empty",
            "message": "No detections in the selected time window. Run a scan in the Analytics Hub first.",
            "clusters": [],
            "raw_points": [],
            "top_regions": [],
            "summary": {"total_points": 0, "total_clusters": 0, "high": 0, "medium": 0, "low": 0},
        }

    now = time.time()

    # ── Step 1: Prepare coordinates ──
    coords = np.array([[p["lat"], p["lon"]] for p in all_points])
    probs = np.array([p.get("probability", 0.5) for p in all_points])
    timestamps = np.array([p.get("timestamp", now) for p in all_points])

    # ── Step 2: DBSCAN clustering ──
    coords_rad = np.radians(coords)
    eps_rad = 0.02 * (math.pi / 180)  # ~2km

    labels = DBSCAN(
        eps=eps_rad,
        min_samples=2,
        metric="haversine",
        algorithm="ball_tree",
    ).fit_predict(coords_rad)

    # ── Step 3: Build cluster intelligence ──
    clusters = []
    unique_labels = set(labels) - {-1}

    # Divide the time window into intervals for persistence detection
    interval_count = max(1, int(max_age_hours / 24))
    interval_duration = (max_age_hours * 3600) / interval_count

    for cid in sorted(unique_labels):
        mask = labels == cid
        cluster_coords = coords[mask]
        cluster_probs = probs[mask]
        cluster_times = timestamps[mask]

        centroid_lat = float(cluster_coords[:, 0].mean())
        centroid_lon = float(cluster_coords[:, 1].mean())

        # Density
        density = int(mask.sum())

        # Radius (max distance from centroid to any point in cluster)
        distances = [_haversine_m(centroid_lat, centroid_lon, c[0], c[1]) for c in cluster_coords]
        radius_m = float(max(distances)) if distances else 100.0

        # Frequency: how many distinct time intervals have detections in this cluster
        interval_hits = set()
        for t in cluster_times:
            interval_idx = int((now - t) / interval_duration) if interval_duration > 0 else 0
            interval_hits.add(interval_idx)
        frequency = len(interval_hits)

        # Persistence: appears in more than half the intervals
        persistence = frequency >= max(2, interval_count // 2)

        # Recency weight: more recent = higher weight (exponential decay)
        most_recent = float(cluster_times.max())
        hours_ago = (now - most_recent) / 3600
        recency_weight = math.exp(-hours_ago / 48)  # half-life of 48h

        # Coastal risk multiplier
        coastal_mult = _coastal_risk_boost(centroid_lat, centroid_lon)

        # Priority score
        priority_score = (density * frequency * recency_weight * coastal_mult) / 10.0
        priority = _classify_priority(priority_score)
        action = _recommend_action(priority, persistence)

        clusters.append({
            "id": int(cid),
            "lat": centroid_lat,
            "lon": centroid_lon,
            "density": density,
            "radius_m": round(radius_m, 1),
            "frequency": frequency,
            "persistence": persistence,
            "recency_hours": round(hours_ago, 1),
            "priority_score": round(priority_score, 2),
            "priority": priority,
            "action": action,
            "avg_confidence": round(float(cluster_probs.mean()), 3),
            "max_confidence": round(float(cluster_probs.max()), 3),
        })

    # Sort by priority score descending
    clusters.sort(key=lambda c: c["priority_score"], reverse=True)

    # ── Step 4: Grid frequency analysis ──
    grid_freq = {}
    for p in all_points:
        grid_lat = round(p["lat"] / GRID_RESOLUTION) * GRID_RESOLUTION
        grid_lon = round(p["lon"] / GRID_RESOLUTION) * GRID_RESOLUTION
        key = f"{grid_lat:.3f},{grid_lon:.3f}"
        if key not in grid_freq:
            grid_freq[key] = {"lat": grid_lat, "lon": grid_lon, "count": 0, "sources": set()}
        grid_freq[key]["count"] += 1
        grid_freq[key]["sources"].add(p.get("source", "unknown"))

    top_regions = sorted(grid_freq.values(), key=lambda g: g["count"], reverse=True)[:10]
    for r in top_regions:
        r["sources"] = list(r["sources"])

    # ── Step 5: Build raw points for heatmap (sampled if too many) ──
    raw_points = all_points
    if len(raw_points) > 2000:
        step = len(raw_points) // 2000
        raw_points = raw_points[::step]

    # Summary counts
    high_count = sum(1 for c in clusters if c["priority"] == "HIGH")
    med_count = sum(1 for c in clusters if c["priority"] == "MEDIUM")
    low_count = sum(1 for c in clusters if c["priority"] == "LOW")

    print(f"[CLEANUP] Processed {len(all_points)} points -> {len(clusters)} clusters "
          f"(H:{high_count} M:{med_count} L:{low_count}), {len(top_regions)} top regions")

    return {
        "status": "ok",
        "time_window_hours": max_age_hours,
        "clusters": clusters,
        "raw_points": [{"lat": p["lat"], "lon": p["lon"], "probability": p.get("probability", 0.5)} for p in raw_points],
        "top_regions": top_regions,
        "summary": {
            "total_points": len(all_points),
            "total_clusters": len(clusters),
            "high": high_count,
            "medium": med_count,
            "low": low_count,
        },
    }

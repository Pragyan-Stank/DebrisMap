"""
seed_service.py
===============
Generates realistic synthetic marine debris detection history
so the Clean-Up Programme features are fully functional without
requiring prior satellite scans. Modelled on Caribbean debris patterns.
"""

import time
import math
import random
from app.services.detection_store import record_detections, clear_history

EARTH_R = 6_371_000.0

# Real-world Caribbean debris accumulation zones (empirically documented)
DEBRIS_HOTSPOTS = [
    # (lat, lon, label, density_scale)
    (16.3,  -86.3, "Bay Islands Dump",        1.4),
    (18.0,  -87.5, "Yucatan Channel Entry",   1.2),
    (17.2,  -86.8, "Honduras Coastal Band",   1.0),
    (15.9,  -85.5, "Gulf of Honduras",        0.9),
    (19.5,  -87.6, "Sian Ka'an Approach",     1.3),
    (17.8,  -88.0, "Belize Shelf Edge",       0.8),
    (16.9,  -87.1, "Inner Barrier Reef",      1.1),
    (15.4,  -86.0, "La Ceiba Offshore",       0.7),
    (18.5,  -86.4, "Lighthouse Reef Lagoon",  1.0),
    (17.0,  -85.2, "Swan Islands Drift",      0.6),
]

# Simulate 7 days of detection history (multiple intervals for persistent zone detection)
N_DAYS = 7
DETECTIONS_PER_DAY_PER_ZONE = 15


def _jitter(val: float, scale: float) -> float:
    return val + random.gauss(0, scale)


def _generate_cluster_points(lat: float, lon: float, n: int, spread_km: float = 5.0) -> list:
    """Generate n random points around a lat/lon cluster center."""
    spread_deg = spread_km / 111.0
    points = []
    for _ in range(n):
        p_lat = _jitter(lat, spread_deg * 0.7)
        p_lon = _jitter(lon, spread_deg)
        prob = round(min(0.97, max(0.35, random.gauss(0.72, 0.18))), 3)
        points.append({"lat": p_lat, "lon": p_lon, "probability": prob})
    return points


def seed_demo_data(clear_existing: bool = False) -> dict:
    """
    Populate the detection store with realistic debris detections
    spanning the past N_DAYS. Called once on demand.
    
    Returns: summary of seeded data
    """
    if clear_existing:
        clear_history()

    now = time.time()
    total_points = 0
    total_entries = 0

    # Generate detections spread across the last 7 days
    for day_offset in range(N_DAYS - 1, -1, -1):  # oldest first
        day_start_ts = now - (day_offset + 1) * 86400
        day_end_ts   = now - day_offset * 86400

        for zone_lat, zone_lon, zone_label, density_scale in DEBRIS_HOTSPOTS:
            # Not every zone fires every day (realistic — some days have clouds)
            if random.random() < 0.25:
                continue  # 25% chance of no detection this day

            # Number of points for this zone-day
            n_pts = int(DETECTIONS_PER_DAY_PER_ZONE * density_scale * random.uniform(0.6, 1.5))
            points = _generate_cluster_points(zone_lat, zone_lon, n_pts)

            # Assign timestamp spread within the day window
            entry_ts = random.uniform(day_start_ts, day_end_ts)

            # Monkey-patch timestamp so record_detections stores correct time
            import app.services.detection_store as ds
            import threading

            entry = {
                "timestamp": entry_ts,
                "iso_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(entry_ts)),
                "source": "satellite_seed",
                "num_points": len(points),
                "num_clusters": 1,
                "points": points[:500],
                "clusters": [{
                    "id": 0,
                    "lat": zone_lat,
                    "lon": zone_lon,
                    "density": len(points),
                    "priority": "HIGH" if density_scale >= 1.3 else "MEDIUM" if density_scale >= 0.9 else "LOW",
                }],
                "metadata": {"label": zone_label, "seeded": True},
            }

            with ds._lock:
                ds._history.append(entry)
                if len(ds._history) > 200:
                    ds._history[:] = ds._history[-200:]

            total_points += len(points)
            total_entries += 1

    # Save to disk
    import app.services.detection_store as ds
    ds._save_to_disk()

    print(f"[SEED] Seeded {total_entries} detection entries, {total_points} total points across {N_DAYS} days")

    return {
        "status": "seeded",
        "entries_created": total_entries,
        "total_points": total_points,
        "days_covered": N_DAYS,
        "zones_seeded": len(DEBRIS_HOTSPOTS),
        "message": f"Demo data seeded: {total_entries} detection events spanning {N_DAYS} days."
    }

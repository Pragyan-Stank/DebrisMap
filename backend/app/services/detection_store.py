"""
detection_store.py
==================
Thread-safe in-memory store for debris detection history.
All detections from patch inference, uploaded TIFs, and trajectory scans
are persisted here for downstream analytics (Clean-Up Programme, etc.).
"""

import threading
import time
import json
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_history: list[dict] = []

# Persist to disk so data survives server restarts
_PERSIST_PATH = Path(__file__).resolve().parents[3] / "data" / "detection_history.json"


def _ensure_dir():
    _PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_from_disk():
    global _history
    _ensure_dir()
    if _PERSIST_PATH.exists():
        try:
            _history = json.loads(_PERSIST_PATH.read_text(encoding="utf-8"))
            print(f"[STORE] Loaded {len(_history)} historical detections from disk.")
        except Exception as e:
            print(f"[STORE] Failed to load history: {e}")
            _history = []


def _save_to_disk():
    _ensure_dir()
    try:
        _PERSIST_PATH.write_text(json.dumps(_history[-5000:], default=str), encoding="utf-8")
    except Exception:
        pass


# Load on import
_load_from_disk()


def record_detections(points: list[dict], clusters: list[dict] = None,
                      source: str = "patch_inference", metadata: dict = None):
    """
    Record a batch of debris detections into history.
    
    Args:
        points: list of {lat, lon, probability} dicts
        clusters: optional list of cluster dicts
        source: "patch_inference" | "upload" | "trajectory"
        metadata: any extra context (bbox, filename, etc.)
    """
    if not points:
        return

    timestamp = time.time()
    entry = {
        "timestamp": timestamp,
        "iso_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
        "source": source,
        "num_points": len(points),
        "num_clusters": len(clusters) if clusters else 0,
        "points": points[:500],  # Cap per-entry to save memory
        "clusters": clusters or [],
        "metadata": metadata or {},
    }

    with _lock:
        _history.append(entry)
        # Keep max 200 entries (each can have up to 500 points)
        if len(_history) > 200:
            _history[:] = _history[-200:]
        _save_to_disk()

    print(f"[STORE] Recorded {len(points)} points from {source} "
          f"(total entries: {len(_history)})")


def get_all_detections(max_age_hours: float = None) -> list[dict]:
    """
    Get all stored detection entries, optionally filtered by recency.
    """
    with _lock:
        if max_age_hours is None:
            return list(_history)
        cutoff = time.time() - (max_age_hours * 3600)
        return [e for e in _history if e["timestamp"] >= cutoff]


def get_all_points(max_age_hours: float = None) -> list[dict]:
    """
    Get flattened list of all detection points with timestamps.
    """
    entries = get_all_detections(max_age_hours)
    points = []
    for entry in entries:
        ts = entry["timestamp"]
        for p in entry.get("points", []):
            points.append({
                "lat": p["lat"],
                "lon": p["lon"],
                "probability": p.get("probability", 0.5),
                "timestamp": ts,
                "source": entry["source"],
            })
    return points


def get_history_summary() -> dict:
    """Quick summary of the detection store."""
    with _lock:
        total_entries = len(_history)
        total_points = sum(e["num_points"] for e in _history)
        sources = {}
        for e in _history:
            sources[e["source"]] = sources.get(e["source"], 0) + 1
        oldest = _history[0]["iso_time"] if _history else None
        newest = _history[-1]["iso_time"] if _history else None
        return {
            "total_entries": total_entries,
            "total_points": total_points,
            "sources": sources,
            "oldest": oldest,
            "newest": newest,
        }


def clear_history():
    """Wipe all stored detections."""
    with _lock:
        _history.clear()
        _save_to_disk()

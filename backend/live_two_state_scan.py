"""
live_two_state_scan.py
======================
Fetch the latest two Sentinel-2 states for a clicked top-left coordinate,
run MARIDA inference on both, and derive debris drift from state A to state B.
"""

from __future__ import annotations

import math
import base64
import io
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import pyproj
import rasterio
from sklearn.cluster import DBSCAN

from inference import infer_patch, load_model
from pixel_latlon import full_pixel_grid
from sentinelhub_client import SentinelPatch, fetch_latest_two_patches, parse_sentinel_time


EARTH_R = 6_371_000.0
FORECAST_HOURS = 72

CLASS_RGBA = {
    1: (255, 65, 54, 230),      # Marine debris
    2: (61, 153, 112, 170),     # Dense sargassum
    3: (127, 219, 255, 155),    # Sparse sargassum
    4: (244, 208, 63, 155),     # Natural organic material
    5: (255, 133, 27, 190),     # Ship
    6: (190, 190, 190, 155),    # Clouds
    7: (0, 116, 217, 50),       # Marine water
    8: (201, 169, 110, 150),    # Sediment-laden water
    9: (255, 255, 255, 165),    # Foam
    10: (57, 204, 204, 145),    # Turbid water
    11: (133, 193, 233, 145),   # Shallow water
    12: (213, 219, 219, 145),   # Waves
    13: (86, 101, 115, 155),    # Cloud shadows
    14: (30, 139, 195, 165),    # Wakes
    15: (93, 173, 226, 145),    # Mixed water
}

_MODEL_CACHE: dict[str, Any] = {"device": None, "model": None}


@dataclass
class DebrisCluster:
    id: int
    lat: float
    lon: float
    n_pixels: int
    mean_conf: float
    max_conf: float
    acquired_at: str
    source_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _MODEL_CACHE["model"] is None or _MODEL_CACHE["device"] != str(device):
        _MODEL_CACHE["model"] = load_model(device)
        _MODEL_CACHE["device"] = str(device)
    return _MODEL_CACHE["model"], device


def _parse_time(value: str) -> datetime:
    return parse_sentinel_time(value)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def _destination_point(lat: float, lon: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    bearing = math.radians(bearing_deg)
    p1 = math.radians(lat)
    l1 = math.radians(lon)
    dr = distance_m / EARTH_R

    p2 = math.asin(math.sin(p1) * math.cos(dr) + math.cos(p1) * math.sin(dr) * math.cos(bearing))
    l2 = l1 + math.atan2(
        math.sin(bearing) * math.sin(dr) * math.cos(p1),
        math.cos(dr) - math.sin(p1) * math.sin(p2),
    )
    return math.degrees(p2), ((math.degrees(l2) + 540) % 360) - 180


def _interpolate_path(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    steps: int = FORECAST_HOURS,
) -> list[dict[str, float]]:
    return [
        {
            "lat": start_lat + (end_lat - start_lat) * i / steps,
            "lon": start_lon + (end_lon - start_lon) * i / steps,
        }
        for i in range(steps + 1)
    ]


def _forecast_from_velocity(
    lat: float,
    lon: float,
    speed_m_s: float,
    bearing_deg: float,
    hours: int = FORECAST_HOURS,
) -> list[dict[str, float]]:
    points = []
    for h in range(hours + 1):
        dest_lat, dest_lon = _destination_point(lat, lon, bearing_deg, speed_m_s * h * 3600)
        points.append({"lat": dest_lat, "lon": dest_lon})
    return points


def _risk_from_cluster(cluster: DebrisCluster, speed_m_s: float) -> str:
    if cluster.n_pixels >= 400 or cluster.mean_conf >= 0.88 or speed_m_s >= 0.35:
        return "CRITICAL"
    if cluster.n_pixels >= 150 or cluster.mean_conf >= 0.75 or speed_m_s >= 0.20:
        return "HIGH"
    if cluster.n_pixels >= 40 or cluster.mean_conf >= 0.60:
        return "MEDIUM"
    return "LOW"


def clusters_from_prediction(
    tif_path: Path,
    pred: np.ndarray,
    probs: np.ndarray,
    acquired_at: str,
    min_confidence: float = 0.50,
    eps_m: float = 120.0,
    min_samples: int = 3,
) -> list[DebrisCluster]:
    debris_mask = (pred == 1) & (probs[1] >= min_confidence)
    rows, cols = np.where(debris_mask)
    if len(rows) == 0:
        return []

    lats, lons = full_pixel_grid(str(tif_path))
    coords = np.column_stack([lats[rows, cols], lons[rows, cols]])
    confs = probs[1, rows, cols]

    labels = DBSCAN(
        eps=eps_m / EARTH_R,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    ).fit_predict(np.radians(coords))

    clusters: list[DebrisCluster] = []
    for label in sorted(set(labels) - {-1}):
        mask = labels == label
        cluster_coords = coords[mask]
        cluster_confs = confs[mask]
        clusters.append(
            DebrisCluster(
                id=int(label),
                lat=float(cluster_coords[:, 0].mean()),
                lon=float(cluster_coords[:, 1].mean()),
                n_pixels=int(mask.sum()),
                mean_conf=float(cluster_confs.mean()),
                max_conf=float(cluster_confs.max()),
                acquired_at=acquired_at,
                source_path=str(tif_path),
            )
        )

    clusters.sort(key=lambda cluster: (cluster.n_pixels, cluster.mean_conf), reverse=True)
    return clusters


def infer_sentinel_patch(patch: SentinelPatch) -> dict[str, Any]:
    model, device = _get_model()
    pred, probs, _img = infer_patch(model, patch.path, device)
    clusters = clusters_from_prediction(
        patch.path,
        pred,
        probs,
        acquired_at=patch.item.acquired_at,
    )
    return {
        "patch": patch.to_dict(),
        "acquired_at": patch.item.acquired_at,
        "diagnostics": patch_diagnostics(patch.path, pred, probs),
        "segmentation_overlay": segmentation_overlay(pred, patch),
        "debris_pixels": int(np.sum(pred == 1)),
        "clusters": clusters,
    }


def geotiff_wgs84_corners(tif_path: Path) -> list[tuple[float, float]]:
    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError(f"{tif_path.name} has no CRS. A georeferenced GeoTIFF is required.")

        pixel_corners = [
            (0, 0),
            (src.width, 0),
            (src.width, src.height),
            (0, src.height),
        ]

        if str(src.crs) == "EPSG:4326":
            corners = []
            for col, row in pixel_corners:
                lon, lat = src.transform * (col, row)
                corners.append((float(lon), float(lat)))
            return corners

        transformer = pyproj.Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        corners = []
        for col, row in pixel_corners:
            x, y = src.transform * (col, row)
            corners.append(transformer.transform(x, y))

    return [(float(lon), float(lat)) for lon, lat in corners]


def geotiff_wgs84_bounds(tif_path: Path) -> tuple[float, float, float, float]:
    corners = geotiff_wgs84_corners(tif_path)
    lons = [point[0] for point in corners]
    lats = [point[1] for point in corners]
    return float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats))


def latlon_corners_from_wgs84(corners_lonlat: list[tuple[float, float]]) -> list[list[float]]:
    return [[lat, lon] for lon, lat in corners_lonlat]


def segmentation_overlay_from_bounds(
    pred: np.ndarray,
    bbox_wgs84: tuple[float, float, float, float],
    acquired_at: str,
    corners_lonlat: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    rgba = np.zeros((pred.shape[0], pred.shape[1], 4), dtype=np.uint8)
    for class_id, color in CLASS_RGBA.items():
        rgba[pred == class_id] = color

    buf = io.BytesIO()
    try:
        from PIL import Image

        image = Image.fromarray(rgba, mode="RGBA")
        image.save(buf, format="PNG", optimize=True)
    except Exception:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.imsave(buf, rgba, format="png")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    west, south, east, north = bbox_wgs84

    return {
        "data_url": f"data:image/png;base64,{encoded}",
        "bounds": [[south, west], [north, east]],
        "corners": latlon_corners_from_wgs84(corners_lonlat) if corners_lonlat else None,
        "opacity": 0.72,
        "acquired_at": acquired_at,
    }


def segmentation_overlay_from_geotiff(pred: np.ndarray, tif_path: Path, acquired_at: str) -> dict[str, Any]:
    from rasterio.transform import array_bounds
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    with rasterio.open(tif_path) as src:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,
            "EPSG:4326",
            src.width,
            src.height,
            *src.bounds,
        )

        warped = np.zeros((dst_height, dst_width), dtype=np.uint8)
        reproject(
            source=pred.astype(np.uint8),
            destination=warped,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
        )

    west, south, east, north = array_bounds(dst_height, dst_width, dst_transform)
    bbox_wgs84 = (float(west), float(south), float(east), float(north))
    corners_lonlat = [
        (float(west), float(north)),
        (float(east), float(north)),
        (float(east), float(south)),
        (float(west), float(south)),
    ]
    return segmentation_overlay_from_bounds(
        warped,
        bbox_wgs84,
        acquired_at,
        corners_lonlat=corners_lonlat,
    )


def segmentation_overlay(pred: np.ndarray, patch: SentinelPatch) -> dict[str, Any]:
    return segmentation_overlay_from_bounds(pred, patch.bounds.bbox_wgs84, patch.item.acquired_at)


def patch_diagnostics(tif_path: Path, pred: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    with rasterio.open(tif_path) as src:
        img = src.read().astype(np.float32)
        profile = {
            "band_count": int(src.count),
            "width": int(src.width),
            "height": int(src.height),
            "dtype": str(src.dtypes[0]) if src.dtypes else "unknown",
            "crs": str(src.crs),
            "bounds": tuple(float(v) for v in src.bounds),
        }

    finite = np.isfinite(img)
    per_band = []
    for idx in range(img.shape[0]):
        band = img[idx]
        valid = np.isfinite(band)
        if np.any(valid):
            per_band.append(
                {
                    "band": idx + 1,
                    "min": round(float(np.nanmin(band)), 6),
                    "mean": round(float(np.nanmean(band)), 6),
                    "max": round(float(np.nanmax(band)), 6),
                }
            )
        else:
            per_band.append({"band": idx + 1, "min": None, "mean": None, "max": None})

    classes, counts = np.unique(pred, return_counts=True)
    class_counts = {str(int(cls)): int(count) for cls, count in zip(classes, counts)}

    debris_probs = probs[1]
    return {
        **profile,
        "all_finite": bool(np.all(finite)),
        "image_min": round(float(np.nanmin(img)), 6),
        "image_mean": round(float(np.nanmean(img)), 6),
        "image_max": round(float(np.nanmax(img)), 6),
        "per_band": per_band,
        "pred_class_counts": class_counts,
        "debris_prob_max": round(float(np.nanmax(debris_probs)), 6),
        "debris_prob_mean": round(float(np.nanmean(debris_probs)), 6),
        "debris_pixels_argmax": int(np.sum(pred == 1)),
        "debris_pixels_prob_gt_050": int(np.sum(debris_probs >= 0.50)),
        "debris_pixels_prob_gt_020": int(np.sum(debris_probs >= 0.20)),
    }


def infer_geotiff(tif_path: Path, acquired_at: str | None = None) -> dict[str, Any]:
    if acquired_at is None:
        acquired_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    with rasterio.open(tif_path) as src:
        if src.count != 11:
            raise ValueError(
                f"{tif_path.name} has {src.count} band(s). The MARIDA model expects exactly 11 bands "
                "(B01, B02, B03, B04, B05, B06, B07, B8A, B09, B11, B12)."
            )
        if src.width % 16 != 0 or src.height % 16 != 0:
            raise ValueError(
                f"{tif_path.name} is {src.width}x{src.height}. The U-Net needs width and height divisible by 16; "
                "256x256 is preferred because that is the MARIDA training patch size."
            )
        if src.crs is None:
            raise ValueError(f"{tif_path.name} has no CRS. Upload a georeferenced GeoTIFF.")

    model, device = _get_model()
    pred, probs, _img = infer_patch(model, tif_path, device)
    clusters = clusters_from_prediction(
        tif_path,
        pred,
        probs,
        acquired_at=acquired_at,
    )
    bbox_wgs84 = geotiff_wgs84_bounds(tif_path)
    corners_lonlat = geotiff_wgs84_corners(tif_path)
    west, south, east, north = bbox_wgs84

    return {
        "mode": "uploaded_geotiff",
        "patch": {
            "source_path": str(tif_path),
            "bbox_wgs84": bbox_wgs84,
            "corners": latlon_corners_from_wgs84(corners_lonlat),
            "center_lat": (south + north) / 2,
            "center_lon": (west + east) / 2,
        },
        "observation": {
            "acquired_at": acquired_at,
            "diagnostics": patch_diagnostics(tif_path, pred, probs),
            "segmentation_overlay": segmentation_overlay_from_geotiff(pred, tif_path, acquired_at),
            "debris_pixels": int(np.sum(pred == 1)),
            "clusters": [cluster.to_dict() for cluster in clusters],
            "source_path": str(tif_path),
        },
        "clusters": [cluster.to_dict() for cluster in clusters],
    }


def _match_clusters(
    older_clusters: list[DebrisCluster],
    newer_clusters: list[DebrisCluster],
    max_match_m: float = 50_000.0,
) -> list[tuple[DebrisCluster | None, DebrisCluster]]:
    pairs: list[tuple[DebrisCluster | None, DebrisCluster]] = []
    used_old: set[int] = set()

    for newer in newer_clusters:
        best_old = None
        best_dist = max_match_m
        for older in older_clusters:
            if older.id in used_old:
                continue
            dist = _haversine_m(older.lat, older.lon, newer.lat, newer.lon)
            if dist < best_dist:
                best_dist = dist
                best_old = older
        if best_old is not None:
            used_old.add(best_old.id)
        pairs.append((best_old, newer))

    return pairs


def build_hotspots_from_two_states(
    older_observation: dict[str, Any],
    newer_observation: dict[str, Any],
) -> list[dict[str, Any]]:
    older_clusters: list[DebrisCluster] = older_observation["clusters"]
    newer_clusters: list[DebrisCluster] = newer_observation["clusters"]
    pairs = _match_clusters(older_clusters, newer_clusters)

    older_time = _parse_time(older_observation["acquired_at"])
    newer_time = _parse_time(newer_observation["acquired_at"])
    dt_seconds = max((newer_time - older_time).total_seconds(), 1.0)
    dt_hours = dt_seconds / 3600

    hotspots = []
    for idx, (older, newer) in enumerate(pairs):
        if older is None:
            distance_m = 0.0
            bearing = 0.0
            speed = 0.0
            observed_path = [{"lat": newer.lat, "lon": newer.lon}]
            matched = False
        else:
            distance_m = _haversine_m(older.lat, older.lon, newer.lat, newer.lon)
            bearing = _bearing_deg(older.lat, older.lon, newer.lat, newer.lon)
            speed = distance_m / dt_seconds
            observed_path = _interpolate_path(older.lat, older.lon, newer.lat, newer.lon)
            matched = True

        forecast_path = _forecast_from_velocity(newer.lat, newer.lon, speed, bearing)
        risk = _risk_from_cluster(newer, speed)

        hotspots.append(
            {
                "id": f"DBR-{newer_time.strftime('%m%d%H%M')}-{idx}",
                "lat": newer.lat,
                "lon": newer.lon,
                "pixels": newer.n_pixels,
                "mass": f"~{max(0.1, newer.n_pixels * 0.03):.1f} t",
                "risk": risk,
                "confidence": round(newer.mean_conf, 3),
                "windSpeed": round(speed, 3),
                "windDir": int(round(bearing)),
                "currentDrift": f"{speed:.3f} m/s observed",
                "threat": "Observed two-state debris drift",
                "threatHours": int(round(dt_hours)),
                "vessel": "Cleanup asset assignment pending",
                "interceptTime": "TBD",
                "trajectory": forecast_path,
                "observedTrajectory": observed_path,
                "observedDrift": {
                    "matched": matched,
                    "distance_m": round(distance_m, 2),
                    "bearing_deg": round(bearing, 2),
                    "speed_m_s": round(speed, 4),
                    "hours_between_observations": round(dt_hours, 2),
                },
                "observations": {
                    "older": older.to_dict() if older else None,
                    "newer": newer.to_dict(),
                },
            }
        )

    return hotspots


def run_live_two_state_scan(
    top_left_lat: float,
    top_left_lon: float,
    lookback_days: int = 45,
    max_cloud_cover: float = 70.0,
) -> dict[str, Any]:
    patches = fetch_latest_two_patches(
        top_left_lat=top_left_lat,
        top_left_lon=top_left_lon,
        lookback_days=lookback_days,
        max_cloud_cover=max_cloud_cover,
    )
    older_patch, newer_patch = patches
    older_observation = infer_sentinel_patch(older_patch)
    newer_observation = infer_sentinel_patch(newer_patch)
    hotspots = build_hotspots_from_two_states(older_observation, newer_observation)

    return {
        "mode": "sentinelhub_two_state",
        "patch": newer_patch.bounds.to_dict(),
        "hotspots": hotspots,
        "observations": [
            {
                "acquired_at": older_observation["acquired_at"],
                "diagnostics": older_observation["diagnostics"],
                "segmentation_overlay": older_observation["segmentation_overlay"],
                "debris_pixels": older_observation["debris_pixels"],
                "clusters": [cluster.to_dict() for cluster in older_observation["clusters"]],
                "source_path": str(older_patch.path),
                "catalog_item": older_patch.item.to_dict(),
            },
            {
                "acquired_at": newer_observation["acquired_at"],
                "diagnostics": newer_observation["diagnostics"],
                "segmentation_overlay": newer_observation["segmentation_overlay"],
                "debris_pixels": newer_observation["debris_pixels"],
                "clusters": [cluster.to_dict() for cluster in newer_observation["clusters"]],
                "source_path": str(newer_patch.path),
                "catalog_item": newer_patch.item.to_dict(),
            },
        ],
    }

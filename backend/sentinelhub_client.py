"""
sentinelhub_client.py
=====================
Small Sentinel Hub client for fetching MARIDA-shaped Sentinel-2 patches.

The map click is interpreted as the top-left corner of a 256x256 patch.
Internally the patch is built in the local UTM CRS so each pixel is 10 m.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pyproj
import requests

from train_unet import Config


CFG = Config()

SENTINEL2_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B09",
    "B11",
    "B12",
]

EVALSCRIPT_11_BAND_FLOAT32 = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B01","B02","B03","B04","B05","B06","B07","B8A","B09","B11","B12"],
      units: "REFLECTANCE"
    }],
    output: {
      bands: 11,
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(s) {
  return [s.B01, s.B02, s.B03, s.B04, s.B05, s.B06, s.B07, s.B8A, s.B09, s.B11, s.B12];
}
""".strip()


class SentinelHubError(RuntimeError):
    """Base error for Sentinel Hub fetch failures."""


class MissingSentinelHubCredentials(SentinelHubError):
    """Raised when the required OAuth credentials are not configured."""


@dataclass(frozen=True)
class PatchBounds:
    top_left_lat: float
    top_left_lon: float
    epsg: int
    crs_uri: str
    bbox_utm: tuple[float, float, float, float]
    bbox_wgs84: tuple[float, float, float, float]
    width: int = 256
    height: int = 256
    resolution_m: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CatalogItem:
    item_id: str
    acquired_at: str
    cloud_cover: float | None
    bbox: list[float] | None

    @property
    def acquired_datetime(self) -> datetime:
        return parse_sentinel_time(self.acquired_at)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SentinelPatch:
    item: CatalogItem
    path: Path
    bounds: PatchBounds

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item.to_dict(),
            "path": str(self.path),
            "bounds": self.bounds.to_dict(),
        }


_TOKEN_CACHE: dict[str, Any] = {"token": None, "expires_at": 0.0}
_LOCAL_ENV_LOADED = False


def _base_url() -> str:
    return os.getenv("SENTINELHUB_BASE_URL", "https://services.sentinel-hub.com").rstrip("/")


def _token_url() -> str:
    return os.getenv(
        "SENTINELHUB_TOKEN_URL",
        "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token",
    )


def _cache_dir() -> Path:
    out = CFG.DATA_DIR / "sentinelhub_cache"
    out.mkdir(exist_ok=True)
    return out


def _load_local_env() -> None:
    global _LOCAL_ENV_LOADED
    if _LOCAL_ENV_LOADED:
        return

    env_path = CFG.DATA_DIR / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and (key.startswith("SENTINELHUB_") or key not in os.environ):
                os.environ[key] = value

    _LOCAL_ENV_LOADED = True


def _normalize_iso_datetime(value: str) -> str:
    text = value.replace("Z", "+00:00")

    # Some Sentinel Hub/CDSE responses use fractional seconds with fewer
    # than 6 digits, e.g. "2026-04-08T16:39:04.2+00:00". Older Python
    # runtimes can reject that, so pad/truncate to microsecond precision.
    match = re.match(r"^(.*T\d{2}:\d{2}:\d{2})\.(\d+)([+-]\d{2}:\d{2})$", text)
    if match:
        prefix, fraction, suffix = match.groups()
        text = f"{prefix}.{fraction[:6].ljust(6, '0')}{suffix}"

    return text


def parse_sentinel_time(value: str) -> datetime:
    text = _normalize_iso_datetime(value)
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def get_access_token() -> str:
    _load_local_env()
    now = time.time()
    if _TOKEN_CACHE["token"] and now < float(_TOKEN_CACHE["expires_at"]) - 60:
        return str(_TOKEN_CACHE["token"])

    client_id = os.getenv("SENTINELHUB_CLIENT_ID")
    client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise MissingSentinelHubCredentials(
            "Set SENTINELHUB_CLIENT_ID and SENTINELHUB_CLIENT_SECRET to enable live Sentinel-2 fetches."
        )

    resp = requests.post(
        _token_url(),
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if resp.status_code >= 400:
        raise SentinelHubError(f"Sentinel Hub token request failed: {resp.status_code} {resp.text[:500]}")

    payload = resp.json()
    token = payload["access_token"]
    expires_in = int(payload.get("expires_in", 300))
    _TOKEN_CACHE.update({"token": token, "expires_at": now + expires_in})
    return token


def utm_epsg_for_latlon(lat: float, lon: float) -> int:
    if lat < -80 or lat > 84:
        raise ValueError("UTM patch extraction is supported between 80S and 84N.")
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def build_top_left_patch_bounds(
    top_left_lat: float,
    top_left_lon: float,
    width: int = 256,
    height: int = 256,
    resolution_m: float = 10.0,
) -> PatchBounds:
    epsg = utm_epsg_for_latlon(top_left_lat, top_left_lon)
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_wgs84 = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    left, top = to_utm.transform(top_left_lon, top_left_lat)
    right = left + width * resolution_m
    bottom = top - height * resolution_m

    corners_lonlat = [
        to_wgs84.transform(left, top),
        to_wgs84.transform(right, top),
        to_wgs84.transform(right, bottom),
        to_wgs84.transform(left, bottom),
    ]
    lons = [p[0] for p in corners_lonlat]
    lats = [p[1] for p in corners_lonlat]

    return PatchBounds(
        top_left_lat=float(top_left_lat),
        top_left_lon=float(top_left_lon),
        epsg=epsg,
        crs_uri=f"http://www.opengis.net/def/crs/EPSG/0/{epsg}",
        bbox_utm=(float(left), float(bottom), float(right), float(top)),
        bbox_wgs84=(float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats))),
        width=width,
        height=height,
        resolution_m=resolution_m,
    )


def search_latest_sentinel2_items(
    bounds: PatchBounds,
    lookback_days: int = 45,
    max_cloud_cover: float = 70.0,
    limit: int = 30,
) -> list[CatalogItem]:
    token = get_access_token()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    payload: dict[str, Any] = {
        "bbox": list(bounds.bbox_wgs84),
        "datetime": f"{start.isoformat().replace('+00:00', 'Z')}/{end.isoformat().replace('+00:00', 'Z')}",
        "collections": ["sentinel-2-l2a"],
        "limit": limit,
        "filter": f"eo:cloud_cover <= {float(max_cloud_cover)}",
        "fields": {
            "include": ["id", "bbox", "properties.datetime", "properties.eo:cloud_cover"],
            "exclude": ["assets", "links"],
        },
    }

    resp = requests.post(
        f"{_base_url()}/api/v1/catalog/1.0.0/search",
        json=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/geo+json",
            "Content-Type": "application/json",
        },
        timeout=60,
    )
    if resp.status_code >= 400:
        raise SentinelHubError(f"Catalog search failed: {resp.status_code} {resp.text[:800]}")

    features = resp.json().get("features", [])
    items = []
    for feature in features:
        props = feature.get("properties", {})
        acquired_at = props.get("datetime")
        if not acquired_at:
            continue
        items.append(
            CatalogItem(
                item_id=str(feature.get("id", "unknown")),
                acquired_at=acquired_at,
                cloud_cover=props.get("eo:cloud_cover"),
                bbox=feature.get("bbox"),
            )
        )

    items.sort(key=lambda item: item.acquired_datetime, reverse=True)
    return _dedupe_acquisition_dates(items)


def _dedupe_acquisition_dates(items: list[CatalogItem], min_gap_hours: float = 12.0) -> list[CatalogItem]:
    selected: list[CatalogItem] = []
    for item in items:
        item_dt = item.acquired_datetime
        if all(abs((item_dt - prev.acquired_datetime).total_seconds()) >= min_gap_hours * 3600 for prev in selected):
            selected.append(item)
    return selected


def fetch_sentinel2_patch(
    item: CatalogItem,
    bounds: PatchBounds,
    max_cloud_cover: float = 70.0,
) -> SentinelPatch:
    token = get_access_token()
    item_dt = item.acquired_datetime
    day_start = item_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    digest = hashlib.sha1(
        f"{item.item_id}|{item.acquired_at}|{bounds.bbox_utm}|{bounds.epsg}".encode("utf-8")
    ).hexdigest()[:12]
    out_path = _cache_dir() / f"s2_{item_dt.strftime('%Y%m%dT%H%M%SZ')}_{digest}.tif"
    if out_path.exists() and out_path.stat().st_size > 0:
        return SentinelPatch(item=item, path=out_path, bounds=bounds)

    payload: dict[str, Any] = {
        "input": {
            "bounds": {
                "bbox": list(bounds.bbox_utm),
                "properties": {"crs": bounds.crs_uri},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": day_start.isoformat().replace("+00:00", "Z"),
                            "to": day_end.isoformat().replace("+00:00", "Z"),
                        },
                        "maxCloudCoverage": float(max_cloud_cover),
                        "mosaickingOrder": "leastCC",
                    },
                    "processing": {
                        "upsampling": "BILINEAR",
                        "downsampling": "BILINEAR",
                    },
                }
            ],
        },
        "output": {
            "width": bounds.width,
            "height": bounds.height,
            "responses": [
                {
                    "identifier": "default",
                    "format": {"type": "image/tiff"},
                }
            ],
        },
        "evalscript": EVALSCRIPT_11_BAND_FLOAT32,
    }

    resp = requests.post(
        f"{_base_url()}/api/v1/process",
        json=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "image/tiff",
            "Content-Type": "application/json",
        },
        timeout=120,
    )
    if resp.status_code >= 400:
        raise SentinelHubError(f"Process API fetch failed: {resp.status_code} {resp.text[:800]}")

    out_path.write_bytes(resp.content)
    return SentinelPatch(item=item, path=out_path, bounds=bounds)


def fetch_latest_two_patches(
    top_left_lat: float,
    top_left_lon: float,
    lookback_days: int = 45,
    max_cloud_cover: float = 70.0,
) -> list[SentinelPatch]:
    bounds = build_top_left_patch_bounds(top_left_lat, top_left_lon)
    items = search_latest_sentinel2_items(
        bounds=bounds,
        lookback_days=lookback_days,
        max_cloud_cover=max_cloud_cover,
    )
    if len(items) < 2:
        raise SentinelHubError(
            f"Need two Sentinel-2 L2A acquisitions, found {len(items)} in the last {lookback_days} days."
        )

    latest_two = items[:2]
    patches = [fetch_sentinel2_patch(item, bounds, max_cloud_cover=max_cloud_cover) for item in latest_two]
    return sorted(patches, key=lambda patch: patch.item.acquired_datetime)

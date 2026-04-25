"""
predict_trajectory.py
======================
Dynamic 72-hour drift forecasting for marine plastic debris.

Pipeline:
  1. Load GPS detections from inference_output/debris_gps_coordinates.csv
  2. Cluster nearby points into debris hotspots (DBSCAN)
  3. Fetch live 72-hour wind forecast from Open-Meteo API (free, no key)
  4. Simulate drift using the Leeway physics model
  5. Monte Carlo uncertainty cone (100 particles per cluster)
  6. Export -> trajectories_72hr.geojson  (for dashboards / GIS tools)
  7. Export -> trajectory_map.html        (interactive map, open in any browser)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import math
import csv
import time
import requests
import numpy as np
import folium
import folium.plugins
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from datetime import datetime, timezone
from sklearn.cluster import DBSCAN

# ── Paths ────────────────────────────────────────────────────
DATA_DIR   = Path(r"c:\Users\omtil\Downloads\MARIDA")
IN_CSV     = DATA_DIR / "inference_output" / "debris_gps_coordinates.csv"
OUT_DIR    = DATA_DIR / "trajectory_output"
OUT_DIR.mkdir(exist_ok=True)

# ── Physics constants ────────────────────────────────────────
EARTH_R    = 6_371_000.0   # metres
ALPHA      = 0.03          # wind leeway factor (3%)
DT_HOURS   = 1             # integration time step (hours)
FORECAST_H = 72            # total forecast window (hours)
N_PARTICLES = 100          # Monte-Carlo ensemble size per cluster
NOISE_STD   = 0.10         # ±10% noise injected each step

# ── Ocean current baseline (Caribbean Current approximation) ─
# These are reasonable background values for the Central America
# Caribbean coast region.  They are perturbed in Monte-Carlo.
BASE_U_CURRENT = 0.12   # m/s  eastward
BASE_V_CURRENT = 0.04   # m/s  northward


# ═══════════════════════════════════════════════════════════
#  STEP 1  -  Load GPS detections
# ═══════════════════════════════════════════════════════════
def load_detections(csv_path: Path) -> list[dict]:
    detections = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            detections.append({
                "patch"     : row["patch"],
                "lat"       : float(row["lat"]),
                "lon"       : float(row["lon"]),
                "confidence": float(row["confidence"]),
            })
    print(f"[1] Loaded {len(detections):,} debris GPS points from {csv_path.name}")
    return detections


# ═══════════════════════════════════════════════════════════
#  STEP 2  -  DBSCAN Clustering
# ═══════════════════════════════════════════════════════════
def cluster_detections(detections: list[dict],
                       eps_deg=0.02, min_samples=3):
    """
    Group nearby GPS points into debris hotspot clusters.
    eps_deg ≈ 0.02° ≈ 2 km at the equator.
    Returns a list of cluster dicts with centroid + pixel count.
    """
    coords = np.array([[d["lat"], d["lon"]] for d in detections])
    # Convert degrees to radians for DBSCAN (uses haversine-aware metric)
    coords_rad = np.radians(coords)
    eps_rad = eps_deg * (math.pi / 180)

    labels = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    ).fit_predict(coords_rad)

    clusters = []
    unique = set(labels) - {-1}
    for cid in sorted(unique):
        mask      = labels == cid
        pts       = coords[mask]
        confs     = np.array([d["confidence"] for d in detections])[mask]
        centroid  = pts.mean(axis=0)  # (lat, lon)
        clusters.append({
            "id"         : int(cid),
            "lat"        : float(centroid[0]),
            "lon"        : float(centroid[1]),
            "n_pixels"   : int(mask.sum()),
            "mean_conf"  : float(confs.mean()),
        })

    noise = int((labels == -1).sum())
    print(f"[2] Clustering -> {len(clusters)} hotspot(s)  "
          f"(noise/outlier pixels ignored: {noise})")
    for c in clusters:
        print(f"       Cluster {c['id']:>2}: "
              f"lat={c['lat']:.5f}, lon={c['lon']:.5f}  "
              f"pixels={c['n_pixels']:>4}  conf={c['mean_conf']:.3f}")
    return clusters


# ═══════════════════════════════════════════════════════════
#  STEP 3  -  Fetch wind from Open-Meteo
# ═══════════════════════════════════════════════════════════
def fetch_wind_forecast(lat: float, lon: float,
                        hours: int = FORECAST_H) -> list[tuple]:
    """
    Returns a list of (u_wind, v_wind) tuples in m/s for `hours` steps.
    Uses the free Open-Meteo API — no API key required.

    u_wind = eastward  component  (+east)
    v_wind = northward component  (+north)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude"       : lat,
        "longitude"      : lon,
        "hourly"         : "windspeed_10m,winddirection_10m",
        "wind_speed_unit": "ms",       # m/s directly
        "forecast_days"  : math.ceil(hours / 24) + 1,
        "timezone"       : "auto",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data  = resp.json()["hourly"]
        spds  = data["windspeed_10m"][:hours]      # m/s
        dirs  = data["winddirection_10m"][:hours]  # degrees FROM

        # Meteorological convention:  dir = "wind FROM north=0, east=90"
        # Eastward  u = -speed * sin(dir_rad)
        # Northward v = -speed * cos(dir_rad)
        uv = []
        for spd, d in zip(spds, dirs):
            dr  = math.radians(d)
            uv.append((-spd * math.sin(dr), -spd * math.cos(dr)))

        print(f"       Open-Meteo OK  "
              f"avg_speed={np.mean(spds):.2f} m/s  "
              f"avg_dir={np.mean(dirs):.0f} deg")
        return uv

    except Exception as exc:
        print(f"       [WARN] Open-Meteo request failed ({exc}). "
              f"Using constant fallback wind.")
        return [(1.5, -0.5)] * hours        # fallback: 1.5 m/s SE wind


# ═══════════════════════════════════════════════════════════
#  STEP 4 & 5  -  Euler integration + Monte Carlo
# ═══════════════════════════════════════════════════════════
def simulate_cluster(cluster: dict, wind_uv: list[tuple]) -> dict:
    """
    Run N_PARTICLES ensemble members for `FORECAST_H` hours.
    Returns per-particle tracks as lat/lon arrays.
    """
    lat0 = cluster["lat"]
    lon0 = cluster["lon"]
    rng  = np.random.default_rng(seed=cluster["id"])

    tracks = []   # list of (lat array, lon array) length FORECAST_H+1
    for _ in range(N_PARTICLES):
        lats = [lat0]
        lons = [lon0]
        lat, lon = lat0, lon0

        for h in range(FORECAST_H):
            u_w, v_w = wind_uv[h]

            # Perturb wind and current with gaussian noise
            noise = rng.normal(1.0, NOISE_STD, size=4)
            u_wind    = u_w * noise[0]
            v_wind    = v_w * noise[1]
            u_current = BASE_U_CURRENT * noise[2]
            v_current = BASE_V_CURRENT * noise[3]

            # Total velocity (m/s)
            u_total = u_current + ALPHA * u_wind
            v_total = v_current + ALPHA * v_wind

            # Convert m/s to degrees per time step
            dt_sec = DT_HOURS * 3600
            dlat   = (v_total * dt_sec) / EARTH_R * (180 / math.pi)
            dlon   = (u_total * dt_sec) / (EARTH_R * math.cos(math.radians(lat))) \
                     * (180 / math.pi)

            lat += dlat
            lon += dlon
            lats.append(lat)
            lons.append(lon)

        tracks.append((np.array(lats), np.array(lons)))

    return tracks   # list of N_PARTICLES x (lat_array, lon_array)


# ═══════════════════════════════════════════════════════════
#  STEP 6  -  Export GeoJSON
# ═══════════════════════════════════════════════════════════
def build_geojson(clusters: list[dict],
                  all_tracks: list,
                  wind_data: list) -> dict:
    """
    Builds a GeoJSON FeatureCollection.
    Each cluster contributes:
      - A Point feature for the detection hotspot
      - A LineString for the median trajectory
      - A Polygon for the 90th-percentile uncertainty cone
    """
    features = []
    now_str  = datetime.now(timezone.utc).isoformat()

    for cluster, tracks in zip(clusters, all_tracks):
        lat_matrix = np.stack([t[0] for t in tracks])  # (N, T+1)
        lon_matrix = np.stack([t[1] for t in tracks])

        med_lats = np.median(lat_matrix, axis=0)
        med_lons = np.median(lon_matrix, axis=0)

        p90_lat_lo = np.percentile(lat_matrix, 5,  axis=0)
        p90_lat_hi = np.percentile(lat_matrix, 95, axis=0)
        p90_lon_lo = np.percentile(lon_matrix, 5,  axis=0)
        p90_lon_hi = np.percentile(lon_matrix, 95, axis=0)

        # ── Detection hotspot point ──────────────────────────
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [cluster["lon"], cluster["lat"]],
            },
            "properties": {
                "type"       : "detection",
                "cluster_id" : cluster["id"],
                "n_pixels"   : cluster["n_pixels"],
                "confidence" : cluster["mean_conf"],
                "detected_at": now_str,
            },
        })

        # ── Median trajectory LineString ──────────────────────
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [float(lo), float(la)]
                    for la, lo in zip(med_lats, med_lons)
                ],
            },
            "properties": {
                "type"           : "median_trajectory",
                "cluster_id"     : cluster["id"],
                "forecast_hours" : FORECAST_H,
                "n_particles"    : N_PARTICLES,
            },
        })

        # ── 90% uncertainty cone (simplified as bbox polygon) ─
        cone_lats = (
            list(p90_lat_hi) + list(p90_lat_lo[::-1]) + [p90_lat_hi[0]]
        )
        cone_lons = (
            list(p90_lon_hi) + list(p90_lon_lo[::-1]) + [p90_lon_hi[0]]
        )
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [float(lo), float(la)]
                    for la, lo in zip(cone_lats, cone_lons)
                ]],
            },
            "properties": {
                "type"       : "uncertainty_cone_90pct",
                "cluster_id" : cluster["id"],
            },
        })

    return {"type": "FeatureCollection", "features": features}


# ═══════════════════════════════════════════════════════════
#  STEP 7  -  Interactive HTML map with Folium
# ═══════════════════════════════════════════════════════════
CLUSTER_COLORS = [
    "#FF4136", "#FF851B", "#FFDC00", "#2ECC40",
    "#7FDBFF", "#0074D9", "#B10DC9", "#F012BE",
    "#01FF70", "#FF69B4",
]

def build_folium_map(clusters, all_tracks, out_path: Path):
    if not clusters:
        return

    center_lat = np.mean([c["lat"] for c in clusters])
    center_lon = np.mean([c["lon"] for c in clusters])

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles="CartoDB dark_matter",
    )

    # Title box
    title_html = """
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
         z-index:9999;background:rgba(15,23,42,0.90);
         padding:10px 22px;border-radius:8px;
         font-family:'Segoe UI',sans-serif;color:#E2E8F0;
         border:1px solid #334155;text-align:center;">
      <b style="font-size:16px;">Marine Debris — 72-Hour Drift Forecast</b><br>
      <span style="font-size:11px;color:#94A3B8;">
        Leeway model &nbsp;|&nbsp; Open-Meteo wind &nbsp;|&nbsp;
        {n} ensemble particles/cluster
      </span>
    </div>""".format(n=N_PARTICLES)
    m.get_root().html.add_child(folium.Element(title_html))

    for cluster, tracks in zip(clusters, all_tracks):
        cid   = cluster["id"]
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]

        lat_matrix = np.stack([t[0] for t in tracks])
        lon_matrix = np.stack([t[1] for t in tracks])
        med_lats   = np.median(lat_matrix, axis=0)
        med_lons   = np.median(lon_matrix, axis=0)

        # ── Draw every ensemble member (thin, low opacity) ───
        for t in tracks[::5]:  # draw every 5th to keep map light
            folium.PolyLine(
                locations=list(zip(t[0], t[1])),
                color=color, weight=0.6, opacity=0.20,
            ).add_to(m)

        # ── Draw median trajectory ───────────────────────────
        folium.PolyLine(
            locations=list(zip(med_lats, med_lons)),
            color=color, weight=2.5, opacity=0.9,
            tooltip=f"Cluster {cid} — median trajectory",
        ).add_to(m)

        # ── Hourly markers every 24 h ────────────────────────
        for h in [24, 48, 72]:
            if h < len(med_lats):
                folium.CircleMarker(
                    location=[med_lats[h], med_lons[h]],
                    radius=5, color=color, fill=True,
                    fill_opacity=0.85,
                    tooltip=f"Cluster {cid} — T+{h}h",
                    popup=folium.Popup(
                        f"<b>Cluster {cid}</b><br>"
                        f"T+{h}h position<br>"
                        f"lat={med_lats[h]:.4f}<br>"
                        f"lon={med_lons[h]:.4f}",
                        max_width=160,
                    ),
                ).add_to(m)

        # ── Detection origin marker ──────────────────────────
        folium.CircleMarker(
            location=[cluster["lat"], cluster["lon"]],
            radius=9, color="#FF4136", fill=True,
            fill_color=color, fill_opacity=1.0,
            tooltip=(
                f"DETECTION — Cluster {cid}<br>"
                f"Pixels: {cluster['n_pixels']}<br>"
                f"Confidence: {cluster['mean_conf']:.2f}"
            ),
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;right:20px;z-index:9999;
         background:rgba(15,23,42,0.92);padding:12px 16px;
         border-radius:8px;font-family:'Segoe UI',sans-serif;
         color:#E2E8F0;border:1px solid #334155;font-size:11px;">
      <b>Legend</b><br><br>
      <span style="color:#FF4136;">&#9679;</span> Detection hotspot<br>
      <span style="opacity:0.5;">&#9135;</span> Ensemble particles (±10% noise)<br>
      <b>&#9135;</b> Median trajectory<br>
      &#11044; T+24h / T+48h / T+72h position
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(str(out_path))


# ═══════════════════════════════════════════════════════════
#  STEP 8  -  Matplotlib quick-check plot
# ═══════════════════════════════════════════════════════════
def save_static_plot(clusters, all_tracks, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    ax.tick_params(colors="#94A3B8")
    ax.spines[:].set_color("#334155")
    ax.set_xlabel("Longitude", color="#94A3B8")
    ax.set_ylabel("Latitude",  color="#94A3B8")
    ax.set_title("72-Hour Marine Debris Drift Forecast\n"
                 "(Leeway model + Open-Meteo wind + Monte Carlo)",
                 color="#E2E8F0", fontsize=12, fontweight="bold")

    handles = []
    for cluster, tracks in zip(clusters, all_tracks):
        cid   = cluster["id"]
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]

        lat_matrix = np.stack([t[0] for t in tracks])
        lon_matrix = np.stack([t[1] for t in tracks])
        med_lats   = np.median(lat_matrix, axis=0)
        med_lons   = np.median(lon_matrix, axis=0)
        p10_lats   = np.percentile(lat_matrix, 10, axis=0)
        p90_lats   = np.percentile(lat_matrix, 90, axis=0)
        p10_lons   = np.percentile(lon_matrix, 10, axis=0)
        p90_lons   = np.percentile(lon_matrix, 90, axis=0)

        # Ensemble spread
        for t in tracks[::10]:
            ax.plot(t[1], t[0], color=color, alpha=0.08, lw=0.7)

        # Uncertainty band (along median lon)
        ax.fill_betweenx(med_lats, p10_lons, p90_lons,
                         alpha=0.15, color=color)

        # Median trajectory
        ln, = ax.plot(med_lons, med_lats, color=color,
                      lw=2, label=f"Cluster {cid} "
                                   f"({cluster['n_pixels']} px, "
                                   f"conf={cluster['mean_conf']:.2f})")
        handles.append(ln)

        # Origin
        ax.scatter(cluster["lon"], cluster["lat"],
                   s=100, color=color, zorder=5, marker="*")

        # T+24, 48, 72
        for h, mk in zip([24, 48, 72], ["o", "s", "D"]):
            if h < len(med_lats):
                ax.scatter(med_lons[h], med_lats[h],
                           s=40, color=color, marker=mk, zorder=6)

    ax.legend(handles=handles, fontsize=8,
              facecolor="#1E293B", labelcolor="#E2E8F0",
              edgecolor="#334155")
    ax.grid(True, color="#1E293B", alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight",
                facecolor="#0F172A")
    plt.close()


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 60)
    print("  Marine Debris Trajectory Forecast")
    print(f"  Window: {FORECAST_H}h  |  Particles: {N_PARTICLES}  |"
          f"  Leeway: {ALPHA*100:.0f}%")
    print("=" * 60)

    # 1. Load
    detections = load_detections(IN_CSV)
    if not detections:
        print("No detections found. Run inference.py first.")
        return

    # 2. Cluster
    clusters = cluster_detections(detections)
    if not clusters:
        print("No clusters found — try lowering min_samples in DBSCAN.")
        return

    # 3-5. Wind + simulation per cluster
    print(f"\n[3-5] Fetching wind & simulating {FORECAST_H}h drift ...")
    all_tracks  = []
    wind_data   = []
    for c in clusters:
        print(f"      Cluster {c['id']} @ lat={c['lat']:.4f}, "
              f"lon={c['lon']:.4f}")
        uv = fetch_wind_forecast(c["lat"], c["lon"])
        wind_data.append(uv)
        tracks = simulate_cluster(c, uv)
        all_tracks.append(tracks)
        print(f"         Simulated {N_PARTICLES} particles OK")

    # 6. GeoJSON
    print("\n[6] Writing GeoJSON ...")
    geojson     = build_geojson(clusters, all_tracks, wind_data)
    geojson_out = OUT_DIR / "trajectories_72hr.geojson"
    with open(geojson_out, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"     Saved: {geojson_out}")

    # 7. Interactive map
    print("[7] Building interactive HTML map ...")
    map_out = OUT_DIR / "trajectory_map.html"
    build_folium_map(clusters, all_tracks, map_out)
    print(f"     Saved: {map_out}")

    # 8. Static plot
    print("[8] Generating static PNG plot ...")
    png_out = OUT_DIR / "trajectory_plot.png"
    save_static_plot(clusters, all_tracks, png_out)
    print(f"     Saved: {png_out}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  FORECAST COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Debris hotspots  : {len(clusters)}")
    print(f"  Total pixels     : {sum(c['n_pixels'] for c in clusters)}")
    print(f"  Forecast window  : {FORECAST_H} hours")
    print(f"  Ensemble size    : {N_PARTICLES} particles/cluster")
    print(f"  Wind source      : Open-Meteo API (live)")
    print(f"  Ocean current    : Caribbean baseline + {NOISE_STD*100:.0f}% noise")
    print(f"\n  Outputs in: {OUT_DIR}")
    print(f"  -> trajectories_72hr.geojson  (paste into geojson.io)")
    print(f"  -> trajectory_map.html        (open in any browser)")
    print(f"  -> trajectory_plot.png        (quick check)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

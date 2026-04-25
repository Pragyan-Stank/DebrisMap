<div align="center">
  <h1>🌊 JalRakshak: Autonomous Detection & Trajectory Mapping of Global Marine Debris</h1>
  <p>
    An end-to-end fullstack platform for detecting, geolocating, and forecasting marine macroplastic distribution using <b>Sentinel-2 Satellite Imagery</b>, <b>Deep Learning</b>, and <b>Physics-Guided Ocean Modeling</b>.
  </p>
  <p>
    <i>Built for the Hackathon</i>
  </p>
</div>

---

## 📋 Table of Contents
1. [Overview & Problem Statement](#-overview--problem-statement)
2. [System Architecture](#-system-architecture)
3. [Deep Learning Pipeline](#-deep-learning-pipeline)
4. [Biofouling & Trajectory Physics Models](#-biofouling--trajectory-physics-models)
5. [Development & Local Setup](#-development--local-setup)
6. [Datasets & Research References](#-datasets--research-references)

---

## 🌍 Overview & Problem Statement
Every year, over 8 million metric tonnes of plastic debris enter our oceans, choking delicate coastal biomes and breaking down into hazardous microplastics. 

**OceanEye** represents a proactive technological response. Instead of mapping garbage retroactively, our system directly intercepts **10m-resolution Sentinel-2 orbital imagery**, passes the 11-band optical signatures through a highly-optimized Lightweight U-Net Neural Network, and autonomously highlights active floating plastic patches on a high-fidelity geospatial dashboard.

We didn't just build an AI—we built an autonomous environmental extraction engine.

---

## ⚙️ System Architecture
The application is split across a powerful deep learning backend and a GPU-accelerated visualization tracking interface.

### 🌐 Frontend (React + Vite)
*   **Geospatial Rendering:** Uses `Deck.GL` (`react-map-gl`) over native ESRI World Imagery satellite tiles to plot hotspots with sub-pixel perfection.
*   **Interactive Patch-Scanning:** Features a custom **Drag-to-Draw** geometric bounding box tool, allowing users to select any global region for real-time spectral analysis.
*   **Dynamic Visualizations:** Renders layered intelligence including **Gaussian Heatmaps** for regional intensity, **Scatterplot overlays** for pixel-level detections, and **Density-Responsive Clusters** for macro-tracking.
*   **Performance:** UI optimized with `FlyToInterpolator` animations and a persistent model context to eliminate weight-loading latency during regional scans.

### 🧠 Backend (FastAPI + PyTorch)
*   **Inference API:** Robust RESTful HTTP endpoints that parse massive multi-spectral `.tif` bytes securely into memory.
*   **Sentinel-2 Live Services:** Integrated with a dynamic regional fetcher and a **Spectral Physics Pipeline** (FDI) that combines Deep Learning with optical index validation (0.7 * U-Net + 0.3 * FDI).
*   **Macro-Clustering Engine:** Implements **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** to group sparse pixel detections into actionable macro-debris fields.
*   **Vector Pipeline:** Actively reverse-projects local mapping grids (`UTM Easting/Northing`) natively into `WGS84 EPSG:4326 Lat/Lon` coordinates using `pyproj`.

---

## 🔬 Deep Learning Pipeline
The cornerstone of our system is a **Lightweight U-Net** fine-tuned on marine data:
- **Band Standardization:** The AI directly intercepts raw Near-Infrared (NIR) and Short-Wave Infrared (SWIR) satellite values and standardizes them using robust Z-score normalization scaling to isolate synthetic atmospheric anomalies.
- **Multiclass Segmentation:** Employs a combination of `Dice` + `Focal Loss` to overcome extreme class imbalance, reliably differentiating Marine Debris against 15 other environmental distractors (Clouds, Wakes, Dense Sargassum, Foam, etc.).
- **FDI Amplification:** Calculates the mathematical proxy `Floating Debris Index (FDI)` strictly using B8A, B6, and B11 Sentinel bands to enhance the ultimate signal-to-noise ratio.

---

## 🌊 Biofouling & Trajectory Physics Models
Marine plastic doesn't just float in place; it drifts, and it decays. Our backend actively corrects for physical environmental factors:

1. **Spatiotemporal Biofouling Decay 🦠**
   - Implements a custom exponential correction algorithm: $CF(t) = e^{-\alpha t}$. Over time, marine algae grows over floating synthetics, mutating their Near-Infrared optical signature. Our system predicts and offsets this signal masking.
   
2. **Dynamic 72-Hour Trajectory Forecast 🗺️**
   - Debris floats according to ocean laws. We integrated a simplified Runge-Kutta 4th order mapping module calculating $dX/dt = V_{current} + \alpha V_{wind} + V_{stokes}$. By factoring in a 3% wind leeway, the platform tracks precisely where the debris will crash over the next three days.

---

## 🚀 Development & Local Setup

### 1. Requirements
- Node.js (v18+)
- Python 3.10+
- NVIDIA GPU + CUDA toolkit (Highly Recommended for inference capability)

### 2. Backend Startup
```bash
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 3. Frontend Startup
```bash
cd frontend
npm install
npm run dev
```

> **Note**: For inference to work, ensure your trained weights (`best_model.pth`) are securely placed in `backend/trained_models/`. 

---

## 📚 Datasets & Research References

Our solution was trained utilizing the open-source **MARIDA** structure layout and was heavily inspired by the following foundational methodologies:

1. **MARIDA Dataset:** *Kikaki K, Kakogeorgiou I, et al. "MARIDA: A benchmark for Marine Debris detection from Sentinel-2 data." PLOS ONE 17(1), 2022.*
2. **Floating Debris Index:** *Biermann L., et al. "Finding Plastic Patches in Coastal Waters using Optical Satellite Data." Scientific Reports 10, 2020.*

---
<div align="center">
<i>Built to combat single-use plastics from atmospheric orbitals.</i>
</div>

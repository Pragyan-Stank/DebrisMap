import React, { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import DeckGL from '@deck.gl/react';
import { FlyToInterpolator } from '@deck.gl/core';
import { ScatterplotLayer, PolygonLayer, PathLayer, IconLayer } from '@deck.gl/layers';
import { Map } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { predictTrajectory, predictClusterTrajectories, predictDebris, fetchWeather } from '../services/api';
import { Navigation, Upload, MapPin, Wind, Clock, AlertTriangle, Shield, Anchor, ChevronRight, Crosshair, BarChart3, Activity, Thermometer, Eye, Droplets, Gauge, Download, Compass, CloudRain, Target, Plus, Info, FastForward, Play, Pause } from 'lucide-react';
import TerminalLoader from '../components/TerminalLoader';

const MAP_STYLE = {
  version: 8,
  sources: {
    satellite: { type: "raster", tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"], tileSize: 256 }
  },
  layers: [
    { id: "bg", type: "background", paint: { "background-color": "#020810" } },
    { id: "sat", type: "raster", source: "satellite", minzoom: 0, maxzoom: 19 }
  ]
};

const RISK_COLORS = {
  CRITICAL: { bg: '#dc262620', border: '#dc2626', text: '#fca5a5', dot: [220, 38, 38] },
  HIGH:     { bg: '#f9731620', border: '#f97316', text: '#fdba74', dot: [249, 115, 22] },
  MEDIUM:   { bg: '#eab30820', border: '#eab308', text: '#fde047', dot: [234, 179, 8] },
  LOW:      { bg: '#22c55e20', border: '#22c55e', text: '#86efac', dot: [34, 197, 94] },
};

const TRAJ_COLORS = [
  [0, 242, 255], [255, 60, 120], [255, 180, 0], [100, 255, 100],
  [180, 100, 255], [255, 100, 180], [0, 200, 150], [255, 220, 100],
];

// Generate animated wind particles
function generateWindParticles(center, windDeg, windSpeed, count = 60) {
  const particles = [];
  const rad = windDeg * Math.PI / 180;
  // Wind comes FROM windDeg, so particles move in opposite direction
  const dx = -Math.sin(rad);
  const dy = -Math.cos(rad);
  for (let i = 0; i < count; i++) {
    const spread = 0.8;
    const baseLon = center[0] + (Math.random() - 0.5) * spread;
    const baseLat = center[1] + (Math.random() - 0.5) * spread;
    const phase = Math.random();
    particles.push({ baseLon, baseLat, dx, dy, phase, speed: windSpeed });
  }
  return particles;
}

// Generate ocean current arrows
function generateCurrentArrows(center, count = 25) {
  const arrows = [];
  for (let i = 0; i < count; i++) {
    const lon = center[0] + (Math.random() - 0.5) * 0.6;
    const lat = center[1] + (Math.random() - 0.5) * 0.6;
    // Caribbean current flows roughly ESE
    const endLon = lon + 0.015;
    const endLat = lat + 0.005;
    arrows.push({ path: [[lon, lat], [endLon, endLat]] });
  }
  return arrows;
}

const Trajectory = () => {
  const [viewState, setViewState] = useState({ longitude: -86.33, latitude: 15.92, zoom: 3, pitch: 0, bearing: 0 });
  const [hotspots, setHotspots] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedHotspot, setSelectedHotspot] = useState(null);
  const [dropMode, setDropMode] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [weather, setWeather] = useState(null);
  const [weatherLoading, setWeatherLoading] = useState(false);
  const [animTick, setAnimTick] = useState(0);
  const [showWind, setShowWind] = useState(true);
  const [showCurrents, setShowCurrents] = useState(true);
  const [showCones, setShowCones] = useState(true);
  const [timeSlider, setTimeSlider] = useState(72);
  const animRef = useRef(null);

  // Animation loop for wind particles
  useEffect(() => {
    const step = () => {
      setAnimTick(t => t + 1);
      animRef.current = requestAnimationFrame(step);
    };
    animRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(animRef.current);
  }, []);

  // Fetch weather when we have a hotspot or after a drop
  const loadWeather = useCallback(async (lat, lon) => {
    setWeatherLoading(true);
    const w = await fetchWeather(lat, lon);
    if (w) setWeather(w);
    setWeatherLoading(false);
  }, []);

  const handleMapClick = async (info) => {
    if (!dropMode || !info.coordinate) return;
    setDropMode(false);
    setLoading(true);

    const [lon, lat] = info.coordinate;
    const result = await predictTrajectory(lat, lon, `DROP-${Date.now() % 10000}`);
    setLoading(false);

    if (result && result.hotspot) {
      setHotspots(prev => [...prev, result.hotspot]);
      loadWeather(lat, lon);
      setViewState(v => ({
        ...v, longitude: lon, latitude: lat, zoom: 8,
        transitionDuration: 2000, transitionInterpolator: new FlyToInterpolator()
      }));
    }
  };

  const handleTifUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setUploadLoading(true);

    const inferResult = await predictDebris(file);
    if (inferResult && inferResult.clusters && inferResult.clusters.length > 0) {
      const clusterData = inferResult.clusters.map(c => ({
        lat: c.center[0], lon: c.center[1],
        n_pixels: c.density * 10, mean_conf: 0.7,
      }));
      const trajResult = await predictClusterTrajectories(clusterData, "upload");
      if (trajResult && trajResult.hotspots) {
        setHotspots(prev => [...prev, ...trajResult.hotspots]);
        const first = trajResult.hotspots[0];
        if (first) {
          loadWeather(first.lat, first.lon);
          setViewState(v => ({
            ...v, longitude: first.lon, latitude: first.lat, zoom: 10,
            transitionDuration: 2500, transitionInterpolator: new FlyToInterpolator()
          }));
        }
      }
    } else if (inferResult && inferResult.points && inferResult.points.length > 0) {
      const clusterData = [{
        lat: inferResult.points[0].lat, lon: inferResult.points[0].lon,
        n_pixels: inferResult.points.length, mean_conf: inferResult.points[0].probability,
      }];
      const trajResult = await predictClusterTrajectories(clusterData, "upload");
      if (trajResult && trajResult.hotspots) {
        setHotspots(prev => [...prev, ...trajResult.hotspots]);
        loadWeather(clusterData[0].lat, clusterData[0].lon);
        setViewState(v => ({
          ...v, longitude: clusterData[0].lon, latitude: clusterData[0].lat, zoom: 10,
          transitionDuration: 2500, transitionInterpolator: new FlyToInterpolator()
        }));
      }
    }
    setUploadLoading(false);
  };

  const handleExportGeoJSON = () => {
    if (!hotspots.length) return;
    const features = [];
    hotspots.forEach(h => {
      features.push({ type: "Feature", geometry: { type: "Point", coordinates: [h.lon, h.lat] }, properties: { id: h.id, risk: h.risk, confidence: h.confidence, mass: h.mass } });
      if (h.trajectory) {
        const slice = h.trajectory.slice(0, timeSlider + 1);
        features.push({ type: "Feature", geometry: { type: "LineString", coordinates: slice.map(p => [p.lon, p.lat]) }, properties: { id: h.id, type: "trajectory", hours: timeSlider } });
      }
    });
    const geojson = { type: "FeatureCollection", features };
    const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `drift_forecast_${Date.now()}.geojson`; a.click();
  };

  // Wind particle positions (animated)
  const windParticles = useMemo(() => {
    if (!showWind || !weather || !hotspots.length) return [];
    const center = [hotspots[0].lon, hotspots[0].lat];
    const base = generateWindParticles(center, weather.current.wind_degree, weather.current.wind_kph);
    const t = animTick * 0.003;
    return base.map(p => {
      const progress = (p.phase + t * (p.speed / 30)) % 1;
      const travelDist = 0.4;
      return {
        position: [
          p.baseLon + p.dx * progress * travelDist,
          p.baseLat + p.dy * progress * travelDist,
        ],
        opacity: Math.sin(progress * Math.PI) * 180,
        size: 2 + p.speed / 10,
      };
    });
  }, [showWind, weather, hotspots, animTick]);

  const currentArrows = useMemo(() => {
    if (!showCurrents || !hotspots.length) return [];
    return generateCurrentArrows([hotspots[0].lon, hotspots[0].lat]);
  }, [showCurrents, hotspots]);

  // Build deck.gl layers
  const layers = useMemo(() => {
    const result = [];

    // Click catcher
    result.push(new PolygonLayer({
      id: 'click-catcher',
      data: [{ polygon: [[-180, 90], [180, 90], [180, -90], [-180, -90]] }],
      getPolygon: d => d.polygon, getFillColor: [0, 0, 0, 0],
      pickable: dropMode, onClick: handleMapClick, visible: dropMode
    }));

    // Ocean current arrows
    if (showCurrents && currentArrows.length) {
      result.push(new PathLayer({
        id: 'current-arrows',
        data: currentArrows,
        getPath: d => d.path,
        getColor: [0, 150, 200, 100],
        widthMinPixels: 2,
        jointRounded: true,
        getDashArray: [4, 4],
        dashJustified: true,
      }));
    }

    // Animated wind particles
    if (showWind && windParticles.length) {
      result.push(new ScatterplotLayer({
        id: `wind-particles-${animTick}`,
        data: windParticles,
        getPosition: d => d.position,
        getFillColor: d => [200, 230, 255, d.opacity],
        getRadius: d => d.size * 20,
        radiusMinPixels: 1.5,
        radiusMaxPixels: 4,
      }));
    }

    // Hotspot layers
    hotspots.forEach((h, i) => {
      const color = TRAJ_COLORS[i % TRAJ_COLORS.length];

      // Trajectory path (trimmed by time slider)
      if (h.trajectory && h.trajectory.length > 1) {
        const slice = h.trajectory.slice(0, timeSlider + 1);
        result.push(new PathLayer({
          id: `traj-${h.id}`,
          data: [{ path: slice.map(p => [p.lon, p.lat]) }],
          getPath: d => d.path, getColor: [...color, 200],
          widthMinPixels: 3, jointRounded: true, capRounded: true,
        }));
      }

      // Monte Carlo uncertainty cone
      if (showCones && h.monteCarlo) {
        const upper = h.monteCarlo.cone_upper.slice(0, timeSlider + 1).map(p => [p.lon, p.lat]);
        const lower = h.monteCarlo.cone_lower.slice(0, timeSlider + 1).map(p => [p.lon, p.lat]).reverse();
        const cone = [...upper, ...lower];
        if (cone.length > 2) {
          result.push(new PolygonLayer({
            id: `cone-${h.id}`, data: [{ polygon: cone }],
            getPolygon: d => d.polygon,
            getFillColor: [...color, 25], getLineColor: [...color, 60],
            lineWidthMinPixels: 1, stroked: true, filled: true,
          }));
        }
      }

      // Time position marker (current slider position)
      if (h.trajectory && h.trajectory[timeSlider]) {
        const tp = h.trajectory[timeSlider];
        result.push(new ScatterplotLayer({
          id: `timepos-${h.id}`,
          data: [tp],
          getPosition: d => [d.lon, d.lat],
          getFillColor: [...color, 255],
          getRadius: 150,
          radiusMinPixels: 6,
          stroked: true, getLineColor: [255, 255, 255], lineWidthMinPixels: 2,
        }));
      }

      // Milestone markers
      if (h.milestones) {
        const ms = [];
        if (timeSlider >= 24) ms.push({ ...h.milestones.t24, label: 'T+24' });
        if (timeSlider >= 48) ms.push({ ...h.milestones.t48, label: 'T+48' });
        if (timeSlider >= 72) ms.push({ ...h.milestones.t72, label: 'T+72' });
        result.push(new ScatterplotLayer({
          id: `mile-${h.id}`,
          data: ms, getPosition: d => [d.lon, d.lat],
          getFillColor: color, getRadius: 80, radiusMinPixels: 4,
          stroked: true, getLineColor: [255, 255, 255], lineWidthMinPixels: 1,
        }));
      }

      // Origin
      result.push(new ScatterplotLayer({
        id: `origin-${h.id}`,
        data: [h], getPosition: d => [d.lon, d.lat],
        getFillColor: [255, 65, 54], getLineColor: color,
        getRadius: 200, radiusMinPixels: 8,
        stroked: true, lineWidthMinPixels: 3,
        pickable: true, onClick: () => setSelectedHotspot(h),
      }));
    });

    return result;
  }, [hotspots, dropMode, animTick, windParticles, currentArrows, showWind, showCurrents, showCones, timeSlider]);

  const summary = useMemo(() => {
    if (!hotspots.length) return null;
    return {
      total: hotspots.length,
      critical: hotspots.filter(h => h.risk === 'CRITICAL').length,
      high: hotspots.filter(h => h.risk === 'HIGH').length,
      medium: hotspots.filter(h => h.risk === 'MEDIUM').length,
      low: hotspots.filter(h => h.risk === 'LOW').length,
    };
  }, [hotspots]);

  return (
    <div className="vis-container">
      <aside className="sidebar glass" style={{ overflowY: 'auto' }}>
        <div style={{ marginTop: '20px' }}>
          <h2 style={{ fontFamily: 'Outfit', color: '#fff', fontSize: '1.2rem' }}>
            <Navigation size={20} style={{ verticalAlign: 'text-bottom', marginRight: '8px' }} />
            Drift Forecast
          </h2>
          <p style={{ color: '#94a3b8', fontSize: '0.8rem', marginTop: '5px' }}>
            72h Leeway Physics + Open-Meteo Wind + Monte Carlo
          </p>
        </div>

        {/* Drop Tracker */}
        <div className="control-card">
          <div className="card-title"><Crosshair size={18} /> Manual Tracker</div>
          <p style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '15px' }}>
            Click on the ocean to drop a tracker and predict its 72h drift path.
          </p>
          <button className={`btn ${dropMode ? 'btn-danger' : 'btn-glow'}`}
            onClick={() => setDropMode(!dropMode)}
            style={{ width: '100%', justifyContent: 'center', backgroundColor: dropMode ? '#ef4444' : '' }}>
            <MapPin size={16} />
            {dropMode ? 'Cancel Drop' : 'Drop Tracker'}
          </button>
        </div>

        {/* Upload Card */}
        <div className="control-card">
          <div className="card-title"><Upload size={18} /> Track from Image</div>
          {uploadLoading ? (
            <TerminalLoader mode="trajectory" height="130px" />
          ) : (
            <div className="upload-zone">
              <input type="file" accept=".tif,.tiff" onChange={handleTifUpload} disabled={uploadLoading} />
              <div className="upload-zone-icon">
                <Upload size={18} color="#00f2ff" />
              </div>
              <div className="upload-zone-text">
                <><strong>Drop .TIF here</strong> or click to browse<br/><span style={{fontSize:'0.7rem'}}>Clusters will be auto-tracked for 72h</span></>
              </div>
            </div>
          )}
        </div>

        {/* Time Slider */}
        {hotspots.length > 0 && (
          <div className="control-card">
            <div className="card-title"><Clock size={18} /> Forecast Time</div>
            <div style={{ padding: '5px 0' }}>
              <input type="range" min={1} max={72} value={timeSlider} onChange={e => setTimeSlider(Number(e.target.value))}
                style={{ width: '100%', accentColor: '#00f2ff' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#64748b', marginTop: '4px' }}>
                <span>Now</span>
                <span style={{ color: '#00f2ff', fontWeight: 700, fontSize: '0.9rem' }}>T+{timeSlider}h</span>
                <span>72h</span>
              </div>
            </div>
          </div>
        )}

        {/* Live Weather Card */}
        {weather && (
          <div className="control-card">
            <div className="card-title"><CloudRain size={18} /> Live Weather</div>
            <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '10px' }}>
              {weather.location.name} · {weather.current.condition}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
              <div style={{ background: '#0a1016', padding: '8px', borderRadius: '8px', textAlign: 'center' }}>
                <Wind size={14} color="#00f2ff" />
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff' }}>{weather.current.wind_kph}</div>
                <div style={{ fontSize: '0.6rem', color: '#64748b' }}>km/h {weather.current.wind_dir}</div>
              </div>
              <div style={{ background: '#0a1016', padding: '8px', borderRadius: '8px', textAlign: 'center' }}>
                <Gauge size={14} color="#f97316" />
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff' }}>{weather.current.gust_kph}</div>
                <div style={{ fontSize: '0.6rem', color: '#64748b' }}>km/h Gust</div>
              </div>
              <div style={{ background: '#0a1016', padding: '8px', borderRadius: '8px', textAlign: 'center' }}>
                <Thermometer size={14} color="#eab308" />
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff' }}>{weather.current.temp_c}°</div>
                <div style={{ fontSize: '0.6rem', color: '#64748b' }}>Temperature</div>
              </div>
              <div style={{ background: '#0a1016', padding: '8px', borderRadius: '8px', textAlign: 'center' }}>
                <Droplets size={14} color="#3b82f6" />
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff' }}>{weather.current.humidity}%</div>
                <div style={{ fontSize: '0.6rem', color: '#64748b' }}>Humidity</div>
              </div>
              <div style={{ background: '#0a1016', padding: '8px', borderRadius: '8px', textAlign: 'center' }}>
                <Eye size={14} color="#22c55e" />
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff' }}>{weather.current.vis_km}</div>
                <div style={{ fontSize: '0.6rem', color: '#64748b' }}>km Visibility</div>
              </div>
              <div style={{ background: '#0a1016', padding: '8px', borderRadius: '8px', textAlign: 'center' }}>
                <Compass size={14} color="#a855f7" />
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff' }}>{weather.current.pressure_mb}</div>
                <div style={{ fontSize: '0.6rem', color: '#64748b' }}>hPa Pressure</div>
              </div>
            </div>
          </div>
        )}

        {/* Layer Toggles */}
        <div className="control-card">
          <div className="card-title"><Eye size={18} /> Layer Controls</div>
          {[
            { label: 'Wind Particles', state: showWind, setter: setShowWind, color: '#c8e6ff' },
            { label: 'Ocean Currents', state: showCurrents, setter: setShowCurrents, color: '#0096c8' },
            { label: 'Uncertainty Cones', state: showCones, setter: setShowCones, color: '#00f2ff' },
          ].map(l => (
            <div key={l.label} onClick={() => l.setter(!l.state)}
              style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', padding: '6px 0' }}>
              <div style={{ width: '12px', height: '12px', borderRadius: '3px', background: l.state ? l.color : '#333', border: `1px solid ${l.color}`, transition: 'all 0.2s' }} />
              <span style={{ fontSize: '0.8rem', color: l.state ? '#e2e8f0' : '#64748b' }}>{l.label}</span>
            </div>
          ))}
        </div>

        {/* Risk Summary */}
        {summary && (
          <div className="control-card">
            <div className="card-title"><BarChart3 size={18} /> Risk Summary</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
              {[
                { label: 'CRITICAL', count: summary.critical, color: '#dc2626' },
                { label: 'HIGH', count: summary.high, color: '#f97316' },
                { label: 'MEDIUM', count: summary.medium, color: '#eab308' },
                { label: 'LOW', count: summary.low, color: '#22c55e' },
              ].map(r => (
                <div key={r.label} style={{ background: '#0a1016', padding: '10px', borderRadius: '8px', textAlign: 'center', borderLeft: `3px solid ${r.color}` }}>
                  <div style={{ fontSize: '1.4rem', fontWeight: 700, color: r.color }}>{r.count}</div>
                  <div style={{ fontSize: '0.6rem', color: '#64748b', textTransform: 'uppercase' }}>{r.label}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Hotspot List */}
        {hotspots.length > 0 && (
          <div className="control-card">
            <div className="card-title"><AlertTriangle size={18} /> Active Hotspots ({hotspots.length})</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '240px', overflowY: 'auto' }}>
              {hotspots.map((h, i) => {
                const rc = RISK_COLORS[h.risk] || RISK_COLORS.LOW;
                return (
                  <div key={h.id}
                    onClick={() => {
                      setSelectedHotspot(h);
                      loadWeather(h.lat, h.lon);
                      setViewState(v => ({ ...v, longitude: h.lon, latitude: h.lat, zoom: 10, transitionDuration: 1500, transitionInterpolator: new FlyToInterpolator() }));
                    }}
                    style={{ background: rc.bg, padding: '10px', borderRadius: '8px', border: `1px solid ${rc.border}30`, cursor: 'pointer' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#e2e8f0' }}>{h.id}</span>
                      <span style={{ fontSize: '0.65rem', fontWeight: 700, color: rc.text, padding: '2px 6px', borderRadius: '4px', background: rc.bg }}>{h.risk}</span>
                    </div>
                    <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>
                      {h.lat.toFixed(4)}°, {h.lon.toFixed(4)}° · <Wind size={10} /> {h.windSpeed} m/s · {h.mass}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Actions */}
        {hotspots.length > 0 && (
          <div style={{ display: 'flex', gap: '8px' }}>
            <button className="btn glass" onClick={handleExportGeoJSON}
              style={{ flex: 1, justifyContent: 'center', borderColor: 'rgba(255,255,255,0.1)', fontSize: '0.75rem' }}>
              <Download size={14} /> GeoJSON
            </button>
            <button className="btn glass" onClick={() => { setHotspots([]); setSelectedHotspot(null); setWeather(null); }}
              style={{ flex: 1, justifyContent: 'center', borderColor: 'rgba(255,255,255,0.1)', color: '#ef4444', fontSize: '0.75rem' }}>
              Clear All
            </button>
          </div>
        )}

        {/* Engine Status */}
        <div className="control-card" style={{ marginTop: 'auto' }}>
          <div className="card-title"><Activity size={18} /> Forecast Engine</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {[
              { label: 'Leeway Model', status: '3% wind' },
              { label: 'Wind API', status: 'Open-Meteo' },
              { label: 'Weather API', status: weather ? 'Connected' : 'Standby' },
              { label: 'Monte Carlo', status: '50 particles' },
              { label: 'Forecast', status: '72 hours' },
            ].map(s => (
              <div key={s.label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem' }}>
                <span style={{ color: '#94a3b8' }}>{s.label}</span>
                <span style={{ color: '#10b981' }}>{s.status}</span>
              </div>
            ))}
          </div>
        </div>
      </aside>

      <main className="map-viewport">
        <DeckGL
          initialViewState={viewState}
          controller={true}
          layers={layers}
          onViewStateChange={({ viewState }) => setViewState(viewState)}
          onClick={dropMode ? handleMapClick : undefined}
          getCursor={() => dropMode ? 'crosshair' : 'grab'}
        >
          <Map mapStyle={MAP_STYLE} />
        </DeckGL>

        {/* Detail Panel */}
        {selectedHotspot && (
          <div style={{
            position: 'absolute', top: '20px', right: '20px', width: '320px',
            background: 'rgba(10,16,22,0.95)', border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '16px', padding: '20px', zIndex: 1000,
            boxShadow: '0 16px 64px rgba(0,0,0,0.6)',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h3 style={{ color: '#fff', fontFamily: 'Outfit', margin: 0 }}>{selectedHotspot.id}</h3>
              <span style={{
                fontSize: '0.7rem', fontWeight: 700, padding: '4px 10px', borderRadius: '6px',
                color: RISK_COLORS[selectedHotspot.risk]?.text,
                background: RISK_COLORS[selectedHotspot.risk]?.bg,
                border: `1px solid ${RISK_COLORS[selectedHotspot.risk]?.border}`,
              }}>{selectedHotspot.risk}</span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '16px' }}>
              {[
                { icon: <MapPin size={12} />, label: 'Position', value: `${selectedHotspot.lat.toFixed(4)}°, ${selectedHotspot.lon.toFixed(4)}°` },
                { icon: <Wind size={12} />, label: 'Wind', value: `${selectedHotspot.windSpeed} m/s @ ${selectedHotspot.windDir}°` },
                { icon: <Shield size={12} />, label: 'Signal', value: `${(selectedHotspot.confidence * 100).toFixed(0)}%` },
                { icon: <Anchor size={12} />, label: 'Mass', value: selectedHotspot.mass },
              ].map((item, i) => (
                <div key={i} style={{ background: '#0a1016', padding: '8px', borderRadius: '8px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.65rem', color: '#64748b', marginBottom: '2px' }}>
                    {item.icon} {item.label}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#e2e8f0', fontWeight: 600 }}>{item.value}</div>
                </div>
              ))}
            </div>

            {selectedHotspot.milestones && (
              <div style={{ marginBottom: '16px' }}>
                <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600, marginBottom: '8px', textTransform: 'uppercase' }}>
                  <Clock size={12} style={{ verticalAlign: 'text-bottom' }} /> Milestones
                </div>
                {['t24', 't48', 't72'].map(key => {
                  const m = selectedHotspot.milestones[key];
                  return (
                    <div key={key} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', padding: '3px 0', borderBottom: '1px solid #112233' }}>
                      <span style={{ color: '#94a3b8' }}>{key.replace('t', 'T+')}h</span>
                      <span style={{ color: '#e2e8f0', fontFamily: 'monospace' }}>{m.lat.toFixed(4)}°, {m.lon.toFixed(4)}°</span>
                    </div>
                  );
                })}
              </div>
            )}

            <div style={{ fontSize: '0.7rem', color: '#64748b', marginBottom: '6px' }}>
              <ChevronRight size={10} /> {selectedHotspot.currentDrift}
            </div>

            <button className="btn glass" onClick={() => setSelectedHotspot(null)}
              style={{ width: '100%', justifyContent: 'center', marginTop: '10px', borderColor: 'rgba(255,255,255,0.1)', color: '#94a3b8', fontSize: '0.8rem' }}>
              Dismiss
            </button>
          </div>
        )}

        {/* HUD */}
        <div className="vis-stats glass">
          <div className="stat-item">
            <div className="stat-value">{hotspots.length}</div>
            <div className="stat-label">Trackers</div>
          </div>
          {summary && (
            <>
              <div className="stat-item">
                <div className="stat-value" style={{ color: '#dc2626' }}>{summary.critical}</div>
                <div className="stat-label">Critical</div>
              </div>
              <div className="stat-item">
                <div className="stat-value" style={{ color: '#00f2ff' }}>T+{timeSlider}h</div>
                <div className="stat-label">Forecast</div>
              </div>
            </>
          )}
          {weather && (
            <div className="stat-item">
              <div className="stat-value" style={{ color: '#fbbf24' }}>{weather.current.wind_kph}</div>
              <div className="stat-label">km/h Wind</div>
            </div>
          )}
        </div>

        {/* Legend */}
        <div style={{
          position: 'absolute', bottom: '20px', left: '20px',
          padding: '10px 14px', borderRadius: '10px', zIndex: 100,
          background: 'rgba(10,16,22,0.85)', border: '1px solid rgba(255,255,255,0.1)', fontSize: '0.7rem'
        }}>
          <div style={{ fontWeight: 600, marginBottom: '6px', color: '#e2e8f0' }}>Legend</div>
          {[
            { color: '#FF4136', shape: 'circle', label: 'Detection Origin' },
            { color: '#00f2ff', shape: 'line', label: '72h Drift Path' },
            { color: 'rgba(0,242,255,0.2)', shape: 'square', label: 'Uncertainty Cone' },
            { color: '#c8e6ff', shape: 'dot', label: 'Wind Particles' },
            { color: '#0096c8', shape: 'line', label: 'Ocean Current' },
          ].map((item, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '3px' }}>
              {item.shape === 'circle' && <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: item.color }} />}
              {item.shape === 'line' && <div style={{ width: '20px', height: '3px', background: item.color, borderRadius: '2px' }} />}
              {item.shape === 'square' && <div style={{ width: '10px', height: '10px', borderRadius: '3px', background: item.color, border: '1px solid rgba(0,242,255,0.4)' }} />}
              {item.shape === 'dot' && <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: item.color, boxShadow: `0 0 6px ${item.color}` }} />}
              <span style={{ color: '#94a3b8' }}>{item.label}</span>
            </div>
          ))}
        </div>

        {dropMode && <div className="draw-instruction">CLICK ON THE OCEAN TO DROP A DEBRIS TRACKER</div>}

        {(loading || uploadLoading) && (
          <div style={{
            position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
            background: 'rgba(10,16,22,0.95)', padding: '24px 36px', borderRadius: '16px',
            border: '1px solid rgba(0,242,255,0.3)', zIndex: 1000, textAlign: 'center'
          }}>
            <div className="pulse-ring" style={{ width: '40px', height: '40px', margin: '0 auto 12px', borderRadius: '50%', border: '3px solid #00f2ff', animation: 'pulse 1.5s infinite' }} />
            <div style={{ color: '#00f2ff', fontSize: '0.9rem', fontWeight: 600 }}>
              {uploadLoading ? 'Detecting Debris & Computing Drift...' : 'Computing Drift Trajectory...'}
            </div>
            <div style={{ color: '#64748b', fontSize: '0.75rem', marginTop: '6px' }}>
              Fetching live wind data from Open-Meteo & WeatherAPI
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default Trajectory;

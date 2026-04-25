import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import DeckGL from '@deck.gl/react';
import { FlyToInterpolator } from '@deck.gl/core';
import { ScatterplotLayer, PolygonLayer, PathLayer } from '@deck.gl/layers';
import { HeatmapLayer } from '@deck.gl/aggregation-layers';
import { Map } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { fetchCleanupHotspots, fetchDispatchPlan, fetchPersistentZones, fetchInterceptPlan, fetchOptimalRoute, seedDemoData, fetchAIDispatch, fetchAIPersistentAnalysis, fetchAIInterceptAnalysis } from '../services/api';
import {
  Trash2, RefreshCw, Clock, AlertTriangle, MapPin, Eye, EyeOff, Shield, Target,
  BarChart3, Activity, ChevronRight, Zap, Radio, Ship, Anchor, Navigation, Crosshair,
  Upload, Fuel, Users, Timer, CircleDot
} from 'lucide-react';
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

const PRIORITY_COLORS = {
  HIGH:   { fill: [220, 38, 38],  line: [255, 80, 80],  bg: '#dc262618', border: '#dc2626', text: '#fca5a5' },
  MEDIUM: { fill: [234, 179, 8],  line: [255, 220, 60], bg: '#eab30818', border: '#eab308', text: '#fde047' },
  LOW:    { fill: [34, 197, 94],  line: [80, 255, 130],  bg: '#22c55e18', border: '#22c55e', text: '#86efac' },
};

const URGENCY_COLORS = {
  IMMEDIATE: { bg: '#dc262620', border: '#dc2626', text: '#fca5a5' },
  PRIORITY:  { bg: '#f9731620', border: '#f97316', text: '#fdba74' },
  SCHEDULED: { bg: '#eab30820', border: '#eab308', text: '#fde047' },
  ROUTINE:   { bg: '#22c55e20', border: '#22c55e', text: '#86efac' },
};

const Cleanup = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [timeWindow, setTimeWindow] = useState(72);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showClusters, setShowClusters] = useState(true);
  const [showPoints, setShowPoints] = useState(false);
  const [showMPA, setShowMPA] = useState(true);
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [highlightRegion, setHighlightRegion] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [viewState, setViewState] = useState({ longitude: -86.33, latitude: 15.92, zoom: 3, pitch: 0, bearing: 0 });
  const [activeTab, setActiveTab] = useState('zones');  // zones | dispatch | intercept | persistent
  const [dispatchData, setDispatchData] = useState(null);
  const [persistentData, setPersistentData] = useState(null);
  const [interceptResult, setInterceptResult] = useState(null);
  const [interceptLoading, setInterceptLoading] = useState(false);
  const [vesselLat, setVesselLat] = useState('');
  const [vesselLon, setVesselLon] = useState('');
  const [vesselSpeed, setVesselSpeed] = useState(22);
  const [interceptTarget, setInterceptTarget] = useState(null);
  const [optimalRoute, setOptimalRoute] = useState(null);
  const [routingLoading, setRoutingLoading] = useState(false);
  const [dispatchLoading, setDispatchLoading] = useState(false);
  const [deployDropMode, setDeployDropMode] = useState(false);
  const [vesselOrigin, setVesselOrigin] = useState(null);
  const intervalRef = useRef(null);
  // AI agent state
  const [aiDispatch, setAiDispatch] = useState(null);
  const [aiDispatchLoading, setAiDispatchLoading] = useState(false);
  const [aiIntercept, setAiIntercept] = useState(null);
  const [aiInterceptLoading, setAiInterceptLoading] = useState(false);
  const [aiPersist, setAiPersist] = useState(null);
  const [aiPersistLoading, setAiPersistLoading] = useState(false);
  const [seedLoading, setSeedLoading] = useState(false);

  // Known MPAs for map display
  const MPA_REGIONS = [
    { name: "Mesoamerican Reef", lat_min: 15.8, lat_max: 18.5, lon_min: -88.5, lon_max: -86.0 },
    { name: "Bay Islands MPA", lat_min: 16.2, lat_max: 16.6, lon_min: -86.8, lon_max: -85.8 },
    { name: "Sian Ka'an Reserve", lat_min: 19.2, lat_max: 20.0, lon_min: -88.0, lon_max: -87.2 },
  ];

  const loadData = useCallback(async (hours) => {
    setLoading(true);
    const result = await fetchCleanupHotspots(hours);
    if (result) setData(result);
    setLoading(false);
  }, []);

  const loadDispatch = useCallback(async () => {
    setDispatchLoading(true);
    const result = await fetchDispatchPlan(timeWindow);
    if (result) setDispatchData(result);
    setDispatchLoading(false);
  }, [timeWindow]);

  const loadPersistent = useCallback(async () => {
    const result = await fetchPersistentZones(168);
    if (result) setPersistentData(result);
  }, []);

  useEffect(() => { loadData(timeWindow); }, []);

  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(() => loadData(timeWindow), 10000);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [autoRefresh, timeWindow, loadData]);

  const handleTimeChange = (hours) => {
    setTimeWindow(hours);
    loadData(hours);
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    if (tab === 'dispatch' && !dispatchData) loadDispatch();
    if (tab === 'persistent' && !persistentData) loadPersistent();
  };

  const handleSeedData = async (clearExisting = false) => {
    setSeedLoading(true);
    const result = await seedDemoData(clearExisting);
    if (result) {
      await loadData(timeWindow);
      if (activeTab === 'dispatch') loadDispatch();
      if (activeTab === 'persistent') loadPersistent();
    }
    setSeedLoading(false);
  };

  const handleAIDispatch = async () => {
    setAiDispatchLoading(true);
    const result = await fetchAIDispatch(timeWindow);
    if (result) {
      if (result.dispatches) setDispatchData(result);
      setAiDispatch(result.ai_analysis || null);
    }
    setAiDispatchLoading(false);
  };

  const handleAIPersist = async () => {
    setAiPersistLoading(true);
    const result = await fetchAIPersistentAnalysis(168);
    if (result) {
      if (result.zones) setPersistentData(result);
      setAiPersist(result.ai_analysis || null);
    }
    setAiPersistLoading(false);
  };

  const handleAIIntercept = async () => {
    if (!interceptTarget || !vesselLat || !vesselLon) return;
    setAiInterceptLoading(true);
    const result = await fetchAIInterceptAnalysis(
      interceptTarget.lat, interceptTarget.lon,
      parseFloat(vesselLat), parseFloat(vesselLon), vesselSpeed
    );
    if (result) {
      if (result.intercept) setInterceptResult(result);
      setAiIntercept(result.ai_analysis || null);
    }
    setAiInterceptLoading(false);
  };

  const flyTo = (lat, lon, zoom = 11) => {
    setViewState(v => ({
      ...v, longitude: lon, latitude: lat, zoom,
      transitionDuration: 2000, transitionInterpolator: new FlyToInterpolator()
    }));
  };

  const handleIntercept = async () => {
    if (!interceptTarget || !vesselLat || !vesselLon) return;
    setInterceptLoading(true);
    const result = await fetchInterceptPlan(
      interceptTarget.lat, interceptTarget.lon,
      parseFloat(vesselLat), parseFloat(vesselLon), vesselSpeed
    );
    if (result) setInterceptResult(result);
    setInterceptLoading(false);
  };

  const handleComputeRoute = async (lat, lon) => {
    setRoutingLoading(true);
    const result = await fetchOptimalRoute(lat, lon, timeWindow);
    if (result) setOptimalRoute(result);
    setRoutingLoading(false);
  };

  // Build layers
  const layers = useMemo(() => {
    if (!data) return [];
    const result = [];

    // MPA boundaries
    if (showMPA) {
      MPA_REGIONS.forEach((mpa, i) => {
        result.push(new PolygonLayer({
          id: `mpa-${i}`,
          data: [{
            polygon: [
              [mpa.lon_min, mpa.lat_min], [mpa.lon_max, mpa.lat_min],
              [mpa.lon_max, mpa.lat_max], [mpa.lon_min, mpa.lat_max],
            ]
          }],
          getPolygon: d => d.polygon,
          getFillColor: [0, 200, 100, 15],
          getLineColor: [0, 200, 100, 120],
          lineWidthMinPixels: 2,
          stroked: true, filled: true,
        }));
      });
    }

    // Heatmap
    if (showHeatmap && data.raw_points && data.raw_points.length > 0) {
      result.push(new HeatmapLayer({
        id: 'cleanup-heatmap',
        data: data.raw_points,
        getPosition: d => [d.lon, d.lat],
        getWeight: d => d.probability || 0.5,
        radiusPixels: 50, intensity: 4, threshold: 0.04,
        colorRange: [
          [0, 25, 50], [0, 80, 120], [0, 180, 200],
          [255, 200, 0], [255, 120, 0], [255, 40, 40]
        ],
      }));
    }

    // Raw points
    if (showPoints && data.raw_points) {
      result.push(new ScatterplotLayer({
        id: 'cleanup-points',
        data: data.raw_points,
        getPosition: d => [d.lon, d.lat],
        getFillColor: [0, 242, 255, 120],
        getRadius: 20, radiusMinPixels: 2,
      }));
    }

    // Cluster zones
    if (showClusters && data.clusters) {
      data.clusters.forEach(cluster => {
        const pc = PRIORITY_COLORS[cluster.priority] || PRIORITY_COLORS.LOW;
        const isSelected = selectedCluster && selectedCluster.id === cluster.id;
        const circlePoints = 32;
        const radiusDeg = Math.max(0.003, (cluster.radius_m / 111320));
        const polygon = [];
        for (let i = 0; i <= circlePoints; i++) {
          const angle = (i / circlePoints) * Math.PI * 2;
          polygon.push([
            cluster.lon + radiusDeg * Math.cos(angle),
            cluster.lat + radiusDeg * Math.sin(angle) * 0.85
          ]);
        }
        result.push(new PolygonLayer({
          id: `cluster-zone-${cluster.id}`,
          data: [{ polygon }],
          getPolygon: d => d.polygon,
          getFillColor: [...pc.fill, isSelected ? 80 : 40],
          getLineColor: [...pc.line, isSelected ? 255 : 150],
          lineWidthMinPixels: isSelected ? 3 : 1.5,
          stroked: true, filled: true,
        }));
      });

      result.push(new ScatterplotLayer({
        id: 'cluster-centers',
        data: data.clusters,
        getPosition: d => [d.lon, d.lat],
        getFillColor: d => [...(PRIORITY_COLORS[d.priority]?.fill || [34,197,94]), 230],
        getRadius: d => Math.max(80, d.density * 3),
        radiusMinPixels: 6, stroked: true,
        getLineColor: [255, 255, 255], lineWidthMinPixels: 2,
        pickable: true,
        onClick: info => {
          if (info.object) {
            setSelectedCluster(info.object);
            setInterceptTarget(info.object);
            flyTo(info.object.lat, info.object.lon, 13);
          }
        },
      }));
    }

    // Intercept path
    if (interceptResult && interceptResult.trajectory) {
      result.push(new PathLayer({
        id: 'intercept-traj',
        data: [{ path: interceptResult.trajectory.map(p => [p.lon, p.lat]) }],
        getPath: d => d.path,
        getColor: [255, 180, 0, 200],
        widthMinPixels: 3, capRounded: true,
      }));

      // Intercept point
      if (interceptResult.intercept) {
        const ip = interceptResult.intercept;
        result.push(new ScatterplotLayer({
          id: 'intercept-point',
          data: [{ lon: ip.intercept_lon, lat: ip.intercept_lat }],
          getPosition: d => [d.lon, d.lat],
          getFillColor: [255, 220, 0, 255],
          getLineColor: [255, 255, 255],
          getRadius: 300, radiusMinPixels: 10,
          stroked: true, lineWidthMinPixels: 3,
        }));
      }

      // Vessel position
      if (vesselLat && vesselLon) {
        result.push(new ScatterplotLayer({
          id: 'vessel-pos',
          data: [{ lon: parseFloat(vesselLon), lat: parseFloat(vesselLat) }],
          getPosition: d => [d.lon, d.lat],
          getFillColor: [59, 130, 246, 255],
          getLineColor: [255, 255, 255],
          getRadius: 200, radiusMinPixels: 8,
          stroked: true, lineWidthMinPixels: 2,
        }));

        // Vessel-to-intercept line
        if (interceptResult.intercept) {
          result.push(new PathLayer({
            id: 'vessel-to-intercept',
            data: [{
              path: [
                [parseFloat(vesselLon), parseFloat(vesselLat)],
                [interceptResult.intercept.intercept_lon, interceptResult.intercept.intercept_lat],
              ]
            }],
            getPath: d => d.path,
            getColor: [59, 130, 246, 150],
            widthMinPixels: 2, getDashArray: [6, 4], dashJustified: true,
          }));
        }
      }
    }

    // Optimal route
    if (optimalRoute && optimalRoute.route && optimalRoute.route.length > 0 && vesselOrigin) {
      const vLat = vesselOrigin.lat;
      const vLon = vesselOrigin.lon;
      
      const routePoints = [[vLon, vLat], ...optimalRoute.route.map(r => [r.lon, r.lat])];
      
      result.push(new PathLayer({
        id: 'optimal-route-path',
        data: [{ path: routePoints }],
        getPath: d => d.path,
        getColor: [0, 242, 255, 200],
        widthMinPixels: 3, capRounded: true, jointRounded: true,
      }));

      result.push(new ScatterplotLayer({
        id: 'vessel-deploy-pos',
        data: [{ lon: vLon, lat: vLat }],
        getPosition: d => [d.lon, d.lat],
        getFillColor: [59, 130, 246, 255],
        getLineColor: [255, 255, 255],
        getRadius: 200, radiusMinPixels: 8,
        stroked: true, lineWidthMinPixels: 2,
      }));
    }

    // Highlight region
    if (highlightRegion) {
      const sz = 0.025;
      result.push(new PolygonLayer({
        id: 'highlight-region',
        data: [{
          polygon: [
            [highlightRegion.lon - sz, highlightRegion.lat - sz],
            [highlightRegion.lon + sz, highlightRegion.lat - sz],
            [highlightRegion.lon + sz, highlightRegion.lat + sz],
            [highlightRegion.lon - sz, highlightRegion.lat + sz],
          ]
        }],
        getPolygon: d => d.polygon,
        getFillColor: [255, 255, 0, 40],
        getLineColor: [255, 255, 0, 200],
        lineWidthMinPixels: 2, stroked: true, filled: true,
      }));
    }

    return result;
  }, [data, showHeatmap, showClusters, showPoints, showMPA, selectedCluster, highlightRegion, interceptResult, vesselLat, vesselLon, optimalRoute, vesselOrigin]);

  const isEmpty = data && data.status === 'empty';

  return (
    <div className="vis-container">
      <aside className="sidebar glass" style={{ overflowY: 'auto' }}>
        <div style={{ marginTop: '16px' }}>
          <h2 style={{ fontFamily: 'Outfit', color: '#fff', fontSize: '1.15rem' }}>
            <Trash2 size={20} style={{ verticalAlign: 'text-bottom', marginRight: '8px' }} />
            Clean-Up Programme
          </h2>
          <p style={{ color: '#94a3b8', fontSize: '0.75rem', marginTop: '4px' }}>
            Debris removal intelligence & coast guard ops
          </p>
        </div>

        {/* Time + Refresh */}
        <div className="control-card">
          <div className="card-title"><Clock size={16} /> Time Window</div>
          <div style={{ display: 'flex', gap: '5px', marginBottom: '10px' }}>
            {[24, 48, 72].map(h => (
              <button key={h}
                className={`btn ${timeWindow === h ? 'btn-glow' : 'btn-outline'}`}
                onClick={() => handleTimeChange(h)}
                style={{ flex: 1, justifyContent: 'center', fontSize: '0.75rem', padding: '6px 0' }}>
                {h}h
              </button>
            ))}
          </div>
          <div style={{ display: 'flex', gap: '5px' }}>
            <button className="btn btn-outline" onClick={() => loadData(timeWindow)}
              style={{ flex: 1, justifyContent: 'center', fontSize: '0.7rem', padding: '5px 0' }}>
              <RefreshCw size={12} className={loading ? 'spin' : ''} /> Refresh
            </button>
            <button className={`btn ${autoRefresh ? 'btn-glow' : 'btn-outline'}`}
              onClick={() => setAutoRefresh(!autoRefresh)}
              style={{ flex: 1, justifyContent: 'center', fontSize: '0.7rem', padding: '5px 0' }}>
              <Radio size={12} /> {autoRefresh ? 'ON' : 'OFF'}
            </button>
          </div>
        </div>

        {/* Summary */}
        {data && data.summary && (
          <div className="control-card">
            <div className="card-title"><BarChart3 size={16} /> Intel Summary</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
              <div style={{ background: '#0a1016', padding: '8px', borderRadius: '8px', textAlign: 'center' }}>
                <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#00f2ff' }}>{data.summary.total_points}</div>
                <div style={{ fontSize: '0.55rem', color: '#64748b', textTransform: 'uppercase' }}>Detections</div>
              </div>
              <div style={{ background: '#0a1016', padding: '8px', borderRadius: '8px', textAlign: 'center' }}>
                <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#e2e8f0' }}>{data.summary.total_clusters}</div>
                <div style={{ fontSize: '0.55rem', color: '#64748b', textTransform: 'uppercase' }}>Zones</div>
              </div>
              {[
                { label: 'Immediate', count: data.summary.high, color: '#dc2626' },
                { label: 'Deploy', count: data.summary.medium, color: '#eab308' },
              ].map(r => (
                <div key={r.label} style={{ background: '#0a1016', padding: '8px', borderRadius: '8px', textAlign: 'center', borderLeft: `3px solid ${r.color}` }}>
                  <div style={{ fontSize: '1.3rem', fontWeight: 700, color: r.color }}>{r.count}</div>
                  <div style={{ fontSize: '0.55rem', color: '#64748b', textTransform: 'uppercase' }}>{r.label}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Operation Tabs */}
        <div style={{ display: 'flex', gap: '3px', background: '#0a1016', padding: '3px', borderRadius: '10px' }}>
          {[
            { key: 'zones', icon: <Target size={12} />, label: 'Zones' },
            { key: 'dispatch', icon: <Ship size={12} />, label: 'Dispatch' },
            { key: 'intercept', icon: <Crosshair size={12} />, label: 'Intercept' },
            { key: 'route', icon: <Navigation size={12} />, label: 'Route' },
            { key: 'persistent', icon: <CircleDot size={12} />, label: 'Persist' },
          ].map(tab => (
            <button key={tab.key}
              onClick={() => handleTabChange(tab.key)}
              style={{
                flex: 1, padding: '6px 0', borderRadius: '8px', border: 'none', cursor: 'pointer',
                background: activeTab === tab.key ? 'rgba(0,242,255,0.15)' : 'transparent',
                color: activeTab === tab.key ? '#00f2ff' : '#64748b',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px',
                fontSize: '0.65rem', fontWeight: 600,
              }}>
              {tab.icon} {tab.label}
            </button>
          ))}
        </div>

        {/* ── TAB: Cleanup Zones ─────────────── */}
        {activeTab === 'zones' && (
          <>
            {/* Layers */}
            <div className="control-card">
              <div className="card-title"><Eye size={16} /> Map Layers</div>
              {[
                { label: 'Density Heatmap', state: showHeatmap, setter: setShowHeatmap, color: '#f97316' },
                { label: 'Cleanup Zones', state: showClusters, setter: setShowClusters, color: '#dc2626' },
                { label: 'Raw Detections', state: showPoints, setter: setShowPoints, color: '#00f2ff' },
                { label: 'Protected Areas', state: showMPA, setter: setShowMPA, color: '#22c55e' },
              ].map(l => (
                <div key={l.label} onClick={() => l.setter(!l.state)}
                  style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', padding: '4px 0' }}>
                  <div style={{ width: '10px', height: '10px', borderRadius: '3px', background: l.state ? l.color : '#333', border: `1px solid ${l.color}`, transition: 'all 0.2s' }} />
                  <span style={{ fontSize: '0.75rem', color: l.state ? '#e2e8f0' : '#64748b' }}>{l.label}</span>
                  {l.state ? <Eye size={10} color="#64748b" style={{ marginLeft: 'auto' }} /> : <EyeOff size={10} color="#334155" style={{ marginLeft: 'auto' }} />}
                </div>
              ))}
            </div>

            {/* Zone List */}
            {data && data.clusters && data.clusters.length > 0 && (
              <div className="control-card">
                <div className="card-title"><Target size={16} /> Cleanup Zones ({data.clusters.length})</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '5px', maxHeight: '220px', overflowY: 'auto' }}>
                  {data.clusters.map(c => {
                    const pc = PRIORITY_COLORS[c.priority] || PRIORITY_COLORS.LOW;
                    const isActive = selectedCluster && selectedCluster.id === c.id;
                    return (
                      <div key={c.id}
                        onClick={() => { setSelectedCluster(c); setInterceptTarget(c); flyTo(c.lat, c.lon, 13); }}
                        style={{
                          background: isActive ? pc.bg : '#0a101680', padding: '8px 10px', borderRadius: '8px',
                          border: `1px solid ${isActive ? pc.border : '#112233'}`, cursor: 'pointer',
                          borderLeft: `4px solid ${pc.border}`, transition: 'all 0.2s',
                        }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
                          <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#e2e8f0' }}>Zone #{c.id + 1}</span>
                          <span style={{ fontSize: '0.55rem', fontWeight: 700, color: pc.text, padding: '2px 6px', borderRadius: '4px', background: pc.bg }}>{c.priority}</span>
                        </div>
                        <div style={{ fontSize: '0.65rem', color: '#94a3b8' }}>
                          <MapPin size={9} style={{ verticalAlign: 'text-bottom' }} /> {c.lat.toFixed(4)}°, {c.lon.toFixed(4)}°
                        </div>
                        <div style={{ display: 'flex', gap: '8px', fontSize: '0.6rem', color: '#64748b', marginTop: '2px' }}>
                          <span>{c.density} pts</span>
                          <span>freq ×{c.frequency}</span>
                          {c.persistence && <span style={{ color: '#f97316' }}>● Persistent</span>}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Top Regions */}
            {data && data.top_regions && data.top_regions.length > 0 && (
              <div className="control-card">
                <div className="card-title"><Zap size={16} /> Top Debris Grounds</div>
                {data.top_regions.slice(0, 5).map((r, i) => (
                  <div key={i}
                    onClick={() => { setHighlightRegion(r); flyTo(r.lat, r.lon, 13); }}
                    style={{
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                      padding: '5px 6px', borderRadius: '6px', cursor: 'pointer', fontSize: '0.7rem',
                      background: highlightRegion === r ? 'rgba(255,255,0,0.1)' : 'transparent',
                    }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span style={{ fontWeight: 700, color: '#fbbf24' }}>#{i + 1}</span>
                      <span style={{ color: '#94a3b8', fontFamily: 'monospace', fontSize: '0.65rem' }}>{r.lat.toFixed(3)}°, {r.lon.toFixed(3)}°</span>
                    </div>
                    <span style={{ fontWeight: 600, color: '#e2e8f0' }}>{r.count}</span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {/* ── TAB: Dispatch Plan ─────────────── */}
        {activeTab === 'dispatch' && (
          <div className="control-card">
            <div className="card-title"><Ship size={16} /> Dispatch Plan</div>
            {!dispatchData ? (
              <div style={{ textAlign: 'center', padding: '16px 0' }}>
                <Activity size={20} className="spin" color="#00f2ff" />
                <div style={{ color: '#64748b', fontSize: '0.75rem', marginTop: '8px' }}>Loading dispatch plan...</div>
              </div>
            ) : dispatchData.dispatches?.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '16px 0', color: '#64748b', fontSize: '0.75rem' }}>
                No zones available for dispatch. Run a scan or load demo data first.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '320px', overflowY: 'auto' }}>
                {dispatchData.dispatches?.map((d, i) => {
                  const uc = URGENCY_COLORS[d.urgency] || URGENCY_COLORS.ROUTINE;
                  return (
                    <div key={i}
                      onClick={() => flyTo(d.lat, d.lon, 13)}
                      style={{
                        background: '#0a1016', padding: '10px 12px', borderRadius: '10px', cursor: 'pointer',
                        borderLeft: `4px solid ${uc.border}`, transition: 'all 0.2s',
                      }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', alignItems: 'center' }}>
                        <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#e2e8f0' }}>Zone #{d.zone_id + 1}</span>
                        <span style={{ fontSize: '0.55rem', fontWeight: 700, color: uc.text, padding: '2px 8px', borderRadius: '4px', background: uc.bg }}>
                          {d.urgency}
                        </span>
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', marginBottom: '8px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.7rem', color: '#94a3b8' }}><Ship size={11} /> {d.assigned_vessel}</div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.7rem', color: '#94a3b8' }}><Users size={11} /> {d.vessel_crew} crew</div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.7rem', color: '#94a3b8' }}><Navigation size={11} /> {d.vessel_speed_knots} kn</div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.7rem', color: '#94a3b8' }}><Timer size={11} /> ~{d.estimated_cleanup_hours}h cleanup</div>
                      </div>
                      {d.threats && d.threats.length > 0 && (
                        <div style={{ padding: '5px 8px', borderRadius: '6px', fontSize: '0.65rem', background: '#dc262615', border: '1px solid #dc262630', color: '#fca5a5', marginBottom: '6px' }}>
                          <AlertTriangle size={10} style={{ verticalAlign: 'text-bottom', marginRight: '4px' }} />{d.threats[0].message}
                        </div>
                      )}
                      {d.nearest_mpa && (
                        <div style={{ fontSize: '0.65rem', color: '#22c55e', display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '4px' }}>
                          <Shield size={10} /> {d.nearest_mpa} — {d.nearest_mpa_km}km
                        </div>
                      )}
                      <div style={{ fontSize: '0.65rem', color: '#64748b', lineHeight: 1.4 }}>
                        <ChevronRight size={10} style={{ verticalAlign: 'text-bottom' }} /> {d.recommended_action}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
            <div style={{ display: 'flex', gap: '5px', marginTop: '10px' }}>
              <button className={`btn ${dispatchLoading ? '' : 'btn-outline'}`} onClick={loadDispatch} disabled={dispatchLoading}
                style={{ flex: 1, justifyContent: 'center', fontSize: '0.65rem', padding: '6px 0', opacity: dispatchLoading ? 0.7 : 1 }}>
                <RefreshCw size={11} className={dispatchLoading ? 'spin' : ''} />
                {dispatchLoading ? 'Generating...' : 'Regenerate'}
              </button>
              <button className={`btn ${aiDispatchLoading ? '' : 'btn-glow'}`} onClick={handleAIDispatch} disabled={aiDispatchLoading}
                style={{ flex: 1, justifyContent: 'center', fontSize: '0.65rem', padding: '6px 0', opacity: aiDispatchLoading ? 0.7 : 1 }}>
                <Zap size={11} className={aiDispatchLoading ? 'spin' : ''} />
                {aiDispatchLoading ? 'Analyzing...' : 'AI Briefing'}
              </button>
            </div>
            {/* AI NAUTILUS Briefing Panel */}
            {aiDispatch && aiDispatch.analysis && (
              <div style={{ marginTop: '10px', padding: '10px 12px', borderRadius: '10px', background: 'rgba(0,242,255,0.05)', border: '1px solid rgba(0,242,255,0.2)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <span style={{ fontSize: '0.65rem', fontWeight: 700, color: '#00f2ff', textTransform: 'uppercase', letterSpacing: '0.08em' }}>⬡ NAUTILUS AI Briefing</span>
                  <span style={{ fontSize: '0.55rem', padding: '2px 6px', borderRadius: '4px',
                    color: aiDispatch.analysis.threat_level === 'CRITICAL' ? '#fca5a5' : aiDispatch.analysis.threat_level === 'HIGH' ? '#fdba74' : '#fde047',
                    background: aiDispatch.analysis.threat_level === 'CRITICAL' ? '#dc262620' : '#f9731620',
                    border: `1px solid ${aiDispatch.analysis.threat_level === 'CRITICAL' ? '#dc262650' : '#f9731650'}`,
                  }}>{aiDispatch.analysis.threat_level}</span>
                </div>
                <p style={{ fontSize: '0.68rem', color: '#cbd5e1', lineHeight: 1.55, margin: '0 0 8px' }}>{aiDispatch.analysis.sitrep}</p>
                {aiDispatch.analysis.recommendations?.length > 0 && (
                  <div>
                    <div style={{ fontSize: '0.55rem', color: '#64748b', textTransform: 'uppercase', marginBottom: '4px' }}>Recommendations</div>
                    {aiDispatch.analysis.recommendations.map((rec, i) => (
                      <div key={i} style={{ display: 'flex', gap: '6px', fontSize: '0.65rem', color: '#94a3b8', marginBottom: '3px', lineHeight: 1.4 }}>
                        <span style={{ color: '#00f2ff', flexShrink: 0 }}>›</span><span>{rec}</span>
                      </div>
                    ))}
                  </div>
                )}
                {aiDispatch.analysis.estimated_ops_window && (
                  <div style={{ fontSize: '0.6rem', color: '#64748b', marginTop: '6px', borderTop: '1px solid #1e293b', paddingTop: '6px' }}>
                    <Timer size={9} style={{ verticalAlign: 'text-bottom', marginRight: '4px' }} />
                    Est. ops window: {aiDispatch.analysis.estimated_ops_window}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ── TAB: Intercept Planner ─────────── */}
        {activeTab === 'intercept' && (
          <div className="control-card">
            <div className="card-title"><Crosshair size={16} /> Vessel Intercept Planner</div>
            <p style={{ fontSize: '0.7rem', color: '#64748b', marginBottom: '10px' }}>
              Enter vessel position and speed to compute the optimal intercept point along the debris drift path.
            </p>

            {interceptTarget ? (
              <div style={{ background: '#0a1016', padding: '8px 10px', borderRadius: '8px', marginBottom: '10px', fontSize: '0.7rem' }}>
                <div style={{ color: '#64748b', fontSize: '0.6rem', textTransform: 'uppercase', marginBottom: '3px' }}>Target Debris</div>
                <div style={{ color: '#e2e8f0' }}>Zone #{interceptTarget.id + 1} — {interceptTarget.lat?.toFixed(4)}°, {interceptTarget.lon?.toFixed(4)}°</div>
              </div>
            ) : (
              <div style={{ background: '#f9731610', padding: '8px', borderRadius: '8px', fontSize: '0.7rem', color: '#fdba74', marginBottom: '10px' }}>
                Click a cleanup zone on the map to set target
              </div>
            )}

            <div className="input-group">
              <span className="input-label">Vessel Latitude</span>
              <input type="number" step="0.001" placeholder="e.g. 16.5" value={vesselLat}
                onChange={e => setVesselLat(e.target.value)} />
            </div>
            <div className="input-group">
              <span className="input-label">Vessel Longitude</span>
              <input type="number" step="0.001" placeholder="e.g. -87.0" value={vesselLon}
                onChange={e => setVesselLon(e.target.value)} />
            </div>
            <div className="input-group">
              <span className="input-label">Speed (knots)</span>
              <select value={vesselSpeed} onChange={e => setVesselSpeed(Number(e.target.value))}>
                <option value={8}>8 kn — Cleanup Barge</option>
                <option value={18}>18 kn — Offshore Patrol</option>
                <option value={22}>22 kn — Patrol Boat</option>
                <option value={28}>28 kn — Fast Response Cutter</option>
              </select>
            </div>

            {/* Compute buttons + AI assessment */}

            {interceptResult && interceptResult.intercept && (
              <div style={{ marginTop: '12px' }}>
                <div style={{ fontSize: '0.7rem', color: '#00f2ff', fontWeight: 700, marginBottom: '8px', textTransform: 'uppercase' }}>Intercept Solution</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
                  {[
                    { icon: <Timer size={11} />, label: 'Arrival', value: `T+${interceptResult.intercept.intercept_hour}h` },
                    { icon: <Navigation size={11} />, label: 'Distance', value: `${interceptResult.intercept.vessel_travel_km} km` },
                    { icon: <Timer size={11} />, label: 'Transit', value: `${interceptResult.intercept.vessel_travel_hours}h` },
                    { icon: <Fuel size={11} />, label: 'Fuel Est.', value: `${interceptResult.intercept.fuel_estimate_liters}L` },
                  ].map((item, i) => (
                    <div key={i} style={{ background: '#0a1016', padding: '6px 8px', borderRadius: '6px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.55rem', color: '#64748b' }}>{item.icon} {item.label}</div>
                      <div style={{ fontSize: '0.8rem', color: '#e2e8f0', fontWeight: 600 }}>{item.value}</div>
                    </div>
                  ))}
                </div>
                <div style={{ marginTop: '8px', fontSize: '0.7rem', color: '#94a3b8', fontFamily: 'monospace', background: '#0a1016', padding: '6px 8px', borderRadius: '6px' }}>
                  <MapPin size={10} style={{ verticalAlign: 'text-bottom' }} /> Intercept: {interceptResult.intercept.intercept_lat?.toFixed(4)}°, {interceptResult.intercept.intercept_lon?.toFixed(4)}°
                </div>
                {interceptResult.intercept.warning && (
                  <div style={{ marginTop: '6px', padding: '5px 8px', borderRadius: '6px', background: '#f9731615', border: '1px solid #f9731640', fontSize: '0.65rem', color: '#fdba74' }}>
                    <AlertTriangle size={10} style={{ verticalAlign: 'text-bottom' }} /> {interceptResult.intercept.warning}
                  </div>
                )}
              </div>
            )}
              <div style={{ display: 'flex', gap: '5px', marginTop: '10px' }}>
                <button className="btn btn-glow" onClick={handleIntercept}
                  disabled={!interceptTarget || !vesselLat || !vesselLon || interceptLoading}
                  style={{ flex: 1, justifyContent: 'center', fontSize: '0.65rem', padding: '6px 0', opacity: (!interceptTarget || !vesselLat || !vesselLon) ? 0.5 : 1 }}>
                  {interceptLoading ? <><Activity size={12} className="spin" /> Computing...</> : <><Navigation size={12} /> Intercept</>}
                </button>
                <button className="btn btn-outline" onClick={handleAIIntercept}
                  disabled={!interceptTarget || !vesselLat || !vesselLon || aiInterceptLoading}
                  style={{ flex: 1, justifyContent: 'center', fontSize: '0.65rem', padding: '6px 0', opacity: (!interceptTarget || !vesselLat || !vesselLon) ? 0.4 : 1 }}>
                  {aiInterceptLoading ? <><Activity size={12} className="spin" /> Thinking...</> : <><Zap size={12} /> AI Assess</>}
                </button>
              </div>
              {/* AI intercept assessment */}
              {aiIntercept && aiIntercept.analysis && (
                <div style={{ marginTop: '10px', padding: '10px 12px', borderRadius: '10px', background: 'rgba(0,242,255,0.05)', border: '1px solid rgba(0,242,255,0.2)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                    <span style={{ fontSize: '0.6rem', fontWeight: 700, color: '#00f2ff', textTransform: 'uppercase' }}>⬡ Tactical Assessment</span>
                    <span style={{ fontSize: '0.55rem', padding: '2px 5px', borderRadius: '4px', color: '#86efac', background: '#22c55e15', border: '1px solid #22c55e40' }}>{aiIntercept.analysis.feasibility}</span>
                  </div>
                  <p style={{ fontSize: '0.68rem', color: '#cbd5e1', lineHeight: 1.5, margin: '0 0 6px' }}>{aiIntercept.analysis.tactical_assessment}</p>
                  <div style={{ fontSize: '0.62rem', color: '#94a3b8' }}><ChevronRight size={9} style={{ verticalAlign: 'text-bottom' }} /> {aiIntercept.analysis.recommended_action}</div>
                  {aiIntercept.analysis.risk_window && (
                    <div style={{ fontSize: '0.6rem', color: '#f97316', marginTop: '4px' }}>
                      <AlertTriangle size={9} style={{ verticalAlign: 'text-bottom' }} /> Window: {aiIntercept.analysis.risk_window}
                    </div>
                  )}
                </div>
              )}
          </div>
        )}

        {/* ── TAB: Optimal Route ──────────── */}
        {activeTab === 'route' && (
          <div className="control-card">
            <div className="card-title"><Navigation size={16} /> Optimal Sweep Route</div>
            <p style={{ fontSize: '0.7rem', color: '#64748b', marginBottom: '10px' }}>
              Deploy a cleanup vessel and instantly compute the shortest-path sweep route across all active debris zones.
            </p>
            
            <button className={`btn ${deployDropMode ? 'btn-danger' : 'btn-glow'}`}
              onClick={() => setDeployDropMode(!deployDropMode)}
              style={{ width: '100%', justifyContent: 'center', backgroundColor: deployDropMode ? '#dc2626' : '', marginBottom: '10px' }}>
              <MapPin size={16} />
              {deployDropMode ? 'Cancel Deployment' : 'Deploy cleanup vessel here'}
            </button>

            {routingLoading && (
              <div style={{ marginTop: '10px' }}>
                <TerminalLoader mode="cleanup" height="150px" />
              </div>
            )}

            {optimalRoute && optimalRoute.route && (
              <div style={{ marginTop: '10px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '0.75rem', fontWeight: 700, color: '#e2e8f0' }}>
                  <span>Optimal Route Plan</span>
                  <span style={{ color: '#00f2ff' }}>{optimalRoute.total_distance_km} km total</span>
                </div>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', maxHeight: '250px', overflowY: 'auto' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '6px', background: '#3b82f620', borderRadius: '6px', border: '1px solid #3b82f650' }}>
                     <Ship size={12} color="#3b82f6" />
                     <span style={{ fontSize: '0.65rem', color: '#bfdbfe' }}>Vessel Deployment Point</span>
                  </div>
                  {optimalRoute.route.map((r, i) => (
                    <div key={i}
                      onClick={() => flyTo(r.lat, r.lon, 13)}
                      style={{ display: 'flex', background: '#0a1016', borderRadius: '6px', overflow: 'hidden', cursor: 'pointer', border: '1px solid #112233' }}>
                      <div style={{ width: '3px', background: '#00f2ff' }}></div>
                      <div style={{ padding: '6px', width: '100%' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: '0.65rem', fontWeight: 600, color: '#e2e8f0' }}>{i+1}. Zone #{r.id + 1}</span>
                          <span style={{ fontSize: '0.6rem', color: '#94a3b8' }}>+{r.distance_from_prev_km} km</span>
                        </div>
                      </div>
                    </div>
                  ))}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '6px', background: '#10b98120', borderRadius: '6px', border: '1px solid #10b98150', marginTop: '4px' }}>
                     <Target size={12} color="#10b981" />
                     <span style={{ fontSize: '0.65rem', color: '#a7f3d0' }}>Sweep Complete</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── TAB: Persistent Zones ──────────── */}
        {activeTab === 'persistent' && (
          <div className="control-card">
            <div className="card-title"><CircleDot size={16} /> Persistent Debris Zones</div>
            <p style={{ fontSize: '0.7rem', color: '#64748b', marginBottom: '10px' }}>
              Zones where debris consistently accumulates across multiple 24h intervals (7-day analysis).
            </p>
            {!persistentData ? (
              <div style={{ textAlign: 'center', padding: '16px 0' }}>
                <Activity size={20} className="spin" color="#00f2ff" />
                <div style={{ color: '#64748b', fontSize: '0.75rem', marginTop: '8px' }}>Analyzing history...</div>
              </div>
            ) : persistentData.zones?.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '16px 0', color: '#64748b', fontSize: '0.75rem' }}>
                No persistent zones detected. More scan data needed.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '350px', overflowY: 'auto' }}>
                {persistentData.zones?.map((z, i) => (
                  <div key={i}
                    onClick={() => flyTo(z.lat, z.lon, 13)}
                    style={{
                      background: '#0a1016', padding: '10px 12px', borderRadius: '10px', cursor: 'pointer',
                      borderLeft: `4px solid ${z.threat?.level === 'CRITICAL' ? '#dc2626' : z.threat?.level === 'HIGH' ? '#f97316' : '#eab308'}`,
                    }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#e2e8f0' }}>Zone #{i + 1}</span>
                      <span style={{
                        fontSize: '0.55rem', fontWeight: 700, padding: '2px 6px', borderRadius: '4px',
                        color: z.threat?.level === 'CRITICAL' ? '#fca5a5' : z.threat?.level === 'HIGH' ? '#fdba74' : '#fde047',
                        background: z.threat?.level === 'CRITICAL' ? '#dc262620' : z.threat?.level === 'HIGH' ? '#f9731620' : '#eab30820',
                      }}>
                        {z.threat?.level || 'LOW'}
                      </span>
                    </div>
                    <div style={{ fontSize: '0.7rem', color: '#94a3b8', fontFamily: 'monospace' }}>
                      {z.lat.toFixed(3)}°, {z.lon.toFixed(3)}°
                    </div>
                    <div style={{ display: 'flex', gap: '10px', fontSize: '0.65rem', color: '#64748b', marginTop: '4px' }}>
                      <span>{z.total_detections} detections</span>
                      <span>{z.intervals_active} intervals</span>
                      <span>score: {z.persistence_score}</span>
                    </div>
                    {z.threat?.nearest_mpa && z.threat.nearest_mpa_km < 100 && (
                      <div style={{ fontSize: '0.6rem', color: '#22c55e', marginTop: '4px' }}>
                        <Shield size={9} style={{ verticalAlign: 'text-bottom' }} /> Near {z.threat.nearest_mpa} ({z.threat.nearest_mpa_km}km)
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
            <div style={{ display: 'flex', gap: '5px', marginTop: '10px' }}>
              <button className="btn btn-outline" onClick={loadPersistent}
                style={{ flex: 1, justifyContent: 'center', fontSize: '0.65rem', padding: '6px 0' }}>
                <RefreshCw size={11} /> Re-analyze
              </button>
              <button className={`btn ${aiPersistLoading ? '' : 'btn-glow'}`} onClick={handleAIPersist} disabled={aiPersistLoading}
                style={{ flex: 1, justifyContent: 'center', fontSize: '0.65rem', padding: '6px 0', opacity: aiPersistLoading ? 0.7 : 1 }}>
                <Zap size={11} className={aiPersistLoading ? 'spin' : ''} />
                {aiPersistLoading ? 'Analyzing...' : 'AI Risk'}
              </button>
            </div>
            {/* AI persistent zone briefing */}
            {aiPersist && aiPersist.analysis && (
              <div style={{ marginTop: '10px', padding: '10px 12px', borderRadius: '10px', background: 'rgba(249,115,22,0.06)', border: '1px solid rgba(249,115,22,0.25)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <span style={{ fontSize: '0.6rem', fontWeight: 700, color: '#f97316', textTransform: 'uppercase' }}>⬡ Ecosystem Risk</span>
                  <span style={{ fontSize: '0.55rem', padding: '2px 5px', borderRadius: '4px', color: '#fdba74', background: '#f9731615', border: '1px solid #f9731640' }}>{aiPersist.analysis.ecosystem_risk}</span>
                </div>
                <p style={{ fontSize: '0.68rem', color: '#cbd5e1', lineHeight: 1.5, margin: '0 0 5px' }}>{aiPersist.analysis.pattern_analysis}</p>
                <p style={{ fontSize: '0.65rem', color: '#94a3b8', lineHeight: 1.4, margin: '0 0 5px' }}>{aiPersist.analysis.recommended_priority}</p>
                <div style={{ fontSize: '0.6rem', color: '#64748b', borderTop: '1px solid #1e293b', paddingTop: '5px', marginTop: '4px' }}>{aiPersist.analysis.long_term_action}</div>
              </div>
            )}
          </div>
        )}

        {/* Legend */}
        <div className="control-card">
          <div className="card-title"><Shield size={16} /> Legend</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px' }}>
            {[
              { color: '#dc2626', label: 'HIGH — Immediate' },
              { color: '#eab308', label: 'MEDIUM — Deploy' },
              { color: '#22c55e', label: 'LOW — Monitor' },
              { color: 'rgba(0,200,100,0.5)', label: 'Protected Area' },
            ].map(l => (
              <div key={l.label} style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '0.6rem', padding: '2px 0' }}>
                <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: l.color, flexShrink: 0 }} />
                <span style={{ color: '#94a3b8' }}>{l.label}</span>
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
          onClick={(info) => {
            if (deployDropMode && info.coordinate) {
              setDeployDropMode(false);
              setVesselOrigin({ lat: info.coordinate[1], lon: info.coordinate[0] });
              handleComputeRoute(info.coordinate[1], info.coordinate[0]);
            }
          }}
          getCursor={({ isHovering }) => deployDropMode ? 'crosshair' : (isHovering ? 'pointer' : 'grab')}
        >
          <Map mapStyle={MAP_STYLE} />
        </DeckGL>

        {/* HUD */}
        <div className="vis-stats glass">
          <div className="stat-item">
            <div className="stat-value">{data?.summary?.total_clusters || 0}</div>
            <div className="stat-label">Zones</div>
          </div>
          <div className="stat-item">
            <div className="stat-value" style={{ color: '#dc2626' }}>{data?.summary?.high || 0}</div>
            <div className="stat-label">Immediate</div>
          </div>
          <div className="stat-item">
            <div className="stat-value" style={{ color: '#eab308' }}>{data?.summary?.medium || 0}</div>
            <div className="stat-label">Deploy</div>
          </div>
          <div className="stat-item">
            <div className="stat-value" style={{ color: '#00f2ff' }}>{data?.summary?.total_points || 0}</div>
            <div className="stat-label">Detections</div>
          </div>
        </div>

        {/* Selected Zone Detail */}
        {selectedCluster && (
          <div style={{
            position: 'absolute', top: '16px', right: '16px', width: '280px',
            background: 'rgba(10,16,22,0.95)', border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: '14px', padding: '16px', zIndex: 1000,
            boxShadow: '0 16px 48px rgba(0,0,0,0.6)',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <h3 style={{ color: '#fff', fontFamily: 'Outfit', margin: 0, fontSize: '0.95rem' }}>Zone #{selectedCluster.id + 1}</h3>
              <span style={{
                fontSize: '0.6rem', fontWeight: 700, padding: '3px 8px', borderRadius: '6px',
                color: PRIORITY_COLORS[selectedCluster.priority]?.text,
                background: PRIORITY_COLORS[selectedCluster.priority]?.bg,
                border: `1px solid ${PRIORITY_COLORS[selectedCluster.priority]?.border}`,
              }}>{selectedCluster.priority}</span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', marginBottom: '12px' }}>
              {[
                { label: 'Density', value: `${selectedCluster.density} pts` },
                { label: 'Frequency', value: `×${selectedCluster.frequency}` },
                { label: 'Radius', value: `${selectedCluster.radius_m}m` },
                { label: 'Recency', value: `${selectedCluster.recency_hours}h ago` },
                { label: 'Confidence', value: `${(selectedCluster.avg_confidence * 100).toFixed(0)}%` },
                { label: 'Score', value: selectedCluster.priority_score.toFixed(2) },
              ].map((item, i) => (
                <div key={i} style={{ background: '#0a1016', padding: '5px 7px', borderRadius: '6px' }}>
                  <div style={{ fontSize: '0.5rem', color: '#64748b', textTransform: 'uppercase' }}>{item.label}</div>
                  <div style={{ fontSize: '0.8rem', color: '#e2e8f0', fontWeight: 600 }}>{item.value}</div>
                </div>
              ))}
            </div>

            {selectedCluster.persistence && (
              <div style={{ padding: '5px 8px', borderRadius: '6px', background: '#f9731615', border: '1px solid #f9731640', marginBottom: '8px', fontSize: '0.65rem', color: '#fdba74' }}>
                <AlertTriangle size={10} style={{ verticalAlign: 'text-bottom', marginRight: '4px' }} />
                Persistent — appears across multiple time intervals
              </div>
            )}

            <div style={{
              padding: '6px 8px', borderRadius: '6px', fontSize: '0.7rem', fontWeight: 600,
              background: PRIORITY_COLORS[selectedCluster.priority]?.bg,
              color: PRIORITY_COLORS[selectedCluster.priority]?.text,
            }}>
              <ChevronRight size={10} style={{ verticalAlign: 'text-bottom' }} /> {selectedCluster.action}
            </div>

            <button className="btn btn-outline" onClick={() => setSelectedCluster(null)}
              style={{ width: '100%', justifyContent: 'center', marginTop: '8px', color: '#94a3b8', fontSize: '0.7rem', padding: '5px 0' }}>
              Dismiss
            </button>
          </div>
        )}

        {/* Empty State */}
        {isEmpty && (
          <div style={{
            position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
            background: 'rgba(10,16,22,0.97)', padding: '28px 36px', borderRadius: '16px',
            border: '1px solid rgba(0,242,255,0.15)', zIndex: 1000, textAlign: 'center', maxWidth: '400px',
            boxShadow: '0 24px 64px rgba(0,0,0,0.7)',
          }}>
            <Trash2 size={36} color="#334155" style={{ marginBottom: '14px' }} />
            <div style={{ color: '#e2e8f0', fontSize: '1rem', fontWeight: 600, marginBottom: '6px' }}>No Detection History</div>
            <div style={{ color: '#64748b', fontSize: '0.8rem', lineHeight: 1.6, marginBottom: '16px' }}>
              Run a scan in the <strong style={{ color: '#00f2ff' }}>Analytics Hub</strong> tab,
              or load 7-day Caribbean demo data to instantly activate all features.
            </div>
            <button className="btn btn-glow" onClick={() => handleSeedData(false)}
              disabled={seedLoading}
              style={{ width: '100%', justifyContent: 'center', marginBottom: '8px', opacity: seedLoading ? 0.7 : 1 }}>
              {seedLoading
                ? <><Activity size={14} className="spin" /> Seeding Caribbean Data...</>
                : <><Zap size={14} /> Load 7-Day Demo Data</>}
            </button>
            <button className="btn btn-outline" onClick={() => handleSeedData(true)}
              disabled={seedLoading}
              style={{ width: '100%', justifyContent: 'center', fontSize: '0.7rem', color: '#475569' }}>
              <RefreshCw size={12} /> Reset & Reseed
            </button>
          </div>
        )}

        {loading && (
          <div style={{
            position: 'absolute', top: '16px', left: '50%', transform: 'translateX(-50%)',
            background: 'rgba(10,16,22,0.92)', padding: '7px 18px', borderRadius: '8px',
            border: '1px solid rgba(0,242,255,0.25)', zIndex: 1000, fontSize: '0.75rem', color: '#00f2ff'
          }}>
            <RefreshCw size={12} className="spin" style={{ verticalAlign: 'text-bottom', marginRight: '6px' }} />
            Processing...
          </div>
        )}
      </main>
    </div>
  );
};

export default Cleanup;

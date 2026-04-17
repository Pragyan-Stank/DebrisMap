import React, { useState, useEffect } from 'react';
import DeckGL from '@deck.gl/react';
import { FlyToInterpolator } from '@deck.gl/core';
import { ScatterplotLayer, PolygonLayer } from '@deck.gl/layers';
import { HeatmapLayer } from '@deck.gl/aggregation-layers';
import { Map } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { getVisualizationData, predictDebris, fetchPatchInference } from '../services/api';
import { Upload, Target, Info, Activity, Layers, Map as MapIcon, ChevronRight } from 'lucide-react';

const INITIAL_VIEW_STATE = {
  longitude: -86.33591,
  latitude: 15.92308,
  zoom: 11,
  pitch: 30,
  bearing: 0
};

const MAP_STYLE = {
  version: 8,
  sources: {
    satellite: {
      type: "raster",
      tiles: [
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
      ],
      tileSize: 256,
      attribution: "Tiles &copy; Esri"
    }
  },
  layers: [
    { id: "background", type: "background", paint: { "background-color": "#020810" } },
    { id: "satellite-layer", type: "raster", source: "satellite", minzoom: 0, maxzoom: 19 }
  ]
};

const Visualization = () => {
  const [data, setData] = useState([]);
  const [clusters, setClusters] = useState([]);
  const [loading, setLoading] = useState(false);
  const [patchLoading, setPatchLoading] = useState(false);
  const [isDrawing, setIsDrawing] = useState(false);
  const [draftBBox, setDraftBBox] = useState(null);
  const [dateRange, setDateRange] = useState("last_3_days");
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);

  const handleRunPatchInference = async () => {
    if (!draftBBox) {
        alert("Please draw a target region on the map first!");
        return;
    }
    
    setPatchLoading(true);
    const lonMin = Math.min(draftBBox.start[0], draftBBox.end[0]);
    const lonMax = Math.max(draftBBox.start[0], draftBBox.end[0]);
    const latMin = Math.min(draftBBox.start[1], draftBBox.end[1]);
    const latMax = Math.max(draftBBox.start[1], draftBBox.end[1]);
    
    const bbox = [lonMin, latMin, lonMax, latMax];
    const centerLon = (lonMin + lonMax) / 2;
    const centerLat = (latMin + latMax) / 2;
    
    const result = await fetchPatchInference(bbox, 10, dateRange); 
    setPatchLoading(false);
    
    if (result && result.points) {
       setData(result.points);
       setClusters(result.clusters || []);
       
       if (result.points.length > 0) {
           setViewState(v => ({
             ...v,
             longitude: centerLon,
             latitude: centerLat,
             zoom: 12,
             transitionDuration: 2500,
             transitionInterpolator: new FlyToInterpolator()
           }));
       }
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    const result = await predictDebris(file);
    setLoading(false);

    if (result && result.points) {
      setData(result.points);
      if (result.points.length > 0) {
         setViewState(v => ({
           ...v,
           longitude: result.points[0].lon,
           latitude: result.points[0].lat,
           zoom: 14,
           transitionDuration: 3000,
           transitionInterpolator: new FlyToInterpolator()
         }));
      }
    }
  };

  const layers = [
    new PolygonLayer({
      id: 'background-catcher',
      data: [{polygon: [[-180, 90], [180, 90], [180, -90], [-180, -90]]}],
      getPolygon: d => d.polygon,
      getFillColor: [0,0,0,0],
      pickable: true,
      visible: isDrawing
    }),
    new PolygonLayer({
      id: 'selection-boundary',
      data: draftBBox ? [{
        polygon: [
          [draftBBox.start[0], draftBBox.start[1]],
          [draftBBox.end[0], draftBBox.start[1]],
          [draftBBox.end[0], draftBBox.end[1]],
          [draftBBox.start[0], draftBBox.end[1]]
        ]
      }] : [],
      getPolygon: d => d.polygon,
      getFillColor: [0, 120, 255, 60],
      getLineColor: [0, 242, 255, 255],
      lineWidthMinPixels: 2,
      stroked: true,
      filled: true
    }),
    new HeatmapLayer({
      id: 'heatmap-layer',
      data,
      getPosition: d => [d.lon, d.lat],
      getWeight: d => d.probability,
      radiusPixels: 40,
      intensity: 3,
      threshold: 0.05
    }),
    new ScatterplotLayer({
      id: 'scatterplot-layer',
      data,
      getPosition: d => [d.lon, d.lat],
      getFillColor: [0, 242, 255, 200],
      getRadius: 30,
      radiusMinPixels: 2,
    }),
    new ScatterplotLayer({
      id: 'cluster-layer',
      data: clusters,
      getPosition: d => d.center,
      getFillColor: [255, 0, 100, 150],
      getLineColor: [255, 0, 100, 255],
      lineWidthMinPixels: 2,
      stroked: true,
      getRadius: d => Math.max(100, d.density * 5),
    })
  ];

  return (
    <div className="vis-container">
      <aside className="sidebar glass">
        <div style={{marginTop: '20px'}}>
            <h2 style={{fontFamily: 'Outfit', color: '#fff', fontSize: '1.2rem'}}>Intelligence Hub</h2>
            <p style={{color: '#94a3b8', fontSize: '0.8rem', marginTop: '5px'}}>Live Regional Detection & Spectral Analytics</p>
        </div>

        <div className="control-card">
            <div className="card-title"><Target size={18}/> Regional Scan</div>
            <p style={{fontSize: '0.75rem', color: '#94a3b8', marginBottom: '15px'}}>Click below then drag on map to select a custom Sentinel-2 swath.</p>
            <button 
                className={`btn ${isDrawing ? 'btn-danger' : 'btn-glow'}`} 
                onClick={() => setIsDrawing(!isDrawing)}
                style={{width: '100%', justifyContent: 'center', backgroundColor: isDrawing ? '#ef4444' : ''}}
            >
                {isDrawing ? 'Cancel Tracking' : 'Define Target Area'}
            </button>
            
            {draftBBox && (
                <div style={{marginTop: '15px'}}>
                    <div className="input-group">
                        <span className="input-label">Temporal Window</span>
                        <select value={dateRange} onChange={e => setDateRange(e.target.value)}>
                            <option value="last_1_day">Current Orbit</option>
                            <option value="last_3_days">Last 72 Hours</option>
                            <option value="last_5_days">Last 5 Orbits</option>
                        </select>
                    </div>
                    <button className="btn btn-glow" onClick={handleRunPatchInference} disabled={patchLoading} style={{width: '100%', justifyContent: 'center'}}>
                        {patchLoading ? 'Analyzing Spectral Data...' : 'Execute Neural Scan'}
                    </button>
                </div>
            )}
        </div>

        <div className="control-card">
            <div className="card-title"><Upload size={18}/> Direct Ingestion</div>
            <p style={{fontSize: '0.75rem', color: '#94a3b8', marginBottom: '15px'}}>Upload localized .TIF patches for high-res deep learning validation.</p>
            <div style={{position: 'relative', overflow: 'hidden', display: 'inline-block', width: '100%'}}>
                <button className="btn glass" style={{width: '100%', justifyContent: 'center', borderColor: 'rgba(255,255,255,0.2)', pointerEvents: 'none'}}>
                    {loading ? 'Processing Raster...' : 'Browse Local Files'}
                </button>
                <input 
                    type="file" 
                    accept=".tif,.tiff" 
                    onChange={handleFileUpload} 
                    disabled={loading}
                    style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        opacity: 0,
                        cursor: 'pointer',
                        zIndex: 10
                    }}
                />
            </div>
        </div>

        <div className="control-card" style={{marginTop: 'auto'}}>
            <div className="card-title"><Activity size={18}/> System Status</div>
            <div style={{display: 'flex', flexDirection: 'column', gap: '8px'}}>
                <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem'}}>
                    <span style={{color: '#94a3b8'}}>U-Net Model</span>
                    <span style={{color: '#10b981'}}>Active</span>
                </div>
                <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem'}}>
                    <span style={{color: '#94a3b8'}}>FDI Compute</span>
                    <span style={{color: '#10b981'}}>Ready</span>
                </div>
                <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem'}}>
                    <span style={{color: '#94a3b8'}}>Lat/Lon Transformation</span>
                    <span style={{color: '#10b981'}}>Locked</span>
                </div>
            </div>
        </div>
      </aside>

      <main className="map-viewport">
        <DeckGL
          initialViewState={viewState}
          controller={{dragPan: !isDrawing}}
          layers={layers}
          onViewStateChange={({viewState}) => setViewState(viewState)}
          onDragStart={(info) => isDrawing && info.coordinate && setDraftBBox({ start: info.coordinate, end: info.coordinate })}
          onDrag={(info) => isDrawing && draftBBox && info.coordinate && setDraftBBox(prev => ({ ...prev, end: info.coordinate }))}
          onDragEnd={(info) => {
             if (isDrawing && draftBBox && info.coordinate) {
                setDraftBBox(prev => ({ ...prev, end: info.coordinate }));
                setIsDrawing(false); 
             }
          }}
          getCursor={({isHovering, isDragging}) => {
             if (isDrawing) return 'crosshair';
             if (isDragging) return 'grabbing';
             if (isHovering) return 'pointer';
             return 'grab';
          }}
        >
          <Map mapStyle={MAP_STYLE} />
        </DeckGL>

        <div className="vis-stats glass">
            <div className="stat-item">
                <div className="stat-value">{data.length}</div>
                <div className="stat-label">Detected Hotspots</div>
            </div>
            <div className="stat-item">
                <div className="stat-value">{clusters.length}</div>
                <div className="stat-label">Active Clusters</div>
            </div>
        </div>

        {isDrawing && (
          <div className="draw-instruction">
            DRAG TO DEFINE SEARCH BBOX
          </div>
        )}
      </main>
    </div>
  );
};

export default Visualization;

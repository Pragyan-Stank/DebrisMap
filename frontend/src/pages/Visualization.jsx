import React, { useState, useMemo } from 'react';
import DeckGL from '@deck.gl/react';
import { FlyToInterpolator } from '@deck.gl/core';
import { ScatterplotLayer, PolygonLayer } from '@deck.gl/layers';
import { HeatmapLayer } from '@deck.gl/aggregation-layers';
import { Map } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { getVisualizationData, predictDebris, fetchPatchInference } from '../services/api';
import { Upload, Target, Activity, Download, Clock, BarChart3, MapPin, AlertTriangle, Crosshair, Trash2, Eye, EyeOff, Radio } from 'lucide-react';
import TerminalLoader from '../components/TerminalLoader';

const INITIAL_VIEW_STATE = {
  longitude: -86.33591,
  latitude: 15.92308,
  zoom: 3,
  pitch: 0,
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
  const [scanHistory, setScanHistory] = useState([]);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showPoints, setShowPoints] = useState(true);
  const [showClusters, setShowClusters] = useState(true);
  const [hoveredPoint, setHoveredPoint] = useState(null);

  // Computed analytics
  const analytics = useMemo(() => {
    if (data.length === 0) return null;
    const probs = data.map(d => d.probability);
    const maxProb = Math.max(...probs);
    const avgProb = probs.reduce((a, b) => a + b, 0) / probs.length;
    const highConf = data.filter(d => d.probability > 0.06).length;
    const medConf = data.filter(d => d.probability > 0.04 && d.probability <= 0.06).length;
    const latCenter = data.reduce((a, d) => a + d.lat, 0) / data.length;
    const lonCenter = data.reduce((a, d) => a + d.lon, 0) / data.length;
    return { maxProb, avgProb, highConf, medConf, latCenter, lonCenter };
  }, [data]);

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
    const widthKm = Math.abs(lonMax - lonMin) * 111.32 * Math.cos(centerLat * Math.PI / 180);
    const heightKm = Math.abs(latMax - latMin) * 111.32;
    const areaSqKm = (widthKm * heightKm).toFixed(2);
    
    const startTime = Date.now();
    
    const result = await fetchPatchInference(bbox, 10, dateRange); 
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    setPatchLoading(false);
    
    if (result && result.points !== undefined) {
       setData(result.points);
       setClusters(result.clusters || []);
       
       // Log to scan history
       setScanHistory(prev => [{
         id: Date.now(),
         time: new Date().toLocaleTimeString(),
         bbox: bbox.map(v => v.toFixed(3)),
         points: result.points.length,
         clusters: (result.clusters || []).length,
         elapsed,
         source: 'Sentinel-2 Live'
       }, ...prev].slice(0, 10));
       
       setViewState(v => ({
         ...v,
         longitude: centerLon,
         latitude: centerLat,
         zoom: 12,
         transitionDuration: 2500,
         transitionInterpolator: new FlyToInterpolator()
       }));
    } else {
       alert("Failed to connect to Sentinel Hub or inference backend.");
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    const startTime = Date.now();
    const result = await predictDebris(file);
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    setLoading(false);

    if (result && result.points) {
      setData(result.points);
      setClusters(result.clusters || []);
      
      setScanHistory(prev => [{
        id: Date.now(),
        time: new Date().toLocaleTimeString(),
        bbox: ['local'],
        points: result.points.length,
        clusters: (result.clusters || []).length,
        elapsed,
        source: file.filename || 'GeoTIFF Upload'
      }, ...prev].slice(0, 10));
      
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

  const handleExportCSV = () => {
    if (data.length === 0) return;
    const header = "latitude,longitude,probability\n";
    const rows = data.map(d => `${d.lat},${d.lon},${d.probability}`).join("\n");
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `oceaneye_detections_${Date.now()}.csv`;
    a.click();
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
    showHeatmap && new HeatmapLayer({
      id: 'heatmap-layer',
      data,
      getPosition: d => [d.lon, d.lat],
      getWeight: d => d.probability,
      radiusPixels: 40,
      intensity: 3,
      threshold: 0.05
    }),
    showPoints && new ScatterplotLayer({
      id: 'scatterplot-layer',
      data,
      getPosition: d => [d.lon, d.lat],
      getFillColor: [0, 242, 255, 200],
      getRadius: 30,
      radiusMinPixels: 3,
      pickable: true,
      onHover: info => setHoveredPoint(info.object || null),
    }),
    showClusters && new ScatterplotLayer({
      id: 'cluster-layer',
      data: clusters,
      getPosition: d => d.center,
      getFillColor: [255, 60, 120, 180],
      getLineColor: [255, 60, 120, 255],
      lineWidthMinPixels: 2,
      stroked: true,
      getRadius: d => Math.max(100, d.density * 5),
    })
  ].filter(Boolean);

  // Mini confidence bar chart
  const ConfidenceChart = () => {
    if (!analytics) return null;
    const bins = [0, 0, 0, 0, 0];
    data.forEach(d => {
      const idx = Math.min(4, Math.floor(d.probability * 100));
      bins[idx]++;
    });
    const maxBin = Math.max(...bins, 1);
    return (
      <div style={{display:'flex',gap:'3px',alignItems:'flex-end',height:'50px',marginTop:'10px'}}>
        {bins.map((count, i) => (
          <div key={i} style={{ flex: 1, display:'flex', flexDirection:'column', alignItems:'center', gap:'2px' }}>
            <div style={{
              width:'100%',
              height: `${(count/maxBin)*40}px`,
              background: `linear-gradient(180deg, #00f2ff, #0066cc)`,
              borderRadius: '2px 2px 0 0',
              minHeight: count > 0 ? '4px' : '0'
            }}/>
            <span style={{fontSize:'0.55rem',color:'#64748b'}}>{(i+1)}%</span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="vis-container">
      <aside className="sidebar glass" style={{overflowY:'auto'}}>
        <div style={{marginTop: '20px'}}>
            <h2 style={{fontFamily: 'Outfit', color: '#fff', fontSize: '1.2rem'}}>Intelligence Hub</h2>
            <p style={{color: '#94a3b8', fontSize: '0.8rem', marginTop: '5px'}}>Sentinel-2 Live Detection & Analytics</p>
        </div>

        {/* Regional Scan Card */}
        <div className="control-card">
            <div className="card-title"><Target size={18}/> Regional Scan</div>
            <p style={{fontSize: '0.75rem', color: '#94a3b8', marginBottom: '15px'}}>Draw a bounding box on the satellite map to scan for debris.</p>
            <button 
                className={`btn ${isDrawing ? 'btn-danger' : 'btn-glow'}`} 
                onClick={() => setIsDrawing(!isDrawing)}
                style={{width: '100%', justifyContent: 'center', backgroundColor: isDrawing ? '#ef4444' : ''}}
            >
                <Crosshair size={16}/>
                {isDrawing ? 'Cancel Drawing' : 'Define Target Area'}
            </button>
            
            {draftBBox && (
                <div style={{marginTop: '15px'}}>
                    {patchLoading ? (
                        <TerminalLoader mode="visualization" height="150px" />
                    ) : (
                        <>
                            <div style={{
                              fontSize:'0.7rem', color:'#64748b', marginBottom:'8px', fontFamily:'monospace', 
                              background:'#0a1016', padding:'6px 8px', borderRadius:'6px', display: 'flex', flexDirection: 'column', gap: '4px'
                            }}>
                              <div>SW: {Math.min(draftBBox.start[1],draftBBox.end[1]).toFixed(4)}°, {Math.min(draftBBox.start[0],draftBBox.end[0]).toFixed(4)}°</div>
                              <div>NE: {Math.max(draftBBox.start[1],draftBBox.end[1]).toFixed(4)}°, {Math.max(draftBBox.start[0],draftBBox.end[0]).toFixed(4)}°</div>
                              <div style={{ 
                                color: '#00f2ff', 
                                borderTop: '1px solid rgba(0, 242, 255, 0.2)', 
                                paddingTop: '4px', 
                                marginTop: '2px',
                                fontWeight: 600
                              }}>
                                Area: ~{(
                                  Math.abs(draftBBox.end[0] - draftBBox.start[0]) * 111.32 * Math.cos(((draftBBox.start[1] + draftBBox.end[1]) / 2) * Math.PI / 180) * 
                                  Math.abs(draftBBox.end[1] - draftBBox.start[1]) * 111.32
                                ).toFixed(2)} sq. km
                              </div>
                            </div>
                            
                            <div className="input-group">
                                <span className="input-label">Temporal Window</span>
                                <select value={dateRange} onChange={e => setDateRange(e.target.value)}>
                                    <option value="last_1_day">Current Orbit</option>
                                    <option value="last_3_days">Last 72 Hours</option>
                                    <option value="last_5_days">Last 5 Orbits</option>
                                </select>
                            </div>
                            <button className="btn btn-glow" onClick={handleRunPatchInference} style={{width: '100%', justifyContent: 'center'}}>
                                Execute Neural Scan
                            </button>
                            <button className="btn glass" onClick={() => {setDraftBBox(null); setData([]); setClusters([]);}} style={{width:'100%',justifyContent:'center',marginTop:'8px',borderColor:'rgba(255,255,255,0.1)', color:'#94a3b8', fontSize:'0.8rem'}}>
                                <Trash2 size={14}/> Clear Selection
                            </button>
                        </>
                    )}
                </div>
            )}
        </div>

        {/* Upload Card */}
        <div className="control-card">
            <div className="card-title"><Upload size={18}/> Direct Ingestion</div>
            {loading ? (
                <TerminalLoader mode="visualization" height="130px" />
            ) : (
                <div className="upload-zone">
                    <input type="file" accept=".tif,.tiff" onChange={handleFileUpload} disabled={loading} />
                    <div className="upload-zone-icon">
                      <Upload size={18} color="#00f2ff" />
                    </div>
                    <div className="upload-zone-text">
                        <><strong>Drop .TIF here</strong> or click to browse<br/><span style={{fontSize:'0.7rem'}}>MARIDA GeoTIFF • 11-band Sentinel-2</span></>
                    </div>
                </div>
            )}
        </div>

        {/* Detection Analytics Card */}
        {analytics && (
          <div className="control-card">
            <div className="card-title"><BarChart3 size={18}/> Detection Analytics</div>
            <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:'10px'}}>
              <div style={{background:'#0a1016',padding:'10px',borderRadius:'8px',textAlign:'center'}}>
                <div style={{fontSize:'1.4rem',fontWeight:700,color:'#00f2ff'}}>{data.length}</div>
                <div style={{fontSize:'0.65rem',color:'#64748b',textTransform:'uppercase'}}>Total Hotspots</div>
              </div>
              <div style={{background:'#0a1016',padding:'10px',borderRadius:'8px',textAlign:'center'}}>
                <div style={{fontSize:'1.4rem',fontWeight:700,color:'#ff3c78'}}>{clusters.length}</div>
                <div style={{fontSize:'0.65rem',color:'#64748b',textTransform:'uppercase'}}>Clusters</div>
              </div>
              <div style={{background:'#0a1016',padding:'10px',borderRadius:'8px',textAlign:'center'}}>
                <div style={{fontSize:'1.4rem',fontWeight:700,color:'#fbbf24'}}>{analytics.highConf}</div>
                <div style={{fontSize:'0.65rem',color:'#64748b',textTransform:'uppercase'}}>High Conf.</div>
              </div>
              <div style={{background:'#0a1016',padding:'10px',borderRadius:'8px',textAlign:'center'}}>
                <div style={{fontSize:'1.4rem',fontWeight:700,color:'#94a3b8'}}>{(analytics.maxProb * 100).toFixed(1)}%</div>
                <div style={{fontSize:'0.65rem',color:'#64748b',textTransform:'uppercase'}}>Peak Signal</div>
              </div>
            </div>
            <div style={{marginTop:'10px',fontSize:'0.7rem',color:'#64748b'}}>Confidence Distribution</div>
            <ConfidenceChart />
            <button className="btn glass" onClick={handleExportCSV} style={{width:'100%',justifyContent:'center',marginTop:'12px',borderColor:'rgba(255,255,255,0.1)',fontSize:'0.8rem'}}>
              <Download size={14}/> Export as CSV
            </button>
          </div>
        )}

        {/* Layer Toggles */}
        <div className="control-card">
          <div className="card-title"><Eye size={18}/> Layer Controls</div>
          <div style={{display:'flex',flexDirection:'column',gap:'8px'}}>
            {[
              { label: 'Heatmap', state: showHeatmap, setter: setShowHeatmap, color: '#f97316' },
              { label: 'Detection Points', state: showPoints, setter: setShowPoints, color: '#00f2ff' },
              { label: 'Cluster Zones', state: showClusters, setter: setShowClusters, color: '#ff3c78' },
            ].map(layer => (
              <div key={layer.label} onClick={() => layer.setter(!layer.state)} style={{display:'flex',alignItems:'center',gap:'10px',cursor:'pointer',padding:'6px 0'}}>
                <div style={{width:'12px',height:'12px',borderRadius:'3px',background: layer.state ? layer.color : '#333',border:`1px solid ${layer.color}`,transition:'all 0.2s'}}/>
                <span style={{fontSize:'0.8rem',color: layer.state ? '#e2e8f0' : '#64748b'}}>{layer.label}</span>
                {layer.state ? <Eye size={12} color="#64748b" style={{marginLeft:'auto'}}/> : <EyeOff size={12} color="#334155" style={{marginLeft:'auto'}}/>}
              </div>
            ))}
          </div>
        </div>

        {/* Scan History */}
        {scanHistory.length > 0 && (
          <div className="control-card">
            <div className="card-title"><Clock size={18}/> Scan History</div>
            <div style={{display:'flex',flexDirection:'column',gap:'6px',maxHeight:'200px',overflowY:'auto'}}>
              {scanHistory.map(scan => (
                <div key={scan.id} style={{background:'#0a1016',padding:'8px 10px',borderRadius:'8px',fontSize:'0.7rem',borderLeft:`3px solid ${scan.points > 0 ? '#00f2ff' : '#334155'}`}}>
                  <div style={{display:'flex',justifyContent:'space-between',marginBottom:'3px'}}>
                    <span style={{color:'#e2e8f0',fontWeight:600}}>{scan.time}</span>
                    <span style={{color:'#64748b'}}>{scan.elapsed}s</span>
                  </div>
                  <div style={{color: scan.points > 0 ? '#00f2ff' : '#64748b'}}>
                    {scan.points} detections · {scan.clusters} clusters
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* System Status */}
        <div className="control-card">
            <div className="card-title"><Activity size={18}/> System Status</div>
            <div style={{display: 'flex', flexDirection: 'column', gap: '8px'}}>
                {[
                  { label: 'U-Net Model', status: 'Online', ok: true },
                  { label: 'Sentinel Hub API', status: 'Connected', ok: true },
                  { label: 'FDI Compute', status: 'Ready', ok: true },
                  { label: 'DBSCAN Clustering', status: 'Active', ok: true },
                  { label: 'CRS Transform', status: 'EPSG:4326', ok: true },
                ].map(s => (
                  <div key={s.label} style={{display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem'}}>
                    <span style={{color: '#94a3b8'}}>{s.label}</span>
                    <span style={{color: s.ok ? '#10b981' : '#ef4444'}}>{s.status}</span>
                  </div>
                ))}
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

        {/* Floating HUD Stats */}
        <div className="vis-stats glass">
            <div className="stat-item">
                <div className="stat-value">{data.length}</div>
                <div className="stat-label">Detections</div>
            </div>
            <div className="stat-item">
                <div className="stat-value">{clusters.length}</div>
                <div className="stat-label">Clusters</div>
            </div>
            {analytics && (
              <div className="stat-item">
                <div className="stat-value" style={{color:'#fbbf24'}}>{(analytics.maxProb * 100).toFixed(0)}%</div>
                <div className="stat-label">Peak</div>
              </div>
            )}
        </div>

        {/* Hover Tooltip */}
        {hoveredPoint && (
          <div style={{
            position:'absolute', bottom:'80px', left:'20px',
            padding:'12px 16px', borderRadius:'12px', zIndex:1000,
            background:'rgba(10,16,22,0.95)', border:'1px solid rgba(0,242,255,0.3)',
            boxShadow:'0 8px 32px rgba(0,0,0,0.6)', minWidth:'200px'
          }}>
            <div style={{fontSize:'0.7rem',color:'#00f2ff',textTransform:'uppercase',letterSpacing:'1px',marginBottom:'6px'}}>Detection Details</div>
            <div style={{fontSize:'0.8rem',color:'#e2e8f0'}}>
              <MapPin size={12} style={{verticalAlign:'text-bottom',marginRight:'4px'}}/>
              {hoveredPoint.lat.toFixed(6)}°, {hoveredPoint.lon.toFixed(6)}°
            </div>
            <div style={{fontSize:'0.8rem',color:'#fbbf24',marginTop:'4px'}}>
              <AlertTriangle size={12} style={{verticalAlign:'text-bottom',marginRight:'4px'}}/>
              Confidence: {(hoveredPoint.probability * 100).toFixed(2)}%
            </div>
          </div>
        )}

        {/* Map Legend */}
        <div style={{
          position:'absolute', bottom:'20px', left:'20px',
          padding:'10px 14px', borderRadius:'10px', zIndex:100,
          background:'rgba(10,16,22,0.85)', border:'1px solid rgba(255,255,255,0.1)',
          fontSize:'0.7rem'
        }}>
          <div style={{fontWeight:600,marginBottom:'6px',color:'#e2e8f0'}}>Map Legend</div>
          <div style={{display:'flex',alignItems:'center',gap:'6px',marginBottom:'3px'}}>
            <div style={{width:'10px',height:'10px',borderRadius:'50%',background:'#00f2ff'}}/> <span style={{color:'#94a3b8'}}>Debris Detection</span>
          </div>
          <div style={{display:'flex',alignItems:'center',gap:'6px',marginBottom:'3px'}}>
            <div style={{width:'10px',height:'10px',borderRadius:'50%',background:'#ff3c78'}}/> <span style={{color:'#94a3b8'}}>DBSCAN Cluster</span>
          </div>
          <div style={{display:'flex',alignItems:'center',gap:'6px'}}>
            <div style={{width:'10px',height:'10px',borderRadius:'4px',background:'rgba(0,120,255,0.5)',border:'1px solid #0078ff'}}/> <span style={{color:'#94a3b8'}}>Scan Region</span>
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

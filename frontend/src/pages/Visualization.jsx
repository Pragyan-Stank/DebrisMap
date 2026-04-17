import React, { useState, useEffect } from 'react';
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer } from '@deck.gl/layers';
import { HeatmapLayer } from '@deck.gl/aggregation-layers';
import { Map } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { getVisualizationData, predictDebris } from '../services/api';
import { Upload } from 'lucide-react';

const INITIAL_VIEW_STATE = {
  longitude: -122.4194,
  latitude: 37.7749,
  zoom: 11,
  pitch: 40,
  bearing: 0
};

// Using a free CARTO basemap
const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

const Visualization = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);

  const loadData = async () => {
    const res = await getVisualizationData();
    if (res && res.points) {
      setData(res.points);
      
      // Auto center map on first point if available
      if (res.points.length > 0) {
        setViewState(v => ({
          ...v,
          longitude: res.points[0].lon,
          latitude: res.points[0].lat
        }));
      }
    }
  };

  useEffect(() => {
    loadData();
    
    // Auto-refresh every 5 seconds to simulate real-time
    const interval = setInterval(() => {
      loadData();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

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
           latitude: result.points[0].lat
         }));
      }
    }
  };

  const layers = [
    new HeatmapLayer({
      id: 'heatmap-layer',
      data,
      getPosition: d => [d.lon, d.lat],
      getWeight: d => d.probability,
      radiusPixels: 40,
      intensity: 1,
      threshold: 0.1
    }),
    new ScatterplotLayer({
      id: 'scatterplot-layer',
      data,
      getPosition: d => [d.lon, d.lat],
      getFillColor: d => [255, 200, 0, 200], // Yellow for points
      getRadius: d => 50,
      radiusMinPixels: 2,
      radiusMaxPixels: 10,
    })
  ];

  return (
    <div className="vis-container">
      <div className="vis-header">
        <div>
          <h2>Marine Plastic Hotspots</h2>
          <p>Real-time detection overlay (Heatmap + Points)</p>
        </div>
        
        <div className="vis-controls">
          <div className="upload-button-wrapper">
            <button className="btn-primary" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Upload size={18} />
              {loading ? 'Processing...' : 'Upload Image'}
            </button>
            <input 
              type="file" 
              accept=".tif,.tiff,.jpg,.jpeg,.png"
              onChange={handleFileUpload} 
              disabled={loading}
            />
          </div>
        </div>
      </div>

      <div className="map-container">
        <DeckGL
          initialViewState={viewState}
          controller={true}
          layers={layers}
          onViewStateChange={({viewState}) => setViewState(viewState)}
        >
          <Map mapStyle={MAP_STYLE} />
        </DeckGL>
      </div>
    </div>
  );
};

export default Visualization;

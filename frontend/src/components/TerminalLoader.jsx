import React, { useState, useEffect } from 'react';
import { Activity, Terminal } from 'lucide-react';

const LOG_PHASES = {
  visualization: [
    "Initializing GeoTIFF ingestion protocol...",
    "Validating image metadata and projection (EPSG:4326)...",
    "Extracting hyper-spectral bands (B02, B03, B04, B08)...",
    "Preprocessing patches for DL pipeline...",
    "Running MARIDA UNet classification model...",
    "Applying Non-Maximum Suppression to detection masks...",
    "Generating spatial density heatmaps...",
    "Finalizing aggregation, drawing inference..."
  ],
  trajectory: [
    "Connecting to meteorological data streams (Open-Meteo)...",
    "Fetching 10m wind vector components (U, V)...",
    "Acquiring global ocean current baseline data...",
    "Seeding Monte-Carlo simulation particles (N=50)...",
    "Simulating physics drift (leeway + Ekman transport)...",
    "Calculating 5th and 95th percentile uncertainty bounds...",
    "Projecting trajectory path out to T+72 hours...",
    "Rendering trajectory arrays for deck.gl payload..."
  ],
  cleanup: [
    "Accessing historical detection intelligence database...",
    "Aggregating spatiotemporal points over window...",
    "Running DBSCAN geographic clustering algorithm...",
    "Cross-referencing Marine Protected Areas (MPAs)...",
    "Assessing asset and environmental threat index...",
    "Formulating vessel dispatch fleet optimization...",
    "Calculating true dynamic interception vectors...",
    "Greedy TSP sweep compilation complete."
  ]
};

const TerminalLoader = ({ 
  mode = "visualization", 
  height = "200px" 
}) => {
  const [logs, setLogs] = useState([]);
  const phases = LOG_PHASES[mode] || LOG_PHASES.visualization;

  useEffect(() => {
    let currentIndex = 0;
    setLogs([`> [INIT] Started ${mode.toUpperCase()} sub-routine`]);

    const interval = setInterval(() => {
      if (currentIndex < phases.length) {
        setLogs(prev => [...prev, `> [${(new Date()).toISOString().split('T')[1].slice(0,-1)}] ${phases[currentIndex]}`]);
        currentIndex++;
      } else {
        setLogs(prev => {
          // keep the length bounded, optionally add some random hex strings
          const randomHex = Math.random().toString(16).substr(2, 8).toUpperCase();
          return [...prev.slice(-6), `> [PROCESS] Analysing buffer payload 0x${randomHex}...`];
        });
      }
    }, 1200); // add a line every 1.2s

    return () => clearInterval(interval);
  }, [mode, phases]);

  return (
    <div style={{
      width: '100%',
      height: height,
      backgroundColor: '#050a0f',
      border: '1px solid #1e293b',
      borderRadius: '8px',
      padding: '12px',
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column',
      position: 'relative'
    }}>
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        gap: '6px', 
        borderBottom: '1px solid #1e293b',
        paddingBottom: '8px',
        marginBottom: '8px'
      }}>
        <Terminal size={14} color="#64748b" />
        <span style={{ fontSize: '0.65rem', textTransform: 'uppercase', color: '#64748b', letterSpacing: '1px', fontWeight: 600 }}>Active Processing Terminal</span>
        <Activity size={14} color="#00f2ff" className="spin" style={{ marginLeft: 'auto' }} />
      </div>
      
      <div style={{
        fontFamily: "'Courier New', Courier, monospace",
        fontSize: '0.65rem',
        color: '#00f2ff',
        display: 'flex',
        flexDirection: 'column',
        gap: '4px',
        lineHeight: 1.4,
        overflowY: 'auto'
      }}>
        {logs.map((log, i) => (
          <div key={i} style={{ 
            opacity: 1 - ((logs.length - 1 - i) * 0.15),
            textShadow: '0 0 5px rgba(0, 242, 255, 0.4)'
          }}>
            {log}
          </div>
        ))}
      </div>
    </div>
  );
};

export default TerminalLoader;

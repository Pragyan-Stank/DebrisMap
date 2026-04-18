import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ChevronRight, Shield, Globe, Cpu, Anchor, Navigation, Trash2, BarChart2, Radio, Satellite, Eye, AlertTriangle, Ship, Activity } from 'lucide-react';

const COUNTER_TARGETS = [
  { value: 14, suffix: 'M+', label: 'Tons Plastic/Year in Oceans' },
  { value: 2560, suffix: '', label: 'km² Scanned Weekly' },
  { value: 72, suffix: 'h', label: 'Forecast Window' },
  { value: 98.6, suffix: '%', label: 'WCR Water Accuracy' },
];

const AnimatedCounter = ({ target, suffix, duration = 2000 }) => {
  const [count, setCount] = useState(0);
  const [started, setStarted] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) setStarted(true);
    }, { threshold: 0.3 });

    const el = document.getElementById('stats-section');
    if (el) observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!started) return;
    const steps = 60;
    const increment = target / steps;
    let current = 0;
    const timer = setInterval(() => {
      current += increment;
      if (current >= target) {
        setCount(target);
        clearInterval(timer);
      } else {
        setCount(Math.floor(current * 10) / 10);
      }
    }, duration / steps);
    return () => clearInterval(timer);
  }, [started, target, duration]);

  const display = Number.isInteger(target) ? Math.floor(count) : count.toFixed(1);
  return <span>{display}{suffix}</span>;
};

const Home = () => {
  const particles = Array.from({ length: 25 }).map((_, i) => ({
    id: i,
    left: `${Math.random() * 100}%`,
    top: `${Math.random() * 100}%`,
    size: `${Math.random() * 6 + 2}px`,
    delay: `${Math.random() * 10}s`,
    duration: `${Math.random() * 20 + 10}s`
  }));

  return (
    <div style={{ overflowX: 'hidden' }}>
      {/* ── Hero ──────────────────────────────────────── */}
      <section className="hero" style={{ height: '100vh' }}>
        <video autoPlay muted loop playsInline className="hero-img-bg"
          style={{ opacity: 0.6, pointerEvents: 'none' }}>
          <source src="/hero_bg.mp4" type="video/mp4" />
        </video>
        <div className="hero-overlay"></div>

        <div className="floating-debris">
          {particles.map(p => (
            <div key={p.id} className="debris-particle"
              style={{ left: p.left, top: p.top, width: p.size, height: p.size, animationDelay: p.delay, animationDuration: p.duration }} />
          ))}
        </div>

        <div className="hero-content">
          <span className="hero-tag animate-fade-in">
            <Satellite size={12} style={{ verticalAlign: 'text-bottom', marginRight: '6px' }} />
            Autonomous Marine Intelligence Platform
          </span>
          <h1 className="hero-title animate-fade-in">
            Cleaning our oceans,<br />one pixel at a time.
          </h1>
          <p className="hero-desc animate-fade-in">
            Real-time Sentinel-2 satellite imagery fused with deep-learning U-Net architectures to detect, track, and forecast marine debris drift across global coastlines.
          </p>
          <div className="hero-btns animate-fade-in" style={{ display: 'flex', gap: '14px', justifyContent: 'center', flexWrap: 'wrap' }}>
            <Link to="/visualization" className="btn btn-glow">
              <BarChart2 size={18} /> Analytics Hub <ChevronRight size={16} />
            </Link>
            <Link to="/trajectory" className="btn btn-outline">
              <Navigation size={18} /> Drift Forecast
            </Link>
            <Link to="/cleanup" className="btn btn-outline">
              <Trash2 size={18} /> Clean-Up Ops
            </Link>
          </div>
        </div>
      </section>

      {/* ── Stats Counter ─────────────────────────────── */}
      <section id="stats-section" className="section-gradient" style={{ padding: '60px 4rem' }}>
        <div className="stats-row">
          {COUNTER_TARGETS.map((s, i) => (
            <div key={i} className="stat-block">
              <div className="stat-num">
                <AnimatedCounter target={s.value} suffix={s.suffix} />
              </div>
              <div className="stat-desc">{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Capabilities Grid ─────────────────────────── */}
      <section className="section-dark">
        <div style={{ textAlign: 'center', marginBottom: '60px' }}>
          <h2 style={{ fontSize: '2.2rem', fontWeight: 800, marginBottom: '14px' }}>
            Operational Capabilities
          </h2>
          <p style={{ color: '#94a3b8', maxWidth: '640px', marginInline: 'auto', fontSize: '0.95rem' }}>
            Built for Coast Guard and maritime environmental agencies — from detection to deployment.
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px', maxWidth: '1100px', marginInline: 'auto' }}>
          {[
            { icon: <Cpu size={22} color="#00f2ff" />, title: 'Neural Inference', desc: 'U-Net architecture with 11-band Sentinel-2 spectral analysis, NDWI water masking, and FDI debris indexing.' },
            { icon: <Satellite size={22} color="#00f2ff" />, title: 'Live Satellite Scan', desc: 'Real-time Copernicus CDSE integration for on-demand regional debris sweeps with auto cloud filtering.' },
            { icon: <Navigation size={22} color="#00f2ff" />, title: '72h Drift Forecast', desc: 'Leeway physics model with live wind forcing from Open-Meteo and Monte Carlo uncertainty estimation.' },
            { icon: <Shield size={22} color="#00f2ff" />, title: 'Risk Classification', desc: 'Priority scoring (CRITICAL → LOW) based on density, frequency, persistence, and coastal proximity.' },
            { icon: <Ship size={22} color="#00f2ff" />, title: 'Cleanup Dispatch', desc: 'Automated zone prioritization with recommended actions: Monitor, Deploy Cleanup, or Immediate Attention.' },
            { icon: <Eye size={22} color="#00f2ff" />, title: 'Weather Intelligence', desc: 'Live WeatherAPI integration showing wind speed, gusts, visibility, pressure, and sea state conditions.' },
          ].map((feat, i) => (
            <div key={i} className="feature-card">
              <div className="feature-icon">{feat.icon}</div>
              <h3 style={{ color: '#fff', fontSize: '1rem', fontWeight: 700, marginBottom: '8px' }}>{feat.title}</h3>
              <p style={{ color: '#94a3b8', fontSize: '0.82rem', lineHeight: 1.6 }}>{feat.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── How It Works ──────────────────────────────── */}
      <section className="section-alt" id="how-it-works">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: '60px', alignItems: 'center', maxWidth: '1100px', marginInline: 'auto' }}>
          <div>
            <h2 style={{ fontSize: '2rem', fontWeight: 800, marginBottom: '10px' }}>Operational Pipeline</h2>
            <p style={{ color: '#64748b', fontSize: '0.85rem', marginBottom: '30px' }}>From raw satellite swath to actionable coast guard intelligence in minutes.</p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              {[
                { n: '1', title: 'Acquire Satellite Pass', desc: 'Sentinel-2 L2A imagery fetched via Copernicus Data Space with automatic cloud masking.' },
                { n: '2', title: 'Deep Neural Segmentation', desc: 'U-Net isolates synthetic debris from natural sargassum and sea foam across 11 spectral bands.' },
                { n: '3', title: 'Geospatial Clustering', desc: 'DBSCAN groups pixel detections into cleanup zones with WGS84 georeferencing.' },
                { n: '4', title: 'Drift & Risk Assessment', desc: '72h trajectory forecast with wind + current forcing. Priority scoring triggers coast guard dispatch.' },
              ].map((step, i) => (
                <div key={i} style={{ display: 'flex', gap: '16px' }}>
                  <div className="step-number">{step.n}</div>
                  <div>
                    <h4 style={{ color: '#fff', fontSize: '0.9rem', marginBottom: '4px', fontWeight: 700 }}>{step.title}</h4>
                    <p style={{ color: '#94a3b8', fontSize: '0.8rem', lineHeight: 1.5 }}>{step.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="glass" style={{ padding: '2rem', borderRadius: '20px', border: '1px solid rgba(0,242,255,0.15)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <Activity size={14} color="#00f2ff" />
                <span style={{ fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1px' }}>Live System Diagnostics</span>
              </div>
              <span style={{ fontSize: '0.65rem', color: '#10b981', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Radio size={10} /> All Systems Nominal
              </span>
            </div>

            <div style={{ display: 'grid', gap: '10px' }}>
              {[
                { label: 'U-Net Model', status: 'Online', val: '62% mIoU', c: '#10b981' },
                { label: 'Sentinel Hub API', status: 'Connected', val: 'CDSE OAuth2', c: '#10b981' },
                { label: 'Open-Meteo Wind', status: 'Live', val: '72h forecast', c: '#10b981' },
                { label: 'WeatherAPI.com', status: 'Active', val: 'Marine data', c: '#10b981' },
                { label: 'DBSCAN Clustering', status: 'Ready', val: 'ε=0.002 rad', c: '#10b981' },
                { label: 'Monte Carlo Engine', status: 'Idle', val: '50 particles', c: '#fbbf24' },
                { label: 'Detection Store', status: 'Persisting', val: 'JSON disk', c: '#10b981' },
              ].map((s, i) => (
                <div key={i} style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  background: '#0a1016', padding: '10px 14px', borderRadius: '10px', fontSize: '0.78rem',
                }}>
                  <span style={{ color: '#94a3b8' }}>{s.label}</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ color: '#64748b', fontSize: '0.7rem' }}>{s.val}</span>
                    <span style={{ color: s.c, fontWeight: 600 }}>{s.status}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── Coast Guard Operations Panel ──────────────── */}
      <section className="section-dark" style={{ paddingTop: '80px' }}>
        <div style={{ textAlign: 'center', marginBottom: '50px' }}>
          <h2 style={{ fontSize: '2rem', fontWeight: 800, marginBottom: '10px' }}>
            <Anchor size={24} style={{ verticalAlign: 'text-bottom', marginRight: '10px' }} />
            Coast Guard Operations
          </h2>
          <p style={{ color: '#64748b', fontSize: '0.9rem' }}>Purpose-built workflow for maritime law enforcement and environmental protection units.</p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '20px', maxWidth: '900px', marginInline: 'auto' }}>
          {[
            { icon: <AlertTriangle size={20} color="#dc2626" />, title: 'Threat Assessment', label: 'Critical', color: '#dc2626',
              desc: 'Automatic risk classification based on debris density, coastal proximity, and protected marine area overlap.' },
            { icon: <Navigation size={20} color="#f97316" />, title: 'Drift Intercept Planning', label: 'Forecast', color: '#f97316',
              desc: 'Predict where debris will be in 24/48/72 hours. Plan vessel intercept routes along the projected path.' },
            { icon: <Ship size={20} color="#eab308" />, title: 'Cleanup Asset Dispatch', label: 'Deploy', color: '#eab308',
              desc: 'Ranked priority zones with recommended actions. High-priority zones trigger immediate asset deployment.' },
            { icon: <Globe size={20} color="#22c55e" />, title: 'Persistent Zone Tracking', label: 'History', color: '#22c55e',
              desc: 'Detection history reveals chronic debris accumulation zones — critical for long-term patrol route planning.' },
          ].map((op, i) => (
            <div key={i} style={{
              display: 'flex', gap: '16px', padding: '20px', borderRadius: '14px',
              background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)',
              borderLeft: `4px solid ${op.color}`,
            }}>
              <div className="feature-icon" style={{ flexShrink: 0 }}>{op.icon}</div>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                  <h4 style={{ color: '#fff', fontSize: '0.9rem', fontWeight: 700 }}>{op.title}</h4>
                  <span style={{ fontSize: '0.6rem', fontWeight: 700, color: op.color, background: `${op.color}15`, 
                    padding: '2px 8px', borderRadius: '4px', textTransform: 'uppercase' }}>{op.label}</span>
                </div>
                <p style={{ color: '#94a3b8', fontSize: '0.78rem', lineHeight: 1.5 }}>{op.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── CTA ───────────────────────────────────────── */}
      <section className="section-alt" style={{ paddingTop: '60px', paddingBottom: '60px' }}>
        <div style={{ textAlign: 'center', maxWidth: '600px', marginInline: 'auto' }}>
          <h2 style={{ fontSize: '1.8rem', fontWeight: 800, marginBottom: '12px' }}>Ready to Deploy?</h2>
          <p style={{ color: '#64748b', fontSize: '0.9rem', marginBottom: '24px' }}>
            Start scanning your coastal sector in seconds. Upload a satellite image or draw a live scan region.
          </p>
          <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
            <Link to="/visualization" className="btn btn-glow">
              <BarChart2 size={16} /> Open Analytics Hub
            </Link>
            <Link to="/cleanup" className="btn btn-outline">
              <Trash2 size={16} /> Clean-Up Programme
            </Link>
          </div>
        </div>
      </section>

      {/* ── Footer ────────────────────────────────────── */}
      <footer className="footer">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ color: '#00f2ff', fontWeight: 700 }}>OceanEye AI</span>
          <span>· 2026</span>
        </div>
        <div style={{ display: 'flex', gap: '16px' }}>
          <a href="#" style={{ color: '#64748b' }}><Globe size={16} /></a>
          <a href="#" style={{ color: '#64748b' }}><Shield size={16} /></a>
        </div>
      </footer>
    </div>
  );
};

export default Home;

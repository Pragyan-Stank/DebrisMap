import React from 'react';
import { Link } from 'react-router-dom';
import { ChevronRight, Shield, Globe, Cpu, Database, Microscope, Zap, Share2, Mail, Activity } from 'lucide-react';

const Home = () => {
  // Generate random debris particles for the dynamic background
  const particles = Array.from({ length: 20 }).map((_, i) => ({
    id: i,
    left: `${Math.random() * 100}%`,
    top: `${Math.random() * 100}%`,
    size: `${Math.random() * 8 + 2}px`,
    delay: `${Math.random() * 10}s`,
    duration: `${Math.random() * 20 + 10}s`
  }));

  return (
    <div className="home-wrapper" style={{ overflowX: 'hidden' }}>
      {/* Cinematic Video Hero */}
      <section className="hero" style={{ height: '100vh', position: 'relative' }}>
        <video 
            autoPlay 
            muted 
            loop 
            playsInline
            className="hero-img-bg"
            style={{ filter: 'brightness(0.3) saturate(1.2)' }}
        >
          <source src="https://assets.mixkit.co/videos/preview/mixkit-top-vantage-point-of-the-ocean-from-the-beach-2479-large.mp4" type="video/mp4" />
        </video>
        <div className="hero-overlay"></div>
        
        <div className="floating-debris">
          {particles.map(p => (
            <div 
              key={p.id}
              className="debris-particle"
              style={{
                left: p.left,
                top: p.top,
                width: p.size,
                height: p.size,
                animationDelay: p.delay,
                animationDuration: p.duration
              }}
            />
          ))}
        </div>

        <div className="hero-content">
          <span className="hero-tag animate-fade-in">Autonomous Marine Intelligence</span>
          <h1 className="hero-title animate-fade-in">Cleaning our seas, one pixel at a time.</h1>
          <p className="hero-desc animate-fade-in">
            Merging multi-spectral satellite imagery with deep-learning neural networks 
            to identify, classify, and track marine debris across the global oceans.
          </p>
          <div className="hero-btns animate-fade-in" style={{ display: 'flex', gap: '20px', justifyContent: 'center' }}>
            <Link to="/visualization" className="btn btn-glow font-bold">
              Launch Tracking Dashboard
              <ChevronRight size={20} />
            </Link>
            <a href="#how-it-works" className="btn glass" style={{ color: 'white' }}>
              Explore Technology
            </a>
          </div>
        </div>
      </section>

      {/* Feature Section */}
      <section style={{ padding: '100px 4rem', background: '#020810', position: 'relative', zIndex: 10 }}>
        <div style={{ textAlign: 'center', marginBottom: '80px' }}>
            <h2 style={{ fontSize: '2.5rem', marginBottom: '20px' }}>Project Foundation</h2>
            <p style={{ color: '#94a3b8', maxWidth: '700px', marginInline: 'auto' }}>
                We combine environmental physics with state-of-the-art AI to build the world's most 
                accurate plastic detection pipeline.
            </p>
        </div>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '40px' }}>
            <div className="control-card">
                <Cpu color="#00f2ff" size={32} style={{marginBottom: '20px'}}/>
                <h3 style={{marginBottom: '10px', color: '#fff'}}>Neural Inference</h3>
                <p style={{color: '#94a3b8', fontSize: '0.9rem'}}>Lightweight U-Net architecture optimized for Sentinel-2 spectral bands, achieving 62%+ val mIoU.</p>
            </div>
            <div className="control-card">
                <Globe color="#00f2ff" size={32} style={{marginBottom: '20px'}}/>
                <h3 style={{marginBottom: '10px', color: '#fff'}}>Regional Scanning</h3>
                <p style={{color: '#94a3b8', fontSize: '0.9rem'}}>Dynamic bounding-box selection allows for real-time monitoring of critical coastal regions & gyres.</p>
            </div>
            <div className="control-card">
                <Shield color="#00f2ff" size={32} style={{marginBottom: '20px'}}/>
                <h3 style={{marginBottom: '10px', color: '#fff'}}>Signal Correction</h3>
                <p style={{color: '#94a3b8', fontSize: '0.9rem'}}>Integrated Biofouling models account for biological signal decay over time for consistent accuracy.</p>
            </div>
        </div>
      </section>

      {/* Dynamic Workflow Section */}
      <section id="how-it-works" style={{ padding: '100px 4rem', background: '#010409', position: 'relative' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: '80px', alignItems: 'center' }}>
              <div>
                  <h2 style={{ fontSize: '2.5rem', marginBottom: '30px', fontFamily: 'Outfit' }}>How it Works.</h2>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
                      <div style={{ display: 'flex', gap: '20px' }}>
                          <div className="glass" style={{ minWidth: '40px', height: '40px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#00f2ff', fontWeight: 'bold' }}>1</div>
                          <div>
                              <h4 style={{ color: '#fff', marginBottom: '8px' }}>Ingestion & Preprocessing</h4>
                              <p style={{ color: '#94a3b8', fontSize: '0.9rem' }}>Raw 11-band optical data is converted into standardized Z-score reflectance tensors.</p>
                          </div>
                      </div>
                      <div style={{ display: 'flex', gap: '20px' }}>
                          <div className="glass" style={{ minWidth: '40px', height: '40px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#00f2ff', fontWeight: 'bold' }}>2</div>
                          <div>
                              <h4 style={{ color: '#fff', marginBottom: '8px' }}>Deep Neural Segmentation</h4>
                              <p style={{ color: '#94a3b8', fontSize: '0.9rem' }}>The model isolates synthetic spectral signatures from natural sargassum and foam.</p>
                          </div>
                      </div>
                      <div style={{ display: 'flex', gap: '20px' }}>
                          <div className="glass" style={{ minWidth: '40px', height: '40px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#00f2ff', fontWeight: 'bold' }}>3</div>
                          <div>
                              <h4 style={{ color: '#fff', marginBottom: '8px' }}>Geospatial Interpolation</h4>
                              <p style={{ color: '#94a3b8', fontSize: '0.9rem' }}>Detected macroplastics are reverse-projected into WGS84 coordinates for global mapping.</p>
                          </div>
                      </div>
                  </div>
              </div>
              <div className="glass" style={{ padding: '2rem', borderRadius: '24px', border: '1px solid rgba(0, 242, 255, 0.2)', boxShadow: '0 0 40px rgba(0, 242, 255, 0.1)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
                      <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                          <Activity size={16} color="#00f2ff" />
                          <span style={{ fontSize: '0.8rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1px' }}>Spectral Response Matrix</span>
                      </div>
                      <span style={{ fontSize: '0.7rem', color: '#10b981' }}>Live Frame Analysed</span>
                  </div>
                  <div style={{ height: '300px', width: '100%', background: 'linear-gradient(45deg, #0a1016, #020810)', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative', overflow: 'hidden' }}>
                      {/* Abstract data visual */}
                      {Array.from({length: 40}).map((_, i) => (
                          <div key={i} style={{ 
                              position: 'absolute', 
                              width: '2px', 
                              height: `${Math.random() * 100}%`,
                              left: `${i * 2.5}%`,
                              background: i % 5 === 0 ? '#00f2ff' : '#112233',
                              opacity: i % 5 === 0 ? 0.6 : 0.2,
                              transition: 'height 0.5s ease'
                          }}/>
                      ))}
                      <div style={{ color: '#fff', zIndex: 1, textAlign: 'center' }}>
                          <Database size={40} style={{ marginBottom: '10px', opacity: 0.5 }} />
                          <div style={{ fontFamily: 'Outfit', fontWeight: 600 }}>Sentinel-2 Swath Processed</div>
                      </div>
                  </div>
              </div>
          </div>
      </section>

      {/* Simple Footer */}
      <footer style={{ padding: '60px 4rem', borderTop: '1px solid #112233', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontWeight: 'bold' }}>
              <span style={{ color: '#00f2ff' }}>OceanEye AI</span> 2026
          </div>
          <div style={{ display: 'flex', gap: '20px' }}>
              <a href="#" style={{ color: '#94a3b8' }}><Globe size={20} /></a>
              <a href="#" style={{ color: '#94a3b8' }}><Share2 size={20} /></a>
              <a href="#" style={{ color: '#94a3b8' }}><Mail size={20} /></a>
          </div>
      </footer>
    </div>
  );
};

export default Home;

import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Waves, BarChart2, Navigation, Trash2 } from 'lucide-react';

const Navbar = () => {
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const isSubPage = location.pathname !== '/';

  return (
    <nav className={`navbar ${scrolled || isSubPage ? 'glass' : ''}`}>
      <Link to="/" className="nav-brand">
        <Waves size={24} color="#00f2ff" />
        <span>OceanEye AI</span>
      </Link>
      <div className="nav-links">
        <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}>
          Mission
        </Link>
        <Link to="/visualization" className={`nav-link ${location.pathname === '/visualization' ? 'active' : ''}`}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <BarChart2 size={16} />
            Analytics Hub
          </div>
        </Link>
        <Link to="/trajectory" className={`nav-link ${location.pathname === '/trajectory' ? 'active' : ''}`}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Navigation size={16} />
            Drift Forecast
          </div>
        </Link>
        <Link to="/cleanup" className={`nav-link ${location.pathname === '/cleanup' ? 'active' : ''}`}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Trash2 size={16} />
            Clean-Up
          </div>
        </Link>
      </div>
    </nav>
  );
};

export default Navbar;

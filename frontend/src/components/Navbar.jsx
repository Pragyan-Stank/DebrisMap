import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Waves, BarChart2 } from 'lucide-react';

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

  const isVisPage = location.pathname === '/visualization';

  return (
    <nav className={`navbar ${scrolled || isVisPage ? 'glass' : ''}`}>
      <Link to="/" className="nav-brand">
        <Waves size={24} color="#00f2ff" />
        <span>OceanEye AI</span>
      </Link>
      <div className="nav-links">
        <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}>
          Mission
        </Link>
        <Link to="/visualization" className={`nav-link ${isVisPage ? 'active' : ''}`}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <BarChart2 size={16} />
            Analytics Hub
          </div>
        </Link>
      </div>
    </nav>
  );
};

export default Navbar;

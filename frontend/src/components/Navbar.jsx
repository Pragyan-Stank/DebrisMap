import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <nav className="navbar">
      <Link to="/" className="nav-brand">
        MarineDebris
      </Link>
      <div className="nav-links">
        <Link to="/" className="nav-link">Home</Link>
        <Link to="/visualization" className="nav-link">Visualization</Link>
      </div>
    </nav>
  );
};

export default Navbar;

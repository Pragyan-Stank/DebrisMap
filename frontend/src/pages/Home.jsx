import React from 'react';
import { Link } from 'react-router-dom';

const Home = () => {
  return (
    <div className="home-container">
      <h1 className="hero-title">Marine Plastic Detection System</h1>
      <p className="hero-description">
        AI-powered system combining U-Net deep learning models and Floating Debris Index (FDI) 
        computations on Sentinel-2 satellite imagery to accurately detect marine plastic pollution.
      </p>
      <Link to="/visualization" className="btn-primary">
        View Visualization
      </Link>
    </div>
  );
};

export default Home;

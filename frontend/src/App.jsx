import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Visualization from './pages/Visualization';
import Trajectory from './pages/Trajectory';
import Cleanup from './pages/Cleanup';

function App() {
  return (
    <Router>
      <div className="app-container">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/visualization" element={<Visualization />} />
            <Route path="/trajectory" element={<Trajectory />} />
            <Route path="/cleanup" element={<Cleanup />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;

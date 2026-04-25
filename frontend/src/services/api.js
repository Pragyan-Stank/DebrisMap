import axios from 'axios';

const API_URL = 'http://localhost:8000/api/v1';

export const getVisualizationData = async () => {
  try {
    const response = await axios.get(`${API_URL}/visualization-data`);
    return response.data;
  } catch (error) {
    console.error("Error fetching visualization data:", error);
    return null;
  }
};

export const predictDebris = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await axios.post(`${API_URL}/predict`, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return response.data;
  } catch (error) {
    console.error("Error predicting debris:", error);
    return null;
  }
};

export const fetchPatchInference = async (bbox, resolution, dateRange) => {
  try {
    const response = await axios.post(`${API_URL}/patch-inference`, {
      bbox,
      resolution,
      date_range: dateRange
    });
    return response.data;
  } catch (error) {
    console.error("Error fetching patch inference:", error);
    return null;
  }
};

export const fetchSegmentationMap = async (bbox, dateRange) => {
  try {
    const response = await axios.post(`${API_URL}/segmentation-map`, {
      bbox,
      resolution: 10,
      date_range: dateRange
    });
    return response.data;
  } catch (error) {
    console.error("Error fetching segmentation map:", error);
    return null;
  }
};

export const predictTrajectory = async (lat, lon, label = "MANUAL", nPixels = 100, confidence = 0.8) => {
  try {
    const response = await axios.post(`${API_URL}/trajectory/predict`, {
      lat, lon, label, n_pixels: nPixels, confidence
    });
    return response.data;
  } catch (error) {
    console.error("Error predicting trajectory:", error);
    return null;
  }
};

export const predictClusterTrajectories = async (clusters, source = "detection") => {
  try {
    const response = await axios.post(`${API_URL}/trajectory/from-clusters`, {
      clusters, source
    });
    return response.data;
  } catch (error) {
    console.error("Error predicting cluster trajectories:", error);
    return null;
  }
};

export const fetchWeather = async (lat, lon) => {
  try {
    const response = await axios.get(`${API_URL}/weather`, { params: { lat, lon } });
    return response.data;
  } catch (error) {
    console.error("Error fetching weather:", error);
    return null;
  }
};

export const fetchCleanupHotspots = async (hours = 72) => {
  try {
    const response = await axios.get(`${API_URL}/cleanup-hotspots`, { params: { hours } });
    return response.data;
  } catch (error) {
    console.error("Error fetching cleanup hotspots:", error);
    return null;
  }
};

export const fetchDetectionHistory = async () => {
  try {
    const response = await axios.get(`${API_URL}/detection-history`);
    return response.data;
  } catch (error) {
    console.error("Error fetching detection history:", error);
    return null;
  }
};

export const fetchThreatAssessment = async (lat, lon, density = 10, confidence = 0.5) => {
  try {
    const response = await axios.post(`${API_URL}/threat-assessment`, null, {
      params: { lat, lon, density, confidence }
    });
    return response.data;
  } catch (error) {
    console.error("Error fetching threat assessment:", error);
    return null;
  }
};

export const fetchInterceptPlan = async (debrisLat, debrisLon, vesselLat, vesselLon, speedKnots = 22) => {
  try {
    const response = await axios.post(`${API_URL}/intercept-plan`, null, {
      params: {
        debris_lat: debrisLat, debris_lon: debrisLon,
        vessel_lat: vesselLat, vessel_lon: vesselLon,
        vessel_speed_knots: speedKnots,
      }
    });
    return response.data;
  } catch (error) {
    console.error("Error computing intercept:", error);
    return null;
  }
};

export const fetchDispatchPlan = async (hours = 72) => {
  try {
    const response = await axios.post(`${API_URL}/dispatch-plan`, null, { params: { hours } });
    return response.data;
  } catch (error) {
    console.error("Error generating dispatch:", error);
    return null;
  }
};

export const fetchPersistentZones = async (hours = 168) => {
  try {
    const response = await axios.get(`${API_URL}/persistent-zones`, { params: { hours } });
    return response.data;
  } catch (error) {
    console.error("Error fetching persistent zones:", error);
    return null;
  }
};

export const fetchOptimalRoute = async (vesselLat, vesselLon, hours = 72) => {
  try {
    const response = await axios.post(`${API_URL}/optimal-route`, null, {
      params: {
        vessel_lat: vesselLat,
        vessel_lon: vesselLon,
        hours: hours
      }
    });
    return response.data;
  } catch (error) {
    console.error("Error fetching optimal route:", error);
    return null;
  }
};

export const seedDemoData = async (clearExisting = false) => {
  try {
    const response = await axios.post(`${API_URL}/seed-demo-data`, null, {
      params: { clear_existing: clearExisting }
    });
    return response.data;
  } catch (error) {
    console.error('Error seeding demo data:', error);
    return null;
  }
};

export const fetchAIDispatch = async (hours = 72) => {
  try {
    const response = await axios.post(`${API_URL}/ai-dispatch`, null, {
      params: { hours }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching AI dispatch:', error);
    return null;
  }
};

export const fetchAIPersistentAnalysis = async (hours = 168) => {
  try {
    const response = await axios.post(`${API_URL}/ai-persistent-analysis`, null, {
      params: { hours }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching AI persistent analysis:', error);
    return null;
  }
};

export const fetchAIInterceptAnalysis = async (debrisLat, debrisLon, vesselLat, vesselLon, speedKnots = 22) => {
  try {
    const response = await axios.post(`${API_URL}/ai-intercept-analysis`, null, {
      params: {
        debris_lat: debrisLat, debris_lon: debrisLon,
        vessel_lat: vesselLat, vessel_lon: vesselLon,
        vessel_speed_knots: speedKnots,
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching AI intercept analysis:', error);
    return null;
  }
};

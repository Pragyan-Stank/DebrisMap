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

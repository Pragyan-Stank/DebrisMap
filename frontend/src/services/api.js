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

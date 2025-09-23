import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const omrAPI = {
  // Upload OMR sheet
  uploadOMR: async (formData) => {
    const response = await api.post('/upload-omr', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Upload answer key
  uploadAnswerKey: async (answerKeyData) => {
    const formData = new FormData();
    Object.keys(answerKeyData).forEach(key => {
      if (key === 'answers') {
        formData.append(key, JSON.stringify(answerKeyData[key]));
      } else {
        formData.append(key, answerKeyData[key]);
      }
    });
    
    const response = await api.post('/upload-answer-key', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Evaluate OMR
  evaluateOMR: async (fileId, answerKeyId) => {
    const formData = new FormData();
    formData.append('file_id', fileId);
    formData.append('answer_key_id', answerKeyId);
    
    const response = await api.post('/evaluate-omr', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Get all answer keys
  getAnswerKeys: async () => {
    const response = await api.get('/answer-keys');
    return response.data;
  },

  // Get all OMR data
  getOMRData: async () => {
    const response = await api.get('/omr-data');
    return response.data;
  },

  // Get evaluation results
  getResults: async () => {
    const response = await api.get('/results');
    return response.data;
  },
};

export default api;
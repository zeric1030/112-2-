import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [videoUrl, setVideoUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [comments, setComments] = useState([]);
  const [keywords, setKeywords] = useState([]);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setComments([]);
    setKeywords([]);

    try {
      const response = await axios.post('http://localhost:5000/analyze', {
        video_url: videoUrl,
        api_key: apiKey
      });
      setComments(response.data.comments);
      setKeywords(response.data.keywords);
    } catch (err) {
      setError(err.response ? err.response.data.error : 'Error connecting to server');
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>YouTube Comments Analyzer</h1>
        <form onSubmit={handleSubmit}>
          <div>
            <label>Video URL:</label>
            <input type="text" value={videoUrl} onChange={(e) => setVideoUrl(e.target.value)} />
          </div>
          <div>
            <label>API Key:</label>
            <input type="text" value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
          </div>
          <button type="submit">Analyze</button>
        </form>
        {error && <p className="error">{error}</p>}
        <div>
          <h2>Comments</h2>
          <ul>
            {comments.map((comment, index) => (
              <li key={index}>{comment}</li>
            ))}
          </ul>
        </div>
        <div>
          <h2>Keywords</h2>
          <ul>
            {keywords.map((keyword, index) => (
              <li key={index}>{keyword.join(', ')}</li>
            ))}
          </ul>
        </div>
      </header>
    </div>
  );
}

export default App;

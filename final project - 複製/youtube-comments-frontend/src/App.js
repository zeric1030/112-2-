import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [videoUrl, setVideoUrl] = useState('');
  const [comments, setComments] = useState([]);
  const [keywords, setKeywords] = useState([]);
  const [positiveWordcloud, setPositiveWordcloud] = useState('');
  const [negativeWordcloud, setNegativeWordcloud] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setComments([]);
    setKeywords([]);
    setPositiveWordcloud('');
    setNegativeWordcloud('');

    try {
      const response = await axios.post('http://localhost:5000/analyze', {
        video_url: videoUrl
      });
      setComments(response.data.comments);
      setKeywords(response.data.keywords);
      setPositiveWordcloud(response.data.positive_wordcloud);
      setNegativeWordcloud(response.data.negative_wordcloud);
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
          <button type="submit">Analyze</button>
        </form>
        {error && <p className="error">{error}</p>}
        <div>
          <h2>Positive Wordcloud</h2>
          {positiveWordcloud && <img src={`http://localhost:5000/static/${positiveWordcloud}`} alt="Positive Wordcloud" />}
        </div>
        <div>
          <h2>Negative Wordcloud</h2>
          {negativeWordcloud && <img src={`http://localhost:5000/static/${negativeWordcloud}`} alt="Negative Wordcloud" />}
        </div>
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

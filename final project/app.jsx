// src/App.js
import React, { useState } from 'react';

function App() {
    const [videoId, setVideoId] = useState('');
    const [apiKey, setApiKey] = useState('');
    const [comments, setComments] = useState([]);
    const [predictions, setPredictions] = useState([]);
    const [loading, setLoading] = useState(false);

    const fetchComments = async () => {
        setLoading(true);
        const response = await fetch('http://localhost:5000/comments', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ videoId, apiKey })
        });
        const data = await response.json();
        setComments(data.comments);
        setLoading(false);
    };

    const predictEmotions = async () => {
        setLoading(true);
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ comments })
        });
        const data = await response.json();
        setPredictions(data.predictions);
        setLoading(false);
    };

    const generateWordcloud = async (title, maskPath, fontPath) => {
        const response = await fetch('http://localhost:5000/wordcloud', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ comments, title, maskPath, fontPath })
        });
        const data = await response.json();
        if (data.status === 'success') {
            alert(`${title} generated successfully!`);
        }
    };

    return (
        <div className="App">
            <h1>YouTube Comments Fetcher</h1>
            <input
                type="text"
                placeholder="Enter YouTube Video ID"
                value={videoId}
                onChange={(e) => setVideoId(e.target.value)}
            />
            <input
                type="text"
                placeholder="Enter API Key"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
            />
            <button onClick={fetchComments} disabled={loading}>Fetch Comments</button>
            <button onClick={predictEmotions} disabled={loading || comments.length === 0}>Predict Emotions</button>
            <button onClick={() => generateWordcloud('Positive Wordcloud', 'positive_mask.png', 'font_path')} disabled={loading || predictions.length === 0}>Generate Positive Wordcloud</button>
            <button onClick={() => generateWordcloud('Negative Wordcloud', 'negative_mask.png', 'font_path')} disabled={loading || predictions.length === 0}>Generate Negative Wordcloud</button>
            <div>
                {comments.map((comment, index) => (
                    <p key={index}>{comment}</p>
                ))}
            </div>
            <div>
                {predictions.map((pred, index) => (
                    <p key={index}>Comment: {comments[index]}, Prediction: {pred}</p>
                ))}
            </div>
        </div>
    );
}

export default App;

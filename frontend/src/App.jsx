import React, { useState } from 'react';
import './App.css';

// API endpoint
const API_ENDPOINT = 'http://localhost:8000/api/query';

// Helper component for displaying the results
const Results = ({ data }) => {
  if (!data) return null;

  if (data.type === 'NO_ANSWER') {
    return (
      <div className="no-answer">
        <div className="no-answer-icon">🔍</div>
        <div className="no-answer-text">
          <p><strong>No specific answer found</strong></p>
          <p style={{ marginTop: '0.5rem' }}>Try rephrasing your question or searching for related terms.</p>
        </div>
      </div>
    );
  }

  if (data.type === 'ANSWER') {
    return (
      <div className="result-card">
        <div className="result-header">
          <div className="result-icon">📜</div>
          <div className="result-title">Answer from NYC Administrative Code</div>
        </div>
        <div 
          className="answer-content" 
          dangerouslySetInnerHTML={{ __html: data.content }} 
        />
        {data.citations && data.citations.length > 0 && (
          <div style={{ marginTop: '1rem' }}>
            {data.citations.map((cite, index) => (
              <a 
                key={index} 
                href={cite.url} 
                target="_blank" 
                rel="noopener noreferrer" 
                className="citation" 
                style={{ color: 'white' }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path>
                  <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>
                </svg>
                View § {cite.section} on NYC Admin Code
              </a>
            ))}
          </div>
        )}
      </div>
    );
  }

  return null;
};


function App() {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [resultData, setResultData] = useState(null);
  const [error, setError] = useState('');

  const parseApiResponse = (answer) => {
    if (answer === 'NO_ANSWER') {
      return { type: 'NO_ANSWER' };
    }

    const citationRegex = /\(§\s*([\d\-\.]+)\)\s*\[([^\]]+)\]/g;
    let formattedContent = answer;
    const citations = [];
    let match;

    // Use a while loop to find all matches
    while ((match = citationRegex.exec(answer)) !== null) {
      citations.push({ section: match[1], url: match[2] });
    }

    // Replace all citation instances after collecting them
    formattedContent = formattedContent.replace(citationRegex, (match, section) => `<strong>(§ ${section})</strong>`);

    return {
      type: 'ANSWER',
      content: formattedContent,
      citations: citations,
    };
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setResultData(null);
    setError('');

    try {
      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch answer from the server.');
      }

      const data = await response.json();
      const parsedData = parseApiResponse(data.answer || 'NO_ANSWER');
      setResultData(parsedData);

    } catch (err) {
      console.error('Search error:', err);
      setError('Could not connect to the server. Please try again later.');
      // Fallback to sample answer for demonstration as in the original script
      const sampleAnswer = query.toLowerCase().includes('zoning') 
        ? 'The provisions of section 28-101.4.5 shall not be construed to affect the status of any non-conforming use or non-complying bulk otherwise permitted to be retained pursuant to the New York city zoning resolution. (§ 28-101.4.5.3) [https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-155473]'
        : 'NO_ANSWER';
      setResultData(parseApiResponse(sampleAnswer));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="header">
        <div className="logo-container">
          <h1>NYC Admin Code Assistant</h1>
        </div>
        <p className="subtitle">Get accurate answers about New York City Administrative Code</p>
      </header>

      <div className="search-container">
        <form className="search-form" onSubmit={handleSearch}>
          <input
            type="text"
            className="search-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask about permits, zoning, construction codes, violations..."
            required
            disabled={isLoading}
          />
          <button type="submit" className="search-button" disabled={isLoading}>
            <span>{isLoading ? 'Searching...' : 'Search'}</span>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8"></circle>
              <path d="m21 21-4.35-4.35"></path>
            </svg>
          </button>
        </form>
        
        {error && <div className="error-message active">{error}</div>}
      </div>

      {isLoading && (
        <div className="loading">
          <div className="spinner"></div>
          <div className="loading-text">Searching NYC Administrative Code...</div>
        </div>
      )}

      <div className="results">
        <Results data={resultData} />
      </div>

      <footer className="footer">
        <p>
          Powered by NYC Administrative Code from <a href="https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-1" target="_blank" rel="noopener noreferrer">American Legal Publishing</a>
        </p>
        <p style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
          This tool provides references to official NYC codes. Always verify with official sources for legal purposes.
        </p>
      </footer>
    </div>
  );
}

export default App;
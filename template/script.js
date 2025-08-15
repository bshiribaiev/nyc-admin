const searchForm = document.getElementById('searchForm');
const queryInput = document.getElementById('queryInput');
const searchButton = document.getElementById('searchButton');
const loading = document.getElementById('loading');
const resultContainer = document.getElementById('resultContainer');
const resultContent = document.getElementById('resultContent');

// API configuration
const API_BASE_URL = 'http://localhost:8000';

function setQuery(query) {
    queryInput.value = query;
    queryInput.focus();
}

function showLoading() {
    loading.style.display = 'block';
    resultContainer.style.display = 'none';
    searchButton.disabled = true;
    searchButton.textContent = '🔍 Searching...';
}

function hideLoading() {
    loading.style.display = 'none';
    searchButton.disabled = false;
    searchButton.textContent = 'Search';
}

function displayResult(data) {
    resultContainer.style.display = 'block';
    
    if (!data.success) {
        resultContent.innerHTML = `
            <div class="error">
                <strong>Error:</strong> ${data.error || 'Unknown error occurred'}
            </div>
        `;
        return;
    }

    if (data.answer === 'NO_ANSWER') {
        resultContent.innerHTML = `
            <div class="no-answer">
                <strong>No Answer Found</strong><br>
                Sorry, I couldn't find relevant information in the NYC Administrative Code for your query.
                Try rephrasing your question or using different keywords.
            </div>
            ${getDebugInfo(data)}
        `;
        return;
    }

    // Convert URLs in answer to clickable links
    const answerWithLinks = data.answer.replace(
        /\[([^\]]+)\]/g, 
        '<a href="$1" target="_blank">📖 View Section</a>'
    );

    resultContent.innerHTML = `
        <div class="answer">${answerWithLinks}</div>
        
        <div class="metadata">
            <div class="meta-item">
                <div class="meta-value">${data.sections_found?.length || 0}</div>
                <div class="meta-label">Sections Found</div>
            </div>
            <div class="meta-item">
                <div class="meta-value">${data.snippets_used || 0}</div>
                <div class="meta-label">Sources Used</div>
            </div>
            <div class="meta-item">
                <div class="meta-value">${data.response_time || 0}s</div>
                <div class="meta-label">Response Time</div>
            </div>
            <div class="meta-item">
                <div class="meta-value">${data.debug_info?.urls_processed || 0}/${data.debug_info?.urls_found || 0}</div>
                <div class="meta-label">URLs Processed</div>
            </div>
        </div>

        ${data.sections_found?.length > 0 ? `
            <div class="sections">
                <strong>Code Sections Referenced:</strong><br>
                ${data.sections_found.map(section => `<span class="section-tag">§ ${section}</span>`).join('')}
            </div>
        ` : ''}

        ${getDebugInfo(data)}
    `;
}

function getDebugInfo(data) {
    if (!data.debug_info) return '';
    
    return `
        <div class="debug-info">
            <button class="debug-toggle" onclick="toggleDebug()">🔧 Show Debug Info</button>
            <div class="debug-content" id="debugContent">
                <strong>Snippets Used:</strong><br>
                ${data.debug_info.snippets?.map(s => `
                    <div style="margin: 0.5rem 0; padding: 0.5rem; background: white; border-radius: 4px;">
                        <strong>${s.section}</strong> (Score: ${s.score})<br>
                        <small>${s.title}</small><br>
                        <a href="${s.url}" target="_blank" style="font-size: 0.8rem;">View Source</a>
                    </div>
                `).join('') || 'No snippets available'}
                
                ${data.debug_info.failed_urls?.length > 0 ? `
                    <br><strong>Failed URLs:</strong><br>
                    ${data.debug_info.failed_urls.map(url => `<small>${url}</small>`).join('<br>')}
                ` : ''}
            </div>
        </div>
    `;
}

function toggleDebug() {
    const debugContent = document.getElementById('debugContent');
    debugContent.style.display = debugContent.style.display === 'none' ? 'block' : 'none';
}

async function search() {
    const query = queryInput.value.trim();
    if (!query) {
        alert('Please enter a question');
        return;
    }

    showLoading();

    try {
        // Use absolute URL to avoid CORS issues
        const response = await fetch(`${API_BASE_URL}/api/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        displayResult(data);
    } catch (error) {
        resultContainer.style.display = 'block';
        resultContent.innerHTML = `
            <div class="error">
                <strong>Connection Error:</strong> ${error.message}<br>
                <small>
                    Make sure:
                    <ul style="margin-top: 0.5rem; text-align: left;">
                        <li>FastAPI backend is running: <code>uvicorn app:app --reload --port 8000</code></li>
                        <li>Backend URL is accessible at: <a href="${API_BASE_URL}/api/health" target="_blank">${API_BASE_URL}/api/health</a></li>
                        <li>Check browser console for CORS errors</li>
                    </ul>
                </small>
            </div>
        `;
    } finally {
        hideLoading();
    }
}

// Event listeners
searchForm.addEventListener('submit', function(e) {
    e.preventDefault();
    search();
});

searchButton.addEventListener('click', function(e) {
    e.preventDefault();
    search();
});

// Focus on input when page loads
queryInput.focus();

// Handle Enter key in input
queryInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        e.preventDefault();
        search();
    }
});

// Check if backend is running on page load
async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        if (response.ok) {
            console.log('✅ Backend is running');
        }
    } catch (error) {
        console.warn('⚠️ Backend not responding:', error.message);
        
        // Show a subtle warning
        const container = document.querySelector('.container');
        const warning = document.createElement('div');
        warning.style.cssText = `
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 0.75rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            text-align: center;
            font-size: 0.9rem;
        `;
        warning.innerHTML = `
            ⚠️ Backend not detected. Make sure to run: <code>uvicorn app:app --reload --port 8000</code>
        `;
        container.insertBefore(warning, container.firstChild);
    }
}

// Check backend health on page load
checkBackendHealth();
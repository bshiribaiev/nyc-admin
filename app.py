#!/usr/bin/env python3
"""
FastAPI backend for NYC Admin Code RAG system
Run with: uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import re
import traceback
import asyncio
from pathlib import Path

# Import your existing RAG functions
from live_rag import (
    pick_urls_for_query, fetch_url, html_to_text, extract_meta, 
    pick_snippets, ask_gemini, enforce_citation_format, validate_answer
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Your question about NYC Administrative Code")

class SnippetInfo(BaseModel):
    section: str
    url: str
    title: str
    score: float

class DebugInfo(BaseModel):
    snippets: List[SnippetInfo]
    failed_urls: List[str]
    urls_found: int
    urls_processed: int

class ValidationInfo(BaseModel):
    valid: bool
    issues: List[str]

class QueryResponse(BaseModel):
    success: bool
    query: str
    answer: str
    sections_found: List[str] = Field(description="Section numbers found in the response")
    urls_in_response: List[str] = Field(description="URLs cited in the response")
    snippets_used: int
    response_time: float = Field(description="Response time in seconds")
    validation: ValidationInfo
    debug_info: Optional[DebugInfo] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: float
    cache_size: int

class StatsResponse(BaseModel):
    cache_size: int
    status: str
    uptime: float

# Initialize FastAPI app
app = FastAPI(
    title="NYC Admin Code RAG API",
    description="Get instant answers about New York City Administrative Code using advanced RAG technology",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track app start time for uptime calculation
START_TIME = time.time()

# Serve static files (for the frontend)
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend page"""
    try:
        with open("templates/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html><body>
        <h1>NYC Admin Code RAG API</h1>
        <p>Frontend template not found. API is running at:</p>
        <ul>
            <li><a href="/docs">Interactive API Documentation</a></li>
            <li><a href="/redoc">ReDoc Documentation</a></li>
            <li><a href="/health">Health Check</a></li>
        </ul>
        </body></html>
        """)

async def process_rag_query(query: str) -> QueryResponse:
    """Async wrapper around your existing RAG pipeline"""
    start_time = time.time()
    
    try:
        # Step 1: Get URLs (run in executor to avoid blocking)
        loop = asyncio.get_event_loop()
        urls = await loop.run_in_executor(None, pick_urls_for_query, query)
        urls = urls[:6]
        
        if not urls:
            return QueryResponse(
                success=False,
                query=query,
                answer="NO_ANSWER",
                sections_found=[],
                urls_in_response=[],
                snippets_used=0,
                response_time=time.time() - start_time,
                validation=ValidationInfo(valid=False, issues=["No relevant URLs found"]),
                error="No relevant URLs found"
            )
        
        # Step 2: Fetch content (can be done concurrently)
        url_to_text, url_meta = {}, {}
        failed_urls = []
        
        async def fetch_single_url(url: str):
            try:
                # Run blocking operations in executor
                html = await loop.run_in_executor(None, fetch_url, url)
                text = await loop.run_in_executor(None, html_to_text, html)
                meta = await loop.run_in_executor(None, extract_meta, html)
                return url, text, meta, None
            except Exception as e:
                return url, None, None, str(e)
        
        # Fetch all URLs concurrently
        fetch_tasks = [fetch_single_url(url) for url in urls]
        fetch_results = await asyncio.gather(*fetch_tasks)
        
        for url, text, meta, error in fetch_results:
            if error:
                failed_urls.append(url)
            else:
                url_to_text[url] = text
                url_meta[url] = meta
        
        # Step 3: Generate snippets
        snippets = await loop.run_in_executor(
            None, pick_snippets, query, url_to_text, url_meta, 5
        )
        
        if not snippets:
            return QueryResponse(
                success=False,
                query=query,
                answer="NO_ANSWER",
                sections_found=[],
                urls_in_response=[],
                snippets_used=0,
                response_time=time.time() - start_time,
                validation=ValidationInfo(valid=False, issues=["No relevant content found"]),
                debug_info=DebugInfo(
                    snippets=[],
                    failed_urls=failed_urls,
                    urls_found=len(urls),
                    urls_processed=len(url_to_text)
                ),
                error="No relevant content found in documents"
            )
        
        # Step 4: Generate answer
        answer = await loop.run_in_executor(None, ask_gemini, query, snippets)
        
        # Step 5: Format enforcement
        if answer != "NO_ANSWER":
            citation_pattern = re.compile(r'\(\s*§\s*[\d\-–.]+\s*\)\s*\[\s*https?://[^\]]+\s*\]')
            if not citation_pattern.search(answer):
                answer = await loop.run_in_executor(None, enforce_citation_format, answer, snippets)
        
        # Step 6: Validate
        validation = await loop.run_in_executor(None, validate_answer, answer, snippets)
        
        if not validation["valid"]:
            answer = "NO_ANSWER"
        
        # Extract sections and URLs for response
        sections_found = re.findall(r'§\s*([\d\-–.]+)', answer) if answer != "NO_ANSWER" else []
        urls_in_response = re.findall(r'https://[^\]]+', answer) if answer != "NO_ANSWER" else []
        
        response_time = time.time() - start_time
        
        return QueryResponse(
            success=answer != "NO_ANSWER",
            query=query,
            answer=answer,
            sections_found=sections_found,
            urls_in_response=urls_in_response,
            snippets_used=len(snippets),
            response_time=round(response_time, 2),
            validation=ValidationInfo(valid=validation["valid"], issues=validation["issues"]),
            debug_info=DebugInfo(
                snippets=[
                    SnippetInfo(
                        section=s['section'],
                        url=s['url'],
                        title=s.get('h1') or s.get('title', ''),
                        score=s['score']
                    ) for s in snippets
                ],
                failed_urls=failed_urls,
                urls_found=len(urls),
                urls_processed=len(url_to_text)
            )
        )
        
    except Exception as e:
        response_time = time.time() - start_time
        print(f"[API] Error: {e}")
        traceback.print_exc()
        
        return QueryResponse(
            success=False,
            query=query,
            answer="ERROR",
            sections_found=[],
            urls_in_response=[],
            snippets_used=0,
            response_time=round(response_time, 2),
            validation=ValidationInfo(valid=False, issues=[f"Internal error: {str(e)}"]),
            error=f"Internal server error: {str(e)}"
        )

@app.post("/api/search", response_model=QueryResponse)
async def search_admin_code(request: QueryRequest) -> QueryResponse:
    """
    Search the NYC Administrative Code and get AI-powered answers
    
    - **query**: Your question about NYC Administrative Code
    - Returns: Detailed response with answer, citations, and metadata
    """
    query = request.query.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    print(f"[API] Processing query: {query}")
    
    result = await process_rag_query(query)
    
    print(f"[API] Completed in {result.response_time}s: {'SUCCESS' if result.success else 'FAILED'}")
    
    return result

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Import here to avoid circular imports
    from live_rag import FETCH_CACHE
    
    return HealthResponse(
        status="healthy",
        service="NYC Admin Code RAG API",
        timestamp=time.time(),
        cache_size=len(FETCH_CACHE)
    )

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    from live_rag import FETCH_CACHE
    
    return StatsResponse(
        cache_size=len(FETCH_CACHE),
        status="operational",
        uptime=time.time() - START_TIME
    )

@app.post("/api/search/batch")
async def search_batch(queries: List[QueryRequest]) -> List[QueryResponse]:
    """
    Process multiple queries concurrently
    Useful for testing or batch processing
    """
    if len(queries) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 queries per batch")
    
    tasks = [process_rag_query(q.query) for q in queries]
    results = await asyncio.gather(*tasks)
    
    return results

@app.post("/api/admin/clear-cache")
async def clear_cache(background_tasks: BackgroundTasks):
    """Admin endpoint to clear the fetch cache"""
    from live_rag import FETCH_CACHE
    
    cache_size = len(FETCH_CACHE)
    FETCH_CACHE.clear()
    
    return {"message": f"Cleared {cache_size} cache entries"}

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting NYC Admin Code RAG API...")
    print("📝 Frontend available at: http://localhost:8000")
    print("🔗 API docs at: http://localhost:8000/docs")
    print("📚 ReDoc at: http://localhost:8000/redoc")
    
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
import logging
import time
from typing import List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis

from config.settings import REDIS_URL, API_CONFIG
from app.database import AsyncDatabasePool
from app.qa_system import QASystem

logger = logging.getLogger(__name__)

# Redis client
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Database pool
db_pool = AsyncDatabasePool()


# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting NYC Admin Code API")
    await db_pool.initialize()
    yield
    # Shutdown
    logger.info("Shutting down NYC Admin Code API")
    await db_pool.close()


# Create FastAPI app
app = FastAPI(
    title="NYC Admin Code Q&A API",
    description="API for querying NYC Administrative Code",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QuestionRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    top_k: Optional[int] = Field(5, ge=1, le=20)


class AnswerResponse(BaseModel):
    answer: str
    sources: List[dict]
    query_time: float


class SectionResponse(BaseModel):
    section_number: str
    title: str
    content: str
    last_updated: datetime


class StatsResponse(BaseModel):
    total_sections: int
    last_updated: Optional[datetime]
    total_queries: int


# API endpoints
@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Answer a question about NYC Admin Code"""
    start_time = time.time()
    
    qa_system = QASystem(db_pool)
    
    # Search for relevant sections
    sections = await qa_system.search_similar_sections(
        request.query, 
        request.top_k
    )
    
    if not sections:
        raise HTTPException(status_code=404, detail="No relevant sections found")
    
    # Generate answer
    answer = await qa_system.generate_answer(request.query, sections)
    
    # Format sources
    sources = [
        {
            "section_number": s["section_number"],
            "title": s["title"],
            "excerpt": s["content"][:300] + "..."
        }
        for s in sections
    ]
    
    # Log query
    await db_pool.execute("""
        INSERT INTO query_logs (query_text, response_time, result_count)
        VALUES ($1, $2, $3)
    """, request.query, time.time() - start_time, len(sections))
    
    # Update metrics
    redis_client.incr("total_queries")
    
    return AnswerResponse(
        answer=answer,
        sources=sources,
        query_time=time.time() - start_time
    )


@app.get("/api/sections/{section_number}", response_model=SectionResponse)
async def get_section(section_number: str):
    """Get a specific section by number"""
    result = await db_pool.fetchrow("""
        SELECT section_number, title, content, last_updated
        FROM sections
        WHERE section_number = $1
    """, section_number)
    
    if not result:
        raise HTTPException(status_code=404, detail="Section not found")
    
    return SectionResponse(**dict(result))


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    # Get database stats
    db_stats = await db_pool.fetchrow("""
        SELECT 
            COUNT(*) as total_sections,
            MAX(last_updated) as last_updated
        FROM sections
    """)
    
    # Get Redis stats
    total_queries = redis_client.get("total_queries") or 0
    
    return StatsResponse(
        total_sections=db_stats["total_sections"],
        last_updated=db_stats["last_updated"],
        total_queries=int(total_queries)
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database
        await db_pool.execute("SELECT 1")
        
        # Check Redis
        redis_client.ping()
        
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "NYC Admin Code Q&A API",
        "version": "1.0.0",
        "endpoints": {
            "ask": "POST /api/ask",
            "section": "GET /api/sections/{section_number}",
            "stats": "GET /api/stats",
            "health": "GET /api/health"
        }
    }
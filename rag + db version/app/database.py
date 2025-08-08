import asyncpg
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager, asynccontextmanager
import logging
from typing import Optional
from config.settings import DATABASE_CONFIG

logger = logging.getLogger(__name__)

# SQL Schema
SCHEMA_SQL = """
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Main sections table
CREATE TABLE IF NOT EXISTS sections (
    id SERIAL PRIMARY KEY,
    section_number VARCHAR(50) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    chapter VARCHAR(100),
    url TEXT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    content_hash VARCHAR(64),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_section_number ON sections(section_number);
CREATE INDEX IF NOT EXISTS idx_full_text ON sections USING gin(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_last_updated ON sections(last_updated);

-- Chunks for embeddings
CREATE TABLE IF NOT EXISTS section_chunks (
    id SERIAL PRIMARY KEY,
    section_id INTEGER REFERENCES sections(id) ON DELETE CASCADE,
    chunk_index INTEGER,
    chunk_text TEXT,
    embedding vector(384),  -- Sentence Transformer (all-MiniLM-L6-v2) embedding size
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE(section_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_section_chunks_section_id ON section_chunks(section_id);
CREATE INDEX IF NOT EXISTS idx_embeddings ON section_chunks USING ivfflat (embedding vector_cosine_ops);

-- Cross references between sections
CREATE TABLE IF NOT EXISTS cross_references (
    id SERIAL PRIMARY KEY,
    from_section_id INTEGER REFERENCES sections(id) ON DELETE CASCADE,
    to_section_number VARCHAR(50),
    reference_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cross_ref_from ON cross_references(from_section_id);
CREATE INDEX IF NOT EXISTS idx_cross_ref_to ON cross_references(to_section_number);

-- Track scraping history
CREATE TABLE IF NOT EXISTS scrape_history (
    id SERIAL PRIMARY KEY,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    sections_processed INTEGER DEFAULT 0,
    sections_updated INTEGER DEFAULT 0,
    errors JSONB DEFAULT '[]'::jsonb
);

-- Query logs for analytics
CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    response_time FLOAT,
    result_count INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_query_logs_timestamp ON query_logs(timestamp);
"""

class DatabaseManager:
    def __init__(self):
        self.connection = None
        
    def connect(self):
        try:
            self.connection = psycopg2.connect(**DATABASE_CONFIG)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
            
    def initialize_schema(self):
        if not self.connection:
            raise RuntimeError("Database not connected")
            
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(SCHEMA_SQL)
                self.connection.commit()
            logger.info("Database schema initialized")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            self.connection.rollback()
            raise
            
    def close(self):
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
            
    @contextmanager
    def get_cursor(self, dict_cursor=True):
        cursor_factory = RealDictCursor if dict_cursor else None
        cursor = self.connection.cursor(cursor_factory=cursor_factory)
        try:
            yield cursor
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        finally:
            cursor.close()


class AsyncDatabasePool:    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        
    async def initialize(self):
        try:
            self.pool = await asyncpg.create_pool(
                **DATABASE_CONFIG,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            logger.info("Async database pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
            
    async def close(self):
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")
    
    # Add async context manager support
    async def __aenter__(self):
        """Enter the async context manager - returns self for use with 'async with'"""
        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call initialize() first.")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager - no cleanup needed here"""
        # We don't close the pool here because it should persist across requests
        # The pool will be closed when the application shuts down
        return False
            
    # Acquire connection from the pool        
    @asynccontextmanager
    async def acquire(self):
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
            
        async with self.pool.acquire() as connection:
            yield connection
            
    async def execute(self, query: str, *args):
        async with self.acquire() as connection:
            return await connection.execute(query, *args)
    
    # Fetch multiple rows        
    async def fetch(self, query: str, *args):
        async with self.acquire() as connection:
            return await connection.fetch(query, *args)
        
    # Fetch single row        
    async def fetchrow(self, query: str, *args):
        async with self.acquire() as connection:
            return await connection.fetchrow(query, *args)
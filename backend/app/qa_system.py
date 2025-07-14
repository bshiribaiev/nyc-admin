import logging
from typing import List, Dict, Optional
import numpy as np
from openai import OpenAI

from config.settings import OPENAI_API_KEY, EMBEDDING_CONFIG

logger = logging.getLogger(__name__)


class QASystem:
    """Question answering system using embeddings and LLM"""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        
    async def search_similar_sections(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Dict]:
        """Search for sections similar to the query"""
        if not self.client:
            # Fallback to keyword search
            return await self._keyword_search(query, top_k)
            
        # Generate query embedding
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_CONFIG["model"],
                input=query
            )
            query_embedding = response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return await self._keyword_search(query, top_k)
            
        # Search similar vectors
        async with self.db_pool.acquire() as conn:
            results = await conn.fetch("""
                SELECT DISTINCT ON (s.id)
                    s.id, s.section_number, s.title, s.content,
                    MIN(c.embedding <-> $1::vector) as distance
                FROM section_chunks c
                JOIN sections s ON c.section_id = s.id
                GROUP BY s.id
                ORDER BY s.id, distance
                LIMIT $2
            """, query_embedding, top_k)
            
        return [dict(row) for row in results]
        
    async def _keyword_search(
        self, 
        query: str, 
        top_k: int
    ) -> List[Dict]:
        """Fallback keyword search"""
        async with self.db_pool.acquire() as conn:
            results = await conn.fetch("""
                SELECT id, section_number, title, content,
                       ts_rank(to_tsvector('english', content), 
                              plainto_tsquery('english', $1)) as rank
                FROM sections
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
                ORDER BY rank DESC
                LIMIT $2
            """, query, top_k)
            
        return [dict(row) for row in results]
        
    async def generate_answer(
        self, 
        query: str, 
        context_sections: List[Dict]
    ) -> str:
        """Generate answer using LLM"""
        if not self.client:
            # Fallback to simple formatting
            return self._format_sections(context_sections)
            
        # Build context
        context = "\n\n".join([
            f"Section {s['section_number']}: {s['title']}\n{s['content'][:1500]}..."
            for s in context_sections
        ])
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert on NYC Administrative Code. 
                        Answer questions based only on the provided context. 
                        Always cite specific section numbers."""
                    },
                    {
                        "role": "user",
                        "content": f"""Context:
{context}

Question: {query}

Provide a clear, accurate answer with section citations."""
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return self._format_sections(context_sections)
            
    def _format_sections(self, sections: List[Dict]) -> str:
        """Format sections as a simple answer"""
        if not sections:
            return "No relevant sections found."
            
        answer = "Based on the NYC Administrative Code:\n\n"
        
        for section in sections[:3]:  # Top 3 sections
            answer += f"**{section['section_number']}: {section['title']}**\n"
            answer += f"{section['content'][:500]}...\n\n"
            
        return answer
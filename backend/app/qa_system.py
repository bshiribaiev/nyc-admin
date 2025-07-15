import logging
from typing import List, Dict, Optional
import google.generativeai as genai

from config.settings import GEMINI_API_KEY

logger = logging.getLogger(__name__)


class QASystem:
    def __init__(self, db_pool):
        self.db_pool = db_pool
        
        # Initialize Gemini
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.embedding_model = 'models/embedding-001'
        else:
            self.model = None
            self.embedding_model = None
            logger.warning("Gemini API key not configured")
    
    async def search_similar_sections(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Dict]:
        """Search for sections similar to the query"""
        
        if not GEMINI_API_KEY:
            # Fallback to keyword search
            return await self._keyword_search(query, top_k)
        
        # Generate query embedding using Gemini
        try:
            # Gemini embeddings API
            result = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query",  # Optimized for search queries
            )
            query_embedding = result['embedding']
            
        except Exception as e:
            logger.error(f"Error generating Gemini embedding: {e}")
            return await self._keyword_search(query, top_k)
        
        # Search similar vectors in database
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
        """Generate answer using Gemini"""
        
        if not self.model:
            # Fallback to simple formatting
            return self._format_sections(context_sections)
        
        # Build context from retrieved sections
        context = "\n\n".join([
            f"Section {s['section_number']}: {s['title']}\n{s['content'][:1500]}..."
            for s in context_sections
        ])
        
        # Create prompt for Gemini
        prompt = f"""You are an expert on NYC Administrative Code. 
Answer the question based ONLY on the provided context. 
Always cite specific section numbers when referencing the code.

Context from NYC Administrative Code:
{context}

Question: {query}

Provide a clear, accurate answer with section citations. If the answer cannot be found in the provided context, say so."""
        
        try:
            # Generate response using Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temperature for factual responses
                    max_output_tokens=1000,
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating Gemini answer: {e}")
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
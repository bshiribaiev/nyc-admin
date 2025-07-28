# app/qa_system.py - Fixed version
import logging
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class QASystem:
    def __init__(self, db_pool):
        self.db_pool = db_pool
        # Initialize Sentence Transformer for embeddings
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded Sentence Transformer model")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    async def search_similar_sections(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Dict]:
        """Search for sections similar to the query"""
        
        if not self.model:
            # Fallback to keyword search if no embedding model
            return await self._keyword_search(query, top_k)
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            
            # Convert numpy array to list for PostgreSQL
            embedding_list = query_embedding.tolist()
            
            # Format the embedding as a PostgreSQL vector string
            # PostgreSQL expects format: '[0.1, 0.2, 0.3, ...]'
            embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'
            
            async with self.db_pool.acquire() as conn:
                # Search using pgvector
                results = await conn.fetch("""
                    SELECT DISTINCT ON (s.id)
                        s.id, s.section_number, s.title, s.content,
                        MIN(c.embedding <-> $1::vector) as distance
                    FROM section_chunks c
                    JOIN sections s ON c.section_id = s.id
                    GROUP BY s.id
                    ORDER BY s.id, distance
                    LIMIT $2
                """, embedding_str, top_k)
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            # Fallback to keyword search
            return await self._keyword_search(query, top_k)
    
    async def _keyword_search(
        self, 
        query: str, 
        top_k: int
    ) -> List[Dict]:
        """Fallback keyword search"""
        try:
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
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    async def generate_answer(
        self, 
        query: str, 
        context_sections: List[Dict]
    ) -> str:
        """Generate answer from context sections"""
        if not context_sections:
            return "No relevant sections found."
        
        # Format answer with relevant sections
        answer = f"Based on NYC Administrative Code:\n\n"
        
        for i, section in enumerate(context_sections[:3], 1):
            answer += f"{i}. **{section['section_number']}: {section['title']}**\n"
            
            # Extract most relevant part
            content = section['content']
            
            # Simple relevance: find sentences containing query words
            query_words = query.lower().split()
            sentences = content.split('. ')
            relevant_sentences = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in query_words if len(word) > 3):
                    relevant_sentences.append(sentence.strip())
                    if len(relevant_sentences) >= 3:
                        break
            
            # Show relevant sentences or beginning of content
            if relevant_sentences:
                answer += '. '.join(relevant_sentences) + ".\n\n"
            else:
                # Show first 400 characters if no specific matches
                answer += content[:400].strip() + "...\n\n"
        
        return answer.strip()


# Alternative: If the above still doesn't work, try this simpler version
class SimpleQASystem:
    """Simpler QA system that relies more on keyword search"""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
        
    async def search_similar_sections(self, query: str, top_k: int = 5) -> List[Dict]:
        """Simple keyword-based search"""
        try:
            async with self.db_pool.acquire() as conn:
                # First try exact section number search
                if query.strip().startswith('§') or any(c.isdigit() for c in query[:3]):
                    results = await conn.fetch("""
                        SELECT id, section_number, title, content
                        FROM sections
                        WHERE section_number ILIKE $1
                        LIMIT $2
                    """, f"%{query.strip()}%", top_k)
                    
                    if results:
                        return [dict(row) for row in results]
                
                # Full text search
                results = await conn.fetch("""
                    SELECT id, section_number, title, content,
                           ts_rank(
                               to_tsvector('english', title || ' ' || content), 
                               plainto_tsquery('english', $1)
                           ) as rank
                    FROM sections
                    WHERE to_tsvector('english', title || ' ' || content) @@ 
                          plainto_tsquery('english', $1)
                    ORDER BY rank DESC
                    LIMIT $2
                """, query, top_k)
                
                if results:
                    return [dict(row) for row in results]
                
                # Fallback: ILIKE search on title and content
                results = await conn.fetch("""
                    SELECT id, section_number, title, content
                    FROM sections
                    WHERE title ILIKE $1 OR content ILIKE $1
                    LIMIT $2
                """, f"%{query}%", top_k)
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    async def generate_answer(self, query: str, context_sections: List[Dict]) -> str:
        """Generate simple formatted answer"""
        if not context_sections:
            return "No relevant sections found in NYC Administrative Code."
        
        answer = f"Results for '{query}':\n\n"
        
        for section in context_sections[:3]:
            answer += f"**{section['section_number']}: {section['title']}**\n"
            answer += f"{section['content'][:500]}...\n\n"
        
        return answer
import logging
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
import google.generativeai as genai

from app.database import AsyncDatabasePool
from config.settings import GEMINI_API_KEY

logger = logging.getLogger(__name__)

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class QASystem:
    def __init__(self, db_pool: AsyncDatabasePool):
        self.db_pool = db_pool
        self.model = None
        self.generative_model = None
        self._initialize_models()

    def _initialize_models(self):
        """Initializes the embedding and generative models."""
        try:
            # For creating vector embeddings for semantic search
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence Transformer model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer model: {e}")

        try:
            # For generating natural language answers
            if GEMINI_API_KEY:
                self.generative_model = genai.GenerativeModel('gemini-pro')
                logger.info("Generative AI model configured.")
            else:
                logger.warning("GEMINI_API_KEY not set. Generative model not available.")
        except Exception as e:
            logger.error(f"Failed to configure Generative AI model: {e}")
    
    async def search_similar_sections(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.model:
            logger.error("Embedding model not available. Cannot perform search.")
            return []

        title_filter = self._detect_title_filter(query)  
        filter_sql = ""
        filter_args = []
        if title_filter:
            filter_sql = "AND s.title_number = ANY($2::int[])"
            filter_args = [title_filter]

        # Use websearch_to_tsquery for “normal” queries
        keyword_sql = f"""
            WITH kw AS (
            SELECT
                s.id,
                s.section_number,
                s.title,
                ts_rank_cd(
                setweight(to_tsvector('english', coalesce(s.title,'')), 'A') ||
                setweight(to_tsvector('english', coalesce(sc.chunk_text,'')), 'C'),
                websearch_to_tsquery('english', $1)
                ) AS rank
            FROM section_chunks sc
            JOIN sections s ON sc.section_id = s.id
            WHERE (
                setweight(to_tsvector('english', coalesce(s.title,'')), 'A') ||
                setweight(to_tsvector('english', coalesce(sc.chunk_text,'')), 'C')
            ) @@ websearch_to_tsquery('english', $1)
            {filter_sql}
            )
            SELECT id, section_number, title, MAX(rank) AS rank
            FROM kw
            GROUP BY id, section_number, title
            ORDER BY MAX(rank) DESC
            LIMIT 100;
        """
        keyword_params = [query] + filter_args
        keyword_results = await self.db_pool.fetch(keyword_sql, *keyword_params)

        # build embedding once
        query_embedding = self.model.encode(query, convert_to_numpy=True).tolist()
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        if keyword_results:
            target_ids = [r['id'] for r in keyword_results]

            semantic_results = await self.db_pool.fetch("""
                SELECT id, section_number, title, content, chunk_text, distance
                FROM (
                SELECT
                    s.id,
                    s.section_number,
                    s.title,
                    s.content,
                    sc.chunk_text,
                    (sc.embedding <=> $1::vector) AS distance,
                    ROW_NUMBER() OVER (
                    PARTITION BY s.id
                    ORDER BY sc.embedding <=> $1::vector
                    ) AS rn
                FROM section_chunks sc
                JOIN sections s ON sc.section_id = s.id
                WHERE s.id = ANY($2::int[])
                ) t
                WHERE rn = 1
                ORDER BY distance ASC
                LIMIT $3;
            """, embedding_str, target_ids, top_k)

            # Blend FTS rank and vector distance
            ts_ranks = {r['id']: r['rank'] for r in keyword_results}
            maxr = max(ts_ranks.values()) if ts_ranks else 1.0
            rows = [dict(r) for r in semantic_results]
            for r in rows:
                r_ts = ts_ranks.get(r['id'], 0.0) / (maxr or 1.0)
                penalty = 0.15 if ("amendments" in r.get("title","").lower() or "definitions" in r.get("title","").lower()) else 0.0
                r['score'] = 0.6*(1.0 - float(r['distance'])) + 0.4*r_ts - penalty
            rows.sort(key=lambda x: x['score'], reverse=True)
            return rows[:top_k]

        if title_filter:
            sem_sql = """
                SELECT id, section_number, title, content, chunk_text, distance
                FROM (
                SELECT
                    s.id,
                    s.section_number,
                    s.title,
                    s.content,
                    sc.chunk_text,
                    (sc.embedding <=> $1::vector) AS distance,
                    ROW_NUMBER() OVER (PARTITION BY s.id ORDER BY sc.embedding <=> $1::vector) AS rn
                FROM section_chunks sc
                JOIN sections s ON sc.section_id = s.id
                WHERE s.title_number = ANY($2::int[])
                ) t
                WHERE rn = 1
                ORDER BY distance ASC
                LIMIT $3;
            """
            semantic_results = await self.db_pool.fetch(sem_sql, embedding_str, title_filter, top_k)
        else:
            sem_sql = """
                SELECT id, section_number, title, content, chunk_text, distance
                FROM (
                SELECT
                    s.id,
                    s.section_number,
                    s.title,
                    s.content,
                    sc.chunk_text,
                    (sc.embedding <=> $1::vector) AS distance,
                    ROW_NUMBER() OVER (PARTITION BY s.id ORDER BY sc.embedding <=> $1::vector) AS rn
                FROM section_chunks sc
                JOIN sections s ON sc.section_id = s.id
                ) t
                WHERE rn = 1
                ORDER BY distance ASC
                LIMIT $2;
            """
            semantic_results = await self.db_pool.fetch(sem_sql, embedding_str, top_k)

        return [dict(r) for r in semantic_results]

    async def generate_answer(self, query: str, context_sections: List[Dict[str, Any]]) -> str:
        if not context_sections:
            return "I couldn’t find any sections that directly answer your question."

        # Tiny must-term guard for certain intents
        ql = query.lower()
        must_terms = []
        if "permit" in ql or "permits" in ql:
            must_terms = ["permit", "§ 28-105", "28-105"]

        if must_terms:
            filtered = []
            for s in context_sections:
                hay = (s.get('title','') + ' ' + s.get('chunk_text','') + ' ' + s.get('content','')).lower()
                if any(t in hay for t in must_terms):
                    filtered.append(s)
            if not filtered:
                return "I didn’t find permit provisions (e.g., § 28-105) in the retrieved sections. Try asking about permits with more detail (e.g., 'When is a work permit required?')."
            context_sections = filtered

        relevant_sections = context_sections[:5]

        # Build compact context from chunks only
        sources_block = []
        for i, s in enumerate(relevant_sections, 1):
            sec = s.get('section_number', 'N/A')
            title = s.get('title', 'N/A')
            chunk = (s.get('chunk_text') or s.get('content',''))[:1200]
            sources_block.append(f"Source {i}: § {sec} — {title}\n\"\"\"\n{chunk}\n\"\"\"")
        context = "\n\n".join(sources_block)

        prompt = f"""
                    You are answering strictly from the NYC Administrative Code sources below.

                    Rules:
                    - Answer with short bullet points.
                    - After every sentence, include the section number in parentheses, e.g., (§ 28-105.1).
                    - If the claim is not explicitly supported by a quoted source, OMIT it.
                    - Do not add disclaimers or outside knowledge.

                    Question:
                    \"\"\"{query}\"\"\"

                    Sources:
                    {context}

                    Now answer. If the sources do not answer, say: "I can’t find that in these sections."
                    """

        if not self.generative_model:
            # Simple non-LLM fallback: list the top snippets only
            bullets = []
            for s in relevant_sections[:3]:
                sec = s.get('section_number', 'N/A')
                title = s.get('title', 'Untitled')
                txt = (s.get('chunk_text') or s.get('content',''))[:300]
                bullets.append(f"• {title} (§ {sec}): {txt}…")
            return "\n".join(bullets)

        try:
            resp = self.generative_model.generate_content(prompt)
            return resp.text.strip()
        except Exception as e:
            logger.error(f"Generative model error: {e}")
            # Same minimal fallback
            bullets = []
            for s in relevant_sections[:3]:
                sec = s.get('section_number', 'N/A')
                title = s.get('title', 'Untitled')
                txt = (s.get('chunk_text') or s.get('content',''))[:300]
                bullets.append(f"• {title} (§ {sec}): {txt}…")
            return "\n".join(bullets)

    @staticmethod
    def _detect_title_filter(query: str) -> list[int] | None:
        q = query.lower()
        # Route examples; expand later if you like
        if "permit" in q or "permits" in q:
            return [28]  # Construction Codes
        if any(t in q for t in ["fire", "sprinkler", "alarm", "egress", "fdny", "evacuation"]):
            return [28, 29]  # Building + Fire Code
        if any(t in q for t in ["noise", "decibel", "sound level"]):
            return [24]  # Noise Code
        return None

    def _provide_helpful_guidance(self, query: str) -> str:
        """Provide helpful guidance when no relevant sections are found."""
        query_lower = query.lower()
        
        if 'noise' in query_lower or 'sound' in query_lower or 'quiet' in query_lower:
            return (
                "I couldn't find the NYC Noise Control Code sections in my database. "
                "The noise ordinance regulations are located in Title 24, Chapter 2 of the NYC Administrative Code "
                "(sections § 24-201 through § 24-273). These sections cover:\n\n"
                "• Prohibited noise levels and decibel limits\n"
                "• Construction noise restrictions (generally prohibited before 7 AM and after 6 PM)\n"
                "• Quiet hours for residential areas\n"
                "• Specific regulations for music, animals, and commercial activities\n"
                "• Penalties for violations\n\n"
                "Unfortunately, these specific sections haven't been indexed in my database yet. "
                "For accurate information, please consult the official NYC Administrative Code Title 24, Chapter 2."
            )
        
        elif 'fire' in query_lower or 'safety' in query_lower:
            return (
                "I couldn't find relevant fire safety sections in my database. "
                "Fire safety requirements are primarily located in:\n\n"
                "• Title 28 (Building Code) - Chapters 7 and 9\n"
                "• Title 29 (Fire Code)\n"
                "• Title 27 (older Housing Maintenance Code)\n\n"
                "These sections may not be fully indexed in my database. "
                "For accurate fire safety requirements, please consult the official NYC Building Code and Fire Code."
            )
        
        elif 'building' in query_lower or 'construction' in query_lower or 'permit' in query_lower:
            return (
                "For building and construction regulations, please refer to:\n\n"
                "• Title 28 (New Building Code) - Current building requirements\n"
                "• Title 27 (Former Building Code) - Some provisions still in effect\n\n"
                "The sections I found may not be the most relevant. "
                "For accurate information, please consult the official NYC Building Code."
            )
        
        else:
            return (
                f"I couldn't find sections directly relevant to '{query}' in my current database. "
                "This might be because:\n\n"
                "1. The relevant sections haven't been indexed yet\n"
                "2. The query uses different terminology than the code\n"
                "3. This topic might be in a specialized section of the code\n\n"
                "Please try rephrasing your question or consult the official NYC Administrative Code."
            )

    def _generate_fallback_answer(self, query: str, context_sections: List[Dict[str, Any]]) -> str:
        """
        Generates a basic answer when the generative model is not available.
        """
        # Check if the sections are actually relevant
        query_lower = query.lower()
        relevant_sections = []
        
        for section in context_sections:
            section_text = (section.get('title', '') + ' ' + 
                          section.get('content', '') + ' ' + 
                          section.get('chunk_text', '')).lower()
            
            # Check if section is actually relevant to the query
            query_words = query_lower.split()
            relevance_score = sum(1 for word in query_words if word in section_text)
            
            if relevance_score >= 2:  # At least 2 query words found
                relevant_sections.append(section)
        
        if not relevant_sections:
            return (f"I couldn't find any sections in the NYC Administrative Code that directly address '{query}'. "
                   f"This might be because:\n\n"
                   f"1. The relevant sections haven't been added to the database yet\n"
                   f"2. The query uses different terminology than the code\n"
                   f"3. This topic might be covered in NYC Fire Code or Building Code, which may not be fully indexed\n\n"
                   f"For fire safety requirements for student housing, you may want to consult:\n"
                   f"- NYC Building Code (Title 28), Chapter 3 (Occupancy Classification) and Chapter 9 (Fire Protection)\n"
                   f"- NYC Fire Code (Title 3)\n"
                   f"- NYC Housing Maintenance Code (Title 27, Chapter 2)")
        
        answer = "Based on the NYC Administrative Code:\n\n"
        
        for i, section in enumerate(relevant_sections[:3], 1):
            section_num = section.get('section_number', 'Unknown')
            title = section.get('title', 'Untitled')
            content = section.get('chunk_text') or section.get('content', '')
            
            # Extract the most relevant snippet
            snippet = content[:300] + "..." if len(content) > 300 else content
            
            answer += f"{i}. **{section_num}** - {title}\n"
            answer += f"   {snippet}\n\n"
        
        answer += f"\nThese sections appear to be most relevant to your question about: {query}"
        
        return answer
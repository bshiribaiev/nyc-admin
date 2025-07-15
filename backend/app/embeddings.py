import logging
from typing import List, Union
import numpy as np

from config.settings import EMBEDDING_CONFIG
from app.database import DatabaseManager
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager
        self.chunk_size = EMBEDDING_CONFIG.get("chunk_size", 1000)
        self.chunk_overlap = EMBEDDING_CONFIG.get("chunk_overlap", 200)
        
        # Try to load embedding model
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        try:
            model_name = "all-MiniLM-L6-v2"  
            logger.info(f"Loading Sentence Transformer model: {model_name}")
            return SentenceTransformer(model_name)
        except ImportError:
            logger.warning("Sentence Transformers not available")
        
    def chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            # Try to break at sentence boundary
            if end < text_len:
                for sep in ['. ', '.\n', '? ', '! ']:
                    sep_pos = text.rfind(sep, start, end)
                    if sep_pos != -1:
                        end = sep_pos + len(sep)
                        break
                        
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            start = end - self.chunk_overlap if end < text_len else text_len
            
        return chunks
        
    def generate_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
            
        # Sentence Transformers
        if hasattr(self.model, 'encode'):
            return self.model.encode(texts, convert_to_numpy=True)
            
        # Cohere
        elif hasattr(self.model, 'embed'):
            response = self.model.embed(
                texts=texts,
                model="embed-english-v3.0",
                input_type="search_document"
            )
            return np.array(response.embeddings)
            
        # TF-IDF fallback
        else:
            # For TF-IDF, we need to fit on all documents first
            if not hasattr(self.model, 'vocabulary_'):
                # Need to fit the model first
                logger.warning("TF-IDF model not fitted, using simple hash")
                # Return random embeddings as last resort
                return np.random.rand(len(texts), 384)
            return self.model.transform(texts).toarray()
            
    def get_embedding_dimension(self) -> int:
        if hasattr(self.model, 'get_sentence_embedding_dimension'):
            return self.model.get_sentence_embedding_dimension()
        elif hasattr(self.model, 'embed'):
            return 1024  # Cohere dimension
        else:
            return 384  # TF-IDF or default
            
    def process_section(self, section_id: int):
        if not self.db:
            raise RuntimeError("Database manager not configured")
            
        with self.db.get_cursor() as cursor:
            # Get section content
            cursor.execute("""
                SELECT content FROM sections WHERE id = %s
            """, (section_id,))
            
            result = cursor.fetchone()
            if not result:
                logger.warning(f"Section {section_id} not found")
                return
                
            content = result['content']
            
            # Delete existing chunks
            cursor.execute("""
                DELETE FROM section_chunks WHERE section_id = %s
            """, (section_id,))
            
            # Create chunks
            chunks = self.chunk_text(content)
            
            # Generate embeddings for all chunks at once (more efficient)
            if chunks:
                embeddings = self.generate_embeddings(chunks)
                
                # Save chunks with embeddings
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    cursor.execute("""
                        INSERT INTO section_chunks 
                        (section_id, chunk_index, chunk_text, embedding)
                        VALUES (%s, %s, %s, %s)
                    """, (section_id, i, chunk, embedding.tolist()))
                    
            logger.info(f"Processed {len(chunks)} chunks for section {section_id}")

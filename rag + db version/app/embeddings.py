import logging
import gc
import time
from typing import List, Dict
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
            model = SentenceTransformer(model_name, device='cpu')
            model.max_seq_length = 512
            return model
        except ImportError:
            logger.warning("Sentence Transformers not available")
            return None
                
    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into chunks efficiently.
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            # If we're not at the end, try to find a natural break
            if end < text_len:
                # Look for sentence boundaries
                for sep in ['. ', '.\n', '\n\n', '? ', '! ', '\n']:
                    pos = text.rfind(sep, start, end)
                    if pos > start:
                        end = pos + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            # Move to next chunk with overlap
            if end < text_len:
                start = max(start + 1, end - self.chunk_overlap)
            else:
                start = end
            
        return chunks
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts efficiently."""
        if not self.model or not texts:
            return np.array([]).reshape(0, 384)
        
        # Process all texts at once - this is much faster than small batches
        # The model will handle internal batching efficiently
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,  # Internal batch size
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        return embeddings
            
    def process_sections_batch(self, section_ids: List[int]) -> Dict[str, int]:
        """
        Process multiple sections at once for better efficiency.
        Returns statistics about processing.
        """
        if not self.db:
            raise RuntimeError("Database manager not configured")
        
        stats = {'processed': 0, 'chunks_created': 0, 'errors': 0}
        
        with self.db.get_cursor() as cursor:
            # Fetch all sections at once
            cursor.execute("""
                SELECT id, content 
                FROM sections 
                WHERE id = ANY(%s)
                ORDER BY id
            """, (section_ids,))
            
            sections = cursor.fetchall()
            
            # Collect all chunks and their metadata
            all_chunks_data = []
            
            for section in sections:
                section_id = section['id']
                content = section['content']
                
                if not content:
                    logger.warning(f"Section {section_id} has no content")
                    continue
                
                # Create chunks for this section
                chunks = self.chunk_text(content)
                
                # Add metadata for each chunk
                for i, chunk in enumerate(chunks):
                    all_chunks_data.append({
                        'section_id': section_id,
                        'chunk_index': i,
                        'chunk_text': chunk[:5000]  # Limit chunk size
                    })
                
                stats['chunks_created'] += len(chunks)
            
            if not all_chunks_data:
                return stats
            
            # Generate embeddings for ALL chunks at once
            logger.info(f"Generating embeddings for {len(all_chunks_data)} chunks from {len(sections)} sections...")
            start_time = time.time()
            
            chunk_texts = [chunk['chunk_text'] for chunk in all_chunks_data]
            embeddings = self.generate_embeddings_batch(chunk_texts)
            
            embedding_time = time.time() - start_time
            logger.info(f"Generated {len(embeddings)} embeddings in {embedding_time:.2f} seconds")
            
            # Delete existing chunks for these sections
            cursor.execute("""
                DELETE FROM section_chunks 
                WHERE section_id = ANY(%s)
            """, (section_ids,))
            
            from psycopg2.extras import execute_values
            
            insert_data = [
                (
                    chunk['section_id'],
                    chunk['chunk_index'],
                    chunk['chunk_text'],
                    embedding.tolist()
                )
                for chunk, embedding in zip(all_chunks_data, embeddings)
            ]
            
            execute_values(
                cursor,
                """
                INSERT INTO section_chunks (section_id, chunk_index, chunk_text, embedding)
                VALUES %s
                """,
                insert_data,
                template="(%s, %s, %s, %s)"
            )
            
            stats['processed'] = len(sections)
            
        return stats
            
    def process_all_sections(self, batch_size: int = 50):
        """
        Process all sections that don't have embeddings yet.
        """
        if not self.db:
            raise RuntimeError("Database manager not configured")
        
        total_stats = {'processed': 0, 'chunks_created': 0, 'errors': 0}
        start_time = time.time()
        
        while True:
            with self.db.get_cursor() as cursor:
                # Find sections without chunks
                cursor.execute("""
                    SELECT s.id
                    FROM sections s
                    LEFT JOIN section_chunks sc ON s.id = sc.section_id
                    WHERE sc.id IS NULL
                    ORDER BY s.id
                    LIMIT %s;
                """, (batch_size,))
                
                section_ids = [row['id'] for row in cursor.fetchall()]
            
            if not section_ids:
                logger.info("All sections have been processed.")
                break
            
            logger.info(f"Processing batch of {len(section_ids)} sections: {section_ids[0]}-{section_ids[-1]}")
            
            try:
                # Process the entire batch at once
                batch_stats = self.process_sections_batch(section_ids)
                
                total_stats['processed'] += batch_stats['processed']
                total_stats['chunks_created'] += batch_stats['chunks_created']
                total_stats['errors'] += batch_stats['errors']
                
                # Log progress
                elapsed = time.time() - start_time
                rate = total_stats['processed'] / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {total_stats['processed']} sections, "
                          f"{total_stats['chunks_created']} chunks, "
                          f"{rate:.1f} sections/sec")
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                total_stats['errors'] += len(section_ids)
                
                # Try to process individually as fallback
                for section_id in section_ids:
                    try:
                        self.process_section_single(section_id)
                        total_stats['processed'] += 1
                    except Exception as e2:
                        logger.error(f"Failed to process section {section_id}: {e2}")
                        total_stats['errors'] += 1
            
            # Garbage collection after each batch
            gc.collect()
        
        # Final statistics
        elapsed = time.time() - start_time
        logger.info(f"Completed embedding generation:")
        logger.info(f"  - Sections processed: {total_stats['processed']}")
        logger.info(f"  - Chunks created: {total_stats['chunks_created']}")
        logger.info(f"  - Errors: {total_stats['errors']}")
        logger.info(f"  - Total time: {elapsed:.1f} seconds")
        logger.info(f"  - Average rate: {total_stats['processed']/elapsed:.1f} sections/sec")
    
    def process_section_single(self, section_id: int):
        """Fallback method to process a single section."""
        if not self.db:
            raise RuntimeError("Database manager not configured")
            
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT content FROM sections WHERE id = %s", (section_id,))
            result = cursor.fetchone()
            if not result:
                logger.warning(f"Section {section_id} not found")
                return
                
            content = result['content']
            
            cursor.execute("DELETE FROM section_chunks WHERE section_id = %s", (section_id,))
            
            chunks = self.chunk_text(content)
            
            if chunks:
                embeddings = self.generate_embeddings_batch(chunks)
                
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    cursor.execute("""
                        INSERT INTO section_chunks 
                        (section_id, chunk_index, chunk_text, embedding)
                        VALUES (%s, %s, %s, %s)
                    """, (section_id, i, chunk[:5000], embedding.tolist()))
                    
            logger.info(f"Processed {len(chunks)} chunks for section {section_id}")
    
    def check_progress(self):
        """Check how many sections have been processed."""
        if not self.db:
            raise RuntimeError("Database manager not configured")
            
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT s.id) as total_sections,
                    COUNT(DISTINCT sc.section_id) as processed_sections,
                    COUNT(sc.id) as total_chunks
                FROM sections s
                LEFT JOIN section_chunks sc ON s.id = sc.section_id
            """)
            
            result = cursor.fetchone()
            
            logger.info(f"Embedding Generation Progress:")
            logger.info(f"  - Total sections: {result['total_sections']}")
            logger.info(f"  - Processed sections: {result['processed_sections']}")
            logger.info(f"  - Remaining: {result['total_sections'] - result['processed_sections']}")
            logger.info(f"  - Total chunks created: {result['total_chunks']}")
            
            return result
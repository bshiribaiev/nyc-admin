import asyncio
import logging
import sys
from typing import Optional

import click
import uvicorn

from config.settings import API_CONFIG, LOG_LEVEL
from app.database import DatabaseManager, AsyncDatabasePool
from app.scraper import NYCAdminCodeScraper
from app.embeddings import EmbeddingGenerator

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """NYC Admin Code Q&A System CLI"""
    pass


@cli.command()
def init_db():
    """Initialize database schema"""
    logger.info("Initializing database...")
    
    db = DatabaseManager()
    db.connect()
    db.initialize_schema()
    db.close()
    
    logger.info("Database initialized successfully")


@cli.command()
@click.option('--full', is_flag=True, help='Run full scrape')
def scrape(full: bool):
    """Scrape NYC Admin Code website"""
    logger.info("Starting scraper...")
    
    db = DatabaseManager()
    db.connect()
    
    try:
        scraper = NYCAdminCodeScraper(db)
        
        if full:
            scraper.run_full_scrape()
        else:
            # For incremental scraping (to be implemented)
            logger.info("Incremental scraping not yet implemented")
            
    finally:
        db.close()


@cli.command()
@click.option('--all', is_flag=True, help='Process all sections')
@click.option('--section-id', type=int, help='Process specific section')
def embeddings(all: bool, section_id: Optional[int]):
    """Generate embeddings for sections"""
    logger.info("Generating embeddings...")
    
    db = DatabaseManager()
    db.connect()
    
    try:
        generator = EmbeddingGenerator(db)
        
        if all:
            generator.process_all_sections()
        elif section_id:
            generator.process_section(section_id)
        else:
            logger.error("Please specify --all or --section-id")
            
    finally:
        db.close()


@cli.command()
@click.option('--host', default=API_CONFIG['host'], help='Host to bind')
@click.option('--port', default=API_CONFIG['port'], help='Port to bind')
@click.option('--workers', default=API_CONFIG['workers'], help='Number of workers')
def serve(host: str, port: int, workers: int):
    """Start the API server"""
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        "app.api:app",
        host=host,
        port=port,
        workers=workers,
        reload=False
    )


@cli.command()
@click.argument('question')
async def ask(question: str):
    """Ask a question from the command line"""
    db_pool = AsyncDatabasePool()
    await db_pool.initialize()
    
    try:
        from app.qa_system import QASystem
        qa = QASystem(db_pool)
        
        # Search for relevant sections
        sections = await qa.search_similar_sections(question, top_k=5)
        
        if not sections:
            print("No relevant sections found.")
            return
            
        # Generate answer
        answer = await qa.generate_answer(question, sections)
        
        print("\nAnswer:")
        print("=" * 50)
        print(answer)
        print("\nSources:")
        print("-" * 50)
        for section in sections[:3]:
            print(f"- {section['section_number']}: {section['title']}")
            
    finally:
        await db_pool.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ask":
        # Handle async ask command
        asyncio.run(ask(sys.argv[2]))
    else:
        cli()
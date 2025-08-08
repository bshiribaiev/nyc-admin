import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).parent.parent

# Database settings
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "nyc_admin"),
    "user": os.getenv("DB_USER", "nyc_user"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}

# Redis settings
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Gemini settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# API settings
API_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", 8000)),
    "workers": int(os.getenv("API_WORKERS", 4)),
}

# Scraping settings
SCRAPE_CONFIG = {
    "base_url": "https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-1",
    "delay": float(os.getenv("SCRAPE_DELAY", 1.0)),
    "batch_size": int(os.getenv("BATCH_SIZE", 10)),
}

# Embedding settings
EMBEDDING_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
}

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
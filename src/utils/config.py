"""
Configuration Management for QA Service

This file handles all the settings for our Q&A service.
Think of it as the control panel for the entire system.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration class that holds all settings
    
    This is like a settings menu for our application.
    All the important numbers and options are stored here.
    """
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    PDFS_DIR = PROJECT_ROOT / "pdfs"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Database settings
    db_filename = os.getenv("DATABASE_PATH", "data/qa_service.db")
    if db_filename.startswith("data/"):
        DATABASE_PATH = PROJECT_ROOT / db_filename
    else:
        DATABASE_PATH = DATA_DIR / db_filename
    SOURCES_JSON = DATA_DIR / "sources.json"
    BM25_PICKLE = DATA_DIR / "bm25_index.pkl"
    
    # AI Model settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Search settings (how we find answers)
    VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))      # How much we trust meaning similarity
    KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "0.3"))    # How much we trust exact word matches
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "20"))         # How many chunks to consider
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))  # When to say "I don't know"
    
    # Text processing settings
    MIN_CHUNK_SIZE = 50      # Smallest chunk size (words)
    MAX_CHUNK_SIZE = 400     # Largest chunk size (words)
    CHUNK_OVERLAP = 50       # How much chunks overlap (words)
    
    # API settings
    FLASK_HOST = os.getenv("FLASK_HOST", "localhost")
    FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    
    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Random seed for reproducible results
    RANDOM_SEED = 42
    
    @classmethod
    def ensure_dirs_exist(cls):
        """
        Create necessary directories if they don't exist
        
        This is like making sure all the filing cabinets 
        are ready before we start organizing documents.
        """
        for directory in [cls.DATA_DIR, cls.PDFS_DIR, cls.LOGS_DIR]:
            directory.mkdir(exist_ok=True)
    
    @classmethod
    def get_summary(cls):
        """
        Get a summary of current configuration
        
        This is useful for debugging - shows all the current settings
        """
        return {
            "database_path": str(cls.DATABASE_PATH),
            "embedding_model": cls.EMBEDDING_MODEL,
            "vector_weight": cls.VECTOR_WEIGHT,
            "keyword_weight": cls.KEYWORD_WEIGHT,
            "top_k_results": cls.TOP_K_RESULTS,
            "confidence_threshold": cls.CONFIDENCE_THRESHOLD,
            "chunk_size_range": f"{cls.MIN_CHUNK_SIZE}-{cls.MAX_CHUNK_SIZE} words"
        }

# Create directories when this module is imported
Config.ensure_dirs_exist()
"""
Application configuration
"""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    app_name: str = "LLM-Powered Intelligent Query-Retrieval System"
    version: str = "1.0.0"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Security
    bearer_token: str = "8915ddf1d1760f2b6a3b027c6fa7b16d2d87a042c41452f49a1d43b3cfa6245b"
    
    # Document Processing
    max_file_size: int = 500 * 1024 * 1024  # 500MB
    temp_dir: Optional[str] = None
    save_parsed_text: bool = True  # Save parsed text to files for validation
    parsed_text_dir: str = "parsed_documents"
    
    # PDF Blob Storage
    save_pdf_blobs: bool = True  # Save PDF files in blob format for caching
    pdf_blob_dir: str = "blob_pdf"  # Directory to save PDF blobs
    
    # Request Timeouts
    download_timeout: int = 60  # seconds
    processing_timeout: int = 300  # seconds
    
    # Embedding Configuration
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"  # Use GPU if available
    chunk_size: int = 512  # Token-based chunk size
    chunk_overlap: int = 50  # Overlap between chunks
    max_tokens_per_chunk: int = 8192  # BGE-M3 max supported tokens
    
    # LLM Configuration
    llm_model: str = "gemini-2.0-flash"  # Gemini model for answer generation
    gemini_api_key: str = ""  # Set via environment variable GEMINI_API_KEY
    llm_max_tokens: int = 8192  # Maximum tokens for LLM response
    llm_temperature: float = 0.1  # Low temperature for factual responses
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
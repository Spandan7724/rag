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
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    temp_dir: Optional[str] = None
    save_parsed_text: bool = True  # Save parsed text to files for validation
    parsed_text_dir: str = "parsed_documents"  # Directory to save parsed text
    
    # Enhanced Document Storage
    document_storage_root: str = "./documents"  # Root directory for persistent storage
    enable_ocr_fallback: bool = True  # Enable OCR for scanned pages
    text_density_threshold: float = 50.0  # Characters per page threshold for OCR trigger
    store_span_map: bool = False  # Store span mapping data (for debugging/development)
    
    # PyMuPDF4LLM Performance
    parallel_processing_workers: int = 0  # Number of workers for parallel page processing (0 = auto-detect)
    
    # Debugging and Logging
    # Note: Clause segmentation features removed
    
    # OCR Configuration
    tesseract_cmd: Optional[str] = None  # Path to tesseract executable (auto-detect if None)
    ocr_languages: str = "eng"  # Tesseract languages to use
    ocr_dpi: int = 300  # DPI for OCR processing
    
    # Request Timeouts
    download_timeout: int = 60  # seconds
    processing_timeout: int = 300  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
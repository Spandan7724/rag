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
    
    # Request Timeouts
    download_timeout: int = 60  # seconds
    processing_timeout: int = 300  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
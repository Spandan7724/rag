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
    
    # File Upload Configuration
    upload_dir: str = "uploads"  # Directory to store uploaded files
    upload_retention_hours: int = 24  # How long to keep uploaded files
    max_upload_size: int = 500 * 1024 * 1024  # 500MB max upload size
    allowed_upload_types: list = ["application/pdf"]  # Only allow PDF files
    upload_cleanup_interval: int = 3600  # Cleanup interval in seconds (1 hour)
    
    # Question Logging Configuration
    enable_question_logging: bool = True  # Enable logging of questions and responses
    question_log_dir: str = "question_logs"  # Directory to store question logs
    log_full_responses: bool = True  # Log complete responses with metadata
    log_retention_days: int = 30  # How long to keep question logs
    
    # Request Timeouts
    download_timeout: int = 60  # seconds
    processing_timeout: int = 3000  # seconds
    
    # Embedding Configuration
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"  # Use GPU if available
    
    # Hybrid PDF Processing Configuration
    enable_hybrid_processing: bool = True  # Enable smart page-level processing
    text_threshold: int = 100  # Minimum chars to avoid OCR processing
    table_confidence: float = 0.7  # Table detection confidence threshold
    use_gpu_ocr: bool = True  # GPU OCR acceleration (not applicable for Tesseract)
    parallel_pages: bool = True  # Process pages in parallel when possible
    ocr_provider: str = "rapidocr"  # Options: "tesseract", "paddleocr", "rapidocr"
    
    # Table Extraction Configuration (Simplified)
    table_extraction_method: str = "pdfplumber"  # Primary method for table extraction
    
    # Vector Storage Configuration
    enable_vector_storage: bool = True  # Enable persistent vector storage to disk
    vector_storage_mode: str = "persistent"  # Options: "persistent", "temporary", "memory_only"
    vector_store_dir: str = "vector_store"  # Directory for persistent vector storage
    auto_cleanup_vectors: bool = False  # Automatically cleanup old vector indices
    vector_retention_days: int = 7  # Days to keep vector indices when auto_cleanup enabled
    
    # Text Chunking Configuration
    chunk_size: int = 450  # Smaller chunks to isolate short clauses like grace period (<600 chars)
    chunk_overlap: int = 100   # Increased overlap for better continuity and accuracy
    k_retrieve: int = 20  # Number of chunks to retrieve from vector store (configurable via .env as K_RETRIEVE)
    max_tokens_per_chunk: int = 8192  # BGE-M3 max supported tokens
    
    # Advanced Retrieval Configuration
    adaptive_k: bool = True  # Dynamically adjust k based on query complexity
    min_k_retrieve: int = 30  # Minimum chunks to retrieve
    max_k_retrieve: int = 60  # Maximum chunks to retrieve for complex queries
    similarity_threshold: float = 0.3  # Minimum similarity score to include chunks
    top_k_reranked: int = 8  # Optimized for larger token budget - fewer chunks, more context per chunk
    enable_boost_rules: bool = False  # Enable insurance-specific boost rules for better accuracy
    
    # Performance Optimization Configuration
    enable_result_caching: bool = False  # Cache query results for faster responses
    cache_ttl_seconds: int = 3600  # Cache time-to-live (1 hour)
    enable_embedding_cache: bool = False  # Cache embeddings for repeated chunks
    enable_reranker_cache: bool = False  # Cache reranker scores
    max_concurrent_requests: int = 4  # Limit concurrent API calls
    max_concurrent_questions: int = 2  # Max questions to process in parallel
    api_timeout_seconds: int = 10  # Faster timeout for API calls
    early_stopping: bool = True  # Stop processing if confident answer found  
    confidence_threshold: float = 0.6  # Relaxed threshold to allow comprehensive multi-clause searches
    
    # Model Optimization

    batch_embedding_size: int = 32  # Batch size for embedding generation
    gpu_memory_limit: Optional[str] = None  # Auto-detect GPU memory limit (remove artificial constraints)

    # LLM Configuration
    llm_provider: str = "copilot"  # Options: "gemini", "copilot"
    llm_model: str = "gpt-4.1-2025-04-14"  # Model for answer generation #gpt-4.1-2025-04-14 #gemini-1.5-flash-latest
    gemini_api_key: str = ""  # Set via environment variable GEMINI_API_KEY
    copilot_access_token: str = ""  # Set via environment variable COPILOT_ACCESS_TOKEN
    llm_max_tokens: int = 768  # Sufficient tokens for complete clause extraction and comprehensive answers
    llm_temperature: float = 0.3  # Balanced temperature for detailed but factual responses
    
    # Competition-specific optimizations
    competition_mode: bool = False  # Enable competition optimizations
    fast_mode: bool = False  # Prioritize accuracy over speed
    max_context_tokens: int = 24000  # Increased context for better accuracy
    
    class Config:   
        env_file = ".env"
        case_sensitive = False

settings = Settings()
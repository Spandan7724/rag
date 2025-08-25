"""
Application configuration
"""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    app_name: str = "Hybrid Document RAG System"
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
    # File upload types - set to None to allow all supported types, or specify a list to restrict
    allowed_upload_types: Optional[list] = None  # None = allow all supported types, or specify list to restrict
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
    embedding_provider: str = "bge-m3"  # BGE-M3 embedding model
    embedding_model: str = "BAAI/bge-m3"  # For BGE-M3 provider
    embedding_device: str = "cuda"  # Use GPU if available for local models
    
    # Hybrid PDF Processing Configuration
    enable_hybrid_processing: bool = True  # Enable smart page-level processing
    text_threshold: int = 100  # Minimum chars to avoid OCR processing
    table_confidence: float = 0.7  # Table detection confidence threshold
    use_gpu_ocr: bool = True  # GPU OCR acceleration (not applicable for Tesseract)
    parallel_pages: bool = True  # Process pages in parallel when possible
    ocr_provider: str = "rapidocr"  # Options: "tesseract", "paddleocr", "rapidocr"
    
    
    table_extraction_method: str = "pdfplumber"  # Primary method for table extraction
    
    # Vector Storage Configuration
    enable_vector_storage: bool = True  # Enable persistent vector storage to disk
    vector_storage_mode: str = "persistent"  # Options: "persistent", "temporary", "memory_only"
    vector_store_dir: str = "vector_store"  # Directory for persistent vector storage
    auto_cleanup_vectors: bool = False  # Automatically cleanup old vector indices
    vector_retention_days: int = 7  # Days to keep vector indices when auto_cleanup enabled
    
    # Text Chunking Configuration
    chunk_size: int = 450  # Optimized chunk size for precise information retrieval (<600 chars)
    chunk_overlap: int = 100   # Increased overlap for better continuity and accuracy
    k_retrieve: int = 35  # Optimized for speed/accuracy balance 
    max_tokens_per_chunk: int = 8192  # BGE-M3 max supported tokens
    
    # Advanced Retrieval Configuration
    adaptive_k: bool = False  # Disable adaptive k - use fixed values for simplicity
    min_k_retrieve: int = 20  # Minimum chunks to retrieve
    max_k_retrieve: int = 40  # Reduced for faster processing
    similarity_threshold: float = 1.5  # Maximum L2 distance to include chunks (significantly increased for multilingual matching)
    # For normalized embeddings: L2=0.0 (identical), L2=0.7 (~85% similar), L2=1.0 (~75% similar), L2=1.4 (~50% similar)
    top_k_reranked: int = 7  # Slightly reduced for faster reranking
    enable_boost_rules: bool = False  # Disable boost rules - too complex and brittle
    
    # Multilingual Retrieval Configuration
    multilingual_k_retrieve: int = 50  # Increased k for cross-language semantic matching
    multilingual_similarity_threshold: float = 1.4  # More lenient threshold for multilingual content
    multilingual_chunk_overlap: int = 150  # Increased overlap for better context preservation
    enable_multilingual_enhancement: bool = True  # Enable enhanced multilingual processing
    
    # Performance Optimization Configuration
    debug_mode: bool = False  # Enable verbose debug logging (impacts performance)
    enable_result_caching: bool = False  # Cache query results for faster responses
    cache_ttl_seconds: int = 43200  # Cache time-to-live (12 hours)
    enable_embedding_cache: bool = True  # Cache embeddings for repeated chunks
    enable_reranker_cache: bool = True  # Cache reranker scores
    enable_answer_cache: bool = True  # Cache complete answers for semantic search documents only
    max_concurrent_requests: int = 4  # Limit concurrent API calls
    max_concurrent_questions: int = 2  # Max questions to process in parallel
    api_timeout_seconds: int = 50  # Balanced timeout for comprehensive responses
    early_stopping: bool = True  # Stop processing if confident answer found  
    confidence_threshold: float = 0.6  # Relaxed threshold to allow comprehensive multi-clause searches
    
    # Model Optimization

    batch_embedding_size: int = 128  # Batch size for embedding generation (increased for throughput)
    gpu_memory_limit: Optional[str] = None  # Auto-detect GPU memory limit (remove artificial constraints)

    # LLM Configuration
    llm_provider: str = "copilot"  # LLM provider: "copilot" or "openai"
    llm_model: str = "claude-sonnet-4"  # Model for answer generation
    copilot_access_token: str = ""  # Set via environment variable COPILOT_ACCESS_TOKEN
    openai_api_key: str = ""  # Set via environment variable OPENAI_API_KEY
    llm_max_tokens: int = 2048  # Increased for comprehensive document analysis with metadata
    llm_temperature: float = 0.2  # Lower temperature for more factual, precise responses
    
    # Performance optimizations
    performance_mode: bool = False  # Enable performance optimizations
    fast_mode: bool = False  # Prioritize accuracy over speed
    max_context_tokens: int = 64000  # Increased context for better accuracy
    
    
    class Config:   
        env_file = ".env"
        case_sensitive = False

settings = Settings()
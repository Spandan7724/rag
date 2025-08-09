"""
Application configuration
"""
from pydantic_settings import BaseSettings
from typing import Optional, List

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
    embedding_provider: str = "bge-m3"  # Options: "bge-m3", "gemini"
    embedding_model: str = "BAAI/bge-m3"  # For BGE-M3 provider
    embedding_device: str = "cuda"  # Use GPU if available for local models
    
    # Gemini Embedding Configuration
    gemini_embedding_model: str = "gemini-embedding-001"  # Gemini model name
    gemini_embedding_dimension: int = 768  # Options: 128-3072, recommended: 768, 1536, 3072
    gemini_task_type_document: str = "RETRIEVAL_DOCUMENT"  # Task type for indexing documents
    gemini_task_type_query: str = "RETRIEVAL_QUERY"  # Task type for search queries
    gemini_api_timeout: int = 30  # API request timeout in seconds
    gemini_batch_size: int = 100  # Maximum texts per API call
    gemini_rate_limit_delay: float = 0.1  # Delay between API calls for rate limiting
    
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
    cache_ttl_seconds: int = 3600  # Cache time-to-live (1 hour)
    enable_embedding_cache: bool = True  # Cache embeddings for repeated chunks
    enable_reranker_cache: bool = True  # Cache reranker scores
    enable_answer_cache: bool = True  # Cache complete answers for specific documents
    cacheable_document_ids: List[str] = ["334ef1720708"]  # Only cache answers for News.pdf
    max_concurrent_requests: int = 4  # Limit concurrent API calls
    max_concurrent_questions: int = 2  # Max questions to process in parallel
    api_timeout_seconds: int = 50  # Balanced timeout for comprehensive responses
    early_stopping: bool = True  # Stop processing if confident answer found  
    confidence_threshold: float = 0.6  # Relaxed threshold to allow comprehensive multi-clause searches
    
    # Model Optimization

    batch_embedding_size: int = 128  # Batch size for embedding generation (increased for throughput)
    gpu_memory_limit: Optional[str] = None  # Auto-detect GPU memory limit (remove artificial constraints)

    # LLM Configuration
    llm_provider: str = "copilot"  # Options: "gemini", "copilot"
    llm_model: str = "claude-sonnet-4"  # Model for answer generation #gpt-4.1-2025-04-14 #gemini-1.5-flash-latest
    gemini_api_key: str = ""  # Set via environment variable G  EMINI_API_KEY
    copilot_access_token: str = ""  # Set via environment variable COPILOT_ACCESS_TOKEN
    llm_max_tokens: int = 2048  # Increased for comprehensive insurance analysis with metadata
    llm_temperature: float = 0.2  # Lower temperature for more factual, precise responses
    
    # Competition-specific optimizations
    competition_mode: bool = False  # Enable competition optimizations
    fast_mode: bool = False  # Prioritize accuracy over speed
    max_context_tokens: int = 64000  # Increased context for better accuracy
    
    # Query Transformation Configuration
    enable_query_transformation: bool = True  # Enable multi-query decomposition
    max_sub_queries: int = 4  # Maximum number of sub-queries to generate
    query_transformation_timeout: int = 30  # Timeout for query decomposition in seconds
    min_query_length: int = 20  # Minimum query length to consider for transformation
    transformation_temperature: float = 0.1  # Temperature for query transformation LLM calls
    
    class Config:   
        env_file = ".env"
        case_sensitive = False

settings = Settings()
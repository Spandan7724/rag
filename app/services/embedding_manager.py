"""
BGE-M3 embedding service for RAG systems
"""
import time
import threading
import torch
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from app.utils.debug import debug_print
from app.services.text_chunker import TextChunk
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Container for embedding results"""
    embeddings: np.ndarray
    processing_time: float
    model_info: Dict[str, Any]


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Generate embeddings for a list of texts"""
        pass
    
    @abstractmethod
    async def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass


class BGEEmbeddingProvider(EmbeddingProvider):
    """BGE-M3 embedding provider (existing implementation)"""
    
    def __init__(self):
        """Initialize BGE-M3 embedding provider"""
        self.model = None
        self.device = settings.embedding_device
        self.model_name = settings.embedding_model
        self._model_lock = threading.Lock()  # Thread-safe model loading
        self._model_loaded = False  # Track model loading state
        
        from app.utils.debug import debug_print
        debug_print(f"Initializing embedding manager: {self.model_name}")
        debug_print(f"Target device: {self.device}")
        
        # Check device availability
        if self.device == "cuda" and not torch.cuda.is_available():
            debug_print("CUDA not available, falling back to CPU")
            self.device = "cpu"
        elif self.device == "cuda":
            debug_print(f"CUDA available: {torch.cuda.get_device_name()}")
            debug_print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def _load_model(self):
        """Thread-safe lazy load the BGE-M3 model"""
        # Quick check without lock (double-checked locking pattern)
        if self._model_loaded and self.model is not None:
            return
        
        with self._model_lock:
            # Double-check inside the lock
            if self._model_loaded and self.model is not None:
                return
            
            try:
                from app.utils.debug import debug_print, info_print
                debug_print(f"Loading BGE-M3 model (thread-safe): {self.model_name}")
                start_time = time.time()
                
                from FlagEmbedding import BGEM3FlagModel
                
                # Use to_empty() to avoid meta tensor issues in multi-threading
                self.model = BGEM3FlagModel(
                    self.model_name,
                    use_fp16=True,  # Use half precision for faster inference
                    device=self.device
                )
                
                load_time = time.time() - start_time
                info_print(f"BGE-M3 model loaded successfully in {load_time:.2f}s")
                
                # Warm up the model
                debug_print("Warming up model...")
                warmup_start = time.time()
                self.model.encode(["This is a warmup text for the embedding model."])
                warmup_time = time.time() - warmup_start
                debug_print(f"Model warmup completed in {warmup_time:.2f}s")
                
                # Mark as loaded
                self._model_loaded = True
                info_print("BGE-M3 model ready for parallel processing!")
                
            except ImportError:
                raise ImportError(
                    "FlagEmbedding not installed. Install with: pip install FlagEmbedding"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load BGE-M3 model: {str(e)}")
    
    def ensure_model_ready(self):
        """Ensure the model is loaded and ready for use"""
        self._load_model()
        return self._model_loaded
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded without triggering loading"""
        return self._model_loaded and self.model is not None
    
    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if not texts:
            return EmbeddingResult(
                embeddings=np.array([]),
                processing_time=0.0,
                model_info=self.get_model_info()
            )
        
        # Run in thread pool since BGE-M3 is CPU/GPU bound
        def _generate_embeddings():
            from app.utils.debug import debug_print
            self._load_model()
            
            debug_print(f"Generating embeddings for {len(texts)} texts...")
            start_time = time.time()
            
            try:
                # Use BGE-M3's encode method
                result = self.model.encode(
                    texts,
                    batch_size=settings.batch_embedding_size,  # Use configured batch size for efficiency
                    max_length=settings.max_tokens_per_chunk,  # BGE-M3 supports up to 8192 tokens
                    return_dense=True,  # We want dense embeddings for vector search
                    return_sparse=False,  # Don't need sparse embeddings for this use case
                    return_colbert_vecs=False  # Don't need ColBERT vectors
                )
                
                # Extract dense embeddings from result dict
                if isinstance(result, dict):
                    embeddings = result['dense_vecs']
                else:
                    embeddings = result
                
                # Convert to numpy array if needed
                if hasattr(embeddings, 'numpy'):
                    embeddings = embeddings.numpy()
                elif torch.is_tensor(embeddings):
                    embeddings = embeddings.cpu().numpy()
                
                # Ensure 2D array
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                
                # Normalize BGE-M3 embeddings for consistent similarity calculations
                embeddings = self._normalize_embeddings(embeddings)
                
                processing_time = time.time() - start_time
                debug_print(f"Generated embeddings in {processing_time:.2f}s")
                debug_print(f"Embedding shape: {embeddings.shape}")
                debug_print(f"Speed: {len(texts) / processing_time:.1f} texts/second")
                
                return EmbeddingResult(
                    embeddings=embeddings,
                    processing_time=processing_time,
                    model_info=self.get_model_info()
                )
                
            except Exception as e:
                raise RuntimeError(f"Embedding generation failed: {str(e)}")
        
        # Execute directly - BGE-M3 is GPU-optimized
        return _generate_embeddings()
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit vectors for consistent similarity calculations"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-12)
        return embeddings / norms
    
    def encode_chunks(self, chunks: List[TextChunk]) -> EmbeddingResult:
        """
        Generate embeddings for text chunks
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        start_time = time.time()
        
        # Extract texts from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.encode_texts(texts)
        
        processing_time = time.time() - start_time
        
        # Collect model info
        model_info = {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": embeddings.shape[1] if embeddings.size > 0 else 0,
            "num_chunks": len(chunks),
            "total_tokens": sum(chunk.token_count for chunk in chunks),
            "processing_speed": len(chunks) / processing_time if processing_time > 0 else 0
        }
        
        debug_print("Chunk embedding completed:")
        debug_print(f"  - Chunks processed: {len(chunks)}")
        debug_print(f"  - Embedding dimension: {model_info['embedding_dimension']}")
        debug_print(f"  - Total tokens: {model_info['total_tokens']:,}")
        debug_print(f"  - Processing time: {processing_time:.2f}s")
        debug_print(f"  - Speed: {model_info['processing_speed']:.1f} chunks/second")
        
        return EmbeddingResult(
            embeddings=embeddings,
            processing_time=processing_time,
            model_info=model_info
        )
    
    async def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query
        
        Args:
            query: Query string
            
        Returns:
            Numpy array of query embedding (1, embedding_dim)
        """
        def _generate_query_embedding():
            from app.utils.debug import debug_print
            self._load_model()
            
            debug_print(f"Generating embedding for query: {query[:50]}...")
            start_time = time.time()
            
            try:
                # Generate query embedding
                result = self.model.encode(
                    [query],
                    batch_size=1,
                    max_length=512,  # Queries are typically shorter
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False
                )
                
                # Extract dense embeddings from result dict
                if isinstance(result, dict):
                    embedding = result['dense_vecs']
                else:
                    embedding = result
                
                # Convert to numpy array
                if hasattr(embedding, 'numpy'):
                    embedding = embedding.numpy()
                elif torch.is_tensor(embedding):
                    embedding = embedding.cpu().numpy()
                
                # Ensure 2D array
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)
                
                # Normalize query embedding for consistent similarity calculations
                embedding = self._normalize_embeddings(embedding)
                
                generation_time = time.time() - start_time
                debug_print(f"Generated query embedding in {generation_time:.2f}s")
                debug_print(f"Embedding shape: {embedding.shape}")
                debug_print(f"Speed: {1 / generation_time:.1f} queries/second")
                
                return embedding
                
            except Exception as e:
                raise RuntimeError(f"Query embedding generation failed: {str(e)}")
        
        # PERFORMANCE OPTIMIZATION: Check LRU cache first (CRITICAL FOR LATENCY)
        from app.services.lru_cache_manager import get_lru_cache_manager
        cache_manager = get_lru_cache_manager()
        
        cached_embedding = cache_manager.get_query_embedding(query)
        if cached_embedding is not None:
            # LRU Cache hit - return immediately (MAJOR LATENCY REDUCTION)
            return cached_embedding
        
        # Generate new embedding
        embedding = _generate_query_embedding()
        
        # Cache the result for future use with LRU eviction
        cache_manager.set_query_embedding(query, embedding)
        
        return embedding
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for BGE-M3"""
        return 1024  # BGE-M3 embedding dimension
    
    def is_available(self) -> bool:
        """Check if BGE-M3 is available"""
        try:
            import FlagEmbedding
            return True
        except ImportError:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {
                "model_name": self.model_name,
                "device": self.device,
                "loaded": False
            }
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "loaded": True,
            "embedding_dimension": 1024,  # BGE-M3 embedding dimension
            "max_input_length": settings.max_tokens_per_chunk,
            "supports_multilingual": True
        }



class EmbeddingManager:
    """
    BGE-M3 embedding manager for RAG systems
    
    This class provides a unified interface for BGE-M3 embeddings while maintaining
    backward compatibility with the existing codebase.
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize embedding manager with BGE-M3 provider
        
        Args:
            provider: Optional provider name (only "bge-m3" supported)
        """
        self.provider_name = provider or settings.embedding_provider
        self._provider: Optional[EmbeddingProvider] = None
        
        logger.info(f"Initializing EmbeddingManager with provider: {self.provider_name}")
        
        # Initialize provider
        self._init_provider()
    
    def _init_provider(self):
        """Initialize the BGE-M3 embedding provider"""
        try:
            if self.provider_name == "bge-m3":
                self._provider = BGEEmbeddingProvider()
                logger.info("Using BGE-M3 embeddings")
            else:
                logger.warning(f"Unknown provider '{self.provider_name}', falling back to BGE-M3")
                self.provider_name = "bge-m3"
                self._provider = BGEEmbeddingProvider()
            
            # Validate provider is available
            if not self._provider.is_available():
                raise RuntimeError("BGE-M3 provider is not available")
        
        except Exception as e:
            logger.error(f"Failed to initialize BGE-M3 provider: {str(e)}")
            raise
    
    async def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts (backward compatibility method)
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of embeddings
        """
        result = await self._provider.embed_texts(texts)
        return result.embeddings
    
    async def encode_chunks(self, chunks: List[TextChunk]) -> EmbeddingResult:
        """
        Generate embeddings for text chunks (backward compatibility method)
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        start_time = time.time()
        
        # Extract texts from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        result = await self._provider.embed_texts(texts)
        
        # Add chunk-specific metadata
        result.model_info.update({
            "num_chunks": len(chunks),
            "total_tokens": sum(chunk.token_count for chunk in chunks),
            "processing_speed": len(chunks) / result.processing_time if result.processing_time > 0 else 0
        })
        
        logger.info("Chunk embedding completed:")
        logger.info(f"  - Chunks processed: {len(chunks)}")
        logger.info(f"  - Embedding dimension: {result.embeddings.shape[1] if result.embeddings.size > 0 else 0}")
        logger.info(f"  - Total tokens: {sum(chunk.token_count for chunk in chunks):,}")
        logger.info(f"  - Processing time: {result.processing_time:.2f}s")
        logger.info(f"  - Speed: {len(chunks) / result.processing_time:.1f} chunks/second")
        
        return result
    
    async def encode_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query with caching (backward compatibility method)
        
        Args:
            query: Query string
            
        Returns:
            Query embedding array
        """
        # PERFORMANCE OPTIMIZATION: Manager-level LRU caching check (CRITICAL FOR LATENCY)
        from app.services.lru_cache_manager import get_lru_cache_manager
        cache_manager = get_lru_cache_manager()
        
        cached_embedding = cache_manager.get_query_embedding(query)
        if cached_embedding is not None:
            return cached_embedding
        
        # Generate and cache with LRU eviction
        embedding = await self._provider.embed_query(query)
        cache_manager.set_query_embedding(query, embedding)
        
        return embedding
    
    def ensure_model_ready(self) -> bool:
        """
        Ensure the model is ready for use (backward compatibility)
        
        Returns:
            True if model is ready
        """
        try:
            return self._provider.is_available()
        except Exception as e:
            logger.error(f"Model readiness check failed: {str(e)}")
            return False
    
    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded (backward compatibility)
        
        Returns:
            True if model is loaded
        """
        return self._provider is not None and self._provider.is_available()
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for the current provider"""
        return self._provider.get_embedding_dimension()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information from current provider"""
        info = self._provider.get_model_info()
        info.update({
            "provider_name": self.provider_name,
            "manager_version": "2.0_multi_provider"
        })
        return info
    
    def get_provider_name(self) -> str:
        """Get the current provider name"""
        return self.provider_name


# Singleton instance
_embedding_manager = None

def get_embedding_manager() -> EmbeddingManager:
    """Get singleton embedding manager instance"""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager
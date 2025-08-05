"""
Embedding service using BGE-M3 for text embeddings
"""
import time
import threading
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.services.text_chunker import TextChunk
from app.core.config import settings


@dataclass
class EmbeddingResult:
    """Container for embedding results"""
    embeddings: np.ndarray
    processing_time: float
    model_info: Dict[str, Any]


class EmbeddingManager:
    """Manages BGE-M3 embeddings generation"""
    
    def __init__(self):
        """Initialize embedding manager"""
        self.model = None
        self.device = settings.embedding_device
        self.model_name = settings.embedding_model
        self._model_lock = threading.Lock()  # Thread-safe model loading
        self._model_loaded = False  # Track model loading state
        
        print(f"Initializing embedding manager: {self.model_name}")
        print(f"Target device: {self.device}")
        
        # Check device availability
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"
        elif self.device == "cuda":
            print(f"CUDA available: {torch.cuda.get_device_name()}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
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
                print(f"Loading BGE-M3 model (thread-safe): {self.model_name}")
                start_time = time.time()
                
                from FlagEmbedding import BGEM3FlagModel
                
                # Use to_empty() to avoid meta tensor issues in multi-threading
                self.model = BGEM3FlagModel(
                    self.model_name,
                    use_fp16=True,  # Use half precision for faster inference
                    device=self.device
                )
                
                load_time = time.time() - start_time
                print(f"BGE-M3 model loaded successfully in {load_time:.2f}s")
                
                # Warm up the model
                print("Warming up model...")
                warmup_start = time.time()
                self.model.encode(["This is a warmup text for the embedding model."])
                warmup_time = time.time() - warmup_start
                print(f"Model warmup completed in {warmup_time:.2f}s")
                
                # Mark as loaded
                self._model_loaded = True
                print(f"BGE-M3 model ready for parallel processing!")
                
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
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        self._load_model()
        
        print(f"Generating embeddings for {len(texts)} texts...")
        start_time = time.time()
        
        try:
            # Use BGE-M3's encode method
            result = self.model.encode(
                texts,
                batch_size=32,  # Process in batches for efficiency
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
            
            generation_time = time.time() - start_time
            print(f"Generated embeddings in {generation_time:.2f}s")
            print(f"Embedding shape: {embeddings.shape}")
            print(f"Speed: {len(texts) / generation_time:.1f} texts/second")
            
            return embeddings
            
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {str(e)}")
    
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
        
        print(f"Chunk embedding completed:")
        print(f"  - Chunks processed: {len(chunks)}")
        print(f"  - Embedding dimension: {model_info['embedding_dimension']}")
        print(f"  - Total tokens: {model_info['total_tokens']:,}")
        print(f"  - Processing time: {processing_time:.2f}s")
        print(f"  - Speed: {model_info['processing_speed']:.1f} chunks/second")
        
        return EmbeddingResult(
            embeddings=embeddings,
            processing_time=processing_time,
            model_info=model_info
        )
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query
        
        Args:
            query: Query string
            
        Returns:
            Numpy array of query embedding (1, embedding_dim)
        """
        self._load_model()
        
        print(f"Generating embedding for query: {query[:50]}...")
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
            
            generation_time = time.time() - start_time
            print(f"Generated query embedding in {generation_time:.2f}s")
            print(f"Embedding shape: {embedding.shape}")
            print(f"Speed: {1 / generation_time:.1f} queries/second")
            
            return embedding
            
        except Exception as e:
            raise RuntimeError(f"Query embedding generation failed: {str(e)}")
    
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


# Singleton instance
_embedding_manager = None

def get_embedding_manager() -> EmbeddingManager:
    """Get singleton embedding manager instance"""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager
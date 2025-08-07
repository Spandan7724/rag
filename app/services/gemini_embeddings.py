"""
Gemini Embedding Provider using Google GenAI API

This module provides Gemini embeddings with task-type optimization for RAG systems.
Supports batch processing, automatic normalization, and various embedding dimensions.
"""

import time
import asyncio
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class GeminiEmbeddingResult:
    """Container for Gemini embedding results"""
    embeddings: np.ndarray
    processing_time: float
    model_info: Dict[str, Any]
    api_calls_made: int
    total_tokens_processed: int


class GeminiEmbeddingProvider:
    """
    Google Gemini embedding provider with task-type optimization and batch processing
    """
    
    def __init__(self):
        """Initialize Gemini embedding provider"""
        self.model_name = settings.gemini_embedding_model
        self.dimension = settings.gemini_embedding_dimension
        self.batch_size = settings.gemini_batch_size
        self.api_timeout = settings.gemini_api_timeout
        self.rate_limit_delay = settings.gemini_rate_limit_delay
        
        # Task types for different operations
        self.task_type_document = settings.gemini_task_type_document
        self.task_type_query = settings.gemini_task_type_query
        
        # API client (initialized lazily)
        self._client = None
        self._api_key = None
        
        logger.info(f"Initialized Gemini embedding provider")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Dimension: {self.dimension}")
        logger.info(f"Batch size: {self.batch_size}")
        
        # Validate API key
        self._validate_api_key()
    
    def _validate_api_key(self):
        """Validate that Gemini API key is available"""
        self._api_key = os.getenv("GEMINI_API_KEY") or settings.gemini_api_key
        if not self._api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required for Gemini embeddings. "
                "Please set it in your .env file or environment."
            )
    
    def _get_client(self):
        """Get or create Gemini API client (lazy initialization)"""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self._api_key)
                logger.info("Gemini API client initialized successfully")
            except ImportError:
                raise ImportError(
                    "Google GenAI SDK not installed. Install with: pip install google-genai"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Gemini client: {str(e)}")
        return self._client
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings for smaller dimensions (< 3072)
        
        Args:
            embeddings: Raw embeddings array
            
        Returns:
            Normalized embeddings array
        """
        if self.dimension >= 3072:
            # 3072-dimensional embeddings are already normalized
            return embeddings
        
        # Normalize for smaller dimensions
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = embeddings / norms
        
        logger.debug(f"Normalized embeddings from {embeddings.shape} to {normalized.shape}")
        return normalized
    
    async def _call_embedding_api(
        self, 
        texts: List[str], 
        task_type: str
    ) -> List[np.ndarray]:
        """
        Call Gemini embedding API with task type optimization
        
        Args:
            texts: List of texts to embed
            task_type: Task type for optimization
            
        Returns:
            List of embedding arrays
        """
        client = self._get_client()
        
        try:
            from google.genai import types
            
            # Create embedding configuration
            config = types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.dimension
            )
            
            # Make API call
            result = await asyncio.to_thread(
                client.models.embed_content,
                model=self.model_name,
                contents=texts,
                config=config
            )
            
            # Extract embeddings
            embeddings = []
            for embedding_obj in result.embeddings:
                embedding_array = np.array(embedding_obj.values, dtype=np.float32)
                embeddings.append(embedding_array)
            
            logger.debug(f"API call successful: {len(embeddings)} embeddings generated")
            return embeddings
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            raise RuntimeError(f"Gemini embedding API error: {str(e)}")
    
    async def embed_texts(
        self, 
        texts: List[str], 
        task_type: str = None
    ) -> GeminiEmbeddingResult:
        """
        Generate embeddings for a list of texts with batch processing
        
        Args:
            texts: List of texts to embed
            task_type: Optional task type override
            
        Returns:
            GeminiEmbeddingResult with embeddings and metadata
        """
        if not texts:
            return GeminiEmbeddingResult(
                embeddings=np.array([]),
                processing_time=0.0,
                model_info=self._get_model_info(),
                api_calls_made=0,
                total_tokens_processed=0
            )
        
        start_time = time.time()
        task_type = task_type or self.task_type_document
        
        logger.info(f"Generating Gemini embeddings for {len(texts)} texts")
        logger.info(f"Task type: {task_type}, Dimension: {self.dimension}")
        
        # Process in batches to respect API limits
        all_embeddings = []
        api_calls_made = 0
        total_tokens_processed = 0
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            logger.debug(f"Processing batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
            
            # Make API call for this batch
            batch_embeddings = await self._call_embedding_api(batch, task_type)
            all_embeddings.extend(batch_embeddings)
            api_calls_made += 1
            
            # Estimate tokens processed (rough estimate: 1 token ≈ 4 characters)
            batch_tokens = sum(len(text) // 4 for text in batch)
            total_tokens_processed += batch_tokens
            
            # Rate limiting delay between batches
            if i + self.batch_size < len(texts) and self.rate_limit_delay > 0:
                await asyncio.sleep(self.rate_limit_delay)
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        
        # Normalize if needed
        embeddings_array = self._normalize_embeddings(embeddings_array)
        
        processing_time = time.time() - start_time
        
        # Create model info
        model_info = self._get_model_info()
        model_info.update({
            "task_type": task_type,
            "api_calls_made": api_calls_made,
            "batch_size": self.batch_size,
            "rate_limit_delay": self.rate_limit_delay,
            "processing_speed": len(texts) / processing_time if processing_time > 0 else 0
        })
        
        logger.info(f"Gemini embeddings completed:")
        logger.info(f"  - Texts processed: {len(texts)}")
        logger.info(f"  - API calls made: {api_calls_made}")
        logger.info(f"  - Total tokens: {total_tokens_processed:,}")
        logger.info(f"  - Processing time: {processing_time:.2f}s")
        logger.info(f"  - Speed: {len(texts) / processing_time:.1f} texts/second")
        logger.info(f"  - Embedding shape: {embeddings_array.shape}")
        
        return GeminiEmbeddingResult(
            embeddings=embeddings_array,
            processing_time=processing_time,
            model_info=model_info,
            api_calls_made=api_calls_made,
            total_tokens_processed=total_tokens_processed
        )
    
    async def embed_documents(self, texts: List[str]) -> GeminiEmbeddingResult:
        """
        Generate embeddings optimized for document indexing
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            GeminiEmbeddingResult with document-optimized embeddings
        """
        return await self.embed_texts(texts, self.task_type_document)
    
    async def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding optimized for search queries
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding array (1, embedding_dim)
        """
        result = await self.embed_texts([query], self.task_type_query)
        return result.embeddings
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get information about the Gemini embedding model"""
        return {
            "provider": "gemini",
            "model_name": self.model_name,
            "embedding_dimension": self.dimension,
            "max_input_length": 2048,  # Gemini token limit
            "supports_batch_processing": True,
            "supports_task_optimization": True,
            "task_type_document": self.task_type_document,
            "task_type_query": self.task_type_query,
            "api_timeout": self.api_timeout
        }
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for this provider"""
        return self.dimension
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for external use"""
        return self._get_model_info()
    
    def is_available(self) -> bool:
        """Check if the Gemini embedding provider is available"""
        try:
            self._validate_api_key()
            return True
        except Exception as e:
            logger.warning(f"Gemini embeddings not available: {str(e)}")
            return False


# Singleton instance
_gemini_embedding_provider = None


def get_gemini_embedding_provider() -> GeminiEmbeddingProvider:
    """Get singleton Gemini embedding provider instance"""
    global _gemini_embedding_provider
    if _gemini_embedding_provider is None:
        _gemini_embedding_provider = GeminiEmbeddingProvider()
    return _gemini_embedding_provider
"""
Performance-optimized cache manager for RAG system
"""
import hashlib
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from app.core.config import settings
from app.utils.debug import conditional_print


@dataclass
class CacheEntry:
    """Cache entry with TTL"""
    value: Any
    timestamp: float
    ttl: float
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return time.time() - self.timestamp > self.ttl


class CacheManager:
    """High-performance in-memory cache manager"""
    
    def __init__(self):
        """Initialize cache manager"""
        self.query_cache: Dict[str, CacheEntry] = {}
        self.embedding_cache: Dict[str, CacheEntry] = {}
        self.query_embedding_cache: Dict[str, CacheEntry] = {}  # Cache for query embeddings specifically
        self.reranker_cache: Dict[str, CacheEntry] = {}
        self.answer_cache: Dict[str, CacheEntry] = {}
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        self.query_embedding_hits = 0  # Track query embedding cache hits
        self.query_embedding_requests = 0  # Track total query embedding requests
        
        conditional_print("Cache Manager initialized")
        if settings.enable_result_caching:
            conditional_print(f"  - Query result caching enabled (TTL: {settings.cache_ttl_seconds}s)")
        if settings.enable_embedding_cache:
            conditional_print("  - Embedding caching enabled")
            conditional_print("  - Query embedding caching enabled (HIGH PERFORMANCE IMPACT)")
        if settings.enable_reranker_cache:
            conditional_print("  - Reranker score caching enabled")
    
    def _generate_cache_key(self, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, str):
            content = data
        elif isinstance(data, dict):
            content = json.dumps(data, sort_keys=True)
        elif isinstance(data, (list, tuple)):
            content = json.dumps(list(data), sort_keys=True)
        else:
            content = str(data)
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cleanup_expired(self, cache_dict: Dict[str, CacheEntry]) -> int:
        """Remove expired entries from cache"""
        expired_keys = [
            key for key, entry in cache_dict.items() 
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del cache_dict[key]
        
        return len(expired_keys)
    
    def get_query_result(self, question: str, doc_id: str, k_retrieve: int) -> Optional[Dict[str, Any]]:
        """Get cached query result"""
        if not settings.enable_result_caching:
            return None
        
        self.total_requests += 1
        
        # Create cache key from query parameters
        cache_key = self._generate_cache_key({
            'question': question.lower().strip(),
            'doc_id': doc_id,
            'k_retrieve': k_retrieve
        })
        
        # Check cache
        if cache_key in self.query_cache:
            entry = self.query_cache[cache_key]
            if not entry.is_expired():
                self.hits += 1
                print(f"  Cache HIT for query: {question[:30]}...")
                return entry.value
            else:
                # Remove expired entry
                del self.query_cache[cache_key]
        
        self.misses += 1
        return None
    
    def set_query_result(self, question: str, doc_id: str, k_retrieve: int, result: Dict[str, Any]) -> None:
        """Cache query result"""
        if not settings.enable_result_caching:
            return
        
        cache_key = self._generate_cache_key({
            'question': question.lower().strip(),
            'doc_id': doc_id,
            'k_retrieve': k_retrieve
        })
        
        self.query_cache[cache_key] = CacheEntry(
            value=result,
            timestamp=time.time(),
            ttl=settings.cache_ttl_seconds
        )
        
        print(f"  Cached query result: {question[:30]}...")
        
        # Periodic cleanup (every 50 entries)
        if len(self.query_cache) % 50 == 0:
            cleaned = self._cleanup_expired(self.query_cache)
            if cleaned > 0:
                print(f"  Cleaned {cleaned} expired query cache entries")
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding"""
        if not settings.enable_embedding_cache:
            return None
        
        cache_key = self._generate_cache_key(text)
        
        if cache_key in self.embedding_cache:
            entry = self.embedding_cache[cache_key]
            if not entry.is_expired():
                return entry.value
            else:
                del self.embedding_cache[cache_key]
        
        return None
    
    def get_query_embedding(self, query: str) -> Optional[Any]:
        """Get cached query embedding (HIGH PERFORMANCE IMPACT)"""
        if not settings.enable_embedding_cache:
            return None
        
        self.query_embedding_requests += 1
        
        # Normalize query for consistent caching
        normalized_query = query.lower().strip()
        cache_key = self._generate_cache_key(normalized_query)
        
        if cache_key in self.query_embedding_cache:
            entry = self.query_embedding_cache[cache_key]
            if not entry.is_expired():
                self.query_embedding_hits += 1
                conditional_print(f"Query embedding cache HIT: {query[:40]}...")
                return entry.value
            else:
                del self.query_embedding_cache[cache_key]
        
        return None
    
    def set_query_embedding(self, query: str, embedding: Any) -> None:
        """Cache query embedding for fast reuse"""
        if not settings.enable_embedding_cache:
            return
        
        # Normalize query for consistent caching
        normalized_query = query.lower().strip()
        cache_key = self._generate_cache_key(normalized_query)
        
        # Cache query embeddings for longer (they're expensive to compute)
        self.query_embedding_cache[cache_key] = CacheEntry(
            value=embedding,
            timestamp=time.time(),
            ttl=settings.cache_ttl_seconds * 6  # 6 hours - longer TTL for queries
        )
        
        conditional_print(f"Cached query embedding: {query[:40]}...")
        
        # Cleanup every 25 entries to maintain performance
        if len(self.query_embedding_cache) % 25 == 0:
            cleaned = self._cleanup_expired(self.query_embedding_cache)
            if cleaned > 0:
                conditional_print(f"Cleaned {cleaned} expired query embedding cache entries")
    
    def set_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache embedding"""
        if not settings.enable_embedding_cache:
            return
        
        cache_key = self._generate_cache_key(text)
        
        # Use longer TTL for embeddings (they don't change)
        self.embedding_cache[cache_key] = CacheEntry(
            value=embedding,
            timestamp=time.time(),
            ttl=settings.cache_ttl_seconds * 24  # 24 hours
        )
    
    def get_reranker_scores(self, question: str, chunk_texts: List[str]) -> Optional[List[float]]:
        """Get cached reranker scores"""
        if not settings.enable_reranker_cache:
            return None
        
        cache_key = self._generate_cache_key({
            'question': question.lower().strip(),
            'texts': chunk_texts
        })
        
        if cache_key in self.reranker_cache:
            entry = self.reranker_cache[cache_key]
            if not entry.is_expired():
                return entry.value
            else:
                del self.reranker_cache[cache_key]
        
        return None
    
    def set_reranker_scores(self, question: str, chunk_texts: List[str], scores: List[float]) -> None:
        """Cache reranker scores"""
        if not settings.enable_reranker_cache:
            return
        
        cache_key = self._generate_cache_key({
            'question': question.lower().strip(),
            'texts': chunk_texts
        })
        
        self.reranker_cache[cache_key] = CacheEntry(
            value=scores,
            timestamp=time.time(),
            ttl=settings.cache_ttl_seconds * 2  # 2 hours
        )
    
    async def get_answer_cache(self, cache_key: str):
        """Get cached complete answer response"""
        if cache_key in self.answer_cache:
            entry = self.answer_cache[cache_key]
            if not entry.is_expired():
                self.hits += 1
                return entry.value
            else:
                del self.answer_cache[cache_key]
        
        self.misses += 1
        return None
    
    async def set_answer_cache(self, cache_key: str, response) -> None:
        """Cache complete answer response"""
        # Cache answers for 1 hour
        self.answer_cache[cache_key] = CacheEntry(
            value=response,
            timestamp=time.time(),
            ttl=settings.cache_ttl_seconds * 1  # 1 hour
        )
        
        # Periodic cleanup (every 25 entries)
        if len(self.answer_cache) % 25 == 0:
            cleaned = self._cleanup_expired(self.answer_cache)
            if cleaned > 0:
                print(f"  Cleaned {cleaned} expired answer cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        hit_rate = (self.hits / self.total_requests * 100) if self.total_requests > 0 else 0
        query_embedding_hit_rate = (self.query_embedding_hits / self.query_embedding_requests * 100) if self.query_embedding_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate_percent": round(hit_rate, 2),
            "query_embedding_stats": {
                "requests": self.query_embedding_requests,
                "hits": self.query_embedding_hits,
                "hit_rate_percent": round(query_embedding_hit_rate, 2),
                "performance_impact": "HIGH - saves embedding computation time"
            },
            "cache_sizes": {
                "query_cache": len(self.query_cache),
                "embedding_cache": len(self.embedding_cache),
                "query_embedding_cache": len(self.query_embedding_cache),
                "reranker_cache": len(self.reranker_cache),
                "answer_cache": len(self.answer_cache)
            },
            "memory_usage_estimate_mb": round(
                (len(self.query_cache) * 0.5 + 
                 len(self.embedding_cache) * 0.1 + 
                 len(self.query_embedding_cache) * 0.2 +
                 len(self.reranker_cache) * 0.05 +
                 len(self.answer_cache) * 1.0), 2  # Answers are larger
            )
        }
    
    def clear_cache(self, cache_type: str = "all") -> Dict[str, int]:
        """Clear cache entries"""
        cleared = {}
        
        if cache_type in ["all", "query"]:
            cleared["query"] = len(self.query_cache)
            self.query_cache.clear()
        
        if cache_type in ["all", "embedding"]:
            cleared["embedding"] = len(self.embedding_cache)
            self.embedding_cache.clear()
            
        if cache_type in ["all", "query_embedding"]:
            cleared["query_embedding"] = len(self.query_embedding_cache)
            self.query_embedding_cache.clear()
        
        if cache_type in ["all", "reranker"]:
            cleared["reranker"] = len(self.reranker_cache)
            self.reranker_cache.clear()
        
        if cache_type in ["all", "answer"]:
            cleared["answer"] = len(self.answer_cache)
            self.answer_cache.clear()
        
        if cache_type == "all":
            self.hits = 0
            self.misses = 0
            self.total_requests = 0
            self.query_embedding_hits = 0
            self.query_embedding_requests = 0
        
        return cleared
    
    def cleanup_expired_entries(self) -> Dict[str, int]:
        """Manual cleanup of expired entries"""
        cleaned = {
            "query": self._cleanup_expired(self.query_cache),
            "embedding": self._cleanup_expired(self.embedding_cache),
            "query_embedding": self._cleanup_expired(self.query_embedding_cache),
            "reranker": self._cleanup_expired(self.reranker_cache),
            "answer": self._cleanup_expired(self.answer_cache)
        }
        
        total_cleaned = sum(cleaned.values())
        if total_cleaned > 0:
            print(f"Cleaned {total_cleaned} expired cache entries")
        
        return cleaned


# Singleton instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get singleton cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
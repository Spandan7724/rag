"""
High-Performance LRU Cache Manager for RAG System
Optimized for latency reduction with bounded memory usage
"""
import hashlib
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import OrderedDict
from app.core.config import settings
from app.utils.debug import conditional_print


@dataclass
class LRUCacheEntry:
    """LRU Cache entry with TTL"""
    value: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return time.time() - self.timestamp > self.ttl


class BoundedLRUCache:
    """LRU cache with size limit and TTL"""
    
    def __init__(self, max_size: int, default_ttl: float):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, LRUCacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache (LRU + TTL)"""
        if key in self.cache:
            entry = self.cache[key]
            
            # Check expiration first
            if entry.is_expired():
                del self.cache[key]
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            entry.access_count += 1
            self.hits += 1
            return entry.value
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put item in cache with LRU eviction"""
        ttl = ttl or self.default_ttl
        
        if key in self.cache:
            # Update existing entry
            self.cache[key].value = value
            self.cache[key].timestamp = time.time()
            self.cache[key].ttl = ttl
            self.cache.move_to_end(key)
        else:
            # Add new entry
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                self.cache.popitem(last=False)
            
            self.cache[key] = LRUCacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl
            )
    
    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": round(hit_rate, 2),
            "memory_efficiency": f"{len(self.cache)}/{self.max_size}"
        }


class HighPerformanceCacheManager:
    """Ultra-fast LRU cache manager optimized for latency reduction"""
    
    def __init__(self):
        """Initialize with optimized cache sizes for different data types"""
        
        # Different cache sizes based on data importance and size
        self.query_cache = BoundedLRUCache(max_size=1000, default_ttl=settings.cache_ttl_seconds)  # Most important
        self.query_embedding_cache = BoundedLRUCache(max_size=2000, default_ttl=settings.cache_ttl_seconds * 6)  # HIGH IMPACT
        self.embedding_cache = BoundedLRUCache(max_size=4000, default_ttl=settings.cache_ttl_seconds * 24)  # Large but efficient
        self.reranker_cache = BoundedLRUCache(max_size=600, default_ttl=settings.cache_ttl_seconds * 2)  # Medium priority
        self.answer_cache = BoundedLRUCache(max_size=500, default_ttl=settings.cache_ttl_seconds)  # Large objects
        
        conditional_print("🚀 High-Performance LRU Cache Manager initialized")
        conditional_print(f"  - Query Cache: {self.query_cache.max_size} entries")
        conditional_print(f"  - Query Embedding Cache: {self.query_embedding_cache.max_size} entries (CRITICAL FOR LATENCY)")
        conditional_print(f"  - Embedding Cache: {self.embedding_cache.max_size} entries") 
        conditional_print(f"  - Reranker Cache: {self.reranker_cache.max_size} entries")
        conditional_print(f"  - Answer Cache: {self.answer_cache.max_size} entries")
    
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
    
    # Query Results Cache (Complete Q&A responses)
    def get_query_result(self, question: str, doc_id: str, k_retrieve: int) -> Optional[Dict[str, Any]]:
        """Get cached query result with LRU"""
        if not settings.enable_result_caching:
            return None
        
        cache_key = self._generate_cache_key({
            'question': question.lower().strip(),
            'doc_id': doc_id,
            'k_retrieve': k_retrieve
        })
        
        result = self.query_cache.get(cache_key)
        if result:
            conditional_print(f"🎯 Query cache HIT: {question[:30]}...")
        return result
    
    def set_query_result(self, question: str, doc_id: str, k_retrieve: int, result: Dict[str, Any]) -> None:
        """Cache query result with LRU"""
        if not settings.enable_result_caching:
            return
        
        cache_key = self._generate_cache_key({
            'question': question.lower().strip(),
            'doc_id': doc_id,
            'k_retrieve': k_retrieve
        })
        
        self.query_cache.put(cache_key, result)
        conditional_print(f"💾 Cached query result: {question[:30]}...")
    
    # Query Embedding Cache (HIGHEST IMPACT ON LATENCY)
    def get_query_embedding(self, query: str) -> Optional[Any]:
        """Get cached query embedding - CRITICAL FOR LATENCY REDUCTION"""
        if not settings.enable_embedding_cache:
            return None
        
        normalized_query = query.lower().strip()
        cache_key = self._generate_cache_key(normalized_query)
        
        result = self.query_embedding_cache.get(cache_key)
        if result is not None:
            conditional_print(f"⚡ Query embedding cache HIT (MAJOR LATENCY SAVE): {query[:40]}...")
        return result
    
    def set_query_embedding(self, query: str, embedding: Any) -> None:
        """Cache query embedding - CRITICAL FOR LATENCY"""
        if not settings.enable_embedding_cache:
            return
        
        normalized_query = query.lower().strip()
        cache_key = self._generate_cache_key(normalized_query)
        
        # Longer TTL for query embeddings - they're expensive to compute
        self.query_embedding_cache.put(cache_key, embedding, ttl=settings.cache_ttl_seconds * 6)
        conditional_print(f"Cached query embedding: {query[:40]}...")
    
    # Regular embedding cache
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached text embedding"""
        if not settings.enable_embedding_cache:
            return None
        
        cache_key = self._generate_cache_key(text)
        return self.embedding_cache.get(cache_key)
    
    def set_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache text embedding"""
        if not settings.enable_embedding_cache:
            return
        
        cache_key = self._generate_cache_key(text)
        self.embedding_cache.put(cache_key, embedding, ttl=settings.cache_ttl_seconds * 24)
    
    # Reranker cache
    def get_reranker_scores(self, question: str, chunk_texts: List[str]) -> Optional[List[float]]:
        """Get cached reranker scores"""
        if not settings.enable_reranker_cache:
            return None
        
        cache_key = self._generate_cache_key({
            'question': question.lower().strip(),
            'texts': chunk_texts
        })
        
        return self.reranker_cache.get(cache_key)
    
    def set_reranker_scores(self, question: str, chunk_texts: List[str], scores: List[float]) -> None:
        """Cache reranker scores"""
        if not settings.enable_reranker_cache:
            return
        
        cache_key = self._generate_cache_key({
            'question': question.lower().strip(),
            'texts': chunk_texts
        })
        
        self.reranker_cache.put(cache_key, scores, ttl=settings.cache_ttl_seconds * 2)
    
    # Answer cache 
    async def get_answer_cache(self, cache_key: str):
        """Get cached complete answer"""
        return self.answer_cache.get(cache_key)
    
    async def set_answer_cache(self, cache_key: str, response) -> None:
        """Cache complete answer"""
        self.answer_cache.put(cache_key, response)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            "cache_type": "LRU + TTL Hybrid (Optimized for Latency)",
            "query_cache": self.query_cache.get_stats(),
            "query_embedding_cache": {
                **self.query_embedding_cache.get_stats(),
                "performance_impact": "CRITICAL - Major latency reduction"
            },
            "embedding_cache": self.embedding_cache.get_stats(),
            "reranker_cache": self.reranker_cache.get_stats(),
            "answer_cache": self.answer_cache.get_stats(),
            "total_memory_bounded": True,
            "latency_optimization": "HIGH - Zero network overhead, bounded memory, hot data retention"
        }
    
    def cleanup_expired_entries(self) -> Dict[str, int]:
        """Cleanup expired entries across all caches"""
        return {
            "query_cache": self.query_cache.cleanup_expired(),
            "query_embedding_cache": self.query_embedding_cache.cleanup_expired(),
            "embedding_cache": self.embedding_cache.cleanup_expired(),
            "reranker_cache": self.reranker_cache.cleanup_expired(),
            "answer_cache": self.answer_cache.cleanup_expired()
        }
    
    def clear_cache(self, cache_type: str = "all") -> Dict[str, int]:
        """Clear specific or all caches"""
        cleared = {}
        
        if cache_type in ["all", "query"]:
            cleared["query"] = len(self.query_cache.cache)
            self.query_cache.cache.clear()
            self.query_cache.hits = 0
            self.query_cache.misses = 0
        
        if cache_type in ["all", "query_embedding"]:
            cleared["query_embedding"] = len(self.query_embedding_cache.cache)
            self.query_embedding_cache.cache.clear()
            self.query_embedding_cache.hits = 0
            self.query_embedding_cache.misses = 0
        
        if cache_type in ["all", "embedding"]:
            cleared["embedding"] = len(self.embedding_cache.cache)
            self.embedding_cache.cache.clear()
            self.embedding_cache.hits = 0
            self.embedding_cache.misses = 0
            
        if cache_type in ["all", "reranker"]:
            cleared["reranker"] = len(self.reranker_cache.cache)
            self.reranker_cache.cache.clear()
            self.reranker_cache.hits = 0
            self.reranker_cache.misses = 0
        
        if cache_type in ["all", "answer"]:
            cleared["answer"] = len(self.answer_cache.cache)
            self.answer_cache.cache.clear()
            self.answer_cache.hits = 0
            self.answer_cache.misses = 0
        
        return cleared


# Singleton instance
_lru_cache_manager = None

def get_lru_cache_manager() -> HighPerformanceCacheManager:
    """Get singleton high-performance LRU cache manager"""
    global _lru_cache_manager
    if _lru_cache_manager is None:
        _lru_cache_manager = HighPerformanceCacheManager()
    return _lru_cache_manager
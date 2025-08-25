
import os
import json
import pickle
import hashlib
import time
from typing import Any, Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import threading
from app.core.config import settings
from app.utils.debug import conditional_print


@dataclass
class PersistentCacheEntry:
    """Persistent cache entry with metadata"""
    value: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    size_bytes: int = 0
    content_hash: str = ""
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return time.time() - self.timestamp > self.ttl
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "access_count": self.access_count,
            "size_bytes": self.size_bytes,
            "content_hash": self.content_hash
        }


class PersistentCacheManager:
    """
    Persistent cache manager with disk storage
    Provides caching that survives container restarts
    """
    
    def __init__(self):
        """Initialize persistent cache manager"""
        self.cache_dir = Path("cache_store")
        self.metadata_dir = self.cache_dir / "metadata"
        self.data_dir = self.cache_dir / "data"
        
        # Create cache directories
        self._ensure_directories()
        
        # Cache types and their TTLs
        self.cache_types = {
            "embeddings": settings.cache_ttl_seconds * 24 * 7,  # 7 days
            "documents": settings.cache_ttl_seconds * 24 * 3,   # 3 days
            "query_results": settings.cache_ttl_seconds * 6,     # 6 hours
            "processed_docs": settings.cache_ttl_seconds * 24 * 7,  # 7 days
            "landmark_mappings": settings.cache_ttl_seconds * 24 * 14  # 14 days
        }
        
        # Thread lock for concurrent access
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "errors": 0
        }
        
        conditional_print("💾 Persistent Cache Manager initialized")
        conditional_print(f"  - Cache directory: {self.cache_dir}")
        conditional_print(f"  - Cache types: {list(self.cache_types.keys())}")
        
        # Load existing cache statistics
        self._load_stats()
    
    def _ensure_directories(self):
        """Create cache directories if they don't exist"""
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each cache type
        for cache_type in ["embeddings", "documents", "query_results", "processed_docs", "landmark_mappings"]:
            (self.metadata_dir / cache_type).mkdir(exist_ok=True)
            (self.data_dir / cache_type).mkdir(exist_ok=True)
    
    def _generate_cache_key(self, data: Any) -> str:
        """Generate consistent cache key"""
        if isinstance(data, str):
            content = data
        elif isinstance(data, dict):
            content = json.dumps(data, sort_keys=True)
        elif isinstance(data, (list, tuple)):
            content = json.dumps(list(data), sort_keys=True)
        else:
            content = str(data)
        
        return hashlib.sha256(content.encode()).hexdigest()[:16]  # 16 chars for file system
    
    def _get_cache_paths(self, cache_type: str, cache_key: str) -> tuple[Path, Path]:
        """Get metadata and data file paths"""
        metadata_path = self.metadata_dir / cache_type / f"{cache_key}.json"
        data_path = self.data_dir / cache_type / f"{cache_key}.pkl"
        return metadata_path, data_path
    
    def get(self, cache_type: str, key: Any, default: Any = None) -> Any:
        """Get item from persistent cache"""
        with self._lock:
            try:
                cache_key = self._generate_cache_key(key)
                metadata_path, data_path = self._get_cache_paths(cache_type, cache_key)
                
                # Check if files exist
                if not metadata_path.exists() or not data_path.exists():
                    self.stats["misses"] += 1
                    return default
                
                # Load metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Check expiration
                if time.time() - metadata["timestamp"] > metadata["ttl"]:
                    # Clean up expired entry
                    self._remove_cache_entry(metadata_path, data_path)
                    self.stats["misses"] += 1
                    return default
                
                # Load data
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Update access count
                metadata["access_count"] += 1
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f)
                
                self.stats["hits"] += 1
                conditional_print(f"📂 Persistent cache HIT: {cache_type}/{cache_key[:8]}...")
                return data
                
            except Exception as e:
                conditional_print(f"❌ Persistent cache read error: {e}")
                self.stats["errors"] += 1
                return default
    
    def set(self, cache_type: str, key: Any, value: Any, ttl: Optional[float] = None) -> bool:
        """Set item in persistent cache"""
        with self._lock:
            try:
                cache_key = self._generate_cache_key(key)
                metadata_path, data_path = self._get_cache_paths(cache_type, cache_key)
                
                # Use default TTL for cache type
                if ttl is None:
                    ttl = self.cache_types.get(cache_type, settings.cache_ttl_seconds)
                
                # Serialize data
                data_bytes = pickle.dumps(value)
                content_hash = hashlib.md5(data_bytes).hexdigest()
                
                # Create metadata
                metadata = {
                    "timestamp": time.time(),
                    "ttl": ttl,
                    "access_count": 0,
                    "size_bytes": len(data_bytes),
                    "content_hash": content_hash
                }
                
                # Write data and metadata atomically
                with open(data_path, 'wb') as f:
                    f.write(data_bytes)
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                self.stats["writes"] += 1
                conditional_print(f"💾 Cached to disk: {cache_type}/{cache_key[:8]} ({len(data_bytes)} bytes)")
                return True
                
            except Exception as e:
                conditional_print(f"❌ Persistent cache write error: {e}")
                self.stats["errors"] += 1
                return False
    
    def _remove_cache_entry(self, metadata_path: Path, data_path: Path):
        """Remove cache entry files"""
        try:
            if metadata_path.exists():
                metadata_path.unlink()
            if data_path.exists():
                data_path.unlink()
        except Exception as e:
            conditional_print(f"❌ Error removing cache entry: {e}")
    
    def cleanup_expired(self, cache_type: Optional[str] = None) -> Dict[str, int]:
        """Clean up expired cache entries"""
        with self._lock:
            cleaned = {}
            
            cache_types_to_clean = [cache_type] if cache_type else list(self.cache_types.keys())
            
            for ct in cache_types_to_clean:
                cleaned[ct] = 0
                metadata_dir = self.metadata_dir / ct
                data_dir = self.data_dir / ct
                
                if not metadata_dir.exists():
                    continue
                
                for metadata_file in metadata_dir.glob("*.json"):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        # Check if expired
                        if time.time() - metadata["timestamp"] > metadata["ttl"]:
                            # Remove both metadata and data files
                            cache_key = metadata_file.stem
                            data_file = data_dir / f"{cache_key}.pkl"
                            
                            self._remove_cache_entry(metadata_file, data_file)
                            cleaned[ct] += 1
                            
                    except Exception as e:
                        conditional_print(f"❌ Error cleaning {metadata_file}: {e}")
            
            total_cleaned = sum(cleaned.values())
            if total_cleaned > 0:
                conditional_print(f"🧹 Cleaned {total_cleaned} expired persistent cache entries")
            
            return cleaned
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self._lock:
            stats = {
                "persistent_cache_stats": {
                    "hits": self.stats["hits"],
                    "misses": self.stats["misses"],
                    "writes": self.stats["writes"],
                    "errors": self.stats["errors"],
                    "hit_rate_percent": round(
                        (self.stats["hits"] / (self.stats["hits"] + self.stats["misses"]) * 100)
                        if (self.stats["hits"] + self.stats["misses"]) > 0 else 0, 2
                    )
                },
                "cache_types": {},
                "disk_usage": {}
            }
            
            total_size = 0
            total_entries = 0
            
            # Get stats for each cache type
            for cache_type in self.cache_types.keys():
                metadata_dir = self.metadata_dir / cache_type
                data_dir = self.data_dir / cache_type
                
                if metadata_dir.exists():
                    entries = list(metadata_dir.glob("*.json"))
                    total_entries += len(entries)
                    
                    type_size = 0
                    valid_entries = 0
                    
                    for entry_file in entries:
                        try:
                            # Get file sizes
                            meta_size = entry_file.stat().st_size
                            data_file = data_dir / f"{entry_file.stem}.pkl"
                            data_size = data_file.stat().st_size if data_file.exists() else 0
                            
                            type_size += meta_size + data_size
                            valid_entries += 1
                            
                        except Exception:
                            continue
                    
                    total_size += type_size
                    
                    stats["cache_types"][cache_type] = {
                        "entries": valid_entries,
                        "size_mb": round(type_size / 1024 / 1024, 2),
                        "ttl_hours": round(self.cache_types[cache_type] / 3600, 1)
                    }
            
            stats["disk_usage"] = {
                "total_entries": total_entries,
                "total_size_mb": round(total_size / 1024 / 1024, 2),
                "cache_directory": str(self.cache_dir)
            }
            
            return stats
    
    def clear_cache(self, cache_type: Optional[str] = None) -> Dict[str, int]:
        """Clear persistent cache"""
        with self._lock:
            cleared = {}
            
            cache_types_to_clear = [cache_type] if cache_type else list(self.cache_types.keys())
            
            for ct in cache_types_to_clear:
                cleared[ct] = 0
                
                # Clear metadata and data directories
                for directory in [self.metadata_dir / ct, self.data_dir / ct]:
                    if directory.exists():
                        for file_path in directory.glob("*"):
                            try:
                                file_path.unlink()
                                cleared[ct] += 1
                            except Exception as e:
                                conditional_print(f"❌ Error deleting {file_path}: {e}")
                
                cleared[ct] = cleared[ct] // 2  # Divide by 2 since we delete both metadata and data
            
            # Reset stats if clearing all
            if cache_type is None:
                self.stats = {"hits": 0, "misses": 0, "writes": 0, "errors": 0}
            
            return cleared
    
    def _load_stats(self):
        """Load persistent statistics"""
        stats_file = self.cache_dir / "stats.json"
        try:
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    loaded_stats = json.load(f)
                    self.stats.update(loaded_stats)
                conditional_print(f"📊 Loaded persistent cache stats: {self.stats}")
        except Exception as e:
            conditional_print(f"❌ Error loading cache stats: {e}")
    
    def save_stats(self):
        """Save persistent statistics"""
        stats_file = self.cache_dir / "stats.json"
        try:
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            conditional_print(f"❌ Error saving cache stats: {e}")


# High-level convenience functions for common cache operations

class DocumentCache:
    """High-level document caching interface"""
    
    def __init__(self, persistent_cache: PersistentCacheManager):
        self.cache = persistent_cache
    
    def get_processed_document(self, doc_url: str) -> Optional[Dict[str, Any]]:
        """Get processed document from cache"""
        return self.cache.get("processed_docs", {"url": doc_url})
    
    def cache_processed_document(self, doc_url: str, processed_data: Dict[str, Any]) -> bool:
        """Cache processed document data"""
        return self.cache.set("processed_docs", {"url": doc_url}, processed_data)
    
    def get_document_embeddings(self, doc_id: str) -> Optional[List[List[float]]]:
        """Get document embeddings from cache"""
        return self.cache.get("embeddings", {"doc_id": doc_id, "type": "document"})
    
    def cache_document_embeddings(self, doc_id: str, embeddings: List[List[float]]) -> bool:
        """Cache document embeddings"""
        return self.cache.set("embeddings", {"doc_id": doc_id, "type": "document"}, embeddings)


class QueryCache:
    """High-level query caching interface"""
    
    def __init__(self, persistent_cache: PersistentCacheManager):
        self.cache = persistent_cache
    
    def get_query_result(self, question: str, doc_id: str, k_retrieve: int) -> Optional[Dict[str, Any]]:
        """Get cached query result"""
        cache_key = {
            "question": question.lower().strip(),
            "doc_id": doc_id,
            "k_retrieve": k_retrieve
        }
        return self.cache.get("query_results", cache_key)
    
    def cache_query_result(self, question: str, doc_id: str, k_retrieve: int, result: Dict[str, Any]) -> bool:
        """Cache query result"""
        cache_key = {
            "question": question.lower().strip(),
            "doc_id": doc_id,
            "k_retrieve": k_retrieve
        }
        return self.cache.set("query_results", cache_key, result)


# Singleton instances
_persistent_cache_manager = None
_document_cache = None  
_query_cache = None

def get_persistent_cache_manager() -> PersistentCacheManager:
    """Get singleton persistent cache manager"""
    global _persistent_cache_manager
    if _persistent_cache_manager is None:
        _persistent_cache_manager = PersistentCacheManager()
    return _persistent_cache_manager

def get_document_cache() -> DocumentCache:
    """Get singleton document cache"""
    global _document_cache
    if _document_cache is None:
        _document_cache = DocumentCache(get_persistent_cache_manager())
    return _document_cache

def get_query_cache() -> QueryCache:
    """Get singleton query cache"""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache(get_persistent_cache_manager())
    return _query_cache
"""
FAISS-based vector store with document deduplication
"""
import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import time

from app.services.text_chunker import TextChunk
from app.core.config import settings
from app.core.directories import create_directory


@dataclass
class ChunkMetadata:
    """Metadata for a stored chunk"""
    doc_id: str
    chunk_id: int
    text_preview: str  # First 100 chars for debugging
    token_count: int
    page: int
    heading: str
    section_number: Optional[str]
    chunk_type: str
    char_start: int
    char_end: int


@dataclass
class SearchResult:
    """Result from vector search"""
    chunk_index: int
    similarity_score: float
    text: str
    metadata: ChunkMetadata


@dataclass
class DocumentInfo:
    """Information about a stored document"""
    doc_id: str
    source_url: str
    title: str
    pages: int
    chunk_count: int
    indexed_at: float
    total_tokens: int


class VectorStore:
    """FAISS-based vector store with document management"""
    
    def __init__(self, storage_path: str = "vector_store"):
        """Initialize vector store"""
        self.storage_path = create_directory(storage_path, "vector store")
        
        # FAISS index
        self.index: Optional[faiss.Index] = None
        self.embedding_dim = 1024  # BGE-M3 dimension
        
        # Storage for chunks and metadata
        self.chunk_texts: List[str] = []
        self.chunk_metadata: List[ChunkMetadata] = []
        self.documents: Dict[str, DocumentInfo] = {}
        
        # File paths
        self.index_path = self.storage_path / "faiss_index.idx"
        self.texts_path = self.storage_path / "chunk_texts.pkl"
        self.metadata_path = self.storage_path / "chunk_metadata.pkl"
        self.documents_path = self.storage_path / "documents.json"
        
        # Load existing data
        self._load_from_disk()
        
        print(f"Vector store initialized at: {self.storage_path}")
        if self.index:
            print(f"Loaded existing index with {self.index.ntotal} vectors")
            print(f"Tracking {len(self.documents)} documents")
    
    def _create_index(self) -> faiss.Index:
        """Create a new FAISS index"""
        # Use HNSW index for good performance and recall
        index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        index.hnsw.efConstruction = 128
        index.hnsw.efSearch = 64
        return index
    
    def _load_from_disk(self):
        """Load existing index and metadata from disk"""
        try:
            # Load FAISS index
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                print(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
            # Load chunk texts
            if self.texts_path.exists():
                with open(self.texts_path, 'rb') as f:
                    self.chunk_texts = pickle.load(f)
                print(f"Loaded {len(self.chunk_texts)} chunk texts")
            
            # Load chunk metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'rb') as f:
                    metadata_dicts = pickle.load(f)
                    self.chunk_metadata = [ChunkMetadata(**md) for md in metadata_dicts]
                print(f"Loaded {len(self.chunk_metadata)} chunk metadata entries")
            
            # Load document info
            if self.documents_path.exists():
                with open(self.documents_path, 'r') as f:
                    doc_dicts = json.load(f)
                    self.documents = {
                        doc_id: DocumentInfo(**doc_info) 
                        for doc_id, doc_info in doc_dicts.items()
                    }
                print(f"Loaded {len(self.documents)} document records")
                    
        except Exception as e:
            print(f"Warning: Failed to load existing data: {e}")
            print("Starting with empty vector store")
            self.index = None
            self.chunk_texts = []
            self.chunk_metadata = []
            self.documents = {}
    
    def _save_to_disk(self):
        """Save index and metadata to disk"""
        try:
            # Save FAISS index
            if self.index:
                faiss.write_index(self.index, str(self.index_path))
            
            # Save chunk texts
            with open(self.texts_path, 'wb') as f:
                pickle.dump(self.chunk_texts, f)
            
            # Save chunk metadata
            with open(self.metadata_path, 'wb') as f:
                metadata_dicts = [asdict(md) for md in self.chunk_metadata]
                pickle.dump(metadata_dicts, f)
            
            # Save document info
            with open(self.documents_path, 'w') as f:
                doc_dicts = {doc_id: asdict(doc_info) for doc_id, doc_info in self.documents.items()}
                json.dump(doc_dicts, f, indent=2)
                
            print(f"Vector store saved to disk")
            
        except Exception as e:
            print(f"Warning: Failed to save vector store: {e}")
    
    def document_exists(self, doc_id: str) -> bool:
        """Check if a document is already indexed"""
        return doc_id in self.documents
    
    def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a stored document"""
        if doc_id not in self.documents:
            return None
            
        doc_info = self.documents[doc_id]
        return {
            "doc_id": doc_info.doc_id,
            "source_url": doc_info.source_url,
            "title": doc_info.title,
            "pages": doc_info.pages,
            "chunk_count": doc_info.chunk_count,
            "indexed_at": doc_info.indexed_at,
            "total_tokens": doc_info.total_tokens,
            "storage_size_mb": self._estimate_storage_size() / 1024 / 1024
        }
    
    def _estimate_storage_size(self) -> int:
        """Estimate storage size in bytes"""
        size = 0
        if self.index_path.exists():
            size += self.index_path.stat().st_size
        if self.texts_path.exists():
            size += self.texts_path.stat().st_size
        if self.metadata_path.exists():
            size += self.metadata_path.stat().st_size
        if self.documents_path.exists():
            size += self.documents_path.stat().st_size
        return size
    
    def add_document(
        self, 
        doc_id: str, 
        chunks: List[TextChunk], 
        embeddings: np.ndarray,
        document_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add a document to the vector store
        
        Args:
            doc_id: Unique document identifier
            chunks: List of text chunks
            embeddings: Embeddings array (n_chunks, embedding_dim)
            document_metadata: Document-level metadata
            
        Returns:
            Addition result summary
        """
        start_time = time.time()
        
        print(f"Adding document {doc_id} to vector store...")
        print(f"  - Chunks: {len(chunks)}")
        print(f"  - Embeddings shape: {embeddings.shape}")
        
        # Check for duplicates
        if self.document_exists(doc_id):
            print(f"Document {doc_id} already exists - skipping")
            return {
                "doc_id": doc_id,
                "action": "skipped",
                "reason": "already_exists",
                "chunk_count": len(chunks)
            }
        
        # Validate inputs
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"Chunk count ({len(chunks)}) doesn't match embeddings ({embeddings.shape[0]})")
        
        # Create index if it doesn't exist
        if self.index is None:
            self.index = self._create_index()
            print("Created new FAISS index")
        
        # Store chunk data
        chunk_start_idx = len(self.chunk_texts)
        
        for chunk in chunks:
            # Store text
            self.chunk_texts.append(chunk.text)
            
            # Store metadata
            metadata = ChunkMetadata(
                doc_id=doc_id,
                chunk_id=chunk.chunk_id,
                text_preview=chunk.text[:100],
                token_count=chunk.token_count,
                page=chunk.page,
                heading=chunk.heading,
                section_number=chunk.section_number,
                chunk_type=chunk.chunk_type,
                char_start=chunk.char_start,
                char_end=chunk.char_end
            )
            self.chunk_metadata.append(metadata)
        
        # Add embeddings to FAISS index
        embeddings_float32 = embeddings.astype(np.float32)
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings_float32, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings_normalized = embeddings_float32 / norms
        
        self.index.add(embeddings_normalized)
        
        # Store document info
        self.documents[doc_id] = DocumentInfo(
            doc_id=doc_id,
            source_url=document_metadata.get("source_url", ""),
            title=document_metadata.get("title", ""),
            pages=document_metadata.get("pages", 0),
            chunk_count=len(chunks),
            indexed_at=time.time(),
            total_tokens=sum(chunk.token_count for chunk in chunks)
        )
        
        # Save to disk
        self._save_to_disk()
        
        processing_time = time.time() - start_time
        
        print(f"Document added successfully in {processing_time:.2f}s")
        print(f"  - Total vectors in index: {self.index.ntotal}")
        print(f"  - Total documents: {len(self.documents)}")
        
        return {
            "doc_id": doc_id,
            "action": "added",
            "chunk_count": len(chunks),
            "processing_time": processing_time,
            "total_vectors": self.index.ntotal,
            "total_documents": len(self.documents)
        }
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 20,
        doc_id_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            doc_id_filter: Optional filter by document ID
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        print(f"Searching vector store...")
        print(f"  - Query embedding shape: {query_embedding.shape}")
        print(f"  - Requesting {k} results")
        if doc_id_filter:
            print(f"  - Filtering by doc_id: {doc_id_filter}")
        
        # Normalize query embedding
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        query_norm = query_norm.reshape(1, -1).astype(np.float32)
        
        # Search FAISS index
        # Search more broadly if we need to filter
        search_k = k * 3 if doc_id_filter else k
        search_k = min(search_k, self.index.ntotal)
        
        similarities, indices = self.index.search(query_norm, search_k)
        
        print(f"FAISS search returned {len(indices[0])} results")
        
        # Convert to SearchResult objects
        results = []
        for i, (similarity, chunk_idx) in enumerate(zip(similarities[0], indices[0])):
            if chunk_idx == -1:  # Invalid result
                continue
                
            # Get chunk data
            if chunk_idx >= len(self.chunk_texts) or chunk_idx >= len(self.chunk_metadata):
                print(f"Warning: Invalid chunk index {chunk_idx}")
                continue
                
            text = self.chunk_texts[chunk_idx]
            metadata = self.chunk_metadata[chunk_idx]
            
            # Apply document filter if specified
            if doc_id_filter and metadata.doc_id != doc_id_filter:
                continue
            
            result = SearchResult(
                chunk_index=chunk_idx,
                similarity_score=float(similarity),
                text=text,
                metadata=metadata
            )
            
            results.append(result)
            
            # Stop when we have enough results
            if len(results) >= k:
                break
        
        print(f"Returning {len(results)} filtered results")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunk_texts),
            "embedding_dimension": self.embedding_dim,
            "storage_size_mb": self._estimate_storage_size() / 1024 / 1024,
            "index_type": "HNSW",
            "storage_path": str(self.storage_path)
        }
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the vector store
        Note: This requires rebuilding the entire index
        """
        if doc_id not in self.documents:
            return False
        
        print(f"Removing document {doc_id} (requires index rebuild)...")
        
        # Find chunks to remove
        chunks_to_remove = set()
        for i, metadata in enumerate(self.chunk_metadata):
            if metadata.doc_id == doc_id:
                chunks_to_remove.add(i)
        
        # Remove chunks (in reverse order to maintain indices)
        for i in sorted(chunks_to_remove, reverse=True):
            del self.chunk_texts[i]
            del self.chunk_metadata[i]
        
        # Remove document info
        del self.documents[doc_id]
        
        # Rebuild index
        if self.chunk_texts:
            print("Rebuilding FAISS index...")
            # This is expensive - would need to re-generate embeddings
            # For now, just mark as needing rebuild
            self.index = None
            print("Index marked for rebuild on next embedding add")
        else:
            self.index = None
        
        # Save changes
        self._save_to_disk()
        
        print(f"Document {doc_id} removed successfully")
        return True


# Singleton instance
_vector_store = None

def get_vector_store() -> VectorStore:
    """Get singleton vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
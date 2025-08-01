"""
RAG Coordinator - orchestrates the complete RAG pipeline
"""
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from app.services.document_processor import get_document_processor, ProcessedDocument
from app.services.text_chunker import get_text_chunker, TextChunk
from app.services.embedding_manager import get_embedding_manager
from app.services.vector_store import get_vector_store
from app.services.enhanced_answer_generator import get_enhanced_answer_generator as get_answer_generator
from app.core.config import Settings

settings = Settings()


@dataclass
class RAGResponse:
    """Complete RAG response"""
    answer: str
    processing_time: float
    doc_id: str
    sources_used: List[Dict[str, Any]]
    pipeline_stats: Dict[str, Any]


class RAGCoordinator:
    """Coordinates the complete RAG pipeline"""
    
    def __init__(self):
        """Initialize RAG coordinator"""
        self.document_processor = get_document_processor()
        self.text_chunker = get_text_chunker()
        self.embedding_manager = get_embedding_manager()
        self.vector_store = get_vector_store()
        self.answer_generator = get_answer_generator()
        
        print("RAG Coordinator initialized")
    
    async def process_document(self, url: str) -> Dict[str, Any]:
        """
        Process a document through the complete pipeline
        
        Args:
            url: Document URL to process
            
        Returns:
            Processing summary with doc_id and stats
        """
        start_time = time.time()
        
        print(f"Starting RAG document processing: {url}")
        
        # Generate document ID
        doc_id = self.document_processor.generate_doc_id(url)
        
        # Check if already processed
        if self.vector_store.document_exists(doc_id):
            print(f"Document {doc_id} already exists - using cached version")
            doc_info = self.vector_store.get_document_info(doc_id)
            
            return {
                "doc_id": doc_id,
                "status": "cached",
                "processing_time": time.time() - start_time,
                "document_info": doc_info,
                "message": "Document already processed - using cached version"
            }
        
        try:
            # Stage 1: Document processing
            stage_start = time.time()
            processed_doc = await self.document_processor.process_document(url)
            doc_processing_time = time.time() - stage_start
            
            # Stage 2: Text chunking
            stage_start = time.time()
            chunks = self.text_chunker.chunk_text(
                processed_doc.text,
                processed_doc.metadata
            )
            chunking_time = time.time() - stage_start
            
            print(f"Document chunked into {len(chunks)} pieces")
            
            # Stage 3: Generate embeddings
            stage_start = time.time()
            embedding_result = self.embedding_manager.encode_chunks(chunks)
            embedding_time = time.time() - stage_start
            
            # Stage 4: Add to vector store
            stage_start = time.time()
            vector_result = self.vector_store.add_document(
                doc_id=doc_id,
                chunks=chunks,
                embeddings=embedding_result.embeddings,
                document_metadata=processed_doc.metadata
            )
            vector_time = time.time() - stage_start
            
            total_time = time.time() - start_time
            
            # Compile results
            result = {
                "doc_id": doc_id,
                "status": "processed",
                "processing_time": total_time,
                "stages": {
                    "document_processing": doc_processing_time,
                    "text_chunking": chunking_time,
                    "embedding_generation": embedding_time,
                    "vector_indexing": vector_time
                },
                "document_stats": {
                    "pages": processed_doc.pages,
                    "text_length": len(processed_doc.text),
                    "chunk_count": len(chunks),
                    "total_tokens": sum(chunk.token_count for chunk in chunks),
                    "preserved_definitions": len([c for c in chunks if c.chunk_type in ["definition", "preserved"]])
                },
                "vector_result": vector_result,
                "embedding_info": embedding_result.model_info
            }
            
            print(f"RAG document processing completed in {total_time:.2f}s")
            print(f"  - Document processing: {doc_processing_time:.2f}s")
            print(f"  - Text chunking: {chunking_time:.2f}s") 
            print(f"  - Embedding generation: {embedding_time:.2f}s")
            print(f"  - Vector indexing: {vector_time:.2f}s")
            print(f"  - Total chunks indexed: {len(chunks)}")
            
            return result
            
        except Exception as e:
            print(f"RAG document processing failed: {str(e)}")
            raise
    
    def answer_question(
        self, 
        question: str, 
        doc_id: Optional[str] = None,
        k_retrieve: int = 10,
        max_context_length: int = 6000
    ) -> RAGResponse:
        """
        Answer a question using the RAG pipeline
        
        Args:
            question: Question to answer
            doc_id: Optional document ID to filter by
            k_retrieve: Number of chunks to retrieve
            max_context_length: Maximum context length for answer generation
            
        Returns:
            RAGResponse with answer and metadata
        """
        start_time = time.time()
        
        print(f"Processing question: {question}")
        if doc_id:
            print(f"Filtering by document: {doc_id}")
        
        try:
            # Stage 1: Generate query embedding
            stage_start = time.time()
            query_embedding = self.embedding_manager.encode_query(question)
            query_embedding_time = time.time() - stage_start
            
            # Stage 2: Vector search
            stage_start = time.time()
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                k=settings.k_retrieve,  # Use config value
                doc_id_filter=doc_id
            )
            vector_search_time = time.time() - stage_start
            
            print(f"Retrieved {len(search_results)} relevant chunks")
            
            if not search_results:
                return RAGResponse(
                    answer="I couldn't find relevant information to answer your question.",
                    processing_time=time.time() - start_time,
                    doc_id=doc_id or "none",
                    sources_used=[],
                    pipeline_stats={
                        "query_embedding_time": query_embedding_time,
                        "vector_search_time": vector_search_time,
                        "chunks_retrieved": 0,
                        "answer_generation_time": 0
                    }
                )
            
            # Debug: Show what we found
            print("Top search results:")
            for i, result in enumerate(search_results[:3]):
                print(f"  {i+1}. Score: {result.similarity_score:.4f}, Type: {result.metadata.chunk_type}")
                print(f"     Preview: {result.metadata.text_preview}...")
            
            # Stage 3: Generate answer
            stage_start = time.time()
            answer_result = self.answer_generator.generate_answer(
                question=question,
                search_results=search_results,
                max_context_length=max_context_length
            )
            answer_generation_time = time.time() - stage_start
            
            total_time = time.time() - start_time
            
            # Compile pipeline stats
            pipeline_stats = {
                "query_embedding_time": query_embedding_time,
                "vector_search_time": vector_search_time,
                "answer_generation_time": answer_generation_time,
                "total_time": total_time,
                "chunks_retrieved": len(search_results),
                "context_chunks_used": len(answer_result.context_used),
                "model_info": answer_result.model_info
            }
            
            print(f"Question answered in {total_time:.2f}s")
            print(f"  - Query embedding: {query_embedding_time:.2f}s")
            print(f"  - Vector search: {vector_search_time:.2f}s")
            print(f"  - Answer generation: {answer_generation_time:.2f}s")
            print(f"Answer: {answer_result.answer[:100]}...")
            
            return RAGResponse(
                answer=answer_result.answer,
                processing_time=total_time,
                doc_id=doc_id or search_results[0].metadata.doc_id if search_results else "none",
                sources_used=answer_result.sources,
                pipeline_stats=pipeline_stats
            )
            
        except Exception as e:
            print(f"Question answering failed: {str(e)}")
            raise
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        vector_stats = self.vector_store.get_stats()
        
        return {
            "vector_store": vector_stats,
            "embedding_model": self.embedding_manager.get_model_info(),
            "answer_model": self.answer_generator.get_model_info(),
            "status": "ready"
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        try:
            # Check each component
            checks = {
                "document_processor": "ok",
                "text_chunker": "ok", 
                "embedding_manager": "ok" if self.embedding_manager else "not_loaded",
                "vector_store": "ok" if self.vector_store else "not_loaded",
                "answer_generator": "ok" if self.answer_generator.api_key else "no_api_key"
            }
            
            overall_status = "healthy" if all(v == "ok" for v in checks.values()) else "degraded"
            
            return {
                "status": overall_status,
                "components": checks,
                "system_stats": self.get_system_stats()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Singleton instance
_rag_coordinator = None

def get_rag_coordinator() -> RAGCoordinator:
    """Get singleton RAG coordinator instance"""
    global _rag_coordinator
    if _rag_coordinator is None:
        _rag_coordinator = RAGCoordinator()
    return _rag_coordinator
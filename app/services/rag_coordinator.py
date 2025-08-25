"""
RAG Coordinator - orchestrates the complete RAG pipeline
"""
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
import hashlib

from app.utils.debug import debug_print, info_print, conditional_print

from app.services.document_processor import get_document_processor
from app.services.text_chunker import get_text_chunker
from app.services.embedding_manager import get_embedding_manager
from app.services.vector_store import get_vector_store, SearchResult
from app.services.enhanced_answer_generator import get_enhanced_answer_generator as get_answer_generator
from app.services.lru_cache_manager import get_lru_cache_manager
from app.core.config import Settings

logger = logging.getLogger(__name__)

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
        self.cache_manager = get_lru_cache_manager()
        
        conditional_print("RAG Coordinator initialized with hybrid processing (RAG + Direct) and multi-provider embeddings")
        
        # Pre-warm embedding model for better parallel processing performance
        conditional_print("Pre-warming embedding model for optimal performance...")
        try:
            self.embedding_manager.ensure_model_ready()
            conditional_print("Embedding model pre-warmed successfully")
        except Exception as e:
            conditional_print(f"Warning: Could not pre-warm embedding model: {e}")
            conditional_print("   Model will be loaded on first use")
    
    async def process_document(self, url: str) -> Dict[str, Any]:
        """
        Process a document through the appropriate pipeline (RAG or Direct)
        
        Args:
            url: Document URL to process
            
        Returns:
            Processing summary with doc_id and stats
        """
        start_time = time.time()
        
        conditional_print(f"Starting document processing: {url}")
        
        # Generate document ID
        doc_id = self.document_processor.generate_doc_id(url)
        
        # Check if already processed in vector store
        if self.vector_store.document_exists(doc_id):
            conditional_print(f"Document {doc_id} already exists in vector store - using cached version")
            doc_info = self.vector_store.get_document_info(doc_id)
            
            return {
                "doc_id": doc_id,
                "status": "cached",
                "processing_time": time.time() - start_time,
                "document_info": doc_info,
                "message": "Document already processed - using cached version",
                "pipeline_used": "vector_store_cached"
            }
        
        # Process the document
        try:
            processed_doc = await self.document_processor.process_document(url)
            
            # Use RAG pipeline for all documents
            return await self._process_document_rag(url, processed_doc, start_time)
                
        except Exception as e:
            print(f"Document processing failed: {str(e)}")
            raise
    
    
    async def _process_document_rag(self, url: str, processed_doc, start_time: float) -> Dict[str, Any]:
        """Process document using traditional RAG pipeline"""
        conditional_print("Using RAG processing pipeline (chunking + vector search)")
        
        doc_id = self.document_processor.generate_doc_id(url)
        
        try:
            # Stage 1: Text chunking with dual-language support
            stage_start = time.time()
            
            # Create chunks from original text
            original_chunks = self.text_chunker.chunk_text(
                processed_doc.text,
                processed_doc.metadata
            )
            
            # Create chunks from translated text if available
            translated_chunks = []
            if processed_doc.translated_text:
                translated_chunks = self.text_chunker.chunk_text(
                    processed_doc.translated_text,
                    {**processed_doc.metadata, "text_version": "translated"}
                )
            
            chunking_time = time.time() - stage_start
            
            conditional_print(f"Document chunked: {len(original_chunks)} original chunks")
            if translated_chunks:
                conditional_print(f"Document chunked: {len(translated_chunks)} translated chunks")
            
            # Store both chunk versions in metadata for language-aware retrieval
            chunks = original_chunks
            
            # Add translated versions to chunks if available
            if translated_chunks:
                # Ensure both have same number of chunks (should be similar)
                min_chunks = min(len(original_chunks), len(translated_chunks))
                for i in range(min_chunks):
                    if i < len(original_chunks):
                        # Add translation fields to the TextChunk
                        translation_text = translated_chunks[i].text if i < len(translated_chunks) else None
                        original_chunks[i].translated_text = translation_text
                        original_chunks[i].source_language = processed_doc.detected_language
                        original_chunks[i].has_translation = True
                        
                        # Combine original and translated text for better semantic search
                        if translation_text:
                            # Create combined searchable content
                            original_chunks[i].text = f"{original_chunks[i].text}\n\n--- English Translation ---\n{translation_text}"
                            print(f"Enhanced chunk {i} with translation for semantic search")
            
            conditional_print(f"Document chunked into {len(chunks)} pieces")
            
            # Handle zero chunks case (prevents division by zero errors)
            if len(chunks) == 0:
                conditional_print("WARNING: Document produced zero chunks - this will cause processing to fail")
                conditional_print("Creating minimal chunk to prevent system failure")
                
                # Create a minimal chunk with error information
                from app.services.text_chunker import TextChunk
                error_chunk = TextChunk(
                    text=processed_doc.text[:500] + "..." if len(processed_doc.text) > 500 else processed_doc.text,
                    start_index=0,
                    end_index=len(processed_doc.text),
                    chunk_type="error_content",
                    metadata={
                        **processed_doc.metadata,
                        'chunk_status': 'minimal_fallback',
                        'warning': 'Original document produced no valid chunks'
                    }
                )
                chunks = [error_chunk]
                conditional_print(f"Created fallback chunk with {len(error_chunk.text)} characters")
            
            # Stage 2: Generate embeddings
            stage_start = time.time()
            embedding_result = await self.embedding_manager.encode_chunks(chunks)
            embedding_time = time.time() - stage_start
            
            # Stage 3: Add to vector store with translation metadata
            stage_start = time.time()
            
            # Prepare enhanced document metadata with translation info
            enhanced_doc_metadata = {
                **processed_doc.metadata,
                "has_translation": processed_doc.translated_text is not None,
                "detected_language": processed_doc.detected_language,
                "document_type": "semantic_search"
            }
            
            vector_result = self.vector_store.add_document(
                doc_id=doc_id,
                chunks=chunks,
                embeddings=embedding_result.embeddings,
                document_metadata=enhanced_doc_metadata
            )
            vector_time = time.time() - stage_start
            
            total_time = time.time() - start_time
            
            # Compile results
            result = {
                "doc_id": doc_id,
                "status": "processed_rag",
                "processing_time": total_time,
                "pipeline_used": "rag_pipeline",
                "document_type": "semantic_search",
                "stages": {
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
                "embedding_info": embedding_result.model_info,
                "message": "Document processed using RAG pipeline"
            }
            
            conditional_print(f"RAG document processing completed in {total_time:.2f}s")
            conditional_print(f"  - Text chunking: {chunking_time:.2f}s") 
            conditional_print(f"  - Embedding generation: {embedding_time:.2f}s")
            conditional_print(f"  - Vector indexing: {vector_time:.2f}s")
            conditional_print(f"  - Total chunks indexed: {len(chunks)}")
            
            return result
            
        except Exception as e:
            print(f"RAG document processing failed: {str(e)}")
            raise

    async def answer_question(
        self, 
        question: str, 
        doc_id: Optional[str] = None,
        k_retrieve: int = 10,
        max_context_length: int = None
    ) -> RAGResponse:
        """
        Answer a question using the RAG pipeline
        
        Args:
            question: Question to answer
            doc_id: Optional document ID to filter by
            k_retrieve: Number of chunks to retrieve
            max_context_length: Maximum context length for answer generation (None = use config default)
            
        Returns:
            RAGResponse with answer and metadata
        """
        start_time = time.time()
        
        # Check answer cache first
        should_cache = False
        if settings.enable_answer_cache and doc_id:
            doc_info = self.vector_store.get_document_info(doc_id)
            if doc_info:
                doc_type = doc_info.get("document_type")
                if doc_type == "semantic_search":
                    should_cache = True
        
        if should_cache:
            # Normalize cache key to ignore retrieval parameters for better cache hit rate  
            normalized_question = question.lower().strip()
            # Use MD5 hash for consistent cache keys across server restarts
            hash_value = hashlib.md5(normalized_question.encode()).hexdigest()[:16]
            cache_key = f"qa:{doc_id}:{hash_value}"
            cached_response = await self.cache_manager.get_answer_cache(cache_key)
            
            if cached_response:
                print(f"Cache hit for question: {question[:50]}...")
                # Update processing time to reflect cache retrieval
                cached_response.processing_time = time.time() - start_time
                return cached_response
        
        conditional_print(f"Processing question: {question}")
        if doc_id:
            print(f"Filtering by document: {doc_id}")
        
        try:
            # Stage 1: Embedding and Search
            stage_start = time.time()
            
            # Generate query embedding
            query_embedding = await self.embedding_manager.encode_query(question)
            
            # Vector search
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                k=k_retrieve,
                doc_id_filter=doc_id
            )
            
            embedding_and_search_time = time.time() - stage_start
            
            print(f"Retrieved {len(search_results)} chunks")
            
            if not search_results:
                total_time = time.time() - start_time
                return RAGResponse(
                    answer="I couldn't find relevant information to answer your question.",
                    processing_time=total_time,
                    doc_id=doc_id or "none",
                    sources_used=[],
                    pipeline_stats={
                        "embedding_and_search_time": embedding_and_search_time,
                        "chunks_retrieved": 0,
                        "answer_generation_time": 0,
                        "total_time": total_time
                    }
                )
            
            # Debug output
            print("Top search results (L2 distance - lower = more similar):")
            for i, result in enumerate(search_results[:3]):
                # Convert L2 distance to cosine similarity for normalized embeddings: cos_sim = 1 - (L2²/2)
                cosine_sim = 1 - (result.similarity_score ** 2) / 2
                print(f"  {i+1}. Distance: {result.similarity_score:.4f} (≈{cosine_sim:.1%} similar)")
            
            # Stage 2: Generate answer
            stage_start = time.time()
            # Use config default if not specified
            context_length = max_context_length if max_context_length is not None else settings.max_context_tokens
            
            answer_result = await self.answer_generator.generate_answer(
                question=question,
                search_results=search_results,
                max_context_length=context_length
            )
            answer_generation_time = time.time() - stage_start
            
            total_time = time.time() - start_time
            
            # Pipeline stats
            pipeline_stats = {
                "embedding_and_search_time": embedding_and_search_time,
                "answer_generation_time": answer_generation_time,
                "total_time": total_time,
                "chunks_retrieved": len(search_results),
                "context_chunks_used": len(answer_result.context_used),
                "model_info": answer_result.model_info
            }
            
            print(f"Question answered in {total_time:.2f}s")
            print(f"  - Embedding & search: {embedding_and_search_time:.2f}s")
            print(f"  - Answer generation: {answer_generation_time:.2f}s")
            print(f"Answer: {answer_result.answer[:100]}...")
            
            # Prepare response
            response = RAGResponse(
                answer=answer_result.answer,
                processing_time=total_time,
                doc_id=doc_id or search_results[0].metadata.doc_id if search_results else "none",
                sources_used=answer_result.sources,
                pipeline_stats=pipeline_stats
            )
            
            # Cache the response for future use
            if should_cache:
                try:
                    await self.cache_manager.set_answer_cache(cache_key, response)
                    print("Cached answer for future requests")
                except Exception as e:
                    print(f"Warning: Failed to cache answer: {e}")
            
            return response
            
        except Exception as e:
            print(f"Question answering failed: {str(e)}")
            raise

    async def answer_question_async(
        self, 
        question: str, 
        doc_id: Optional[str] = None,
        k_retrieve: int = 10,
        max_context_length: int = None
    ) -> RAGResponse:
        """
        Async version of answer_question
        
        Args:
            question: Question to answer
            doc_id: Optional document ID to filter by
            k_retrieve: Number of chunks to retrieve
            max_context_length: Maximum context length for answer generation (None = use config default)
            
        Returns:
            RAGResponse with answer and metadata
        """
        return await self.answer_question(
            question=question,
            doc_id=doc_id,
            k_retrieve=k_retrieve,
            max_context_length=max_context_length
        )
    
    async def process_questions_individually(
        self,
        questions: List[str],
        doc_id: Optional[str] = None,
        k_retrieve: int = None,
        max_context_length: int = None
    ) -> List[RAGResponse]:
        """
        Process multiple questions individually
        
        Args:
            questions: List of questions to answer
            doc_id: Optional document ID to filter by  
            k_retrieve: Number of chunks to retrieve per query
            max_context_length: Maximum context length for answer generation
            
        Returns:
            List of RAGResponse objects
        """
        if not questions:
            return []
        
        if k_retrieve is None:
            k_retrieve = settings.k_retrieve
        if max_context_length is None:
            max_context_length = settings.max_context_tokens
            
        debug_print(f"Processing {len(questions)} questions individually")
        start_time = time.time()
        
        responses = []
        for i, question in enumerate(questions):
            debug_print(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
            
            response = await self.answer_question(
                question=question,
                doc_id=doc_id,
                k_retrieve=k_retrieve,
                max_context_length=max_context_length
            )
            
            responses.append(response)
            debug_print(f"  Question {i+1} completed in {response.processing_time:.2f}s")
        
        total_time = time.time() - start_time
        info_print(f"✓ Individual processing completed in {total_time:.2f}s")
        debug_print(f"  Average per question: {total_time/len(questions):.2f}s")
        
        return responses
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics with performance metrics"""
        vector_stats = self.vector_store.get_stats()
        cache_stats = self.cache_manager.get_cache_stats()
        
        return {
            "vector_store": vector_stats,
            "embedding_model": self.embedding_manager.get_model_info(),
            "answer_model": self.answer_generator.get_model_info(),
            "cache_performance": cache_stats,
            "optimization_settings": {
                "fast_mode": settings.fast_mode,
                "performance_mode": settings.performance_mode,
                "result_caching": settings.enable_result_caching,
                "embedding_caching": settings.enable_embedding_cache,
                "k_retrieve": settings.k_retrieve,
                "max_tokens": settings.llm_max_tokens
            },
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
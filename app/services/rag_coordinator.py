"""
RAG Coordinator - orchestrates the complete RAG pipeline with query transformation
"""
import asyncio
import time
import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
import logging
import re

from app.utils.debug import debug_print, info_print, conditional_print

from app.services.document_processor import get_document_processor, ProcessedDocument
from app.services.text_chunker import get_text_chunker, TextChunk
from app.services.embedding_manager import get_embedding_manager
from app.services.vector_store import get_vector_store, SearchResult
from app.services.enhanced_answer_generator import get_enhanced_answer_generator as get_answer_generator
from app.services.cache_manager import get_cache_manager
from app.services.query_transformer import get_query_transformer, QueryTransformationResult
from app.services.challenge_detector import get_challenge_detector, ChallengeType
from app.services.challenge_solver import get_challenge_solver
from app.services.web_client import WebClient
from app.services.document_type_detector import get_document_type_detector, DocumentType
from app.services.direct_processor import get_direct_processor
from app.core.config import Settings

logger = logging.getLogger(__name__)

settings = Settings()


@dataclass
class RAGResponse:
    """Complete RAG response with intelligent challenge detection"""
    answer: str
    processing_time: float
    doc_id: str
    sources_used: List[Dict[str, Any]]
    pipeline_stats: Dict[str, Any]
    query_transformation: Optional[Dict[str, Any]] = None  # Information about query transformation
    challenge_detection: Optional[Dict[str, Any]] = None  # Information about detected challenge type
    special_processing: Optional[Dict[str, Any]] = None   # Information about special processing used


class RAGCoordinator:
    """Coordinates the complete RAG pipeline"""
    
    def __init__(self):
        """Initialize RAG coordinator"""
        self.document_processor = get_document_processor()
        self.text_chunker = get_text_chunker()
        self.embedding_manager = get_embedding_manager()
        self.vector_store = get_vector_store()
        self.answer_generator = get_answer_generator()
        self.cache_manager = get_cache_manager()
        self.query_transformer = get_query_transformer()
        self.challenge_detector = get_challenge_detector()
        self.document_type_detector = get_document_type_detector()
        self.direct_processor = get_direct_processor()
        
        # Initialize challenge solver (lazy loaded when needed)
        self._challenge_solver = None
        
        # Single embedding manager based on config
        
        conditional_print("RAG Coordinator initialized with hybrid processing (RAG + Direct), query transformation, and multi-provider embeddings")
        
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
        
        conditional_print(f"Starting hybrid document processing: {url}")
        
        # Check if this URL is actually a token API endpoint, not a document
        if self._is_token_api_endpoint(url):
            conditional_print(f"Detected token API endpoint, not a document: {url}")
            # Return a special response indicating this should be handled as web scraping
            return {
                "doc_id": None,
                "status": "token_api_detected",
                "processing_time": time.time() - start_time,
                "message": "URL detected as token API endpoint, not a document",
                "pipeline_used": "token_api_detection",
                "requires_web_scraping": True,
                "api_url": url
            }
        
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
        
        # First, get the document text to determine processing approach
        try:
            processed_doc = await self.document_processor.process_document(url)
            
            # Detect document type to choose processing pipeline
            type_result = self.document_type_detector.detect_document_type(
                processed_doc.text,
                url
            )
            
            conditional_print(f"Document type detected: {type_result.document_type.value} "
                  f"(confidence: {type_result.confidence:.2f})")
            conditional_print(f"Suggested pipeline: {type_result.suggested_pipeline}")
            
            # Route to appropriate processing pipeline
            if type_result.suggested_pipeline == "direct_processing":
                return await self._process_document_direct(url, processed_doc, type_result, start_time)
            else:
                return await self._process_document_rag(url, processed_doc, type_result, start_time)
                
        except Exception as e:
            print(f"Hybrid document processing failed: {str(e)}")
            raise
    
    async def _process_document_direct(self, url: str, processed_doc, type_result, start_time: float) -> Dict[str, Any]:
        """Process document using direct processing pipeline"""
        conditional_print("Using DIRECT processing pipeline (no vector search)")
        
        doc_id = self.document_processor.generate_doc_id(url)
        
        # Use direct processor
        direct_result = await self.direct_processor.process_document_direct(url)
        
        total_time = time.time() - start_time
        
        return {
            "doc_id": doc_id,
            "status": "processed_direct",
            "processing_time": total_time,
            "pipeline_used": "direct_processing",
            "document_type": direct_result.document_type.value,
            "type_confidence": type_result.confidence,
            "document_stats": direct_result.metadata["document_stats"],
            "landmark_mappings": {
                "total_landmarks": sum(len(mappings) for mappings in direct_result.landmark_mappings.values()),
                "cities_covered": len(direct_result.landmark_mappings),
                "cities": list(direct_result.landmark_mappings.keys())
            },
            "message": f"Document processed using direct pipeline for {direct_result.document_type.value}"
        }
    
    async def _process_document_rag(self, url: str, processed_doc, type_result, start_time: float) -> Dict[str, Any]:
        """Process document using traditional RAG pipeline"""
        conditional_print("Using RAG processing pipeline (chunking + vector search)")
        
        doc_id = self.document_processor.generate_doc_id(url)
        
        try:
            # Stage 1: Document processing (already done)
            stage_start = time.time()
            # Use already processed document
            # processed_doc is already available from caller
            
            # Stage 2: Text chunking with dual-language support
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
            
            # Stage 3: Generate embeddings
            stage_start = time.time()
            embedding_result = await self.embedding_manager.encode_chunks(chunks)
            embedding_time = time.time() - stage_start
            
            # Stage 4: Add to vector store with translation metadata
            stage_start = time.time()
            
            # Prepare enhanced document metadata with translation info
            enhanced_doc_metadata = {
                **processed_doc.metadata,
                "has_translation": processed_doc.translated_text is not None,
                "detected_language": processed_doc.detected_language
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
                "document_type": type_result.document_type.value,
                "type_confidence": type_result.confidence,
                "stages": {
                    "document_processing": 0.0,  # Already done
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
                "message": f"Document processed using RAG pipeline for {type_result.document_type.value}"
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
    
    def _is_token_api_endpoint(self, url: str) -> bool:
        """
        Check if a URL is a token API endpoint rather than a document to process
        
        Args:
            url: URL to check
            
        Returns:
            True if this appears to be a token API endpoint
        """
        # Check for token-related keywords in the URL
        token_indicators = [
            'get-secret-token',
            'secret-token', 
            'token',
            '/api/',
            '/utils/',
            'auth',
            'credential'
        ]
        
        url_lower = url.lower()
        
        # Check for token API patterns
        for indicator in token_indicators:
            if indicator in url_lower:
                # Additional check for common document extensions - if present, likely a document
                document_extensions = ['.pdf', '.doc', '.docx', '.txt', '.json', '.xml']
                has_doc_extension = any(ext in url_lower for ext in document_extensions)
                
                if not has_doc_extension:
                    return True
        
        # Check for hackrx specific endpoints
        if 'hackrx.in' in url_lower and ('utils' in url_lower or 'token' in url_lower):
            return True
        
        return False
    
    def _get_complete_document_text(self, doc_id: str) -> Optional[str]:
        """
        Get complete document text for direct processing
        
        Args:
            doc_id: Document ID
            
        Returns:
            Complete document text or None if not available
        """
        try:
            # Check if we have the document in parsed text storage
            import os
            from glob import glob
            
            # Look for parsed document files
            parsed_files = glob(f"parsed_documents/parsed_*_{doc_id[:8]}*.txt")
            
            if parsed_files:
                # Read the most recent parsed file
                parsed_file = max(parsed_files, key=os.path.getmtime)
                print(f"Found parsed document file: {parsed_file}")
                
                with open(parsed_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Extract just the document text (skip metadata)
                    if "PARSED TEXT CONTENT" in content:
                        text_start = content.find("PARSED TEXT CONTENT")
                        text_start = content.find("\n", text_start) + 1
                        document_text = content[text_start:].strip()
                        
                        print(f"Loaded complete document text: {len(document_text)} characters")
                        return document_text
            
            # Fallback: reconstruct from vector store chunks
            if hasattr(self, 'vector_store') and self.vector_store and doc_id in self.vector_store.documents:
                print(f"Reconstructing document text from vector store chunks for {doc_id}")
                
                doc_chunks = []
                for i in range(len(self.vector_store.chunk_texts)):
                    metadata = self.vector_store.chunk_metadata[i]
                    if metadata.doc_id == doc_id:
                        doc_chunks.append(self.vector_store.chunk_texts[i])
                
                if doc_chunks:
                    full_text = "\n".join(doc_chunks)
                    print(f"Reconstructed document text: {len(full_text)} characters from {len(doc_chunks)} chunks")
                    return full_text
            
            print(f"No complete document text available for {doc_id}")
            return None
            
        except Exception as e:
            print(f"Error getting complete document text for {doc_id}: {e}")
            return None
    
    def _deduplicate_search_results(
        self,
        all_results: List[SearchResult],
        similarity_threshold: float = 0.85
    ) -> List[SearchResult]:
        """
        Fast hash-based deduplication for search results
        
        Args:
            all_results: All search results from multiple queries
            similarity_threshold: Not used in this optimized version
            
        Returns:
            Deduplicated search results sorted by relevance score
        """
        if len(all_results) <= 1:
            return all_results
        
        # Use hash-based deduplication for O(n) performance
        seen_exact: Set[Tuple[str, int, int]] = set()  # Exact chunk matches
        seen_overlapping = {}  # For overlap detection: {doc_id: [(start, end, result)]}
        deduplicated = []
        
        # Sort by similarity score (ascending) - FAISS IndexHNSWFlat uses L2 distance (smaller = more similar)
        sorted_results = sorted(all_results, key=lambda x: x.similarity_score)
        
        for result in sorted_results:
            chunk_id = (result.metadata.doc_id, result.metadata.char_start, result.metadata.char_end)
            
            # Quick exact match check
            if chunk_id in seen_exact:
                continue
            
            # Check for overlaps within the same document
            doc_id = result.metadata.doc_id
            is_duplicate = False
            
            if doc_id in seen_overlapping:
                current_start, current_end = result.metadata.char_start, result.metadata.char_end
                current_length = current_end - current_start
                
                # Check against existing chunks in this document
                for existing_start, existing_end, _ in seen_overlapping[doc_id]:
                    # Quick range overlap check
                    if current_end <= existing_start or current_start >= existing_end:
                        continue  # No overlap
                    
                    # Calculate overlap ratio
                    overlap_start = max(existing_start, current_start)
                    overlap_end = min(existing_end, current_end)
                    overlap_length = overlap_end - overlap_start
                    overlap_ratio = overlap_length / current_length if current_length > 0 else 0
                    
                    if overlap_ratio > 0.7:  # 70% overlap threshold
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                deduplicated.append(result)
                seen_exact.add(chunk_id)
                
                # Track for overlap detection
                if doc_id not in seen_overlapping:
                    seen_overlapping[doc_id] = []
                seen_overlapping[doc_id].append((result.metadata.char_start, result.metadata.char_end, result))
        
        logger.info(f"Deduplication: {len(all_results)} → {len(deduplicated)} chunks")
        return deduplicated

    async def answer_question(
        self, 
        question: str, 
        doc_id: Optional[str] = None,
        k_retrieve: int = 10,
        max_context_length: int = None,
        enable_transformation: Optional[bool] = None  # Keep for compatibility but ignore
    ) -> RAGResponse:
        """
        Intelligent question answering with automatic challenge detection and caching
        
        Args:
            question: Question to answer
            doc_id: Optional document ID to filter by
            k_retrieve: Number of chunks to retrieve
            max_context_length: Maximum context length for answer generation (None = use config default)
            enable_transformation: Ignored - no transformation used
            
        Returns:
            RAGResponse with answer and metadata including challenge detection
        """
        start_time = time.time()
        
        # Check answer cache first - only for configured cacheable documents
        should_cache = (settings.enable_answer_cache and doc_id and 
                       any(cacheable_id in str(doc_id) for cacheable_id in settings.cacheable_document_ids))
        
        if should_cache:
            cache_key = f"qa:{doc_id}:{hash(question)}:{k_retrieve}:{max_context_length}"
            cached_response = await self.cache_manager.get_answer_cache(cache_key)
            
            if cached_response:
                print(f"✅ Cache hit for News.pdf question: {question[:50]}...")
                # Update processing time to reflect cache retrieval
                cached_response.processing_time = time.time() - start_time
                return cached_response
        
        conditional_print(f"Processing question: {question}")
        if doc_id:
            print(f"Filtering by document: {doc_id}")
        
        try:
            # Step 1: Get document content for context (if doc_id specified)
            document_content = ""
            if doc_id:
                # Try to get complete document text first (for direct processed docs)
                complete_text = self._get_complete_document_text(doc_id)
                
                if complete_text:
                    # Use first 2000 chars of complete text for challenge detection
                    document_content = complete_text[:20000]
                    print(f"Using complete document text for challenge detection ({len(document_content)} chars)")
                else:
                    # Fallback: Extract document content from vector store chunks
                    doc_chunks = []
                    for i in range(len(self.vector_store.chunk_texts)):
                        metadata = self.vector_store.chunk_metadata[i]
                        if metadata.doc_id == doc_id:
                            doc_chunks.append(self.vector_store.chunk_texts[i])
                    document_content = " ".join(doc_chunks[:5])  # First 5 chunks for context
                    print(f"Using vector store chunks for challenge detection ({len(document_content)} chars)")
            
            # Step 2: Detect challenge type
            detections = self.challenge_detector.detect_challenge_type([question], document_content)
            primary_challenge = self.challenge_detector.get_primary_challenge(detections)
            
            print(f"Challenge detected: {primary_challenge.challenge_type.value} (confidence: {primary_challenge.confidence:.2f})")
            print(f"Suggested approach: {primary_challenge.suggested_approach}")
            
            challenge_info = {
                "type": primary_challenge.challenge_type.value,
                "confidence": primary_challenge.confidence,
                "patterns": primary_challenge.detected_patterns,
                "approach": primary_challenge.suggested_approach,
                "metadata": primary_challenge.metadata
            }
            
            # Step 3: Route to appropriate handler based on challenge type
            special_processing = None
            
            if primary_challenge.challenge_type == ChallengeType.GEOGRAPHIC_PUZZLE and primary_challenge.confidence > 0.6:
                # Use geographic challenge solver
                return await self._handle_geographic_challenge(question, doc_id, start_time, challenge_info)
            
            elif primary_challenge.challenge_type == ChallengeType.FLIGHT_NUMBER and primary_challenge.confidence > 0.8:
                # Try to extract flight number or trigger geographic solver
                return await self._handle_flight_request(question, doc_id, start_time, challenge_info)
            
            elif primary_challenge.challenge_type == ChallengeType.SECRET_TOKEN and primary_challenge.confidence > 0.7:
                # Use web client to get token
                return await self._handle_token_request(question, doc_id, start_time, challenge_info)
            
            elif primary_challenge.challenge_type == ChallengeType.WEB_SCRAPING and primary_challenge.confidence > 0.6:
                # Use web scraping
                return await self._handle_web_scraping(question, doc_id, start_time, challenge_info)
            
            elif primary_challenge.challenge_type == ChallengeType.MULTILINGUAL_QA and primary_challenge.confidence > 0.5:
                # Enhanced multilingual processing - but only if the QUESTION is multilingual
                detected_lang = primary_challenge.metadata.get("language_detected", "unknown")
                question_languages = primary_challenge.metadata.get("question_languages", {})
                
                # Only use multilingual mode if question itself contains non-English content
                if question_languages and "english" not in question_languages:
                    special_processing = {
                        "multilingual_mode": True,
                        "detected_language": detected_lang,
                        "language_processing": True
                    }
                    print(f"Using enhanced multilingual processing for {detected_lang} content")
                else:
                    # Question is in English, use standard processing even if document is multilingual
                    print(f"Question is in English, using standard processing (document contains {detected_lang})")
            
            elif primary_challenge.challenge_type == ChallengeType.COMPLEX_REASONING and primary_challenge.confidence > 0.5:
                # Enhanced reasoning with more context
                k_retrieve = min(k_retrieve * 2, 50)  # Increase context for complex reasoning
                special_processing = {"complex_reasoning_mode": True}
            
            # Step 4: Use standard RAG pipeline with any special processing
            # SIMPLIFIED: Use original question directly - no transformation
            queries_to_process = [question]
            
            # Remove query transformation completely
            query_transformation_result = {
                "enabled": False,
                "successful": False,
                "original_query": question,
                "sub_queries": [question],
                "processing_time": 0.0,
                "error": "Query transformation disabled for simplicity"
            }
            
            # SIMPLIFIED Stage 2: Single Query Embedding and Search (no parallel processing)
            stage_start = time.time()
            
            # Simple single query processing - no caching complexity
            query_embedding = await self.embedding_manager.encode_query(question)
            
            # Determine if this is multilingual content for enhanced retrieval
            is_multilingual = (primary_challenge.challenge_type == ChallengeType.MULTILINGUAL_QA or
                              (document_content and bool(re.search(r'[\u0D00-\u0D7F]', document_content))))
            
            # Use enhanced parameters for multilingual content
            if is_multilingual and settings.enable_multilingual_enhancement:
                k_search = settings.multilingual_k_retrieve
                print(f"Using enhanced multilingual retrieval: k={k_search} (document contains Malayalam text)")
            else:
                k_search = settings.k_retrieve
            
            # Simple vector search with dynamic k based on content type
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                k=k_search,
                doc_id_filter=doc_id
            )
            
            print(f"Retrieved {len(search_results)} chunks")
            
            # No deduplication needed - single query results
            deduplicated_results = search_results
            
            embedding_and_search_time = time.time() - stage_start
            
            print(f"Retrieved {len(deduplicated_results)} chunks")
            
            if not deduplicated_results:
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
                    },
                    query_transformation=query_transformation_result,
                    challenge_detection=challenge_info,
                    special_processing=special_processing
                )
            
            # Simple debug output
            print("Top search results (L2 distance - lower = more similar):")
            for i, result in enumerate(deduplicated_results[:3]):
                # Convert L2 distance to cosine similarity for normalized embeddings: cos_sim = 1 - (L2²/2)
                cosine_sim = 1 - (result.similarity_score ** 2) / 2
                print(f"  {i+1}. Distance: {result.similarity_score:.4f} (≈{cosine_sim:.1%} similar)")
            
            # SIMPLIFIED Stage 3: Generate answer
            stage_start = time.time()
            # Use config default if not specified
            context_length = max_context_length if max_context_length is not None else settings.max_context_tokens
            
            # Language-aware context selection: Use translated text for English questions
            question_is_english = not bool(re.search(r'[\u0900-\u097F\u0D00-\u0D7F\u0B80-\u0BFF\u0980-\u09FF\u0C00-\u0C7F]', question))
            
            # Create language-appropriate search results  
            context_results = []
            for result in deduplicated_results:
                # For now, use original results - translation will be handled at document level
                # TODO: Implement proper chunk-level translation storage in vector store
                context_results.append(result)
            
            print(f"Language-aware context: Using {'translated' if question_is_english else 'original'} text for {'English' if question_is_english else 'native'} question")
            
            # Pass multilingual info to answer generator if available
            multilingual_mode = special_processing and special_processing.get("multilingual_mode", False)
            detected_language = special_processing and special_processing.get("detected_language")
            
            answer_result = await self.answer_generator.generate_answer(
                question=question,
                search_results=context_results,  # Use language-appropriate results
                max_context_length=context_length,
                multilingual_mode=multilingual_mode,
                detected_language=detected_language
            )
            answer_generation_time = time.time() - stage_start
            
            total_time = time.time() - start_time
            
            # Simplified pipeline stats
            pipeline_stats = {
                "embedding_and_search_time": embedding_and_search_time,
                "answer_generation_time": answer_generation_time,
                "total_time": total_time,
                "chunks_retrieved": len(deduplicated_results),
                "context_chunks_used": len(answer_result.context_used),
                "model_info": answer_result.model_info
            }
            
            print(f"Enhanced question answered in {total_time:.2f}s")
            if query_transformation_result["enabled"]:
                print(f"  - Query transformation: {query_transformation_result['processing_time']:.2f}s")
            print(f"  - Embedding & search: {embedding_and_search_time:.2f}s")
            print(f"  - Answer generation: {answer_generation_time:.2f}s")
            print(f"Answer: {answer_result.answer[:100]}...")
            
            # Prepare response
            response = RAGResponse(
                answer=answer_result.answer,
                processing_time=total_time,
                doc_id=doc_id or deduplicated_results[0].metadata.doc_id if deduplicated_results else "none",
                sources_used=answer_result.sources,
                pipeline_stats=pipeline_stats,
                query_transformation=query_transformation_result,
                challenge_detection=challenge_info,
                special_processing=special_processing
            )
            
            # Cache the response for future use - ONLY for News.pdf document
            if should_cache:
                try:
                    await self.cache_manager.set_answer_cache(cache_key, response)
                    print(f"📝 Cached News.pdf answer for future requests")
                except Exception as e:
                    print(f"⚠ Warning: Failed to cache answer: {e}")
            
            return response
            
        except Exception as e:
            print(f"Enhanced question answering failed: {str(e)}")
            raise
    
    async def _handle_geographic_challenge(self, question: str, doc_id: str, start_time: float, challenge_info: Dict[str, Any]) -> RAGResponse:
        """Handle geographic puzzle challenge with multi-step solving"""
        conditional_print("Handling geographic puzzle challenge...")
        
        try:
            # Get challenge solver instance
            if self._challenge_solver is None:
                from app.services.challenge_solver import get_challenge_solver
                self._challenge_solver = await get_challenge_solver()
            
            # Try to get complete document text for direct processing
            document_text = self._get_complete_document_text(doc_id)
            
            # Solve the geographic challenge - use direct method if text available
            if document_text:
                print("Using direct geographic challenge solving with complete document text")
                challenge_result = await self._challenge_solver.solve_geographic_challenge_direct(
                    doc_id, document_text
                )
            else:
                print("Using legacy geographic challenge solving with vector store chunks")
                challenge_result = await self._challenge_solver.solve_geographic_challenge(doc_id)
            
            # Extract the answer - handle multiple flight numbers
            if challenge_result.flight_numbers:
                if len(challenge_result.flight_numbers) == 1:
                    answer = f"Flight Number: {challenge_result.flight_numbers[0]}"
                else:
                    answer = f"Flight Numbers: {', '.join(challenge_result.flight_numbers)}"
            else:
                # If challenge failed, provide detailed error info
                failed_steps = [step for step in challenge_result.steps_completed if not step.success]
                if failed_steps:
                    last_error = failed_steps[-1].error
                    answer = f"Geographic challenge failed at step {failed_steps[-1].step.value}: {last_error}"
                else:
                    answer = "Geographic challenge completed but no flight numbers were obtained."
            
            # Create sources from challenge steps
            sources_used = []
            for step in challenge_result.steps_completed:
                if step.success and step.data:
                    sources_used.append({
                        "step": step.step.value,
                        "data": step.data,
                        "processing_time": step.processing_time
                    })
            
            total_time = time.time() - start_time
            
            # Create special processing metadata
            special_processing = {
                "challenge_type": "geographic_puzzle",
                "multi_step_process": True,
                "steps_completed": len(challenge_result.steps_completed),
                "successful_steps": len([s for s in challenge_result.steps_completed if s.success]),
                "challenge_processing_time": challenge_result.total_processing_time,
                "final_result": {
                    "city": challenge_result.city,
                    "landmarks": [landmark.landmark for landmark in challenge_result.landmarks] if challenge_result.landmarks else [],
                    "flight_numbers": challenge_result.flight_numbers,
                    "landmark_count": len(challenge_result.landmarks),
                    "flight_count": len(challenge_result.flight_numbers),
                    "has_token": challenge_result.secret_token is not None
                }
            }
            
            return RAGResponse(
                answer=answer,
                processing_time=total_time,
                doc_id=doc_id,
                sources_used=sources_used,
                pipeline_stats={
                    "challenge_solving_time": challenge_result.total_processing_time,
                    "total_time": total_time,
                    "steps_completed": len(challenge_result.steps_completed),
                    "model_info": {"challenge_solver": "geographic_puzzle"}
                },
                challenge_detection=challenge_info,
                special_processing=special_processing
            )
            
        except Exception as e:
            print(f"Geographic challenge handling failed: {e}")
            # Fallback to standard RAG
            return await self._fallback_to_standard_rag(question, doc_id, start_time, challenge_info, f"Geographic challenge failed: {e}")
    
    async def _handle_flight_request(self, question: str, doc_id: str, start_time: float, challenge_info: Dict[str, Any]) -> RAGResponse:
        """Handle direct flight number requests"""
        conditional_print("Handling flight number request...")
        
        try:
            # First try to extract flight number directly from document using RAG
            rag_response = await self._standard_rag_query(question, doc_id, k_retrieve=15)
            
            # Check if the answer contains a flight number pattern
            import re
            flight_pattern = r'[A-Z]{2,3}[0-9]{2,4}|[0-9]{3,4}'
            flight_matches = re.findall(flight_pattern, rag_response.answer)
            
            if flight_matches:
                # Found flight number in document
                answer = f"Flight Number: {flight_matches[0]}"
                special_processing = {
                    "approach": "direct_extraction",
                    "flight_patterns_found": flight_matches
                }
            else:
                # No flight number found, trigger geographic challenge
                print("No flight number found directly, triggering geographic challenge...")
                return await self._handle_geographic_challenge(question, doc_id, start_time, challenge_info)
            
            total_time = time.time() - start_time
            
            return RAGResponse(
                answer=answer,
                processing_time=total_time,
                doc_id=doc_id,
                sources_used=rag_response.sources_used,
                pipeline_stats={
                    **rag_response.pipeline_stats,
                    "flight_extraction": True
                },
                challenge_detection=challenge_info,
                special_processing=special_processing
            )
            
        except Exception as e:
            print(f"Flight request handling failed: {e}")
            return await self._fallback_to_standard_rag(question, doc_id, start_time, challenge_info, f"Flight request failed: {e}")
    
    async def _handle_token_request(self, question: str, doc_id: str, start_time: float, challenge_info: Dict[str, Any]) -> RAGResponse:
        """Handle secret token requests using web client"""
        conditional_print("Handling secret token request...")
        
        try:
            from app.services.web_client import WebClient
            import re
            
            # Extract team ID from the URL in the question
            team_id = "2836"  # Default fallback
            url_pattern = r'https?://[^\s]+'
            urls = re.findall(url_pattern, question)
            
            if urls:
                url = urls[0]
                # Extract hackTeam parameter from URL
                team_match = re.search(r'hackTeam=(\d+)', url)
                if team_match:
                    team_id = team_match.group(1)
                    print(f"Extracted team ID from URL: {team_id}")
                else:
                    print(f"No team ID found in URL, using default: {team_id}")
            else:
                print(f"No URL found in question, using default team ID: {team_id}")
            
            async with WebClient() as client:
                token = await client.hackrx_get_secret_token(hack_team=team_id)
                
                if token:
                    answer = f"Secret Token: {token}"
                    special_processing = {
                        "approach": "web_api_call",
                        "token_obtained": True,
                        "token_length": len(token),
                        "team_id_used": team_id,
                        "url_extracted": urls[0] if urls else None
                    }
                else:
                    answer = "Failed to retrieve secret token from API endpoint."
                    special_processing = {
                        "approach": "web_api_call", 
                        "token_obtained": False,
                        "team_id_used": team_id,
                        "url_extracted": urls[0] if urls else None
                    }
            
            total_time = time.time() - start_time
            
            return RAGResponse(
                answer=answer,
                processing_time=total_time,
                doc_id=doc_id,
                sources_used=[{
                    "source": "hackrx_api_endpoint",
                    "method": "web_scraping",
                    "success": token is not None
                }],
                pipeline_stats={
                    "web_scraping_time": total_time - (time.time() - start_time),
                    "total_time": total_time,
                    "model_info": {"web_client": "hackrx_token_api"}
                },
                challenge_detection=challenge_info,
                special_processing=special_processing
            )
            
        except Exception as e:
            print(f"Token request handling failed: {e}")
            return await self._fallback_to_standard_rag(question, doc_id, start_time, challenge_info, f"Token request failed: {e}")
    
    async def _handle_web_scraping(self, question: str, doc_id: str, start_time: float, challenge_info: Dict[str, Any]) -> RAGResponse:
        """Handle web scraping requests"""
        conditional_print("Handling web scraping request...")
        
        try:
            from app.services.web_client import WebClient
            import re
            
            # Extract URL from question if present
            url_pattern = r'https?://[^\s]+'
            urls = re.findall(url_pattern, question)
            
            if urls:
                async with WebClient() as client:
                    url = urls[0]
                    response = await client.fetch_url(url)
                    
                    if response.status_code == 200:
                        # Extract relevant content based on question
                        if "token" in question.lower():
                            # Look for token-like patterns
                            token_pattern = r'[A-Za-z0-9]{20,}'
                            tokens = re.findall(token_pattern, response.content[:1000])
                            if tokens:
                                answer = f"Found token: {tokens[0]}"
                            else:
                                answer = "No token pattern found in the webpage."
                        else:
                            # General content extraction
                            content = response.content[:500] + "..." if len(response.content) > 500 else response.content
                            answer = f"Web content from {url}:\n{content}"
                        
                        special_processing = {
                            "approach": "web_scraping",
                            "url_scraped": url,
                            "content_length": len(response.content),
                            "status_code": response.status_code
                        }
                    else:
                        answer = f"Failed to access {url}. Status: {response.status_code}"
                        special_processing = {
                            "approach": "web_scraping",
                            "url_attempted": url,
                            "error": f"HTTP {response.status_code}"
                        }
            else:
                # No URL found, fallback to standard RAG
                return await self._fallback_to_standard_rag(question, doc_id, start_time, challenge_info, "No URL found for scraping")
            
            total_time = time.time() - start_time
            
            return RAGResponse(
                answer=answer,
                processing_time=total_time,
                doc_id=doc_id,
                sources_used=[{
                    "source": urls[0] if urls else "unknown",
                    "method": "web_scraping",
                    "status": response.status_code if 'response' in locals() else "failed"
                }],
                pipeline_stats={
                    "web_scraping_time": total_time,
                    "total_time": total_time,
                    "model_info": {"web_client": "general_scraping"}
                },
                challenge_detection=challenge_info,
                special_processing=special_processing
            )
            
        except Exception as e:
            print(f"Web scraping handling failed: {e}")
            return await self._fallback_to_standard_rag(question, doc_id, start_time, challenge_info, f"Web scraping failed: {e}")
    
    async def _standard_rag_query(self, question: str, doc_id: str, k_retrieve: int = 10) -> RAGResponse:
        """Execute standard RAG query without challenge detection"""
        # Simple single query processing
        query_embedding = await self.embedding_manager.encode_query(question)
        
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            k=k_retrieve,
            doc_id_filter=doc_id
        )
        
        if not search_results:
            return RAGResponse(
                answer="No relevant information found.",
                processing_time=0.0,
                doc_id=doc_id,
                sources_used=[],
                pipeline_stats={"chunks_retrieved": 0}
            )
        
        # Generate answer
        answer_result = await self.answer_generator.generate_answer(
            question=question,
            search_results=search_results,
            max_context_length=settings.max_context_tokens
        )
        
        return RAGResponse(
            answer=answer_result.answer,
            processing_time=0.0,  # Will be set by caller
            doc_id=doc_id,
            sources_used=answer_result.sources,
            pipeline_stats={
                "chunks_retrieved": len(search_results),
                "context_chunks_used": len(answer_result.context_used),
                "model_info": answer_result.model_info
            }
        )
    
    async def _fallback_to_standard_rag(self, question: str, doc_id: str, start_time: float, challenge_info: Dict[str, Any], error_msg: str) -> RAGResponse:
        """Fallback to standard RAG when challenge handling fails"""
        conditional_print(f"Falling back to standard RAG: {error_msg}")
        
        try:
            rag_response = await self._standard_rag_query(question, doc_id)
            total_time = time.time() - start_time
            
            # Update response with timing and challenge info
            rag_response.processing_time = total_time
            rag_response.challenge_detection = challenge_info
            rag_response.special_processing = {
                "fallback_used": True,
                "original_error": error_msg,
                "approach": "standard_rag"
            }
            
            return rag_response
            
        except Exception as fallback_error:
            total_time = time.time() - start_time
            
            return RAGResponse(
                answer=f"Processing failed: {error_msg}. Fallback also failed: {fallback_error}",
                processing_time=total_time,
                doc_id=doc_id,
                sources_used=[],
                pipeline_stats={"total_time": total_time},
                challenge_detection=challenge_info,
                special_processing={"fallback_failed": True, "errors": [error_msg, str(fallback_error)]}
            )
    
    async def answer_question_async(
        self, 
        question: str, 
        doc_id: Optional[str] = None,
        k_retrieve: int = 10,
        max_context_length: int = None,
        enable_transformation: Optional[bool] = None
    ) -> RAGResponse:
        """
        Async version of answer_question with query transformation support
        
        Args:
            question: Question to answer
            doc_id: Optional document ID to filter by
            k_retrieve: Number of chunks to retrieve per sub-query
            max_context_length: Maximum context length for answer generation (None = use config default)
            enable_transformation: Override global transformation setting
            
        Returns:
            RAGResponse with answer and metadata including transformation info
        """
        # Since answer_question is now async-aware, call it directly
        return await self.answer_question(
            question=question,
            doc_id=doc_id,
            k_retrieve=k_retrieve,
            max_context_length=max_context_length,
            enable_transformation=enable_transformation
        )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics with performance metrics"""
        vector_stats = self.vector_store.get_stats()
        cache_stats = self.cache_manager.get_cache_stats()
        query_transformer_stats = self.query_transformer.get_stats()
        
        return {
            "vector_store": vector_stats,
            "embedding_model": self.embedding_manager.get_model_info(),
            "answer_model": self.answer_generator.get_model_info(),
            "query_transformer": query_transformer_stats,
            "cache_performance": cache_stats,
            "optimization_settings": {
                "fast_mode": settings.fast_mode,
                "competition_mode": settings.competition_mode,
                "result_caching": settings.enable_result_caching,
                "embedding_caching": settings.enable_embedding_cache,
                "k_retrieve": settings.k_retrieve,
                "max_tokens": settings.llm_max_tokens
            },
            "query_transformation_settings": {
                "enabled": settings.enable_query_transformation,
                "max_sub_queries": settings.max_sub_queries,
                "min_query_length": settings.min_query_length,
                "transformation_temperature": settings.transformation_temperature,
                "timeout": settings.query_transformation_timeout
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
    

    async def process_questions_individually(
        self,
        questions: List[str],
        doc_id: Optional[str] = None,
        k_retrieve: int = None,
        max_context_length: int = None
    ) -> List[RAGResponse]:
        """
        Process multiple questions individually for reliable challenge detection
        
        This method processes each question through the full pipeline individually,
        ensuring consistent challenge detection and reliable results.
        
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
            
        debug_print(f"Processing {len(questions)} questions individually for reliable results")
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


# Singleton instance
_rag_coordinator = None

def get_rag_coordinator() -> RAGCoordinator:
    """Get singleton RAG coordinator instance"""
    global _rag_coordinator
    if _rag_coordinator is None:
        _rag_coordinator = RAGCoordinator()
    return _rag_coordinator
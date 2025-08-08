"""
RAG Coordinator - orchestrates the complete RAG pipeline with query transformation
"""
import asyncio
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
import logging

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
        
        # Initialize challenge solver (lazy loaded when needed)
        self._challenge_solver = None
        
        # Single embedding manager based on config
        
        print("RAG Coordinator initialized with performance optimizations, query transformation, and multi-provider embeddings")
        
        # Pre-warm embedding model for better parallel processing performance
        print("Pre-warming embedding model for optimal performance...")
        try:
            self.embedding_manager.ensure_model_ready()
            print("Embedding model pre-warmed successfully")
        except Exception as e:
            print(f"Warning: Could not pre-warm embedding model: {e}")
            print("   Model will be loaded on first use")
    
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
            
            # Stage 2: Text chunking with zero content validation
            stage_start = time.time()
            chunks = self.text_chunker.chunk_text(
                processed_doc.text,
                processed_doc.metadata
            )
            chunking_time = time.time() - stage_start
            
            print(f"Document chunked into {len(chunks)} pieces")
            
            # Handle zero chunks case (prevents division by zero errors)
            if len(chunks) == 0:
                print("WARNING: Document produced zero chunks - this will cause processing to fail")
                print("Creating minimal chunk to prevent system failure")
                
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
                print(f"Created fallback chunk with {len(error_chunk.text)} characters")
            
            # Stage 3: Generate embeddings
            stage_start = time.time()
            embedding_result = await self.embedding_manager.encode_chunks(chunks)
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
        Intelligent question answering with automatic challenge detection
        
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
        
        print(f"Processing question: {question}")
        if doc_id:
            print(f"Filtering by document: {doc_id}")
        
        try:
            # Step 1: Get document content for context (if doc_id specified)
            document_content = ""
            if doc_id:
                # Extract document content from vector store for challenge detection
                doc_chunks = []
                for i in range(len(self.vector_store.chunk_texts)):
                    metadata = self.vector_store.chunk_metadata[i]
                    if metadata.doc_id == doc_id:
                        doc_chunks.append(self.vector_store.chunk_texts[i])
                document_content = " ".join(doc_chunks[:5])  # First 5 chunks for context
            
            # Step 2: Detect challenge type
            detections = self.challenge_detector.detect_challenge_type([question], document_content)
            primary_challenge = self.challenge_detector.get_primary_challenge(detections)
            
            print(f"🔍 Challenge detected: {primary_challenge.challenge_type.value} (confidence: {primary_challenge.confidence:.2f})")
            print(f"💡 Suggested approach: {primary_challenge.suggested_approach}")
            
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
                # Enhanced multilingual processing
                detected_lang = primary_challenge.metadata.get("language_detected", "unknown")
                special_processing = {
                    "multilingual_mode": True,
                    "detected_language": detected_lang,
                    "language_processing": True
                }
                print(f"🌍 Using enhanced multilingual processing for {detected_lang} content")
            
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
            
            # Simple vector search with fixed k - no adaptive logic
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                k=settings.k_retrieve,  # Use fixed value from config
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
            
            # Pass multilingual info to answer generator if available
            multilingual_mode = special_processing and special_processing.get("multilingual_mode", False)
            detected_language = special_processing and special_processing.get("detected_language")
            
            answer_result = self.answer_generator.generate_answer(
                question=question,
                search_results=deduplicated_results,
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
            
            # No caching - keep it simple
            
            return response
            
        except Exception as e:
            print(f"Enhanced question answering failed: {str(e)}")
            raise
    
    async def _handle_geographic_challenge(self, question: str, doc_id: str, start_time: float, challenge_info: Dict[str, Any]) -> RAGResponse:
        """Handle geographic puzzle challenge with multi-step solving"""
        print("🌍 Handling geographic puzzle challenge...")
        
        try:
            # Get challenge solver instance
            if self._challenge_solver is None:
                from app.services.challenge_solver import get_challenge_solver
                self._challenge_solver = await get_challenge_solver()
            
            # Solve the geographic challenge
            challenge_result = await self._challenge_solver.solve_geographic_challenge(doc_id)
            
            # Extract the answer
            if challenge_result.flight_number:
                answer = f"Flight Number: {challenge_result.flight_number}"
            else:
                # If challenge failed, provide detailed error info
                failed_steps = [step for step in challenge_result.steps_completed if not step.success]
                if failed_steps:
                    last_error = failed_steps[-1].error
                    answer = f"Geographic challenge failed at step {failed_steps[-1].step.value}: {last_error}"
                else:
                    answer = "Geographic challenge completed but no flight number was obtained."
            
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
                    "landmark": challenge_result.landmark.landmark if challenge_result.landmark else None,
                    "flight_number": challenge_result.flight_number,
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
        print("✈️ Handling flight number request...")
        
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
        print("🔐 Handling secret token request...")
        
        try:
            from app.services.web_client import WebClient
            
            async with WebClient() as client:
                token = await client.hackrx_get_secret_token()
                
                if token:
                    answer = f"Secret Token: {token}"
                    special_processing = {
                        "approach": "web_api_call",
                        "token_obtained": True,
                        "token_length": len(token)
                    }
                else:
                    answer = "Failed to retrieve secret token from API endpoint."
                    special_processing = {
                        "approach": "web_api_call", 
                        "token_obtained": False
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
        print("🌐 Handling web scraping request...")
        
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
        answer_result = self.answer_generator.generate_answer(
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
        print(f"Falling back to standard RAG: {error_msg}")
        
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


# Singleton instance
_rag_coordinator = None

def get_rag_coordinator() -> RAGCoordinator:
    """Get singleton RAG coordinator instance"""
    global _rag_coordinator
    if _rag_coordinator is None:
        _rag_coordinator = RAGCoordinator()
    return _rag_coordinator
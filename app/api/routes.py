"""
API routes for the RAG application
"""
import time
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from app.models.requests import (
    QueryRequest, SimpleQueryResponse, HealthResponse, UploadResponse, 
    FileInfoResponse, FileListResponse
)
from app.services.rag_coordinator import get_rag_coordinator
from app.services.file_manager import get_file_manager
from app.services.question_logger import get_question_logger
from app.services.persistent_cache import get_persistent_cache_manager
from app.core.security import verify_token
from app.core.config import settings
from app.core.directories import get_directory_manager
from app.utils.debug import debug_print, info_print

router = APIRouter()


@router.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": f"{settings.app_name} is running",
        "version": settings.version,
        "status": "ready"
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        rag_coordinator = get_rag_coordinator()
        health_status = rag_coordinator.health_check()
        
        return HealthResponse(
            status=health_status["status"],
            service="rag-pipeline"
        )
    except Exception:
        return HealthResponse(
            status="unhealthy",
            service="rag-pipeline"
        )


@router.post("/query", response_model=SimpleQueryResponse)
async def process_queries(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """
    Complete RAG document query endpoint
    
    This endpoint:
    1. Processes the document (with caching to prevent duplicates)
    2. Answers each question using the RAG pipeline
    3. Returns clean, direct answers
    """
    import time
    
    try:
        # Start total processing timer
        total_start_time = time.time()
        
        rag_coordinator = get_rag_coordinator()
        question_logger = get_question_logger()
        
        # Step 1: Process document (with automatic caching) or skip for web scraping
        if str(request.documents) == 'web-scraping://no-document':
            debug_print("Skipping document processing for web scraping request")
            doc_id = None
            doc_processing_time = 0.0
            doc_result = {"status": "skipped_for_web_scraping", "doc_id": None}
        else:
            debug_print(f"Processing document: {request.documents}")
            debug_print(f"Using embedding provider: {settings.embedding_provider}")
            doc_start_time = time.time()
            doc_result = await rag_coordinator.process_document(
                url=str(request.documents)
            )
            doc_processing_time = time.time() - doc_start_time
            doc_id = doc_result["doc_id"]
        
        # Start question logging session
        session_metadata = {
            "embedding_provider": settings.embedding_provider,
            "llm_provider": settings.llm_provider,
            "embedding_model": settings.embedding_model,
            "total_questions": len(request.questions),
            "query_transformation_enabled": request.enable_query_transformation if request.enable_query_transformation is not None else settings.enable_query_transformation
        }
        
        session_id = question_logger.start_session(
            document_url=str(request.documents),
            doc_id=doc_id or "web-scraping-request",
            metadata=session_metadata
        )
        
        if doc_result["status"] == "cached":
            info_print(f"Using cached document: {doc_id}")
        elif doc_result["status"] == "skipped_for_web_scraping":
            info_print("Skipped document processing for web scraping request")
        elif doc_result["status"] == "token_api_detected":
            info_print(f"Token API endpoint detected: {doc_result.get('api_url', 'unknown')}")
            # For token API endpoints, we'll process the questions differently
            # The questions will be processed as web scraping requests
        else:
            info_print(f"Processed new document: {doc_id}")
            info_print(f"  - Processing time: {doc_result['processing_time']:.2f}s")
            
            # Handle different processing pipeline results
            if doc_result.get("pipeline_used") == "direct_processing":
                debug_print(f"  - Document type: {doc_result.get('document_type', 'unknown')}")
                if "landmark_mappings" in doc_result:
                    debug_print(f"  - Landmark mappings: {doc_result['landmark_mappings']['total_landmarks']}")
                    debug_print(f"  - Cities covered: {doc_result['landmark_mappings']['cities_covered']}")
            elif "document_stats" in doc_result:
                # Traditional RAG pipeline results
                doc_stats = doc_result["document_stats"]
                if "chunk_count" in doc_stats:
                    debug_print(f"  - Chunks created: {doc_stats['chunk_count']}")
                if "preserved_definitions" in doc_stats:
                    debug_print(f"  - Preserved definitions: {doc_stats['preserved_definitions']}")
        
        # Step 2: Pre-warm embedding model for parallel processing
        debug_print("Pre-warming embedding model for parallel processing...")
        embedding_warmup_start = time.time()
        
        # Ensure the embedding model is loaded before parallel processing
        if not rag_coordinator.embedding_manager.ensure_model_ready():
            raise HTTPException(
                status_code=500, 
                detail="Failed to initialize embedding model for parallel processing"
            )
        
        embedding_warmup_time = time.time() - embedding_warmup_start
        debug_print(f"Embedding model ready in {embedding_warmup_time:.2f}s")
        
        # Step 3: Answer all questions with batch optimization (OPTIMIZED PARALLEL PROCESSING)
        info_print(f"Processing {len(request.questions)} questions with batch optimization")
        
        questions_start_time = time.time()
        
        # Handle token API detection - modify questions to include the URL for web scraping
        questions_to_process = list(request.questions)
        if doc_result["status"] == "token_api_detected":
            api_url = doc_result.get("api_url", "")
            info_print(f"Modifying questions to include token API URL: {api_url}")
            # Modify questions to include the URL for proper web scraping detection
            questions_to_process = [f"{q} {api_url}" for q in request.questions]
        
        # Process questions individually for reliable and consistent results
        rag_responses = []
        for i, question in enumerate(questions_to_process):
            debug_print(f"Processing question {i+1}/{len(questions_to_process)}: {question[:50]}...")
            if request.use_universal_solver:
                debug_print(f"  Using Universal LLM Solver for question {i+1}")
            response = await rag_coordinator.answer_question(
                question=question,
                doc_id=doc_id,
                k_retrieve=settings.k_retrieve,
                max_context_length=settings.max_context_tokens,
                use_universal_solver=request.use_universal_solver or False
            )
            rag_responses.append(response)
            debug_print(f"  Question {i+1} completed in {response.processing_time:.2f}s")
        
        # Log each question and response
        for i, (question, rag_response) in enumerate(zip(request.questions, rag_responses)):
            debug_print(f"  Question {i+1}: {question}")
            debug_print(f"    Answered in {rag_response.processing_time:.2f}s")
            debug_print(f"    Used {len(rag_response.sources_used)} sources")
            
            # Log question and response
            try:
                # Extract similarity scores from pipeline stats if available
                similarity_scores = []
                if hasattr(rag_response, 'pipeline_stats') and 'similarity_scores' in rag_response.pipeline_stats:
                    similarity_scores = rag_response.pipeline_stats.get('similarity_scores', [])
                elif len(rag_response.sources_used) > 0:
                    # Try to extract from sources metadata
                    similarity_scores = [source.get('similarity_score', 0.0) for source in rag_response.sources_used[:3]]
                
                question_logger.log_question(
                    session_id=session_id,
                    question_id=i+1,
                    question=question,
                    processing_time=rag_response.processing_time,
                    answer=rag_response.answer,
                    sources_used=len(rag_response.sources_used),
                    similarity_scores=similarity_scores,
                    error=None
                )
            except Exception as log_error:
                debug_print(f"Warning: Failed to log question {i+1}: {log_error}")
        
        # Extract answers, sources, timing info, and transformation metadata
        answers = [response.answer for response in rag_responses]
        sources = [response.sources_used for response in rag_responses]
        individual_times = [response.processing_time for response in rag_responses]
        query_transformations = [response.query_transformation for response in rag_responses]
        
        # Calculate transformation statistics
        transformations_successful = sum(1 for qt in query_transformations 
                                       if qt and qt.get('successful', False))
        
        # Calculate total times
        questions_processing_time = time.time() - questions_start_time
        total_processing_time = time.time() - total_start_time
        
        # Display comprehensive time metrics with parallel processing benefits
        sequential_time_estimate = sum(individual_times)  # What it would have taken sequentially
        parallel_speedup = sequential_time_estimate / questions_processing_time if questions_processing_time > 0 else 1
        
        info_print("\n" + "="*70)
        info_print("PARALLEL PROCESSING COMPLETED - PERFORMANCE METRICS")
        info_print("="*70)
        info_print(f"Document Processing Time: {doc_processing_time:.2f}s")
        info_print(f"Embedding Model Warmup: {embedding_warmup_time:.2f}s")
        info_print(f"Questions Processing Time: {questions_processing_time:.2f}s (PARALLEL)")
        info_print(f"Total Processing Time: {total_processing_time:.2f}s")
        
        debug_print("\nPARALLEL PROCESSING BENEFITS:")
        debug_print(f"  - Parallel Time: {questions_processing_time:.2f}s")
        debug_print(f"  - Concurrent Questions Limit: {settings.max_concurrent_questions}")
        
        debug_print("\nQuestion-by-Question Breakdown:")
        for i, q_time in enumerate(individual_times, 1):
            debug_print(f"  Question {i}: {q_time:.2f}s")
        debug_print("\nQuestions Statistics:")
        debug_print(f"  - Total Questions: {len(request.questions)}")
        debug_print(f"  - Average Time per Question: {sum(individual_times) / len(individual_times):.2f}s")
        debug_print(f"  - Fastest Question: {min(individual_times):.2f}s")
        debug_print(f"  - Slowest Question: {max(individual_times):.2f}s")
        debug_print(f"  - Effective Questions per Second: {len(request.questions) / questions_processing_time:.2f}")
        
        debug_print("\nQuery Transformation Statistics:")
        debug_print(f"  - Transformations Enabled: {request.enable_query_transformation if request.enable_query_transformation is not None else settings.enable_query_transformation}")
        debug_print(f"  - Questions with Successful Transformation: {transformations_successful}/{len(request.questions)}")
        info_print("="*70)
        
        # Create processing summary
        processing_summary = {
            "total_questions": len(request.questions),
            "questions_with_transformation": transformations_successful,
            "parallel_processing_time": questions_processing_time,
            "total_processing_time": total_processing_time,
            "average_time_per_question": sum(individual_times) / len(individual_times),
            "document_processing_time": doc_processing_time,
            "embedding_warmup_time": embedding_warmup_time,
            "transformation_enabled": request.enable_query_transformation if request.enable_query_transformation is not None else settings.enable_query_transformation,
            "embedding_provider": settings.embedding_provider,
            "embedding_dimension": rag_responses[0].pipeline_stats.get("model_info", {}).get("embedding_dimension", "unknown") if rag_responses else "unknown"
        }
        
        # End question logging session
        try:
            session_end_metadata = {
                **session_metadata,
                "processing_summary": processing_summary,
                "questions_with_transformation": transformations_successful,
                "parallel_speedup": f"{parallel_speedup:.2f}x" if parallel_speedup > 1 else "1.0x",
                "average_similarity_score": sum(sum(scores) for scores in [
                    [source.get('similarity_score', 0.0) for source in resp.sources_used[:3]] 
                    for resp in rag_responses
                ]) / max(sum(len(resp.sources_used) for resp in rag_responses), 1)
            }
            
            question_logger.end_session(
                session_id=session_id,
                document_url=str(request.documents),
                doc_id=doc_id,
                session_metadata=session_end_metadata
            )
        except Exception as log_error:
            debug_print(f"Warning: Failed to end question logging session: {log_error}")
        
        return SimpleQueryResponse(
            answers=answers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        info_print(f"RAG pipeline error: {str(e)}")
        
        # Log error in question logging if session was started
        try:
            if 'session_id' in locals() and session_id:
                question_logger.log_question(
                    session_id=session_id,
                    question_id=0,  # Error entry
                    question="[SYSTEM ERROR]",
                    processing_time=time.time() - total_start_time if 'total_start_time' in locals() else 0,
                    answer=f"Processing failed: {str(e)}",
                    sources_used=0,
                    similarity_scores=[],
                    error=str(e)
                )
                question_logger.end_session(
                    session_id=session_id,
                    document_url=str(request.documents) if 'request' in locals() else "unknown",
                    doc_id="error",
                    session_metadata={"error": str(e), "status": "failed"}
                )
        except Exception as log_error:
            debug_print(f"Failed to log error: {log_error}")
        
        raise HTTPException(
            status_code=500,
            detail=f"RAG processing failed: {str(e)}"
        )


@router.get("/debug/system-stats")
async def get_system_stats(token: str = Depends(verify_token)):
    """Debug endpoint to inspect system statistics"""
    try:
        rag_coordinator = get_rag_coordinator()
        stats = rag_coordinator.get_system_stats()
        
        return {
            "system_status": "operational",
            "statistics": stats,
            "health": rag_coordinator.health_check()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Stats collection failed: {str(e)}"
        )



@router.get("/debug/documents")
async def list_documents(token: str = Depends(verify_token)):
    """Debug endpoint to list processed documents"""
    try:
        rag_coordinator = get_rag_coordinator()
        vector_store = rag_coordinator.vector_store
        
        documents = []
        for doc_id in vector_store.documents.keys():
            doc_info = vector_store.get_document_info(doc_id)
            if doc_info:
                documents.append(doc_info)
        
        return {
            "total_documents": len(documents),
            "documents": documents
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document listing failed: {str(e)}"
        )


@router.get("/debug/directories")
async def get_directory_info(token: str = Depends(verify_token)):
    """Debug endpoint to check directory status and information"""
    try:
        directory_manager = get_directory_manager()
        
        # Get comprehensive directory information
        dir_info = directory_manager.get_directory_info()
        
        # Validate all directories
        all_valid = directory_manager.validate_directories()
        
        return {
            "status": "healthy" if all_valid else "degraded",
            "validation_passed": all_valid,
            "directory_info": dir_info,
            "summary": {
                "total_directories": len(dir_info["directories"]),
                "total_size_mb": dir_info["total_size_mb"],
                "healthy_directories": sum(1 for d in dir_info["directories"].values() 
                                         if d["exists"] and d["is_directory"] and d["readable"] and d["writable"]),
                "total_files": sum(d["file_count"] for d in dir_info["directories"].values())
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Directory info collection failed: {str(e)}"
        )


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    """
    Upload a PDF file for processing
    
    This endpoint:
    1. Validates the uploaded file (PDF only, size limits)
    2. Saves the file with a unique ID
    3. Returns file information and upload URL for use in queries
    """
    try:
        file_manager = get_file_manager()
        
        # Save the uploaded file
        file_info = await file_manager.save_uploaded_file(file)
        
        return UploadResponse(
            file_id=file_info.file_id,
            original_filename=file_info.original_filename,
            file_size=file_info.file_size,
            content_type=file_info.content_type,
            uploaded_at=file_info.uploaded_at,
            expires_at=file_info.expires_at,
            upload_url=f"upload://{file_info.file_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {str(e)}"
        )




@router.get("/uploads", response_model=FileListResponse)
async def list_uploaded_files(token: str = Depends(verify_token)):
    """List all uploaded files"""
    try:
        file_manager = get_file_manager()
        files_data = file_manager.list_files()
        
        files = []
        for file_data in files_data:
            files.append(FileInfoResponse(
                file_id=file_data["file_id"],
                original_filename=file_data["original_filename"],
                file_size=file_data["file_size"],
                content_type="application/pdf",  # We only allow PDFs
                uploaded_at=file_data["uploaded_at"],
                expires_at=file_data["expires_at"],
                expired=file_data["expired"],
                exists=file_data["exists"]
            ))
        
        return FileListResponse(
            total_files=len(files),
            files=files
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list uploaded files: {str(e)}"
        )


@router.get("/uploads/{file_id}", response_model=FileInfoResponse)
async def get_uploaded_file_info(
    file_id: str,
    token: str = Depends(verify_token)
):
    """Get information about a specific uploaded file"""
    try:
        file_manager = get_file_manager()
        file_info = file_manager.get_file_info(file_id)
        
        if not file_info:
            raise HTTPException(
                status_code=404,
                detail=f"Uploaded file not found: {file_id}"
            )
        
        from datetime import datetime
        current_time = datetime.now()
        
        return FileInfoResponse(
            file_id=file_info.file_id,
            original_filename=file_info.original_filename,
            file_size=file_info.file_size,
            content_type=file_info.content_type,
            uploaded_at=file_info.uploaded_at,
            expires_at=file_info.expires_at,
            expired=current_time > file_info.expires_at,
            exists=file_manager.get_file_path(file_id) is not None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get file info: {str(e)}"
        )


@router.delete("/uploads/{file_id}")
async def delete_uploaded_file(
    file_id: str,
    token: str = Depends(verify_token)
):
    """Delete a specific uploaded file"""
    try:
        file_manager = get_file_manager()
        
        if not file_manager.get_file_info(file_id):
            raise HTTPException(
                status_code=404,
                detail=f"Uploaded file not found: {file_id}"
            )
        
        success = file_manager.remove_file(file_id)
        
        if success:
            return {"message": f"File {file_id} deleted successfully"}
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete file"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete file: {str(e)}"
        )


@router.get("/debug/file-manager")
async def get_file_manager_stats(token: str = Depends(verify_token)):
    """Debug endpoint to inspect file manager statistics"""
    try:
        file_manager = get_file_manager()
        stats = file_manager.get_stats()
        
        return {
            "status": "operational",
            "file_manager_stats": stats
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File manager stats collection failed: {str(e)}"
        )


@router.get("/debug/cache-stats")
async def get_cache_statistics(token: str = Depends(verify_token)):
    """Get comprehensive caching statistics"""
    try:
        rag_coordinator = get_rag_coordinator()
        persistent_cache = get_persistent_cache_manager()
        
        # Get stats from all cache layers
        in_memory_stats = rag_coordinator.cache_manager.get_cache_stats()
        persistent_stats = persistent_cache.get_cache_stats()
        
        return {
            "cache_overview": {
                "layers": ["in_memory_ttl", "in_memory_lru", "persistent_disk"],
                "total_cache_types": 5,
                "cache_enabled": {
                    "result_caching": settings.enable_result_caching,
                    "embedding_caching": settings.enable_embedding_cache,
                    "reranker_caching": settings.enable_reranker_cache
                }
            },
            "in_memory_cache": in_memory_stats,
            "persistent_cache": persistent_stats,
            "performance_impact": {
                "query_embedding_cache": "CRITICAL - Saves 200-500ms per query",
                "document_processing_cache": "HIGH - Saves 2-10s per document", 
                "query_result_cache": "MEDIUM - Saves 100-300ms per duplicate query",
                "persistent_benefits": "Survives container restarts, reduced cold start time"
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cache stats collection failed: {str(e)}"
        )


@router.post("/debug/cache-cleanup")
async def cleanup_caches(
    cache_type: str = "expired_only",
    token: str = Depends(verify_token)
):
    """Cleanup cache entries (expired_only, all, or specific type)"""
    try:
        rag_coordinator = get_rag_coordinator()
        persistent_cache = get_persistent_cache_manager()
        
        results = {}
        
        if cache_type == "expired_only":
            # Clean up only expired entries
            in_memory_cleaned = rag_coordinator.cache_manager.cleanup_expired_entries()
            persistent_cleaned = persistent_cache.cleanup_expired()
            
            results = {
                "operation": "cleanup_expired",
                "in_memory_cleaned": in_memory_cleaned,
                "persistent_cleaned": persistent_cleaned,
                "message": "Cleaned up expired cache entries only"
            }
            
        elif cache_type == "all":
            # Clear all caches
            in_memory_cleared = rag_coordinator.cache_manager.clear_cache("all")
            persistent_cleared = persistent_cache.clear_cache()
            
            results = {
                "operation": "clear_all",
                "in_memory_cleared": in_memory_cleared,
                "persistent_cleared": persistent_cleared,
                "message": "All cache entries cleared"
            }
            
        else:
            # Clear specific cache type
            valid_types = ["query", "embedding", "query_embedding", "reranker", "answer"]
            if cache_type not in valid_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid cache type. Valid types: {valid_types}"
                )
            
            in_memory_cleared = rag_coordinator.cache_manager.clear_cache(cache_type)
            persistent_cleared = persistent_cache.clear_cache(cache_type)
            
            results = {
                "operation": f"clear_{cache_type}",
                "in_memory_cleared": in_memory_cleared,
                "persistent_cleared": persistent_cleared,
                "message": f"Cleared {cache_type} cache entries"
            }
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cache cleanup failed: {str(e)}"
        )



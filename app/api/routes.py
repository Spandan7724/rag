"""
API routes for the RAG application
"""
import asyncio
import time
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.security import HTTPAuthorizationCredentials
from typing import List

from app.models.requests import (
    QueryRequest, QueryResponse, SimpleQueryResponse, HealthResponse, UploadResponse, 
    UploadRequest, FileInfoResponse, FileListResponse
)
from app.services.rag_coordinator import get_rag_coordinator
from app.services.file_manager import get_file_manager
from app.services.question_logger import get_question_logger
from app.services.challenge_solver import get_challenge_solver, solve_hackrx_challenge
from app.services.universal_llm_solver import get_universal_llm_solver
from app.services.web_client import WebClient, fetch_web_content
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
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            service="rag-pipeline"
        )


@router.post("/hackrx/run", response_model=SimpleQueryResponse)
async def process_queries(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """
    Complete RAG endpoint with clean architecture
    
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
        debug_print(f"Pre-warming embedding model for parallel processing...")
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
        
        # Process questions individually for reliable challenge detection and consistent results
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
        
        info_print(f"\n" + "="*70)
        info_print(f"PARALLEL PROCESSING COMPLETED - PERFORMANCE METRICS")
        info_print(f"="*70)
        info_print(f"Document Processing Time: {doc_processing_time:.2f}s")
        info_print(f"Embedding Model Warmup: {embedding_warmup_time:.2f}s")
        info_print(f"Questions Processing Time: {questions_processing_time:.2f}s (PARALLEL)")
        info_print(f"Total Processing Time: {total_processing_time:.2f}s")
        
        debug_print(f"\nPARALLEL PROCESSING BENEFITS:")
        debug_print(f"  - Parallel Time: {questions_processing_time:.2f}s")
        debug_print(f"  - Concurrent Questions Limit: {settings.max_concurrent_questions}")
        
        debug_print(f"\nQuestion-by-Question Breakdown:")
        for i, q_time in enumerate(individual_times, 1):
            debug_print(f"  Question {i}: {q_time:.2f}s")
        debug_print(f"\nQuestions Statistics:")
        debug_print(f"  - Total Questions: {len(request.questions)}")
        debug_print(f"  - Average Time per Question: {sum(individual_times) / len(individual_times):.2f}s")
        debug_print(f"  - Fastest Question: {min(individual_times):.2f}s")
        debug_print(f"  - Slowest Question: {max(individual_times):.2f}s")
        debug_print(f"  - Effective Questions per Second: {len(request.questions) / questions_processing_time:.2f}")
        
        debug_print(f"\nQuery Transformation Statistics:")
        debug_print(f"  - Transformations Enabled: {request.enable_query_transformation if request.enable_query_transformation is not None else settings.enable_query_transformation}")
        debug_print(f"  - Questions with Successful Transformation: {transformations_successful}/{len(request.questions)}")
        info_print(f"="*70)
        
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


@router.get("/debug/query-transformation-metrics")
async def get_query_transformation_metrics(token: str = Depends(verify_token)):
    """Debug endpoint for detailed query transformation metrics and performance analysis"""
    try:
        rag_coordinator = get_rag_coordinator()
        query_transformer = rag_coordinator.query_transformer
        
        # Get current configuration
        transformer_stats = query_transformer.get_stats()
        
        # Create detailed metrics response
        metrics = {
            "transformation_status": "enabled" if settings.enable_query_transformation else "disabled",
            "configuration": {
                "enable_query_transformation": settings.enable_query_transformation,
                "max_sub_queries": settings.max_sub_queries,
                "query_transformation_timeout": settings.query_transformation_timeout,
                "min_query_length": settings.min_query_length,
                "transformation_temperature": settings.transformation_temperature
            },
            "model_info": {
                "provider": transformer_stats["provider"],
                "model": transformer_stats["model"],
                "timeout": transformer_stats["timeout"]
            },
            "performance_guidelines": {
                "best_for": [
                    "Complex multi-part questions",
                    "Comparison queries (e.g., 'Compare A and B')",
                    "Questions with multiple concepts",
                    "Policy documents with interconnected clauses"
                ],
                "expected_latency_increase": "1-3 seconds per query",
                "recommended_use_cases": [
                    "Insurance policy analysis",
                    "Legal document review", 
                    "Technical specification queries",
                    "Compliance requirement analysis"
                ]
            },
            "optimization_tips": [
                "Questions under 20 characters are automatically skipped",
                "Simple questions (single concept) won't be transformed",
                "Set enable_query_transformation=false for speed-critical applications",
                "Use transformation selectively for complex analytical queries"
            ],
            "health_check": {
                "transformation_service": "operational",
                "llm_provider_available": transformer_stats["provider"] != "none",
                "configuration_valid": all([
                    settings.max_sub_queries > 1,
                    settings.query_transformation_timeout > 0,
                    settings.min_query_length > 0
                ])
            }
        }
        
        return {
            "query_transformation_metrics": metrics,
            "timestamp": time.time()
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


@router.post("/upload-and-query", response_model=QueryResponse)
async def upload_file_and_query(
    file: UploadFile = File(...),
    questions: str = Form(...),
    enable_query_transformation: str = Form(None),
    token: str = Depends(verify_token)
):
    """
    Upload a PDF file and immediately process queries
    
    This endpoint combines upload and query processing:
    1. Uploads and validates the PDF file
    2. Processes the document through the RAG pipeline
    3. Answers the provided questions
    4. Cleans up the uploaded file after processing
    """
    import json
    
    try:
        # Parse questions JSON and transformation parameter
        try:
            questions_list = json.loads(questions)
            if not isinstance(questions_list, list):
                raise ValueError("Questions must be a list")
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Questions must be valid JSON array"
            )
        
        # Parse transformation parameter
        transformation_enabled = None
        if enable_query_transformation is not None:
            if enable_query_transformation.lower() in ['true', '1', 'yes']:
                transformation_enabled = True
            elif enable_query_transformation.lower() in ['false', '0', 'no']:
                transformation_enabled = False
        
        debug_print(f"Using embedding provider from config: {settings.embedding_provider}")
        
        file_manager = get_file_manager()
        rag_coordinator = get_rag_coordinator()
        question_logger = get_question_logger()
        
        # Upload the file
        file_info = await file_manager.save_uploaded_file(file)
        
        try:
            # Create upload URL for processing
            upload_url = f"upload://{file_info.file_id}"
            
            # Process document and answer questions
            doc_result = await rag_coordinator.process_document(
                url=upload_url
            )
            doc_id = doc_result["doc_id"]
            
            info_print(f"Processed uploaded document: {file_info.original_filename}")
            debug_print(f"  - File ID: {file_info.file_id}")
            debug_print(f"  - Document ID: {doc_id}")
            
            # Start question logging session for uploaded file
            session_metadata = {
                "embedding_provider": settings.embedding_provider,
                "llm_provider": settings.llm_provider,
                "embedding_model": settings.embedding_model,
                "total_questions": len(questions_list),
                "query_transformation_enabled": transformation_enabled if transformation_enabled is not None else settings.enable_query_transformation,
                "upload_method": "file_upload",
                "original_filename": file_info.original_filename,
                "file_id": file_info.file_id
            }
            
            session_id = question_logger.start_session(
                document_url=upload_url,
                doc_id=doc_id,
                metadata=session_metadata
            )
            
            # Answer questions with transformation support
            answers = []
            sources = []
            query_transformations = []
            processing_times = []
            
            for i, question in enumerate(questions_list, 1):
                rag_response = await rag_coordinator.answer_question_async(
                    question=question,
                    doc_id=doc_id,
                    k_retrieve=settings.k_retrieve,
                    max_context_length=settings.max_context_tokens,
                    enable_transformation=transformation_enabled
                )
                answers.append(rag_response.answer)
                sources.append(rag_response.sources_used)
                query_transformations.append(rag_response.query_transformation)
                processing_times.append(rag_response.processing_time)
                
                # Log question and response
                try:
                    similarity_scores = [source.get('similarity_score', 0.0) for source in rag_response.sources_used[:3]]
                    
                    question_logger.log_question(
                        session_id=session_id,
                        question_id=i,
                        question=question,
                        processing_time=rag_response.processing_time,
                        answer=rag_response.answer,
                        sources_used=len(rag_response.sources_used),
                        similarity_scores=similarity_scores,
                        error=None
                    )
                except Exception as log_error:
                    debug_print(f"Warning: Failed to log question {i}: {log_error}")
            
            # Calculate statistics
            transformations_successful = sum(1 for qt in query_transformations 
                                           if qt and qt.get('successful', False))
            
            processing_summary = {
                "total_questions": len(questions_list),
                "questions_with_transformation": transformations_successful,
                "average_time_per_question": sum(processing_times) / len(processing_times),
                "transformation_enabled": transformation_enabled if transformation_enabled is not None else settings.enable_query_transformation,
                "embedding_provider": settings.embedding_provider,
                "embedding_dimension": query_transformations[0].get("model_info", {}).get("embedding_dimension", "unknown") if query_transformations and query_transformations[0] else "unknown"
            }
            
            info_print(f"Answered {len(questions_list)} questions for uploaded file")
            if transformations_successful > 0:
                debug_print(f"  - {transformations_successful} questions used query transformation")
            
            # End question logging session
            try:
                session_end_metadata = {
                    **session_metadata,
                    "processing_summary": processing_summary,
                    "questions_with_transformation": transformations_successful,
                    "upload_processing": "completed"
                }
                
                question_logger.end_session(
                    session_id=session_id,
                    document_url=upload_url,
                    doc_id=doc_id,
                    session_metadata=session_end_metadata
                )
            except Exception as log_error:
                debug_print(f"Warning: Failed to end upload question logging session: {log_error}")
            
            return SimpleQueryResponse(
                answers=answers
            )
            
        finally:
            # Clean up uploaded file after processing
            file_manager.remove_file(file_info.file_id)
            debug_print(f"Cleaned up uploaded file: {file_info.file_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload and query failed: {str(e)}"
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


@router.get("/debug/question-logs")
async def get_question_logs(
    days: int = 7,
    token: str = Depends(verify_token)
):
    """Debug endpoint to view recent question logs"""
    try:
        question_logger = get_question_logger()
        
        # Get recent logs
        recent_logs = question_logger.get_recent_logs(days=days)
        
        # Get statistics
        stats = question_logger.get_stats()
        
        return {
            "status": "operational",
            "question_logging_stats": stats,
            "recent_logs_count": len(recent_logs),
            "recent_logs": recent_logs[:10] if recent_logs else [],  # Limit to 10 most recent
            "days_requested": days
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Question logs retrieval failed: {str(e)}"
        )


@router.post("/debug/cleanup-question-logs")
async def cleanup_question_logs(token: str = Depends(verify_token)):
    """Debug endpoint to manually cleanup old question logs"""
    try:
        question_logger = get_question_logger()
        archived_count = question_logger.cleanup_old_logs()
        
        return {
            "status": "success",
            "archived_files": archived_count,
            "message": f"Successfully archived {archived_count} old log files"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Question log cleanup failed: {str(e)}"
        )


@router.post("/universal-solver", response_model=SimpleQueryResponse)
async def universal_llm_solver(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """
    Universal LLM Solver endpoint for pure LLM-driven reasoning
    
    This endpoint bypasses traditional challenge detection and uses
    the Universal LLM Solver for all questions, providing pure
    LLM-driven reasoning with web scraping capabilities.
    """
    try:
        start_time = time.time()
        
        rag_coordinator = get_rag_coordinator()
        universal_solver = get_universal_llm_solver()
        
        # Force use of Universal LLM Solver
        debug_print("Using Universal LLM Solver for all questions")
        debug_print(f"Processing {len(request.questions)} questions")
        
        # Get document content if provided
        document_content = ""
        doc_id = None
        
        if request.documents and request.documents != 'web-scraping://no-document':
            debug_print(f"Processing document: {request.documents}")
            doc_result = await rag_coordinator.process_document(url=str(request.documents))
            doc_id = doc_result.get("doc_id")
            
            # Try to get complete document text
            if doc_id:
                complete_text = rag_coordinator._get_complete_document_text(doc_id)
                if complete_text:
                    document_content = complete_text
                    debug_print(f"Using complete document text ({len(document_content)} chars)")
        
        # Process questions with Universal LLM Solver
        answers = []
        processing_times = []
        
        for i, question in enumerate(request.questions):
            debug_print(f"Processing question {i+1}/{len(request.questions)}: {question[:50]}...")
            
            question_start_time = time.time()
            
            result = await universal_solver.solve(
                question=question,
                document_content=document_content,
                context={
                    "doc_id": doc_id,
                    "document_url": request.documents
                }
            )
            
            question_time = time.time() - question_start_time
            processing_times.append(question_time)
            
            if result.success:
                answers.append(result.answer)
                debug_print(f"  Question {i+1} completed in {question_time:.2f}s")
                if result.api_calls_made:
                    debug_print(f"  API calls made: {', '.join(result.api_calls_made)}")
            else:
                answers.append(f"Processing failed: {result.error}")
                debug_print(f"  Question {i+1} failed in {question_time:.2f}s: {result.error}")
        
        total_time = time.time() - start_time
        
        debug_print(f"\nUniversal LLM Solver completed:")
        debug_print(f"  - Total time: {total_time:.2f}s")
        debug_print(f"  - Average per question: {sum(processing_times)/len(processing_times):.2f}s")
        debug_print(f"  - Questions processed: {len(request.questions)}")
        
        return SimpleQueryResponse(
            answers=answers,
            processing_time=total_time,
            total_questions=len(request.questions),
            avg_time_per_question=sum(processing_times) / len(processing_times),
            processing_metadata={
                "solver_type": "universal_llm_solver",
                "document_provided": bool(document_content),
                "document_length": len(document_content),
                "individual_times": processing_times,
                "llm_driven": True
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Universal LLM Solver failed: {str(e)}"
        )



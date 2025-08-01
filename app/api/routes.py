"""
API routes for the RAG application
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from typing import List

from app.models.requests import QueryRequest, QueryResponse, HealthResponse
from app.services.rag_coordinator import get_rag_coordinator
from app.core.security import verify_token
from app.core.config import settings
from app.core.directories import get_directory_manager

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


@router.post("/hackrx/run", response_model=QueryResponse)
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
        
        # Step 1: Process document (with automatic caching)
        print(f"Processing document: {request.documents}")
        doc_start_time = time.time()
        doc_result = await rag_coordinator.process_document(str(request.documents))
        doc_processing_time = time.time() - doc_start_time
        doc_id = doc_result["doc_id"]
        
        if doc_result["status"] == "cached":
            print(f"Using cached document: {doc_id}")
        else:
            print(f"Processed new document: {doc_id}")
            print(f"  - Processing time: {doc_result['processing_time']:.2f}s")
            print(f"  - Chunks created: {doc_result['document_stats']['chunk_count']}")
            print(f"  - Preserved definitions: {doc_result['document_stats']['preserved_definitions']}")
        
        # Step 2: Answer each question
        answers = []
        print(f"Processing {len(request.questions)} questions")
        
        questions_start_time = time.time()
        individual_times = []
        
        for i, question in enumerate(request.questions):
            print(f"  Question {i+1}: {question[:50]}...")
            
            # Use RAG pipeline to answer question
            rag_response = rag_coordinator.answer_question(
                question=question,
                doc_id=doc_id,
                k_retrieve=10,  # Retrieve top 10 chunks
                max_context_length=6000  # Max context for LLM
            )
            
            # Add clean answer to results
            answers.append(rag_response.answer)
            
            # Track individual question time
            individual_times.append(rag_response.processing_time)
            
            print(f"    Answer generated in {rag_response.processing_time:.2f}s")
            print(f"    Used {len(rag_response.sources_used)} sources")
        
        # Calculate total times
        questions_processing_time = time.time() - questions_start_time
        total_processing_time = time.time() - total_start_time
        
        # Display comprehensive time metrics
        print(f"\n" + "="*60)
        print(f"📊 PROCESSING COMPLETED - TIME METRICS")
        print(f"="*60)
        print(f"Document Processing Time: {doc_processing_time:.2f}s")
        print(f"Questions Processing Time: {questions_processing_time:.2f}s")
        print(f"Total Processing Time: {total_processing_time:.2f}s")
        print(f"\nQuestion-by-Question Breakdown:")
        for i, q_time in enumerate(individual_times, 1):
            print(f"  Question {i}: {q_time:.2f}s")
        print(f"\nQuestions Statistics:")
        print(f"  - Total Questions: {len(request.questions)}")
        print(f"  - Average Time per Question: {sum(individual_times) / len(individual_times):.2f}s")
        print(f"  - Fastest Question: {min(individual_times):.2f}s")
        print(f"  - Slowest Question: {max(individual_times):.2f}s")
        print(f"  - Questions per Second: {len(request.questions) / questions_processing_time:.2f}")
        print(f"="*60)
        
        return QueryResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"RAG pipeline error: {str(e)}")
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
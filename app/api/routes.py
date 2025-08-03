"""
API routes for the RAG application
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.security import HTTPAuthorizationCredentials
from typing import List

from app.models.requests import (
    QueryRequest, QueryResponse, HealthResponse, UploadResponse, 
    UploadRequest, FileInfoResponse, FileListResponse
)
from app.services.rag_coordinator import get_rag_coordinator
from app.services.file_manager import get_file_manager
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
        # Parse questions JSON
        try:
            questions_list = json.loads(questions)
            if not isinstance(questions_list, list):
                raise ValueError("Questions must be a list")
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Questions must be valid JSON array"
            )
        
        file_manager = get_file_manager()
        rag_coordinator = get_rag_coordinator()
        
        # Upload the file
        file_info = await file_manager.save_uploaded_file(file)
        
        try:
            # Create upload URL for processing
            upload_url = f"upload://{file_info.file_id}"
            
            # Process document and answer questions
            doc_result = await rag_coordinator.process_document(upload_url)
            doc_id = doc_result["doc_id"]
            
            print(f"Processed uploaded document: {file_info.original_filename}")
            print(f"  - File ID: {file_info.file_id}")
            print(f"  - Document ID: {doc_id}")
            
            # Answer questions
            answers = []
            for question in questions_list:
                rag_response = rag_coordinator.answer_question(
                    question=question,
                    doc_id=doc_id,
                    k_retrieve=10,
                    max_context_length=6000
                )
                answers.append(rag_response.answer)
            
            print(f"Answered {len(questions_list)} questions for uploaded file")
            
            return QueryResponse(answers=answers)
            
        finally:
            # Clean up uploaded file after processing
            file_manager.remove_file(file_info.file_id)
            print(f"Cleaned up uploaded file: {file_info.file_id}")
        
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
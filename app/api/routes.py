"""
API routes for the application
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Header
from fastapi.security import HTTPAuthorizationCredentials
from typing import List, Optional
import time
import uuid

from app.models.requests import (
    QueryRequest, QueryResponse, HealthResponse, 
    DocumentUploadResponse, DocumentProcessingStatus
)
from app.services.document_service import DocumentService
from app.services.document_storage import DocumentStorageService
from app.services.enhanced_document_service import EnhancedDocumentService
from app.core.security import verify_token
from app.core.config import settings

router = APIRouter()


# Initialize services
storage_service = DocumentStorageService()
enhanced_document_service = EnhancedDocumentService(storage_service)

@router.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": f"{settings.app_name} is running",
        "version": settings.version,
        "features": {
            "file_upload": True,
            "ocr_fallback": settings.enable_ocr_fallback,
            "enhanced_processing": True
        }
    }

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy", 
        service="document-ingestion"
    )

@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    token: str = Depends(verify_token),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """
    Upload a document for processing
    
    This endpoint:
    1. Validates the uploaded file
    2. Generates a unique document ID
    3. Stores the raw document persistently
    4. Returns upload confirmation with document ID
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        # Read file content
        content = await file.read()
        
        # Validate file size
        if len(content) > settings.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size} bytes"
            )
        
        # Generate document ID
        document_id = storage_service.generate_document_id(content, file.filename)
        
        # Store document
        metadata = await storage_service.store_raw_document(
            document_id=document_id,
            content=content,
            filename=file.filename,
            source_type="upload",
            source_reference=file.filename
        )
        
        print(f"[UPLOAD] Document uploaded: {document_id} ({len(content):,} bytes)")
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            file_size=len(content),
            upload_timestamp=metadata.upload_timestamp,
            status="uploaded"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )

@router.get("/documents/{document_id}/status", response_model=DocumentProcessingStatus)
async def get_document_status(
    document_id: str,
    token: str = Depends(verify_token)
):
    """Get document processing status"""
    try:
        if not storage_service.document_exists(document_id):
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )
        
        metadata = await storage_service.load_metadata(document_id)
        artifacts = await storage_service.get_document_artifacts(document_id)
        
        return DocumentProcessingStatus(
            document_id=document_id,
            status=metadata.processing_status,
            processing_time_ms=metadata.processing_time_ms,
            error_message=metadata.error_message,
            artifacts=artifacts
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Status check failed: {str(e)}"
        )

@router.post("/documents/{document_id}/process", response_model=QueryResponse)
async def process_document_enhanced(
    document_id: str,
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """
    Process a stored document with enhanced extraction
    
    This endpoint:
    1. Processes the stored document using enhanced PyMuPDF4LLM extraction
    2. Processes the questions against the document content
    3. Returns structured answers
    """
    try:
        request_start_time = time.time()
        
        # Step 1: Enhanced document processing
        print(f"[ENHANCED] Processing stored document: {document_id}")
        document_processing_start = time.time()
        document_content = await enhanced_document_service.process_document_from_storage(
            document_id
        )
        document_processing_time = time.time() - document_processing_start
        
        print(f"[ENHANCED] Successfully processed PDF:")
        print(f"  - Pages: {document_content.pages}")
        print(f"  - Text length: {len(document_content.text)} characters")
        print(f"  - Processing method: {document_content.processing_method}")
        print(f"  - Document processing time: {document_processing_time:.3f}s")
        if hasattr(document_content, 'markdown') and document_content.markdown:
            print(f"  - Markdown length: {len(document_content.markdown)} characters")
        
        # Step 2: Process questions
        print(f"Processing {len(request.questions)} questions with enhanced extraction")
        question_processing_start = time.time()
        
        # TODO: Implement actual query processing pipeline
        # For now, return placeholder answers with document info
        placeholder_answers = [
            f"[Enhanced PyMuPDF4LLM] Document processed using {document_content.processing_method} extraction. Question {i+1}: {question[:50]}..."
            for i, question in enumerate(request.questions)
        ]
        
        question_processing_time = time.time() - question_processing_start
        total_request_time = time.time() - request_start_time
        
        print(f"[TIMING] Request completed:")
        print(f"  - Document processing: {document_processing_time:.3f}s")
        print(f"  - Question processing: {question_processing_time:.3f}s")
        print(f"  - Total request time: {total_request_time:.3f}s")
        
        return QueryResponse(answers=placeholder_answers)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Enhanced processing error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Enhanced processing failed: {str(e)}"
        )

@router.post("/hackrx/run", response_model=QueryResponse)
async def process_queries(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """
    Main endpoint to process document queries with enhanced extraction
    
    This endpoint:
    1. Downloads and processes the document using enhanced PyMuPDF4LLM extraction
    2. Extracts text with markdown formatting and span mapping
    3. Processes the questions (currently returns placeholders)
    4. Returns structured answers
    """
    try:
        request_start_time = time.time()
        
        # Step 1: Enhanced document processing
        print(f"[ENHANCED] Processing document from: {request.documents}")
        document_processing_start = time.time()
        document_content = await enhanced_document_service.process_document_from_url(
            str(request.documents)
        )
        document_processing_time = time.time() - document_processing_start
        
        print(f"[ENHANCED] Successfully processed PDF:")
        print(f"  - Pages: {document_content.pages}")
        print(f"  - Text length: {len(document_content.text)} characters")
        print(f"  - Processing method: {document_content.processing_method}")
        print(f"  - Document ID: {document_content.document_id}")
        print(f"  - Document processing time: {document_processing_time:.3f}s")
        if hasattr(document_content, 'markdown') and document_content.markdown:
            print(f"  - Markdown length: {len(document_content.markdown)} characters")
        
        # Step 2: Process questions
        print(f"Processing {len(request.questions)} questions with enhanced extraction")
        question_processing_start = time.time()
        
        # TODO: Implement actual query processing pipeline
        # For now, return placeholder answers with document info
        placeholder_answers = [
            f"[Enhanced PyMuPDF4LLM] Document processed using {document_content.processing_method} extraction. Document ID: {document_content.document_id}. Question {i+1}: {question[:50]}..."
            for i, question in enumerate(request.questions)
        ]
        
        question_processing_time = time.time() - question_processing_start
        total_request_time = time.time() - request_start_time
        
        print(f"[TIMING] Request completed:")
        print(f"  - Document processing: {document_processing_time:.3f}s")
        print(f"  - Question processing: {question_processing_time:.3f}s")
        print(f"  - Total request time: {total_request_time:.3f}s")
        
        return QueryResponse(answers=placeholder_answers)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Enhanced processing error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Enhanced processing failed: {str(e)}"
        )
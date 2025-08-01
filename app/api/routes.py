"""
API routes for the application
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Header
from fastapi.security import HTTPAuthorizationCredentials
from typing import List, Optional
import time
import uuid
import os
import json
from app.models.requests import (
    QueryRequest, QueryResponse, HealthResponse,
    DocumentUploadResponse, DocumentProcessingStatus
)
from app.services.document_service import DocumentService
from app.services.document_storage import DocumentStorageService
from app.services.enhanced_document_service import EnhancedDocumentService
from app.services.adaptive_segmentation import AdaptiveSegmentationService
from app.services.clause_extractor import ClauseExtractor
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
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        content = await file.read()

        if len(content) > settings.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size} bytes"
            )

        document_id = storage_service.generate_document_id(content, file.filename)

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
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/documents/{document_id}/status", response_model=DocumentProcessingStatus)
async def get_document_status(
    document_id: str,
    token: str = Depends(verify_token)
):
    try:
        if not storage_service.document_exists(document_id):
            raise HTTPException(status_code=404, detail="Document not found")

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
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.post("/documents/{document_id}/process", response_model=QueryResponse)
async def process_document_enhanced(
    document_id: str,
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    try:
        request_start_time = time.time()

        print(f"[ENHANCED] Processing stored document: {document_id}")
        document_processing_start = time.time()
        document_content = await enhanced_document_service.process_document_from_storage(document_id)
        document_processing_time = time.time() - document_processing_start

        # CLAUSE EXTRACTION STEP
        clause_extraction_start = time.time()
        clause_extractor = ClauseExtractor()
        clauses = clause_extractor.extract_clauses_from_markdown_text(document_content.markdown or document_content.text)
        clause_extraction_time = time.time() - clause_extraction_start
        print(f"[CLAUSES] Extracted {len(clauses)} clauses in {clause_extraction_time:.3f}s")

        # SAVE CLAUSES
        os.makedirs("clauses", exist_ok=True)
        with open(f"clauses/{document_id}.json", "w", encoding="utf-8") as f:
            json.dump(clauses, f, indent=2, ensure_ascii=False)

        placeholder_answers = [
            f"Clause Match: {clauses[i % len(clauses)][:150]}..." for i in range(len(request.questions))
        ]

        question_processing_time = time.time() - document_processing_start
        total_request_time = time.time() - request_start_time
        print(f"[TIMING] Doc: {document_processing_time:.3f}s | Clause: {clause_extraction_time:.3f}s | Qs: {question_processing_time:.3f}s | Total: {total_request_time:.3f}s")

        return QueryResponse(answers=placeholder_answers)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Enhanced processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced processing failed: {str(e)}")

@router.post("/hackrx/run", response_model=QueryResponse)
async def process_queries(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    try:
        request_start_time = time.time()

        print(f"[ENHANCED] Processing document from: {request.documents}")
        document_processing_start = time.time()
        document_content = await enhanced_document_service.process_document_from_url(str(request.documents))
        document_processing_time = time.time() - document_processing_start

        # CLAUSE EXTRACTION STEP
        clause_extraction_start = time.time()
        clause_extractor = ClauseExtractor()
        clauses = clause_extractor.extract_clauses_from_markdown_text(document_content.markdown or document_content.text)
        clause_extraction_time = time.time() - clause_extraction_start
        print(f"[CLAUSES] Extracted {len(clauses)} clauses in {clause_extraction_time:.3f}s")

        # SAVE CLAUSES
        os.makedirs("clauses", exist_ok=True)
        with open("clauses/from_url.json", "w", encoding="utf-8") as f:
            json.dump(clauses, f, indent=2, ensure_ascii=False)

        placeholder_answers = [
            f"Clause Match: {clauses[i % len(clauses)][:150]}..." for i in range(len(request.questions))
        ]

        question_processing_time = time.time() - document_processing_start
        total_request_time = time.time() - request_start_time
        print(f"[TIMING] Doc: {document_processing_time:.3f}s | Clause: {clause_extraction_time:.3f}s | Qs: {question_processing_time:.3f}s | Total: {total_request_time:.3f}s")

        return QueryResponse(answers=placeholder_answers)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Enhanced processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced processing failed: {str(e)}")

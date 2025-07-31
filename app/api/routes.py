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
from app.services.adaptive_segmentation import AdaptiveSegmentationService
from app.core.security import verify_token
from app.core.config import settings

router = APIRouter()

def _log_detailed_clauses(segmentation_result):
    """Log detailed information about each segmented clause"""
    if not settings.log_segmented_clauses or not segmentation_result.clauses:
        return
    
    print(f"\n[CLAUSE DETAILS] Segmented {len(segmentation_result.clauses)} clauses:")
    print("=" * 80)
    
    # Console logging (limited for readability)
    for i, clause in enumerate(segmentation_result.clauses[:50]):  # Limit to first 50 for readability
        # Truncate text for display
        text_preview = clause.text[:100] + "..." if len(clause.text) > 100 else clause.text
        text_preview = text_preview.replace('\n', ' ').replace('\r', ' ')
        
        print(f"[{clause.id}] Type: {clause.clause_type.value}")
        print(f"  Section: {clause.section_heading}")
        if clause.sub_section:
            print(f"  Sub-section: {clause.sub_section}")
        print(f"  Page: {clause.page_number}")
        print(f"  Length: {len(clause.text)} chars")
        print(f"  Text: {text_preview}")
        print(f"  Position: {clause.char_start}-{clause.char_end}")
        print("-" * 40)
    
    if len(segmentation_result.clauses) > 50:
        print(f"... and {len(segmentation_result.clauses) - 50} more clauses")
    
    # Summary by clause type
    clause_type_counts = {}
    for clause in segmentation_result.clauses:
        clause_type = clause.clause_type.value
        clause_type_counts[clause_type] = clause_type_counts.get(clause_type, 0) + 1
    
    print(f"\n[CLAUSE SUMMARY] Breakdown by type:")
    for clause_type, count in sorted(clause_type_counts.items()):
        print(f"  {clause_type}: {count} clauses")
    print("=" * 80)
    
    # Optional: Save detailed clause info to file
    if settings.save_clause_details:
        _save_clause_details_to_file(segmentation_result)

def _save_clause_details_to_file(segmentation_result):
    """Save detailed clause information to a file for inspection"""
    try:
        import json
        from datetime import datetime
        from pathlib import Path
        
        # Create logs directory if it doesn't exist
        logs_dir = Path("clause_logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = logs_dir / f"clauses_{timestamp}.json"
        
        # Prepare clause data for JSON serialization
        clause_data = {
            "timestamp": datetime.now().isoformat(),
            "total_clauses": len(segmentation_result.clauses),
            "processing_time_seconds": segmentation_result.processing_time_seconds,
            "pages_processed": segmentation_result.pages_processed,
            "sections_found": segmentation_result.sections_found,
            "errors": segmentation_result.errors,
            "warnings": segmentation_result.warnings,
            "clauses": []
        }
        
        for clause in segmentation_result.clauses:
            clause_dict = {
                "id": clause.id,
                "type": clause.clause_type.value,
                "text": clause.text,
                "section_heading": clause.section_heading,
                "sub_section": clause.sub_section,
                "page_number": clause.page_number,
                "char_start": clause.char_start,
                "char_end": clause.char_end,
                "source_type": clause.source_type.value,
                "created_at": clause.created_at.isoformat() if clause.created_at else None,
                "metadata": clause.metadata
            }
            clause_data["clauses"].append(clause_dict)
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(clause_data, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Detailed clause information saved to: {filename}")
        
    except Exception as e:
        print(f"[ERROR] Failed to save clause details to file: {str(e)}")

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
    2. Segments the document into clauses using adaptive strategies
    3. Processes the questions against the segmented content
    4. Returns structured answers with enhanced citations
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
        
        # Step 2: Segment document into clauses using adaptive strategy
        print("[INFO] Starting adaptive clause segmentation...")
        segmentation_start = time.time()
        segmentation_service = AdaptiveSegmentationService()
        segmentation_result = segmentation_service.segment_document(document_content)
        segmentation_time = time.time() - segmentation_start
        
        if segmentation_result.errors:
            print(f"[ERROR] Segmentation errors: {segmentation_result.errors}")
        if segmentation_result.warnings:
            print(f"[WARNING] Segmentation warnings: {segmentation_result.warnings}")
        
        print("[SUCCESS] Clause segmentation completed:")
        print(f"  - Total clauses: {len(segmentation_result.clauses)}")
        print(f"  - Segmentation time: {segmentation_time:.3f}s")
        print(f"  - Strategy processing time: {segmentation_result.processing_time_seconds}s")
        print(f"  - Clause types: {segmentation_result.clause_types_summary}")
        print(f"  - Segmentation speed: {len(segmentation_result.clauses) / segmentation_time:.1f} clauses/sec")
        
        # Log detailed clause information if enabled
        _log_detailed_clauses(segmentation_result)
        
        # Step 3: Process questions
        print(f"Processing {len(request.questions)} questions with enhanced extraction")
        question_processing_start = time.time()
        
        # TODO: Implement actual query processing pipeline with embeddings and span mapping
        # For now, return enhanced placeholder answers with enhanced extraction info
        placeholder_answers = [
            f"[Enhanced PyMuPDF4LLM] Based on {len(segmentation_result.clauses)} clauses from {document_content.processing_method} extraction. Question {i+1}: {question[:50]}..."
            for i, question in enumerate(request.questions)
        ]
        
        question_processing_time = time.time() - question_processing_start
        total_request_time = time.time() - request_start_time
        
        print(f"[TIMING] Request completed:")
        print(f"  - Document processing: {document_processing_time:.3f}s")
        print(f"  - Clause segmentation: {segmentation_time:.3f}s") 
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
    3. Segments the document into clauses using adaptive strategies
    4. Processes the questions (currently returns placeholders)
    5. Returns structured answers with enhanced citations
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
        
        # Step 2: Segment document into clauses using adaptive strategy
        print("[INFO] Starting adaptive clause segmentation...")
        segmentation_start = time.time()
        segmentation_service = AdaptiveSegmentationService()
        segmentation_result = segmentation_service.segment_document(document_content)
        segmentation_time = time.time() - segmentation_start
        
        if segmentation_result.errors:
            print(f"[ERROR] Segmentation errors: {segmentation_result.errors}")
        if segmentation_result.warnings:
            print(f"[WARNING] Segmentation warnings: {segmentation_result.warnings}")
        
        print("[SUCCESS] Clause segmentation completed:")
        print(f"  - Total clauses: {len(segmentation_result.clauses)}")
        print(f"  - Segmentation time: {segmentation_time:.3f}s")
        print(f"  - Strategy processing time: {segmentation_result.processing_time_seconds}s")
        print(f"  - Clause types: {segmentation_result.clause_types_summary}")
        print(f"  - Segmentation speed: {len(segmentation_result.clauses) / segmentation_time:.1f} clauses/sec")
        
        # Log detailed clause information if enabled
        _log_detailed_clauses(segmentation_result)
        
        # Show which strategy was successful
        if segmentation_result.clauses:
            strategy_used = segmentation_result.clauses[0].metadata.get('segmentation_metadata', {}).get('strategy_used', 'unknown')
            print(f"  - Strategy used: {strategy_used}")
        
        # Step 3: Process questions
        print(f"Processing {len(request.questions)} questions with enhanced extraction")
        question_processing_start = time.time()
        
        # TODO: Implement actual query processing pipeline with embeddings and span mapping
        # For now, return enhanced placeholder answers with enhanced extraction info
        placeholder_answers = [
            f"[Enhanced PyMuPDF4LLM] Based on {len(segmentation_result.clauses)} clauses from {document_content.processing_method} extraction. Document ID: {document_content.document_id}. Question {i+1}: {question[:50]}..."
            for i, question in enumerate(request.questions)
        ]
        
        question_processing_time = time.time() - question_processing_start
        total_request_time = time.time() - request_start_time
        
        print(f"[TIMING] Request completed:")
        print(f"  - Document processing: {document_processing_time:.3f}s")
        print(f"  - Clause segmentation: {segmentation_time:.3f}s")
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
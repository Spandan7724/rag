"""
API routes for the application
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from typing import List

from app.models.requests import QueryRequest, QueryResponse, HealthResponse
from app.services.document_service import DocumentService
from app.services.clause_segmentation import ClauseSegmentationService
from app.core.security import verify_token
from app.core.config import settings

router = APIRouter()

@router.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": f"{settings.app_name} is running",
        "version": settings.version
    }

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy", 
        service="document-ingestion"
    )

@router.post("/hackrx/run", response_model=QueryResponse)
async def process_queries(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """
    Main endpoint to process document queries
    
    This endpoint:
    1. Downloads the document from the provided URL
    2. Extracts text and metadata from the PDF
    3. Processes the questions (currently returns placeholders)
    4. Returns structured answers
    """
    try:
        # Step 1: Process document
        print(f"Processing document from: {request.documents}")
        document_content = await DocumentService.process_document(str(request.documents))
        
        print(f"Successfully processed PDF:")
        print(f"  - Pages: {document_content.pages}")
        print(f"  - Text length: {len(document_content.text)} characters")
        print(f"  - Title: {document_content.metadata.get('title', 'N/A')}")
        
        # Step 2: Segment document into clauses
        print("🔪 Starting clause segmentation...")
        segmentation_service = ClauseSegmentationService()
        segmentation_result = segmentation_service.segment_document(document_content)
        
        if segmentation_result.errors:
            print(f"⚠️  Segmentation errors: {segmentation_result.errors}")
        
        print(f"📝 Clause segmentation completed:")
        print(f"  - Total clauses: {len(segmentation_result.clauses)}")
        print(f"  - Processing time: {segmentation_result.processing_time_seconds}s")
        print(f"  - Clause types: {segmentation_result.clause_types_summary}")
        
        # Step 3: Process questions
        print(f"Processing {len(request.questions)} questions")
        
        # TODO: Implement actual query processing pipeline with embeddings
        # For now, return enhanced placeholder answers with clause info
        placeholder_answers = [
            f"[Based on {len(segmentation_result.clauses)} clauses] Placeholder answer for question {i+1}: {question[:50]}..."
            for i, question in enumerate(request.questions)
        ]
        
        return QueryResponse(answers=placeholder_answers)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Processing failed: {str(e)}"
        )
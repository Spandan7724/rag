"""
Request/Response models for the API
"""
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

class DocumentSource(BaseModel):
    """Document source - either URL or uploaded file reference"""
    url: Optional[HttpUrl] = None
    file_id: Optional[str] = None  # For uploaded files
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/document.pdf"
            }
        }

class QueryRequest(BaseModel):
    """Request model for document query processing"""
    documents: Union[HttpUrl, DocumentSource]  # Support both legacy and new format
    questions: List[str]
    request_id: Optional[str] = Field(None, description="Client-provided request ID for tracking")
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": "https://example.com/document.pdf",
                "questions": [
                    "What is the grace period for premium payment?",
                    "What are the coverage limits?"
                ],
                "request_id": "req_abc123"
            }
        }

class QueryResponse(BaseModel):
    """Response model for document query processing"""
    answers: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "answers": [
                    "The grace period for premium payment is 30 days.",
                    "The coverage limit is $100,000 per incident."
                ]
            }
        }

class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    document_id: str
    filename: str
    file_size: int
    upload_timestamp: datetime
    status: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_abc123def456",
                "filename": "policy.pdf",
                "file_size": 1024000,
                "upload_timestamp": "2025-01-30T10:30:00Z",
                "status": "uploaded"
            }
        }

class DocumentProcessingStatus(BaseModel):
    """Document processing status"""
    document_id: str
    status: str  # "uploaded", "processing", "completed", "failed"
    progress_percentage: Optional[int] = None
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    artifacts: Dict[str, str] = Field(default_factory=dict)  # artifact_type -> file_path
    
class DocumentContent(BaseModel):
    """Enhanced model for processed document content"""
    document_id: str
    text: str
    markdown: Optional[str] = None  # PyMuPDF4LLM output
    pages: int
    metadata: Dict[str, Any]
    span_mapping: Optional[Dict[str, Any]] = None  # Character span to bbox mapping
    ocr_pages: List[int] = Field(default_factory=list)  # Pages that required OCR
    processing_method: str = "pymupdf4llm"  # Always enhanced processing with PyMuPDF4LLM
    
class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    service: str
    
class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str
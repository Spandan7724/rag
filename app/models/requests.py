"""
Request/Response models for the API
"""
from pydantic import BaseModel, HttpUrl, field_validator
from typing import List, Dict, Any, Union, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    """Request model for document query processing with query transformation support"""
    documents: str  # Accept HTTP URLs, file:// paths, or upload file IDs
    questions: List[str]
    enable_query_transformation: Optional[bool] = None  # Override global transformation setting
    
    @field_validator('documents')
    @classmethod
    def validate_documents(cls, v: str) -> str:
        """Validate that documents is either a valid URL, file path, or upload file ID"""
        if v.startswith('file://'):
            # For file:// URLs, just ensure the path exists and is readable
            file_path = v[7:]  # Remove 'file://' prefix
            if not file_path:
                raise ValueError('file:// URL must specify a path')
            return v
        elif v.startswith('upload://'):
            # For upload file IDs, validate format
            file_id = v[9:]  # Remove 'upload://' prefix
            if not file_id or len(file_id) < 10:
                raise ValueError('upload:// URL must specify a valid file ID')
            return v
        else:
            # For HTTP URLs, validate using HttpUrl
            try:
                HttpUrl(v)
                return v
            except Exception:
                raise ValueError('documents must be a valid HTTP URL, file:// path, or upload:// file ID')
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": "https://example.com/document.pdf",
                "questions": [
                    "What is the grace period for premium payment?",
                    "What are the coverage limits?"
                ]
            },
            "examples": {
                "http_url": {
                    "summary": "HTTP URL Example",
                    "value": {
                        "documents": "https://example.com/document.pdf",
                        "questions": ["What is covered?", "What are the limits?"]
                    }
                },
                "local_file": {
                    "summary": "Local File Example", 
                    "value": {
                        "documents": "file:///absolute/path/to/document.pdf",
                        "questions": ["What is covered?", "What are the limits?"]
                    }
                },
                "uploaded_file": {
                    "summary": "Uploaded File Example",
                    "value": {
                        "documents": "upload://abc123def456ghi789",
                        "questions": ["What is covered?", "What are the limits?"]
                    }
                }
            }
        }

class SimpleQueryResponse(BaseModel):
    """Simple response model with just answers"""
    answers: List[str]

class QueryResponse(BaseModel):
    """Response model for document query processing with transformation metadata"""
    answers: List[str]
    sources: Optional[List[List[Dict[str, Any]]]] = None  # Source citations for each answer
    query_transformations: Optional[List[Dict[str, Any]]] = None  # Transformation info per query
    processing_summary: Optional[Dict[str, Any]] = None  # Overall processing statistics
    
    class Config:
        json_schema_extra = {
            "example": {
                "answers": [
                    "The grace period for premium payment is 30 days.",
                    "The coverage limit is $100,000 per incident."
                ],
                "sources": [
                    [
                        {
                            "number": 1,
                            "doc_id": "abc123",
                            "page": 5,
                            "heading": "Premium Payment Terms",
                            "section": "2.1",
                            "similarity_score": 0.92,
                            "text_preview": "Grace period provisions...",
                            "chunk_type": "text"
                        }
                    ],
                    [
                        {
                            "number": 1,
                            "doc_id": "abc123",
                            "page": 12,
                            "heading": "Coverage Limits",
                            "section": "4.3",
                            "similarity_score": 0.89,
                            "text_preview": "Maximum coverage amounts...",
                            "chunk_type": "table"
                        }
                    ]
                ],
                "query_transformations": [
                    {
                        "original_query": "What is the grace period?",
                        "transformation_successful": False,
                        "sub_queries": ["What is the grace period?"]
                    },
                    {
                        "original_query": "Compare coverage and limits",
                        "transformation_successful": True,
                        "sub_queries": [
                            "What coverage is provided?",
                            "What are the coverage limits?"
                        ]
                    }
                ],
                "processing_summary": {
                    "total_questions": 2,
                    "questions_with_transformation": 1,
                    "parallel_processing_time": 1.5,
                    "average_time_per_question": 0.75,
                    "embedding_provider": "gemini",
                    "embedding_dimension": 1024
                }
            }
        }

class DocumentContent(BaseModel):
    """Internal model for processed document content"""
    text: str
    pages: int
    metadata: Dict[str, Any]
    
class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    service: str
    
class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str


class UploadResponse(BaseModel):
    """Response model for file upload"""
    file_id: str
    original_filename: str
    file_size: int
    content_type: str
    uploaded_at: datetime
    expires_at: datetime
    upload_url: str  # The upload:// URL to use in queries
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "abc123def456ghi789",
                "original_filename": "policy_document.pdf",
                "file_size": 2048576,
                "content_type": "application/pdf",
                "uploaded_at": "2025-01-15T10:30:00",
                "expires_at": "2025-01-16T10:30:00",
                "upload_url": "upload://abc123def456ghi789"
            }
        }


class UploadRequest(BaseModel):
    """Request model for file upload with immediate processing and query transformation support"""
    questions: List[str]
    enable_query_transformation: Optional[bool] = None  # Override global transformation setting
    
    class Config:
        json_schema_extra = {
            "example": {
                "questions": [
                    "What is the grace period for premium payment?",
                    "What are the coverage limits?",
                    "What conditions are excluded?"
                ],
                "enable_query_transformation": True
            }
        }


class FileInfoResponse(BaseModel):
    """Response model for file information"""
    file_id: str
    original_filename: str
    file_size: int
    content_type: str
    uploaded_at: datetime
    expires_at: datetime
    expired: bool
    exists: bool


class FileListResponse(BaseModel):
    """Response model for listing uploaded files"""
    total_files: int
    files: List[FileInfoResponse]
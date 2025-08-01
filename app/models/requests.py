"""
Request/Response models for the API
"""
from pydantic import BaseModel, HttpUrl, field_validator
from typing import List, Dict, Any, Union

class QueryRequest(BaseModel):
    """Request model for document query processing"""
    documents: str  # Accept both HTTP URLs and file:// paths
    questions: List[str]
    
    @field_validator('documents')
    @classmethod
    def validate_documents(cls, v: str) -> str:
        """Validate that documents is either a valid HTTP URL or file:// path"""
        if v.startswith('file://'):
            # For file:// URLs, just ensure the path exists and is readable
            file_path = v[7:]  # Remove 'file://' prefix
            if not file_path:
                raise ValueError('file:// URL must specify a path')
            return v
        else:
            # For HTTP URLs, validate using HttpUrl
            try:
                HttpUrl(v)
                return v
            except Exception:
                raise ValueError('documents must be a valid HTTP URL or file:// path')
    
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
                }
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
"""
Request/Response models for the API
"""
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any

class QueryRequest(BaseModel):
    """Request model for document query processing"""
    documents: HttpUrl
    questions: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": "https://example.com/document.pdf",
                "questions": [
                    "What is the grace period for premium payment?",
                    "What are the coverage limits?"
                ]
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
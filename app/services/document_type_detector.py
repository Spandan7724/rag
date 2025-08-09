#!/usr/bin/env python3
"""
Document Type Detector Service
Identifies document types to route to appropriate processing pipeline
"""
import re
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

class DocumentType(Enum):
    """Document processing types"""
    HACKRX_CHALLENGE = "hackrx_challenge"
    SEMANTIC_SEARCH = "semantic_search"
    STRUCTURED_DATA = "structured_data"
    UNKNOWN = "unknown"

@dataclass
class DocumentTypeResult:
    """Document type detection result"""
    document_type: DocumentType
    confidence: float
    detected_patterns: list
    suggested_pipeline: str
    metadata: Dict[str, Any]

class DocumentTypeDetector:
    """
    Detects document types to route to appropriate processing pipelines
    """
    
    def __init__(self):
        """Initialize document type detector"""
        # HackRX challenge patterns
        self.hackrx_patterns = [
            # Challenge-specific terms
            r"hackrx|hack\s*rx|challenge|puzzle|parallel\s*world",
            r"sachin.{0,20}parallel\s*world",
            r"mission\s*brief|mission\s*objective",
            r"flight\s*number|get.*flight",
            r"landmark.{0,30}current\s*location",
            
            # Geographic puzzle indicators
            r"gateway\s*of\s*india|taj\s*mahal|eiffel\s*tower|big\s*ben",
            r"delhi|mumbai|chennai|hyderabad|new\s*york|london|tokyo",
            r"favorite\s*city|favourite\s*city",
            
            # API endpoint patterns
            r"register\.hackrx\.in|hackrx\.in",
            r"submissions/myFavouriteCity",
            r"getFirstCityFlightNumber|getSecondCityFlightNumber",
            
            # Challenge workflow patterns
            r"step\s*[1-9].*:|step\s*by\s*step|query\s*the\s*secret",
            r"decode\s*the\s*city|choose.*flight\s*path",
            r"final\s*deliverable|secret\s*token"
        ]
        
        # Structured data patterns (tables, mappings)
        self.structured_data_patterns = [
            r"landmark\s+current\s*location",
            r"[🏛️🗼🕌🏖️🌉🏰❤️⛩️🌸👑🪨🏢🌞✨🗽🕰️🏟️🎭✝️🏙️🖼️🌊🪨🕍🏞️🗿⛪🛰️🌆]+",  # Emojis
            r"(\w+\s*){1,3}\n\s*(\w+\s*){1,3}",  # Two-line patterns
            r"table|mapping|list.*items"
        ]
        
        # Semantic search document patterns (general text content)
        self.semantic_patterns = [
            r"policy|insurance|terms\s*and\s*conditions",
            r"coverage|premium|claim|benefit",
            r"article|section|chapter|paragraph",
            r"constitution|law|legal|regulation"
        ]
    
    def detect_document_type(self, text: str, url: str = None) -> DocumentTypeResult:
        """
        Detect document type from content and metadata
        
        Args:
            text: Document text content
            url: Optional document URL for additional context
            
        Returns:
            DocumentTypeResult with detection information
        """
        # Count pattern matches for each type
        hackrx_matches = self._count_pattern_matches(self.hackrx_patterns, text)
        structured_matches = self._count_pattern_matches(self.structured_data_patterns, text)
        semantic_matches = self._count_pattern_matches(self.semantic_patterns, text)
        
        detected_patterns = []
        metadata = {}
        
        # Check URL for additional context
        url_score = 0
        if url and "hackrx" in url.lower():
            url_score = 0.3
            detected_patterns.append("hackrx_url")
        
        # Calculate scores
        text_length = len(text)
        hackrx_score = min(hackrx_matches / max(text_length / 1000, 1), 1.0) + url_score
        structured_score = min(structured_matches / max(text_length / 1000, 1), 1.0)
        semantic_score = min(semantic_matches / max(text_length / 1000, 1), 1.0)
        
        # Special boost for HackRX indicators
        if any(re.search(pattern, text, re.IGNORECASE | re.MULTILINE) 
               for pattern in self.hackrx_patterns[:5]):  # Core HackRX patterns
            hackrx_score += 0.4
            detected_patterns.append("hackrx_core_patterns")
        
        # Document type decision logic with improved prioritization
        if hackrx_score > 0.6:
            document_type = DocumentType.HACKRX_CHALLENGE
            confidence = min(hackrx_score, 1.0)
            suggested_pipeline = "direct_processing"
            metadata = {
                "hackrx_indicators": hackrx_matches,
                "requires_api_calls": True,
                "structured_data": True
            }
            
        elif structured_score > 0.6 and structured_score > (semantic_score * 2) and hackrx_score < 0.3:
            # Only use structured processing if:
            # 1. High structured score (>0.6)
            # 2. Structured score significantly higher than semantic (2x)
            # 3. Low hackrx score (not a challenge doc)
            document_type = DocumentType.STRUCTURED_DATA
            confidence = min(structured_score, 1.0)
            suggested_pipeline = "direct_processing"
            metadata = {
                "structured_elements": structured_matches,
                "table_like": True
            }
            
        elif semantic_score > 0.1 or text_length > 500:
            # Prefer semantic search for:
            # 1. Any semantic indicators
            # 2. Documents longer than 500 chars (likely text content)
            document_type = DocumentType.SEMANTIC_SEARCH
            confidence = max(min(semantic_score, 1.0), 0.5)  # Minimum confidence of 0.5
            suggested_pipeline = "rag_pipeline"
            metadata = {
                "semantic_indicators": semantic_matches,
                "requires_similarity_search": True,
                "text_length": text_length
            }
            
        else:
            document_type = DocumentType.UNKNOWN
            confidence = 0.2
            suggested_pipeline = "rag_pipeline"  # Default fallback to RAG
            metadata = {
                "hackrx_score": hackrx_score,
                "structured_score": structured_score,
                "semantic_score": semantic_score,
                "fallback_reason": "no_clear_indicators"
            }
        
        # Add pattern details
        if hackrx_matches > 0:
            detected_patterns.append(f"hackrx_patterns({hackrx_matches})")
        if structured_matches > 0:
            detected_patterns.append(f"structured_patterns({structured_matches})")
        if semantic_matches > 0:
            detected_patterns.append(f"semantic_patterns({semantic_matches})")
        
        return DocumentTypeResult(
            document_type=document_type,
            confidence=confidence,
            detected_patterns=detected_patterns,
            suggested_pipeline=suggested_pipeline,
            metadata=metadata
        )
    
    def _count_pattern_matches(self, patterns: list, text: str) -> int:
        """Count total pattern matches in text"""
        total_matches = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            total_matches += len(matches)
        return total_matches
    
    def is_hackrx_document(self, text: str, url: str = None) -> bool:
        """Quick check if document is HackRX challenge type"""
        result = self.detect_document_type(text, url)
        return result.document_type == DocumentType.HACKRX_CHALLENGE
    
    def requires_direct_processing(self, text: str, url: str = None) -> bool:
        """Check if document should use direct processing instead of RAG"""
        result = self.detect_document_type(text, url)
        return result.suggested_pipeline == "direct_processing"


# Singleton instance
_document_type_detector = None

def get_document_type_detector() -> DocumentTypeDetector:
    """Get or create document type detector instance"""
    global _document_type_detector
    if _document_type_detector is None:
        _document_type_detector = DocumentTypeDetector()
    return _document_type_detector
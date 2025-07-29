"""
Data models for clause segmentation
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

class ClauseType(str, Enum):
    """Types of clauses found in policy documents"""
    DEFINITION = "definition"
    COVERAGE = "coverage" 
    EXCLUSION = "exclusion"
    CONDITION = "condition"
    PROCEDURE = "procedure"
    BENEFIT = "benefit"
    LIMITATION = "limitation"
    GENERAL = "general"

class SourceType(str, Enum):
    """Source of the text content"""
    NATIVE = "native"  # Direct PDF text extraction
    OCR = "ocr"       # OCR-processed text

@dataclass
class PageContent:
    """Content from a single page"""
    page_number: int
    raw_text: str
    cleaned_text: str
    char_start: int
    char_end: int

@dataclass
class Section:
    """A document section (e.g., DEFINITIONS, COVERAGE)"""
    title: str           # "DEFINITIONS"
    section_number: str  # "1"
    text: str           # Section content
    page_number: int    # Primary page number
    char_start: int     # Position in document
    char_end: int       # End position

@dataclass
class Clause:
    """A single clause extracted from the document"""
    id: str                          # Unique identifier "clause_001"
    text: str                        # Clean clause text
    page_number: int                 # Source page number
    section_heading: str             # "DEFINITIONS", "COVERAGE"
    sub_section: str                 # "1.1", "2.1", etc.
    clause_type: ClauseType          # Type of clause
    source_type: SourceType          # How text was extracted
    char_start: int                  # Character position in original document
    char_end: int                    # End character position
    metadata: Dict[str, Any]         # Additional context
    created_at: datetime             # When clause was extracted
    
    def __post_init__(self):
        """Validate and clean up clause data"""
        if not self.text.strip():
            raise ValueError("Clause text cannot be empty")
        
        # Clean up text
        self.text = " ".join(self.text.split())
        
        # Set creation time if not provided
        if not hasattr(self, 'created_at') or self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class ClauseExtractionResult:
    """Result of clause segmentation process"""
    clauses: List[Clause]
    pages_processed: int
    sections_found: int
    total_chars: int
    processing_time_seconds: float
    errors: List[str]
    warnings: List[str]
    
    @property
    def clause_count(self) -> int:
        """Total number of clauses extracted"""
        return len(self.clauses)
    
    @property
    def clause_types_summary(self) -> Dict[ClauseType, int]:
        """Count of each clause type"""
        summary = {}
        for clause in self.clauses:
            summary[clause.clause_type] = summary.get(clause.clause_type, 0) + 1
        return summary

@dataclass
class SegmentationStats:
    """Statistics about the segmentation process"""
    average_clause_length: float
    median_clause_length: float
    longest_clause_length: int
    shortest_clause_length: int
    clauses_per_page: float
    sections_by_type: Dict[str, int]
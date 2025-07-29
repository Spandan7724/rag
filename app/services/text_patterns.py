"""
Text pattern recognition utilities for clause segmentation
"""
import re
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

@dataclass
class PatternMatch:
    """A pattern match in text"""
    text: str
    start: int
    end: int
    pattern_type: str
    metadata: Dict[str, str]

class TextPatterns:
    """Pattern recognition for policy document structures"""
    
    # Page markers
    PAGE_MARKER_PATTERN = r'---\s*Page\s+(\d+)\s*---'
    
    # Section headers (e.g., "1 DEFINITIONS", "2 COVERAGE")
    SECTION_HEADER_PATTERN = r'^(\d+)\s+([A-Z][A-Z\s&/\-]{2,})\s*$'
    
    # Sub-clauses (e.g., "1.1", "2.1", "3.1.1")
    SUB_CLAUSE_PATTERN = r'^(\d+(?:\.\d+)*)\s+'
    
    # List items (a), (b), (c) or a), b), c)
    LIST_ITEM_PATTERN = r'^\s*\(?([a-z]|[ivx]+)\)\s+'
    
    # Bullet points
    BULLET_PATTERN = r'^\s*[•·▪▫-]\s+'
    
    # Definition patterns
    DEFINITION_PATTERN = r'([A-Za-z\s]+)\s+means\s+'
    
    # Policy titles (ALL CAPS lines)
    TITLE_PATTERN = r'^[A-Z][A-Z\s&/\-]{10,}$'
    
    @classmethod
    def extract_page_markers(cls, text: str) -> List[PatternMatch]:
        """Extract page markers from text"""
        matches = []
        for match in re.finditer(cls.PAGE_MARKER_PATTERN, text, re.MULTILINE):
            matches.append(PatternMatch(
                text=match.group(0),
                start=match.start(),
                end=match.end(),
                pattern_type="page_marker",
                metadata={"page_number": match.group(1)}
            ))
        return matches
    
    @classmethod
    def extract_section_headers(cls, text: str) -> List[PatternMatch]:
        """Extract section headers (1. DEFINITIONS, etc.)"""
        matches = []
        for match in re.finditer(cls.SECTION_HEADER_PATTERN, text, re.MULTILINE):
            matches.append(PatternMatch(
                text=match.group(0),
                start=match.start(),
                end=match.end(),
                pattern_type="section_header",
                metadata={
                    "section_number": match.group(1),
                    "section_title": match.group(2).strip()
                }
            ))
        return matches
    
    @classmethod
    def extract_sub_clauses(cls, text: str) -> List[PatternMatch]:
        """Extract sub-clause markers (1.1, 2.1, etc.)"""
        matches = []
        for match in re.finditer(cls.SUB_CLAUSE_PATTERN, text, re.MULTILINE):
            matches.append(PatternMatch(
                text=match.group(0),
                start=match.start(),
                end=match.end(),
                pattern_type="sub_clause",
                metadata={"clause_number": match.group(1)}
            ))
        return matches
    
    @classmethod
    def extract_list_items(cls, text: str) -> List[PatternMatch]:
        """Extract list items (a), (b), etc."""
        matches = []
        for match in re.finditer(cls.LIST_ITEM_PATTERN, text, re.MULTILINE):
            matches.append(PatternMatch(
                text=match.group(0),
                start=match.start(),
                end=match.end(),
                pattern_type="list_item",
                metadata={"item_marker": match.group(1)}
            ))
        return matches
    
    @classmethod
    def extract_definitions(cls, text: str) -> List[PatternMatch]:
        """Extract definition patterns"""
        matches = []
        for match in re.finditer(cls.DEFINITION_PATTERN, text, re.IGNORECASE):
            matches.append(PatternMatch(
                text=match.group(0),
                start=match.start(),
                end=match.end(),
                pattern_type="definition",
                metadata={"term": match.group(1).strip()}
            ))
        return matches
    
    @classmethod
    def extract_titles(cls, text: str) -> List[PatternMatch]:
        """Extract policy titles (ALL CAPS)"""
        matches = []
        for match in re.finditer(cls.TITLE_PATTERN, text, re.MULTILINE):
            matches.append(PatternMatch(
                text=match.group(0),
                start=match.start(),
                end=match.end(),
                pattern_type="title",
                metadata={"title": match.group(0).strip()}
            ))
        return matches
    
    @classmethod
    def split_by_pages(cls, text: str) -> List[Tuple[int, str, int, int]]:
        """Split text by page markers
        
        Returns:
            List of (page_number, page_text, start_pos, end_pos)
        """
        page_markers = cls.extract_page_markers(text)
        pages = []
        
        for i, marker in enumerate(page_markers):
            page_num = int(marker.metadata["page_number"])
            start_pos = marker.end
            
            # Find end position (next page marker or end of text)
            if i + 1 < len(page_markers):
                end_pos = page_markers[i + 1].start
            else:
                end_pos = len(text)
            
            page_text = text[start_pos:end_pos].strip()
            pages.append((page_num, page_text, start_pos, end_pos))
        
        return pages
    
    @classmethod
    def clean_text(cls, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        
        # Normalize dashes
        text = re.sub(r'[–—]', '-', text)
        
        return text
    
    @classmethod
    def extract_sentences(cls, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with spaCy/NLTK)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
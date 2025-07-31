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
    
    # Section headers - updated for PyMuPDF4LLM markdown format
    # Matches: "**1.** **PREAMBLE**", "**2.** **OPERATIVE CLAUSE**", "**3.** **DEFINITIONS**"
    SECTION_HEADER_PATTERNS = [
        (r'^\*\*(\d+)\.\*\*\s+\*\*([A-Z][A-Z\s&/\-]{2,})\*\*\s*$', "markdown_bold_section"),
        (r'^(\d+)\.\s+([A-Z][A-Z\s&/\-]{2,})\s*$', "plain_section"),
        (r'^(\d+)\s+([A-Z][A-Z\s&/\-]{2,})\s*$', "legacy_section"),
    ]
    
    # Sub-clauses - updated for PyMuPDF4LLM markdown format
    SUB_CLAUSE_PATTERNS = [
        # Pattern 1: PyMuPDF4LLM format "**3.1.** **Accident** means..." 
        (r'^\*\*(\d+(?:\.\d+)*)\.\*\*\s+\*\*([^*]+)\*\*\s+', "markdown_bold_definition"),
        # Pattern 2: PyMuPDF4LLM format "**3.1.** text without bold definition"
        (r'^\*\*(\d+(?:\.\d+)*)\.\*\*\s+', "markdown_bold_numbered"),
        # Pattern 3: Legacy "3.1. text" (with period after number - Arogya Sanjeevani format)
        (r'^(\d+(?:\.\d+)*)\.\s+', "numbered_with_period"),
        # Pattern 4: Legacy "3.1 text" (no period after number - original format)  
        (r'^(\d+(?:\.\d+)*)\s+(?![A-Z\s&/\-]{3,})', "numbered_space_only"),
        # Pattern 5: Roman numerals "i. text", "ii. text"
        (r'^([ivxlcdm]+)\.\s+', "roman_with_period"),
        # Pattern 6: Letters "a. text", "b. text"  
        (r'^([a-z])\.\s+', "letter_with_period"),
        # Pattern 7: Parenthetical "(i) text", "(a) text"
        (r'^\(([ivxlcdm]+|[a-z])\)\s+', "parenthetical"),
        # Pattern 8: Spaced period format "10.15.  text" (with extra spaces)
        (r'^(\d+(?:\.\d+)*)\.\s{2,}', "numbered_with_multiple_spaces"),
    ]
    
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
        """Extract section headers with multiple pattern attempts and exclusion filtering"""
        matches = []
        
        # Multiple patterns to handle different document formats including PyMuPDF4LLM markdown
        # IMPORTANT: Most specific patterns first to avoid less specific patterns capturing them
        patterns = [
            # PyMuPDF4LLM markdown format: "**1.** **PREAMBLE**"
            (r'^\*\*(\d+)\.\*\*\s+\*\*([A-Z][A-Z\s&/\-]{2,})\*\*\s*$', "markdown_bold_section"),
            # Standalone caps with colon: "DEFINITIONS:", "PREAMBLE:"  
            (r'^([A-Z][A-Z\s&/\-]{3,})\s*:\s*$', "caps_colon"),
            # Pure standalone caps: "DEFINITIONS", "PREAMBLE", "AYUSH"
            (r'^([A-Z][A-Z\s&/\-]{3,})\s*$', "caps_only"),
            # Current format: "2 DEFINITIONS"
            (r'^(\d+)\s+([A-Z][A-Z\s&/\-]{2,})\s*$', "numbered_space"),
            # Traditional format: "2. DEFINITIONS"  
            (r'^(\d+)\.\s*([A-Z][A-Z\s&/\-]{2,})\s*$', "numbered_period"),
            # Roman numerals: "II. DEFINITIONS"
            (r'^([IVX]+)\.\s*([A-Z][A-Z\s&/\-]{2,})\s*$', "roman_period"),
        ]
        
        for pattern, pattern_name in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                # Handle different group structures
                groups = match.groups()
                if len(groups) == 2:
                    section_number, section_title = groups
                elif len(groups) == 1:
                    # Single group patterns (caps_colon, caps_only)
                    section_number = ""
                    section_title = groups[0].rstrip(':')
                else:
                    continue
                
                # Skip if this looks like an exclusion list item
                if cls._is_exclusion_list_item(section_title, text, match.start()):
                    continue
                
                matches.append(PatternMatch(
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    pattern_type="section_header",
                    metadata={
                        "section_number": section_number,
                        "section_title": section_title.strip(),
                        "pattern_used": pattern_name
                    }
                ))
        
        # Sort by position and remove duplicates
        matches.sort(key=lambda x: x.start)
        return cls._remove_overlapping_matches(matches)
    
    @classmethod
    def extract_sub_clauses(cls, text: str) -> List[PatternMatch]:
        """Extract sub-clause markers using multiple patterns for different document formats"""
        matches = []
        
        # Try each pattern and collect all matches
        for pattern, pattern_name in cls.SUB_CLAUSE_PATTERNS:
            pattern_matches = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))
            
            for match in pattern_matches:
                clause_number = match.group(1)
                matches.append(PatternMatch(
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    pattern_type="sub_clause",
                    metadata={
                        "clause_number": clause_number,
                        "pattern_used": pattern_name
                    }
                ))
        
        # Sort by position and remove overlapping matches
        matches.sort(key=lambda x: x.start)
        return cls._remove_overlapping_matches(matches)
    
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
    def extract_exclusion_list_items(cls, text: str) -> List[PatternMatch]:
        """Extract individual items from exclusion lists (Annexure, etc.)"""
        matches = []
        
        # Pattern for numbered exclusion items like "1 BABY FOOD", "57 NEBULISATION KIT"
        exclusion_item_patterns = [
            (r'^(\d+)\s+([A-Z][A-Z\s/\-&\(\)]+)$', "numbered_exclusion"),
            (r'^([A-Z][A-Z\s/\-&\(\)]{3,})$', "caps_exclusion"),
        ]
        
        for pattern, pattern_name in exclusion_item_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                # Check if this is in an exclusion context
                if cls._is_exclusion_list_item(match.group(0), text, match.start()):
                    groups = match.groups()
                    if len(groups) == 2:
                        item_number, item_name = groups
                    else:
                        item_number = ""
                        item_name = groups[0] if groups else match.group(0)
                    
                    matches.append(PatternMatch(
                        text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        pattern_type="exclusion_item",
                        metadata={
                            "item_number": item_number,
                            "item_name": item_name.strip(),
                            "pattern_used": pattern_name
                        }
                    ))
        
        return matches
    
    @classmethod
    def split_by_pages(cls, text: str) -> List[Tuple[int, str, int, int]]:
        """Split text by page markers or treat as single page if no markers found
        
        Returns:
            List of (page_number, page_text, start_pos, end_pos)
        """
        page_markers = cls.extract_page_markers(text)
        pages = []
        
        if page_markers:
            # Traditional page marker splitting
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
        else:
            # No page markers found (e.g., PyMuPDF4LLM text) - treat as single page
            pages.append((1, text.strip(), 0, len(text)))
        
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
    
    @classmethod
    def _is_exclusion_list_item(cls, title: str, full_text: str, position: int) -> bool:
        """Check if a potential section header is actually an exclusion list item"""
        title_lower = title.lower()
        
        # Check if we're in an exclusion/annexure context
        context_start = max(0, position - 1000)
        context_end = min(len(full_text), position + 1000)
        context = full_text[context_start:context_end].lower()
        
        # Strong indicators this is an exclusion list
        exclusion_context_indicators = [
            'annexure', 'list of which coverage is not available',
            'exclusion', 'not covered', 'excluded items'
        ]
        
        is_in_exclusion_context = any(indicator in context for indicator in exclusion_context_indicators)
        
        if is_in_exclusion_context:
            # Common exclusion list item patterns
            exclusion_item_indicators = [
                'charges', 'kit', 'equipment', 'services', 'food', 'utilities',
                'treatment', 'procedure', 'collar', 'bandage', 'mask', 'pad',
                'bag', 'belt', 'cover', 'tape', 'solution', 'container'
            ]
            
            # If title contains exclusion item indicators, it's likely a list item
            if any(indicator in title_lower for indicator in exclusion_item_indicators):
                return True
            
            # Very short titles (< 15 chars) in caps in exclusion context are list items
            if len(title) < 15 and title.isupper():
                return True
            
            # Enhanced consecutive pattern detection
            lines_before = full_text[max(0, position-2000):position].split('\n')[-20:]
            lines_after = full_text[position:position+2000].split('\n')[:20]
            
            # Count different types of list patterns
            numbered_items = 0
            caps_items = 0
            consecutive_items = 0
            
            all_lines = lines_before + lines_after
            for i, line in enumerate(all_lines):
                line_clean = line.strip()
                if not line_clean or len(line_clean) > 100:
                    continue
                
                # Pattern 1: "23 ITEM NAME"
                if re.match(r'^\d+\s+[A-Z][A-Z\s/\-&\(\)]+$', line_clean):
                    numbered_items += 1
                    
                # Pattern 2: "ITEM NAME" (standalone caps)
                elif (line_clean.isupper() and len(line_clean) < 60 and 
                      len(line_clean) > 3):
                    caps_items += 1
                
                # Check for consecutive numbering
                if i > 0 and numbered_items > 0:
                    prev_line = all_lines[i-1].strip()
                    if (re.match(r'^\d+\s+', line_clean) and 
                        re.match(r'^\d+\s+', prev_line)):
                        consecutive_items += 1
            
            # Strong indicators of list sections
            total_list_items = numbered_items + caps_items
            
            # If many similar patterns nearby, this is likely a list section
            if (total_list_items > 10 or  # Many list-like items
                consecutive_items > 5 or  # Sequential numbering
                (numbered_items > 6 and consecutive_items > 3)):  # Mixed strong signals
                return True
        
        return False
    
    @classmethod
    def _remove_overlapping_matches(cls, matches: List[PatternMatch]) -> List[PatternMatch]:
        """Remove overlapping matches, keeping the first one found"""
        if not matches:
            return matches
        
        filtered = [matches[0]]
        
        for match in matches[1:]:
            # Check if this match overlaps with the last kept match
            last_match = filtered[-1]
            if match.start >= last_match.end:
                filtered.append(match)
        
        return filtered
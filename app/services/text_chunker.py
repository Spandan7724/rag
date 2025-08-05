"""
Text chunking service with smart section preservation for policy documents
"""
import re
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from app.core.config import settings


@dataclass
class TextChunk:
    """Container for a single text chunk with metadata"""
    chunk_id: int
    text: str
    token_count: int
    char_start: int
    char_end: int
    page: int
    heading: str
    section_number: Optional[str] = None
    chunk_type: str = "content"  # content, definition, section_header


class TextChunker:
    """Smart text chunker for policy documents"""
    
    def __init__(self):
        """Initialize text chunker"""
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Standard tokenizer
        self.max_tokens = settings.chunk_size  # Use configured chunk size from settings
        self.overlap_tokens = settings.chunk_overlap  # Use configured overlap from settings
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def detect_section_headers(self, text: str) -> List[Tuple[int, str, str]]:
        """
        Detect section headers and their positions
        
        Returns:
            List of (position, section_number, heading) tuples
        """
        headers = []
        
        # Pattern for numbered sections (3.1, 3.22, etc.)
        section_pattern = r'^(\d+\.\d+\.?)\s+(.+?)(?=\n|$)'
        
        for match in re.finditer(section_pattern, text, re.MULTILINE):
            pos = match.start()
            section_num = match.group(1).rstrip('.')
            heading = match.group(2).strip()
            headers.append((pos, section_num, heading))
        
        # Pattern for main sections (1. PREAMBLE, 2. OPERATIVE CLAUSE, etc.)
        main_section_pattern = r'^(\d+\.)\s+([A-Z][A-Z\s]+)(?=\n|$)'
        
        for match in re.finditer(main_section_pattern, text, re.MULTILINE):
            pos = match.start()
            section_num = match.group(1).rstrip('.')
            heading = match.group(2).strip()
            headers.append((pos, section_num, heading))
        
        # Sort by position
        headers.sort(key=lambda x: x[0])
        return headers
    
    def extract_page_numbers(self, text: str) -> Dict[int, int]:
        """
        Extract page numbers and their positions in text
        
        Returns:
            Dict mapping character position to page number
        """
        page_map = {}
        
        # Look for page markers (--- Page X ---)
        page_pattern = r'--- Page (\d+) ---'
        current_page = 1
        
        for match in re.finditer(page_pattern, text):
            pos = match.start()
            page_num = int(match.group(1))
            page_map[pos] = page_num
            current_page = page_num
        
        return page_map
    
    def get_page_for_position(self, pos: int, page_map: Dict[int, int]) -> int:
        """Get page number for a character position"""
        current_page = 1
        
        for page_pos, page_num in sorted(page_map.items()):
            if pos >= page_pos:
                current_page = page_num
            else:
                break
        
        return current_page
    
    def get_section_for_position(self, pos: int, headers: List[Tuple[int, str, str]]) -> Tuple[str, str]:
        """Get current section number and heading for a position"""
        current_section = ""
        current_heading = ""
        
        for header_pos, section_num, heading in headers:
            if pos >= header_pos:
                current_section = section_num
                current_heading = heading
            else:
                break
        
        return current_section, current_heading
    
    def is_definition_section(self, text: str) -> bool:
        """Check if a chunk contains a definition (more restrictive)"""
        # Only consider sections with clear definition structure
        definition_patterns = [
            r'^\d+\.\d+\.?\s+[A-Z][a-z\s]+(means|refers to|defined as)',  # "3.22. Grace Period means"
            r'^[A-Z][a-z\s]+:?\s+(means|refers to|defined as)',  # "Grace Period: means" or "Grace Period means"
        ]
        
        # Check if it's a short section (likely a real definition)
        if len(text) > 5000:  # If section is too large, probably not a single definition
            return False
        
        text_lower = text.lower()
        for pattern in definition_patterns:
            if re.search(pattern, text_lower, re.MULTILINE):
                return True
        
        return False
    
    def preserve_definitions(self, text: str, headers: List[Tuple[int, str, str]]) -> List[Tuple[int, int, str]]:
        """
        Identify definition sections that should be kept intact (with size limits)
        
        Returns:
            List of (start_pos, end_pos, reason) for sections to preserve
        """
        preserve_sections = []
        max_definition_size = 2000  # Maximum size for a definition section
        
        # Find definition sections from headers (only if they're reasonably sized)
        for i, (pos, section_num, heading) in enumerate(headers):
            next_pos = headers[i + 1][0] if i + 1 < len(headers) else len(text)
            section_text = text[pos:next_pos]
            
            # Only preserve if it's a reasonable size and actually a definition
            if len(section_text) <= max_definition_size and self.is_definition_section(section_text):
                preserve_sections.append((pos, next_pos, f"Definition section: {section_num} {heading}"))
        
        # Look for specific important definitions (with better boundary detection)
        important_definitions = [
            r'grace period.*?means.*?(?:thirty|30)\s*days',
            r'waiting period.*?means.*?(?:years?|months?|days?)',
            r'pre[-\s]?existing.*?disease.*?means',
        ]
        
        for pattern in important_definitions:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                start = match.start()
                end = match.end()
                
                # Find better boundaries - look for sentence or small paragraph
                # Look backwards for sentence start
                para_start = start
                for i in range(start, max(0, start - 500), -1):  # Look back max 500 chars
                    if text[i:i+2] == '\n\n':  # Paragraph break
                        para_start = i + 2
                        break
                    elif i > 0 and text[i-1:i+1] == '. ' and text[i].isupper():  # Sentence start
                        para_start = i
                        break
                
                # Look forwards for sentence/paragraph end
                para_end = end
                for i in range(end, min(len(text), end + 800)):  # Look forward max 800 chars
                    if text[i:i+2] == '\n\n':  # Paragraph break
                        para_end = i
                        break
                    elif text[i:i+2] == '. ' and (i+2 >= len(text) or text[i+2].isupper() or text[i+2] == '\n'):  # Sentence end
                        para_end = i + 1
                        break
                
                # Only preserve if it's a reasonable size
                if para_end - para_start <= max_definition_size:
                    preserve_sections.append((para_start, para_end, f"Important definition: {match.group()[:50]}..."))
        
        # Remove overlaps and merge adjacent sections (with size limits)
        preserve_sections.sort(key=lambda x: x[0])
        merged_sections = []
        max_merged_size = 3000  # Maximum size after merging
        
        for start, end, reason in preserve_sections:
            if merged_sections and start <= merged_sections[-1][1]:
                # Check if merging would create too large a section
                prev_start, prev_end, prev_reason = merged_sections[-1]
                merged_end = max(end, prev_end)
                
                if merged_end - prev_start <= max_merged_size:
                    # Safe to merge
                    merged_sections[-1] = (prev_start, merged_end, f"{prev_reason}; {reason}")
                else:
                    # Don't merge - would be too large
                    merged_sections.append((start, end, reason))
            else:
                merged_sections.append((start, end, reason))
        
        return merged_sections
    
    def create_chunk(
        self, 
        chunk_id: int, 
        text: str, 
        char_start: int, 
        char_end: int,
        page_map: Dict[int, int],
        headers: List[Tuple[int, str, str]],
        chunk_type: str = "content"
    ) -> TextChunk:
        """Create a TextChunk with proper metadata"""
        
        # Get page number
        page = self.get_page_for_position(char_start, page_map)
        
        # Get section info
        section_num, heading = self.get_section_for_position(char_start, headers)
        
        # Count tokens
        token_count = self.count_tokens(text)
        
        return TextChunk(
            chunk_id=chunk_id,
            text=text.strip(),
            token_count=token_count,
            char_start=char_start,
            char_end=char_end,
            page=page,
            heading=heading,
            section_number=section_num,
            chunk_type=chunk_type
        )
    
    def chunk_text(self, text: str, document_metadata: Dict[str, Any]) -> List[TextChunk]:
        """
        Chunk text into small uniform chunks
        
        Args:
            text: Full document text
            document_metadata: Document metadata
            
        Returns:
            List of TextChunk objects
        """
        print(f"Starting uniform text chunking...")
        print(f"Document length: {len(text):,} characters")
        print(f"Target: {self.max_tokens} tokens per chunk with {self.overlap_tokens} token overlap")
        
        # Simple page detection for metadata
        page_map = self.extract_page_numbers(text)
        headers = self.detect_section_headers(text)
        
        chunks = []
        chunk_id = 0
        pos = 0
        
        while pos < len(text):
            # Calculate rough character end based on token estimate (1 token ≈ 4 chars)
            target_end = min(pos + self.max_tokens * 4, len(text))
            
            # Smart chunking to preserve table structure
            chunk_end = self._find_smart_break_point(text, pos, target_end)
            
            # Extract chunk text and preserve structure
            chunk_text = text[pos:chunk_end].strip()
            
            # Detect and mark table chunks
            chunk_type = "table" if self._is_table_content(chunk_text) else "content"
            
            if chunk_text:  # Only create non-empty chunks
                chunk = self.create_chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    char_start=pos,
                    char_end=chunk_end,
                    page_map=page_map,
                    headers=headers,
                    chunk_type=chunk_type
                )
                
                chunks.append(chunk)
                chunk_id += 1
                
                # Move position with overlap (convert tokens to approximate chars)
                overlap_chars = self.overlap_tokens * 4
                pos = max(chunk_end - overlap_chars, pos + 1)
            else:
                pos = chunk_end
        
        # Final statistics  
        total_tokens = sum(chunk.token_count for chunk in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        table_chunks = sum(1 for chunk in chunks if chunk.chunk_type == "table")
        
        print(f"Smart chunking completed:")
        print(f"  - Total chunks: {len(chunks)}")
        print(f"  - Table chunks: {table_chunks}")
        print(f"  - Total tokens: {total_tokens:,}")
        print(f"  - Average tokens per chunk: {avg_tokens:.1f}")
        print(f"  - Chunk size optimized for short clauses (<600 chars)")
        
        return chunks
    
    def _find_smart_break_point(self, text: str, start: int, target_end: int) -> int:
        """Find a smart break point that preserves table structure and sentence boundaries"""
        
        if target_end >= len(text):
            return len(text)
        
        # Look for good break points in order of preference
        search_window = min(200, target_end - start)  # Search within 200 chars of target
        
        # 1. Try to break at paragraph boundaries (double newlines)
        for i in range(target_end, max(start, target_end - search_window), -1):
            if i < len(text) - 1 and text[i:i+2] == '\n\n':
                return i
        
        # 2. Try to break at single line breaks (preserve table rows)
        for i in range(target_end, max(start, target_end - search_window), -1):
            if text[i] == '\n':
                return i + 1  # Include the newline
        
        # 3. Try to break at sentence boundaries
        for i in range(target_end, max(start, target_end - search_window), -1):
            if i < len(text) - 1 and text[i] == '.' and text[i+1] == ' ':
                return i + 1
        
        # 4. Fallback to target end
        return target_end
    
    def _is_table_content(self, text: str) -> bool:
        """Detect if chunk contains table-like content"""
        
        lines = text.split('\n')
        if len(lines) < 2:
            return False
        
        # Check for table indicators
        table_indicators = 0
        
        # Count lines with percentage symbols (common in benefits tables)
        percentage_lines = sum(1 for line in lines if '%' in line)
        if percentage_lines >= 2:
            table_indicators += 2
        
        # Count lines with specific patterns (Sum Insured, Plan A/B, etc.)
        pattern_lines = sum(1 for line in lines if any(pattern in line.lower() for pattern in [
            'plan a', 'plan b', 'sum insured', 'sub-limit', 'coverage', 'benefit'
        ]))
        if pattern_lines >= 2:
            table_indicators += 1
        
        # Count lines with multiple spaces (table alignment)
        spaced_lines = sum(1 for line in lines if '  ' in line)  # Two or more spaces
        if spaced_lines >= len(lines) // 2:
            table_indicators += 1
        
        return table_indicators >= 2


# Singleton instance
_text_chunker = None

def get_text_chunker() -> TextChunker:
    """Get singleton text chunker instance"""
    global _text_chunker
    if _text_chunker is None:
        _text_chunker = TextChunker()
    return _text_chunker
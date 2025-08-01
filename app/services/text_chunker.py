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
        self.max_tokens = settings.chunk_size
        self.overlap_tokens = settings.chunk_overlap
    
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
        Chunk text with smart section preservation
        
        Args:
            text: Full document text
            document_metadata: Document metadata
            
        Returns:
            List of TextChunk objects
        """
        print(f"Starting text chunking...")
        print(f"Document length: {len(text):,} characters")
        
        # Detect structure
        headers = self.detect_section_headers(text)
        page_map = self.extract_page_numbers(text)
        preserve_sections = self.preserve_definitions(text, headers)
        
        print(f"Found {len(headers)} section headers")
        print(f"Found {len(preserve_sections)} sections to preserve intact")
        
        # Debug: Show what we're preserving
        for start, end, reason in preserve_sections:
            preview = text[start:start+100].replace('\n', ' ')
            print(f"  Preserving: {reason} - '{preview}...'")
        
        chunks = []
        chunk_id = 0
        pos = 0
        
        while pos < len(text):
            # Check if we're in a preserve section
            in_preserve_section = False
            preserve_info = None
            
            for preserve_start, preserve_end, reason in preserve_sections:
                if preserve_start <= pos < preserve_end:
                    in_preserve_section = True
                    preserve_info = (preserve_start, preserve_end, reason)
                    break
            
            if in_preserve_section:
                # Take the entire preserved section as one chunk
                preserve_start, preserve_end, reason = preserve_info
                section_text = text[preserve_start:preserve_end]
                
                chunk = self.create_chunk(
                    chunk_id=chunk_id,
                    text=section_text,
                    char_start=preserve_start,
                    char_end=preserve_end,
                    page_map=page_map,
                    headers=headers,
                    chunk_type="definition" if "definition" in reason.lower() else "preserved"
                )
                
                chunks.append(chunk)
                chunk_id += 1
                pos = preserve_end
                
                print(f"  Created preserved chunk {chunk_id-1}: {chunk.token_count} tokens - {reason}")
                
            else:
                # Regular chunking
                # Find a good breaking point within token limit
                chunk_start = pos
                chunk_end = pos
                chunk_tokens = 0
                
                # Expand chunk until we hit token limit
                while chunk_end < len(text) and chunk_tokens < self.max_tokens:
                    # Look for next good breaking point (paragraph, sentence, etc.)
                    para_break = text.find('\n\n', chunk_end)
                    sent_break = text.find('. ', chunk_end)
                    
                    # Find the next reasonable breaking point
                    breaks = []
                    if para_break != -1:
                        breaks.append(para_break)
                    if sent_break != -1:
                        breaks.append(sent_break + 1)  # Include the period
                    
                    if breaks:
                        next_break = min(breaks)
                    else:
                        # No good break points found, advance by a reasonable amount
                        next_break = min(chunk_end + 200, len(text))
                    
                    # Ensure we're making progress
                    if next_break <= chunk_end:
                        next_break = min(chunk_end + 50, len(text))
                    
                    # Check if adding this segment would exceed token limit
                    test_text = text[chunk_start:next_break]
                    test_tokens = self.count_tokens(test_text)
                    
                    if test_tokens > self.max_tokens and chunk_end > chunk_start:
                        # Would exceed limit, use current position
                        break
                    
                    chunk_end = next_break
                    chunk_tokens = test_tokens
                    
                    # Check if we're about to enter a preserve section
                    should_stop = False
                    for preserve_start, preserve_end, reason in preserve_sections:
                        if chunk_start < preserve_start <= chunk_end:
                            # Stop before the preserve section
                            chunk_end = preserve_start
                            should_stop = True
                            break
                    
                    if should_stop:
                        break
                
                # Ensure minimum chunk size
                if chunk_end <= chunk_start + 50:  # Too small
                    chunk_end = min(chunk_start + 200, len(text))
                
                # Create chunk
                chunk_text = text[chunk_start:chunk_end]
                
                if chunk_text.strip():  # Only create non-empty chunks
                    chunk = self.create_chunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        char_start=chunk_start,
                        char_end=chunk_end,
                        page_map=page_map,
                        headers=headers,
                        chunk_type="content"
                    )
                    
                    chunks.append(chunk)
                    chunk_id += 1
                    
                    # Apply overlap for next chunk
                    overlap_chars = min(self.overlap_tokens * 4, chunk_end - chunk_start - 50)  # Rough char estimate
                    pos = max(chunk_end - overlap_chars, chunk_start + 1)
                else:
                    pos = chunk_end
        
        # Final statistics
        total_tokens = sum(chunk.token_count for chunk in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        
        print(f"Text chunking completed:")
        print(f"  - Total chunks: {len(chunks)}")
        print(f"  - Total tokens: {total_tokens:,}")
        print(f"  - Average tokens per chunk: {avg_tokens:.1f}")
        print(f"  - Preserved sections: {len(preserve_sections)}")
        
        # Check for grace period specifically
        grace_chunks = [c for c in chunks if 'grace period' in c.text.lower()]
        if grace_chunks:
            print(f"  - Found Grace Period in {len(grace_chunks)} chunk(s)")
            for i, chunk in enumerate(grace_chunks):
                if 'thirty days' in chunk.text.lower():
                    print(f"    ✓ Chunk {chunk.chunk_id} contains complete grace period definition")
                else:
                    print(f"    ⚠ Chunk {chunk.chunk_id} mentions grace period but may be incomplete")
        else:
            print("  - ⚠ WARNING: No chunks found containing 'grace period'")
        
        return chunks


# Singleton instance
_text_chunker = None

def get_text_chunker() -> TextChunker:
    """Get singleton text chunker instance"""
    global _text_chunker
    if _text_chunker is None:
        _text_chunker = TextChunker()
    return _text_chunker
"""
Text chunking service with smart section preservation for policy documents
"""
import re
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from app.core.config import settings
from app.utils.debug import debug_print, info_print


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
    chunk_type: str = "content"  # content, definition, section_header, table
    
    # Enhanced semantic metadata
    parent_sections: Optional[List[str]] = None  # ["3", "3.2", "3.22"] - hierarchical path
    section_hierarchy: Optional[str] = None      # "Definitions > Grace Period" - readable path
    semantic_completeness: float = 1.0           # 0.0-1.0 score for how complete this chunk is
    preservation_reason: Optional[str] = None    # Why this chunk was preserved intact
    has_definitions: bool = False                # Whether chunk contains definition patterns
    table_structure: Optional[Dict[str, Any]] = None  # Table metadata if applicable
    
    # Multilingual support
    translated_text: Optional[str] = None        # English translation of this chunk
    source_language: str = "english"            # Source language of the chunk
    has_translation: bool = False               # Whether chunk has translation available


class TextChunker:
    """Smart text chunker for policy documents"""
    
    def __init__(self):
        """Initialize text chunker"""
        # Use BGE-M3 compatible tokenizer - BGE-M3 uses XLM-RoBERTa tokenizer
        # For now, approximate with cl100k_base but adjust token counts by factor
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  
        self.bge_token_adjustment = 1.2  # BGE-M3 typically uses ~20% more tokens than cl100k_base
        self.max_tokens = settings.chunk_size  # Use configured chunk size from settings
        self.overlap_tokens = settings.chunk_overlap  # Use configured overlap from settings
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text (adjusted for BGE-M3 compatibility)"""
        base_tokens = len(self.tokenizer.encode(text))
        # Adjust for BGE-M3's XLM-RoBERTa tokenizer which typically uses more tokens
        return int(base_tokens * self.bge_token_adjustment)
    
    def detect_section_headers(self, text: str) -> List[Tuple[int, str, str]]:
        """
        Enhanced section header detection with comprehensive patterns
        
        Returns:
            List of (position, section_number, heading) tuples
        """
        headers = []
        
        # Enhanced patterns for better coverage
        patterns = [
            # Multi-level numbered sections (3.1, 3.1.1, 3.22.1, etc.)
            (r'^(\d+\.\d+\.?\d*\.?)\s+(.+?)(?=\n|$)', 'numbered_subsection'),
            
            # Main sections (1. PREAMBLE, 2. OPERATIVE CLAUSE, etc.)
            (r'^(\d+\.)\s+([A-Z][A-Z\s]+)(?=\n|$)', 'main_section'),
            
            # Letter-based subsections (a), (b), (i), (ii), etc.)
            (r'^\(([a-z])\)\s+(.+?)(?=\n|$)', 'letter_subsection'),
            (r'^\(([ivx]+)\)\s+(.+?)(?=\n|$)', 'roman_subsection'),
            
            # Schedule and Appendix sections
            (r'^(SCHEDULE\s+[A-Z]+)\s*:?\s*(.+?)(?=\n|$)', 'schedule'),
            (r'^(APPENDIX\s+[A-Z]+)\s*:?\s*(.+?)(?=\n|$)', 'appendix'),
            
            # Definition headers (for better definition detection)
            (r'^(\d+\.\d+\.?\d*\.?)\s+(.*(?:means|definition|defined as).*)(?=\n|$)', 'definition_header'),
            
            # Table of contents style
            (r'^([A-Z][A-Z\s]+)\s*\.{3,}\s*\d+', 'toc_entry'),
        ]
        
        for pattern, pattern_type in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
                pos = match.start()
                section_num = match.group(1).rstrip('.')
                heading = match.group(2).strip() if len(match.groups()) > 1 else ""
                
                # Add pattern type as metadata (we'll use this later)
                headers.append((pos, section_num, heading, pattern_type))
        
        # Sort by position and remove duplicates
        headers = list(set(headers))
        headers.sort(key=lambda x: x[0])
        
        # Convert back to original format but keep the enhanced data
        enhanced_headers = [(pos, section_num, heading) for pos, section_num, heading, _ in headers]
        return enhanced_headers
    
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
        chunk_type: str = "content",
        parent_sections: Optional[List[str]] = None,
        preservation_reason: Optional[str] = None
    ) -> TextChunk:
        """Create a TextChunk with enhanced semantic metadata"""
        
        # Get page number
        page = self.get_page_for_position(char_start, page_map)
        
        # Get section info
        section_num, heading = self.get_section_for_position(char_start, headers)
        
        # Count tokens
        token_count = self.count_tokens(text)
        
        # Build section hierarchy
        if parent_sections is None:
            parent_sections = self._build_section_hierarchy(char_start, headers)
        
        section_hierarchy = self._build_readable_hierarchy(parent_sections, headers)
        
        # Detect definitions in chunk
        has_definitions = self._chunk_has_definitions(text)
        
        # Calculate semantic completeness
        semantic_completeness = self._calculate_semantic_completeness(text, chunk_type)
        
        # Detect table structure if applicable
        table_structure = None
        if chunk_type == "table" or self._is_table_content(text):
            table_structure = self._analyze_table_structure(text)
            chunk_type = "table"
        
        return TextChunk(
            chunk_id=chunk_id,
            text=text.strip(),
            token_count=token_count,
            char_start=char_start,
            char_end=char_end,
            page=page,
            heading=heading,
            section_number=section_num,
            chunk_type=chunk_type,
            parent_sections=parent_sections,
            section_hierarchy=section_hierarchy,
            semantic_completeness=semantic_completeness,
            preservation_reason=preservation_reason,
            has_definitions=has_definitions,
            table_structure=table_structure
        )
    
    def chunk_text(self, text: str, document_metadata: Dict[str, Any]) -> List[TextChunk]:
        """
        Semantic-aware chunking that respects document structure
        
        This method implements recursive semantic chunking:
        1. First preserves complete definitions and important sections
        2. Then chunks remaining text by section boundaries
        3. Uses hierarchical splitting: sections → paragraphs → sentences
        4. Maintains semantic completeness and context
        
        Args:
            text: Full document text
            document_metadata: Document metadata
            
        Returns:
            List of TextChunk objects with enhanced semantic metadata
        """
        debug_print("Starting semantic text chunking...")
        debug_print(f"Document length: {len(text):,} characters")
        debug_print(f"Target: {self.max_tokens} tokens per chunk with {self.overlap_tokens} token overlap")
        
        # Extract structural elements
        page_map = self.extract_page_numbers(text)
        headers = self.detect_section_headers(text)
        preserve_sections = self.preserve_definitions(text, headers)  # FIXED: Actually use this!
        
        debug_print(f"Detected {len(headers)} section headers")
        debug_print(f"Found {len(preserve_sections)} sections to preserve intact")
        
        chunks = []
        chunk_id = 0
        processed_ranges = set()
        
        # Step 1: Handle preserved sections (definitions, important clauses)
        for start, end, reason in preserve_sections:
            preserved_text = text[start:end].strip()
            if preserved_text:
                # Validate preserved section isn't too large
                token_count = self.count_tokens(preserved_text)
                if token_count <= self.max_tokens * 1.5:  # Allow 50% overflow for definitions
                    chunk = self.create_chunk(
                        chunk_id=chunk_id,
                        text=preserved_text,
                        char_start=start,
                        char_end=end,
                        page_map=page_map,
                        headers=headers,
                        chunk_type="definition",
                        preservation_reason=reason
                    )
                    chunks.append(chunk)
                    processed_ranges.update(range(start, end))
                    chunk_id += 1
                    debug_print(f"Preserved definition chunk: {reason[:50]}...")
                else:
                    debug_print(f"Warning: Preserved section too large ({token_count} tokens), will be split")
        
        # Step 2: Process remaining text by sections using recursive chunking
        section_ranges = self._get_section_ranges(headers, len(text))
        
        for i, (section_start, section_end, section_num, section_heading) in enumerate(section_ranges):
            # Skip if this section is largely already preserved
            section_size = section_end - section_start
            
            # Skip empty sections
            if section_size <= 0:
                continue
            
            section_processed = sum(1 for pos in range(section_start, section_end) 
                                  if pos in processed_ranges)
            
            # Prevent division by zero
            if section_size > 0 and section_processed / section_size > 0.8:  # 80% already processed
                continue
            
            # Get unprocessed parts of this section
            section_text = text[section_start:section_end]
            section_chunks = self._chunk_section_recursive(
                section_text, 
                section_start, 
                section_num, 
                section_heading,
                page_map, 
                headers, 
                chunk_id,
                processed_ranges
            )
            
            chunks.extend(section_chunks)
            chunk_id += len(section_chunks)
            
            # Mark this section as processed
            processed_ranges.update(range(section_start, section_end))
        
        # Step 3: Handle any remaining unprocessed text (edge cases)
        remaining_chunks = self._process_remaining_text(
            text, processed_ranges, page_map, headers, chunk_id
        )
        chunks.extend(remaining_chunks)
        
        # Validation and post-processing
        chunks = self._validate_and_fix_chunks(chunks)
        
        # Enhanced statistics
        total_tokens = sum(chunk.token_count for chunk in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        chunk_types = {}
        for chunk in chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
        
        info_print("Semantic chunking completed:")
        debug_print(f"  - Total chunks: {len(chunks)}")
        debug_print(f"  - Average tokens per chunk: {avg_tokens:.1f}")
        print(f"  - Chunk types: {dict(chunk_types)}")
        print(f"  - Preserved definitions: {chunk_types.get('definition', 0)}")
        print(f"  - Table chunks: {chunk_types.get('table', 0)}")
        print(f"  - Content chunks: {chunk_types.get('content', 0)}")
        
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
    
    def _build_section_hierarchy(self, char_pos: int, headers: List[Tuple[int, str, str]]) -> List[str]:
        """Build hierarchical section path for a character position"""
        hierarchy = []
        
        for header_pos, section_num, heading in headers:
            if header_pos <= char_pos:
                # This header comes before our position
                parts = section_num.split('.')
                
                # Build hierarchy - only add if it's a parent of existing or new branch
                if not hierarchy:
                    hierarchy = parts
                else:
                    # Check if this is a parent, sibling, or child
                    common_parts = 0
                    for i, part in enumerate(parts):
                        if i < len(hierarchy) and hierarchy[i] == part:
                            common_parts += 1
                        else:
                            break
                    
                    # Keep only the common part and add new parts
                    hierarchy = hierarchy[:common_parts] + parts[common_parts:]
            else:
                break
        
        return hierarchy
    
    def _build_readable_hierarchy(self, parent_sections: List[str], headers: List[Tuple[int, str, str]]) -> str:
        """Build a readable hierarchy string"""
        if not parent_sections:
            return ""
        
        # Map section numbers to headings
        section_to_heading = {}
        for _, section_num, heading in headers:
            section_to_heading[section_num] = heading
        
        # Build readable path
        path_parts = []
        for i, section in enumerate(parent_sections):
            section_key = '.'.join(parent_sections[:i+1])
            heading = section_to_heading.get(section_key, f"Section {section_key}")
            path_parts.append(heading)
        
        return " > ".join(path_parts)
    
    def _chunk_has_definitions(self, text: str) -> bool:
        """Check if chunk contains definition patterns"""
        definition_patterns = [
            r'\bmeans\b',
            r'\bdefined as\b',
            r'\brefers to\b',
            r'\bshall mean\b',
            r'\bin this policy\b.*\bmeans\b',
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in definition_patterns)
    
    def _calculate_semantic_completeness(self, text: str, chunk_type: str) -> float:
        """Calculate how semantically complete a chunk is (0.0 to 1.0)"""
        score = 1.0
        
        # Check for incomplete sentences
        if not text.strip().endswith(('.', '!', '?', ':', ';')):
            score -= 0.2
        
        # Check for incomplete paragraphs
        if text.startswith(' ') or not text[0].isupper():
            score -= 0.1
        
        # Check for broken lists or numbered items
        if re.search(r'\n\s*\d+\.\s*$', text) or re.search(r'\n\s*[a-z]\)\s*$', text):
            score -= 0.3
        
        # Bonus for definitions (they should be complete)
        if chunk_type == "definition":
            if "means" in text.lower() and text.strip().endswith('.'):
                score = 1.0
        
        return max(0.0, score)
    
    def _analyze_table_structure(self, text: str) -> Dict[str, Any]:
        """Analyze table structure and return metadata"""
        lines = text.split('\n')
        structure = {
            'rows': len(lines),
            'has_headers': False,
            'alignment_pattern': None,
            'column_indicators': [],
            'table_type': 'unknown'
        }
        
        # Detect headers (usually first line with specific patterns)
        if lines and any(indicator in lines[0].lower() for indicator in 
                        ['plan', 'coverage', 'benefit', 'sum insured', 'premium']):
            structure['has_headers'] = True
        
        # Detect alignment patterns
        space_patterns = [line.count('  ') for line in lines[:5]]  # Check first 5 lines
        if space_patterns and max(space_patterns) > 2:
            structure['alignment_pattern'] = 'spaces'
        elif any('\t' in line for line in lines[:3]):
            structure['alignment_pattern'] = 'tabs'
        
        # Detect table type
        text_lower = text.lower()
        if 'premium' in text_lower and '%' in text:
            structure['table_type'] = 'premium_rates'
        elif 'sum insured' in text_lower:
            structure['table_type'] = 'coverage_limits'
        elif 'plan' in text_lower and any(c in text for c in ['a', 'b', 'c']):
            structure['table_type'] = 'plan_comparison'
        
        return structure
    
    def _get_section_ranges(self, headers: List[Tuple[int, str, str]], text_length: int) -> List[Tuple[int, int, str, str]]:
        """Get section ranges with start, end, section_num, heading"""
        ranges = []
        
        for i, (pos, section_num, heading) in enumerate(headers):
            start = pos
            end = headers[i + 1][0] if i + 1 < len(headers) else text_length
            ranges.append((start, end, section_num, heading))
        
        return ranges
    
    def _chunk_section_recursive(
        self, 
        section_text: str, 
        base_offset: int, 
        section_num: str, 
        section_heading: str,
        page_map: Dict[int, int], 
        headers: List[Tuple[int, str, str]], 
        start_chunk_id: int,
        processed_ranges: set
    ) -> List[TextChunk]:
        """
        Recursively chunk a section using hierarchical splitting:
        1. Try to keep section intact if small enough
        2. Split by paragraphs (double newlines)
        3. Split by sentences if paragraphs too large
        4. Use token-based splitting as last resort
        """
        chunks = []
        chunk_id = start_chunk_id
        
        # Check if entire section fits in one chunk
        section_tokens = self.count_tokens(section_text)
        if section_tokens <= self.max_tokens:
            chunk = self.create_chunk(
                chunk_id=chunk_id,
                text=section_text,
                char_start=base_offset,
                char_end=base_offset + len(section_text),
                page_map=page_map,
                headers=headers,
                chunk_type="content",
                parent_sections=[section_num]
            )
            return [chunk]
        
        # Split by paragraphs first
        paragraphs = self._split_by_paragraphs(section_text)
        current_chunk_text = ""
        current_chunk_start = base_offset
        
        for para_start, para_end, para_text in paragraphs:
            # Skip if this paragraph is already processed
            abs_para_start = base_offset + para_start
            abs_para_end = base_offset + para_end
            
            if any(pos in processed_ranges for pos in range(abs_para_start, abs_para_end)):
                continue
            
            # Check if adding this paragraph would exceed token limit
            test_text = (current_chunk_text + "\n\n" + para_text).strip()
            test_tokens = self.count_tokens(test_text)
            
            if test_tokens <= self.max_tokens and current_chunk_text:
                # Safe to add paragraph
                current_chunk_text = test_text
            else:
                # Finalize current chunk if it exists
                if current_chunk_text:
                    chunk = self.create_chunk(
                        chunk_id=chunk_id,
                        text=current_chunk_text,
                        char_start=current_chunk_start,
                        char_end=current_chunk_start + len(current_chunk_text),
                        page_map=page_map,
                        headers=headers,
                        chunk_type="content",
                        parent_sections=[section_num]
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                # Start new chunk with current paragraph
                current_chunk_text = para_text
                current_chunk_start = abs_para_start
                
                # If single paragraph is too large, split it further
                if self.count_tokens(para_text) > self.max_tokens:
                    sentence_chunks = self._split_paragraph_by_sentences(
                        para_text, abs_para_start, section_num, page_map, headers, chunk_id
                    )
                    chunks.extend(sentence_chunks)
                    chunk_id += len(sentence_chunks)
                    current_chunk_text = ""
        
        # Don't forget the last chunk
        if current_chunk_text:
            chunk = self.create_chunk(
                chunk_id=chunk_id,
                text=current_chunk_text,
                char_start=current_chunk_start,
                char_end=current_chunk_start + len(current_chunk_text),
                page_map=page_map,
                headers=headers,
                chunk_type="content",
                parent_sections=[section_num]
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[Tuple[int, int, str]]:
        """Split text by paragraphs and return (start, end, text) tuples"""
        paragraphs = []
        
        # Split by double newlines (paragraph boundaries)
        para_splits = text.split('\n\n')
        current_pos = 0
        
        for i, para in enumerate(para_splits):
            para = para.strip()
            if para:
                # Find the actual position of this paragraph in the original text
                start = text.find(para, current_pos)
                if start == -1:  # Fallback if not found
                    start = current_pos
                end = start + len(para)
                paragraphs.append((start, end, para))
                current_pos = end
            
            # Move past the delimiter for next search
            if i < len(para_splits) - 1:  # Not the last paragraph
                current_pos = text.find('\n\n', current_pos) + 2
                if current_pos == 1:  # find returned -1
                    current_pos = len(text)
        
        return paragraphs
    
    def _split_paragraph_by_sentences(
        self, 
        para_text: str, 
        para_start: int, 
        section_num: str,
        page_map: Dict[int, int], 
        headers: List[Tuple[int, str, str]], 
        start_chunk_id: int
    ) -> List[TextChunk]:
        """Split an oversized paragraph by sentences"""
        chunks = []
        chunk_id = start_chunk_id
        
        # Split by sentences (more sophisticated than just periods)
        sentences = self._split_by_sentences(para_text)
        current_chunk = ""
        current_start = para_start
        
        for sentence in sentences:
            test_chunk = (current_chunk + " " + sentence).strip()
            
            if self.count_tokens(test_chunk) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                # Finalize current chunk
                if current_chunk:
                    chunk = self.create_chunk(
                        chunk_id=chunk_id,
                        text=current_chunk,
                        char_start=current_start,
                        char_end=current_start + len(current_chunk),
                        page_map=page_map,
                        headers=headers,
                        chunk_type="content",
                        parent_sections=[section_num]
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                # Start new chunk
                current_chunk = sentence
                current_start = para_start + para_text.find(sentence)
        
        # Final chunk
        if current_chunk:
            chunk = self.create_chunk(
                chunk_id=chunk_id,
                text=current_chunk,
                char_start=current_start,
                char_end=current_start + len(current_chunk),
                page_map=page_map,
                headers=headers,
                chunk_type="content",
                parent_sections=[section_num]
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences with better handling of abbreviations"""
        # Simple sentence splitting - could be enhanced with more sophisticated NLP
        sentences = []
        current_sentence = ""
        
        i = 0
        while i < len(text):
            char = text[i]
            current_sentence += char
            
            if char == '.':
                # Check if this is end of sentence or abbreviation
                next_char = text[i + 1] if i + 1 < len(text) else ''
                
                # Simple heuristic: if next char is space and then capital letter, it's sentence end
                if next_char == ' ' and i + 2 < len(text) and text[i + 2].isupper():
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
                elif next_char == '\n':
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
            
            i += 1
        
        # Add remaining text
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return [s for s in sentences if s]
    
    def _process_remaining_text(
        self, 
        text: str, 
        processed_ranges: set, 
        page_map: Dict[int, int], 
        headers: List[Tuple[int, str, str]], 
        start_chunk_id: int
    ) -> List[TextChunk]:
        """Process any text that wasn't handled by section processing"""
        chunks = []
        chunk_id = start_chunk_id
        
        # Find unprocessed ranges
        unprocessed_ranges = []
        current_start = None
        
        for i in range(len(text)):
            if i not in processed_ranges:
                if current_start is None:
                    current_start = i
            else:
                if current_start is not None:
                    unprocessed_ranges.append((current_start, i))
                    current_start = None
        
        # Handle final range
        if current_start is not None:
            unprocessed_ranges.append((current_start, len(text)))
        
        # Process each unprocessed range
        for start, end in unprocessed_ranges:
            if end - start > 10:  # Only process ranges larger than 10 characters
                remaining_text = text[start:end].strip()
                if remaining_text:
                    # Use simple token-based chunking for remaining text
                    remaining_chunks = self._simple_token_chunk(
                        remaining_text, start, page_map, headers, chunk_id
                    )
                    chunks.extend(remaining_chunks)
                    chunk_id += len(remaining_chunks)
        
        return chunks
    
    def _simple_token_chunk(
        self, 
        text: str, 
        base_offset: int, 
        page_map: Dict[int, int], 
        headers: List[Tuple[int, str, str]], 
        start_chunk_id: int
    ) -> List[TextChunk]:
        """Simple token-based chunking for edge cases"""
        chunks = []
        chunk_id = start_chunk_id
        pos = 0
        
        while pos < len(text):
            # Find chunk end based on actual token count
            chunk_end = self._find_token_based_end(text, pos, self.max_tokens)
            chunk_text = text[pos:chunk_end].strip()
            
            if chunk_text:
                chunk = self.create_chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    char_start=base_offset + pos,
                    char_end=base_offset + chunk_end,
                    page_map=page_map,
                    headers=headers,
                    chunk_type="content"
                )
                chunks.append(chunk)
                chunk_id += 1
                
                # Move with overlap
                overlap_size = min(len(chunk_text) // 4, self.overlap_tokens * 4)
                pos = max(chunk_end - overlap_size, pos + 1)
            else:
                pos = chunk_end
        
        return chunks
    
    def _find_token_based_end(self, text: str, start: int, max_tokens: int) -> int:
        """Find chunk end based on actual token count, not character estimation"""
        current_end = start
        binary_search_range = (start + 100, min(len(text), start + max_tokens * 6))  # Conservative range
        
        # Binary search for optimal end position
        low, high = binary_search_range
        best_end = start + 100  # Minimum chunk size
        
        while low <= high:
            mid = (low + high) // 2
            test_text = text[start:mid]
            token_count = self.count_tokens(test_text)
            
            if token_count <= max_tokens:
                best_end = mid
                low = mid + 1
            else:
                high = mid - 1
        
        # Find a good break point near the best position
        final_end = self._find_smart_break_point(text, start, best_end)
        return final_end
    
    def _validate_and_fix_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Validate chunks and fix any issues"""
        validated_chunks = []
        
        for chunk in chunks:
            # Check token count
            actual_tokens = self.count_tokens(chunk.text)
            if actual_tokens != chunk.token_count:
                # Update token count
                chunk.token_count = actual_tokens
            
            # Split oversized chunks
            if chunk.token_count > self.max_tokens * 1.5:  # 50% tolerance
                print(f"Warning: Chunk {chunk.chunk_id} oversized ({chunk.token_count} tokens), splitting...")
                split_chunks = self._split_oversized_chunk(chunk)
                validated_chunks.extend(split_chunks)
            else:
                validated_chunks.append(chunk)
        
        # Re-assign chunk IDs
        for i, chunk in enumerate(validated_chunks):
            chunk.chunk_id = i
        
        return validated_chunks
    
    def _split_oversized_chunk(self, chunk: TextChunk) -> List[TextChunk]:
        """Split an oversized chunk into smaller pieces"""
        # Simple splitting for oversized chunks
        text = chunk.text
        target_size = self.max_tokens
        
        # Split by sentences first
        sentences = self._split_by_sentences(text)
        sub_chunks = []
        current_text = ""
        
        for sentence in sentences:
            test_text = (current_text + " " + sentence).strip()
            if self.count_tokens(test_text) <= target_size:
                current_text = test_text
            else:
                if current_text:
                    sub_chunks.append(current_text)
                current_text = sentence
        
        if current_text:
            sub_chunks.append(current_text)
        
        # Create new chunk objects
        result_chunks = []
        char_offset = chunk.char_start
        
        for i, sub_text in enumerate(sub_chunks):
            new_chunk = TextChunk(
                chunk_id=chunk.chunk_id + i,
                text=sub_text,
                token_count=self.count_tokens(sub_text),
                char_start=char_offset,
                char_end=char_offset + len(sub_text),
                page=chunk.page,
                heading=chunk.heading,
                section_number=chunk.section_number,
                chunk_type=chunk.chunk_type,
                parent_sections=chunk.parent_sections,
                section_hierarchy=chunk.section_hierarchy,
                semantic_completeness=max(0.5, chunk.semantic_completeness - 0.3),  # Reduce completeness
                preservation_reason=chunk.preservation_reason,
                has_definitions=chunk.has_definitions,
                table_structure=chunk.table_structure
            )
            result_chunks.append(new_chunk)
            char_offset += len(sub_text)
        
        return result_chunks


# Singleton instance
_text_chunker = None

def get_text_chunker() -> TextChunker:
    """Get singleton text chunker instance"""
    global _text_chunker
    if _text_chunker is None:
        _text_chunker = TextChunker()
    return _text_chunker
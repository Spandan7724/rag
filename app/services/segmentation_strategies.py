"""
Segmentation strategies for different document types and formats
"""
import re
import spacy
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.models.requests import DocumentContent
from app.models.clause import (
    Clause, ClauseType, SourceType, PageContent, Section,
    ClauseExtractionResult
)
from app.services.text_patterns import TextPatterns

# Module-level spaCy singleton for performance
_nlp_instance: Optional[spacy.Language] = None

def get_nlp_instance() -> Optional[spacy.Language]:
    """Get thread-safe spaCy instance, loading only once per process"""
    global _nlp_instance
    if _nlp_instance is None:
        try:
            _nlp_instance = spacy.load("en_core_web_sm")
            # Keep parser for sentence segmentation, disable others for performance
            if "tagger" in _nlp_instance.pipe_names:
                _nlp_instance.disable_pipes(["tagger", "ner"])
            # Ensure sentence segmentation is working
            if "parser" not in _nlp_instance.pipe_names:
                _nlp_instance.add_pipe("sentencizer")
            print("[INFO] Module-level spaCy singleton loaded successfully")
        except OSError:
            print("[WARNING] spaCy model not found, sentence processing will be disabled")
            _nlp_instance = None
    return _nlp_instance
from app.services.clause_classifier import ClauseClassifier

class SegmentationStrategy(ABC):
    """Abstract base class for document segmentation strategies"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.patterns = TextPatterns()
        self.classifier = ClauseClassifier()
    
    @abstractmethod
    def segment(self, document_content: DocumentContent) -> ClauseExtractionResult:
        """Segment document into clauses"""
        pass
    
    def _is_valid_result(self, result: ClauseExtractionResult) -> bool:
        """Check if segmentation result is valid with adaptive thresholds"""
        clause_count = len(result.clauses)
        
        # Dynamic thresholds based on document characteristics
        min_clauses = max(5, result.total_chars // 3000)  # Roughly 1 clause per 3000 chars
        max_clauses = min(1000, result.total_chars // 100)  # Not more than 1 per 100 chars
        
        # For definition-heavy documents, expect more clauses
        if result.sections_found > 0:
            definition_ratio = result.sections_found / max(1, result.pages_processed)
            if definition_ratio > 2:  # Many sections per page = definition document
                min_clauses = max(min_clauses, 15)
        
        return (
            clause_count >= min_clauses and
            clause_count <= max_clauses and
            len(result.errors) == 0  # No critical errors
        )

class PatternBasedStrategy(SegmentationStrategy):
    """Enhanced regex-based segmentation with multiple patterns"""
    
    def __init__(self):
        super().__init__()
        # Use module-level singleton for performance
        self.nlp = get_nlp_instance()
        if self.nlp:
            print("[SUCCESS] PatternBasedStrategy using spaCy singleton for sentence fallback")
        else:
            print("[WARNING] PatternBasedStrategy: spaCy not available, sentence fallback disabled")
    
    def segment(self, document_content: DocumentContent) -> ClauseExtractionResult:
        """Segment using enhanced pattern matching"""
        start_time = datetime.now()
        errors = []
        warnings = []
        
        try:
            print(f"[INFO] {self.name}: Starting pattern-based segmentation...")
            
            # Extract pages
            pages = self._extract_pages(document_content.text)
            
            # Process pages for sections and clauses
            all_clauses = []
            sections_found = 0
            
            for page in pages:
                sections = self._extract_sections(page)
                sections_found += len(sections)
                
                for section in sections:
                    clauses = self._extract_clauses_from_section(section, page.page_number)
                    all_clauses.extend(clauses)
                
                # Process exclusion list items separately
                exclusion_clauses = self._extract_exclusion_clauses(page)
                all_clauses.extend(exclusion_clauses)
                if exclusion_clauses:
                    print(f"   [PAGE] Page {page.page_number}: Found {len(exclusion_clauses)} exclusion list items")
            
            # Generate IDs
            for i, clause in enumerate(all_clauses):
                clause.id = f"clause_{i+1:03d}"
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate metrics
            metrics = self._calculate_metrics(all_clauses, document_content)
            print(f"[SUCCESS] {self.name}: Found {len(all_clauses)} clauses, {sections_found} sections")
            print(f"[METRICS] {metrics}")
            
            return ClauseExtractionResult(
                clauses=all_clauses,
                pages_processed=len(pages),
                sections_found=sections_found,
                total_chars=len(document_content.text),
                processing_time_seconds=round(processing_time, 2),
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            errors.append(f"Pattern-based segmentation failed: {str(e)}")
            
            return ClauseExtractionResult(
                clauses=[],
                pages_processed=0,
                sections_found=0,
                total_chars=len(document_content.text),
                processing_time_seconds=round(processing_time, 2),
                errors=errors,
                warnings=warnings
            )
    
    def _extract_pages(self, text: str) -> List[PageContent]:
        """Extract pages preserving line structure"""
        pages = []
        page_splits = self.patterns.split_by_pages(text)
        
        for page_num, page_text, start_pos, end_pos in page_splits:
            pages.append(PageContent(
                page_number=page_num,
                raw_text=page_text,
                cleaned_text=page_text,  # Keep raw for pattern matching
                char_start=start_pos,
                char_end=end_pos
            ))
        
        return pages
    
    def _extract_sections(self, page: PageContent) -> List[Section]:
        """Extract sections using enhanced pattern matching"""
        sections = []
        section_headers = self.patterns.extract_section_headers(page.cleaned_text)
        
        print(f"   [PAGE] Page {page.page_number}: Found {len(section_headers)} section headers")
        for header in section_headers:
            print(f"      - {header.metadata.get('section_number', '')} {header.metadata['section_title']} ({header.metadata.get('pattern_used', 'unknown')})")
        
        for i, header in enumerate(section_headers):
            start_pos = header.end
            
            if i + 1 < len(section_headers):
                end_pos = section_headers[i + 1].start
            else:
                end_pos = len(page.cleaned_text)
            
            section_text = page.cleaned_text[start_pos:end_pos].strip()
            
            if section_text:
                sections.append(Section(
                    title=header.metadata["section_title"],
                    section_number=header.metadata.get("section_number", ""),
                    text=section_text,
                    page_number=page.page_number,
                    char_start=page.char_start + start_pos,
                    char_end=page.char_start + end_pos
                ))
        
        return sections
    
    def _extract_clauses_from_section(self, section: Section, page_number: int) -> List[Clause]:
        """Extract clauses from section using sub-clause patterns"""
        clauses = []
        sub_clauses = self.patterns.extract_sub_clauses(section.text)
        
        print(f"      [SECTION] Section '{section.title}': Found {len(sub_clauses)} sub-clauses")
        if sub_clauses:
            # Show which patterns were used
            pattern_counts = {}
            for sub in sub_clauses:
                pattern_used = sub.metadata.get('pattern_used', 'unknown')
                pattern_counts[pattern_used] = pattern_counts.get(pattern_used, 0) + 1
            pattern_summary = ', '.join([f"{pattern}: {count}" for pattern, count in pattern_counts.items()])
            print(f"         Patterns used: {pattern_summary}")
        
        if not sub_clauses:
            # Try sentence-level fallback first
            sentence_clauses = self._split_by_sentences(section, page_number)
            if sentence_clauses:
                print(f"         Using sentence fallback: {len(sentence_clauses)} sentences")
                return sentence_clauses
            
            # Try paragraph-level fallback if sentence splitting fails
            paragraph_clauses = self._split_by_paragraphs(section, page_number)
            if paragraph_clauses:
                print(f"         Using paragraph fallback: {len(paragraph_clauses)} paragraphs")
                return paragraph_clauses
            
            # Last resort: treat entire section as one clause
            clause = self._create_clause(
                text=section.text,
                page_number=page_number,
                section_heading=section.title,
                sub_section=section.section_number,
                char_start=section.char_start,
                char_end=section.char_end
            )
            clauses.append(clause)
            return clauses
        
        # Process sub-clauses
        for i, sub_clause in enumerate(sub_clauses):
            start_pos = sub_clause.start
            end_pos = sub_clauses[i + 1].start if i + 1 < len(sub_clauses) else len(section.text)
            
            clause_text = section.text[start_pos:end_pos].strip()
            clause_text = clause_text[len(sub_clause.text):].strip()  # Remove marker
            
            if clause_text and len(clause_text) > 20:
                clause = self._create_clause(
                    text=clause_text,
                    page_number=page_number,
                    section_heading=section.title,
                    sub_section=sub_clause.metadata["clause_number"],
                    char_start=section.char_start + start_pos,
                    char_end=section.char_start + end_pos
                )
                clauses.append(clause)
        
        return clauses
    
    def _extract_exclusion_clauses(self, page: PageContent) -> List[Clause]:
        """Extract individual exclusion list items as clauses"""
        clauses = []
        
        # Get exclusion list items
        exclusion_items = self.patterns.extract_exclusion_list_items(page.cleaned_text)
        
        for item in exclusion_items:
            # Create clause for each exclusion item
            clause = self._create_clause(
                text=item.metadata['item_name'],
                page_number=page.page_number,
                section_heading="EXCLUSIONS",
                sub_section=item.metadata.get('item_number', ''),
                char_start=page.char_start + item.start,
                char_end=page.char_start + item.end
            )
            # Override clause type to exclusion
            clause.clause_type = ClauseType.EXCLUSION
            clauses.append(clause)
        
        return clauses
    
    def _split_by_sentences(self, section: Section, page_number: int) -> List[Clause]:
        """Split long narrative sections by sentences when no sub-clauses found"""
        clauses = []
        
        # Only apply sentence splitting to sections with substantial text
        if len(section.text) < 200:  # Skip short sections
            return []
        
        try:
            # Check if spaCy is available
            if self.nlp is None:
                print(f"spaCy not available, skipping sentence splitting")
                return []
            
            # Use spaCy for sentence segmentation
            doc = self.nlp(section.text)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
            
            # Only split if we get multiple meaningful sentences
            if len(sentences) < 2:
                return []
            
            print(f"         Splitting section into {len(sentences)} sentences")
            
            char_offset = 0
            for i, sentence_text in enumerate(sentences):
                # Find sentence position in original text
                sentence_start = section.text.find(sentence_text, char_offset)
                if sentence_start == -1:
                    sentence_start = char_offset
                
                sentence_end = sentence_start + len(sentence_text)
                char_offset = sentence_end
                
                clause = self._create_clause(
                    text=sentence_text,
                    page_number=page_number,
                    section_heading=section.title,
                    sub_section=f"{section.section_number}.s{i+1}",  # .s1, .s2 for sentences
                    char_start=section.char_start + sentence_start,
                    char_end=section.char_start + sentence_end
                )
                clauses.append(clause)
            
            return clauses
            
        except Exception as e:
            print(f"         Sentence splitting failed: {str(e)}")
            return []
    
    def _split_by_paragraphs(self, section: Section, page_number: int) -> List[Clause]:
        """Split sections by paragraph boundaries when sentence splitting fails"""
        clauses = []
        
        # Only apply paragraph splitting to substantial sections
        if len(section.text) < 100:  # Skip very short sections
            return []
        
        try:
            # Split on double newlines (paragraph boundaries)
            paragraphs = re.split(r'\n\s*\n', section.text)
            
            # Filter paragraphs with minimum length and content
            meaningful_paragraphs = []
            for para in paragraphs:
                para_clean = para.strip()
                if len(para_clean) >= 30 and not para_clean.isupper():  # Skip headers
                    meaningful_paragraphs.append(para_clean)
            
            # Only split if we get multiple meaningful paragraphs
            if len(meaningful_paragraphs) < 2:
                return []
            
            print(f"         Splitting section into {len(meaningful_paragraphs)} paragraphs")
            
            char_offset = 0
            for i, paragraph_text in enumerate(meaningful_paragraphs):
                # Find paragraph position in original text
                para_start = section.text.find(paragraph_text, char_offset)
                if para_start == -1:
                    para_start = char_offset
                
                para_end = para_start + len(paragraph_text)
                char_offset = para_end
                
                clause = self._create_clause(
                    text=paragraph_text,
                    page_number=page_number,
                    section_heading=section.title,
                    sub_section=f"{section.section_number}.p{i+1}",  # .p1, .p2 for paragraphs
                    char_start=section.char_start + para_start,
                    char_end=section.char_start + para_end
                )
                clauses.append(clause)
            
            return clauses
            
        except Exception as e:
            print(f"         Paragraph splitting failed: {str(e)}")
            return []
    
    def _calculate_metrics(self, clauses: List[Clause], document_content: DocumentContent) -> str:
        """Calculate comprehensive metrics for segmentation quality"""
        if not clauses:
            return "No clauses to analyze"
        
        # Basic counts
        total_clauses = len(clauses)
        
        # Token and character analysis
        total_tokens = 0
        total_chars = 0
        clause_lengths = []
        
        for clause in clauses:
            tokens = len(clause.text.split())
            chars = len(clause.text)
            total_tokens += tokens
            total_chars += chars
            clause_lengths.append(tokens)
        
        avg_tokens_per_clause = total_tokens / total_clauses if total_clauses > 0 else 0
        avg_chars_per_clause = total_chars / total_clauses if total_clauses > 0 else 0
        
        # Clause type distribution
        type_distribution = {}
        for clause in clauses:
            clause_type = clause.clause_type.value if hasattr(clause.clause_type, 'value') else str(clause.clause_type)
            type_distribution[clause_type] = type_distribution.get(clause_type, 0) + 1
        
        # Sort by count
        sorted_types = sorted(type_distribution.items(), key=lambda x: x[1], reverse=True)
        type_summary = ', '.join([f"{t}: {c}" for t, c in sorted_types])
        
        # Granularity analysis
        chars_per_clause_ratio = len(document_content.text) / total_clauses if total_clauses > 0 else 0
        
        # Format metrics string
        return (f"Avg tokens/clause: {avg_tokens_per_clause:.1f}, "
                f"Avg chars/clause: {avg_chars_per_clause:.0f}, "
                f"Granularity: {chars_per_clause_ratio:.0f} chars/clause, "
                f"Types: [{type_summary}]")
    
    def _create_clause(self, text: str, page_number: int, section_heading: str, 
                      sub_section: str, char_start: int, char_end: int) -> Clause:
        """Create clause with metadata"""
        cleaned_text = self.patterns.clean_text(text)
        clause_type, confidence = self.classifier.classify_with_confidence(cleaned_text, section_heading)
        
        metadata = {
            "confidence_score": confidence,
            "char_length": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
            "strategy_used": self.name
        }
        
        return Clause(
            id="",  # Will be set later
            text=cleaned_text,
            page_number=page_number,
            section_heading=section_heading,
            sub_section=sub_section,
            clause_type=clause_type,
            source_type=SourceType.NATIVE,
            char_start=char_start,
            char_end=char_end,
            metadata=metadata,
            created_at=datetime.now()
        )

class SpacyNLPStrategy(SegmentationStrategy):
    """NLP-based segmentation using spaCy"""
    
    def __init__(self):
        super().__init__()
        # Use module-level singleton for performance
        self.nlp = get_nlp_instance()
        if self.nlp is None:
            raise Exception("spaCy English model not found. Please install with: python -m spacy download en_core_web_sm")
        print("[SUCCESS] SpacyNLPStrategy using spaCy singleton")
    
    def segment(self, document_content: DocumentContent) -> ClauseExtractionResult:
        """Segment using spaCy NLP pipeline"""
        start_time = datetime.now()
        errors = []
        warnings = []
        
        try:
            print(f"[INFO] {self.name}: Starting NLP-based segmentation...")
            
            # Process with spaCy
            doc = self.nlp(document_content.text)
            
            # Extract meaningful chunks using sentence boundaries and content analysis
            chunks = self._extract_semantic_chunks(doc, document_content.text)
            
            # Create clauses from chunks
            clauses = []
            for i, chunk_data in enumerate(chunks):
                clause = self._create_nlp_clause(chunk_data, i + 1)
                clauses.append(clause)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            print(f"[SUCCESS] {self.name}: Found {len(clauses)} semantic chunks")
            
            return ClauseExtractionResult(
                clauses=clauses,
                pages_processed=self._estimate_pages(document_content.text),
                sections_found=len(set(clause.section_heading for clause in clauses)),
                total_chars=len(document_content.text),
                processing_time_seconds=round(processing_time, 2),
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            errors.append(f"NLP segmentation failed: {str(e)}")
            
            return ClauseExtractionResult(
                clauses=[],
                pages_processed=0,
                sections_found=0,
                total_chars=len(document_content.text),
                processing_time_seconds=round(processing_time, 2),
                errors=errors,
                warnings=warnings
            )
    
    def _extract_semantic_chunks(self, doc, original_text: str) -> List[Dict[str, Any]]:
        """Extract semantic chunks using NLP analysis"""
        chunks = []
        current_chunk = []
        current_length = 0
        target_length = 400  # Target chunk size
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_length = len(sent_text)
            
            # Skip very short sentences (likely formatting artifacts)
            if sent_length < 20:
                continue
            
            # Start new chunk if current is getting too long
            if current_length + sent_length > target_length * 1.5 and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk_data(chunk_text, original_text, len(chunks) + 1))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sent_text)
            current_length += sent_length
            
            # Complete chunk if we hit target length or find a natural break
            if (current_length >= target_length and self._is_natural_break(sent_text)) or current_length > target_length * 2:
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk_data(chunk_text, original_text, len(chunks) + 1))
                current_chunk = []
                current_length = 0
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_chunk_data(chunk_text, original_text, len(chunks) + 1))
        
        return chunks
    
    def _is_natural_break(self, text: str) -> bool:
        """Detect natural breaking points in text"""
        # End of definitions, sections, etc.
        indicators = [
            text.endswith('.') and ('means' in text or 'refers to' in text),
            'provided that' in text.lower(),
            'subject to' in text.lower(),
            len(text.split()) < 10 and text.isupper()  # Section headers
        ]
        return any(indicators)
    
    def _create_chunk_data(self, text: str, original_text: str, chunk_num: int) -> Dict[str, Any]:
        """Create chunk data with metadata"""
        # Estimate page number
        char_pos = original_text.find(text)
        page_num = self._estimate_page_from_position(char_pos, original_text)
        
        # Infer section heading from content
        section_heading = self._infer_section_heading(text, chunk_num)
        
        return {
            'text': text,
            'page_number': page_num,
            'section_heading': section_heading,
            'char_position': char_pos,
            'chunk_number': chunk_num
        }
    
    def _create_nlp_clause(self, chunk_data: Dict[str, Any], clause_id: int) -> Clause:
        """Create clause from NLP chunk data"""
        text = chunk_data['text']
        clause_type, confidence = self.classifier.classify_with_confidence(text, chunk_data['section_heading'])
        
        metadata = {
            "confidence_score": confidence,
            "char_length": len(text),
            "word_count": len(text.split()),
            "strategy_used": self.name,
            "chunk_number": chunk_data['chunk_number']
        }
        
        return Clause(
            id=f"clause_{clause_id:03d}",
            text=text,
            page_number=chunk_data['page_number'],
            section_heading=chunk_data['section_heading'],
            sub_section=f"nlp_{chunk_data['chunk_number']}",
            clause_type=clause_type,
            source_type=SourceType.NATIVE,
            char_start=chunk_data.get('char_position', 0),
            char_end=chunk_data.get('char_position', 0) + len(text),
            metadata=metadata,
            created_at=datetime.now()
        )
    
    def _estimate_pages(self, text: str) -> int:
        """Estimate number of pages from text"""
        page_markers = len(re.findall(r'---\s*Page\s+\d+\s*---', text))
        return max(page_markers, 1)
    
    def _estimate_page_from_position(self, char_pos: int, text: str) -> int:
        """Estimate page number from character position"""
        if char_pos == -1:
            return 1
        
        # Find page markers before this position
        page_markers = list(re.finditer(r'---\s*Page\s+(\d+)\s*---', text))
        current_page = 1
        
        for match in page_markers:
            if match.start() <= char_pos:
                current_page = int(match.group(1))
            else:
                break
        
        return current_page
    
    def _infer_section_heading(self, text: str, position: int) -> str:
        """Infer section heading based on content and position"""
        text_lower = text.lower()
        
        # Early chunks are likely definitions
        if position <= 20 and ('means' in text_lower or 'refers to' in text_lower):
            return "DEFINITIONS"
        elif 'cover' in text_lower or 'benefit' in text_lower:
            return "COVERAGE"
        elif 'exclud' in text_lower or 'not covered' in text_lower:
            return "EXCLUSIONS"
        elif 'claim' in text_lower or 'process' in text_lower:
            return "CLAIMS"
        else:
            return "GENERAL"

class FallbackChunkingStrategy(SegmentationStrategy):
    """Simple fallback strategy that always produces results"""
    
    def segment(self, document_content: DocumentContent) -> ClauseExtractionResult:
        """Simple paragraph-based segmentation as fallback"""
        start_time = datetime.now()
        
        try:
            print(f"[INFO] {self.name}: Using fallback paragraph-based segmentation...")
            
            # Simple paragraph splitting
            paragraphs = [p.strip() for p in document_content.text.split('\n\n') if p.strip()]
            
            # Filter and process paragraphs
            clauses = []
            for i, paragraph in enumerate(paragraphs):
                if 50 <= len(paragraph) <= 2000:  # Reasonable size
                    clause = self._create_fallback_clause(paragraph, i + 1)
                    clauses.append(clause)
                elif len(paragraph) > 2000:
                    # Split long paragraphs
                    sub_chunks = self._split_long_paragraph(paragraph)
                    for j, chunk in enumerate(sub_chunks):
                        clause = self._create_fallback_clause(chunk, len(clauses) + 1)
                        clauses.append(clause)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            print(f"[SUCCESS] {self.name}: Created {len(clauses)} fallback chunks")
            
            return ClauseExtractionResult(
                clauses=clauses,
                pages_processed=self._estimate_pages(document_content.text),
                sections_found=len(set(clause.section_heading for clause in clauses)),
                total_chars=len(document_content.text),
                processing_time_seconds=round(processing_time, 2),
                errors=[],
                warnings=["Used fallback segmentation - results may be less accurate"]
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ClauseExtractionResult(
                clauses=[],
                pages_processed=0,
                sections_found=0,
                total_chars=len(document_content.text),
                processing_time_seconds=round(processing_time, 2),
                errors=[f"Fallback segmentation failed: {str(e)}"],
                warnings=[]
            )
    
    def _split_long_paragraph(self, text: str) -> List[str]:
        """Split long paragraphs at sentence boundaries"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if current_length + len(sentence) > 800 and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += len(sentence)
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks
    
    def _create_fallback_clause(self, text: str, clause_id: int) -> Clause:
        """Create clause with minimal processing"""
        clause_type, confidence = self.classifier.classify_with_confidence(text, "")
        
        metadata = {
            "confidence_score": confidence,
            "char_length": len(text),
            "word_count": len(text.split()),
            "strategy_used": self.name
        }
        
        return Clause(
            id=f"clause_{clause_id:03d}",
            text=text,
            page_number=1,  # Default page
            section_heading=self._infer_section(text),
            sub_section=f"para_{clause_id}",
            clause_type=clause_type,
            source_type=SourceType.NATIVE,
            char_start=0,
            char_end=len(text),
            metadata=metadata,
            created_at=datetime.now()
        )
    
    def _infer_section(self, text: str) -> str:
        """Simple section inference"""
        text_lower = text.lower()
        if 'means' in text_lower:
            return "DEFINITIONS"
        elif 'cover' in text_lower:
            return "COVERAGE"
        elif 'exclud' in text_lower:
            return "EXCLUSIONS"
        else:
            return "GENERAL"
    
    def _estimate_pages(self, text: str) -> int:
        """Estimate pages"""
        return max(len(re.findall(r'---\s*Page\s+\d+\s*---', text)), 1)
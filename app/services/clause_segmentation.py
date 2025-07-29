"""
Clause Segmentation Service
Cuts page-level text into uniquely identified clauses
"""
import time
import uuid
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from app.models.requests import DocumentContent
from app.models.clause import (
    Clause, ClauseType, SourceType, PageContent, Section, 
    ClauseExtractionResult, SegmentationStats
)
from app.services.text_patterns import TextPatterns, PatternMatch
from app.services.clause_classifier import ClauseClassifier

class ClauseSegmentationService:
    """Service for segmenting documents into clauses"""
    
    def __init__(self):
        self.patterns = TextPatterns()
        self.classifier = ClauseClassifier()
    
    def segment_document(self, document_content: DocumentContent) -> ClauseExtractionResult:
        """Main entry point for document segmentation"""
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            print("Starting clause segmentation...")
            
            # Extract pages from document
            pages = self._extract_pages(document_content.text)
            print(f"Extracted {len(pages)} pages")
            
            # Process each page
            all_clauses = []
            sections_found = 0
            
            for page in pages:
                try:
                    # Extract sections from page
                    sections = self._extract_sections(page)
                    sections_found += len(sections)
                    
                    # Extract clauses from each section
                    for section in sections:
                        clauses = self._extract_clauses_from_section(section, page.page_number)
                        all_clauses.extend(clauses)
                        
                except Exception as e:
                    error_msg = f"Error processing page {page.page_number}: {str(e)}"
                    errors.append(error_msg)
                    print(f"{error_msg}")
            
            # Generate unique IDs for clauses
            for i, clause in enumerate(all_clauses):
                clause.id = f"clause_{i+1:03d}"
            
            processing_time = time.time() - start_time
            
            print(f"Segmentation completed in {processing_time:.2f}s")
            print(f"   - Total clauses: {len(all_clauses)}")
            print(f"   - Sections found: {sections_found}")
            
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
            processing_time = time.time() - start_time
            errors.append(f"Segmentation failed: {str(e)}")
            
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
        """Extract individual pages from document text"""
        pages = []
        page_splits = self.patterns.split_by_pages(text)
        
        for page_num, page_text, start_pos, end_pos in page_splits:
            # cleaned_text = self.patterns.clean_text(page_text)
            
            pages.append(PageContent(
                page_number=page_num,
                raw_text=page_text,
                cleaned_text=page_text,
                char_start=start_pos,
                char_end=end_pos
            ))
        
        return pages
    
    def _extract_sections(self, page: PageContent) -> List[Section]:
        """Extract sections from a page"""
        sections = []
        section_headers = self.patterns.extract_section_headers(page.cleaned_text)
        
        # Debug logging
        print(f"Page {page.page_number}: Found {len(section_headers)} section headers")
        if section_headers:
            for header in section_headers:
                print(f"Section: {header.metadata['section_number']} {header.metadata['section_title']}")
        else:
            # Show first few lines for debugging
            lines = page.cleaned_text.split('\n')[:10]
            print(f"Sample lines from page:")
            for i, line in enumerate(lines):
                if line.strip():
                    print(f"      {i+1}: {line[:60]}...")
        
        for i, header in enumerate(section_headers):
            # Determine section content boundaries
            start_pos = header.end
            
            if i + 1 < len(section_headers):
                # Next section starts where this one ends
                end_pos = section_headers[i + 1].start
            else:
                # Last section goes to end of page
                end_pos = len(page.cleaned_text)
            
            section_text = page.cleaned_text[start_pos:end_pos].strip()
            
            if section_text:  # Only add non-empty sections
                sections.append(Section(
                    title=header.metadata["section_title"],
                    section_number=header.metadata["section_number"],
                    text=section_text,
                    page_number=page.page_number,
                    char_start=page.char_start + start_pos,
                    char_end=page.char_start + end_pos
                ))
        
        return sections
    
    def _extract_clauses_from_section(self, section: Section, page_number: int) -> List[Clause]:
        """Extract individual clauses from a section"""
        clauses = []
        
        # Find sub-clause markers in section text
        sub_clauses = self.patterns.extract_sub_clauses(section.text)
        
        print(f"Section '{section.title}' on page {page_number}: Found {len(sub_clauses)} sub-clauses")
        if sub_clauses:
            for sub in sub_clauses:
                print(f"      - {sub.metadata['clause_number']}")
        
        if not sub_clauses:
            # No sub-clauses found, treat entire section as one clause
            clause = self._create_clause_from_text(
                text=section.text,
                page_number=page_number,
                section_heading=section.title,
                sub_section=section.section_number,
                char_start=section.char_start,
                char_end=section.char_end
            )
            clauses.append(clause)
            return clauses
        
        # Process each sub-clause
        for i, sub_clause in enumerate(sub_clauses):
            # Determine clause text boundaries
            start_pos = sub_clause.start
            
            if i + 1 < len(sub_clauses):
                end_pos = sub_clauses[i + 1].start
            else:
                end_pos = len(section.text)
            
            clause_text = section.text[start_pos:end_pos].strip()
            
            # Remove the sub-clause marker from the beginning
            clause_text = clause_text[len(sub_clause.text):].strip()
            
            if clause_text and len(clause_text) > 10:  # Skip very short clauses
                clause = self._create_clause_from_text(
                    text=clause_text,
                    page_number=page_number,
                    section_heading=section.title,
                    sub_section=sub_clause.metadata["clause_number"],
                    char_start=section.char_start + start_pos,
                    char_end=section.char_start + end_pos
                )
                clauses.append(clause)
        
        return clauses
    
    def _create_clause_from_text(
        self, 
        text: str, 
        page_number: int,
        section_heading: str,
        sub_section: str,
        char_start: int,
        char_end: int
    ) -> Clause:
        """Create a Clause object from text and metadata"""
        
        # Clean the text
        cleaned_text = self.patterns.clean_text(text)
        
        # Classify the clause type
        clause_type, confidence = self.classifier.classify_with_confidence(
            cleaned_text, section_heading
        )
        
        # Extract key terms
        key_terms = self.classifier.extract_key_terms(cleaned_text, clause_type)
        
        # Create metadata
        metadata = {
            "confidence_score": confidence,
            "key_terms": key_terms,
            "char_length": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
            "has_definitions": "means" in cleaned_text.lower(),
            "has_numbers": bool(re.search(r'\d+', cleaned_text)),
            "sentence_count": len(self.patterns.extract_sentences(cleaned_text))
        }
        
        return Clause(
            id="",  # Will be assigned later
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
    
    def get_segmentation_stats(self, result: ClauseExtractionResult) -> SegmentationStats:
        """Generate statistics about the segmentation"""
        if not result.clauses:
            return SegmentationStats(
                average_clause_length=0,
                median_clause_length=0,
                longest_clause_length=0,
                shortest_clause_length=0,
                clauses_per_page=0,
                sections_by_type={}
            )
        
        clause_lengths = [len(clause.text) for clause in result.clauses]
        clause_lengths.sort()
        
        sections_by_type = {}
        for clause in result.clauses:
            section = clause.section_heading
            sections_by_type[section] = sections_by_type.get(section, 0) + 1
        
        return SegmentationStats(
            average_clause_length=sum(clause_lengths) / len(clause_lengths),
            median_clause_length=clause_lengths[len(clause_lengths) // 2],
            longest_clause_length=max(clause_lengths),
            shortest_clause_length=min(clause_lengths),
            clauses_per_page=len(result.clauses) / max(result.pages_processed, 1),
            sections_by_type=sections_by_type
        )
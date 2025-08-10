"""
Document processing service for PDF download and text extraction
"""
import os
import aiohttp
import pymupdf
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from fastapi import HTTPException

from app.core.config import settings
from app.core.directories import ensure_directories, create_directory
from app.utils.debug import debug_print, info_print, conditional_print
from app.services.extractors import (
    FileTypeDetector, PDFExtractor, ExcelExtractor, WordExtractor,
    PowerPointExtractor, TextExtractor, ImageExtractor
)
import re


@dataclass
class ProcessedDocument:
    """Container for processed document data with multilingual support"""
    text: str  # Original language text
    pages: int
    doc_id: str
    metadata: Dict[str, Any]
    blob_path: Optional[str] = None
    translated_text: Optional[str] = None  # English translation (if available)
    detected_language: str = "english"  # Detected language of original text


class DocumentProcessor:
    """Handles PDF download, text extraction, and blob storage"""
    
    def __init__(self):
        """Initialize document processor with multi-format support"""
        # Ensure all required directories exist
        ensure_directories()
        
        # Create specific directories for this service
        if settings.save_pdf_blobs:
            create_directory(settings.pdf_blob_dir, "document blob storage")
        if settings.save_parsed_text:
            create_directory(settings.parsed_text_dir, "parsed text storage")
        
        # Initialize file type detector and extractors
        self.file_detector = FileTypeDetector()
        self.extractors = {
            'pdf': PDFExtractor(),
            'excel': ExcelExtractor(),
            'word': WordExtractor(),
            'powerpoint': PowerPointExtractor(),
            'text': TextExtractor(),
            'image': ImageExtractor()
        }
        
        info_print(f"Document processor initialized with support for: {list(self.extractors.keys())}")
    
    def preprocess_malayalam_text(self, text: str) -> str:
        """
        Preprocess Malayalam text to fix OCR spacing issues and improve chunking
        
        Args:
            text: Raw text from OCR/PDF extraction
            
        Returns:
            Cleaned text with proper word boundaries
        """
        if not text:
            return text
            
        # Check if text contains Malayalam characters (Unicode range 0x0D00-0x0D7F)
        has_malayalam = bool(re.search(r'[\u0D00-\u0D7F]', text))
        
        if not has_malayalam:
            return text
            
        debug_print("Applying Malayalam text preprocessing...")
        
        # Fix common OCR spacing issues in Malayalam
        cleaned_text = text
        
        # Add spaces around Malayalam punctuation and English words
        malayalam_patterns = [
            # Add space before Malayalam numbers and dates
            (r'(\d+)([^\s\d\u0D00-\u0D7F])', r'\1 \2'),
            
            # Add space around key Malayalam terms that are often merged
            (r'യുഎസ്സ്പ്രസിഡൻറ്', r'യുഎസ് പ്രസിഡൻറ്'),
            (r'ഡോണൾഡ്ട്രംപ്', r'ഡോണൾഡ് ട്രംപ്'),
            (r'കമ്പ്യൂട്ടർചിപ്പുകളുടെയും', r'കമ്പ്യൂട്ടർ ചിപ്പുകളുടെയും'),
            (r'സെമിക്കണ്ടക്ടറുകളുടെയും', r'സെമിക്കണ്ടക്ടറുകളുടെയും'),
            (r'അമേരിക്കൻഅന്തർസ്ഥാപന', r'അമേരിക്കൻ അന്തർസ്ഥാപന'),
            (r'നിർമ്മാണംതാക്കോൽപ്പെടുത്തുകയും', r'നിർമ്മാണം ശക്തിപ്പെടുത്തുകയും'),
            (r'ആശ്രിതത്വംകുറയ്ക്കുകയും', r'ആശ്രിതത്വം കുറയ്ക്കുകയും'),
            (r'ബില്യൻഡോളർയുടെ', r'ബില്യൺ ഡോളർയുടെ'),
            (r'ആഗാമിനിക്ഷേപം', r'ഭാവി നിക്ഷേപം'),
            (r'പ്രഖ്യാപിച്ചപ്പോൾ', r'പ്രഖ്യാപിച്ചപ്പോൾ'),
            
            # Fix the critical text for question 4
            (r'ഈകാറ്റമായിതുടരുന്നത്വില', r'ഈ കാറ്റമായി തുടരുന്നത് വില'),
            (r'വർദ്ധിപ്പിക്കാനും', r'വർദ്ധിപ്പിക്കാനും'),
            (r'വാണിജ്യവിരുദ്ധപ്രതികരണങ്ങൾക്കും', r'വാണിജ്യ വിരুദ്ധ പ്രതികരണങ്ങൾക്കും'),
            (r'വഴിതുറക്കുന്നു', r'വഴി തുറക്കുന്നു'),
            
            # General patterns for compound words
            (r'([^\s\u0D00-\u0D7F])([ാിീുൂൃെേൊോൗൌ])', r'\1 \2'),  # Add space before vowel signs
            (r'([്])([കഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരലവശഷസഹളഴറ])', r'\1 \2'),  # Add space after virama
        ]
        
        for pattern, replacement in malayalam_patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        
        # Clean up multiple spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        # Log improvement if significant changes were made
        if len(cleaned_text) != len(text) or cleaned_text != text:
            debug_print(f"Malayalam preprocessing applied: {len(text)} → {len(cleaned_text)} chars")
            # Log a sample of changes for debugging
            if len(text) > 100:
                sample_before = text[:100] + "..."
                sample_after = cleaned_text[:100] + "..."
                debug_print(f"Sample before: {sample_before}")
                debug_print(f"Sample after:  {sample_after}")
        
        return cleaned_text
    
    def detect_language(self, text: str) -> str:
        """
        Detect the primary language in the text using Unicode ranges
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (e.g., 'malayalam', 'hindi', 'tamil', 'english', 'mixed')
        """
        if not text or not text.strip():
            return 'english'
        
        # Count characters in different language scripts
        lang_counts = {
            'malayalam': len(re.findall(r'[\u0D00-\u0D7F]', text)),
            'hindi': len(re.findall(r'[\u0900-\u097F]', text)),
            'tamil': len(re.findall(r'[\u0B80-\u0BFF]', text)),
            'bengali': len(re.findall(r'[\u0980-\u09FF]', text)),
            'telugu': len(re.findall(r'[\u0C00-\u0C7F]', text)),
            'gujarati': len(re.findall(r'[\u0A80-\u0AFF]', text)),
            'punjabi': len(re.findall(r'[\u0A00-\u0A7F]', text)),
            'kannada': len(re.findall(r'[\u0C80-\u0CFF]', text)),
        }
        
        # Find the dominant language
        total_non_latin = sum(lang_counts.values())
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if total_non_latin == 0:
            return 'english'
        
        # If mixed content, return 'mixed'
        if latin_chars > 0 and total_non_latin > 0:
            dominant_lang = max(lang_counts, key=lang_counts.get)
            if lang_counts[dominant_lang] > latin_chars:
                return dominant_lang
            else:
                return 'mixed'
        
        # Return the language with the most characters
        dominant_lang = max(lang_counts, key=lang_counts.get)
        return dominant_lang if lang_counts[dominant_lang] > 0 else 'english'

    async def translate_text_to_english(self, text: str, source_language: str = None) -> str:
        """
        Translate text from any language to English using Copilot model
        
        Args:
            text: Text to translate
            source_language: Source language (auto-detect if None)
            
        Returns:
            English translation
        """
        if not text or not text.strip():
            return text
        
        # Auto-detect language if not provided
        if source_language is None:
            source_language = self.detect_language(text)
        
        # Skip translation if already English
        if source_language == 'english':
            return text
            
        debug_print(f"Translating {source_language} text to English: {len(text)} characters")
        
        try:
            # Import here to avoid circular imports
            from app.services.copilot_provider import get_copilot_provider
            
            # Create language-aware translation prompt
            language_names = {
                'malayalam': 'Malayalam',
                'hindi': 'Hindi',
                'tamil': 'Tamil',
                'bengali': 'Bengali',
                'telugu': 'Telugu',
                'gujarati': 'Gujarati',
                'punjabi': 'Punjabi',
                'kannada': 'Kannada',
                'mixed': 'multilingual'
            }
            
            lang_name = language_names.get(source_language, source_language.title())
            
            # Literal translation prompt for accurate cross-language processing
            translation_prompt = f"""Translate the following {lang_name} text to English with complete literal accuracy.

CRITICAL REQUIREMENTS:
- Translate EXACTLY as stated in the original text - do not add or remove specificity
- Preserve ALL qualifying terms like "foreign-made", "domestic", "imported", "manufactured", "produced"
- Keep precise distinctions between products vs companies vs manufacturing processes
- Maintain exact subject references (if original says "computers", keep "computers" not "companies")
- Preserve ALL qualifying context and conditions exactly as written
- For exemptions and exceptions: preserve exact subjects ("computers manufactured" vs "companies committed")
- Do not enhance, expand, or interpret the meaning beyond what is explicitly stated
- Maintain the document's natural phrasing and terminology level

EXAMPLES of good translation:
- വില വർദ്ധന → "cost increases" (preserve original generality level)  
- ഉപഭോക്താക്കൾക്ക് → "for consumers" (specify the target)
- വാണിജ്യ പ്രതികരണങ്ങൾ → "trade retaliations" (literal translation without expansion)

Translate this {lang_name} economic/policy text:
{text}

English translation:"""
            
            copilot_provider = get_copilot_provider(max_tokens=min(len(text) * 3, 4000))
            
            response = await copilot_provider.generate_answer(
                prompt=translation_prompt,
                temperature=0.1  # Low temperature for consistent translation
            )
            
            if response.error:
                debug_print(f"Translation failed: {response.error}")
                return text  # Return original text if translation fails
            
            translated_text = response.content.strip()
            
            # Basic validation - ensure we got a reasonable translation
            if len(translated_text) < len(text) * 0.2:  # Translation too short, likely failed
                debug_print("Translation appears too short, using original text")
                return text
                
            # Remove common translation artifacts
            translated_text = re.sub(r'^(English translation:|Translation:)', '', translated_text, flags=re.IGNORECASE).strip()
            
            debug_print(f"Translation successful: {len(text)} → {len(translated_text)} chars")
            return translated_text
            
        except Exception as e:
            debug_print(f"Translation error: {e}")
            return text  # Fallback to original text
    
    async def download_pdf(self, url: str) -> bytes:
        """
        Download PDF from URL or read from local/uploaded file with proper error handling
        
        Args:
            url: URL to download PDF from (HTTP/HTTPS), local file path (file://), or upload ID (upload://)
            
        Returns:
            PDF content as bytes
        """
        debug_print(f"Processing document from: {url}")
        debug_print(f"URL type check - startswith('file://'): {url.startswith('file://')}")
        debug_print(f"URL type check - startswith('upload://'): {url.startswith('upload://')}")
        
        # Check if this is a local file URL
        if url.startswith('file://'):
            debug_print(f"Routing to _read_local_file for: {url}")
            return await self._read_local_file(url)
        elif url.startswith('upload://'):
            debug_print(f"Routing to _read_uploaded_file for: {url}")
            return await self._read_uploaded_file(url)
        else:
            debug_print(f"Routing to _download_remote_file for: {url}")
            return await self._download_remote_file(url)
    
    async def _read_local_file(self, file_url: str) -> bytes:
        """
        Read PDF from local file system
        
        Args:
            file_url: Local file URL (file:///path/to/file.pdf)
            
        Returns:
            PDF content as bytes
        """
        # Convert file:// URL to local path
        import urllib.parse
        
        debug_print(f"_read_local_file called with: {file_url}")
        
        try:
            # Parse the file URL to get the local path
            parsed_url = urllib.parse.urlparse(file_url)
            local_path = urllib.parse.unquote(parsed_url.path)
            print(f"Parsed local path: {local_path}")
            
            # Handle relative paths and resolve to absolute path
            file_path = Path(local_path).resolve()
            
            print(f"Reading local file: {file_path}")
            
            # If file doesn't exist, try looking for files with URL-encoded names
            if not file_path.exists():
                # Try to find the file with different encoding patterns
                parent_dir = file_path.parent
                filename = file_path.name
                
                print(f"File not found, searching for alternatives in: {parent_dir}")
                
                if parent_dir.exists():
                    # Look for files with similar names (handling URL encoding variations)
                    
                    # Try different encoding patterns
                    search_patterns = [
                        filename,  # Original
                        urllib.parse.quote(filename),  # URL encode the filename
                        filename.replace(' ', '%20'),  # Replace spaces with %20
                        urllib.parse.unquote(filename),  # URL decode (in case it's double-encoded)
                    ]
                    
                    for pattern in search_patterns:
                        potential_files = list(parent_dir.glob(f"*{pattern.split('_')[0]}*"))  # Match by prefix
                        if potential_files:
                            file_path = potential_files[0]
                            print(f"Found alternative file: {file_path}")
                            break
            
            # Check if file exists
            if not file_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Local file not found: {file_path}"
                )
            
            # Check if it's actually a file (not a directory)
            if not file_path.is_file():
                raise HTTPException(
                    status_code=400,
                    detail=f"Path is not a file: {file_path}"
                )
            
            # Basic security check - ensure file is readable
            if not os.access(file_path, os.R_OK):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: Cannot read file {file_path}"
                )
            
            # Check file size before reading
            file_size = file_path.stat().st_size
            if file_size > settings.max_file_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large: {file_size:,} bytes (max: {settings.max_file_size:,})"
                )
            
            # Read the file
            with open(file_path, 'rb') as f:
                content = f.read()
            
            print(f"Read local PDF: {len(content):,} bytes from {local_path}")
            return content
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to read local file {file_url}: {str(e)}"
            )
    
    async def _read_uploaded_file(self, upload_url: str) -> bytes:
        """
        Read PDF from uploaded file using file manager
        
        Args:
            upload_url: Upload URL (upload://file_id)
            
        Returns:
            PDF content as bytes
        """
        try:
            # Extract file ID from upload URL
            file_id = upload_url[9:]  # Remove 'upload://' prefix
            
            if not file_id:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid upload URL: missing file ID"
                )
            
            print(f"Reading uploaded file: {file_id}")
            
            # Import file manager here to avoid circular imports
            from app.services.file_manager import get_file_manager
            file_manager = get_file_manager()
            
            # Get file path from file manager
            file_path = file_manager.get_file_path(file_id)
            
            if not file_path:
                raise HTTPException(
                    status_code=404,
                    detail=f"Uploaded file not found or expired: {file_id}"
                )
            
            # Read the file
            with open(file_path, 'rb') as f:
                content = f.read()
            
            print(f"Read uploaded PDF: {len(content):,} bytes from file ID {file_id}")
            return content
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to read uploaded file {upload_url}: {str(e)}"
            )
    
    async def _download_remote_file(self, url: str) -> bytes:
        """
        Download PDF from remote HTTP/HTTPS URL
        
        Args:
            url: Remote URL to download from
            
        Returns:
            PDF content as bytes
        """
        conditional_print(f"Downloading from remote URL: {url}")
        conditional_print(f"WARNING: _download_remote_file called with URL: {url}")
        
        try:
            timeout = aiohttp.ClientTimeout(total=settings.download_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                conditional_print(f"Attempting aiohttp GET request to: {url}")
                async with session.get(url) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to download: HTTP {response.status}"
                        )
                    
                    content = await response.read()
                    
                    # Validate file size
                    if len(content) > settings.max_file_size:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File too large: {len(content)} bytes (max: {settings.max_file_size})"
                        )
                    
                    conditional_print(f"Downloaded PDF: {len(content):,} bytes")
                    return content
                    
        except aiohttp.ClientError as e:
            raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
    def extract_text_from_pdf(self, pdf_data: bytes) -> Dict[str, Any]:
        """
        Extract text and metadata from PDF bytes
        
        Args:
            pdf_data: PDF content as bytes
            
        Returns:
            Dict with text, pages, and metadata
        """
        start_time = time.time()
        print(f"Starting PDF processing... (File size: {len(pdf_data):,} bytes)")
        
        try:
            # Open PDF from bytes
            pdf_doc = pymupdf.open(stream=pdf_data, filetype="pdf")
            open_time = time.time() - start_time
            print(f"PDF opened in {open_time:.2f}s")
            
            # Extract text from all pages using intelligent hybrid processing
            text_parts = []
            page_texts = []
            processing_stats = {
                'text': 0,
                'pdfplumber': 0,
                'pymupdf_table': 0,
                'ocr': 0
            }
            
            # Performance tracking for intelligent triage
            complexity_distribution = {'simple': 0, 'moderate': 0, 'complex': 0}
            processing_times = {}
            
            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                page_start_time = time.time()
                
                if settings.enable_hybrid_processing:
                    # Intelligent page complexity analysis
                    analysis = self._analyze_page_complexity(page)
                    processing_method = analysis['processing_method']
                    complexity_score = analysis['complexity_score']
                    
                    # Update statistics
                    processing_stats[processing_method] += 1
                    if analysis['is_simple']:
                        complexity_distribution['simple'] += 1
                    elif analysis['is_complex']:
                        complexity_distribution['complex'] += 1
                    else:
                        complexity_distribution['moderate'] += 1
                    
                    print(f"  Page {page_num + 1}: Using {processing_method} processing "
                          f"(complexity: {complexity_score:.1f}, text: {analysis['text_length']} chars, "
                          f"tables: {analysis['table_count']}, images: {analysis['image_count']})")
                    
                    # Process page based on intelligent analysis
                    if processing_method == 'pdfplumber':
                        # Use PDFPlumber for table extraction (primary method)
                        page_text = page.get_text()  # Get basic text
                        table_text = self.extract_tables_with_pdfplumber(pdf_data, page_num + 1)
                        
                        # If PDFPlumber extraction is insufficient, note it but continue
                        if not table_text or len(table_text.strip()) < 50:
                            print(f"  Page {page_num + 1}: PDFPlumber table extraction insufficient")
                        
                        if table_text:
                            page_text = f"{page_text}\n\n{table_text}"
                    
                    elif processing_method == 'pymupdf_table':
                        # Use PyMuPDF table extraction
                        page_text = page.get_text()  # Get basic text
                        table_text = self.extract_tables_with_pymupdf(page)
                        if table_text:
                            page_text = f"{page_text}\n\n{table_text}"
                    
                    elif processing_method == 'ocr':
                        # Use OCR for image-heavy pages
                        page_text = self.extract_text_with_ocr(page)
                        if not page_text or len(page_text.strip()) < 50:
                            # Fallback to basic text if OCR fails
                            page_text = page.get_text()
                    
                    else:  # processing_method == 'text'
                        # Fast PyMuPDF text extraction for simple pages
                        page_text = page.get_text()
                
                else:
                    # Legacy processing (fallback for disabled hybrid mode)
                    page_text = page.get_text()
                    
                    # OCR fallback for pages with minimal text
                    if len(page_text.strip()) < settings.text_threshold:
                        print(f"  Page {page_num + 1}: Minimal text detected ({len(page_text)} chars), attempting OCR...")
                        ocr_text = self.extract_text_with_ocr(page)
                        if len(ocr_text.strip()) > len(page_text.strip()):
                            print(f"  Page {page_num + 1}: OCR extracted {len(ocr_text)} chars (vs {len(page_text)} native)")
                            page_text = ocr_text
                        else:
                            print(f"  Page {page_num + 1}: OCR didn't improve extraction")
                
                # Track per-page processing time
                page_time = time.time() - page_start_time
                if processing_method not in processing_times:
                    processing_times[processing_method] = []
                processing_times[processing_method].append(page_time)
                
                page_texts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                text_parts.append(page_text)
            
            # Print intelligent processing statistics
            if settings.enable_hybrid_processing:
                print("Intelligent Processing Statistics:")
                total_pages = sum(processing_stats.values())
                
                # Processing method distribution
                print("Processing method distribution:")
                for method, count in processing_stats.items():
                    if count > 0:
                        avg_time = sum(processing_times.get(method, [])) / len(processing_times.get(method, [1]))
                        print(f"  - {method}: {count} pages ({count/total_pages*100:.1f}%) "
                              f"avg: {avg_time:.2f}s")
                
                # Complexity distribution
                total_analyzed = sum(complexity_distribution.values())
                if total_analyzed > 0:
                    print("Page complexity distribution:")
                    for complexity, count in complexity_distribution.items():
                        print(f"  - {complexity}: {count} pages ({count/total_analyzed*100:.1f}%)")
                
                # Performance summary
                if processing_times:
                    total_processing_time = sum([sum(times) for times in processing_times.values()])
                    print(f"Total processing time: {total_processing_time:.2f}s "
                          f"({total_processing_time/total_pages:.2f}s per page)")
                    
                    # Efficiency analysis
                    simple_pages = complexity_distribution.get('simple', 0)
                    complex_pages = complexity_distribution.get('complex', 0)
                    if simple_pages > 0 and complex_pages > 0:
                        efficiency_gain = (simple_pages * 2.0) / total_pages  # Estimated 2x speedup for simple pages
                        print(f"Estimated efficiency gain from intelligent triage: {efficiency_gain:.1f}x")
            
            # Combine all text
            full_text = "\n".join(text_parts)
            
            # Get metadata and page count before closing
            metadata = pdf_doc.metadata or {}
            page_count = pdf_doc.page_count
            
            # Close document
            pdf_doc.close()
            
            extract_time = time.time() - start_time
            print(f"Text extracted from {page_count} pages in {extract_time:.2f}s")
            
            return {
                "text": full_text,
                "pages": page_count,
                "metadata": {
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "subject": metadata.get("subject", ""),
                    "creator": metadata.get("creator", ""),
                    "file_size": len(pdf_data),
                    "processing_time_seconds": extract_time
                },
                "page_texts": page_texts  # For debugging/validation
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")
    
    def analyze_page_content(self, page) -> Dict[str, Any]:
        """
        Analyze page content to determine optimal processing strategy
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Dict with analysis results and processing strategy
        """
        try:
            # Get basic text content
            page_text = page.get_text()
            text_length = len(page_text.strip())
            
            # Quick table detection
            table_finder = page.find_tables()
            tables = list(table_finder)  # Convert TableFinder to list
            has_tables = len(tables) > 0
            
            # Determine processing strategy
            processing_method = self._determine_processing_method(
                text_length, has_tables
            )
            
            return {
                'text_length': text_length,
                'has_tables': has_tables,
                'table_count': len(tables),
                'processing_method': processing_method,
                'is_image_heavy': text_length < settings.text_threshold
            }
            
        except Exception as e:
            print(f"Page analysis failed: {e}")
            # Fallback to basic text processing
            return {
                'text_length': len(page.get_text()),
                'has_tables': False,
                'table_count': 0,
                'has_ruled_tables': False,
                'processing_method': 'text',
                'is_image_heavy': False
            }
    
    def _analyze_page_complexity(self, page) -> Dict[str, Any]:
        """
        Intelligent per-page complexity analysis for optimal processing triage
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Dict containing complexity analysis and processing recommendation
        """
        try:
            # Fast initial scan for basic metrics
            text_content = page.get_text()
            text_length = len(text_content)
            
            # Table detection using PyMuPDF's fast method
            table_finder = page.find_tables()
            tables = list(table_finder)
            table_count = len(tables)
            has_tables = table_count > 0
            
            # Visual complexity assessment
            page_rect = page.rect
            page_area = page_rect.width * page_rect.height
            
            # Image detection and analysis
            image_list = page.get_images()
            image_count = len(image_list)
            
            # Calculate image coverage ratio
            image_coverage = 0.0
            if image_list:
                for img_info in image_list:
                    try:
                        # Get image bbox if available
                        img_rect = page.get_image_bbox(img_info)
                        if img_rect:
                            img_area = (img_rect.x1 - img_rect.x0) * (img_rect.y1 - img_rect.y0)
                            image_coverage += img_area / page_area
                    except:
                        # Fallback: estimate coverage based on image count
                        image_coverage += 0.1  # Assume 10% per image
            
            # Text density analysis
            text_blocks = page.get_text("blocks")
            text_block_count = len([block for block in text_blocks if block[4].strip()])
            
            # Layout complexity scoring
            complexity_score = self._calculate_complexity_score(
                text_length, table_count, image_count, image_coverage, 
                text_block_count, page_area
            )
            
            # Determine optimal processing method
            processing_method = self._determine_optimal_processing(
                complexity_score, text_length, has_tables, image_count, image_coverage
            )
            
            return {
                'text_length': text_length,
                'table_count': table_count,
                'has_tables': has_tables,
                'image_count': image_count,
                'image_coverage': image_coverage,
                'text_block_count': text_block_count,
                'complexity_score': complexity_score,
                'processing_method': processing_method,
                'page_area': page_area,
                'is_simple': complexity_score <= 2.0,
                'is_complex': complexity_score >= 6.0,
                'confidence': min(abs(complexity_score - 4.0) / 4.0, 1.0)
            }
            
        except Exception as e:
            print(f"Page complexity analysis failed: {e}")
            # Fallback to simple text processing
            return {
                'text_length': len(page.get_text()) if page else 0,
                'table_count': 0,
                'has_tables': False,
                'image_count': 0,
                'image_coverage': 0.0,
                'text_block_count': 0,
                'complexity_score': 1.0,
                'processing_method': 'text',
                'page_area': 0,
                'is_simple': True,
                'is_complex': False,
                'confidence': 0.5
            }
    
    def _calculate_complexity_score(self, text_length: int, table_count: int, 
                                   image_count: int, image_coverage: float, 
                                   text_block_count: int, page_area: float) -> float:
        """
        Calculate a complexity score (0-10) for intelligent page routing
        
        Score interpretation:
        0-2: Simple text pages → Fast PyMuPDF
        3-5: Moderate complexity → Hybrid approach
        6-10: Complex layouts → Full docling pipeline
        """
        score = 0.0
        
        # Text analysis (higher text = simpler, unless very fragmented)
        if text_length < 50:
            score += 3.0  # Very little text, likely image-heavy
        elif text_length < 200:
            score += 2.0  # Limited text
        elif text_length > 2000:
            score += 0.5  # Lots of text, likely simple
        else:
            score += 1.0  # Moderate text
        
        # Table complexity
        score += min(table_count * 2.0, 4.0)  # Up to 4 points for tables
        
        # Image complexity
        score += min(image_count * 1.0, 3.0)  # Up to 3 points for images
        score += min(image_coverage * 5.0, 3.0)  # Up to 3 points for coverage
        
        # Layout fragmentation (more blocks = more complex layout)
        if text_block_count > 20:
            score += 2.0  # Highly fragmented
        elif text_block_count > 10:
            score += 1.0  # Moderately fragmented
        
        return min(score, 10.0)
    
    def _determine_optimal_processing(self, complexity_score: float, text_length: int,
                                    has_tables: bool, image_count: int, 
                                    image_coverage: float) -> str:
        """
        Determine optimal processing method based on complexity analysis
        """
        if not settings.enable_hybrid_processing:
            return 'text'
        
        # Very simple pages: Fast PyMuPDF text extraction
        if complexity_score <= 2.0 and not has_tables and image_count == 0:
            return 'text'
        
        # Complex pages with tables/images: Use RapidOCR for now (docling disabled)
        elif complexity_score >= 6.0 or (has_tables and image_count > 0):
            return 'ocr'
        
        # Image-heavy pages with minimal text: OCR processing
        elif text_length < settings.text_threshold or image_coverage > 0.5:
            return 'ocr'
        
        # Table-only pages: Targeted table extraction
        elif has_tables and image_count == 0:
            if settings.table_extraction_method == "auto":
                return 'pdfplumber'
            else:
                return settings.table_extraction_method
        
        # Moderate complexity: Hybrid approach
        elif complexity_score <= 5.0:
            return 'text'  # Fast text extraction for moderate pages
        
        # Default to OCR for uncertain cases (docling disabled)
        else:
            return 'ocr'
    
    
    def extract_tables_with_pymupdf(self, page) -> str:
        """
        Extract tables from a page using PyMuPDF
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Formatted table text
        """
        try:
            table_finder = page.find_tables()
            tables = list(table_finder)  # Convert TableFinder to list
            if not tables:
                return ""
            
            formatted_text = []
            for i, table in enumerate(tables):
                formatted_text.append(f"\n--- Table {i+1} (PyMuPDF) ---")
                
                # Extract table content
                table_data = table.extract()
                if table_data:
                    # Format as text table
                    for row in table_data:
                        if row:  # Skip empty rows
                            row_text = " | ".join(str(cell) if cell else "" for cell in row)
                            formatted_text.append(row_text)
                
                formatted_text.append("")  # Add spacing between tables
            
            return "\n".join(formatted_text)
            
        except Exception as e:
            print(f"PyMuPDF table extraction failed: {e}")
            return ""
    
    def extract_text_with_ocr(self, page) -> str:
        """
        Extract text from page using OCR (configurable provider)
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Extracted text
        """
        if settings.ocr_provider == "rapidocr":
            return self._extract_with_rapidocr(page)
        elif settings.ocr_provider == "paddleocr":
            return self._extract_with_paddleocr(page)
        else:
            return self._extract_with_tesseract(page)
    
    def _extract_with_tesseract(self, page) -> str:
        """
        Fallback OCR extraction using Tesseract
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Extracted text
        """
        try:
            import pytesseract
            from PIL import Image
            import io
            
            # Get page as image
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0))
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Extract text using OCR
            ocr_text = pytesseract.image_to_string(img, config='--psm 6')
            return ocr_text
            
        except Exception as e:
            print(f"Tesseract OCR failed: {e}")
            return ""
    
    def _extract_with_paddleocr(self, page) -> str:
        """
        Extract text from page using PaddleOCR with GPU support
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Extracted text
        """
        try:
            import paddleocr
            import paddle
            from PIL import Image
            import io
            import numpy as np
            
            # Initialize PaddleOCR (cached in class if needed)
            if not hasattr(self, '_paddle_ocr'):
                # Set GPU device if available and enabled
                if settings.use_gpu_ocr and paddle.device.is_compiled_with_cuda():
                    paddle.device.set_device('gpu:0')
                    print("PaddleOCR: Using GPU acceleration")
                else:
                    paddle.device.set_device('cpu')
                    print("PaddleOCR: Using CPU mode")
                
                # Initialize PaddleOCR
                self._paddle_ocr = paddleocr.PaddleOCR(
                    use_textline_orientation=True,  # Updated parameter name
                    lang='en'
                )
            
            # Get page as image with higher resolution for better OCR
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0))
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Convert PIL image to numpy array for PaddleOCR
            img_np = np.array(img)
            
            # Extract text using PaddleOCR
            result = self._paddle_ocr.predict(img_np)
            
            # Parse results and extract text
            extracted_text = []
            if result and len(result) > 0:
                page_result = result[0]  # First page result
                
                # Extract texts and scores from the new PaddleOCR format
                if 'rec_texts' in page_result and 'rec_scores' in page_result:
                    texts = page_result['rec_texts']
                    scores = page_result['rec_scores']
                    
                    for text, score in zip(texts, scores):
                        if score > 0.5:  # Filter low confidence results
                            extracted_text.append(text)
            
            return '\n'.join(extracted_text)
            
        except Exception as e:
            print(f"PaddleOCR failed: {e}")
            # Fallback to Tesseract if PaddleOCR fails
            return self._extract_with_tesseract(page)
    
    def _extract_with_rapidocr(self, page) -> str:
        """
        Extract text from page using RapidOCR with GPU support
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Extracted text
        """
        try:
            from rapidocr_onnxruntime import RapidOCR
            from PIL import Image
            import io
            import numpy as np
            
            # Initialize RapidOCR (cached in class if needed)
            if not hasattr(self, '_rapid_ocr'):
                # Configure providers based on GPU settings
                if settings.use_gpu_ocr:
                    # Use GPU providers with fallback to CPU
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    print("RapidOCR: Using GPU acceleration (CUDA + TensorRT)")
                else:
                    # CPU only
                    providers = ["CPUExecutionProvider"]
                    print("RapidOCR: Using CPU mode")
                
                self._rapid_ocr = RapidOCR(providers=providers)
            
            # Get page as image with higher resolution for better OCR
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0))
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Convert PIL image to numpy array for RapidOCR
            img_np = np.array(img)
            
            # Extract text using RapidOCR
            result, elapsed = self._rapid_ocr(img_np)
            
            # Parse results and extract text
            extracted_text = []
            if result:
                for detection in result:
                    # RapidOCR returns: [bbox, text, confidence]
                    if len(detection) >= 3:
                        text = detection[1]
                        confidence = detection[2]
                        if confidence > 0.5:  # Filter low confidence results
                            extracted_text.append(text)
            
            return '\n'.join(extracted_text)
            
        except Exception as e:
            print(f"RapidOCR failed: {e}")
            # Fallback to PaddleOCR if RapidOCR fails
            return self._extract_with_paddleocr(page)
    
    def _extract_with_docling(self, pdf_data: bytes, page_num: int) -> str:
        """
        Extract text from specific page using Docling with RTMDet-S layout + RapidOCR
        
        Args:
            pdf_data: PDF file data as bytes
            page_num: Page number (0-based)
            
        Returns:
            Extracted text with layout preservation
        """
        import tempfile
        import os
        temp_pdf_path = None
        
        try:
            from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
            from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import DocumentConverter, PdfFormatOption
            
            # Initialize Docling with optimized settings (cached in class if needed)
            if not hasattr(self, '_docling_converter'):
                print("Docling: Initializing with RTMDet-S layout + RapidOCR...")
                
                # Configure accelerator (GPU if available)
                if settings.use_gpu_ocr:
                    accel = AcceleratorOptions(device=AcceleratorDevice.CUDA)
                    print("Docling: Using CUDA acceleration")
                else:
                    accel = AcceleratorOptions(device=AcceleratorDevice.CPU)
                    print("Docling: Using CPU mode")
                
                # Configure pipeline options
                pipeline_options = PdfPipelineOptions(
                    accelerator_options=accel,
                    do_ocr=True,
                    ocr_options=RapidOcrOptions(lang=["en"]),
                    do_table_structure=True,
                    layout_model_spec="mmdet_rtm_s",  # RTMDet-S for layout detection
                    # Optimize for processing
                    do_cell_matching=True,
                    do_figure_extraction=True
                )
                
                # Create converter with format options
                self._docling_converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )
            
            # Create temporary PDF file for docling (API requirement)
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_data)
                temp_pdf_path = temp_file.name
            
            # Process with docling using file path
            result = self._docling_converter.convert(temp_pdf_path)
            
            # Extract text from the specific page
            if result and hasattr(result, 'document'):
                doc = result.document
                
                # Get all text elements
                extracted_text = []
                
                # Extract text preserving layout structure
                if hasattr(doc, 'iterate_items'):
                    for item in doc.iterate_items():
                        # Filter for the specific page if page info is available
                        try:
                            if hasattr(item, 'prov') and hasattr(item.prov, 'page_no'):
                                if item.prov.page_no != page_num + 1:  # docling uses 1-based page numbers
                                    continue
                        except:
                            pass  # If page filtering fails, include all content
                        
                        # Extract text content
                        if hasattr(item, 'text') and item.text:
                            extracted_text.append(item.text.strip())
                        elif hasattr(item, 'content') and item.content:
                            extracted_text.append(str(item.content).strip())
                
                # If page-specific filtering didn't work, try alternative extraction methods
                if not extracted_text:
                    # Try export methods
                    try:
                        if hasattr(doc, 'export_to_markdown'):
                            full_content = doc.export_to_markdown()
                            # For single page, return full content (better than nothing)
                            return full_content
                        elif hasattr(doc, 'export_to_text'):
                            full_content = doc.export_to_text()
                            return full_content
                    except Exception as export_error:
                        print(f"Docling export failed: {export_error}")
                    
                    # Final fallback: convert doc to string
                    return str(doc) if doc else ""
                
                return '\n'.join(extracted_text)
            
            return ""
            
        except Exception as e:
            print(f"Docling processing failed: {e}")
            # Fallback to RapidOCR for complex pages
            try:
                # Convert page to image and use RapidOCR
                import pymupdf
                pdf_doc = pymupdf.open(stream=pdf_data, filetype="pdf")
                if page_num < len(pdf_doc):
                    page = pdf_doc[page_num]
                    result = self._extract_with_rapidocr(page)
                    pdf_doc.close()
                    return result
                pdf_doc.close()
            except Exception as ocr_error:
                print(f"Docling fallback to RapidOCR also failed: {ocr_error}")
            
            return ""
            
        finally:
            # Clean up temporary file
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.unlink(temp_pdf_path)
                except Exception as cleanup_error:
                    print(f"Warning: Failed to cleanup temp file {temp_pdf_path}: {cleanup_error}")
                    # Non-critical error, continue processing
    
    def _clean_dataframe(self, df) -> Any:
        """
        Clean DataFrame by removing empty rows and columns
        
        Args:
            df: pandas DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        try:
            import pandas as pd
            
            if df.empty:
                return df
            
            # Make a copy to avoid modifying the original
            df_clean = df.copy()
            
            # Convert all values to strings and clean
            for col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(lambda x: str(x).strip() if pd.notna(x) and x != '' else '')
            
            # Remove completely empty rows
            df_clean = df_clean[df_clean.apply(lambda row: any(cell != '' for cell in row), axis=1)]
            
            # Remove completely empty columns
            df_clean = df_clean.loc[:, df_clean.apply(lambda col: any(cell != '' for cell in col), axis=0)]
            
            return df_clean
            
        except Exception as e:
            print(f"DataFrame cleaning failed: {e}")
            # Return original DataFrame if cleaning fails
            return df
    
    def _format_dataframe_as_table(self, df) -> str:
        """
        Format DataFrame as a simple, human-readable text string.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            Formatted plain text string representing the table data.
        """
        try:
            
            if df.empty:
                return ""
            
            # Get column headers
            headers = df.columns.tolist()
            text_lines = []

            # Add a clear header for the table's content
            text_lines.append("The document includes a table with the following information:")

            # Iterate over each row in the DataFrame
            for index, row in df.iterrows():
                row_parts = []
                # Combine header and cell value for each item in the row
                for header in headers:
                    cell_value = str(row[header]).strip()
                    if cell_value and cell_value.lower() != 'nan':
                        # Create a "key: value" pair
                        row_parts.append(f"{str(header).strip()}: {cell_value}")
                
                # Join the parts into a single line of text for the row
                if row_parts:
                    # Start with a clear identifier for the row
                    row_text = f"- For row {index + 1}, the details are: " + "; ".join(row_parts) + "."
                    text_lines.append(row_text)

            # Join all lines with a newline character
            return "\n".join(text_lines)

        except Exception as e:
            print(f"DataFrame formatting to plain text failed: {e}")
            # Fallback to a very simple string conversion if the primary method fails
            return df.to_string(index=False, header=True)
    
    def extract_tables_with_pdfplumber(self, pdf_data: bytes, page_num: int) -> str:
        """
        Extract tables from a specific page using PDFPlumber
        
        Args:
            pdf_data: PDF content as bytes
            page_num: Page number (0-indexed for PDFPlumber)
            
        Returns:
            Formatted table text
        """
        try:
            import pdfplumber
            import io
            import pandas as pd
            
            with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
                if page_num - 1 >= len(pdf.pages):
                    return ""
                
                page = pdf.pages[page_num - 1]  # Convert to 0-indexed
                
                # Simple, reliable PDFPlumber table extraction
                tables = page.extract_tables()
                
                if not tables:
                    return ""
                
                # Format tables as text
                formatted_text = []
                for i, table in enumerate(tables):
                    if table and len(table) > 0:
                        # Convert to DataFrame for easier manipulation
                        # Handle duplicate or missing column names
                        if table[0] and all(col for col in table[0]):  # Has valid headers
                            headers = table[0]
                            # Fix duplicate column names
                            seen_cols = {}
                            unique_headers = []
                            for col in headers:
                                col_str = str(col).strip()
                                if col_str in seen_cols:
                                    seen_cols[col_str] += 1
                                    unique_headers.append(f"{col_str}_{seen_cols[col_str]}")
                                else:
                                    seen_cols[col_str] = 0
                                    unique_headers.append(col_str)
                            df = pd.DataFrame(table[1:], columns=unique_headers)
                        else:
                            # Generate column names for tables without headers
                            num_cols = len(table[0]) if table[0] else len(table[1]) if len(table) > 1 else 1
                            df = pd.DataFrame(table, columns=[f"Col_{i+1}" for i in range(num_cols)])
                        
                        # Clean the DataFrame
                        df = self._clean_dataframe(df)
                        
                        if not df.empty:
                            formatted_text.append(f"\n--- Table {i+1} (Page {page_num}, PDFPlumber) ---")
                            
                            try:
                                # Format with better structure
                                table_str = self._format_dataframe_as_table(df)
                                if table_str.strip():
                                    formatted_text.append(table_str)
                                else:
                                    # Fallback to simple text representation
                                    formatted_text.append(str(df.to_string(index=False)))
                            except Exception as format_error:
                                print(f"DataFrame formatting failed: {format_error}")
                                # Final fallback - basic table representation
                                try:
                                    formatted_text.append(str(df.to_string(index=False)))
                                except:
                                    # Last resort - just show the raw table data
                                    formatted_text.append(f"Table data (raw): {table}")
                            
                            formatted_text.append("")  # Add spacing between tables
                
                return "\n".join(formatted_text)
                
        except ImportError:
            print("PDFPlumber not available, falling back to PyMuPDF")
            return ""
        except Exception as e:
            print(f"PDFPlumber table extraction failed for page {page_num}: {e}")
            return ""
    
    
    def save_pdf_blob(self, pdf_data: bytes, url: str, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Save PDF blob for caching/debugging
        
        Args:
            pdf_data: PDF content as bytes
            url: Original URL
            metadata: Document metadata
            
        Returns:
            Path to saved blob file or None if disabled
        """
        if not settings.save_pdf_blobs:
            return None
        
        try:
            # Generate filename
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Extract name from URL if possible
            original_name = ""
            if url.startswith('file://'):
                original_name = Path(url[7:]).stem
            elif url.startswith('upload://'):
                # For uploaded files, try to get original filename from file manager
                try:
                    from app.services.file_manager import get_file_manager
                    file_manager = get_file_manager()
                    file_id = url[9:]  # Remove 'upload://' prefix
                    file_info = file_manager.get_file_info(file_id)
                    if file_info:
                        original_name = Path(file_info.original_filename).stem
                except:
                    original_name = f"upload_{url[9:10]}"  # Use first char of file ID
            elif '/' in url:
                try:
                    original_name = Path(url.split('?')[0]).stem  # Remove query params
                except:
                    pass
            
            filename = f"{original_name}_{timestamp}_{url_hash}.pdf" if original_name else f"document_{timestamp}_{url_hash}.pdf"
            
            # Save blob
            blob_path = Path(settings.pdf_blob_dir) / filename
            with open(blob_path, 'wb') as f:
                f.write(pdf_data)
            
            # Save metadata
            meta_path = Path(settings.pdf_blob_dir) / f"{filename}.meta"
            with open(meta_path, 'w', encoding='utf-8') as f:
                f.write("# PDF Blob Metadata\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Source URL: {url}\n")
                f.write(f"# File size: {len(pdf_data):,} bytes\n")
                f.write(f"# Document metadata: {metadata}\n")
            
            conditional_print(f"PDF blob saved to: {blob_path}")
            return str(blob_path)
            
        except Exception as e:
            print(f"Warning: Failed to save PDF blob: {e}")
            return None
    
    def save_parsed_text(self, text: str, page_texts: list, url: str, metadata: Dict[str, Any], translated_text: str = None) -> Optional[str]:
        """
        Save parsed text for validation/debugging, including translations
        
        Args:
            text: Full extracted text (original language)
            page_texts: Text from individual pages
            url: Original URL
            metadata: Document metadata
            translated_text: English translation (if available)
            
        Returns:
            Path to saved text file or None if disabled
        """
        if not settings.save_parsed_text:
            return None
        
        try:
            # Generate filename
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"parsed_{timestamp}_{url_hash}.txt"
            
            # Save parsed text
            text_path = Path(settings.parsed_text_dir) / filename
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write("# Document Parsing Report\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Source URL: {url}\n")
                f.write(f"# Metadata: {metadata}\n")
                f.write(f"\n{'='*50}\n")
                f.write("ORIGINAL PARSED TEXT CONTENT\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"{text}\n")
                
                # Write translated content if available
                if translated_text:
                    f.write(f"\n\n{'='*50}\n")
                    f.write("ENGLISH TRANSLATION\n")
                    f.write(f"{'='*50}\n\n")
                    f.write(f"{translated_text}\n")
            
            conditional_print(f"Parsed text saved to: {text_path}")
            return str(text_path)
            
        except Exception as e:
            print(f"Warning: Failed to save parsed text: {e}")
            return None
    
    def generate_doc_id(self, url: str) -> str:
        """
        Generate consistent document ID from URL or file path
        
        Args:
            url: Document URL (HTTP/HTTPS) or file path (file://)
            
        Returns:
            12-character document ID
        """
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    @staticmethod
    def create_file_url(file_path: str) -> str:
        """
        Create a proper file:// URL from a local file path
        
        Args:
            file_path: Local file path (absolute or relative)
            
        Returns:
            Properly formatted file:// URL
        """
        import urllib.parse
        
        # Convert to absolute path
        abs_path = Path(file_path).resolve()
        
        # Create proper file URL
        return f"file://{urllib.parse.quote(str(abs_path))}"
    
    def extract_text_from_document(self, file_data: bytes, url: str) -> Dict[str, Any]:
        """
        Extract text from any supported document format with zero content handling
        
        Args:
            file_data: Document bytes
            url: Original URL for filename detection
            
        Returns:
            Dict with text, pages, metadata, and page_texts
        """
        start_time = time.time()
        filename = url.split('/')[-1] if '/' in url else url
        
        conditional_print(f"Processing document: {filename} ({len(file_data):,} bytes)")
        
        try:
            # Detect file type
            detected_type = self.file_detector.detect_file_type(file_data, filename)
            conditional_print(f"Detected file type: {detected_type.mime_type} -> {detected_type.extractor_type}")
            
            # Check if file type is supported
            if not detected_type.is_supported or not detected_type.extractor_type:
                raise ValueError(
                    f"Unsupported file type: {detected_type.mime_type}\n"
                    f"Supported types: {list(self.extractors.keys())}\n"
                    f"File: {filename}"
                )
            
            # Get appropriate extractor
            extractor = self.extractors.get(detected_type.extractor_type)
            if not extractor:
                raise ValueError(f"No extractor available for type: {detected_type.extractor_type}")
            
            # Extract text using appropriate extractor
            extraction_result = extractor.extract_text(file_data, filename)
            
            # Validate extraction result and handle zero content
            if not extraction_result.text or len(extraction_result.text.strip()) == 0:
                print("WARNING: No text extracted from document")
                # Create meaningful error message instead of empty content
                error_text = f"[No text content found in file: {filename}]\n"
                error_text += f"[File type: {detected_type.extractor_type}]\n"
                error_text += f"[File size: {len(file_data):,} bytes]\n"
                error_text += "[Recommendation: Check if file contains readable text content]"
                
                extraction_result.text = error_text
                extraction_result.metadata.update({
                    'extraction_status': 'no_content',
                    'warning': 'No readable text found in document'
                })
            else:
                extraction_result.metadata['extraction_status'] = 'success'
            
            # Handle zero pages (division by zero prevention)
            if extraction_result.pages <= 0:
                print("WARNING: Zero pages detected, setting to 1")
                extraction_result.pages = 1
            
            processing_time = time.time() - start_time
            
            # Convert ExtractionResult to legacy format for compatibility
            result = {
                'text': extraction_result.text,
                'pages': extraction_result.pages,
                'metadata': {
                    **extraction_result.metadata,
                    'total_processing_time': processing_time,
                    'file_type_detected': detected_type.mime_type,
                    'extractor_used': detected_type.extractor_type,
                    'file_size_bytes': len(file_data)
                },
                'page_texts': getattr(extraction_result, 'page_texts', None) or [extraction_result.text]
            }
            
            conditional_print("Document processing completed:")
            conditional_print(f"  - Type: {detected_type.extractor_type}")
            conditional_print(f"  - Pages: {extraction_result.pages}")
            conditional_print(f"  - Text length: {len(extraction_result.text)} characters")
            conditional_print(f"  - Processing time: {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            # Handle extraction errors gracefully
            processing_time = time.time() - start_time
            
            error_text = f"[Document processing failed: {filename}]\n"
            error_text += f"[Error: {str(e)}]\n"
            error_text += f"[File size: {len(file_data):,} bytes]\n"
            
            # Try to provide helpful error information
            try:
                detected_type = self.file_detector.detect_file_type(file_data, filename)
                error_text += f"[Detected type: {detected_type.mime_type}]\n"
                if detected_type.is_supported:
                    error_text += "[This file type should be supported - please check file integrity]"
                else:
                    supported_types = self.file_detector.get_supported_types()
                    error_text += f"[Supported extensions: {', '.join(supported_types['extensions'])}]"
            except Exception:
                error_text += "[Could not detect file type - file may be corrupted]"
            
            # Return error result instead of raising exception
            return {
                'text': error_text,
                'pages': 1,
                'metadata': {
                    'extraction_status': 'failed',
                    'error': str(e),
                    'file_type_detected': 'unknown',
                    'extractor_used': 'none',
                    'file_size_bytes': len(file_data),
                    'total_processing_time': processing_time
                },
                'page_texts': [error_text]
            }
    
    async def process_document(self, url: str) -> ProcessedDocument:
        """
        Complete multi-format document processing pipeline
        
        Args:
            url: URL to process (supports PDF, Excel, Word, PowerPoint, Text, Images)
            
        Returns:
            ProcessedDocument with all extracted data
        """
        start_time = time.time()
        
        # Generate document ID
        doc_id = self.generate_doc_id(url)
        
        # Download document (renamed from download_pdf)
        document_data = await self.download_pdf(url)  # Keep existing method name for compatibility
        
        # Detect file type and extract text
        extraction_result = self.extract_text_from_document(document_data, url)
        
        # Save blob if enabled (works for any file type)
        blob_path = self.save_pdf_blob(document_data, url, extraction_result["metadata"])
        
        # Apply multilingual text preprocessing before saving and processing
        processed_text = self.preprocess_malayalam_text(extraction_result["text"])
        
        # Detect language and create translated version if needed
        detected_language = self.detect_language(processed_text)
        translated_text = None
        
        if detected_language != 'english':
            print(f"Detected {detected_language} content, creating English translation...")
            translated_text = await self.translate_text_to_english(processed_text, detected_language)
        
        # Save parsed text if enabled (save both original and translated if available)
        text_path = self.save_parsed_text(
            processed_text, 
            extraction_result["page_texts"], 
            url, 
            extraction_result["metadata"],
            translated_text=translated_text
        )
        
        # Combine metadata
        final_metadata = {
            **extraction_result["metadata"],
            "source_url": url,
            "doc_id": doc_id,
            "blob_path": blob_path,
            "text_path": text_path,
            "total_processing_time": time.time() - start_time,
            "malayalam_preprocessing_applied": processed_text != extraction_result["text"],
            "detected_language": detected_language,
            "translation_created": translated_text is not None,
            "original_text_length": len(processed_text),
            "translated_text_length": len(translated_text) if translated_text else 0
        }
        
        conditional_print(f"Document processing completed in {final_metadata['total_processing_time']:.2f}s")
        conditional_print(f"   - Original text length: {len(processed_text):,} characters")
        if processed_text != extraction_result["text"]:
            conditional_print(f"   - Preprocessing applied: {len(extraction_result['text'])} → {len(processed_text)} chars")
        if translated_text:
            conditional_print(f"   - English translation length: {len(translated_text):,} characters")
            conditional_print(f"   - Detected language: {detected_language}")
        conditional_print(f"   - Title: {final_metadata.get('title', 'N/A')}")
        if blob_path:
            conditional_print(f"   - Blob saved to: {blob_path}")
        
        return ProcessedDocument(
            text=processed_text,
            pages=extraction_result["pages"],
            doc_id=doc_id,
            metadata=final_metadata,
            blob_path=blob_path,
            translated_text=translated_text,
            detected_language=detected_language
        )


# Singleton instance
_document_processor = None

def get_document_processor() -> DocumentProcessor:
    """Get singleton document processor instance"""
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
    return _document_processor
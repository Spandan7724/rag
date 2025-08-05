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


@dataclass
class ProcessedDocument:
    """Container for processed document data"""
    text: str
    pages: int
    doc_id: str
    metadata: Dict[str, Any]
    blob_path: Optional[str] = None


class DocumentProcessor:
    """Handles PDF download, text extraction, and blob storage"""
    
    def __init__(self):
        """Initialize document processor"""
        # Ensure all required directories exist
        ensure_directories()
        
        # Create specific directories for this service
        if settings.save_pdf_blobs:
            create_directory(settings.pdf_blob_dir, "PDF blob storage")
        if settings.save_parsed_text:
            create_directory(settings.parsed_text_dir, "parsed text storage")
    
    async def download_pdf(self, url: str) -> bytes:
        """
        Download PDF from URL or read from local/uploaded file with proper error handling
        
        Args:
            url: URL to download PDF from (HTTP/HTTPS), local file path (file://), or upload ID (upload://)
            
        Returns:
            PDF content as bytes
        """
        print(f"Processing document from: {url}")
        print(f"URL type check - startswith('file://'): {url.startswith('file://')}")
        print(f"URL type check - startswith('upload://'): {url.startswith('upload://')}")
        
        # Check if this is a local file URL
        if url.startswith('file://'):
            print(f"Routing to _read_local_file for: {url}")
            return await self._read_local_file(url)
        elif url.startswith('upload://'):
            print(f"Routing to _read_uploaded_file for: {url}")
            return await self._read_uploaded_file(url)
        else:
            print(f"Routing to _download_remote_file for: {url}")
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
        
        print(f"_read_local_file called with: {file_url}")
        
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
                    import fnmatch
                    
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
        print(f"Downloading from remote URL: {url}")
        print(f"WARNING: _download_remote_file called with URL: {url}")
        
        try:
            timeout = aiohttp.ClientTimeout(total=settings.download_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                print(f"Attempting aiohttp GET request to: {url}")
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
                    
                    print(f"Downloaded PDF: {len(content):,} bytes")
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
            
            # Extract text from all pages using hybrid processing
            text_parts = []
            page_texts = []
            processing_stats = {
                'text': 0,
                'pdfplumber': 0,
                'pymupdf_table': 0,
                'ocr': 0
            }
            
            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                
                if settings.enable_hybrid_processing:
                    # Analyze page content to determine processing strategy
                    analysis = self.analyze_page_content(page)
                    processing_method = analysis['processing_method']
                    processing_stats[processing_method] += 1
                    
                    print(f"  Page {page_num + 1}: Using {processing_method} processing "
                          f"(text: {analysis['text_length']} chars, tables: {analysis['table_count']})")
                    
                    # Process page based on analysis
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
                        # Fast PyMuPDF text extraction
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
                
                page_texts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                text_parts.append(page_text)
            
            # Print processing statistics
            if settings.enable_hybrid_processing:
                print(f"Processing statistics: {processing_stats}")
                total_pages = sum(processing_stats.values())
                if total_pages > 0:
                    for method, count in processing_stats.items():
                        if count > 0:
                            print(f"  - {method}: {count} pages ({count/total_pages*100:.1f}%)")
            
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
    
    def _determine_processing_method(self, text_length: int, has_tables: bool) -> str:
        """
        Determine the optimal processing method based on page analysis
        
        Args:
            text_length: Length of text content
            has_tables: Whether page has tables
            
        Returns:
            Processing method string
        """
        if not settings.enable_hybrid_processing:
            return 'text'  # Default to current behavior
        
        if text_length < settings.text_threshold:
            return 'ocr'  # PaddleOCR for image-heavy pages
        elif has_tables:
            if settings.table_extraction_method == "auto":
                # PDFPlumber-first approach for better table accuracy
                return 'pdfplumber'  # Primary choice for all table types
            elif settings.table_extraction_method == "pdfplumber":
                return 'pdfplumber'  # Direct PDFPlumber selection
            else:
                return settings.table_extraction_method  # Use configured method
        else:
            return 'text'  # Fast PyMuPDF text extraction
    
    
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
        Format DataFrame as a well-structured table string
        
        Args:
            df: pandas DataFrame
            
        Returns:
            Formatted table string
        """
        try:
            import pandas as pd
            
            if df.empty:
                return ""
            
            # Convert all cells to strings and clean them
            df_str = df.astype(str)
            
            # Smart column width calculation
            num_cols = len(df_str.columns)
            col_widths = []
            
            for i, col in enumerate(df_str.columns):
                header_text = str(col) if col else f"Col_{i+1}"
                header_width = len(header_text)
                
                # Calculate max content width for this column
                max_content_width = 0
                if len(df_str[col]) > 0:
                    for cell in df_str[col]:
                        cell_lines = str(cell).split('\n')
                        max_line_width = max(len(line) for line in cell_lines) if cell_lines else 0
                        max_content_width = max(max_content_width, max_line_width)
                
                # Set reasonable column widths based on position and content
                if i == 0:  # First column (Sr. No.) - keep narrow
                    col_width = max(header_width, min(max_content_width, 15))
                elif i == 1:  # Second column (Treatment Methods) - medium width
                    col_width = max(header_width, min(max_content_width, 60))
                else:  # Other columns - flexible width
                    col_width = max(header_width, min(max_content_width, 80))
                
                col_widths.append(max(col_width, 8))  # Minimum width of 8
            
            # Format header
            formatted_lines = []
            if num_cols > 1:
                header_cells = []
                for i, col in enumerate(df_str.columns):
                    header_text = str(col) if col else f"Col_{i+1}"
                    # Clean up header text
                    header_text = ' '.join(header_text.split())  # Remove extra whitespace
                    header_cells.append(header_text.ljust(col_widths[i]))
                
                header_line = " | ".join(header_cells)
                formatted_lines.append(header_line)
                formatted_lines.append("-" * len(header_line))
            
            # Format data rows with better text handling
            for _, row in df_str.iterrows():
                # Handle multi-line cells by processing each line
                max_lines = 1
                processed_cells = []
                
                for i, cell in enumerate(row):
                    cell_text = str(cell) if pd.notna(cell) and cell != 'nan' and cell != '' else ""
                    # Clean up cell text
                    cell_text = ' '.join(cell_text.split())  # Remove extra whitespace
                    
                    # Handle long content with intelligent wrapping
                    if len(cell_text) > col_widths[i]:
                        # For long content, wrap intelligently
                        wrapped_lines = []
                        remaining = cell_text
                        while remaining:
                            if len(remaining) <= col_widths[i]:
                                wrapped_lines.append(remaining)
                                break
                            else:
                                # Try to break at word boundary
                                break_point = col_widths[i]
                                space_pos = remaining.rfind(' ', 0, break_point)
                                if space_pos > col_widths[i] * 0.7:  # Good break point found
                                    wrapped_lines.append(remaining[:space_pos])
                                    remaining = remaining[space_pos + 1:]
                                else:
                                    # No good break point, just cut
                                    wrapped_lines.append(remaining[:break_point])
                                    remaining = remaining[break_point:]
                        processed_cells.append(wrapped_lines)
                        max_lines = max(max_lines, len(wrapped_lines))
                    else:
                        processed_cells.append([cell_text])
                
                # Output the row(s)
                for line_idx in range(max_lines):
                    row_cells = []
                    for i, cell_lines in enumerate(processed_cells):
                        if line_idx < len(cell_lines):
                            text = cell_lines[line_idx]
                        else:
                            text = ""
                        row_cells.append(text.ljust(col_widths[i]))
                    
                    row_line = " | ".join(row_cells)
                    if row_line.strip():  # Only add non-empty rows
                        formatted_lines.append(row_line)
            
            return "\n".join(formatted_lines)
            
        except Exception as e:
            print(f"DataFrame formatting failed: {e}")
            # Fallback to simple string conversion
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
                f.write(f"# PDF Blob Metadata\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Source URL: {url}\n")
                f.write(f"# File size: {len(pdf_data):,} bytes\n")
                f.write(f"# Document metadata: {metadata}\n")
            
            print(f"PDF blob saved to: {blob_path}")
            return str(blob_path)
            
        except Exception as e:
            print(f"Warning: Failed to save PDF blob: {e}")
            return None
    
    def save_parsed_text(self, text: str, page_texts: list, url: str, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Save parsed text for validation/debugging
        
        Args:
            text: Full extracted text
            page_texts: Text from individual pages
            url: Original URL
            metadata: Document metadata
            
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
                f.write(f"# Document Parsing Report\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Source URL: {url}\n")
                f.write(f"# Metadata: {metadata}\n")
                f.write(f"\n{'='*50}\n")
                f.write(f"PARSED TEXT CONTENT\n")
                f.write(f"{'='*50}\n\n")
                
                # Write page-by-page content
                for page_text in page_texts:
                    f.write(f"{page_text}\n")
            
            print(f"Parsed text saved to: {text_path}")
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
    
    async def process_document(self, url: str) -> ProcessedDocument:
        """
        Complete document processing pipeline
        
        Args:
            url: URL to process
            
        Returns:
            ProcessedDocument with all extracted data
        """
        start_time = time.time()
        
        # Generate document ID
        doc_id = self.generate_doc_id(url)
        
        # Download PDF
        pdf_data = await self.download_pdf(url)
        
        # Extract text and metadata
        extraction_result = self.extract_text_from_pdf(pdf_data)
        
        # Save blob if enabled
        blob_path = self.save_pdf_blob(pdf_data, url, extraction_result["metadata"])
        
        # Save parsed text if enabled
        text_path = self.save_parsed_text(
            extraction_result["text"], 
            extraction_result["page_texts"], 
            url, 
            extraction_result["metadata"]
        )
        
        # Combine metadata
        final_metadata = {
            **extraction_result["metadata"],
            "source_url": url,
            "doc_id": doc_id,
            "blob_path": blob_path,
            "text_path": text_path,
            "total_processing_time": time.time() - start_time
        }
        
        print(f"Document processing completed in {final_metadata['total_processing_time']:.2f}s")
        print(f"   - Text length: {len(extraction_result['text']):,} characters")
        print(f"   - Title: {final_metadata.get('title', 'N/A')}")
        if blob_path:
            print(f"   - Blob saved to: {blob_path}")
        
        return ProcessedDocument(
            text=extraction_result["text"],
            pages=extraction_result["pages"],
            doc_id=doc_id,
            metadata=final_metadata,
            blob_path=blob_path
        )


# Singleton instance
_document_processor = None

def get_document_processor() -> DocumentProcessor:
    """Get singleton document processor instance"""
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
    return _document_processor
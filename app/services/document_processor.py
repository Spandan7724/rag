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
            
            # Extract text from all pages
            text_parts = []
            page_texts = []
            
            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                page_text = page.get_text()
                
                # OCR fallback for pages with minimal text (likely scanned/images)
                if len(page_text.strip()) < 100:  # Less than 100 chars suggests scanned content
                    print(f"  Page {page_num + 1}: Minimal text detected ({len(page_text)} chars), attempting OCR...")
                    try:
                        import pytesseract
                        from PIL import Image
                        import io
                        
                        # Get page as image
                        pix = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0))  # 2x resolution for better OCR
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        
                        # Extract text using OCR
                        ocr_text = pytesseract.image_to_string(img, config='--psm 6')
                        
                        if len(ocr_text.strip()) > len(page_text.strip()):
                            print(f"  Page {page_num + 1}: OCR extracted {len(ocr_text)} chars (vs {len(page_text)} native)")
                            page_text = ocr_text
                        else:
                            print(f"  Page {page_num + 1}: OCR didn't improve extraction")
                            
                    except ImportError:
                        print(f"  Page {page_num + 1}: OCR libraries not available (install: pip install pytesseract pillow)")
                    except Exception as e:
                        print(f"  Page {page_num + 1}: OCR failed: {e}")
                
                page_texts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                text_parts.append(page_text)
            
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
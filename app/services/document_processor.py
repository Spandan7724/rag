"""
Document processing service for PDF download and text extraction
"""
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
        Download PDF from URL with proper error handling
        
        Args:
            url: URL to download PDF from
            
        Returns:
            PDF content as bytes
        """
        print(f"Downloading document from URL: {url}")
        
        try:
            timeout = aiohttp.ClientTimeout(total=settings.download_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
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
        Generate consistent document ID from URL
        
        Args:
            url: Document URL
            
        Returns:
            12-character document ID
        """
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
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
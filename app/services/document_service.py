"""
Document processing service
"""
import aiohttp
import pymupdf
import tempfile
import os
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from fastapi import HTTPException

from app.models.requests import DocumentContent
from app.core.config import settings

class DocumentService:
    """Service for downloading and processing documents"""
    
    @staticmethod
    async def download_document(url: str) -> bytes:
        """Download document from URL"""
        try:
            timeout = aiohttp.ClientTimeout(total=settings.download_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(str(url)) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Failed to download document: HTTP {response.status}"
                        )
                    
                    content = await response.read()
                    
                    # Check file size
                    if len(content) > settings.max_file_size:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File too large. Maximum size: {settings.max_file_size} bytes"
                        )
                    
                    return content
                    
        except aiohttp.ClientError as e:
            raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected download error: {str(e)}")

    @staticmethod
    def process_pdf(pdf_data: bytes, url: str = "") -> DocumentContent:
        """Extract text and metadata from PDF"""
        temp_file_path = None
        start_time = time.time()
        
        try:
            print(f"Starting PDF processing... (File size: {len(pdf_data):,} bytes)")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                suffix='.pdf', 
                delete=False,
                dir=settings.temp_dir
            ) as temp_file:
                temp_file.write(pdf_data)
                temp_file_path = temp_file.name
            
            # Open and process PDF
            pdf_open_time = time.time()
            doc = pymupdf.open(temp_file_path)
            print(f"PDF opened in {(time.time() - pdf_open_time):.2f}s")
            
            # Extract text from all pages
            text_extraction_start = time.time()
            full_text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():  # Only add non-empty pages
                    full_text += f"\n--- Page {page_num + 1} ---\n{text}"
            
            text_extraction_time = time.time() - text_extraction_start
            print(f"Text extracted from {len(doc)} pages in {text_extraction_time:.2f}s")
            
            # Extract metadata before closing
            page_count = len(doc)
            metadata = {
                "pages": page_count,
                "title": doc.metadata.get("title", "").strip(),
                "author": doc.metadata.get("author", "").strip(),
                "subject": doc.metadata.get("subject", "").strip(),
                "creator": doc.metadata.get("creator", "").strip(),
                "file_size": len(pdf_data),
                "processing_time_seconds": round(time.time() - start_time, 2)
            }
            
            doc.close()
            
            # Optional: Save parsed text for validation
            if settings.save_parsed_text:
                DocumentService._save_parsed_text(full_text, metadata, url)
            
            total_time = time.time() - start_time
            print(f"PDF processing completed in {total_time:.2f}s")
            print(f"   - Text length: {len(full_text):,} characters")
            print(f"   - Title: {metadata.get('title', 'N/A')}")
            
            return DocumentContent(
                text=full_text.strip(),
                pages=page_count,
                metadata=metadata
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"PDF processing failed: {str(e)}"
            )
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass  # File cleanup failed, but don't crash the app

    @staticmethod
    def _save_parsed_text(text: str, metadata: Dict[str, Any], url: str) -> None:
        """Save parsed text to file for validation purposes"""
        try:
            # Create directory if it doesn't exist
            parsed_dir = Path(settings.parsed_text_dir)
            parsed_dir.mkdir(exist_ok=True)
            
            # Generate filename based on URL hash and timestamp
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"parsed_{timestamp}_{url_hash}.txt"
            
            # Save to file
            file_path = parsed_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Document Parsing Report\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Source URL: {url}\n")
                f.write(f"# Metadata: {metadata}\n")
                f.write(f"\n{'='*50}\n")
                f.write(f"PARSED TEXT CONTENT\n")
                f.write(f"{'='*50}\n\n")
                f.write(text)
            
            print(f"Parsed text saved to: {file_path}")
            
        except Exception as e:
            print(f"Failed to save parsed text: {str(e)}")
            # Don't raise - this is optional functionality

    @classmethod
    async def process_document(cls, url: str) -> DocumentContent:
        """Complete document processing pipeline"""
        # Download document
        pdf_data = await cls.download_document(url)
        
        # Process PDF with URL for optional saving
        document_content = cls.process_pdf(pdf_data, url)
        
        return document_content
"""
Document processing service
"""
import aiohttp
import pymupdf
import tempfile
import os
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
    def process_pdf(pdf_data: bytes) -> DocumentContent:
        """Extract text and metadata from PDF"""
        temp_file_path = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                suffix='.pdf', 
                delete=False,
                dir=settings.temp_dir
            ) as temp_file:
                temp_file.write(pdf_data)
                temp_file_path = temp_file.name
            
            # Open and process PDF
            doc = pymupdf.open(temp_file_path)
            
            # Extract text from all pages
            full_text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():  # Only add non-empty pages
                    full_text += f"\n--- Page {page_num + 1} ---\n{text}"
            
            # Extract metadata before closing
            page_count = len(doc)
            metadata = {
                "pages": page_count,
                "title": doc.metadata.get("title", "").strip(),
                "author": doc.metadata.get("author", "").strip(),
                "subject": doc.metadata.get("subject", "").strip(),
                "creator": doc.metadata.get("creator", "").strip(),
                "file_size": len(pdf_data)
            }
            
            doc.close()
            
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

    @classmethod
    async def process_document(cls, url: str) -> DocumentContent:
        """Complete document processing pipeline"""
        # Download document
        pdf_data = await cls.download_document(url)
        
        # Process PDF
        document_content = cls.process_pdf(pdf_data)
        
        return document_content
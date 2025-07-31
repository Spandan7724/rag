"""
Document storage service for persistent artifact management
"""
import os
import json
import uuid
import hashlib
import aiofiles
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from app.core.config import settings


@dataclass
class DocumentMetadata:
    """Document metadata for storage"""
    document_id: str
    original_filename: str
    file_size: int
    content_hash: str
    upload_timestamp: datetime
    processing_status: str
    source_type: str  # "url", "upload"
    source_reference: str  # URL or original filename
    processing_method: str
    pages: Optional[int] = None
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None


class DocumentStorageService:
    """Service for persistent document artifact storage"""
    
    def __init__(self, storage_root: str = None):
        self.storage_root = Path(storage_root or getattr(settings, 'document_storage_root', './documents'))
        self.storage_root.mkdir(parents=True, exist_ok=True)
    
    def generate_document_id(self, content: bytes = None, source: str = "") -> str:
        """Generate unique document ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if content:
            content_hash = hashlib.sha256(content).hexdigest()[:8]
            return f"doc_{timestamp}_{content_hash}"
        else:
            unique_id = str(uuid.uuid4())[:8]
            return f"doc_{timestamp}_{unique_id}"
    
    def get_document_path(self, document_id: str) -> Path:
        """Get document storage directory path"""
        return self.storage_root / document_id
    
    async def store_raw_document(self, document_id: str, content: bytes, 
                               filename: str, source_type: str, source_reference: str) -> DocumentMetadata:
        """Store raw PDF document with metadata"""
        doc_path = self.get_document_path(document_id)
        doc_path.mkdir(parents=True, exist_ok=True)
        
        # Store raw PDF
        raw_pdf_path = doc_path / "raw.pdf"
        async with aiofiles.open(raw_pdf_path, 'wb') as f:
            await f.write(content)
        
        # Calculate content hash
        content_hash = hashlib.sha256(content).hexdigest()
        
        # Create metadata
        metadata = DocumentMetadata(
            document_id=document_id,
            original_filename=filename,
            file_size=len(content),
            content_hash=content_hash,
            upload_timestamp=datetime.now(),
            processing_status="uploaded",
            source_type=source_type,
            source_reference=source_reference,
            processing_method="pending"
        )
        
        # Store metadata
        await self.store_metadata(document_id, metadata)
        
        # Create processing log
        await self.log_processing_event(document_id, "document_stored", {
            "filename": filename,
            "file_size": len(content),
            "source_type": source_type
        })
        
        return metadata
    
    
    async def store_markdown(self, document_id: str, markdown: str, 
                           span_mapping: Optional[Dict] = None) -> None:
        """Store PyMuPDF4LLM markdown output"""
        from app.core.config import settings
        
        doc_path = self.get_document_path(document_id)
        
        # Store markdown
        markdown_path = doc_path / "extracted.md"
        async with aiofiles.open(markdown_path, 'w', encoding='utf-8') as f:
            await f.write(markdown)
        
        # Store span mapping if provided and feature flag is enabled
        if span_mapping and settings.store_span_map:
            span_path = doc_path / "span_map.json"
            async with aiofiles.open(span_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(span_mapping, indent=2))
    
    async def store_ocr_results(self, document_id: str, ocr_pages: List[int], 
                              ocr_data: Dict[int, str]) -> None:
        """Store OCR results for specific pages"""
        doc_path = self.get_document_path(document_id)
        ocr_path = doc_path / "ocr_pages.json"
        
        ocr_results = {
            "ocr_pages": ocr_pages,
            "ocr_data": ocr_data,
            "timestamp": datetime.now().isoformat()
        }
        
        async with aiofiles.open(ocr_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(ocr_results, indent=2))
    
    async def store_metadata(self, document_id: str, metadata: DocumentMetadata) -> None:
        """Store document metadata"""
        doc_path = self.get_document_path(document_id)
        metadata_path = doc_path / "metadata.json"
        
        # Convert dataclass to dict with datetime serialization
        metadata_dict = asdict(metadata)
        if isinstance(metadata_dict.get('upload_timestamp'), datetime):
            metadata_dict['upload_timestamp'] = metadata_dict['upload_timestamp'].isoformat()
        
        async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(metadata_dict, indent=2))
    
    async def load_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        """Load document metadata"""
        doc_path = self.get_document_path(document_id)
        metadata_path = doc_path / "metadata.json"
        
        if not metadata_path.exists():
            return None
        
        async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            metadata_dict = json.loads(content)
        
        # Convert timestamp back to datetime
        if 'upload_timestamp' in metadata_dict:
            metadata_dict['upload_timestamp'] = datetime.fromisoformat(metadata_dict['upload_timestamp'])
        
        return DocumentMetadata(**metadata_dict)
    
    async def log_processing_event(self, document_id: str, event: str, data: Dict[str, Any]) -> None:
        """Log processing events for audit trail"""
        doc_path = self.get_document_path(document_id)
        log_path = doc_path / "processing.log"
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "data": data
        }
        
        # Append to log file
        async with aiofiles.open(log_path, 'a', encoding='utf-8') as f:
            await f.write(json.dumps(log_entry) + '\n')
    
    async def update_processing_status(self, document_id: str, status: str, 
                                     processing_time_ms: Optional[int] = None,
                                     error_message: Optional[str] = None) -> None:
        """Update document processing status"""
        metadata = await self.load_metadata(document_id)
        if metadata:
            metadata.processing_status = status
            if processing_time_ms:
                metadata.processing_time_ms = processing_time_ms
            if error_message:
                metadata.error_message = error_message
            
            await self.store_metadata(document_id, metadata)
            
            # Log status change
            await self.log_processing_event(document_id, f"status_changed_{status}", {
                "status": status,
                "processing_time_ms": processing_time_ms,
                "error_message": error_message
            })
    
    async def get_document_artifacts(self, document_id: str) -> Dict[str, str]:
        """Get available artifacts for a document"""
        doc_path = self.get_document_path(document_id)
        artifacts = {}
        
        # Check for various artifacts
        artifact_files = {
            "raw_pdf": "raw.pdf",
            "markdown": "extracted.md",  # Enhanced processing with PyMuPDF4LLM
            "span_mapping": "span_map.json",
            "ocr_results": "ocr_pages.json",
            "metadata": "metadata.json",
            "processing_log": "processing.log"
        }
        
        for artifact_type, filename in artifact_files.items():
            artifact_path = doc_path / filename
            if artifact_path.exists():
                artifacts[artifact_type] = str(artifact_path)
        
        return artifacts
    
    def document_exists(self, document_id: str) -> bool:
        """Check if document exists in storage"""
        doc_path = self.get_document_path(document_id)
        return doc_path.exists() and (doc_path / "raw.pdf").exists()
    
    async def get_text_content(self, document_id: str) -> Optional[str]:
        """Get text content from markdown (enhanced processing)"""
        doc_path = self.get_document_path(document_id)
        
        # Get text from markdown (enhanced processing)
        markdown_path = doc_path / "extracted.md"
        if markdown_path.exists():
            async with aiofiles.open(markdown_path, 'r', encoding='utf-8') as f:
                markdown_content = await f.read()
            return self._markdown_to_text(markdown_content)
        
        return None
    
    async def get_markdown_content(self, document_id: str) -> Optional[str]:
        """Get markdown content (only available for enhanced processing)"""
        doc_path = self.get_document_path(document_id)
        markdown_path = doc_path / "extracted.md"
        
        if not markdown_path.exists():
            return None
        
        async with aiofiles.open(markdown_path, 'r', encoding='utf-8') as f:
            return await f.read()
    
    def _markdown_to_text(self, markdown: str) -> str:
        """Convert markdown to plain text"""
        import re
        
        # Remove markdown formatting
        text = re.sub(r'#{1,6}\s+', '', markdown)  # Headers
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)  # Code
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Links
        
        return text

    async def cleanup_document(self, document_id: str) -> bool:
        """Remove all artifacts for a document"""
        doc_path = self.get_document_path(document_id)
        if doc_path.exists():
            import shutil
            shutil.rmtree(doc_path)
            return True
        return False
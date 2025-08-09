"""
File type detection for document processing
"""
import mimetypes
import magic
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from app.utils.debug import conditional_print


@dataclass
class DetectedFileType:
    """Information about detected file type"""
    mime_type: str
    extension: str
    filename: str
    size_bytes: int
    is_supported: bool = False
    extractor_type: Optional[str] = None


class FileTypeDetector:
    """Detects file types using MIME detection and extensions"""
    
    # Supported file type mappings
    SUPPORTED_TYPES = {
        # PDF
        'application/pdf': 'pdf',
        
        # Excel
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'excel',  # .xlsx
        'application/vnd.ms-excel': 'excel',  # .xls
        'application/excel': 'excel',
        'application/x-excel': 'excel',
        
        # Word
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'word',  # .docx
        'application/msword': 'word',  # .doc
        'application/vnd.ms-word': 'word',
        
        # PowerPoint
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'powerpoint',  # .pptx
        'application/vnd.ms-powerpoint': 'powerpoint',  # .ppt
        
        # Text files
        'text/plain': 'text',
        'text/csv': 'text',
        'application/csv': 'text',
        'text/tab-separated-values': 'text',
        
        # Images
        'image/png': 'image',
        'image/jpeg': 'image', 
        'image/jpg': 'image',
        'image/gif': 'image',
        'image/bmp': 'image',
        'image/tiff': 'image',
        'image/webp': 'image',
    }
    
    EXTENSION_MAPPING = {
        '.pdf': 'pdf',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.docx': 'word', 
        '.doc': 'word',
        '.pptx': 'powerpoint',
        '.ppt': 'powerpoint',
        '.txt': 'text',
        '.csv': 'text',
        '.tsv': 'text',
        '.png': 'image',
        '.jpg': 'image',
        '.jpeg': 'image',
        '.gif': 'image',
        '.bmp': 'image',
        '.tiff': 'image',
        '.tif': 'image',
        '.webp': 'image',
    }
    
    def __init__(self):
        """Initialize file type detector"""
        # Initialize python-magic if available
        self.magic_detector = None
        try:
            self.magic_detector = magic.Magic(mime=True)
            conditional_print("File type detector initialized with python-magic")
        except Exception as e:
            conditional_print(f"Warning: python-magic not available, falling back to mimetypes: {e}")
    
    def detect_file_type(self, file_data: bytes, filename: str = "") -> DetectedFileType:
        """
        Detect file type from data and filename
        
        Args:
            file_data: Raw file bytes
            filename: Original filename
            
        Returns:
            DetectedFileType with all detection information
        """
        # Extract extension from filename
        path = Path(filename) if filename else Path("unknown")
        extension = path.suffix.lower()
        
        # Detect MIME type
        mime_type = self._detect_mime_type(file_data, filename)
        
        # Determine extractor type
        extractor_type = self._determine_extractor_type(mime_type, extension)
        is_supported = extractor_type is not None
        
        return DetectedFileType(
            mime_type=mime_type,
            extension=extension,
            filename=filename,
            size_bytes=len(file_data),
            is_supported=is_supported,
            extractor_type=extractor_type
        )
    
    def _detect_mime_type(self, file_data: bytes, filename: str = "") -> str:
        """Detect MIME type using available methods"""
        
        # Try python-magic first (most reliable)
        if self.magic_detector:
            try:
                mime_type = self.magic_detector.from_buffer(file_data)
                if mime_type and mime_type != 'application/octet-stream':
                    return mime_type
            except Exception as e:
                print(f"Warning: python-magic detection failed: {e}")
        
        # Fallback to mimetypes based on filename
        if filename:
            mime_type, _ = mimetypes.guess_type(filename)
            if mime_type:
                return mime_type
        
        # Check for specific file signatures
        mime_type = self._detect_by_signature(file_data)
        if mime_type:
            return mime_type
            
        # Default fallback
        return 'application/octet-stream'
    
    def _detect_by_signature(self, file_data: bytes) -> Optional[str]:
        """Detect file type by examining file signatures"""
        if len(file_data) < 8:
            return None
        
        # Common file signatures
        signatures = {
            b'%PDF': 'application/pdf',
            b'PK\x03\x04': 'application/zip',  # Could be Office docs
            b'\x89PNG': 'image/png',
            b'\xff\xd8\xff': 'image/jpeg',
            b'GIF87a': 'image/gif',
            b'GIF89a': 'image/gif',
            b'BM': 'image/bmp',
        }
        
        for signature, mime_type in signatures.items():
            if file_data.startswith(signature):
                # Special handling for Office documents (ZIP-based)
                if mime_type == 'application/zip':
                    return self._detect_office_document(file_data)
                return mime_type
        
        return None
    
    def _detect_office_document(self, file_data: bytes) -> str:
        """Detect specific Office document type from ZIP data"""
        try:
            import zipfile
            import io
            
            with zipfile.ZipFile(io.BytesIO(file_data), 'r') as zf:
                filenames = zf.namelist()
                
                # Excel
                if any('xl/' in f for f in filenames):
                    return 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                
                # Word
                if any('word/' in f for f in filenames):
                    return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                
                # PowerPoint
                if any('ppt/' in f for f in filenames):
                    return 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
                    
        except Exception:
            pass
        
        return 'application/zip'
    
    def _determine_extractor_type(self, mime_type: str, extension: str) -> Optional[str]:
        """Determine which extractor should handle this file type"""
        
        # Check MIME type first
        extractor_type = self.SUPPORTED_TYPES.get(mime_type.lower())
        if extractor_type:
            return extractor_type
        
        # Check extension as fallback
        return self.EXTENSION_MAPPING.get(extension.lower())
    
    def get_supported_types(self) -> dict:
        """Get all supported file types and their extractors"""
        return {
            'mime_types': list(self.SUPPORTED_TYPES.keys()),
            'extensions': list(self.EXTENSION_MAPPING.keys()),
            'extractors': list(set(self.SUPPORTED_TYPES.values()))
        }
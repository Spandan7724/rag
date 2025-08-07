"""
Base extractor interface for document processing
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class ExtractionResult:
    """Result of document extraction"""
    text: str
    pages: int
    metadata: Dict[str, Any]
    page_texts: Optional[List[str]] = None
    processing_time: Optional[float] = None
    
    def __post_init__(self):
        """Validate extraction result"""
        if not isinstance(self.text, str):
            raise ValueError("Extracted text must be a string")
        if self.pages < 0:
            raise ValueError("Number of pages cannot be negative")
        if not isinstance(self.metadata, dict):
            raise ValueError("Metadata must be a dictionary")


class BaseExtractor(ABC):
    """Base class for document extractors"""
    
    @property
    @abstractmethod
    def supported_mime_types(self) -> List[str]:
        """Return list of supported MIME types"""
        pass
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions"""
        pass
    
    @abstractmethod
    def extract_text(self, file_data: bytes, filename: str = "") -> ExtractionResult:
        """
        Extract text from file data
        
        Args:
            file_data: Raw file bytes
            filename: Original filename (for extension detection)
            
        Returns:
            ExtractionResult containing text and metadata
            
        Raises:
            ValueError: If file format is not supported
            Exception: If extraction fails
        """
        pass
    
    def can_extract(self, mime_type: str, extension: str) -> bool:
        """
        Check if this extractor can handle the given file type
        
        Args:
            mime_type: MIME type of the file
            extension: File extension (without dot)
            
        Returns:
            True if this extractor can handle the file
        """
        return (mime_type.lower() in [mt.lower() for mt in self.supported_mime_types] or
                extension.lower() in [ext.lower() for ext in self.supported_extensions])
    
    def validate_file_size(self, file_data: bytes, max_size_mb: int = 100) -> None:
        """
        Validate file size
        
        Args:
            file_data: File data to validate
            max_size_mb: Maximum allowed file size in MB
            
        Raises:
            ValueError: If file is too large
        """
        size_mb = len(file_data) / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValueError(f"File size ({size_mb:.1f} MB) exceeds maximum allowed size ({max_size_mb} MB)")
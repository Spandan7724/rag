"""
File Manager Service for handling uploaded files
"""
import uuid
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from fastapi import UploadFile, HTTPException

from app.core.config import settings
from app.core.directories import create_directory


@dataclass
class UploadedFileInfo:
    """Information about an uploaded file"""
    file_id: str
    original_filename: str
    file_path: str
    content_type: str
    file_size: int
    uploaded_at: datetime
    expires_at: datetime


class FileManager:
    """Manages uploaded files with automatic cleanup"""
    
    def __init__(self):
        """Initialize file manager"""
        # Create upload directory
        self.upload_dir = create_directory(settings.upload_dir, "file uploads")
        
        # Track uploaded files
        self.uploaded_files: Dict[str, UploadedFileInfo] = {}
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
        
        print(f"File Manager initialized at: {self.upload_dir}")
    
    def _start_cleanup_task(self):
        """Start the cleanup background task"""
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._cleanup_task = loop.create_task(self._cleanup_expired_files())
        except RuntimeError:
            # No event loop running, cleanup will be called manually
            print("No event loop available for automatic cleanup - will cleanup on demand")
    
    async def _cleanup_expired_files(self):
        """Background task to cleanup expired files"""
        while True:
            try:
                await asyncio.sleep(settings.upload_cleanup_interval)
                self.cleanup_expired_files()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Cleanup task error: {e}")
    
    def cleanup_expired_files(self):
        """Remove expired uploaded files"""
        current_time = datetime.now()
        expired_files = []
        
        for file_id, file_info in self.uploaded_files.items():
            if current_time > file_info.expires_at:
                expired_files.append(file_id)
        
        for file_id in expired_files:
            try:
                self.remove_file(file_id)
                print(f"Cleaned up expired file: {file_id}")
            except Exception as e:
                print(f"Failed to cleanup file {file_id}: {e}")
        
        if expired_files:
            print(f"Cleaned up {len(expired_files)} expired files")
    
    def _generate_file_id(self) -> str:
        """Generate a unique file ID"""
        return str(uuid.uuid4())
    
    def _validate_upload_file(self, upload_file: UploadFile) -> None:
        """Validate uploaded file"""
        # Check file size
        if upload_file.size and upload_file.size > settings.max_upload_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {upload_file.size:,} bytes (max: {settings.max_upload_size:,})"
            )
        
        # Check content type (if restrictions are configured)
        if settings.allowed_upload_types is not None:
            if upload_file.content_type not in settings.allowed_upload_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {upload_file.content_type}. Allowed: {settings.allowed_upload_types}"
                )
        # If allowed_upload_types is None, we rely on the file type detector to validate supported types
        
        # Check filename extension against supported types
        if upload_file.filename:
            file_ext = Path(upload_file.filename).suffix.lower()
            # Get supported extensions from file type detector
            from app.services.extractors import FileTypeDetector
            detector = FileTypeDetector()
            supported_extensions = list(detector.EXTENSION_MAPPING.keys())
            
            if file_ext not in supported_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file extension: {file_ext}. Supported: {supported_extensions}"
                )
    
    async def save_uploaded_file(self, upload_file: UploadFile) -> UploadedFileInfo:
        """
        Save an uploaded file and return file info
        
        Args:
            upload_file: FastAPI UploadFile object
            
        Returns:
            UploadedFileInfo with file details
        """
        # Validate file
        self._validate_upload_file(upload_file)
        
        # Generate unique file ID
        file_id = self._generate_file_id()
        
        # Create filename with timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file_id}_{upload_file.filename}"
        file_path = self.upload_dir / safe_filename
        
        # Calculate expiry time
        uploaded_at = datetime.now()
        expires_at = uploaded_at + timedelta(hours=settings.upload_retention_hours)
        
        try:
            # Save file to disk
            with open(file_path, "wb") as buffer:
                content = await upload_file.read()
                buffer.write(content)
            
            # Get actual file size
            file_size = len(content)
            
            # Create file info
            file_info = UploadedFileInfo(
                file_id=file_id,
                original_filename=upload_file.filename or "unknown.pdf",
                file_path=str(file_path),
                content_type=upload_file.content_type or "application/pdf",
                file_size=file_size,
                uploaded_at=uploaded_at,
                expires_at=expires_at
            )
            
            # Track the file
            self.uploaded_files[file_id] = file_info
            
            print(f"Saved uploaded file: {upload_file.filename} -> {file_id}")
            print(f"  - Size: {file_size:,} bytes")
            print(f"  - Expires: {expires_at}")
            
            return file_info
            
        except Exception as e:
            # Cleanup file if something went wrong
            if file_path.exists():
                try:
                    file_path.unlink()
                except:
                    pass
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save uploaded file: {str(e)}"
            )
    
    def get_file_info(self, file_id: str) -> Optional[UploadedFileInfo]:
        """Get information about an uploaded file"""
        return self.uploaded_files.get(file_id)
    
    def get_file_path(self, file_id: str) -> Optional[str]:
        """Get the file path for a file ID"""
        file_info = self.uploaded_files.get(file_id)
        if file_info:
            # Check if file still exists and hasn't expired
            if Path(file_info.file_path).exists() and datetime.now() <= file_info.expires_at:
                return file_info.file_path
            else:
                # File expired or deleted, remove from tracking
                self.uploaded_files.pop(file_id, None)
        return None
    
    def create_file_url(self, file_id: str) -> Optional[str]:
        """Create a file:// URL for an uploaded file"""
        file_path = self.get_file_path(file_id)
        if file_path:
            return f"file://{file_path}"
        return None
    
    def remove_file(self, file_id: str) -> bool:
        """Remove an uploaded file"""
        file_info = self.uploaded_files.pop(file_id, None)
        if file_info:
            try:
                file_path = Path(file_info.file_path)
                if file_path.exists():
                    file_path.unlink()
                print(f"Removed uploaded file: {file_id}")
                return True
            except Exception as e:
                print(f"Failed to remove file {file_id}: {e}")
        return False
    
    def list_files(self) -> List[Dict[str, Any]]:
        """List all uploaded files"""
        files = []
        current_time = datetime.now()
        
        for file_id, file_info in self.uploaded_files.items():
            files.append({
                "file_id": file_id,
                "original_filename": file_info.original_filename,
                "file_size": file_info.file_size,
                "uploaded_at": file_info.uploaded_at.isoformat(),
                "expires_at": file_info.expires_at.isoformat(),
                "expired": current_time > file_info.expires_at,
                "exists": Path(file_info.file_path).exists()
            })
        
        return files
    
    def get_stats(self) -> Dict[str, Any]:
        """Get file manager statistics"""
        current_time = datetime.now()
        total_files = len(self.uploaded_files)
        active_files = sum(1 for info in self.uploaded_files.values() 
                          if current_time <= info.expires_at and Path(info.file_path).exists())
        expired_files = total_files - active_files
        
        total_size = sum(info.file_size for info in self.uploaded_files.values() 
                        if Path(info.file_path).exists())
        
        return {
            "upload_directory": str(self.upload_dir),
            "total_files": total_files,
            "active_files": active_files,
            "expired_files": expired_files,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / 1024 / 1024,
            "retention_hours": settings.upload_retention_hours,
            "max_upload_size_mb": settings.max_upload_size / 1024 / 1024,
            "allowed_types": settings.allowed_upload_types
        }
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()


# Singleton instance
_file_manager = None

def get_file_manager() -> FileManager:
    """Get singleton file manager instance"""
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManager()
    return _file_manager
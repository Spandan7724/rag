"""
Directory management utilities for the RAG application
"""
import os
from pathlib import Path
from typing import Optional
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class DirectoryManager:
    """Manages creation and validation of required directories"""
    
    def __init__(self):
        """Initialize directory manager"""
        self.required_directories = [
            settings.pdf_blob_dir,
            settings.parsed_text_dir,
            settings.upload_dir,  # File upload storage
            "vector_store",  # Vector database storage
            settings.question_log_dir,  # Question logging storage
        ]
    
    def ensure_directories_exist(self) -> None:
        """
        Create all required directories if they don't exist
        
        This method creates directories with proper permissions and
        handles nested directory creation automatically.
        """
        created_dirs = []
        failed_dirs = []
        
        for dir_path in self.required_directories:
            try:
                path = Path(dir_path)
                
                # Create directory with parents if needed
                path.mkdir(parents=True, exist_ok=True)
                
                # Verify directory is writable
                if not os.access(path, os.W_OK):
                    logger.warning(f"Directory {dir_path} exists but is not writable")
                else:
                    created_dirs.append(str(path.absolute()))
                    
            except Exception as e:
                logger.error(f"Failed to create directory {dir_path}: {str(e)}")
                failed_dirs.append(dir_path)
        
        # Log results
        if created_dirs:
            logger.info(f"Directory setup completed. Ensured {len(created_dirs)} directories exist:")
            for dir_path in created_dirs:
                logger.info(f"  ✓ {dir_path}")
        
        if failed_dirs:
            logger.error(f"Failed to create {len(failed_dirs)} directories:")
            for dir_path in failed_dirs:
                logger.error(f"  ✗ {dir_path}")
            raise RuntimeError(f"Failed to create required directories: {failed_dirs}")
    
    def create_directory(self, path: str, description: Optional[str] = None) -> Path:
        """
        Create a single directory with error handling
        
        Args:
            path: Directory path to create
            description: Optional description for logging
            
        Returns:
            Path object of created directory
            
        Raises:
            RuntimeError: If directory creation fails
        """
        try:
            dir_path = Path(path)
            dir_path.mkdir(parents=True, exist_ok=True)
            
            if description:
                logger.info(f"Created directory for {description}: {dir_path.absolute()}")
            
            return dir_path
            
        except Exception as e:
            error_msg = f"Failed to create directory {path}"
            if description:
                error_msg += f" for {description}"
            error_msg += f": {str(e)}"
            
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def validate_directories(self) -> bool:
        """
        Validate that all required directories exist and are accessible
        
        Returns:
            True if all directories are valid, False otherwise
        """
        all_valid = True
        
        for dir_path in self.required_directories:
            path = Path(dir_path)
            
            if not path.exists():
                logger.error(f"Required directory does not exist: {dir_path}")
                all_valid = False
            elif not path.is_dir():
                logger.error(f"Path exists but is not a directory: {dir_path}")
                all_valid = False
            elif not os.access(path, os.R_OK | os.W_OK):
                logger.error(f"Directory exists but is not readable/writable: {dir_path}")
                all_valid = False
            else:
                logger.debug(f"Directory validated: {dir_path}")
        
        return all_valid
    
    def get_directory_info(self) -> dict:
        """
        Get information about all managed directories
        
        Returns:
            Dictionary with directory information
        """
        info = {
            "directories": {},
            "total_size_mb": 0,
            "status": "healthy"
        }
        
        for dir_path in self.required_directories:
            path = Path(dir_path)
            
            dir_info = {
                "path": str(path.absolute()),
                "exists": path.exists(),
                "is_directory": path.is_dir() if path.exists() else False,
                "readable": os.access(path, os.R_OK) if path.exists() else False,
                "writable": os.access(path, os.W_OK) if path.exists() else False,
                "size_mb": 0,
                "file_count": 0
            }
            
            if path.exists() and path.is_dir():
                try:
                    # Calculate directory size and file count
                    total_size = 0
                    file_count = 0
                    
                    for item in path.rglob("*"):
                        if item.is_file():
                            total_size += item.stat().st_size
                            file_count += 1
                    
                    dir_info["size_mb"] = round(total_size / (1024 * 1024), 2)
                    dir_info["file_count"] = file_count
                    info["total_size_mb"] += dir_info["size_mb"]
                    
                except Exception as e:
                    logger.warning(f"Could not calculate size for {dir_path}: {str(e)}")
            
            info["directories"][dir_path] = dir_info
            
            # Update overall status
            if not (dir_info["exists"] and dir_info["is_directory"] and 
                   dir_info["readable"] and dir_info["writable"]):
                info["status"] = "degraded"
        
        info["total_size_mb"] = round(info["total_size_mb"], 2)
        return info


# Singleton instance
_directory_manager = None

def get_directory_manager() -> DirectoryManager:
    """Get singleton directory manager instance"""
    global _directory_manager
    if _directory_manager is None:
        _directory_manager = DirectoryManager()
    return _directory_manager


def ensure_directories() -> None:
    """Convenience function to ensure all directories exist"""
    directory_manager = get_directory_manager()
    directory_manager.ensure_directories_exist()


def create_directory(path: str, description: Optional[str] = None) -> Path:
    """Convenience function to create a single directory"""
    directory_manager = get_directory_manager()
    return directory_manager.create_directory(path, description)
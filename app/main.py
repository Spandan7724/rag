"""
Main FastAPI application
"""
import logging
import sys
import os
from contextlib import contextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.core.directories import ensure_directories

@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def configure_logging():
    """Configure logging levels based on debug_mode"""
    if not settings.debug_mode:
        # Suppress debug and info logs when debug_mode is False
        logging.getLogger().setLevel(logging.WARNING)
        
        # Specifically suppress uvicorn logs
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.ERROR)
        
        # Suppress FastAPI logs
        logging.getLogger("fastapi").setLevel(logging.WARNING)
        
        # Suppress other common loggers
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        
        # Suppress ALL stdout (print statements) globally when not in debug mode
        sys.stdout = open(os.devnull, "w")
    else:
        # Enable detailed logging for debug mode
        logging.getLogger().setLevel(logging.DEBUG)

def create_application() -> FastAPI:
    """Create and configure FastAPI application"""
    
    # Configure logging based on debug_mode
    configure_logging()
    
    # Ensure all required directories exist before starting the application
    if settings.debug_mode:
        print("Setting up application directories...")
    ensure_directories()
    if settings.debug_mode:
        print("All required directories are ready")
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        debug=settings.debug,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(router)
    
    return app

# Create the application instance
app = create_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug_mode,  # Use debug_mode instead of debug
        log_level="warning" if not settings.debug_mode else "info",  # Suppress logs in production
        access_log=settings.debug_mode  # Disable access logs in production
    )
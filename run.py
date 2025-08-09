"""
Application entry point
"""
import uvicorn
import signal
import sys
import warnings
from app.core.config import settings

def signal_handler(signum, frame):
    """Clean signal handler to suppress multiprocessing warnings"""
    # Suppress the resource tracker warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")
        sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug_mode,  # Only reload in debug mode
        log_level="warning" if not settings.debug_mode else "info",  # Suppress logs in production
        access_log=settings.debug_mode  # Disable access logs in production
    )
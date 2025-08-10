"""
Application entry point
"""
import uvicorn
import signal
import sys
import warnings

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

    # Run similar to: uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 3
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=3,
    )
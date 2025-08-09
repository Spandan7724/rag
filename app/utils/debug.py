"""
Debug utilities for conditional logging based on configuration
"""
from app.core.config import settings

def debug_print(*args, **kwargs):
    """Conditional debug printing based on config"""
    if settings.debug_mode:
        print(*args, **kwargs)

def info_print(*args, **kwargs):
    """Print important info only in debug mode (errors, results, timing)"""
    if settings.debug_mode:
        print(*args, **kwargs)

def conditional_print(*args, **kwargs):
    """Print only when debug_mode is True - use this instead of direct print() calls"""
    if settings.debug_mode:
        print(*args, **kwargs)
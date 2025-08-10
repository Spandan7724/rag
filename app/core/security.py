"""
Security and authentication utilities
"""
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import settings

# Security scheme
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Verify bearer token authentication"""
    if credentials.credentials != settings.bearer_token:
        raise HTTPException(
            status_code=401, 
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return credentials.credentials
"""
Production Security Configuration for NIS Protocol v4.0
Implements JWT auth, rate limiting, input validation
"""

from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
from jose import JWTError, jwt
from typing import Optional, Dict, Any
import hashlib
import time
from collections import defaultdict
import re

# Security constants
SECRET_KEY = "CHANGE_THIS_IN_PRODUCTION"  # TODO: Load from environment
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Rate limiting
rate_limit_store = defaultdict(list)
RATE_LIMIT = 100  # requests per minute
RATE_WINDOW = 60  # seconds

security = HTTPBearer()


# JWT Token Management
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=403, detail="Invalid authentication token")
        return payload
    except JWTError:
        raise HTTPException(status_code=403, detail="Invalid authentication token")


# Rate Limiting
def check_rate_limit(client_id: str) -> bool:
    """
    Check if client has exceeded rate limit
    Returns True if allowed, False if rate limited
    """
    now = time.time()
    
    # Clean old requests
    rate_limit_store[client_id] = [
        req_time for req_time in rate_limit_store[client_id]
        if now - req_time < RATE_WINDOW
    ]
    
    # Check limit
    if len(rate_limit_store[client_id]) >= RATE_LIMIT:
        return False
    
    # Add current request
    rate_limit_store[client_id].append(now)
    return True


def rate_limit_dependency(request):
    """
    FastAPI dependency for rate limiting
    Usage: @app.post("/endpoint", dependencies=[Depends(rate_limit_dependency)])
    """
    client_id = request.client.host
    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Maximum 100 requests per minute."
        )


# Input Validation
def sanitize_input(text: str) -> str:
    """
    Sanitize text input to prevent injection attacks
    """
    # Remove potentially dangerous characters
    text = re.sub(r'[<>\"\'%;()&+]', '', text)
    # Limit length
    return text[:10000]


def validate_json_payload(payload: Dict[str, Any], max_depth: int = 10) -> bool:
    """
    Validate JSON payload to prevent deeply nested structures
    """
    def check_depth(obj, depth=0):
        if depth > max_depth:
            return False
        if isinstance(obj, dict):
            return all(check_depth(v, depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            return all(check_depth(item, depth + 1) for item in obj)
        return True
    
    return check_depth(payload)


# Security Headers
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
}


def add_security_headers(response):
    """Add security headers to response"""
    for header, value in SECURITY_HEADERS.items():
        response.headers[header] = value
    return response


# API Key Management (simpler alternative to JWT)
API_KEYS = {
    # Hash of API keys - in production, load from secure storage
    "production_key_hash": hashlib.sha256(b"your_api_key_here").hexdigest()
}


def verify_api_key(api_key: str) -> bool:
    """Verify API key"""
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    return key_hash in API_KEYS.values()


# To integrate in main.py:
"""
from security_config import verify_token, rate_limit_dependency, add_security_headers

# Add middleware
@app.middleware("http")
async def add_security_headers_middleware(request, call_next):
    response = await call_next(request)
    return add_security_headers(response)

# Protect endpoints
@app.post("/v4/consciousness/evolve", dependencies=[Depends(rate_limit_dependency)])
async def evolve_consciousness(
    request: Request,
    token: dict = Depends(verify_token)
):
    ...
"""

#!/usr/bin/env python3
"""
NIS Protocol Authentication & Rate Limiting
Simple but effective security for production

Usage:
    from src.security.auth import verify_api_key, rate_limiter, require_auth
    
    @app.middleware("http")
    async def auth_middleware(request, call_next):
        # Check API key for protected endpoints
        if request.url.path.startswith("/api/"):
            api_key = request.headers.get("X-API-Key")
            if not verify_api_key(api_key):
                return JSONResponse({"error": "Invalid API key"}, status_code=401)
        return await call_next(request)
"""

import os
import time
import hashlib
import secrets
from typing import Optional, Dict, Any
from collections import defaultdict
from functools import wraps
import logging
from enum import Enum

logger = logging.getLogger(__name__)

# ====== USER TIER SYSTEM ======
class UserTier(Enum):
    """User tier levels with different rate limits and access levels"""
    FREE = "free"
    PREMIUM = "premium"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"
    MASTER = "master"

# Rate limits per tier (requests per minute)
TIER_RATE_LIMITS = {
    UserTier.FREE: 10,           # 10 requests/min
    UserTier.PREMIUM: 60,        # 60 requests/min
    UserTier.PRO: 300,           # 300 requests/min
    UserTier.ENTERPRISE: 1000,   # 1000 requests/min
    UserTier.ADMIN: 5000,        # 5000 requests/min
    UserTier.MASTER: float('inf') # Unlimited
}

# API key prefix to tier mapping
API_KEY_TIER_PREFIXES = {
    'sk-master-': UserTier.MASTER,
    'sk-admin-': UserTier.ADMIN,
    'sk-ent-': UserTier.ENTERPRISE,
    'sk-pro-': UserTier.PRO,
    'sk-prem-': UserTier.PREMIUM,
    'sk-free-': UserTier.FREE,
}


# =============================================================================
# API KEY AUTHENTICATION
# =============================================================================

class APIKeyAuth:
    """Simple API key authentication"""
    
    def __init__(self):
        # Master API key from environment (for admin access)
        self.master_key = os.getenv("NIS_API_KEY", "")
        
        # Valid API keys (can be loaded from DB in production)
        self.valid_keys: Dict[str, Dict[str, Any]] = {}
        
        # Initialize master key if set
        if self.master_key:
            self.valid_keys[self._hash_key(self.master_key)] = {
                "name": "master",
                "permissions": ["*"],
                "rate_limit": 1000  # requests per minute
            }
    
    def _hash_key(self, key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def generate_key(self, name: str, permissions: list = None, rate_limit: int = 60) -> str:
        """Generate a new API key"""
        key = f"nis_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(key)
        
        self.valid_keys[key_hash] = {
            "name": name,
            "permissions": permissions or ["read", "write"],
            "rate_limit": rate_limit,
            "created": time.time()
        }
        
        logger.info(f"Generated API key for: {name}")
        return key
    
    def verify(self, api_key: Optional[str]) -> bool:
        """Verify an API key"""
        if not api_key:
            return False
        
        key_hash = self._hash_key(api_key)
        return key_hash in self.valid_keys
    
    def get_key_info(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Get info about an API key"""
        if not api_key:
            return None
        key_hash = self._hash_key(api_key)
        return self.valid_keys.get(key_hash)
    
    def revoke(self, api_key: str) -> bool:
        """Revoke an API key"""
        key_hash = self._hash_key(api_key)
        if key_hash in self.valid_keys:
            del self.valid_keys[key_hash]
            return True
        return False


# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """
    Simple in-memory rate limiter using sliding window
    
    HOW IT WORKS:
    - Tracks requests per IP/key in a time window
    - Returns (allowed: bool, remaining: int, reset_time: int)
    - Can be bypassed with master key
    """
    
    def __init__(self, default_requests_per_minute: int = 60):
        self.default_requests_per_minute = default_requests_per_minute
        self.requests: Dict[str, list] = {}  # identifier -> list of timestamps
        self.tier_cache: Dict[str, UserTier] = {}  # api_key -> tier
    
    def get_tier_from_api_key(self, api_key: str) -> UserTier:
        """Determine user tier from API key prefix"""
        if not api_key:
            return UserTier.FREE
        
        # Check cache first
        if api_key in self.tier_cache:
            return self.tier_cache[api_key]
        
        # Check prefix
        for prefix, tier in API_KEY_TIER_PREFIXES.items():
            if api_key.startswith(prefix):
                self.tier_cache[api_key] = tier
                return tier
        
        # Default to free tier
        self.tier_cache[api_key] = UserTier.FREE
        return UserTier.FREE
    
    def get_rate_limit_for_tier(self, tier: UserTier) -> int:
        """Get rate limit for a specific tier"""
        return TIER_RATE_LIMITS.get(tier, self.default_requests_per_minute)
    
    def check_limit(self, identifier: str, api_key: Optional[str] = None, custom_limit: Optional[int] = None) -> tuple:
        """
        Check if request is allowed based on user tier
        
        Returns: (allowed, remaining, reset_time, tier)
        """
        now = time.time()
        tier = self.get_tier_from_api_key(api_key) if api_key else UserTier.FREE
        limit = custom_limit if custom_limit else self.get_rate_limit_for_tier(tier)
        
        # Master tier has unlimited access
        if tier == UserTier.MASTER or limit == float('inf'):
            return True, float('inf'), 0, tier.value
        
        # Clean old requests
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        self.requests[identifier] = [t for t in self.requests[identifier] if t > now - 60]
        
        # Check count
        current_count = len(self.requests[identifier])
        
        if current_count >= limit:
            # Calculate reset time
            oldest = min(self.requests[identifier]) if self.requests[identifier] else now
            reset_time = int(oldest + 60 - now)
            return False, 0, reset_time, tier.value
        
        # Record this request
        self.requests[identifier].append(now)
        
        remaining = limit - current_count - 1
        reset_time = int(60)
        
        return True, remaining, reset_time, tier.value
    
    def get_headers(self, identifier: str, api_key: Optional[str] = None, limit: int = None) -> Dict[str, str]:
        """Get rate limit headers for response"""
        allowed, remaining, reset, tier = self.check_limit(identifier, api_key, limit)
        # Don't actually count this check
        if allowed and identifier in self.requests and self.requests[identifier]:
            self.requests[identifier].pop()  # Remove the check we just added
        
        return {
            "X-RateLimit-Limit": str(limit or self.default_requests_per_minute),
            "X-RateLimit-Remaining": str(max(0, int(remaining)) if remaining != float('inf') else 999999),
            "X-RateLimit-Reset": str(reset),
            "X-RateLimit-Tier": tier
        }


# =============================================================================
# MIDDLEWARE HELPERS
# =============================================================================

# Global instances
api_auth = APIKeyAuth()
rate_limiter = RateLimiter(default_requests_per_minute=60)


def verify_api_key(api_key: Optional[str]) -> bool:
    """Verify API key (convenience function)"""
    # If no master key is set, allow all requests (dev mode)
    if not api_auth.master_key:
        return True
    return api_auth.verify(api_key)


def get_rate_limit_for_key(api_key: Optional[str]) -> int:
    """Get rate limit for an API key"""
    if not api_key:
        return 30  # Anonymous limit
    
    info = api_auth.get_key_info(api_key)
    if info:
        return info.get("rate_limit", 60)
    return 60  # Default limit


def check_rate_limit(identifier: str, api_key: Optional[str] = None) -> tuple:
    """
    Check rate limit for identifier with tier-based limits
    Returns: (allowed, remaining, reset_time, tier)
    """
    return rate_limiter.check_limit(identifier, api_key)


# =============================================================================
# FASTAPI INTEGRATION
# =============================================================================

def require_auth(func):
    """
    Decorator to require API key authentication
    
    Usage:
        @app.post("/protected")
        @require_auth
        async def protected_endpoint(request: Request):
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Find request object in args/kwargs
        request = kwargs.get('request')
        if not request:
            for arg in args:
                if hasattr(arg, 'headers'):
                    request = arg
                    break
        
        if request:
            api_key = request.headers.get("X-API-Key")
            if not verify_api_key(api_key):
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    {"error": "Invalid or missing API key"},
                    status_code=401
                )
        
        return await func(*args, **kwargs)
    return wrapper


# =============================================================================
# DOCUMENTATION
# =============================================================================

"""
SECURITY SETUP GUIDE
====================

1. SET MASTER API KEY:
   Add to .env:
   NIS_API_KEY=your_secret_master_key_here

2. GENERATE USER KEYS:
   from src.security.auth import api_auth
   new_key = api_auth.generate_key("user_name", rate_limit=100)
   # Give this key to the user

3. USE IN REQUESTS:
   curl -H "X-API-Key: nis_xxxxx" http://localhost:8000/chat

4. RATE LIMITS:
   - Anonymous: 30 requests/minute
   - Authenticated: 60 requests/minute (default)
   - Master key: 1000 requests/minute

5. PROTECTED ENDPOINTS:
   Add @require_auth decorator to sensitive endpoints

6. MIDDLEWARE EXAMPLE:
   @app.middleware("http")
   async def security_middleware(request: Request, call_next):
       # Skip health checks
       if request.url.path in ["/health", "/metrics"]:
           return await call_next(request)
       
       # Check rate limit
       client_ip = request.client.host
       api_key = request.headers.get("X-API-Key")
       allowed, remaining, reset = check_rate_limit(client_ip, api_key)
       
       if not allowed:
           return JSONResponse(
               {"error": "Rate limit exceeded", "reset_in": reset},
               status_code=429
           )
       
       response = await call_next(request)
       
       # Add rate limit headers
       response.headers["X-RateLimit-Remaining"] = str(remaining)
       return response
"""

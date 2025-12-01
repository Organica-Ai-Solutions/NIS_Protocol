"""
NIS Protocol v4.0 - Authentication Routes

This module contains authentication and user management endpoints:
- User signup/login/logout
- Token management
- User profile management
- API key management
- Usage statistics

MIGRATION STATUS: Ready for testing
- These routes mirror the ones in main.py
- Can be tested independently before switching over

Usage:
    from routes.auth import router as auth_router
    app.include_router(auth_router, tags=["Authentication"])
"""

import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("nis.routes.auth")

# Create router
router = APIRouter(tags=["Authentication"])


# ====== Dependency Injection ======

def get_user_manager():
    return getattr(router, '_user_manager', None)


# ====== Authentication Endpoints ======

@router.post("/auth/signup")
async def auth_signup(request: Dict[str, Any]):
    """
    📝 Create new user account
    """
    user_manager = get_user_manager()
    
    if not user_manager:
        raise HTTPException(status_code=503, detail="User manager not initialized")
    
    email = request.get("email", "")
    password = request.get("password", "")
    name = request.get("name", "")
    
    if not email or not password or not name:
        raise HTTPException(status_code=400, detail="Email, password, and name required")
    
    result = user_manager.signup(email, password, name)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Signup failed"))
    
    return result


@router.post("/auth/login")
async def auth_login(request: Dict[str, Any]):
    """
    🔑 Authenticate user
    """
    user_manager = get_user_manager()
    
    if not user_manager:
        raise HTTPException(status_code=503, detail="User manager not initialized")
    
    email = request.get("email", "")
    password = request.get("password", "")
    
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")
    
    result = user_manager.login(email, password)
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result.get("error", "Login failed"))
    
    return result


@router.post("/auth/logout")
async def auth_logout(request: Dict[str, Any]):
    """
    🚪 Logout and invalidate token
    """
    user_manager = get_user_manager()
    
    if not user_manager:
        raise HTTPException(status_code=503, detail="User manager not initialized")
    
    token = request.get("token", "")
    return user_manager.logout(token)


@router.post("/auth/recover")
async def auth_recover(request: Dict[str, Any]):
    """
    🔄 Initiate password recovery
    """
    user_manager = get_user_manager()
    
    if not user_manager:
        raise HTTPException(status_code=503, detail="User manager not initialized")
    
    email = request.get("email", "")
    if not email:
        raise HTTPException(status_code=400, detail="Email required")
    
    result = user_manager.recover_password(email)
    return result


@router.post("/auth/refresh")
async def auth_refresh(request: Dict[str, Any]):
    """
    🔄 Refresh authentication token
    """
    user_manager = get_user_manager()
    
    if not user_manager:
        raise HTTPException(status_code=503, detail="User manager not initialized")
    
    token = request.get("token", "")
    if not token:
        raise HTTPException(status_code=400, detail="Token required")
    
    result = user_manager.refresh_token(token)
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result.get("error", "Refresh failed"))
    
    return result


@router.get("/auth/verify")
async def auth_verify(token: str):
    """
    ✅ Verify token validity
    """
    user_manager = get_user_manager()
    
    if not user_manager:
        raise HTTPException(status_code=503, detail="User manager not initialized")
    
    user = user_manager.verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return {"valid": True, "user": user}


# ====== User Management Endpoints ======

@router.get("/users/profile")
async def get_user_profile(token: str):
    """
    👤 Get current user profile
    """
    user_manager = get_user_manager()
    
    if not user_manager:
        raise HTTPException(status_code=503, detail="User manager not initialized")
    
    user = user_manager.verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


@router.put("/users/profile")
async def update_user_profile(request: Dict[str, Any]):
    """
    ✏️ Update user profile
    """
    user_manager = get_user_manager()
    
    if not user_manager:
        raise HTTPException(status_code=503, detail="User manager not initialized")
    
    token = request.get("token", "")
    user = user_manager.verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    updates = {k: v for k, v in request.items() if k in ["name", "settings", "password"]}
    result = user_manager.update_user(user["id"], updates)
    return result


@router.post("/users/api-keys")
async def create_api_key(request: Dict[str, Any]):
    """
    🔑 Create new API key
    """
    user_manager = get_user_manager()
    
    if not user_manager:
        raise HTTPException(status_code=503, detail="User manager not initialized")
    
    token = request.get("token", "")
    name = request.get("name", "Unnamed Key")
    
    user = user_manager.verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    result = user_manager.create_api_key(user["id"], name)
    return result


@router.delete("/users/api-keys/{key_id}")
async def delete_api_key(key_id: str, token: str):
    """
    🗑️ Delete API key
    """
    user_manager = get_user_manager()
    
    if not user_manager:
        raise HTTPException(status_code=503, detail="User manager not initialized")
    
    user = user_manager.verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    result = user_manager.delete_api_key(user["id"], key_id)
    return result


@router.get("/users/usage")
async def get_user_usage(token: str):
    """
    📊 Get user usage statistics
    """
    user_manager = get_user_manager()
    
    if not user_manager:
        raise HTTPException(status_code=503, detail="User manager not initialized")
    
    user = user_manager.verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    return user_manager.get_usage(user["id"])


# ====== Dependency Injection Helper ======

def set_dependencies(user_manager=None):
    """Set dependencies for the auth router"""
    router._user_manager = user_manager

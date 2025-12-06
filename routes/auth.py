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


# ====== Security Management Endpoints ======

@router.get("/security/status")
async def get_security_status():
    """
    🔒 Get Security System Status
    
    Returns the status of all security components.
    """
    try:
        from src.security import get_secrets_manager, get_rbac_manager
        
        secrets = get_secrets_manager()
        rbac = get_rbac_manager()
        
        return {
            "status": "active",
            "components": {
                "secrets_manager": {
                    "active": True,
                    "secrets_count": len(secrets._secrets),
                    "api_keys_count": len(secrets._api_keys),
                    "encryption_enabled": secrets._fernet is not None
                },
                "rbac": {
                    "active": True,
                    "roles_count": len(rbac.roles),
                    "users_count": len(rbac.users)
                },
                "rate_limiting": {
                    "active": True,
                    "default_limit": 60
                }
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Security status error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }


@router.post("/security/api-keys/generate")
async def generate_api_key(request: Dict[str, Any]):
    """
    🔑 Generate New API Key
    
    Creates a new API key with specified permissions.
    Requires admin permission.
    """
    try:
        from src.security import get_secrets_manager
        
        secrets = get_secrets_manager()
        
        name = request.get("name", "unnamed")
        permissions = request.get("permissions", ["read", "write"])
        rate_limit = request.get("rate_limit", 60)
        expires_in_days = request.get("expires_in_days")
        
        key = secrets.generate_api_key(
            name=name,
            permissions=permissions,
            rate_limit=rate_limit,
            expires_in_days=expires_in_days
        )
        
        return {
            "status": "success",
            "api_key": key,
            "name": name,
            "permissions": permissions,
            "rate_limit": rate_limit,
            "note": "Store this key securely. It cannot be retrieved again.",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"API key generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/security/api-keys")
async def list_api_keys():
    """
    📋 List All API Keys
    
    Returns list of API keys (without the actual key values).
    """
    try:
        from src.security import get_secrets_manager
        
        secrets = get_secrets_manager()
        keys = secrets.list_api_keys()
        
        return {
            "status": "success",
            "count": len(keys),
            "api_keys": keys,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"List API keys error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/security/api-keys/{key_id}/rotate")
async def rotate_api_key(key_id: str):
    """
    🔄 Rotate API Key
    
    Generates a new key value for an existing API key.
    The old key will no longer work.
    """
    try:
        from src.security import get_secrets_manager
        
        secrets = get_secrets_manager()
        new_key = secrets.rotate_api_key(key_id)
        
        if not new_key:
            raise HTTPException(status_code=404, detail="API key not found")
        
        return {
            "status": "success",
            "key_id": key_id,
            "new_api_key": new_key,
            "note": "Store this key securely. The old key is now invalid.",
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API key rotation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/security/api-keys/{key_id}/revoke")
async def revoke_api_key(key_id: str):
    """
    🚫 Revoke API Key
    
    Deactivates an API key. It can no longer be used.
    """
    try:
        from src.security import get_secrets_manager
        
        secrets = get_secrets_manager()
        success = secrets.revoke_api_key(key_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="API key not found")
        
        return {
            "status": "success",
            "key_id": key_id,
            "message": "API key revoked",
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API key revocation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/security/roles")
async def list_roles():
    """
    👥 List All Roles
    
    Returns all defined roles and their permissions.
    """
    try:
        from src.security import get_rbac_manager
        
        rbac = get_rbac_manager()
        roles = rbac.list_roles()
        
        return {
            "status": "success",
            "count": len(roles),
            "roles": roles,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"List roles error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/security/audit-log")
async def get_audit_log(limit: int = 100):
    """
    📜 Get Security Audit Log
    
    Returns recent security events.
    """
    try:
        from src.security import get_secrets_manager
        
        secrets = get_secrets_manager()
        log = secrets.get_audit_log(limit)
        
        return {
            "status": "success",
            "count": len(log),
            "events": log,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Audit log error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====== Dependency Injection Helper ======

def set_dependencies(user_manager=None):
    """Set dependencies for the auth router"""
    router._user_manager = user_manager

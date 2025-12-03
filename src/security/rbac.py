#!/usr/bin/env python3
"""
NIS Protocol Role-Based Access Control (RBAC)
Fine-grained permission management for multi-tenant deployments

Features:
- Role definitions with hierarchical permissions
- Resource-based access control
- Permission inheritance
- Audit logging
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

logger = logging.getLogger("nis.security.rbac")


class Permission(Enum):
    """Available permissions in NIS Protocol"""
    # Read permissions
    READ_HEALTH = "read:health"
    READ_METRICS = "read:metrics"
    READ_SYSTEM = "read:system"
    READ_INFRASTRUCTURE = "read:infrastructure"
    
    # Chat permissions
    CHAT_SIMPLE = "chat:simple"
    CHAT_ADVANCED = "chat:advanced"
    CHAT_RESEARCH = "chat:research"
    
    # Robotics permissions
    ROBOTICS_READ = "robotics:read"
    ROBOTICS_CONTROL = "robotics:control"
    ROBOTICS_EMERGENCY = "robotics:emergency"
    
    # Physics permissions
    PHYSICS_READ = "physics:read"
    PHYSICS_VALIDATE = "physics:validate"
    PHYSICS_SIMULATE = "physics:simulate"
    
    # Training permissions
    TRAINING_READ = "training:read"
    TRAINING_TRIGGER = "training:trigger"
    TRAINING_EXPORT = "training:export"
    
    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_KEYS = "admin:keys"
    ADMIN_SYSTEM = "admin:system"
    ADMIN_FULL = "admin:*"
    
    # Wildcard
    ALL = "*"


@dataclass
class Role:
    """A role with a set of permissions"""
    name: str
    description: str
    permissions: Set[str]
    inherits_from: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    is_system: bool = False


@dataclass
class User:
    """A user with roles"""
    user_id: str
    name: str
    roles: List[str]
    direct_permissions: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


# Default system roles
DEFAULT_ROLES = {
    "anonymous": Role(
        name="anonymous",
        description="Unauthenticated users",
        permissions={
            Permission.READ_HEALTH.value,
            Permission.READ_METRICS.value,
        },
        is_system=True
    ),
    "user": Role(
        name="user",
        description="Basic authenticated user",
        permissions={
            Permission.READ_HEALTH.value,
            Permission.READ_METRICS.value,
            Permission.READ_SYSTEM.value,
            Permission.CHAT_SIMPLE.value,
            Permission.ROBOTICS_READ.value,
            Permission.PHYSICS_READ.value,
        },
        inherits_from=["anonymous"],
        is_system=True
    ),
    "developer": Role(
        name="developer",
        description="Developer with extended access",
        permissions={
            Permission.CHAT_ADVANCED.value,
            Permission.CHAT_RESEARCH.value,
            Permission.ROBOTICS_CONTROL.value,
            Permission.PHYSICS_VALIDATE.value,
            Permission.PHYSICS_SIMULATE.value,
            Permission.TRAINING_READ.value,
        },
        inherits_from=["user"],
        is_system=True
    ),
    "operator": Role(
        name="operator",
        description="System operator with control access",
        permissions={
            Permission.READ_INFRASTRUCTURE.value,
            Permission.ROBOTICS_CONTROL.value,
            Permission.ROBOTICS_EMERGENCY.value,
            Permission.TRAINING_TRIGGER.value,
        },
        inherits_from=["developer"],
        is_system=True
    ),
    "admin": Role(
        name="admin",
        description="Full administrative access",
        permissions={
            Permission.ADMIN_USERS.value,
            Permission.ADMIN_KEYS.value,
            Permission.ADMIN_SYSTEM.value,
            Permission.TRAINING_EXPORT.value,
        },
        inherits_from=["operator"],
        is_system=True
    ),
    "superadmin": Role(
        name="superadmin",
        description="Unrestricted access",
        permissions={Permission.ALL.value},
        is_system=True
    )
}


# Endpoint to permission mapping
ENDPOINT_PERMISSIONS = {
    # Health & System
    "/health": Permission.READ_HEALTH.value,
    "/metrics": Permission.READ_METRICS.value,
    "/system/status": Permission.READ_SYSTEM.value,
    "/infrastructure/status": Permission.READ_INFRASTRUCTURE.value,
    "/infrastructure/kafka": Permission.READ_INFRASTRUCTURE.value,
    "/infrastructure/redis": Permission.READ_INFRASTRUCTURE.value,
    "/runner/status": Permission.READ_INFRASTRUCTURE.value,
    
    # Chat
    "/chat/simple": Permission.CHAT_SIMPLE.value,
    "/chat/advanced": Permission.CHAT_ADVANCED.value,
    "/research/deep": Permission.CHAT_RESEARCH.value,
    "/reasoning/collaborative": Permission.CHAT_ADVANCED.value,
    
    # Robotics
    "/robotics/capabilities": Permission.ROBOTICS_READ.value,
    "/robotics/forward_kinematics": Permission.ROBOTICS_CONTROL.value,
    "/robotics/inverse_kinematics": Permission.ROBOTICS_CONTROL.value,
    "/robotics/plan_trajectory": Permission.ROBOTICS_CONTROL.value,
    "/robotics/can/status": Permission.ROBOTICS_READ.value,
    "/robotics/can/emergency_stop": Permission.ROBOTICS_EMERGENCY.value,
    "/robotics/obd/status": Permission.ROBOTICS_READ.value,
    "/robotics/obd/vehicle": Permission.ROBOTICS_READ.value,
    
    # Physics
    "/physics/capabilities": Permission.PHYSICS_READ.value,
    "/physics/constants": Permission.PHYSICS_READ.value,
    "/physics/validate": Permission.PHYSICS_VALIDATE.value,
    "/physics/solve/heat-equation": Permission.PHYSICS_SIMULATE.value,
    
    # Training
    "/bitnet/status": Permission.TRAINING_READ.value,
    "/bitnet/training/status": Permission.TRAINING_READ.value,
    "/training/bitnet/force": Permission.TRAINING_TRIGGER.value,
    "/models/bitnet/download": Permission.TRAINING_EXPORT.value,
    
    # Admin
    "/auth/users": Permission.ADMIN_USERS.value,
    "/auth/keys": Permission.ADMIN_KEYS.value,
    "/v4/consciousness/status": Permission.READ_SYSTEM.value,
    "/v4/dashboard/complete": Permission.READ_SYSTEM.value,
}


class RBACManager:
    """
    Role-Based Access Control Manager
    
    Manages roles, users, and permission checks
    """
    
    def __init__(self):
        self.roles: Dict[str, Role] = dict(DEFAULT_ROLES)
        self.users: Dict[str, User] = {}
        self._permission_cache: Dict[str, Set[str]] = {}
        
        logger.info(f"RBAC Manager initialized with {len(self.roles)} roles")
    
    def _get_role_permissions(self, role_name: str, visited: Set[str] = None) -> Set[str]:
        """Get all permissions for a role, including inherited ones"""
        if visited is None:
            visited = set()
        
        if role_name in visited:
            return set()  # Prevent circular inheritance
        
        visited.add(role_name)
        
        role = self.roles.get(role_name)
        if not role:
            return set()
        
        permissions = set(role.permissions)
        
        # Add inherited permissions
        for parent_role in role.inherits_from:
            permissions.update(self._get_role_permissions(parent_role, visited))
        
        return permissions
    
    def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for a user"""
        # Check cache
        if user_id in self._permission_cache:
            return self._permission_cache[user_id]
        
        user = self.users.get(user_id)
        if not user or not user.is_active:
            # Return anonymous permissions
            return self._get_role_permissions("anonymous")
        
        permissions = set(user.direct_permissions)
        
        for role_name in user.roles:
            permissions.update(self._get_role_permissions(role_name))
        
        # Cache the result
        self._permission_cache[user_id] = permissions
        
        return permissions
    
    def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if a user has a specific permission"""
        permissions = self.get_user_permissions(user_id)
        
        # Check for wildcard
        if Permission.ALL.value in permissions:
            return True
        
        # Check for admin wildcard
        if Permission.ADMIN_FULL.value in permissions and permission.startswith("admin:"):
            return True
        
        return permission in permissions
    
    def check_endpoint_access(self, user_id: str, endpoint: str) -> bool:
        """Check if a user can access an endpoint"""
        # Find the permission for this endpoint
        permission = None
        
        # Exact match
        if endpoint in ENDPOINT_PERMISSIONS:
            permission = ENDPOINT_PERMISSIONS[endpoint]
        else:
            # Prefix match
            for ep, perm in ENDPOINT_PERMISSIONS.items():
                if endpoint.startswith(ep):
                    permission = perm
                    break
        
        if not permission:
            # No permission defined, allow by default (or deny based on policy)
            return True
        
        return self.check_permission(user_id, permission)
    
    def create_role(
        self,
        name: str,
        description: str,
        permissions: List[str],
        inherits_from: List[str] = None
    ) -> Role:
        """Create a new role"""
        if name in self.roles and self.roles[name].is_system:
            raise ValueError(f"Cannot modify system role: {name}")
        
        role = Role(
            name=name,
            description=description,
            permissions=set(permissions),
            inherits_from=inherits_from or []
        )
        
        self.roles[name] = role
        self._permission_cache.clear()  # Invalidate cache
        
        logger.info(f"Created role: {name}")
        return role
    
    def delete_role(self, name: str) -> bool:
        """Delete a role"""
        if name not in self.roles:
            return False
        
        if self.roles[name].is_system:
            raise ValueError(f"Cannot delete system role: {name}")
        
        del self.roles[name]
        self._permission_cache.clear()
        
        logger.info(f"Deleted role: {name}")
        return True
    
    def create_user(
        self,
        user_id: str,
        name: str,
        roles: List[str],
        direct_permissions: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> User:
        """Create a new user"""
        user = User(
            user_id=user_id,
            name=name,
            roles=roles,
            direct_permissions=set(direct_permissions or []),
            metadata=metadata or {}
        )
        
        self.users[user_id] = user
        self._permission_cache.pop(user_id, None)
        
        logger.info(f"Created user: {user_id} with roles: {roles}")
        return user
    
    def update_user_roles(self, user_id: str, roles: List[str]) -> bool:
        """Update a user's roles"""
        if user_id not in self.users:
            return False
        
        self.users[user_id].roles = roles
        self._permission_cache.pop(user_id, None)
        
        logger.info(f"Updated roles for user {user_id}: {roles}")
        return True
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user"""
        if user_id not in self.users:
            return False
        
        self.users[user_id].is_active = False
        self._permission_cache.pop(user_id, None)
        
        logger.info(f"Deactivated user: {user_id}")
        return True
    
    def list_roles(self) -> List[Dict[str, Any]]:
        """List all roles"""
        return [
            {
                "name": role.name,
                "description": role.description,
                "permissions": list(role.permissions),
                "inherits_from": role.inherits_from,
                "is_system": role.is_system
            }
            for role in self.roles.values()
        ]
    
    def list_users(self) -> List[Dict[str, Any]]:
        """List all users"""
        return [
            {
                "user_id": user.user_id,
                "name": user.name,
                "roles": user.roles,
                "is_active": user.is_active,
                "created_at": user.created_at
            }
            for user in self.users.values()
        ]


# Singleton instance
_rbac_manager: Optional[RBACManager] = None


def get_rbac_manager() -> RBACManager:
    """Get the RBAC manager singleton"""
    global _rbac_manager
    if _rbac_manager is None:
        _rbac_manager = RBACManager()
    return _rbac_manager


def require_permission(permission: str):
    """
    Decorator to require a specific permission
    
    Usage:
        @app.get("/admin/users")
        @require_permission("admin:users")
        async def list_users(request: Request):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from fastapi import Request
            from fastapi.responses import JSONResponse
            
            # Find request in args/kwargs
            request = kwargs.get('request')
            if not request:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
            
            if not request:
                return await func(*args, **kwargs)
            
            # Get user ID from request (from API key or session)
            user_id = getattr(request.state, 'user_id', 'anonymous')
            
            rbac = get_rbac_manager()
            if not rbac.check_permission(user_id, permission):
                return JSONResponse(
                    {
                        "error": "Permission denied",
                        "required_permission": permission
                    },
                    status_code=403
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

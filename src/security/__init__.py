"""
NIS Protocol Security Module

Features:
- API Key Authentication
- Rate Limiting
- Secrets Management with Rotation
- Role-Based Access Control (RBAC)
- Audit Logging
"""

from .auth import (
    api_auth,
    rate_limiter,
    verify_api_key,
    check_rate_limit,
    require_auth,
    APIKeyAuth,
    RateLimiter
)

from .secrets_manager import (
    SecretsManager,
    get_secrets_manager,
    SecretMetadata,
    APIKeyInfo
)

from .rbac import (
    RBACManager,
    get_rbac_manager,
    require_permission,
    Permission,
    Role,
    User,
    ENDPOINT_PERMISSIONS,
    DEFAULT_ROLES
)

__all__ = [
    # Auth
    'api_auth',
    'rate_limiter', 
    'verify_api_key',
    'check_rate_limit',
    'require_auth',
    'APIKeyAuth',
    'RateLimiter',
    # Secrets
    'SecretsManager',
    'get_secrets_manager',
    'SecretMetadata',
    'APIKeyInfo',
    # RBAC
    'RBACManager',
    'get_rbac_manager',
    'require_permission',
    'Permission',
    'Role',
    'User',
    'ENDPOINT_PERMISSIONS',
    'DEFAULT_ROLES'
]

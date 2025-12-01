"""
User Management System for NIS Protocol v4.0
Handles authentication, API keys, and user profiles
"""

import hashlib
import secrets
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Dev bypass key - CHANGE IN PRODUCTION
DEV_MASTER_KEY = "NIS_DEV_2024_MASTER"

@dataclass
class ApiKey:
    id: str
    name: str
    key_hash: str  # Store hash, not plain key
    created_at: str
    last_used: Optional[str] = None
    permissions: List[str] = field(default_factory=lambda: ["read", "write"])

@dataclass
class UserProfile:
    id: str
    email: str
    name: str
    password_hash: str
    role: str = "user"  # user, admin, dev
    api_keys: List[ApiKey] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_login: Optional[str] = None
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> Dict[str, Any]:
        """Return user data without sensitive fields"""
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "api_keys": [{"id": k.id, "name": k.name, "created_at": k.created_at} for k in self.api_keys],
            "created_at": self.created_at,
            "last_login": self.last_login,
        }


class UserManager:
    """Manages user authentication and profiles"""
    
    def __init__(self, storage_path: str = "data/users"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.users_file = self.storage_path / "users.json"
        self.tokens_file = self.storage_path / "tokens.json"
        self._users: Dict[str, UserProfile] = {}
        self._tokens: Dict[str, Dict[str, Any]] = {}  # token -> {user_id, expires}
        self._load()

    def _load(self):
        """Load users and tokens from storage"""
        try:
            if self.users_file.exists():
                with open(self.users_file, 'r') as f:
                    data = json.load(f)
                    for uid, udata in data.items():
                        api_keys = [ApiKey(**k) for k in udata.pop('api_keys', [])]
                        self._users[uid] = UserProfile(**udata, api_keys=api_keys)
            if self.tokens_file.exists():
                with open(self.tokens_file, 'r') as f:
                    self._tokens = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load user data: {e}")

    def _save(self):
        """Save users and tokens to storage"""
        try:
            users_data = {}
            for uid, user in self._users.items():
                udict = asdict(user)
                users_data[uid] = udict
            with open(self.users_file, 'w') as f:
                json.dump(users_data, f, indent=2)
            with open(self.tokens_file, 'w') as f:
                json.dump(self._tokens, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save user data: {e}")

    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = "nis_protocol_v4_salt"  # In production, use unique salt per user
        return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()

    def _generate_token(self) -> str:
        """Generate secure token"""
        return secrets.token_urlsafe(32)

    def _generate_api_key(self) -> str:
        """Generate API key"""
        return f"nis_{secrets.token_urlsafe(24)}"

    # ==================== AUTH ====================

    def signup(self, email: str, password: str, name: str) -> Dict[str, Any]:
        """Create new user account"""
        # Check if email exists
        for user in self._users.values():
            if user.email.lower() == email.lower():
                return {"success": False, "error": "Email already registered"}

        # Create user
        user_id = f"user_{secrets.token_hex(8)}"
        user = UserProfile(
            id=user_id,
            email=email.lower(),
            name=name,
            password_hash=self._hash_password(password),
            role="user",
        )
        self._users[user_id] = user

        # Generate token
        token = self._generate_token()
        self._tokens[token] = {
            "user_id": user_id,
            "expires": (datetime.utcnow() + timedelta(days=7)).isoformat()
        }

        self._save()
        logger.info(f"New user registered: {email}")

        return {
            "success": True,
            "token": token,
            "user": user.to_public_dict()
        }

    def login(self, email: str, password: str) -> Dict[str, Any]:
        """Authenticate user"""
        # Dev bypass
        if password == DEV_MASTER_KEY:
            return self._dev_bypass_login()

        # Find user
        user = None
        for u in self._users.values():
            if u.email.lower() == email.lower():
                user = u
                break

        if not user:
            return {"success": False, "error": "Invalid credentials"}

        # Verify password
        if user.password_hash != self._hash_password(password):
            return {"success": False, "error": "Invalid credentials"}

        # Update last login
        user.last_login = datetime.utcnow().isoformat()

        # Generate token
        token = self._generate_token()
        self._tokens[token] = {
            "user_id": user.id,
            "expires": (datetime.utcnow() + timedelta(days=7)).isoformat()
        }

        self._save()
        logger.info(f"User logged in: {email}")

        return {
            "success": True,
            "token": token,
            "user": user.to_public_dict()
        }

    def _dev_bypass_login(self) -> Dict[str, Any]:
        """Dev bypass authentication"""
        dev_user = UserProfile(
            id="dev_master",
            email="dev@nisprotocol.ai",
            name="Master Developer",
            password_hash="",
            role="admin",
            api_keys=[ApiKey(
                id="dev_key",
                name="Dev Unlimited",
                key_hash="dev",
                created_at=datetime.utcnow().isoformat(),
                permissions=["read", "write", "admin"]
            )]
        )
        token = f"dev_master_{secrets.token_hex(16)}"
        self._tokens[token] = {
            "user_id": "dev_master",
            "expires": (datetime.utcnow() + timedelta(days=365)).isoformat(),
            "is_dev": True
        }
        logger.warning("DEV BYPASS LOGIN USED")
        return {
            "success": True,
            "token": token,
            "user": dev_user.to_public_dict(),
            "message": "Developer bypass active"
        }

    def logout(self, token: str) -> Dict[str, Any]:
        """Invalidate token"""
        if token in self._tokens:
            del self._tokens[token]
            self._save()
        return {"success": True}

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify token and return user info"""
        if token not in self._tokens:
            return None

        token_data = self._tokens[token]
        expires = datetime.fromisoformat(token_data["expires"])
        
        if datetime.utcnow() > expires:
            del self._tokens[token]
            self._save()
            return None

        user_id = token_data["user_id"]
        
        # Dev user
        if token_data.get("is_dev"):
            return {"id": "dev_master", "role": "admin", "is_dev": True}

        user = self._users.get(user_id)
        if not user:
            return None

        return user.to_public_dict()

    def recover_password(self, email: str) -> Dict[str, Any]:
        """Initiate password recovery"""
        # In production, send email with reset link
        for user in self._users.values():
            if user.email.lower() == email.lower():
                logger.info(f"Password recovery requested for: {email}")
                return {"success": True, "message": "Recovery email sent"}
        return {"success": False, "error": "Email not found"}

    def refresh_token(self, token: str) -> Dict[str, Any]:
        """Refresh authentication token"""
        user_info = self.verify_token(token)
        if not user_info:
            return {"success": False, "error": "Invalid token"}

        # Invalidate old token
        del self._tokens[token]

        # Generate new token
        new_token = self._generate_token()
        self._tokens[new_token] = {
            "user_id": user_info["id"],
            "expires": (datetime.utcnow() + timedelta(days=7)).isoformat()
        }
        self._save()

        return {"success": True, "token": new_token}

    # ==================== API KEYS ====================

    def create_api_key(self, user_id: str, name: str) -> Dict[str, Any]:
        """Create new API key for user"""
        user = self._users.get(user_id)
        if not user:
            return {"success": False, "error": "User not found"}

        # Generate key
        plain_key = self._generate_api_key()
        key_hash = hashlib.sha256(plain_key.encode()).hexdigest()

        api_key = ApiKey(
            id=f"key_{secrets.token_hex(8)}",
            name=name,
            key_hash=key_hash,
            created_at=datetime.utcnow().isoformat()
        )
        user.api_keys.append(api_key)
        self._save()

        # Return plain key only once
        return {
            "success": True,
            "key": plain_key,
            "id": api_key.id,
            "name": api_key.name,
            "created_at": api_key.created_at
        }

    def delete_api_key(self, user_id: str, key_id: str) -> Dict[str, Any]:
        """Delete API key"""
        user = self._users.get(user_id)
        if not user:
            return {"success": False, "error": "User not found"}

        user.api_keys = [k for k in user.api_keys if k.id != key_id]
        self._save()
        return {"success": True}

    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return user info"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        for user in self._users.values():
            for key in user.api_keys:
                if key.key_hash == key_hash:
                    key.last_used = datetime.utcnow().isoformat()
                    self._save()
                    return user.to_public_dict()
        return None

    # ==================== USER PROFILE ====================

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile"""
        user = self._users.get(user_id)
        return user.to_public_dict() if user else None

    def update_user(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile"""
        user = self._users.get(user_id)
        if not user:
            return {"success": False, "error": "User not found"}

        if "name" in updates:
            user.name = updates["name"]
        if "settings" in updates:
            user.settings.update(updates["settings"])
        if "password" in updates:
            user.password_hash = self._hash_password(updates["password"])

        self._save()
        return {"success": True, "user": user.to_public_dict()}

    def get_usage(self, user_id: str) -> Dict[str, Any]:
        """Get user usage statistics"""
        # In production, track actual usage
        return {
            "api_calls": 0,
            "tokens_used": 0,
            "storage_used": 0,
            "last_activity": datetime.utcnow().isoformat()
        }


# Global instance
user_manager = UserManager()

#!/usr/bin/env python3
"""
NIS Protocol Secrets Manager
Secure secrets management with rotation support

Features:
- Environment-based secrets loading
- Encrypted secrets storage
- API key rotation
- Audit logging
- Integration with HashiCorp Vault (optional)
"""

import os
import json
import time
import hashlib
import secrets
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import base64

logger = logging.getLogger("nis.security.secrets")

# Try to import cryptography for encryption
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography not available, using basic encoding")


@dataclass
class SecretMetadata:
    """Metadata for a secret"""
    name: str
    created_at: float
    rotated_at: Optional[float] = None
    expires_at: Optional[float] = None
    rotation_count: int = 0
    last_accessed: Optional[float] = None
    access_count: int = 0
    tags: List[str] = field(default_factory=list)


@dataclass
class APIKeyInfo:
    """Information about an API key"""
    key_id: str
    name: str
    key_hash: str
    permissions: List[str]
    rate_limit: int
    created_at: float
    expires_at: Optional[float]
    rotated_at: Optional[float] = None
    is_active: bool = True
    last_used: Optional[float] = None
    use_count: int = 0


class SecretsManager:
    """
    Centralized secrets management for NIS Protocol
    
    Supports:
    - Environment variables
    - Encrypted file storage
    - HashiCorp Vault (optional)
    - API key rotation
    """
    
    def __init__(
        self,
        secrets_file: str = None,
        encryption_key: str = None,
        vault_addr: str = None,
        vault_token: str = None
    ):
        self.secrets_file = secrets_file or os.getenv(
            "NIS_SECRETS_FILE", 
            "data/secrets.enc"
        )
        
        # Encryption setup
        self.encryption_key = encryption_key or os.getenv("NIS_ENCRYPTION_KEY")
        self._fernet = None
        if CRYPTO_AVAILABLE and self.encryption_key:
            self._init_encryption()
        
        # Vault setup (optional)
        self.vault_addr = vault_addr or os.getenv("VAULT_ADDR")
        self.vault_token = vault_token or os.getenv("VAULT_TOKEN")
        self._vault_client = None
        
        # In-memory secrets cache
        self._secrets: Dict[str, str] = {}
        self._metadata: Dict[str, SecretMetadata] = {}
        
        # API keys
        self._api_keys: Dict[str, APIKeyInfo] = {}
        
        # Audit log
        self._audit_log: List[Dict[str, Any]] = []
        
        # Load secrets
        self._load_secrets()
        
        logger.info("Secrets Manager initialized")
    
    def _init_encryption(self):
        """Initialize encryption with the provided key"""
        if not CRYPTO_AVAILABLE:
            return
        
        # Derive a key from the encryption key
        salt = b'nis_protocol_salt'  # In production, use a random salt stored securely
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.encryption_key.encode()))
        self._fernet = Fernet(key)
    
    def _encrypt(self, data: str) -> str:
        """Encrypt data"""
        if self._fernet:
            return self._fernet.encrypt(data.encode()).decode()
        return base64.b64encode(data.encode()).decode()
    
    def _decrypt(self, data: str) -> str:
        """Decrypt data"""
        if self._fernet:
            return self._fernet.decrypt(data.encode()).decode()
        return base64.b64decode(data.encode()).decode()
    
    def _load_secrets(self):
        """Load secrets from file and environment"""
        # Load from environment first
        self._load_from_env()
        
        # Load from encrypted file if exists
        if os.path.exists(self.secrets_file):
            self._load_from_file()
        
        # Load from Vault if configured
        if self.vault_addr and self.vault_token:
            self._load_from_vault()
    
    def _load_from_env(self):
        """Load secrets from environment variables"""
        secret_prefixes = ["NIS_", "API_KEY_", "SECRET_"]
        
        for key, value in os.environ.items():
            for prefix in secret_prefixes:
                if key.startswith(prefix):
                    self._secrets[key] = value
                    self._metadata[key] = SecretMetadata(
                        name=key,
                        created_at=time.time(),
                        tags=["env"]
                    )
    
    def _load_from_file(self):
        """Load secrets from encrypted file"""
        try:
            with open(self.secrets_file, 'r') as f:
                encrypted_data = f.read()
            
            decrypted = self._decrypt(encrypted_data)
            data = json.loads(decrypted)
            
            for key, value in data.get("secrets", {}).items():
                self._secrets[key] = value
            
            for key, meta in data.get("metadata", {}).items():
                self._metadata[key] = SecretMetadata(**meta)
            
            for key_id, key_info in data.get("api_keys", {}).items():
                self._api_keys[key_id] = APIKeyInfo(**key_info)
            
            logger.info(f"Loaded {len(self._secrets)} secrets from file")
            
        except Exception as e:
            logger.error(f"Failed to load secrets from file: {e}")
    
    def _save_to_file(self):
        """Save secrets to encrypted file"""
        try:
            data = {
                "secrets": self._secrets,
                "metadata": {k: vars(v) for k, v in self._metadata.items()},
                "api_keys": {k: vars(v) for k, v in self._api_keys.items()}
            }
            
            encrypted = self._encrypt(json.dumps(data))
            
            # Ensure directory exists
            Path(self.secrets_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.secrets_file, 'w') as f:
                f.write(encrypted)
            
            logger.info("Secrets saved to file")
            
        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
    
    def _load_from_vault(self):
        """Load secrets from HashiCorp Vault"""
        try:
            import hvac
            self._vault_client = hvac.Client(
                url=self.vault_addr,
                token=self.vault_token
            )
            
            if self._vault_client.is_authenticated():
                # Load NIS secrets from Vault
                secret = self._vault_client.secrets.kv.v2.read_secret_version(
                    path='nis-protocol'
                )
                
                for key, value in secret['data']['data'].items():
                    self._secrets[key] = value
                    self._metadata[key] = SecretMetadata(
                        name=key,
                        created_at=time.time(),
                        tags=["vault"]
                    )
                
                logger.info("Loaded secrets from Vault")
            
        except ImportError:
            logger.debug("hvac not installed, Vault integration disabled")
        except Exception as e:
            logger.error(f"Failed to load from Vault: {e}")
    
    def _audit(self, action: str, secret_name: str, details: Dict = None):
        """Log an audit event"""
        event = {
            "timestamp": time.time(),
            "action": action,
            "secret": secret_name,
            "details": details or {}
        }
        self._audit_log.append(event)
        logger.info(f"Audit: {action} on {secret_name}")
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def get(self, name: str, default: str = None) -> Optional[str]:
        """Get a secret value"""
        value = self._secrets.get(name, default)
        
        if name in self._metadata:
            self._metadata[name].last_accessed = time.time()
            self._metadata[name].access_count += 1
        
        return value
    
    def set(self, name: str, value: str, tags: List[str] = None, expires_in: int = None):
        """Set a secret value"""
        self._secrets[name] = value
        
        expires_at = None
        if expires_in:
            expires_at = time.time() + expires_in
        
        self._metadata[name] = SecretMetadata(
            name=name,
            created_at=time.time(),
            expires_at=expires_at,
            tags=tags or []
        )
        
        self._audit("set", name)
        self._save_to_file()
    
    def delete(self, name: str) -> bool:
        """Delete a secret"""
        if name in self._secrets:
            del self._secrets[name]
            if name in self._metadata:
                del self._metadata[name]
            self._audit("delete", name)
            self._save_to_file()
            return True
        return False
    
    def rotate(self, name: str, new_value: str) -> bool:
        """Rotate a secret to a new value"""
        if name not in self._secrets:
            return False
        
        old_value = self._secrets[name]
        self._secrets[name] = new_value
        
        if name in self._metadata:
            self._metadata[name].rotated_at = time.time()
            self._metadata[name].rotation_count += 1
        
        self._audit("rotate", name, {"old_hash": hashlib.sha256(old_value.encode()).hexdigest()[:8]})
        self._save_to_file()
        return True
    
    def list_secrets(self) -> List[Dict[str, Any]]:
        """List all secrets (without values)"""
        return [
            {
                "name": name,
                "created_at": meta.created_at,
                "rotated_at": meta.rotated_at,
                "expires_at": meta.expires_at,
                "rotation_count": meta.rotation_count,
                "tags": meta.tags
            }
            for name, meta in self._metadata.items()
        ]
    
    # ========================================================================
    # API KEY MANAGEMENT
    # ========================================================================
    
    def generate_api_key(
        self,
        name: str,
        permissions: List[str] = None,
        rate_limit: int = 60,
        expires_in_days: int = None
    ) -> str:
        """Generate a new API key"""
        key = f"nis_{secrets.token_urlsafe(32)}"
        key_id = secrets.token_hex(8)
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        expires_at = None
        if expires_in_days:
            expires_at = time.time() + (expires_in_days * 86400)
        
        self._api_keys[key_id] = APIKeyInfo(
            key_id=key_id,
            name=name,
            key_hash=key_hash,
            permissions=permissions or ["read", "write"],
            rate_limit=rate_limit,
            created_at=time.time(),
            expires_at=expires_at
        )
        
        self._audit("generate_api_key", name, {"key_id": key_id})
        self._save_to_file()
        
        return key
    
    def verify_api_key(self, key: str) -> Optional[APIKeyInfo]:
        """Verify an API key and return its info"""
        if not key:
            return None
        
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        for key_info in self._api_keys.values():
            if key_info.key_hash == key_hash:
                # Check if expired
                if key_info.expires_at and time.time() > key_info.expires_at:
                    return None
                
                # Check if active
                if not key_info.is_active:
                    return None
                
                # Update usage stats
                key_info.last_used = time.time()
                key_info.use_count += 1
                
                return key_info
        
        return None
    
    def rotate_api_key(self, key_id: str) -> Optional[str]:
        """Rotate an API key, returning the new key"""
        if key_id not in self._api_keys:
            return None
        
        old_info = self._api_keys[key_id]
        
        # Generate new key
        new_key = f"nis_{secrets.token_urlsafe(32)}"
        new_hash = hashlib.sha256(new_key.encode()).hexdigest()
        
        # Update key info
        old_info.key_hash = new_hash
        old_info.rotated_at = time.time()
        
        self._audit("rotate_api_key", old_info.name, {"key_id": key_id})
        self._save_to_file()
        
        return new_key
    
    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key"""
        if key_id not in self._api_keys:
            return False
        
        self._api_keys[key_id].is_active = False
        self._audit("revoke_api_key", self._api_keys[key_id].name, {"key_id": key_id})
        self._save_to_file()
        return True
    
    def list_api_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (without the actual keys)"""
        return [
            {
                "key_id": info.key_id,
                "name": info.name,
                "permissions": info.permissions,
                "rate_limit": info.rate_limit,
                "created_at": info.created_at,
                "expires_at": info.expires_at,
                "rotated_at": info.rotated_at,
                "is_active": info.is_active,
                "last_used": info.last_used,
                "use_count": info.use_count
            }
            for info in self._api_keys.values()
        ]
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries"""
        return self._audit_log[-limit:]
    
    def cleanup_expired(self) -> int:
        """Clean up expired secrets and keys"""
        now = time.time()
        cleaned = 0
        
        # Clean expired secrets
        for name, meta in list(self._metadata.items()):
            if meta.expires_at and now > meta.expires_at:
                self.delete(name)
                cleaned += 1
        
        # Deactivate expired API keys
        for key_info in self._api_keys.values():
            if key_info.expires_at and now > key_info.expires_at and key_info.is_active:
                key_info.is_active = False
                cleaned += 1
        
        if cleaned > 0:
            self._save_to_file()
            logger.info(f"Cleaned up {cleaned} expired items")
        
        return cleaned


# Singleton instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get the secrets manager singleton"""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager

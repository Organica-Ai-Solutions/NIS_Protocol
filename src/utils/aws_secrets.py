"""
AWS Secrets Manager Integration
Loads API keys from AWS Secrets Manager for production deployment
"""

import os
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Check if boto3 is available
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available - AWS Secrets Manager disabled")


class AWSSecretsManager:
    """Manages API keys from AWS Secrets Manager"""
    
    def __init__(self, region: str = "us-east-2"):
        self.region = region
        self.enabled = os.getenv("AWS_SECRETS_ENABLED", "false").lower() == "true"
        self.client = None
        self._cache: Dict[str, str] = {}
        
        if self.enabled and BOTO3_AVAILABLE:
            try:
                self.client = boto3.client('secretsmanager', region_name=region)
                logger.info(f"AWS Secrets Manager initialized (region: {region})")
            except Exception as e:
                logger.error(f"Failed to initialize AWS Secrets Manager: {e}")
                self.enabled = False
        elif self.enabled and not BOTO3_AVAILABLE:
            logger.error("AWS_SECRETS_ENABLED=true but boto3 not installed")
            self.enabled = False
    
    def get_secret(self, secret_name: str, cache: bool = True) -> Optional[str]:
        """
        Retrieve a secret from AWS Secrets Manager
        
        Args:
            secret_name: Name or ARN of the secret
            cache: Whether to cache the secret value
            
        Returns:
            Secret value as string, or None if not found
        """
        if not self.enabled or not self.client:
            return None
        
        # Check cache first
        if cache and secret_name in self._cache:
            return self._cache[secret_name]
        
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            
            # Parse secret value
            if 'SecretString' in response:
                secret_value = response['SecretString']
                
                # Try to parse as JSON (for complex secrets)
                try:
                    secret_dict = json.loads(secret_value)
                    # If it's a dict with a single key, return that value
                    if len(secret_dict) == 1:
                        secret_value = list(secret_dict.values())[0]
                except json.JSONDecodeError:
                    # Not JSON, use as-is
                    pass
                
                if cache:
                    self._cache[secret_name] = secret_value
                
                return secret_value
            else:
                logger.error(f"Secret {secret_name} has no SecretString")
                return None
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                logger.warning(f"Secret not found: {secret_name}")
            elif error_code == 'AccessDeniedException':
                logger.error(f"Access denied to secret: {secret_name}")
            else:
                logger.error(f"Error retrieving secret {secret_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving secret {secret_name}: {e}")
            return None
    
    def load_api_keys(self) -> Dict[str, str]:
        """
        Load all API keys from AWS Secrets Manager
        
        Returns:
            Dictionary of API key name -> value
        """
        if not self.enabled:
            return {}
        
        # AWS Secret ARNs from the documentation
        secrets_map = {
            "OPENAI_API_KEY": "arn:aws:secretsmanager:us-east-2:774518279463:secret:nis/openai-api-key-x0UEEi",
            "DEEPSEEK_API_KEY": "arn:aws:secretsmanager:us-east-2:774518279463:secret:nis/openai-api-key-x0UEEi",  # Same as OpenAI per docs
            "GOOGLE_API_KEY": "arn:aws:secretsmanager:us-east-2:774518279463:secret:nis/google-api-key-UpwtiO",
            "ANTHROPIC_API_KEY": "arn:aws:secretsmanager:us-east-2:774518279463:secret:nis/anthropic-api-key-00TnSn",
        }
        
        api_keys = {}
        for key_name, secret_arn in secrets_map.items():
            value = self.get_secret(secret_arn)
            if value:
                api_keys[key_name] = value
                logger.info(f"Loaded {key_name} from AWS Secrets Manager")
            else:
                logger.warning(f"Failed to load {key_name} from AWS Secrets Manager")
        
        return api_keys


# Global instance
_secrets_manager: Optional[AWSSecretsManager] = None


def get_secrets_manager() -> AWSSecretsManager:
    """Get or create the global AWS Secrets Manager instance"""
    global _secrets_manager
    if _secrets_manager is None:
        region = os.getenv("AWS_REGION", "us-east-2")
        _secrets_manager = AWSSecretsManager(region=region)
    return _secrets_manager


def get_api_key(key_name: str, fallback_env: bool = True) -> Optional[str]:
    """
    Get an API key from AWS Secrets Manager or environment variable
    
    Args:
        key_name: Name of the API key (e.g., "OPENAI_API_KEY")
        fallback_env: Whether to fall back to environment variable if AWS fails
        
    Returns:
        API key value or None
    """
    # Try AWS Secrets Manager first
    secrets_manager = get_secrets_manager()
    if secrets_manager.enabled:
        value = secrets_manager.get_secret(f"nis/{key_name.lower().replace('_', '-')}")
        if value:
            return value
    
    # Fall back to environment variable
    if fallback_env:
        return os.getenv(key_name)
    
    return None


def load_all_api_keys() -> Dict[str, str]:
    """
    Load all API keys from AWS Secrets Manager and environment variables
    
    Returns:
        Dictionary of API key name -> value
    """
    secrets_manager = get_secrets_manager()
    
    # Start with AWS Secrets Manager
    api_keys = secrets_manager.load_api_keys()
    
    # Fill in missing keys from environment variables
    env_keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "GOOGLE_API_KEY",
        "NVIDIA_API_KEY",
        "ELEVENLABS_API_KEY",
        "KIMI_K2_API_KEY",
    ]
    
    for key_name in env_keys:
        if key_name not in api_keys:
            env_value = os.getenv(key_name)
            if env_value:
                api_keys[key_name] = env_value
    
    return api_keys

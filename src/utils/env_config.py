"""
Environment Configuration Utility for NIS Protocol v3

This module provides centralized environment variable loading and configuration
management for the NIS Protocol system, particularly for LLM providers.
"""

import os
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class EnvironmentConfig:
    """Centralized environment configuration manager."""
    
    def __init__(self, env_path: Optional[str] = None):
        """Initialize environment configuration.
        
        Args:
            env_path: Optional path to .env file. If None, searches for .env in project root.
        """
        # Load .env file if exists
        if env_path is None:
            # Search for .env file in project root
            project_root = Path(__file__).parent.parent.parent
            env_path = project_root / ".env"
        
        if os.path.exists(env_path):
            load_dotenv(env_path)
            logger.info(f"Loaded environment configuration from {env_path}")
        else:
            logger.warning(f"No .env file found at {env_path}, using system environment variables only")
    
    @staticmethod
    def get_env(key: str, default: Any = None, required: bool = False) -> str:
        """Get environment variable with optional default and required validation.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            required: Whether the variable is required
            
        Returns:
            Environment variable value or default
            
        Raises:
            ValueError: If required variable is not found
        """
        value = os.getenv(key, default)
        
        if required and (value is None or value == ""):
            raise ValueError(f"Required environment variable {key} is not set")
        
        return value
    
    @staticmethod
    def get_bool(key: str, default: bool = False) -> bool:
        """Get boolean environment variable.
        
        Args:
            key: Environment variable name
            default: Default boolean value
            
        Returns:
            Boolean value
        """
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on', 'enabled')
    
    @staticmethod
    def get_int(key: str, default: int = 0) -> int:
        """Get integer environment variable.
        
        Args:
            key: Environment variable name
            default: Default integer value
            
        Returns:
            Integer value
        """
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            logger.warning(f"Invalid integer value for {key}, using default {default}")
            return default
    
    @staticmethod
    def get_float(key: str, default: float = 0.0) -> float:
        """Get float environment variable.
        
        Args:
            key: Environment variable name
            default: Default float value
            
        Returns:
            Float value
        """
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            logger.warning(f"Invalid float value for {key}, using default {default}")
            return default
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM provider configuration from environment variables.
        
        Returns:
            Dictionary containing LLM configuration
        """
        # Get API keys
        openai_key = self.get_env("OPENAI_API_KEY", "")
        anthropic_key = self.get_env("ANTHROPIC_API_KEY", "")
        deepseek_key = self.get_env("DEEPSEEK_API_KEY", "")
        bitnet_enabled = self.get_bool("BITNET_ENABLED", False)
        
        config = {
            "providers": {
                "openai": {
                    "enabled": self.get_bool("OPENAI_ENABLED", bool(openai_key)),
                    "api_key": openai_key,
                    "api_base": self.get_env("OPENAI_API_BASE", "https://api.openai.com/v1"),
                    "organization": self.get_env("OPENAI_ORGANIZATION", ""),
                    "models": {
                        "chat": {
                            "name": "gpt-4o",
                            "max_tokens": 4096,
                            "temperature": 0.7
                        },
                        "fast_chat": {
                            "name": "gpt-4o-mini",
                            "max_tokens": 2048,
                            "temperature": 0.5
                        },
                        "embedding": {
                            "name": "text-embedding-3-small",
                            "dimensions": 1536
                        }
                    }
                },
                "anthropic": {
                    "enabled": self.get_bool("ANTHROPIC_ENABLED", bool(anthropic_key)),
                    "api_key": anthropic_key,
                    "api_base": self.get_env("ANTHROPIC_API_BASE", "https://api.anthropic.com/v1"),
                    "version": self.get_env("ANTHROPIC_VERSION", "2023-06-01"),
                    "models": {
                        "chat": {
                            "name": "claude-3-5-sonnet-20241022",
                            "max_tokens": 4096,
                            "temperature": 0.7
                        }
                    }
                },
                "deepseek": {
                    "enabled": self.get_bool("DEEPSEEK_ENABLED", bool(deepseek_key)),
                    "api_key": deepseek_key,
                    "api_base": self.get_env("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
                    "models": {
                        "chat": {
                            "name": "deepseek-chat",
                            "max_tokens": 4096,
                            "temperature": 0.7
                        }
                    }
                },
                "bitnet": {
                    "enabled": self.get_bool("BITNET_ENABLED", False),
                    "model_path": self.get_env("BITNET_MODEL_PATH", "./models/bitnet/model.bin"),
                    "executable_path": self.get_env("BITNET_EXECUTABLE_PATH", "bitnet"),
                    "context_length": self.get_int("BITNET_CONTEXT_LENGTH", 4096),
                    "cpu_threads": self.get_int("BITNET_CPU_THREADS", 8),
                    "models": {
                        "chat": {
                            "name": "bitnet-b1.58-3b",
                            "max_tokens": 2048,
                            "temperature": 0.7
                        }
                    }
                }
            },
            "agent_llm_config": {
                "default_provider": self.get_env("DEFAULT_LLM_PROVIDER", None),
                "fallback_to_mock": self.get_bool("FALLBACK_TO_MOCK", True),
                "cognitive_functions": {
                    "consciousness": {
                        "primary_provider": self.get_env("CONSCIOUSNESS_PROVIDER", "anthropic"),
                        "fallback_providers": ["openai", "deepseek", "mock"],
                        "temperature": 0.5,
                        "max_tokens": 4096
                    },
                    "reasoning": {
                        "primary_provider": self.get_env("REASONING_PROVIDER", "anthropic"),
                        "fallback_providers": ["openai", "deepseek", "mock"],
                        "temperature": 0.3,
                        "max_tokens": 3072
                    },
                    "creativity": {
                        "primary_provider": self.get_env("CREATIVITY_PROVIDER", "openai"),
                        "fallback_providers": ["anthropic", "deepseek", "mock"],
                        "temperature": 0.8,
                        "max_tokens": 2048
                    },
                    "cultural": {
                        "primary_provider": self.get_env("CULTURAL_PROVIDER", "anthropic"),
                        "fallback_providers": ["openai", "deepseek", "mock"],
                        "temperature": 0.6,
                        "max_tokens": 3072
                    },
                    "archaeological": {
                        "primary_provider": self.get_env("ARCHAEOLOGICAL_PROVIDER", "anthropic"),
                        "fallback_providers": ["openai", "deepseek", "mock"],
                        "temperature": 0.4,
                        "max_tokens": 4096
                    },
                    "execution": {
                        "primary_provider": self.get_env("EXECUTION_PROVIDER", "bitnet"),
                        "fallback_providers": ["deepseek", "openai", "mock"],
                        "temperature": 0.2,
                        "max_tokens": 1024
                    },
                    "memory": {
                        "primary_provider": self.get_env("MEMORY_PROVIDER", "deepseek"),
                        "fallback_providers": ["anthropic", "openai", "mock"],
                        "temperature": 0.3,
                        "max_tokens": 4096
                    },
                    "perception": {
                        "primary_provider": self.get_env("PERCEPTION_PROVIDER", "openai"),
                        "fallback_providers": ["bitnet", "anthropic", "mock"],
                        "temperature": 0.4,
                        "max_tokens": 2048
                    }
                }
            },
            "cognitive_orchestra": {
                "enabled": self.get_bool("COGNITIVE_ORCHESTRA_ENABLED", True),
                "parallel_processing": self.get_bool("COGNITIVE_PARALLEL_PROCESSING", True),
                "max_concurrent_functions": self.get_int("COGNITIVE_MAX_CONCURRENT", 6),
                "harmony_threshold": self.get_float("COGNITIVE_HARMONY_THRESHOLD", 0.7)
            },
            "default_config": {
                "max_retries": self.get_int("LLM_MAX_RETRIES", 3),
                "retry_delay": self.get_int("LLM_RETRY_DELAY", 1),
                "timeout": self.get_int("LLM_TIMEOUT", 30),
                "cache_enabled": self.get_bool("LLM_CACHE_ENABLED", True),
                "cache_ttl": self.get_int("LLM_CACHE_TTL", 3600)
            },
            "monitoring": {
                "log_level": self.get_env("LOG_LEVEL", "INFO"),
                "track_usage": self.get_bool("TRACK_LLM_USAGE", True),
                "track_performance": self.get_bool("TRACK_PERFORMANCE", True),
                "alert_on_errors": self.get_bool("ALERT_ON_ERRORS", True),
                "cognitive_metrics": self.get_bool("COGNITIVE_METRICS", True)
            }
        }
        
        return config
    
    def get_infrastructure_config(self) -> Dict[str, Any]:
        """Get infrastructure configuration from environment variables.
        
        Returns:
            Dictionary containing infrastructure configuration
        """
        return {
            "redis": {
                "host": self.get_env("REDIS_HOST", "localhost"),
                "port": self.get_int("REDIS_PORT", 6379),
                "password": self.get_env("REDIS_PASSWORD", ""),
                "db": self.get_int("REDIS_DB", 0)
            },
            "kafka": {
                "bootstrap_servers": self.get_env("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
                "security_protocol": self.get_env("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
                "sasl_mechanism": self.get_env("KAFKA_SASL_MECHANISM", "PLAIN"),
                "sasl_username": self.get_env("KAFKA_SASL_USERNAME", ""),
                "sasl_password": self.get_env("KAFKA_SASL_PASSWORD", "")
            },
            "search": {
                "search_api_key": self.get_env("SEARCH_API_KEY", ""),
                "search_engine_id": self.get_env("SEARCH_ENGINE_ID", ""),
                "bing_search_api_key": self.get_env("BING_SEARCH_API_KEY", "")
            }
        }

# Global environment config instance
env_config = EnvironmentConfig() 
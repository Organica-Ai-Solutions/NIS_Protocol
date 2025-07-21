"""
NIS Protocol LLM Manager

This module provides centralized management of LLM providers and caching.
Enhanced to read configuration from environment variables.
"""

import json
import os
import sys
from typing import Dict, Any, Optional, Type, List
import aiohttp
import asyncio
from cachetools import TTLCache
import logging

from .base_llm_provider import BaseLLMProvider, LLMResponse, LLMMessage

# Import env_config with error handling for different execution contexts
try:
    from ..utils.env_config import env_config
except ImportError:
    try:
        from src.utils.env_config import env_config
    except ImportError:
        # Fallback for testing/standalone execution
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from utils.env_config import env_config

class LLMManager:
    """Manager for LLM providers and caching."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the LLM manager.
        
        Args:
            config_path: Path to LLM configuration file (deprecated - now uses environment variables)
        """
        self.logger = logging.getLogger("llm_manager")
        
        # Load configuration from environment variables
        self.config = env_config.get_llm_config()
        
        # If config_path is provided, try to merge with JSON config (for backward compatibility)
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    json_config = json.load(f)
                    self.logger.info(f"Merging JSON config from {config_path} with environment configuration")
                    self._merge_configs(json_config)
            except Exception as e:
                self.logger.warning(f"Failed to load JSON config from {config_path}: {e}")
        
        # Initialize provider registry with lazy loading
        self.provider_registry: Dict[str, Type[BaseLLMProvider]] = {}
        self._register_available_providers()
        
        # Initialize provider instances
        self.providers: Dict[str, BaseLLMProvider] = {}
        
        # Initialize cache if enabled
        if self.config["default_config"]["cache_enabled"]:
            self.cache = TTLCache(
                maxsize=1000,
                ttl=self.config["default_config"]["cache_ttl"]
            )
        else:
            self.cache = None
            
        self.logger.info("LLM Manager initialized with environment-based configuration")
    
    def _merge_configs(self, json_config: Dict[str, Any]):
        """Merge JSON configuration with environment configuration.
        
        Environment variables take precedence over JSON configuration.
        
        Args:
            json_config: Configuration loaded from JSON file
        """
        # Only merge if API keys are not set in environment
        for provider_name, provider_config in json_config.get("providers", {}).items():
            if provider_name in self.config["providers"]:
                env_provider = self.config["providers"][provider_name]
                
                # If API key is not set in environment, use JSON config
                if not env_provider.get("api_key") and provider_config.get("api_key"):
                    if provider_config["api_key"] not in ["YOUR_API_KEY_HERE", "YOUR_OPENAI_API_KEY", "YOUR_ANTHROPIC_API_KEY", "YOUR_DEEPSEEK_API_KEY"]:
                        env_provider["api_key"] = provider_config["api_key"]
                        env_provider["enabled"] = provider_config.get("enabled", False)
                        self.logger.info(f"Using API key from JSON config for {provider_name}")
    
    def _register_available_providers(self):
        """Register available LLM providers with lazy loading."""
        # Try to import each provider and register if available
        provider_modules = {
            "openai": "openai_provider.OpenAIProvider",
            "anthropic": "anthropic_provider.AnthropicProvider", 
            "deepseek": "deepseek_provider.DeepseekProvider",
            "bitnet": "bitnet_provider.BitNetProvider",
            "mock": "mock_provider.MockProvider"
        }
        
        for provider_name, module_path in provider_modules.items():
            try:
                module_name, class_name = module_path.split(".")
                
                # Try different import paths
                import_paths = [
                    f"llm.providers.{module_name}",
                    f"src.llm.providers.{module_name}",
                    f".providers.{module_name}"
                ]
                
                module = None
                for import_path in import_paths:
                    try:
                        if import_path.startswith("."):
                            # Skip relative imports for now
                            continue
                        module = __import__(import_path, fromlist=[class_name])
                        break
                    except ImportError:
                        continue
                
                if module is None:
                    raise ImportError(f"Could not import {module_name}")
                
                provider_class = getattr(module, class_name)
                self.provider_registry[provider_name] = provider_class
                self.logger.info(f"Registered LLM provider: {provider_name}")
            except ImportError as e:
                self.logger.warning(f"Could not import {provider_name} provider: {e}")
            except Exception as e:
                self.logger.error(f"Error registering {provider_name} provider: {e}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return list(self.provider_registry.keys())
    
    def is_provider_configured(self, provider_name: str) -> bool:
        """Check if a provider is properly configured with API key."""
        if provider_name not in self.config.get("providers", {}):
            return False
        
        provider_config = self.config["providers"][provider_name]
        
        # Check if provider is enabled
        if not provider_config.get("enabled", False):
            return False
        
        # Check for API key (not needed for local providers like BitNet or mock)
        if provider_name in ["bitnet", "mock"]:
            # For local providers, just check if enabled
            return provider_config.get("enabled", False)
        else:
            # For API providers, check for API key and enabled status
            api_key = provider_config.get("api_key", "")
            return (api_key and 
                   api_key not in ["YOUR_API_KEY_HERE", "your_openai_api_key_here", "your_anthropic_api_key_here", "your_deepseek_api_key_here"] and 
                   provider_config.get("enabled", False))
    
    def get_provider(self, provider_name: Optional[str] = None) -> BaseLLMProvider:
        """Get or initialize a provider instance.
        
        Args:
            provider_name: Name of the provider. If None, uses default or fallback.
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If no provider is available
        """
        # Determine which provider to use
        actual_provider = self._resolve_provider(provider_name)
        
        if actual_provider not in self.providers:
            if actual_provider not in self.provider_registry:
                raise ValueError(f"Unknown provider: {actual_provider}")
            
            if actual_provider == "mock":
                # Mock provider doesn't need real config
                provider_config = {"model": "mock-provider", "max_tokens": 2048, "temperature": 0.7}
            else:
                provider_config = self.config["providers"][actual_provider]
                
            self.providers[actual_provider] = self.provider_registry[actual_provider](provider_config)
        
        return self.providers[actual_provider]
    
    def _resolve_provider(self, requested_provider: Optional[str] = None) -> str:
        """Resolve which provider to actually use.
        
        Args:
            requested_provider: The requested provider name
            
        Returns:
            The actual provider name to use
        """
        # If specific provider requested and configured, use it
        if requested_provider and self.is_provider_configured(requested_provider):
            return requested_provider
        
        # Try default provider from environment
        default_provider = self.config.get("agent_llm_config", {}).get("default_provider")
        if default_provider and self.is_provider_configured(default_provider):
            return default_provider
        
        # Try to find any configured provider
        for provider_name in self.provider_registry.keys():
            if provider_name != "mock" and self.is_provider_configured(provider_name):
                return provider_name
        
        # Fall back to mock if enabled
        if self.config.get("agent_llm_config", {}).get("fallback_to_mock", True):
            self.logger.warning("No LLM providers configured, falling back to mock provider")
            return "mock"
        
        raise ValueError("No LLM providers configured and mock fallback disabled")
    
    def get_configured_providers(self) -> List[str]:
        """Get list of properly configured provider names."""
        configured = []
        for provider_name in self.provider_registry.keys():
            if provider_name != "mock" and self.is_provider_configured(provider_name):
                configured.append(provider_name)
        return configured
    
    def get_provider_for_cognitive_function(self, function: str) -> str:
        """Get the best provider for a specific cognitive function.
        
        Args:
            function: The cognitive function name
            
        Returns:
            Provider name to use for this function
        """
        cognitive_config = self.config.get("agent_llm_config", {}).get("cognitive_functions", {})
        function_config = cognitive_config.get(function, {})
        
        # Try primary provider
        primary_provider = function_config.get("primary_provider")
        if primary_provider and self.is_provider_configured(primary_provider):
            return primary_provider
        
        # Try fallback providers
        fallback_providers = function_config.get("fallback_providers", [])
        for provider in fallback_providers:
            if provider != "mock" and self.is_provider_configured(provider):
                return provider
        
        # Use general resolution
        return self._resolve_provider()
    
    def get_cognitive_config(self, function: str) -> Dict[str, Any]:
        """Get configuration for a specific cognitive function.
        
        Args:
            function: The cognitive function name
            
        Returns:
            Configuration dictionary for the function
        """
        cognitive_config = self.config.get("agent_llm_config", {}).get("cognitive_functions", {})
        return cognitive_config.get(function, {})

    async def generate_with_function(
        self,
        messages: List[LLMMessage],
        function: str,
        **kwargs
    ) -> LLMResponse:
        """Generate response using the optimal provider for a cognitive function.
        
        Args:
            messages: List of messages for the conversation
            function: The cognitive function to use
            **kwargs: Additional parameters
            
        Returns:
            LLM response
        """
        provider_name = self.get_provider_for_cognitive_function(function)
        provider = self.get_provider(provider_name)
        
        # Get function-specific configuration
        function_config = self.get_cognitive_config(function)
        
        # Override parameters with function-specific settings
        temperature = kwargs.get("temperature", function_config.get("temperature", 0.7))
        max_tokens = kwargs.get("max_tokens", function_config.get("max_tokens", 2048))
        
        return await provider.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    async def generate_with_cache(
        self,
        provider: BaseLLMProvider,
        messages: List[LLMMessage],
        cache_key: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response with caching if enabled.
        
        Args:
            provider: LLM provider to use
            messages: Messages for generation
            cache_key: Optional cache key (default: hash of messages)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        if not cache_key:
            cache_key = str(hash(tuple(
                (msg.role.value, msg.content, msg.name)
                for msg in messages
            )))
        
        if self.cache is not None:
            cached_response = self.cache.get(cache_key)
            if cached_response:
                self.logger.debug(f"Cache hit for key: {cache_key}")
                return cached_response
        
        response = await provider.generate(messages, **kwargs)
        
        if self.cache is not None:
            self.cache[cache_key] = response
            self.logger.debug(f"Cached response for key: {cache_key}")
        
        return response
    
    async def close(self):
        """Close all provider sessions."""
        for provider in self.providers.values():
            await provider.close() 
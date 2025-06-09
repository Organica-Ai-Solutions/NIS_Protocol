"""
NIS Protocol LLM Manager

This module provides centralized management of LLM providers and caching.
"""

import json
import os
from typing import Dict, Any, Optional, Type, List
import aiohttp
import asyncio
from cachetools import TTLCache
import logging

from .base_llm_provider import BaseLLMProvider, LLMResponse, LLMMessage

class LLMManager:
    """Manager for LLM providers and caching."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the LLM manager.
        
        Args:
            config_path: Path to LLM configuration file
        """
        self.logger = logging.getLogger("llm_manager")
        
        # Load configuration
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config",
                "llm_config.json"
            )
        
        with open(config_path) as f:
            self.config = json.load(f)
        
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
        
        # Check for API key (not needed for local providers like BitNet or mock)
        if provider_name in ["bitnet", "mock"]:
            # For local providers, just check if enabled
            return provider_config.get("enabled", False)
        else:
            # For API providers, check for API key and enabled status
            api_key = provider_config.get("api_key", "")
            return (api_key and 
                   api_key not in ["YOUR_API_KEY_HERE", "YOUR_OPENAI_API_KEY", "YOUR_ANTHROPIC_API_KEY", "YOUR_DEEPSEEK_API_KEY"] and 
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
        
        # Try default provider
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
    
    def get_agent_llm(self, agent_type: str) -> BaseLLMProvider:
        """Get LLM provider for an agent type.
        
        Args:
            agent_type: Type of agent (e.g., "perception_agent")
            
        Returns:
            Configured LLM provider for the agent
            
        Raises:
            ValueError: If agent type not found in config
        """
        if agent_type not in self.config["agent_llm_config"]:
            raise ValueError(f"No LLM configuration for agent type: {agent_type}")
        
        agent_config = self.config["agent_llm_config"][agent_type]
        requested_provider = agent_config.get("provider")
        
        # Pass the requested provider, which may be None (will trigger fallback logic)
        return self.get_provider(requested_provider)
    
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
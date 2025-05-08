"""
NIS Protocol LLM Manager

This module provides centralized management of LLM providers and caching.
"""

import json
import os
from typing import Dict, Any, Optional, Type
import aiohttp
import asyncio
from cachetools import TTLCache
import logging

from .base_llm_provider import BaseLLMProvider, LLMResponse, LLMMessage
from .providers.deepseek_provider import DeepseekProvider

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
        
        # Initialize provider registry
        self.provider_registry: Dict[str, Type[BaseLLMProvider]] = {
            "deepseek": DeepseekProvider
        }
        
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
    
    def get_provider(self, provider_name: str) -> BaseLLMProvider:
        """Get or initialize a provider instance.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If provider not found in registry
        """
        if provider_name not in self.providers:
            if provider_name not in self.provider_registry:
                raise ValueError(f"Unknown provider: {provider_name}")
            
            provider_config = self.config["providers"][provider_name]
            self.providers[provider_name] = self.provider_registry[provider_name](provider_config)
        
        return self.providers[provider_name]
    
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
        return self.get_provider(agent_config["provider"])
    
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
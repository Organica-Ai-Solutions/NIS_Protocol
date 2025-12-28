#!/usr/bin/env python3
"""
Multi-Provider LLM Strategy for NIS Protocol
Ensures no single-provider dependency with rotation, fallback, and load balancing

Copyright 2025 Organica AI Solutions
Licensed under Apache License 2.0
"""

import logging
import random
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ProviderStats:
    """Track statistics for each LLM provider."""
    name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency: float = 0.0
    last_failure_time: Optional[float] = None
    consecutive_failures: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls
    
    @property
    def avg_latency(self) -> float:
        """Calculate average latency."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_latency / self.successful_calls
    
    @property
    def is_healthy(self) -> bool:
        """Check if provider is healthy."""
        # Circuit breaker: if 3+ consecutive failures, mark unhealthy for 60s
        if self.consecutive_failures >= 3:
            if self.last_failure_time and (time.time() - self.last_failure_time) < 60:
                return False
        return True


class MultiProviderStrategy:
    """
    Multi-provider LLM strategy with rotation, fallback, and load balancing.
    
    Ensures no single-provider dependency by:
    1. Rotating between all available providers
    2. Automatic fallback on provider failure
    3. Load balancing based on success rates
    4. Circuit breaker for unhealthy providers
    5. Provider health monitoring
    """
    
    def __init__(self, llm_provider):
        """Initialize multi-provider strategy."""
        self.llm_provider = llm_provider
        
        # Provider priority order (can be customized)
        self.provider_priority = [
            "anthropic",   # Claude Sonnet 4 - best for planning
            "openai",      # GPT-4o - reliable
            "google",      # Gemini 2.5 Flash - fast
            "deepseek",    # DeepSeek V3 - good reasoning
            "kimi",        # Kimi K2 - long context
            "nvidia",      # Llama 3.1 - open source
            "bitnet"       # Local fallback
        ]
        
        # Track stats for each provider
        self.stats: Dict[str, ProviderStats] = {
            provider: ProviderStats(name=provider)
            for provider in self.provider_priority
        }
        
        # Current provider index for round-robin
        self.current_index = 0
        
        logger.info(f"🔄 Multi-provider strategy initialized with {len(self.provider_priority)} providers")
    
    def get_available_providers(self) -> List[str]:
        """Get list of providers with valid API keys."""
        available = []
        
        if not self.llm_provider:
            return available
        
        for provider in self.provider_priority:
            api_key = self.llm_provider.api_keys.get(provider, "")
            if api_key and api_key != "":
                # Check if provider is healthy
                if self.stats[provider].is_healthy:
                    available.append(provider)
        
        return available
    
    def select_provider(self, strategy: str = "round_robin") -> Optional[str]:
        """
        Select next provider based on strategy.
        
        Args:
            strategy: Selection strategy
                - "round_robin": Rotate through providers
                - "best_performance": Choose provider with best success rate
                - "random": Random selection
                - "priority": Use priority order
        
        Returns:
            Provider name or None if no providers available
        """
        available = self.get_available_providers()
        
        if not available:
            logger.warning("⚠️ No LLM providers available!")
            return None
        
        if strategy == "round_robin":
            # Round-robin through available providers
            provider = available[self.current_index % len(available)]
            self.current_index = (self.current_index + 1) % len(available)
            return provider
        
        elif strategy == "best_performance":
            # Select provider with best success rate
            best_provider = max(
                available,
                key=lambda p: (
                    self.stats[p].success_rate,
                    -self.stats[p].avg_latency
                )
            )
            return best_provider
        
        elif strategy == "random":
            # Random selection
            return random.choice(available)
        
        elif strategy == "priority":
            # Use priority order
            for provider in self.provider_priority:
                if provider in available:
                    return provider
            return None
        
        else:
            logger.warning(f"Unknown strategy: {strategy}, using round_robin")
            return self.select_provider("round_robin")
    
    async def call_with_fallback(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        strategy: str = "round_robin"
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Call LLM with automatic fallback to other providers.
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            strategy: Provider selection strategy
        
        Returns:
            Tuple of (response, provider_used)
        """
        available = self.get_available_providers()
        
        if not available:
            logger.error("❌ No LLM providers available for fallback")
            return None, "none"
        
        # Try providers in order until one succeeds
        attempted_providers = []
        
        for attempt in range(len(available)):
            provider = self.select_provider(strategy)
            
            if not provider or provider in attempted_providers:
                continue
            
            attempted_providers.append(provider)
            
            logger.info(f"🔄 Attempting provider: {provider} (attempt {attempt + 1}/{len(available)})")
            
            start_time = time.time()
            
            try:
                # Call LLM provider
                response = await self.llm_provider.generate_response(
                    messages=messages,
                    provider=provider,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Success!
                latency = time.time() - start_time
                self._record_success(provider, latency)
                
                logger.info(f"✅ Provider {provider} succeeded in {latency:.2f}s")
                return response, provider
                
            except Exception as e:
                # Failure - try next provider
                self._record_failure(provider)
                logger.warning(f"⚠️ Provider {provider} failed: {e}")
                continue
        
        # All providers failed
        logger.error(f"❌ All {len(attempted_providers)} providers failed")
        return None, "all_failed"
    
    def _record_success(self, provider: str, latency: float):
        """Record successful provider call."""
        stats = self.stats[provider]
        stats.total_calls += 1
        stats.successful_calls += 1
        stats.total_latency += latency
        stats.consecutive_failures = 0
    
    def _record_failure(self, provider: str):
        """Record failed provider call."""
        stats = self.stats[provider]
        stats.total_calls += 1
        stats.failed_calls += 1
        stats.consecutive_failures += 1
        stats.last_failure_time = time.time()
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of provider statistics."""
        return {
            "providers": {
                name: {
                    "total_calls": stats.total_calls,
                    "successful_calls": stats.successful_calls,
                    "failed_calls": stats.failed_calls,
                    "success_rate": f"{stats.success_rate * 100:.1f}%",
                    "avg_latency": f"{stats.avg_latency:.2f}s",
                    "is_healthy": stats.is_healthy,
                    "consecutive_failures": stats.consecutive_failures
                }
                for name, stats in self.stats.items()
                if stats.total_calls > 0
            },
            "total_calls": sum(s.total_calls for s in self.stats.values()),
            "available_providers": self.get_available_providers()
        }


def get_multi_provider_strategy(llm_provider) -> MultiProviderStrategy:
    """Get or create multi-provider strategy instance."""
    return MultiProviderStrategy(llm_provider=llm_provider)

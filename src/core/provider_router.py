#!/usr/bin/env python3
"""
🎯 NIS Protocol v3.2 - Dynamic Provider Router
Intelligent routing of requests to optimal AI providers based on capabilities, cost, and performance

Features:
- YAML-based provider registry
- Capability-based routing  
- Cost optimization
- Failover & fallback logic
- Performance monitoring
- Environment-specific overrides
"""

import os
import yaml
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import concurrent.futures

logger = logging.getLogger("provider_router")

class RoutingStrategy(Enum):
    """Provider selection strategies"""
    QUALITY_FIRST = "quality_first"
    COST_PERFORMANCE_BALANCE = "cost_performance_balance"
    SPEED_FIRST = "speed_first"
    DOMAIN_EXPERTISE = "domain_expertise"
    BALANCED = "balanced"

@dataclass
class ProviderMetrics:
    """Runtime metrics for provider performance"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    total_cost: float = 0.0
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    consecutive_failures: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def is_healthy(self) -> bool:
        return self.consecutive_failures < 3 and self.success_rate > 0.8

@dataclass
class ProviderConfig:
    """Provider configuration from registry"""
    provider: str
    model: str
    capabilities: List[str]
    max_tokens: int
    cost_per_million_tokens: float
    avg_latency_ms: float
    quality_score: float
    availability: float
    features: List[str]

@dataclass
class RoutingRequest:
    """Request for provider routing"""
    task_type: str
    requested_provider: Optional[str] = None
    requested_model: Optional[str] = None
    strategy: Optional[RoutingStrategy] = None
    max_cost: Optional[float] = None
    max_latency: Optional[float] = None
    required_features: List[str] = field(default_factory=list)

@dataclass
class RoutingResult:
    """Result of provider routing"""
    provider: str
    model: str
    config: ProviderConfig
    reason: str
    fallback_used: bool = False
    estimated_cost: float = 0.0
    estimated_latency: float = 0.0

class ProviderRouter:
    """
    🎯 Dynamic Provider Router for NIS Protocol
    
    Intelligently routes requests to optimal providers based on:
    - Task capabilities
    - Cost optimization  
    - Performance metrics
    - Availability & health
    - Environment settings
    """
    
    def __init__(self, registry_path: str = "configs/provider_registry.yaml"):
        self.registry_path = registry_path
        self.registry = self._load_registry()
        self.metrics: Dict[str, ProviderMetrics] = defaultdict(ProviderMetrics)
        self.environment = os.getenv("NIS_ENVIRONMENT", "development")
        
        # Performance tracking
        self.request_history = deque(maxlen=1000)
        self.last_registry_reload = time.time()
        
        logger.info(f"🎯 Provider Router initialized - Environment: {self.environment}")
        logger.info(f"📊 Loaded {len(self.registry['providers'])} providers")
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load provider registry from YAML"""
        try:
            if not os.path.exists(self.registry_path):
                logger.warning(f"Provider registry not found: {self.registry_path}")
                return self._get_fallback_registry()
                
            with open(self.registry_path, 'r') as f:
                registry = yaml.safe_load(f)
                
            logger.info(f"✅ Loaded provider registry v{registry.get('version', 'unknown')}")
            return registry
            
        except Exception as e:
            logger.error(f"Failed to load provider registry: {e}")
            return self._get_fallback_registry()
    
    def _get_fallback_registry(self) -> Dict[str, Any]:
        """Fallback registry if YAML loading fails"""
        return {
            "providers": {
                "openai": {
                    "gpt-4o": {
                        "capabilities": ["default", "reasoning"],
                        "max_tokens": 128000,
                        "cost_per_million_tokens": 5.0,
                        "avg_latency_ms": 1500,
                        "quality_score": 9.5,
                        "availability": 99.9,
                        "features": ["function_calling"]
                    }
                }
            },
            "task_routing": {
                "default": {
                    "primary": ["openai/gpt-4o"],
                    "fallback": [],
                    "strategy": "balanced"
                }
            }
        }
    
    def reload_registry(self):
        """Reload provider registry (for live updates)"""
        self.registry = self._load_registry()
        self.last_registry_reload = time.time()
        logger.info("🔄 Provider registry reloaded")
    
    def route_request(self, request: RoutingRequest) -> RoutingResult:
        """
        Route a request to the optimal provider
        
        Args:
            request: RoutingRequest with task type and preferences
            
        Returns:
            RoutingResult with selected provider and reasoning
        """
        start_time = time.time()
        
        # Check for explicit provider override
        if request.requested_provider and request.requested_model:
            if self._is_provider_available(request.requested_provider, request.requested_model):
                config = self._get_provider_config(request.requested_provider, request.requested_model)
                if config:
                    return RoutingResult(
                        provider=request.requested_provider,
                        model=request.requested_model,
                        config=config,
                        reason="Explicit provider request",
                        estimated_cost=self._estimate_cost(config, 1000),
                        estimated_latency=config.avg_latency_ms
                    )
        
        # Check environment overrides
        env_override = self._check_environment_override(request.task_type)
        if env_override:
            return env_override
        
        # Get task routing configuration
        task_config = self.registry.get("task_routing", {}).get(request.task_type)
        if not task_config:
            task_config = self.registry.get("task_routing", {}).get("default", {})
        
        # Try primary providers first
        primary_providers = task_config.get("primary", [])
        for provider_model in primary_providers:
            result = self._try_provider(provider_model, request, task_config)
            if result:
                self._record_routing_decision(result, time.time() - start_time)
                return result
        
        # Try fallback providers
        fallback_providers = task_config.get("fallback", [])
        for provider_model in fallback_providers:
            result = self._try_provider(provider_model, request, task_config)
            if result:
                result.fallback_used = True
                result.reason += " (fallback)"
                self._record_routing_decision(result, time.time() - start_time)
                return result
        
        # Last resort: find any available provider with required capabilities
        return self._emergency_fallback(request)
    
    def _try_provider(self, provider_model: str, request: RoutingRequest, task_config: Dict) -> Optional[RoutingResult]:
        """Try to route to a specific provider/model"""
        try:
            provider, model = provider_model.split("/")
            
            # Check availability and health
            if not self._is_provider_healthy(provider, model):
                return None
            
            config = self._get_provider_config(provider, model)
            if not config:
                return None
            
            # Check capability match
            if not self._has_required_capabilities(config, request):
                return None
            
            # Check cost constraints
            estimated_cost = self._estimate_cost(config, 1000)  # Estimate for 1k tokens
            if request.max_cost and estimated_cost > request.max_cost:
                return None
            
            # Check latency constraints  
            if request.max_latency and config.avg_latency_ms > request.max_latency:
                return None
            
            return RoutingResult(
                provider=provider,
                model=model,
                config=config,
                reason=f"Optimal match for {request.task_type}",
                estimated_cost=estimated_cost,
                estimated_latency=config.avg_latency_ms
            )
            
        except Exception as e:
            logger.warning(f"Failed to try provider {provider_model}: {e}")
            return None
    
    def _is_provider_healthy(self, provider: str, model: str) -> bool:
        """Check if provider is healthy based on recent metrics"""
        key = f"{provider}/{model}"
        metrics = self.metrics.get(key)
        
        if not metrics:
            return True  # No metrics yet, assume healthy
        
        return metrics.is_healthy
    
    def _has_required_capabilities(self, config: ProviderConfig, request: RoutingRequest) -> bool:
        """Check if provider has required capabilities"""
        # Check task capability
        if request.task_type not in config.capabilities:
            return False
        
        # Check required features
        for feature in request.required_features:
            if feature not in config.features:
                return False
        
        return True
    
    def _estimate_cost(self, config: ProviderConfig, estimated_tokens: int) -> float:
        """Estimate cost for request"""
        return (estimated_tokens / 1_000_000) * config.cost_per_million_tokens
    
    def _get_provider_config(self, provider: str, model: str) -> Optional[ProviderConfig]:
        """Get provider configuration"""
        try:
            provider_data = self.registry["providers"][provider][model]
            return ProviderConfig(
                provider=provider,
                model=model,
                capabilities=provider_data["capabilities"],
                max_tokens=provider_data["max_tokens"],
                cost_per_million_tokens=provider_data["cost_per_million_tokens"],
                avg_latency_ms=provider_data["avg_latency_ms"],
                quality_score=provider_data["quality_score"],
                availability=provider_data["availability"],
                features=provider_data.get("features", [])
            )
        except KeyError:
            return None
    
    def _is_provider_available(self, provider: str, model: str) -> bool:
        """Check if provider/model is available in registry"""
        return (provider in self.registry.get("providers", {}) and 
                model in self.registry.get("providers", {}).get(provider, {}))
    
    def _check_environment_override(self, task_type: str) -> Optional[RoutingResult]:
        """Check for environment-specific overrides"""
        env_config = self.registry.get("environments", {}).get(self.environment, {})
        
        # Check for forced provider in testing
        forced_provider = env_config.get("force_provider")
        if forced_provider:
            try:
                provider, model = forced_provider.split("/")
                config = self._get_provider_config(provider, model)
                if config:
                    return RoutingResult(
                        provider=provider,
                        model=model,
                        config=config,
                        reason=f"Environment override ({self.environment})",
                        estimated_cost=self._estimate_cost(config, 1000),
                        estimated_latency=config.avg_latency_ms
                    )
            except Exception:
                pass
        
        return None
    
    def _emergency_fallback(self, request: RoutingRequest) -> RoutingResult:
        """Emergency fallback when no providers match"""
        logger.warning(f"Emergency fallback for task: {request.task_type}")
        
        # Try to find any working provider
        for provider_name, models in self.registry.get("providers", {}).items():
            for model_name, model_data in models.items():
                if self._is_provider_healthy(provider_name, model_name):
                    config = self._get_provider_config(provider_name, model_name)
                    if config:
                        return RoutingResult(
                            provider=provider_name,
                            model=model_name,
                            config=config,
                            reason="Emergency fallback",
                            fallback_used=True,
                            estimated_cost=self._estimate_cost(config, 1000),
                            estimated_latency=config.avg_latency_ms
                        )
        
        # Absolute fallback
        raise Exception(f"No available providers for task: {request.task_type}")
    
    def record_request_result(self, provider: str, model: str, success: bool, 
                            latency_ms: float, cost: float = 0.0):
        """Record the result of a request for metrics tracking"""
        key = f"{provider}/{model}"
        metrics = self.metrics[key]
        
        metrics.total_requests += 1
        if success:
            metrics.successful_requests += 1
            metrics.last_success = time.time()
            metrics.consecutive_failures = 0
        else:
            metrics.failed_requests += 1
            metrics.last_failure = time.time()
            metrics.consecutive_failures += 1
        
        # Update rolling average latency
        if metrics.avg_latency_ms == 0:
            metrics.avg_latency_ms = latency_ms
        else:
            metrics.avg_latency_ms = (metrics.avg_latency_ms * 0.9) + (latency_ms * 0.1)
        
        metrics.total_cost += cost
        
        logger.debug(f"📊 Recorded result for {key}: success={success}, latency={latency_ms}ms")
    
    def _record_routing_decision(self, result: RoutingResult, decision_time_ms: float):
        """Record routing decision for analysis"""
        self.request_history.append({
            "timestamp": time.time(),
            "provider": result.provider,
            "model": result.model,
            "reason": result.reason,
            "fallback_used": result.fallback_used,
            "decision_time_ms": decision_time_ms
        })
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get current provider statistics"""
        stats = {}
        for key, metrics in self.metrics.items():
            stats[key] = {
                "total_requests": metrics.total_requests,
                "success_rate": metrics.success_rate,
                "avg_latency_ms": metrics.avg_latency_ms,
                "total_cost": metrics.total_cost,
                "is_healthy": metrics.is_healthy,
                "consecutive_failures": metrics.consecutive_failures
            }
        return stats
    
    def get_task_recommendations(self, task_type: str) -> List[str]:
        """Get recommended providers for a task type"""
        task_config = self.registry.get("task_routing", {}).get(task_type, {})
        recommendations = []
        recommendations.extend(task_config.get("primary", []))
        recommendations.extend(task_config.get("fallback", []))
        return recommendations

# Convenience functions for easy integration

def create_provider_router() -> ProviderRouter:
    """Create a provider router instance"""
    return ProviderRouter()

def route_to_provider(task_type: str, requested_provider: str = None, 
                     requested_model: str = None) -> RoutingResult:
    """Quick routing function"""
    router = create_provider_router()
    request = RoutingRequest(
        task_type=task_type,
        requested_provider=requested_provider,
        requested_model=requested_model
    )
    return router.route_request(request)
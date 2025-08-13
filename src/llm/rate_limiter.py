#!/usr/bin/env python3
"""
🎯 NIS Protocol Rate Limiting & Request Batching
Advanced rate limiting for LLM API calls with intelligent batching

Features:
- Provider-specific rate limits
- Sliding window rate limiting
- Request queuing and batching
- Cost-aware throttling
- Burst handling with backoff
- Request prioritization
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import threading

logger = logging.getLogger(__name__)

class RequestPriority(Enum):
    """Request priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class RateLimitConfig:
    """Rate limiting configuration for a provider"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    max_concurrent: int = 5
    max_queue_size: int = 100
    backoff_factor: float = 1.5
    max_retry_delay: float = 60.0
    cost_per_request: float = 0.01

@dataclass
class QueuedRequest:
    """Queued API request"""
    request_id: str
    provider: str
    model: str
    request_func: Callable[[], Awaitable[Any]]
    priority: RequestPriority
    created_at: float
    cost_estimate: float
    retry_count: int = 0
    
    def __post_init__(self):
        if not hasattr(self, 'created_at'):
            self.created_at = time.time()

@dataclass
class RateLimitStats:
    """Rate limiting statistics"""
    total_requests: int = 0
    queued_requests: int = 0
    rejected_requests: int = 0
    total_wait_time: float = 0.0
    avg_queue_time: float = 0.0
    peak_queue_size: int = 0

class ProviderRateLimiter:
    """
    🚦 Provider-specific rate limiter with intelligent queuing
    
    Features:
    - Sliding window rate limiting
    - Request prioritization
    - Automatic backoff
    - Cost tracking
    - Queue management
    """
    
    def __init__(self, provider: str, config: RateLimitConfig):
        self.provider = provider
        self.config = config
        
        # Sliding window tracking
        self.request_times = deque()
        self.hourly_requests = deque()
        
        # Concurrency control
        self.active_requests = 0
        self.max_concurrent = config.max_concurrent
        
        # Request queue
        self.request_queue: List[QueuedRequest] = []
        self.processing_queue = False
        
        # Statistics
        self.stats = RateLimitStats()
        
        # Locks for thread safety
        self._queue_lock = threading.RLock()
        self._stats_lock = threading.RLock()
        
        # Cost tracking
        self.hourly_cost = 0.0
        self.daily_cost = 0.0
        self.cost_limit_per_hour = 10.0  # $10/hour default
        
        logger.info(f"🚦 Rate limiter initialized for {provider}: {config.requests_per_minute}/min")
    
    def _cleanup_old_requests(self):
        """Remove old requests from sliding windows"""
        current_time = time.time()
        
        # Clean minute window
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()
        
        # Clean hour window
        while self.hourly_requests and current_time - self.hourly_requests[0] > 3600:
            old_time = self.hourly_requests.popleft()
            # Also remove from hourly cost tracking
            # (simplified - in production, track costs with timestamps)
    
    def _can_make_request(self) -> tuple[bool, float]:
        """Check if a request can be made now"""
        self._cleanup_old_requests()
        current_time = time.time()
        
        # Check concurrent requests
        if self.active_requests >= self.max_concurrent:
            return False, 1.0  # Wait 1 second for concurrent limit
        
        # Check minute rate limit
        if len(self.request_times) >= self.config.requests_per_minute:
            oldest_request = self.request_times[0]
            wait_time = 60 - (current_time - oldest_request)
            if wait_time > 0:
                return False, wait_time
        
        # Check hourly rate limit
        if len(self.hourly_requests) >= self.config.requests_per_hour:
            oldest_request = self.hourly_requests[0]
            wait_time = 3600 - (current_time - oldest_request)
            if wait_time > 0:
                return False, wait_time
        
        # Check cost limits
        if self.hourly_cost >= self.cost_limit_per_hour:
            return False, 300  # Wait 5 minutes if cost limit hit
        
        return True, 0.0
    
    def _record_request(self, cost: float = 0.0):
        """Record a successful request"""
        current_time = time.time()
        self.request_times.append(current_time)
        self.hourly_requests.append(current_time)
        
        # Update cost tracking
        self.hourly_cost += cost
        self.daily_cost += cost
        
        # Update stats
        with self._stats_lock:
            self.stats.total_requests += 1
    
    async def execute_request(self, 
                            request_id: str,
                            request_func: Callable[[], Awaitable[Any]],
                            priority: RequestPriority = RequestPriority.NORMAL,
                            cost_estimate: float = None,
                            model: str = "unknown") -> Any:
        """Execute request with rate limiting"""
        
        if cost_estimate is None:
            cost_estimate = self.config.cost_per_request
        
        # Create queued request
        queued_request = QueuedRequest(
            request_id=request_id,
            provider=self.provider,
            model=model,
            request_func=request_func,
            priority=priority,
            created_at=time.time(),
            cost_estimate=cost_estimate
        )
        
        # Check if we can execute immediately
        can_execute, wait_time = self._can_make_request()
        
        if can_execute:
            return await self._execute_immediate(queued_request)
        else:
            return await self._queue_request(queued_request)
    
    async def _execute_immediate(self, request: QueuedRequest) -> Any:
        """Execute request immediately"""
        self.active_requests += 1
        start_time = time.time()
        
        try:
            result = await request.request_func()
            
            # Record successful request
            self._record_request(request.cost_estimate)
            
            # Update wait time stats
            with self._stats_lock:
                wait_time = start_time - request.created_at
                self.stats.total_wait_time += wait_time
                if self.stats.total_requests > 0:
                    self.stats.avg_queue_time = self.stats.total_wait_time / self.stats.total_requests
            
            return result
            
        except Exception as e:
            logger.error(f"Request {request.request_id} failed: {e}")
            raise
        finally:
            self.active_requests -= 1
    
    async def _queue_request(self, request: QueuedRequest) -> Any:
        """Queue request for later execution"""
        with self._queue_lock:
            # Check queue size limit
            if len(self.request_queue) >= self.config.max_queue_size:
                with self._stats_lock:
                    self.stats.rejected_requests += 1
                raise Exception(f"Request queue full for provider {self.provider}")
            
            # Add to queue (sorted by priority)
            self.request_queue.append(request)
            self.request_queue.sort(key=lambda r: (-r.priority.value, r.created_at))
            
            # Update queue stats
            with self._stats_lock:
                self.stats.queued_requests += 1
                queue_size = len(self.request_queue)
                if queue_size > self.stats.peak_queue_size:
                    self.stats.peak_queue_size = queue_size
        
        # Start queue processing if not already running
        if not self.processing_queue:
            asyncio.create_task(self._process_queue())
        
        # Wait for request to be processed
        return await self._wait_for_request_completion(request.request_id)
    
    async def _process_queue(self):
        """Process queued requests"""
        if self.processing_queue:
            return
        
        self.processing_queue = True
        
        try:
            while True:
                with self._queue_lock:
                    if not self.request_queue:
                        break
                    
                    # Get next request
                    request = self.request_queue.pop(0)
                
                # Check if we can process this request
                can_execute, wait_time = self._can_make_request()
                
                if can_execute:
                    # Execute the request
                    try:
                        await self._execute_immediate(request)
                    except Exception as e:
                        # Handle retry logic
                        if request.retry_count < 3:
                            request.retry_count += 1
                            # Add back to queue with exponential backoff
                            await asyncio.sleep(self.config.backoff_factor ** request.retry_count)
                            with self._queue_lock:
                                self.request_queue.append(request)
                                self.request_queue.sort(key=lambda r: (-r.priority.value, r.created_at))
                        else:
                            logger.error(f"Request {request.request_id} failed after retries: {e}")
                else:
                    # Put request back and wait
                    with self._queue_lock:
                        self.request_queue.insert(0, request)
                    await asyncio.sleep(min(wait_time, 10.0))  # Cap wait time
        
        finally:
            self.processing_queue = False
    
    async def _wait_for_request_completion(self, request_id: str) -> Any:
        """Wait for a specific request to complete"""
        # This is a simplified implementation
        # In production, you'd use proper event handling
        max_wait = 300  # 5 minutes max wait
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            # Check if request is still in queue
            with self._queue_lock:
                still_queued = any(r.request_id == request_id for r in self.request_queue)
            
            if not still_queued:
                # Request was processed
                return {"status": "completed", "request_id": request_id}
            
            await asyncio.sleep(0.5)
        
        raise TimeoutError(f"Request {request_id} timed out in queue")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        with self._stats_lock:
            self._cleanup_old_requests()
            
            return {
                "provider": self.provider,
                "config": {
                    "requests_per_minute": self.config.requests_per_minute,
                    "requests_per_hour": self.config.requests_per_hour,
                    "max_concurrent": self.config.max_concurrent
                },
                "current_state": {
                    "active_requests": self.active_requests,
                    "queue_size": len(self.request_queue),
                    "requests_last_minute": len(self.request_times),
                    "requests_last_hour": len(self.hourly_requests),
                    "hourly_cost": self.hourly_cost,
                    "daily_cost": self.daily_cost
                },
                "statistics": {
                    "total_requests": self.stats.total_requests,
                    "queued_requests": self.stats.queued_requests,
                    "rejected_requests": self.stats.rejected_requests,
                    "avg_queue_time": self.stats.avg_queue_time,
                    "peak_queue_size": self.stats.peak_queue_size
                }
            }

class GlobalRateLimiter:
    """
    🌐 Global rate limiter managing all providers
    
    Features:
    - Provider-specific limits
    - Cross-provider cost tracking
    - Global emergency throttling
    - Request batching optimization
    """
    
    def __init__(self):
        # Provider configurations
        self.provider_configs = {
            "openai": RateLimitConfig(
                requests_per_minute=60,
                requests_per_hour=1000,
                max_concurrent=5,
                cost_per_request=0.02
            ),
            "anthropic": RateLimitConfig(
                requests_per_minute=50,
                requests_per_hour=800,
                max_concurrent=3,
                cost_per_request=0.03
            ),
            "google": RateLimitConfig(
                requests_per_minute=100,
                requests_per_hour=2000,
                max_concurrent=8,
                cost_per_request=0.005
            ),
            "deepseek": RateLimitConfig(
                requests_per_minute=40,
                requests_per_hour=600,
                max_concurrent=4,
                cost_per_request=0.01
            ),
            "nvidia": RateLimitConfig(
                requests_per_minute=30,
                requests_per_hour=400,
                max_concurrent=2,
                cost_per_request=0.05
            ),
            "multimodel": RateLimitConfig(
                requests_per_minute=20,  # Lower limit for expensive consensus
                requests_per_hour=200,
                max_concurrent=2,
                cost_per_request=0.10  # High cost due to multiple calls
            )
        }
        
        # Provider rate limiters
        self.provider_limiters: Dict[str, ProviderRateLimiter] = {}
        
        # Global limits
        self.global_hourly_cost_limit = 50.0  # $50/hour emergency limit
        self.global_daily_cost_limit = 200.0  # $200/day emergency limit
        self.total_hourly_cost = 0.0
        self.total_daily_cost = 0.0
        
        # Initialize provider limiters
        for provider, config in self.provider_configs.items():
            self.provider_limiters[provider] = ProviderRateLimiter(provider, config)
        
        logger.info("🌐 Global rate limiter initialized")
    
    def get_provider_limiter(self, provider: str) -> ProviderRateLimiter:
        """Get rate limiter for specific provider"""
        if provider not in self.provider_limiters:
            # Create default limiter for unknown provider
            default_config = RateLimitConfig()
            self.provider_limiters[provider] = ProviderRateLimiter(provider, default_config)
            logger.warning(f"Created default rate limiter for unknown provider: {provider}")
        
        return self.provider_limiters[provider]
    
    async def execute_request(self,
                            provider: str,
                            request_id: str,
                            request_func: Callable[[], Awaitable[Any]],
                            priority: RequestPriority = RequestPriority.NORMAL,
                            cost_estimate: float = None,
                            model: str = "unknown") -> Any:
        """Execute request through appropriate provider limiter"""
        
        # Check global cost limits
        if self.total_hourly_cost >= self.global_hourly_cost_limit:
            raise Exception(f"Global hourly cost limit exceeded: ${self.total_hourly_cost:.2f}")
        
        if self.total_daily_cost >= self.global_daily_cost_limit:
            raise Exception(f"Global daily cost limit exceeded: ${self.total_daily_cost:.2f}")
        
        # Get provider limiter
        limiter = self.get_provider_limiter(provider)
        
        # Execute through provider limiter
        try:
            result = await limiter.execute_request(
                request_id, request_func, priority, cost_estimate, model
            )
            
            # Update global cost tracking
            if cost_estimate:
                self.total_hourly_cost += cost_estimate
                self.total_daily_cost += cost_estimate
            
            return result
            
        except Exception as e:
            logger.error(f"Rate limited request failed for {provider}: {e}")
            raise
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get comprehensive rate limiting statistics"""
        provider_stats = {}
        total_requests = 0
        total_queued = 0
        total_rejected = 0
        
        for provider, limiter in self.provider_limiters.items():
            stats = limiter.get_stats()
            provider_stats[provider] = stats
            total_requests += stats["statistics"]["total_requests"]
            total_queued += stats["statistics"]["queued_requests"]
            total_rejected += stats["statistics"]["rejected_requests"]
        
        return {
            "global_limits": {
                "hourly_cost_limit": self.global_hourly_cost_limit,
                "daily_cost_limit": self.global_daily_cost_limit,
                "current_hourly_cost": self.total_hourly_cost,
                "current_daily_cost": self.total_daily_cost
            },
            "totals": {
                "total_requests": total_requests,
                "total_queued": total_queued,
                "total_rejected": total_rejected
            },
            "providers": provider_stats
        }
    
    def update_provider_config(self, provider: str, config: RateLimitConfig):
        """Update configuration for specific provider"""
        self.provider_configs[provider] = config
        if provider in self.provider_limiters:
            self.provider_limiters[provider].config = config
        logger.info(f"Updated rate limit config for {provider}")
    
    def emergency_throttle(self, factor: float = 0.5):
        """Emergency throttling - reduce all limits by factor"""
        for provider, limiter in self.provider_limiters.items():
            original_config = limiter.config
            limiter.config.requests_per_minute = int(original_config.requests_per_minute * factor)
            limiter.config.requests_per_hour = int(original_config.requests_per_hour * factor)
            limiter.config.max_concurrent = max(1, int(original_config.max_concurrent * factor))
        
        logger.warning(f"🚨 Emergency throttling activated - all limits reduced by {factor}")


# Global rate limiter instance
_global_rate_limiter: Optional[GlobalRateLimiter] = None

def get_rate_limiter() -> GlobalRateLimiter:
    """Get global rate limiter instance"""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = GlobalRateLimiter()
    return _global_rate_limiter

def init_rate_limiter() -> GlobalRateLimiter:
    """Initialize global rate limiter"""
    global _global_rate_limiter
    _global_rate_limiter = GlobalRateLimiter()
    return _global_rate_limiter

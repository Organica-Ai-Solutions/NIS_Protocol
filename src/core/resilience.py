#!/usr/bin/env python3
"""
NIS Protocol Resilience Patterns
Production-grade fault tolerance and graceful degradation

Features:
- Circuit Breaker pattern
- Retry with exponential backoff
- Graceful shutdown handling
- Health check improvements
- Timeout management
"""

import asyncio
import time
import logging
import signal
import functools
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

logger = logging.getLogger("nis.core.resilience")


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5      # Failures before opening
    success_threshold: int = 2      # Successes to close from half-open
    timeout: float = 30.0           # Seconds before trying half-open
    half_open_max_calls: int = 3    # Max calls in half-open state


class CircuitBreaker:
    """
    Circuit Breaker implementation
    
    Prevents cascading failures by stopping calls to failing services.
    
    Usage:
        breaker = CircuitBreaker("external_api")
        
        @breaker
        async def call_external_api():
            ...
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        
        self._lock = asyncio.Lock()
    
    async def _can_execute(self) -> bool:
        """Check if request can be executed"""
        async with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            
            if self.state == CircuitState.OPEN:
                # Check if timeout has passed
                if time.time() - self.last_failure_time >= self.config.timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info(f"Circuit {self.name}: OPEN -> HALF_OPEN")
                    return True
                return False
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls < self.config.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                return False
            
            return False
    
    async def _record_success(self):
        """Record successful call"""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"Circuit {self.name}: HALF_OPEN -> CLOSED")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0  # Reset on success
    
    async def _record_failure(self):
        """Record failed call"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.success_count = 0
                logger.warning(f"Circuit {self.name}: HALF_OPEN -> OPEN (failure)")
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit {self.name}: CLOSED -> OPEN (threshold reached)")
    
    def __call__(self, func: Callable):
        """Decorator for circuit breaker"""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not await self._can_execute():
                raise CircuitOpenError(f"Circuit {self.name} is OPEN")
            
            try:
                result = await func(*args, **kwargs)
                await self._record_success()
                return result
            except Exception as e:
                await self._record_failure()
                raise
        
        return wrapper
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time
        }


class CircuitOpenError(Exception):
    """Raised when circuit is open"""
    pass


# ============================================================================
# RETRY WITH BACKOFF
# ============================================================================

@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True


def retry_with_backoff(config: RetryConfig = None):
    """
    Decorator for retry with exponential backoff
    
    Usage:
        @retry_with_backoff(RetryConfig(max_attempts=5))
        async def unreliable_operation():
            ...
    """
    config = config or RetryConfig()
    
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts - 1:
                        delay = min(
                            config.base_delay * (config.exponential_base ** attempt),
                            config.max_delay
                        )
                        
                        if config.jitter:
                            import random
                            delay *= (0.5 + random.random())
                        
                        logger.warning(
                            f"Retry {attempt + 1}/{config.max_attempts} for {func.__name__} "
                            f"after {delay:.2f}s: {e}"
                        )
                        await asyncio.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


# ============================================================================
# TIMEOUT MANAGEMENT
# ============================================================================

def with_timeout(seconds: float):
    """
    Decorator to add timeout to async functions
    
    Usage:
        @with_timeout(5.0)
        async def slow_operation():
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"{func.__name__} timed out after {seconds}s")
        
        return wrapper
    return decorator


# ============================================================================
# GRACEFUL SHUTDOWN
# ============================================================================

class GracefulShutdown:
    """
    Graceful shutdown handler
    
    Ensures clean shutdown of all services.
    
    Usage:
        shutdown = GracefulShutdown()
        shutdown.register(cleanup_kafka)
        shutdown.register(cleanup_redis)
        shutdown.setup_signals()
    """
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.shutdown_event = asyncio.Event()
        self.cleanup_handlers: List[Callable] = []
        self._is_shutting_down = False
    
    def register(self, handler: Callable):
        """Register a cleanup handler"""
        self.cleanup_handlers.append(handler)
        logger.debug(f"Registered cleanup handler: {handler.__name__}")
    
    def setup_signals(self):
        """Setup signal handlers for graceful shutdown"""
        try:
            loop = asyncio.get_event_loop()
            
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(self._handle_signal(s))
                )
            
            logger.info("Signal handlers registered for graceful shutdown")
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            logger.warning("Signal handlers not supported on this platform")
    
    async def _handle_signal(self, sig):
        """Handle shutdown signal"""
        if self._is_shutting_down:
            logger.warning("Forced shutdown requested")
            return
        
        self._is_shutting_down = True
        logger.info(f"Received signal {sig.name}, initiating graceful shutdown...")
        
        await self.shutdown()
    
    async def shutdown(self):
        """Execute graceful shutdown"""
        logger.info("Starting graceful shutdown...")
        
        # Signal shutdown to waiting tasks
        self.shutdown_event.set()
        
        # Run cleanup handlers with timeout
        for handler in reversed(self.cleanup_handlers):
            try:
                logger.info(f"Running cleanup: {handler.__name__}")
                
                if asyncio.iscoroutinefunction(handler):
                    await asyncio.wait_for(handler(), timeout=self.timeout / len(self.cleanup_handlers))
                else:
                    handler()
                    
            except asyncio.TimeoutError:
                logger.error(f"Cleanup handler {handler.__name__} timed out")
            except Exception as e:
                logger.error(f"Cleanup handler {handler.__name__} failed: {e}")
        
        logger.info("Graceful shutdown complete")
    
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress"""
        return self._is_shutting_down
    
    async def wait_for_shutdown(self):
        """Wait for shutdown signal"""
        await self.shutdown_event.wait()


# ============================================================================
# ENHANCED HEALTH CHECK
# ============================================================================

@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    healthy: bool
    latency_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class HealthChecker:
    """
    Enhanced health checker with dependency checks
    
    Usage:
        checker = HealthChecker()
        checker.add_check("kafka", check_kafka)
        checker.add_check("redis", check_redis)
        
        status = await checker.check_all()
    """
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.check_interval = 30.0  # seconds
        self._last_check_time = 0
    
    def add_check(self, name: str, check_func: Callable):
        """Add a health check"""
        self.checks[name] = check_func
    
    async def check_one(self, name: str) -> HealthCheckResult:
        """Run a single health check"""
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                healthy=False,
                latency_ms=0,
                error="Check not found"
            )
        
        start = time.time()
        try:
            check_func = self.checks[name]
            
            if asyncio.iscoroutinefunction(check_func):
                result = await asyncio.wait_for(check_func(), timeout=5.0)
            else:
                result = check_func()
            
            latency = (time.time() - start) * 1000
            
            if isinstance(result, dict):
                healthy = result.get("healthy", True)
                details = result
            else:
                healthy = bool(result)
                details = {}
            
            return HealthCheckResult(
                name=name,
                healthy=healthy,
                latency_ms=latency,
                details=details
            )
            
        except asyncio.TimeoutError:
            return HealthCheckResult(
                name=name,
                healthy=False,
                latency_ms=(time.time() - start) * 1000,
                error="Health check timed out"
            )
        except Exception as e:
            return HealthCheckResult(
                name=name,
                healthy=False,
                latency_ms=(time.time() - start) * 1000,
                error=str(e)
            )
    
    async def check_all(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = await asyncio.gather(
            *[self.check_one(name) for name in self.checks],
            return_exceptions=True
        )
        
        self._last_check_time = time.time()
        
        check_results = {}
        all_healthy = True
        
        for i, name in enumerate(self.checks):
            result = results[i]
            if isinstance(result, Exception):
                result = HealthCheckResult(
                    name=name,
                    healthy=False,
                    latency_ms=0,
                    error=str(result)
                )
            
            self.last_results[name] = result
            check_results[name] = {
                "healthy": result.healthy,
                "latency_ms": result.latency_ms,
                "details": result.details,
                "error": result.error
            }
            
            if not result.healthy:
                all_healthy = False
        
        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "checks": check_results,
            "timestamp": time.time()
        }
    
    def get_cached_status(self) -> Dict[str, Any]:
        """Get cached health status"""
        if not self.last_results:
            return {"status": "unknown", "message": "No health checks run yet"}
        
        all_healthy = all(r.healthy for r in self.last_results.values())
        
        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "checks": {
                name: {
                    "healthy": r.healthy,
                    "latency_ms": r.latency_ms,
                    "error": r.error
                }
                for name, r in self.last_results.items()
            },
            "last_check": self._last_check_time
        }


# ============================================================================
# CIRCUIT BREAKER REGISTRY
# ============================================================================

class CircuitBreakerRegistry:
    """Registry for all circuit breakers"""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_or_create(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create a circuit breaker"""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config)
        return self.breakers[name]
    
    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers"""
        return {
            name: breaker.get_status()
            for name, breaker in self.breakers.items()
        }


# Singleton instances
_shutdown_handler: Optional[GracefulShutdown] = None
_health_checker: Optional[HealthChecker] = None
_circuit_registry: Optional[CircuitBreakerRegistry] = None


def get_shutdown_handler() -> GracefulShutdown:
    """Get the shutdown handler singleton"""
    global _shutdown_handler
    if _shutdown_handler is None:
        _shutdown_handler = GracefulShutdown()
    return _shutdown_handler


def get_health_checker() -> HealthChecker:
    """Get the health checker singleton"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def get_circuit_registry() -> CircuitBreakerRegistry:
    """Get the circuit breaker registry singleton"""
    global _circuit_registry
    if _circuit_registry is None:
        _circuit_registry = CircuitBreakerRegistry()
    return _circuit_registry

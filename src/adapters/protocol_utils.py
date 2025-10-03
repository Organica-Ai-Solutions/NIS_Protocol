"""
Protocol Utilities for NIS Protocol v3.2

Retry logic, circuit breaker, and metrics collection for protocol adapters.
Enables production-grade error handling and monitoring.
"""

import asyncio
import functools
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Optional, Dict, Any, Tuple, Type
from collections import defaultdict

from .protocol_errors import (
    ProtocolError,
    CircuitBreakerOpenError,
    ProtocolTimeoutError,
    ProtocolConnectionError
)

logger = logging.getLogger(__name__)


# =============================================================================
# RETRY LOGIC WITH EXPONENTIAL BACKOFF
# =============================================================================

def with_retry(
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    backoff_max: float = 60.0,
    retryable_errors: Optional[Tuple[Type[Exception], ...]] = None
):
    """
    Retry decorator with exponential backoff.
    
    Automatically retries failed operations with increasing delays.
    
    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        backoff_base: Base for exponential backoff calculation (default: 2.0)
        backoff_max: Maximum backoff time in seconds (default: 60.0)
        retryable_errors: Tuple of exception types to retry (default: common network errors)
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @with_retry(max_attempts=3, backoff_base=2.0)
        async def make_request():
            return await api_call()
    """
    if retryable_errors is None:
        # Default retryable errors
        try:
            import requests
            retryable_errors = (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                ProtocolTimeoutError,
                ProtocolConnectionError
            )
        except ImportError:
            retryable_errors = (
                ProtocolTimeoutError,
                ProtocolConnectionError
            )
    
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                    
                except retryable_errors as e:
                    last_error = e
                    
                    # Don't retry on last attempt
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    
                    # Calculate exponential backoff
                    wait_time = min(
                        backoff_base ** attempt,
                        backoff_max
                    )
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_attempts} failed: {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    
                    await asyncio.sleep(wait_time)
            
            # Should never reach here, but just in case
            raise last_error
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                    
                except retryable_errors as e:
                    last_error = e
                    
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    
                    wait_time = min(
                        backoff_base ** attempt,
                        backoff_max
                    )
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_attempts} failed: {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    
                    time.sleep(wait_time)
            
            raise last_error
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# =============================================================================
# CIRCUIT BREAKER PATTERN
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation, requests flow through
    OPEN = "open"          # Failing, reject requests immediately
    HALF_OPEN = "half_open"  # Testing recovery, allow limited requests


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascade failures by temporarily blocking requests to failing services.
    Automatically attempts recovery after a timeout period.
    
    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Service failing, reject all requests
    - HALF_OPEN: Testing recovery, allow limited requests
    
    Example:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        result = await breaker.call(make_request, arg1, arg2)
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before testing recovery
            success_threshold: Successful calls needed to close circuit in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time: Optional[datetime] = None
        
        logger.info(
            f"Circuit breaker initialized: "
            f"failure_threshold={failure_threshold}, "
            f"recovery_timeout={recovery_timeout}s"
        )
    
    async def call(self, func: Callable, *args, **kwargs):
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Other exceptions: From the wrapped function
        """
        # Check if circuit should transition states
        self._check_state_transition()
        
        # Reject if circuit is open
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(
                f"Circuit breaker is OPEN - service unavailable. "
                f"Failed {self.failure_count} times. "
                f"Will retry after {self.recovery_timeout}s"
            )
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                logger.info(
                    f"Circuit breaker CLOSED - service recovered "
                    f"({self.success_threshold} successful calls)"
                )
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.success_count = 0  # Reset success count on any failure
        
        if self.failure_count >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                self.state = CircuitState.OPEN
                logger.error(
                    f"Circuit breaker OPEN - {self.failure_count} consecutive failures. "
                    f"Blocking requests for {self.recovery_timeout}s"
                )
    
    def _check_state_transition(self):
        """Check if circuit should transition from OPEN to HALF_OPEN"""
        if self.state == CircuitState.OPEN and self.last_failure_time:
            time_since_failure = datetime.now() - self.last_failure_time
            
            if time_since_failure >= timedelta(seconds=self.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self.failure_count = 0
                logger.info(
                    f"Circuit breaker HALF_OPEN - testing recovery "
                    f"(need {self.success_threshold} successful calls)"
                )
    
    def reset(self):
        """Manually reset circuit breaker to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker manually reset to CLOSED")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


# =============================================================================
# METRICS COLLECTION
# =============================================================================

@dataclass
class ProtocolMetrics:
    """
    Performance metrics for protocol adapter.
    
    Tracks success rates, response times, errors, and circuit breaker state.
    """
    protocol_name: str
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retried_requests: int = 0
    
    # Timing metrics (in seconds)
    total_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    
    # Error tracking
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Circuit breaker state
    circuit_state: str = "closed"
    
    # Timestamps
    first_request_time: Optional[float] = None
    last_request_time: Optional[float] = None
    
    def record_request(
        self,
        success: bool,
        response_time: float,
        error_type: Optional[str] = None,
        retried: bool = False
    ):
        """
        Record request metrics.
        
        Args:
            success: Whether request succeeded
            response_time: Request duration in seconds
            error_type: Type of error if failed
            retried: Whether this was a retry attempt
        """
        self.total_requests += 1
        
        if self.first_request_time is None:
            self.first_request_time = time.time()
        self.last_request_time = time.time()
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error_type:
                self.error_counts[error_type] += 1
        
        if retried:
            self.retried_requests += 1
        
        # Update timing metrics
        self.total_response_time += response_time
        self.min_response_time = min(self.min_response_time, response_time)
        self.max_response_time = max(self.max_response_time, response_time)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate (0.0 to 1.0)"""
        return 1.0 - self.success_rate
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time in seconds"""
        if self.total_requests == 0:
            return 0.0
        return self.total_response_time / self.total_requests
    
    @property
    def requests_per_second(self) -> float:
        """Calculate requests per second"""
        if not self.first_request_time or not self.last_request_time:
            return 0.0
        
        duration = self.last_request_time - self.first_request_time
        if duration == 0:
            return 0.0
        
        return self.total_requests / duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary"""
        return {
            "protocol": self.protocol_name,
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "retried": self.retried_requests
            },
            "performance": {
                "success_rate": round(self.success_rate, 4),
                "error_rate": round(self.error_rate, 4),
                "avg_response_time_ms": round(self.average_response_time * 1000, 2),
                "min_response_time_ms": round(self.min_response_time * 1000, 2) if self.min_response_time != float('inf') else 0,
                "max_response_time_ms": round(self.max_response_time * 1000, 2),
                "requests_per_second": round(self.requests_per_second, 2)
            },
            "errors": dict(self.error_counts),
            "circuit_breaker": self.circuit_state,
            "uptime_seconds": round(self.last_request_time - self.first_request_time, 2) if self.first_request_time and self.last_request_time else 0
        }
    
    def reset(self):
        """Reset all metrics to initial state"""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.retried_requests = 0
        self.total_response_time = 0.0
        self.min_response_time = float('inf')
        self.max_response_time = 0.0
        self.error_counts.clear()
        self.first_request_time = None
        self.last_request_time = None


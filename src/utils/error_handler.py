"""
Error Handler - NIS Protocol v4.0
Graceful error handling with retries, fallbacks, and clear messages.
"""

import asyncio
import functools
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("nis.error_handler")

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"           # Recoverable, user doesn't need to know
    MEDIUM = "medium"     # Recoverable with fallback, inform user
    HIGH = "high"         # Not recoverable, user action needed
    CRITICAL = "critical" # System-level failure


class ErrorCategory(Enum):
    """Error categories for better handling"""
    NETWORK = "network"           # Connection issues
    AUTH = "auth"                 # Authentication/authorization
    RATE_LIMIT = "rate_limit"     # API rate limiting
    VALIDATION = "validation"     # Input validation
    PROVIDER = "provider"         # LLM provider issues
    INTERNAL = "internal"         # Internal system errors
    TIMEOUT = "timeout"           # Operation timeout
    RESOURCE = "resource"         # Resource exhaustion


@dataclass
class NISError:
    """Structured error with context"""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    user_message: str
    details: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None
    retry_after: Optional[int] = None  # Seconds
    original_error: Optional[Exception] = None


# Error classification rules
ERROR_PATTERNS = {
    # Network errors
    "connection refused": (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM, "Service temporarily unavailable"),
    "timeout": (ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM, "Request took too long"),
    "connection reset": (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM, "Connection interrupted"),
    "dns": (ErrorCategory.NETWORK, ErrorSeverity.HIGH, "Network configuration issue"),
    
    # Auth errors
    "401": (ErrorCategory.AUTH, ErrorSeverity.HIGH, "Authentication required"),
    "403": (ErrorCategory.AUTH, ErrorSeverity.HIGH, "Access denied"),
    "invalid api key": (ErrorCategory.AUTH, ErrorSeverity.HIGH, "Invalid API key"),
    "unauthorized": (ErrorCategory.AUTH, ErrorSeverity.HIGH, "Not authorized"),
    
    # Rate limits
    "429": (ErrorCategory.RATE_LIMIT, ErrorSeverity.MEDIUM, "Too many requests"),
    "rate limit": (ErrorCategory.RATE_LIMIT, ErrorSeverity.MEDIUM, "Rate limit reached"),
    "quota exceeded": (ErrorCategory.RATE_LIMIT, ErrorSeverity.HIGH, "API quota exceeded"),
    
    # Provider errors
    "openai": (ErrorCategory.PROVIDER, ErrorSeverity.MEDIUM, "OpenAI service issue"),
    "anthropic": (ErrorCategory.PROVIDER, ErrorSeverity.MEDIUM, "Anthropic service issue"),
    "model not found": (ErrorCategory.PROVIDER, ErrorSeverity.HIGH, "Model unavailable"),
    
    # Validation
    "validation": (ErrorCategory.VALIDATION, ErrorSeverity.LOW, "Invalid input"),
    "invalid": (ErrorCategory.VALIDATION, ErrorSeverity.LOW, "Invalid request"),
    
    # Resource
    "out of memory": (ErrorCategory.RESOURCE, ErrorSeverity.CRITICAL, "System overloaded"),
    "disk full": (ErrorCategory.RESOURCE, ErrorSeverity.CRITICAL, "Storage full"),
}


def classify_error(error: Exception) -> NISError:
    """Classify an exception into a structured NISError"""
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    
    # Check patterns
    for pattern, (category, severity, user_msg) in ERROR_PATTERNS.items():
        if pattern in error_str or pattern in error_type:
            return NISError(
                category=category,
                severity=severity,
                message=str(error),
                user_message=user_msg,
                details={"error_type": type(error).__name__},
                suggestion=_get_suggestion(category),
                retry_after=_get_retry_delay(category),
                original_error=error
            )
    
    # Default classification
    return NISError(
        category=ErrorCategory.INTERNAL,
        severity=ErrorSeverity.MEDIUM,
        message=str(error),
        user_message="An unexpected error occurred",
        details={
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc()
        },
        suggestion="Please try again. If the problem persists, contact support.",
        original_error=error
    )


def _get_suggestion(category: ErrorCategory) -> str:
    """Get user-friendly suggestion for error category"""
    suggestions = {
        ErrorCategory.NETWORK: "Check your internet connection and try again.",
        ErrorCategory.AUTH: "Verify your API key in the settings.",
        ErrorCategory.RATE_LIMIT: "Please wait a moment before trying again.",
        ErrorCategory.VALIDATION: "Check your input and try again.",
        ErrorCategory.PROVIDER: "The AI service is temporarily unavailable. Trying fallback...",
        ErrorCategory.INTERNAL: "Please try again. If the problem persists, contact support.",
        ErrorCategory.TIMEOUT: "The request took too long. Try a simpler query.",
        ErrorCategory.RESOURCE: "System is under heavy load. Please try again later.",
    }
    return suggestions.get(category, "Please try again.")


def _get_retry_delay(category: ErrorCategory) -> Optional[int]:
    """Get recommended retry delay in seconds"""
    delays = {
        ErrorCategory.NETWORK: 5,
        ErrorCategory.RATE_LIMIT: 30,
        ErrorCategory.PROVIDER: 10,
        ErrorCategory.TIMEOUT: 5,
        ErrorCategory.RESOURCE: 60,
    }
    return delays.get(category)


class RetryConfig:
    """Configuration for retry behavior"""
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        retryable_categories: Optional[List[ErrorCategory]] = None
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retryable_categories = retryable_categories or [
            ErrorCategory.NETWORK,
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.PROVIDER,
            ErrorCategory.TIMEOUT
        ]
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt (exponential backoff)"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)
    
    def should_retry(self, error: NISError, attempt: int) -> bool:
        """Determine if we should retry"""
        if attempt >= self.max_retries:
            return False
        if error.category not in self.retryable_categories:
            return False
        if error.severity == ErrorSeverity.CRITICAL:
            return False
        return True


def with_retry(
    config: Optional[RetryConfig] = None,
    fallback: Optional[Callable[[], T]] = None
):
    """
    Decorator for automatic retry with exponential backoff.
    
    Usage:
        @with_retry(RetryConfig(max_retries=3))
        async def call_api():
            ...
    """
    config = config or RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_error = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    nis_error = classify_error(e)
                    last_error = nis_error
                    
                    if not config.should_retry(nis_error, attempt):
                        break
                    
                    delay = config.get_delay(attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_retries} for {func.__name__} "
                        f"after {delay:.1f}s: {nis_error.user_message}"
                    )
                    await asyncio.sleep(delay)
            
            # All retries failed
            if fallback:
                logger.info(f"Using fallback for {func.__name__}")
                return fallback()
            
            raise ErrorHandlerException(last_error)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_error = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    nis_error = classify_error(e)
                    last_error = nis_error
                    
                    if not config.should_retry(nis_error, attempt):
                        break
                    
                    delay = config.get_delay(attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_retries} for {func.__name__} "
                        f"after {delay:.1f}s: {nis_error.user_message}"
                    )
                    time.sleep(delay)
            
            if fallback:
                logger.info(f"Using fallback for {func.__name__}")
                return fallback()
            
            raise ErrorHandlerException(last_error)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class ErrorHandlerException(Exception):
    """Exception wrapper with NISError context"""
    def __init__(self, nis_error: NISError):
        self.nis_error = nis_error
        super().__init__(nis_error.user_message)


def safe_execute(
    func: Callable[..., T],
    *args,
    default: Optional[T] = None,
    log_error: bool = True,
    **kwargs
) -> T:
    """
    Safely execute a function, returning default on error.
    
    Usage:
        result = safe_execute(risky_function, arg1, arg2, default="fallback")
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_error:
            nis_error = classify_error(e)
            logger.warning(f"Safe execute failed: {nis_error.user_message}")
        return default


async def safe_execute_async(
    func: Callable[..., T],
    *args,
    default: Optional[T] = None,
    log_error: bool = True,
    **kwargs
) -> T:
    """Async version of safe_execute"""
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        if log_error:
            nis_error = classify_error(e)
            logger.warning(f"Safe execute async failed: {nis_error.user_message}")
        return default


def format_error_response(error: Union[Exception, NISError]) -> Dict[str, Any]:
    """Format error for API response"""
    if isinstance(error, NISError):
        nis_error = error
    elif isinstance(error, ErrorHandlerException):
        nis_error = error.nis_error
    else:
        nis_error = classify_error(error)
    
    response = {
        "success": False,
        "error": {
            "message": nis_error.user_message,
            "category": nis_error.category.value,
            "severity": nis_error.severity.value,
        }
    }
    
    if nis_error.suggestion:
        response["error"]["suggestion"] = nis_error.suggestion
    
    if nis_error.retry_after:
        response["error"]["retry_after_seconds"] = nis_error.retry_after
    
    return response


class CircuitBreaker:
    """
    Circuit breaker pattern for failing fast on repeated errors.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Failing fast (too many errors)
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.failures = 0
        self.last_failure_time = 0
        self.state = "closed"
        self.half_open_calls = 0
    
    def can_execute(self) -> bool:
        """Check if we can execute (circuit not open)"""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            # Check if recovery timeout passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "half_open"
                self.half_open_calls = 0
                return True
            return False
        
        # Half-open: allow limited calls
        if self.half_open_calls < self.half_open_max_calls:
            return True
        return False
    
    def record_success(self):
        """Record successful call"""
        if self.state == "half_open":
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                # Recovered
                self.state = "closed"
                self.failures = 0
        else:
            self.failures = max(0, self.failures - 1)
    
    def record_failure(self):
        """Record failed call"""
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.state == "half_open":
            # Failed during recovery test
            self.state = "open"
        elif self.failures >= self.failure_threshold:
            self.state = "open"
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        return {
            "state": self.state,
            "failures": self.failures,
            "threshold": self.failure_threshold,
            "time_until_retry": max(0, self.recovery_timeout - (time.time() - self.last_failure_time))
            if self.state == "open" else 0
        }

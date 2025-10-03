"""
Protocol Error Classes for NIS Protocol v3.2

Custom error types for protocol adapter error handling.
Enables specific error handling and better debugging.
"""

from typing import Optional


class ProtocolError(Exception):
    """
    Base error class for all protocol-related errors.
    
    All custom protocol errors inherit from this class for easy catching.
    """
    pass


class ProtocolConnectionError(ProtocolError):
    """
    Failed to establish connection to protocol endpoint.
    
    Raised when:
    - Server is unreachable
    - DNS resolution fails
    - Network connectivity issues
    """
    pass


class ProtocolTimeoutError(ProtocolError):
    """
    Request exceeded timeout threshold.
    
    Raised when:
    - Server doesn't respond within timeout
    - Slow network conditions
    - Server is overloaded
    """
    def __init__(self, message: str, timeout: Optional[float] = None):
        self.timeout = timeout
        super().__init__(message)


class ProtocolAuthError(ProtocolError):
    """
    Authentication or authorization failed.
    
    Raised when:
    - Invalid API key
    - Expired token
    - Insufficient permissions
    - Missing credentials
    """
    pass


class ProtocolValidationError(ProtocolError):
    """
    Response validation failed.
    
    Raised when:
    - Invalid response format
    - Missing required fields
    - Type mismatch in response
    - Protocol spec violation
    """
    def __init__(self, message: str, response: Optional[dict] = None):
        self.response = response
        super().__init__(message)


class ProtocolRateLimitError(ProtocolError):
    """
    Rate limit exceeded.
    
    Raised when:
    - Too many requests to endpoint
    - Quota exhausted
    - Throttling enforced by server
    """
    def __init__(self, message: str, retry_after: Optional[int] = None):
        self.retry_after = retry_after
        msg = message
        if retry_after:
            msg = f"{message} (retry after {retry_after}s)"
        super().__init__(msg)


class ProtocolServerError(ProtocolError):
    """
    Server-side error (5xx responses).
    
    Raised when:
    - Internal server error
    - Service unavailable
    - Gateway timeout
    """
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(message)


class CircuitBreakerOpenError(ProtocolError):
    """
    Circuit breaker is open - service unavailable.
    
    Raised when:
    - Circuit breaker detects repeated failures
    - Service is temporarily disabled
    - Waiting for recovery period
    """
    pass


def get_error_from_response(status_code: int, response_data: dict) -> ProtocolError:
    """
    Create appropriate error from HTTP response.
    
    Args:
        status_code: HTTP status code
        response_data: Response body
        
    Returns:
        Appropriate ProtocolError subclass
    """
    message = response_data.get("error", {}).get("message", "Unknown error")
    
    if status_code == 401 or status_code == 403:
        return ProtocolAuthError(f"Authentication failed: {message}")
    
    elif status_code == 429:
        retry_after = response_data.get("retry_after")
        return ProtocolRateLimitError(message, retry_after)
    
    elif status_code >= 500:
        return ProtocolServerError(message, status_code)
    
    elif status_code == 408 or status_code == 504:
        return ProtocolTimeoutError(message)
    
    else:
        return ProtocolError(f"Request failed ({status_code}): {message}")


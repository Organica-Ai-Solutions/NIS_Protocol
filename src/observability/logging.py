#!/usr/bin/env python3
"""
NIS Protocol Structured Logging
JSON-formatted logging for production observability

Features:
- JSON structured output
- Log levels with filtering
- Context injection (trace_id, user_id)
- Performance metrics
- Error tracking with stack traces
"""

import os
import sys
import json
import time
import logging
import traceback
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging"""
    
    def __init__(self, service_name: str = "nis-protocol"):
        super().__init__()
        self.service_name = service_name
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add trace context if available
        if hasattr(record, 'trace_id'):
            log_data["trace_id"] = record.trace_id
        if hasattr(record, 'span_id'):
            log_data["span_id"] = record.span_id
        if hasattr(record, 'user_id'):
            log_data["user_id"] = record.user_id
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        # Add exception info
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stacktrace": traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_data)


class StructuredLogger:
    """
    Structured logger with context injection
    
    Usage:
        logger = StructuredLogger("my_module")
        logger.info("User logged in", user_id="123", action="login")
    """
    
    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        json_output: bool = True
    ):
        self.name = name
        self.logger = logging.getLogger(name)
        self.json_output = json_output
        
        # Context that gets added to all logs
        self._context: Dict[str, Any] = {}
        
        # Set level
        self.logger.setLevel(getattr(logging, level.value))
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal log method with context injection"""
        # Merge context with kwargs
        extra_fields = {**self._context, **kwargs}
        
        # Create log record with extra fields
        record = self.logger.makeRecord(
            self.name,
            level,
            "(unknown)",
            0,
            message,
            (),
            None
        )
        record.extra_fields = extra_fields
        
        # Add trace context if available
        try:
            from .tracing import get_current_span
            span = get_current_span()
            if span:
                record.trace_id = span.trace_id
                record.span_id = span.span_id
        except:
            pass
        
        self.logger.handle(record)
    
    def set_context(self, **kwargs):
        """Set persistent context fields"""
        self._context.update(kwargs)
    
    def clear_context(self):
        """Clear context fields"""
        self._context.clear()
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message"""
        if exc_info:
            kwargs['exc_info'] = sys.exc_info()
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, exc_info: bool = False, **kwargs):
        """Log critical message"""
        if exc_info:
            kwargs['exc_info'] = sys.exc_info()
        self._log(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with stack trace"""
        self.error(message, exc_info=True, **kwargs)


# Logger registry
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(name: str) -> StructuredLogger:
    """Get or create a structured logger"""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)
    return _loggers[name]


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    json_output: bool = True,
    service_name: str = "nis-protocol"
):
    """
    Configure global logging settings
    
    Args:
        level: Minimum log level
        json_output: Use JSON formatting
        service_name: Service name for logs
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.value))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add new handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.value))
    
    if json_output:
        handler.setFormatter(JSONFormatter(service_name))
    else:
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
    
    root_logger.addHandler(handler)
    
    logging.info(f"Logging configured: level={level.value}, json={json_output}")


class RequestLogger:
    """
    Middleware-style request logger
    
    Usage:
        @app.middleware("http")
        async def log_requests(request, call_next):
            with RequestLogger(request) as logger:
                response = await call_next(request)
                logger.set_response(response)
                return response
    """
    
    def __init__(self, request):
        self.request = request
        self.start_time = time.time()
        self.response = None
        self.logger = get_logger("nis.http")
    
    def __enter__(self):
        self.logger.info(
            "Request started",
            method=self.request.method,
            path=str(self.request.url.path),
            client=self.request.client.host if self.request.client else "unknown"
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        
        if exc_type:
            self.logger.error(
                "Request failed",
                method=self.request.method,
                path=str(self.request.url.path),
                duration_ms=duration_ms,
                error=str(exc_val),
                exc_info=True
            )
        else:
            status_code = self.response.status_code if self.response else 0
            self.logger.info(
                "Request completed",
                method=self.request.method,
                path=str(self.request.url.path),
                status_code=status_code,
                duration_ms=duration_ms
            )
    
    def set_response(self, response):
        """Set the response for logging"""
        self.response = response

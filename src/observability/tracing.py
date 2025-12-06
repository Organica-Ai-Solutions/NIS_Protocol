#!/usr/bin/env python3
"""
NIS Protocol Distributed Tracing
OpenTelemetry-based tracing for request tracking across services

Features:
- Automatic span creation
- Context propagation
- Trace ID injection
- Performance timing
- Error tracking
"""

import os
import time
import uuid
import logging
import functools
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from contextvars import ContextVar

logger = logging.getLogger("nis.observability.tracing")

# Try to import OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.info("OpenTelemetry not available, using built-in tracing")


# Context variable for current span
_current_span: ContextVar[Optional['Span']] = ContextVar('current_span', default=None)


@dataclass
class Span:
    """A trace span representing a unit of work"""
    trace_id: str
    span_id: str
    name: str
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "OK"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: list = field(default_factory=list)
    
    def set_attribute(self, key: str, value: Any):
        """Set a span attribute"""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Dict[str, Any] = None):
        """Add an event to the span"""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })
    
    def set_status(self, status: str, description: str = None):
        """Set span status"""
        self.status = status
        if description:
            self.attributes["status_description"] = description
    
    def end(self):
        """End the span"""
        self.end_time = time.time()
    
    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds"""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary"""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events
        }


class TracingManager:
    """
    Distributed tracing manager for NIS Protocol
    
    Supports:
    - OpenTelemetry (if available)
    - Built-in lightweight tracing
    - Jaeger/Zipkin export
    """
    
    def __init__(
        self,
        service_name: str = "nis-protocol",
        otlp_endpoint: str = None,
        enable_console_export: bool = False
    ):
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        self.enable_console = enable_console_export
        
        # Span storage (for built-in tracing)
        self._spans: Dict[str, Span] = {}
        self._completed_spans: list = []
        self._max_completed = 1000
        
        # Statistics
        self.stats = {
            "spans_created": 0,
            "spans_completed": 0,
            "errors": 0
        }
        
        # Initialize OpenTelemetry if available
        self._otel_tracer = None
        if OTEL_AVAILABLE and self.otlp_endpoint:
            self._init_otel()
        
        logger.info(f"Tracing initialized (OpenTelemetry: {OTEL_AVAILABLE})")
    
    def _init_otel(self):
        """Initialize OpenTelemetry tracing"""
        try:
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": "4.0.1"
            })
            
            provider = TracerProvider(resource=resource)
            
            # Add OTLP exporter
            if self.otlp_endpoint:
                exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))
            
            trace.set_tracer_provider(provider)
            self._otel_tracer = trace.get_tracer(self.service_name)
            
            logger.info(f"OpenTelemetry initialized with endpoint: {self.otlp_endpoint}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
    
    def create_span(
        self,
        name: str,
        parent: Optional[Span] = None,
        attributes: Dict[str, Any] = None
    ) -> Span:
        """Create a new span"""
        parent_span = parent or _current_span.get()
        
        trace_id = parent_span.trace_id if parent_span else uuid.uuid4().hex[:32]
        span_id = uuid.uuid4().hex[:16]
        parent_id = parent_span.span_id if parent_span else None
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            name=name,
            parent_id=parent_id,
            attributes=attributes or {}
        )
        
        # Add service info
        span.set_attribute("service.name", self.service_name)
        
        self._spans[span_id] = span
        self.stats["spans_created"] += 1
        
        return span
    
    @contextmanager
    def start_span(self, name: str, attributes: Dict[str, Any] = None):
        """Context manager for creating and managing a span"""
        span = self.create_span(name, attributes=attributes)
        token = _current_span.set(span)
        
        try:
            yield span
            span.set_status("OK")
        except Exception as e:
            span.set_status("ERROR", str(e))
            span.add_event("exception", {
                "type": type(e).__name__,
                "message": str(e)
            })
            self.stats["errors"] += 1
            raise
        finally:
            span.end()
            self._complete_span(span)
            _current_span.reset(token)
    
    def _complete_span(self, span: Span):
        """Complete a span and store it"""
        if span.span_id in self._spans:
            del self._spans[span.span_id]
        
        self._completed_spans.append(span.to_dict())
        self.stats["spans_completed"] += 1
        
        # Trim old spans
        if len(self._completed_spans) > self._max_completed:
            self._completed_spans = self._completed_spans[-self._max_completed:]
        
        # Log if console export enabled
        if self.enable_console:
            logger.info(f"TRACE: {span.name} [{span.duration_ms:.2f}ms] {span.status}")
    
    def get_current_span(self) -> Optional[Span]:
        """Get the current active span"""
        return _current_span.get()
    
    def get_trace(self, trace_id: str) -> list:
        """Get all spans for a trace"""
        return [s for s in self._completed_spans if s["trace_id"] == trace_id]
    
    def get_recent_spans(self, limit: int = 100) -> list:
        """Get recent completed spans"""
        return self._completed_spans[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracing statistics"""
        return {
            **self.stats,
            "active_spans": len(self._spans),
            "stored_spans": len(self._completed_spans),
            "otel_enabled": self._otel_tracer is not None
        }


# Singleton instance
_tracer: Optional[TracingManager] = None


def get_tracer() -> TracingManager:
    """Get the tracing manager singleton"""
    global _tracer
    if _tracer is None:
        _tracer = TracingManager()
    return _tracer


def create_span(name: str, attributes: Dict[str, Any] = None):
    """Create a new span using the global tracer"""
    return get_tracer().start_span(name, attributes)


def get_current_span() -> Optional[Span]:
    """Get the current active span"""
    return get_tracer().get_current_span()


def trace_function(name: str = None, attributes: Dict[str, Any] = None):
    """
    Decorator to trace a function
    
    Usage:
        @trace_function("my_operation")
        async def my_function():
            ...
    """
    def decorator(func: Callable):
        span_name = name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with get_tracer().start_span(span_name, attributes) as span:
                span.set_attribute("function", func.__name__)
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with get_tracer().start_span(span_name, attributes) as span:
                span.set_attribute("function", func.__name__)
                return func(*args, **kwargs)
        
        if asyncio_iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def asyncio_iscoroutinefunction(func):
    """Check if function is async"""
    import asyncio
    return asyncio.iscoroutinefunction(func)

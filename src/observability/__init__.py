"""
NIS Protocol Observability Module

Features:
- Distributed Tracing (OpenTelemetry)
- Structured Logging (JSON format)
- Enhanced Prometheus Metrics
- Performance Monitoring
"""

from .tracing import (
    TracingManager,
    get_tracer,
    trace_function,
    create_span,
    get_current_span
)

from .logging import (
    StructuredLogger,
    get_logger,
    configure_logging,
    LogLevel
)

from .metrics import (
    MetricsCollector,
    get_metrics,
    Counter,
    Gauge,
    Histogram
)

__all__ = [
    # Tracing
    'TracingManager',
    'get_tracer',
    'trace_function',
    'create_span',
    'get_current_span',
    # Logging
    'StructuredLogger',
    'get_logger',
    'configure_logging',
    'LogLevel',
    # Metrics
    'MetricsCollector',
    'get_metrics',
    'Counter',
    'Gauge',
    'Histogram'
]

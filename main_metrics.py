"""
Production Monitoring Metrics for NIS Protocol v4.0
Adds Prometheus-compatible metrics to all endpoints
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
from functools import wraps
from typing import Callable
import logging

# Initialize logger
logger = logging.getLogger("nis.metrics")

# Define metrics
request_count = Counter(
    'nis_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'nis_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

active_requests = Gauge(
    'nis_active_requests',
    'Number of active requests'
)

error_count = Counter(
    'nis_errors_total',
    'Total errors',
    ['endpoint', 'error_type']
)

# Phase-specific metrics
phase_execution_count = Counter(
    'nis_phase_executions_total',
    'Total phase executions',
    ['phase_name']
)

phase_execution_duration = Histogram(
    'nis_phase_duration_seconds',
    'Phase execution duration',
    ['phase_name']
)

# System metrics
system_memory_usage = Gauge('nis_memory_usage_bytes', 'Memory usage in bytes')
system_cpu_usage = Gauge('nis_cpu_usage_percent', 'CPU usage percentage')


def track_metrics(endpoint: str):
    """
    Decorator to track metrics for endpoints
    Usage:
    @track_metrics("evolution")
    async def evolve_endpoint(...):
        ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            active_requests.inc()
            
            try:
                result = await func(*args, **kwargs)
                status = "success"
                request_count.labels(
                    method="POST",
                    endpoint=endpoint,
                    status=status
                ).inc()
                return result
            except Exception as e:
                status = "error"
                error_count.labels(
                    endpoint=endpoint,
                    error_type=type(e).__name__
                ).inc()
                logger.error(f"Error in {endpoint}: {e}")
                raise
            finally:
                duration = time.time() - start_time
                request_duration.labels(
                    method="POST",
                    endpoint=endpoint
                ).observe(duration)
                active_requests.dec()
        
        return wrapper
    return decorator


def track_phase(phase_name: str):
    """
    Decorator to track phase-specific metrics
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            phase_execution_count.labels(phase_name=phase_name).inc()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                phase_execution_duration.labels(phase_name=phase_name).observe(duration)
        
        return wrapper
    return decorator


# To add to main.py:
"""
from main_metrics import track_metrics, generate_latest

@app.get("/metrics")
async def metrics():
    '''Prometheus metrics endpoint'''
    return Response(content=generate_latest(), media_type="text/plain")

@app.post("/v4/consciousness/evolve")
@track_metrics("evolution")
async def evolve_consciousness(...):
    ...
"""

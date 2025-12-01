#!/usr/bin/env python3
"""
Prometheus Metrics for NIS Protocol
Exposes metrics for monitoring with Prometheus/Grafana

Usage:
    from src.monitoring.prometheus_metrics import metrics, track_request
    
    @track_request("chat")
    async def chat_endpoint(...):
        ...
"""

import time
import functools
from typing import Callable, Any
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST

# =============================================================================
# METRICS DEFINITIONS
# =============================================================================

# Request metrics
REQUEST_COUNT = Counter(
    'nis_requests_total',
    'Total number of requests',
    ['endpoint', 'method', 'status']
)

REQUEST_LATENCY = Histogram(
    'nis_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# LLM metrics
LLM_REQUESTS = Counter(
    'nis_llm_requests_total',
    'Total LLM API requests',
    ['provider', 'model', 'success']
)

LLM_TOKENS = Counter(
    'nis_llm_tokens_total',
    'Total tokens used',
    ['provider', 'type']  # type: input/output
)

LLM_LATENCY = Histogram(
    'nis_llm_latency_seconds',
    'LLM response latency',
    ['provider'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

LLM_FALLBACKS = Counter(
    'nis_llm_fallbacks_total',
    'Number of provider fallbacks',
    ['from_provider', 'to_provider']
)

# System metrics
ACTIVE_CONVERSATIONS = Gauge(
    'nis_active_conversations',
    'Number of active conversations'
)

ACTIVE_AGENTS = Gauge(
    'nis_active_agents',
    'Number of registered agents'
)

CACHE_HITS = Counter(
    'nis_cache_hits_total',
    'Cache hit count',
    ['cache_type']
)

CACHE_MISSES = Counter(
    'nis_cache_misses_total',
    'Cache miss count',
    ['cache_type']
)

# Consciousness metrics
CONSCIOUSNESS_LEVEL = Gauge(
    'nis_consciousness_level',
    'Current consciousness level (0-1)'
)

ETHICS_SCORE = Gauge(
    'nis_ethics_score',
    'Current ethics score (0-1)'
)

# Error metrics
ERRORS = Counter(
    'nis_errors_total',
    'Total errors',
    ['error_type', 'endpoint']
)

# System info
SYSTEM_INFO = Info(
    'nis_system',
    'NIS Protocol system information'
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def track_request(endpoint: str):
    """Decorator to track request metrics"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            status = "success"
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                ERRORS.labels(error_type=type(e).__name__, endpoint=endpoint).inc()
                raise
            finally:
                latency = time.time() - start_time
                REQUEST_COUNT.labels(endpoint=endpoint, method="POST", status=status).inc()
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
        return wrapper
    return decorator


def track_llm_request(provider: str, model: str, success: bool, latency: float, tokens: int = 0):
    """Track LLM request metrics"""
    LLM_REQUESTS.labels(provider=provider, model=model, success=str(success)).inc()
    LLM_LATENCY.labels(provider=provider).observe(latency)
    if tokens > 0:
        LLM_TOKENS.labels(provider=provider, type="total").inc(tokens)


def track_fallback(from_provider: str, to_provider: str):
    """Track provider fallback"""
    LLM_FALLBACKS.labels(from_provider=from_provider, to_provider=to_provider).inc()


def track_cache(cache_type: str, hit: bool):
    """Track cache hit/miss"""
    if hit:
        CACHE_HITS.labels(cache_type=cache_type).inc()
    else:
        CACHE_MISSES.labels(cache_type=cache_type).inc()


def update_consciousness_metrics(level: float, ethics: float):
    """Update consciousness metrics"""
    CONSCIOUSNESS_LEVEL.set(level)
    ETHICS_SCORE.set(ethics)


def set_system_info(version: str, environment: str):
    """Set system info"""
    SYSTEM_INFO.info({
        'version': version,
        'environment': environment,
        'protocol': 'NIS v4.0'
    })


def get_metrics() -> bytes:
    """Get all metrics in Prometheus format"""
    return generate_latest()


def get_metrics_content_type() -> str:
    """Get content type for metrics response"""
    return CONTENT_TYPE_LATEST


# =============================================================================
# METRICS CLASS FOR EASY ACCESS
# =============================================================================

class NISMetrics:
    """Centralized metrics access"""
    
    request_count = REQUEST_COUNT
    request_latency = REQUEST_LATENCY
    llm_requests = LLM_REQUESTS
    llm_tokens = LLM_TOKENS
    llm_latency = LLM_LATENCY
    llm_fallbacks = LLM_FALLBACKS
    active_conversations = ACTIVE_CONVERSATIONS
    active_agents = ACTIVE_AGENTS
    cache_hits = CACHE_HITS
    cache_misses = CACHE_MISSES
    consciousness_level = CONSCIOUSNESS_LEVEL
    ethics_score = ETHICS_SCORE
    errors = ERRORS
    
    @staticmethod
    def track_request(endpoint: str):
        return track_request(endpoint)
    
    @staticmethod
    def track_llm(provider: str, model: str, success: bool, latency: float, tokens: int = 0):
        track_llm_request(provider, model, success, latency, tokens)
    
    @staticmethod
    def track_fallback(from_prov: str, to_prov: str):
        track_fallback(from_prov, to_prov)
    
    @staticmethod
    def get_prometheus_metrics() -> bytes:
        return get_metrics()


# Global instance
metrics = NISMetrics()

#!/usr/bin/env python3
"""
NIS Protocol Enhanced Metrics
Prometheus-compatible metrics collection

Features:
- Counter, Gauge, Histogram metrics
- Automatic labeling
- Prometheus export format
- Real-time statistics
"""

import os
import time
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import math

import logging

logger = logging.getLogger("nis.observability.metrics")


@dataclass
class MetricValue:
    """A single metric value with labels"""
    value: float
    labels: Dict[str, str]
    timestamp: float = field(default_factory=time.time)


class Counter:
    """
    A counter metric that only increases
    
    Usage:
        requests = Counter("http_requests_total", "Total HTTP requests", ["method", "path"])
        requests.inc(method="GET", path="/health")
    """
    
    def __init__(self, name: str, description: str, labels: List[str] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()
    
    def inc(self, value: float = 1, **labels):
        """Increment the counter"""
        label_key = tuple(labels.get(l, "") for l in self.label_names)
        with self._lock:
            self._values[label_key] += value
    
    def get(self, **labels) -> float:
        """Get current counter value"""
        label_key = tuple(labels.get(l, "") for l in self.label_names)
        return self._values.get(label_key, 0)
    
    def get_all(self) -> List[MetricValue]:
        """Get all counter values"""
        result = []
        for label_key, value in self._values.items():
            labels = dict(zip(self.label_names, label_key))
            result.append(MetricValue(value=value, labels=labels))
        return result
    
    def to_prometheus(self) -> str:
        """Export in Prometheus format"""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} counter"
        ]
        for label_key, value in self._values.items():
            labels = dict(zip(self.label_names, label_key))
            label_str = ",".join(f'{k}="{v}"' for k, v in labels.items() if v)
            if label_str:
                lines.append(f"{self.name}{{{label_str}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        return "\n".join(lines)


class Gauge:
    """
    A gauge metric that can increase or decrease
    
    Usage:
        active_connections = Gauge("active_connections", "Active connections", ["service"])
        active_connections.set(10, service="api")
        active_connections.inc(service="api")
        active_connections.dec(service="api")
    """
    
    def __init__(self, name: str, description: str, labels: List[str] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()
    
    def set(self, value: float, **labels):
        """Set the gauge value"""
        label_key = tuple(labels.get(l, "") for l in self.label_names)
        with self._lock:
            self._values[label_key] = value
    
    def inc(self, value: float = 1, **labels):
        """Increment the gauge"""
        label_key = tuple(labels.get(l, "") for l in self.label_names)
        with self._lock:
            self._values[label_key] += value
    
    def dec(self, value: float = 1, **labels):
        """Decrement the gauge"""
        label_key = tuple(labels.get(l, "") for l in self.label_names)
        with self._lock:
            self._values[label_key] -= value
    
    def get(self, **labels) -> float:
        """Get current gauge value"""
        label_key = tuple(labels.get(l, "") for l in self.label_names)
        return self._values.get(label_key, 0)
    
    def get_all(self) -> List[MetricValue]:
        """Get all gauge values"""
        result = []
        for label_key, value in self._values.items():
            labels = dict(zip(self.label_names, label_key))
            result.append(MetricValue(value=value, labels=labels))
        return result
    
    def to_prometheus(self) -> str:
        """Export in Prometheus format"""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} gauge"
        ]
        for label_key, value in self._values.items():
            labels = dict(zip(self.label_names, label_key))
            label_str = ",".join(f'{k}="{v}"' for k, v in labels.items() if v)
            if label_str:
                lines.append(f"{self.name}{{{label_str}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        return "\n".join(lines)


class Histogram:
    """
    A histogram metric for measuring distributions
    
    Usage:
        request_duration = Histogram("request_duration_seconds", "Request duration", 
                                     ["method"], buckets=[0.01, 0.05, 0.1, 0.5, 1, 5])
        request_duration.observe(0.123, method="GET")
    """
    
    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
    
    def __init__(self, name: str, description: str, labels: List[str] = None, 
                 buckets: List[float] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        
        self._counts: Dict[tuple, Dict[float, int]] = defaultdict(lambda: defaultdict(int))
        self._sums: Dict[tuple, float] = defaultdict(float)
        self._totals: Dict[tuple, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    def observe(self, value: float, **labels):
        """Observe a value"""
        label_key = tuple(labels.get(l, "") for l in self.label_names)
        with self._lock:
            self._sums[label_key] += value
            self._totals[label_key] += 1
            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[label_key][bucket] += 1
    
    def get_stats(self, **labels) -> Dict[str, Any]:
        """Get histogram statistics"""
        label_key = tuple(labels.get(l, "") for l in self.label_names)
        total = self._totals.get(label_key, 0)
        sum_val = self._sums.get(label_key, 0)
        
        return {
            "count": total,
            "sum": sum_val,
            "avg": sum_val / total if total > 0 else 0,
            "buckets": dict(self._counts.get(label_key, {}))
        }
    
    def to_prometheus(self) -> str:
        """Export in Prometheus format"""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} histogram"
        ]
        
        for label_key in self._totals.keys():
            labels = dict(zip(self.label_names, label_key))
            label_str = ",".join(f'{k}="{v}"' for k, v in labels.items() if v)
            
            # Bucket values
            cumulative = 0
            for bucket in self.buckets:
                cumulative += self._counts[label_key].get(bucket, 0)
                bucket_labels = f'{label_str},le="{bucket}"' if label_str else f'le="{bucket}"'
                lines.append(f"{self.name}_bucket{{{bucket_labels}}} {cumulative}")
            
            # +Inf bucket
            inf_labels = f'{label_str},le="+Inf"' if label_str else 'le="+Inf"'
            lines.append(f"{self.name}_bucket{{{inf_labels}}} {self._totals[label_key]}")
            
            # Sum and count
            if label_str:
                lines.append(f"{self.name}_sum{{{label_str}}} {self._sums[label_key]}")
                lines.append(f"{self.name}_count{{{label_str}}} {self._totals[label_key]}")
            else:
                lines.append(f"{self.name}_sum {self._sums[label_key]}")
                lines.append(f"{self.name}_count {self._totals[label_key]}")
        
        return "\n".join(lines)


class MetricsCollector:
    """
    Central metrics collector for NIS Protocol
    
    Pre-defined metrics for common operations
    """
    
    def __init__(self):
        # HTTP metrics
        self.http_requests = Counter(
            "nis_http_requests_total",
            "Total HTTP requests",
            ["method", "path", "status"]
        )
        self.http_duration = Histogram(
            "nis_http_request_duration_seconds",
            "HTTP request duration",
            ["method", "path"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
        )
        
        # System metrics
        self.uptime = Gauge("nis_uptime_seconds", "System uptime")
        self.health_status = Gauge("nis_health_status", "Health status (1=healthy)")
        
        # Infrastructure metrics
        self.kafka_messages = Counter(
            "nis_kafka_messages_total",
            "Kafka messages",
            ["topic", "direction"]
        )
        self.redis_operations = Counter(
            "nis_redis_operations_total",
            "Redis operations",
            ["operation", "status"]
        )
        
        # Robotics metrics
        self.robotics_commands = Counter(
            "nis_robotics_commands_total",
            "Robotics commands",
            ["robot_type", "command"]
        )
        self.kinematics_duration = Histogram(
            "nis_kinematics_duration_seconds",
            "Kinematics computation duration",
            ["type"]
        )
        
        # AI metrics
        self.llm_requests = Counter(
            "nis_llm_requests_total",
            "LLM requests",
            ["provider", "model"]
        )
        self.llm_tokens = Counter(
            "nis_llm_tokens_total",
            "LLM tokens used",
            ["provider", "direction"]
        )
        self.llm_duration = Histogram(
            "nis_llm_request_duration_seconds",
            "LLM request duration",
            ["provider"]
        )
        
        # Physics metrics
        self.physics_validations = Counter(
            "nis_physics_validations_total",
            "Physics validations",
            ["domain", "result"]
        )
        
        # Security metrics
        self.auth_attempts = Counter(
            "nis_auth_attempts_total",
            "Authentication attempts",
            ["result"]
        )
        self.rate_limit_hits = Counter(
            "nis_rate_limit_hits_total",
            "Rate limit hits",
            ["endpoint"]
        )
        
        # Active connections
        self.active_connections = Gauge(
            "nis_active_connections",
            "Active connections",
            ["type"]
        )
        
        # Start time for uptime calculation
        self._start_time = time.time()
        
        logger.info("Metrics collector initialized")
    
    def record_request(self, method: str, path: str, status: int, duration: float):
        """Record an HTTP request"""
        self.http_requests.inc(method=method, path=path, status=str(status))
        self.http_duration.observe(duration, method=method, path=path)
    
    def record_llm_request(self, provider: str, model: str, 
                           input_tokens: int, output_tokens: int, duration: float):
        """Record an LLM request"""
        self.llm_requests.inc(provider=provider, model=model)
        self.llm_tokens.inc(input_tokens, provider=provider, direction="input")
        self.llm_tokens.inc(output_tokens, provider=provider, direction="output")
        self.llm_duration.observe(duration, provider=provider)
    
    def record_robotics_command(self, robot_type: str, command: str, duration: float = None):
        """Record a robotics command"""
        self.robotics_commands.inc(robot_type=robot_type, command=command)
        if duration:
            self.kinematics_duration.observe(duration, type=command)
    
    def record_physics_validation(self, domain: str, is_valid: bool):
        """Record a physics validation"""
        result = "valid" if is_valid else "invalid"
        self.physics_validations.inc(domain=domain, result=result)
    
    def to_prometheus(self) -> str:
        """Export all metrics in Prometheus format"""
        # Update uptime
        self.uptime.set(time.time() - self._start_time)
        self.health_status.set(1)
        
        metrics = [
            self.http_requests,
            self.http_duration,
            self.uptime,
            self.health_status,
            self.kafka_messages,
            self.redis_operations,
            self.robotics_commands,
            self.kinematics_duration,
            self.llm_requests,
            self.llm_tokens,
            self.llm_duration,
            self.physics_validations,
            self.auth_attempts,
            self.rate_limit_hits,
            self.active_connections
        ]
        
        return "\n\n".join(m.to_prometheus() for m in metrics)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary as JSON"""
        return {
            "uptime_seconds": time.time() - self._start_time,
            "http": {
                "total_requests": sum(self.http_requests._values.values()),
                "duration_stats": self.http_duration.get_stats()
            },
            "llm": {
                "total_requests": sum(self.llm_requests._values.values()),
                "total_tokens": sum(self.llm_tokens._values.values())
            },
            "robotics": {
                "total_commands": sum(self.robotics_commands._values.values())
            },
            "physics": {
                "total_validations": sum(self.physics_validations._values.values())
            }
        }


# Singleton instance
_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get the metrics collector singleton"""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics

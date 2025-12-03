"""
NIS Protocol v3 - Enhanced Infrastructure Module

This module provides comprehensive Kafka and Redis integration with self-audit
capabilities for all NIS Protocol agents.

Features:
- Async Kafka message streaming with integrity monitoring
- Redis caching with performance tracking
- Auto-failover and resilience patterns
- Integration with self-audit engine
- Performance monitoring and optimization
"""

from .message_streaming import (
    NISKafkaManager,
    MessageType,
    StreamingTopics,
    MessagePriority
)

from .caching_system import (
    NISRedisManager,
    CacheStrategy,
    CacheMetrics,
    PerformanceTracker
)

from .integration_coordinator import (
    InfrastructureCoordinator,
    ServiceHealth,
    IntegrationStatus
)

__all__ = [
    'NISKafkaManager',
    'NISRedisManager', 
    'InfrastructureCoordinator',
    'MessageType',
    'StreamingTopics',
    'MessagePriority',
    'CacheStrategy',
    'CacheMetrics',
    'PerformanceTracker',
    'ServiceHealth',
    'IntegrationStatus'
] 
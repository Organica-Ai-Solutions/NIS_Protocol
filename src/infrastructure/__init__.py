"""
NIS Protocol v4 - Enhanced Infrastructure Module

This module provides comprehensive Kafka and Redis integration with self-audit
capabilities for all NIS Protocol agents.

Features:
- Async Kafka message streaming with integrity monitoring
- Redis caching with performance tracking
- Auto-failover and resilience patterns
- Integration with self-audit engine
- Performance monitoring and optimization
- OBD-II automotive data streaming
- CAN bus message routing
- Unified NIS infrastructure management
"""

# New unified message broker
try:
    from .message_broker import (
        KafkaMessageBroker,
        RedisCache,
        InfrastructureManager,
        KafkaConfig,
        RedisConfig,
        MessageType as BrokerMessageType,
        get_infrastructure_manager,
        KAFKA_AVAILABLE,
        REDIS_AVAILABLE
    )
except ImportError:
    KafkaMessageBroker = None
    RedisCache = None
    InfrastructureManager = None
    KAFKA_AVAILABLE = False
    REDIS_AVAILABLE = False

# NIS Infrastructure (unified access)
try:
    from .nis_infrastructure import (
        NISInfrastructure,
        NISEventType,
        CacheNamespace,
        NISInfrastructureConfig,
        get_nis_infrastructure,
        initialize_infrastructure,
        shutdown_infrastructure
    )
except ImportError:
    NISInfrastructure = None
    NISEventType = None
    CacheNamespace = None
    get_nis_infrastructure = None
    initialize_infrastructure = None
    shutdown_infrastructure = None

# Legacy imports (for backward compatibility)
try:
    from .message_streaming import (
        NISKafkaManager,
        MessageType,
        StreamingTopics,
        MessagePriority
    )
except ImportError:
    NISKafkaManager = None
    MessageType = None
    StreamingTopics = None
    MessagePriority = None

try:
    from .caching_system import (
        NISRedisManager,
        CacheStrategy,
        CacheMetrics,
        PerformanceTracker
    )
except ImportError:
    NISRedisManager = None
    CacheStrategy = None
    CacheMetrics = None
    PerformanceTracker = None

try:
    from .integration_coordinator import (
        InfrastructureCoordinator,
        ServiceHealth,
        IntegrationStatus
    )
except ImportError:
    InfrastructureCoordinator = None
    ServiceHealth = None
    IntegrationStatus = None

__all__ = [
    # New unified broker
    'KafkaMessageBroker',
    'RedisCache',
    'InfrastructureManager',
    'KafkaConfig',
    'RedisConfig',
    'get_infrastructure_manager',
    'KAFKA_AVAILABLE',
    'REDIS_AVAILABLE',
    # Legacy
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
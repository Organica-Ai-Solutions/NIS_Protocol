"""
NIS Protocol v3 - Infrastructure Integration Coordinator

This module provides centralized coordination of Kafka and Redis services
with health monitoring, auto-recovery, and seamless integration for all agents.

Features:
- Unified interface for message streaming and caching
- Health monitoring and auto-recovery
- Load balancing and failover management
- Performance optimization and monitoring
- Self-audit integration across infrastructure
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from datetime import datetime, timedelta

# Infrastructure components
from .message_streaming import NISKafkaManager, NISMessage, MessageType, MessagePriority, StreamingTopics
from .caching_system import NISRedisManager, CacheStrategy, CacheNamespace

# Self-audit integration
from src.utils.self_audit import self_audit_engine


class ServiceHealth(Enum):
    """Health status of infrastructure services"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class IntegrationStatus(Enum):
    """Overall integration status"""
    FULLY_OPERATIONAL = "fully_operational"
    PARTIALLY_OPERATIONAL = "partially_operational"
    DEGRADED = "degraded"
    OFFLINE = "offline"


@dataclass
class ServiceStatus:
    """Status information for a service"""
    service_name: str
    health: ServiceHealth
    last_check: float
    uptime: float
    error_count: int
    response_time: float
    metadata: Dict[str, Any]


@dataclass
class InfrastructureMetrics:
    """Comprehensive infrastructure metrics"""
    kafka_metrics: Dict[str, Any]
    redis_metrics: Dict[str, Any]
    overall_health: ServiceHealth
    integration_status: IntegrationStatus
    total_messages: int
    total_cache_operations: int
    error_rate: float
    avg_response_time: float
    uptime: float
    last_update: float


class InfrastructureCoordinator:
    """
    Central coordinator for Kafka and Redis infrastructure with health monitoring,
    auto-recovery, and unified agent interface.
    
    Features:
    - Unified infrastructure management
    - Health monitoring and auto-recovery
    - Load balancing and performance optimization
    - Self-audit integration
    - Seamless agent integration
    """
    
    def __init__(
        self,
        kafka_config: Optional[Dict[str, Any]] = None,
        redis_config: Optional[Dict[str, Any]] = None,
        enable_self_audit: bool = True,
        health_check_interval: float = 30.0,
        auto_recovery: bool = True
    ):
        """Initialize the infrastructure coordinator"""
        self.kafka_config = kafka_config or {}
        self.redis_config = redis_config or {}
        self.enable_self_audit = enable_self_audit
        self.health_check_interval = health_check_interval
        self.auto_recovery = auto_recovery
        
        # Initialize components
        self.logger = logging.getLogger("nis.infrastructure_coordinator")
        self.kafka_manager: Optional[NISKafkaManager] = None
        self.redis_manager: Optional[NISRedisManager] = None
        
        # Service status tracking
        self.service_status: Dict[str, ServiceStatus] = {}
        self.integration_metrics = InfrastructureMetrics(
            kafka_metrics={},
            redis_metrics={},
            overall_health=ServiceHealth.UNKNOWN,
            integration_status=IntegrationStatus.OFFLINE,
            total_messages=0,
            total_cache_operations=0,
            error_rate=0.0,
            avg_response_time=0.0,
            uptime=0.0,
            last_update=time.time()
        )
        
        # Health monitoring
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        self.start_time = time.time()
        
        # Message routing
        self.message_routes: Dict[str, List[Callable]] = {}
        self.agent_subscriptions: Dict[str, List[str]] = {}
        
        # Self-audit integration
        if self.enable_self_audit:
            self._init_self_audit()
        
        self.logger.info("InfrastructureCoordinator initialized")
    
    def _init_self_audit(self):
        """Initialize self-audit capabilities for infrastructure"""
        try:
            self.self_audit_enabled = True
            self.audit_threshold = 80.0
            self.infrastructure_violations: List[Dict[str, Any]] = []
            
            self.logger.info("Self-audit integration enabled for infrastructure")
        except Exception as e:
            self.logger.error(f"Failed to initialize self-audit: {e}")
            self.self_audit_enabled = False
    
    async def initialize(self) -> bool:
        """Initialize all infrastructure components"""
        try:
            self.logger.info("Initializing infrastructure components...")
            
            # Initialize Kafka manager
            kafka_success = await self._initialize_kafka()
            
            # Initialize Redis manager
            redis_success = await self._initialize_redis()
            
            # Determine overall status
            if kafka_success and redis_success:
                self.integration_metrics.integration_status = IntegrationStatus.FULLY_OPERATIONAL
                self.integration_metrics.overall_health = ServiceHealth.HEALTHY
            elif kafka_success or redis_success:
                self.integration_metrics.integration_status = IntegrationStatus.PARTIALLY_OPERATIONAL
                self.integration_metrics.overall_health = ServiceHealth.DEGRADED
            else:
                self.integration_metrics.integration_status = IntegrationStatus.OFFLINE
                self.integration_metrics.overall_health = ServiceHealth.UNHEALTHY
            
            # Start health monitoring
            if kafka_success or redis_success:
                await self.start_health_monitoring()
            
            self.logger.info(f"Infrastructure initialization complete: {self.integration_metrics.integration_status.value}")
            return kafka_success or redis_success
            
        except Exception as e:
            self.logger.error(f"Failed to initialize infrastructure: {e}")
            return False
    
    async def _initialize_kafka(self) -> bool:
        """Initialize Kafka manager"""
        try:
            self.kafka_manager = NISKafkaManager(
                bootstrap_servers=self.kafka_config.get("bootstrap_servers", ["kafka:9092"]),
                enable_self_audit=self.enable_self_audit,
                **self.kafka_config.get("options", {})
            )
            
            success = await self.kafka_manager.initialize()
            
            if success:
                # Set up default topic consumers
                await self._setup_default_consumers()
                
                self.service_status["kafka"] = ServiceStatus(
                    service_name="kafka",
                    health=ServiceHealth.HEALTHY,
                    last_check=time.time(),
                    uptime=0.0,
                    error_count=0,
                    response_time=0.0,
                    metadata={}
                )
                
                self.logger.info("Kafka manager initialized successfully")
            else:
                self.service_status["kafka"] = ServiceStatus(
                    service_name="kafka",
                    health=ServiceHealth.UNHEALTHY,
                    last_check=time.time(),
                    uptime=0.0,
                    error_count=1,
                    response_time=0.0,
                    metadata={"error": "initialization_failed"}
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Kafka: {e}")
            return False
    
    async def _initialize_redis(self) -> bool:
        """Initialize Redis manager"""
        try:
            self.redis_manager = NISRedisManager(
                host=self.redis_config.get("host", "redis"),
                port=self.redis_config.get("port", 6379),
                db=self.redis_config.get("db", 0),
                enable_self_audit=self.enable_self_audit,
                **self.redis_config.get("options", {})
            )
            
            success = await self.redis_manager.initialize()
            
            if success:
                self.service_status["redis"] = ServiceStatus(
                    service_name="redis",
                    health=ServiceHealth.HEALTHY,
                    last_check=time.time(),
                    uptime=0.0,
                    error_count=0,
                    response_time=0.0,
                    metadata={}
                )
                
                self.logger.info("Redis manager initialized successfully")
            else:
                self.service_status["redis"] = ServiceStatus(
                    service_name="redis",
                    health=ServiceHealth.UNHEALTHY,
                    last_check=time.time(),
                    uptime=0.0,
                    error_count=1,
                    response_time=0.0,
                    metadata={"error": "initialization_failed"}
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis: {e}")
            return False
    
    async def _setup_default_consumers(self):
        """Set up default Kafka consumers for system topics"""
        if not self.kafka_manager:
            return
        
        # System health consumer
        await self.kafka_manager.create_consumer(
            StreamingTopics.SYSTEM_HEALTH,
            "infrastructure-health",
            self._handle_system_health_message
        )
        
        # Audit alerts consumer
        await self.kafka_manager.create_consumer(
            StreamingTopics.AUDIT_ALERTS,
            "infrastructure-audit",
            self._handle_audit_alert_message
        )
        
        # Performance metrics consumer
        await self.kafka_manager.create_consumer(
            StreamingTopics.PERFORMANCE,
            "infrastructure-performance",
            self._handle_performance_message
        )
    
    async def start_health_monitoring(self):
        """Start continuous health monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        self.logger.info("Health monitoring started")
    
    async def stop_health_monitoring(self):
        """Stop health monitoring"""
        self.is_monitoring = False
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Health monitoring stopped")
    
    async def _health_monitor_loop(self):
        """Continuous health monitoring loop"""
        while self.is_monitoring:
            try:
                await self._perform_health_checks()
                await self._update_metrics()
                
                # Auto-recovery if enabled
                if self.auto_recovery:
                    await self._perform_auto_recovery()
                
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _perform_health_checks(self):
        """Perform health checks on all services"""
        current_time = time.time()
        
        # Check Kafka health
        if self.kafka_manager:
            kafka_health = self.kafka_manager.get_health_status()
            self.service_status["kafka"].health = (
                ServiceHealth.HEALTHY if kafka_health["initialized"] 
                else ServiceHealth.UNHEALTHY
            )
            self.service_status["kafka"].last_check = current_time
        
        # Check Redis health
        if self.redis_manager:
            redis_health = self.redis_manager.get_health_status()
            self.service_status["redis"].health = (
                ServiceHealth.HEALTHY if redis_health["initialized"]
                else ServiceHealth.UNHEALTHY
            )
            self.service_status["redis"].last_check = current_time
        
        # Update overall health
        healthy_services = sum(1 for status in self.service_status.values() 
                             if status.health == ServiceHealth.HEALTHY)
        total_services = len(self.service_status)
        
        if healthy_services == total_services:
            self.integration_metrics.overall_health = ServiceHealth.HEALTHY
            self.integration_metrics.integration_status = IntegrationStatus.FULLY_OPERATIONAL
        elif healthy_services > 0:
            self.integration_metrics.overall_health = ServiceHealth.DEGRADED
            self.integration_metrics.integration_status = IntegrationStatus.PARTIALLY_OPERATIONAL
        else:
            self.integration_metrics.overall_health = ServiceHealth.UNHEALTHY
            self.integration_metrics.integration_status = IntegrationStatus.OFFLINE
    
    async def _update_metrics(self):
        """Update comprehensive infrastructure metrics"""
        current_time = time.time()
        
        # Update uptime
        self.integration_metrics.uptime = current_time - self.start_time
        
        # Get Kafka metrics
        if self.kafka_manager:
            kafka_metrics = self.kafka_manager.get_metrics()
            self.integration_metrics.kafka_metrics = asdict(kafka_metrics)
            self.integration_metrics.total_messages = kafka_metrics.messages_sent + kafka_metrics.messages_received
        
        # Get Redis metrics
        if self.redis_manager:
            redis_metrics = self.redis_manager.get_metrics()
            self.integration_metrics.redis_metrics = asdict(redis_metrics)
            self.integration_metrics.total_cache_operations = redis_metrics.hits + redis_metrics.misses + redis_metrics.sets
        
        # Calculate error rate
        total_operations = self.integration_metrics.total_messages + self.integration_metrics.total_cache_operations
        total_errors = (
            self.integration_metrics.kafka_metrics.get("errors_encountered", 0) +
            self.integration_metrics.redis_metrics.get("errors_encountered", 0)
        )
        
        if total_operations > 0:
            self.integration_metrics.error_rate = total_errors / total_operations
        
        self.integration_metrics.last_update = current_time
    
    async def _perform_auto_recovery(self):
        """Perform auto-recovery for unhealthy services"""
        for service_name, status in self.service_status.items():
            if status.health == ServiceHealth.UNHEALTHY:
                self.logger.info(f"Attempting auto-recovery for {service_name}")
                
                if service_name == "kafka" and self.kafka_manager:
                    try:
                        await self.kafka_manager.initialize()
                        self.logger.info("Kafka auto-recovery successful")
                    except Exception as e:
                        self.logger.error(f"Kafka auto-recovery failed: {e}")
                
                elif service_name == "redis" and self.redis_manager:
                    try:
                        await self.redis_manager.initialize()
                        self.logger.info("Redis auto-recovery successful")
                    except Exception as e:
                        self.logger.error(f"Redis auto-recovery failed: {e}")
    
    # =============================================================================
    # UNIFIED AGENT INTERFACE
    # =============================================================================
    
    async def send_message(
        self,
        message_type: MessageType,
        content: Dict[str, Any],
        source_agent: str,
        target_agent: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        topic: Optional[str] = None
    ) -> bool:
        """
        Unified interface for agents to send messages
        
        Args:
            message_type: Type of message
            content: Message content
            source_agent: Source agent ID
            target_agent: Target agent ID (optional)
            priority: Message priority
            topic: Kafka topic (auto-determined if not provided)
            
        Returns:
            bool: True if message sent successfully
        """
        if not self.kafka_manager:
            self.logger.warning("Kafka manager not available")
            return False
        
        # Auto-determine topic if not provided
        if topic is None:
            topic = self._determine_topic(message_type)
        
        # Create NIS message
        message = NISMessage(
            message_id=f"{source_agent}_{int(time.time() * 1000)}",
            message_type=message_type,
            priority=priority,
            source_agent=source_agent,
            target_agent=target_agent,
            topic=topic,
            content=content,
            timestamp=time.time()
        )
        
        # Send message
        return await self.kafka_manager.send_message(message)
    
    async def cache_data(
        self,
        key: str,
        value: Any,
        agent_id: str,
        ttl: Optional[int] = None,
        strategy: CacheStrategy = CacheStrategy.TTL
    ) -> bool:
        """
        Unified interface for agents to cache data
        
        Args:
            key: Cache key
            value: Value to cache
            agent_id: Agent ID for namespace
            ttl: Time to live in seconds
            strategy: Cache strategy
            
        Returns:
            bool: True if data cached successfully
        """
        if not self.redis_manager:
            self.logger.warning("Redis manager not available")
            return False
        
        # Use agent ID as namespace
        namespace = self._determine_cache_namespace(agent_id)
        
        return await self.redis_manager.set(key, value, namespace, ttl, strategy)
    
    async def get_cached_data(
        self,
        key: str,
        agent_id: str,
        default: Any = None
    ) -> Any:
        """
        Unified interface for agents to retrieve cached data
        
        Args:
            key: Cache key
            agent_id: Agent ID for namespace
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        if not self.redis_manager:
            self.logger.warning("Redis manager not available")
            return default
        
        namespace = self._determine_cache_namespace(agent_id)
        return await self.redis_manager.get(key, namespace, default)
    
    async def subscribe_to_messages(
        self,
        agent_id: str,
        message_types: List[MessageType],
        handler: Callable[[NISMessage], None]
    ) -> bool:
        """
        Subscribe agent to specific message types
        
        Args:
            agent_id: Agent ID
            message_types: List of message types to subscribe to
            handler: Message handler function
            
        Returns:
            bool: True if subscription successful
        """
        if not self.kafka_manager:
            self.logger.warning("Kafka manager not available")
            return False
        
        for message_type in message_types:
            topic = self._determine_topic(message_type)
            group_id = f"{agent_id}_group"
            
            success = await self.kafka_manager.create_consumer(topic, group_id, handler)
            if not success:
                return False
            
            # Track subscription
            if agent_id not in self.agent_subscriptions:
                self.agent_subscriptions[agent_id] = []
            self.agent_subscriptions[agent_id].append(topic)
        
        self.logger.info(f"Agent {agent_id} subscribed to {len(message_types)} message types")
        return True
    
    def _determine_topic(self, message_type: MessageType) -> str:
        """Determine Kafka topic based on message type"""
        topic_mapping = {
            MessageType.CONSCIOUSNESS_EVENT: StreamingTopics.CONSCIOUSNESS,
            MessageType.GOAL_GENERATION: StreamingTopics.GOALS,
            MessageType.SIMULATION_RESULT: StreamingTopics.SIMULATION,
            MessageType.ALIGNMENT_CHECK: StreamingTopics.ALIGNMENT,
            MessageType.MEMORY_OPERATION: StreamingTopics.MEMORY,
            MessageType.AGENT_COORDINATION: StreamingTopics.COORDINATION,
            MessageType.SYSTEM_HEALTH: StreamingTopics.SYSTEM_HEALTH,
            MessageType.AUDIT_ALERT: StreamingTopics.AUDIT_ALERTS,
            MessageType.PERFORMANCE_METRIC: StreamingTopics.PERFORMANCE
        }
        
        return topic_mapping.get(message_type, StreamingTopics.COORDINATION)
    
    def _determine_cache_namespace(self, agent_id: str) -> str:
        """Determine cache namespace based on agent ID"""
        # Extract agent category from ID
        if "consciousness" in agent_id.lower():
            return CacheNamespace.CONSCIOUSNESS.value
        elif "memory" in agent_id.lower():
            return CacheNamespace.MEMORY.value
        elif "simulation" in agent_id.lower():
            return CacheNamespace.SIMULATION.value
        elif "alignment" in agent_id.lower():
            return CacheNamespace.ALIGNMENT.value
        elif "goal" in agent_id.lower():
            return CacheNamespace.GOALS.value
        elif "coordination" in agent_id.lower():
            return CacheNamespace.COORDINATION.value
        else:
            return CacheNamespace.SYSTEM.value
    
    # =============================================================================
    # MESSAGE HANDLERS
    # =============================================================================
    
    def _handle_system_health_message(self, message: NISMessage):
        """Handle system health messages"""
        try:
            content = message.content
            self.logger.info(f"System health update from {message.source_agent}: {content}")
            
            # Update service status if relevant
            if message.source_agent in self.service_status:
                status = self.service_status[message.source_agent]
                status.last_check = time.time()
                
                if content.get("health") == "healthy":
                    status.health = ServiceHealth.HEALTHY
                elif content.get("health") == "degraded":
                    status.health = ServiceHealth.DEGRADED
                else:
                    status.health = ServiceHealth.UNHEALTHY
                
                status.error_count = content.get("error_count", status.error_count)
                status.response_time = content.get("response_time", status.response_time)
                
        except Exception as e:
            self.logger.error(f"Error handling system health message: {e}")
    
    def _handle_audit_alert_message(self, message: NISMessage):
        """Handle audit alert messages"""
        try:
            content = message.content
            self.logger.warning(f"Audit alert from {message.source_agent}: {content}")
            
            # Track infrastructure violations
            if self.self_audit_enabled:
                violation = {
                    "timestamp": time.time(),
                    "source": message.source_agent,
                    "alert": content,
                    "severity": content.get("severity", "medium")
                }
                self.infrastructure_violations.append(violation)
                
                # Keep only recent violations
                cutoff_time = time.time() - 3600  # 1 hour
                self.infrastructure_violations = [
                    v for v in self.infrastructure_violations 
                    if v["timestamp"] > cutoff_time
                ]
                
        except Exception as e:
            self.logger.error(f"Error handling audit alert message: {e}")
    
    def _handle_performance_message(self, message: NISMessage):
        """Handle performance metric messages"""
        try:
            content = message.content
            self.logger.debug(f"Performance metrics from {message.source_agent}: {content}")
            
            # Update relevant metrics
            response_time = content.get("response_time", 0.0)
            if response_time > 0:
                self.integration_metrics.avg_response_time = (
                    self.integration_metrics.avg_response_time * 0.9 + response_time * 0.1
                )
                
        except Exception as e:
            self.logger.error(f"Error handling performance message: {e}")
    
    # =============================================================================
    # STATUS AND METRICS
    # =============================================================================
    
    async def shutdown(self):
        """Gracefully shutdown all infrastructure components"""
        try:
            self.logger.info("Shutting down infrastructure coordinator...")
            
            # Stop health monitoring
            await self.stop_health_monitoring()
            
            # Shutdown Kafka
            if self.kafka_manager:
                await self.kafka_manager.shutdown()
            
            # Shutdown Redis
            if self.redis_manager:
                await self.redis_manager.shutdown()
            
            self.logger.info("Infrastructure coordinator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive infrastructure status"""
        return {
            "integration_status": self.integration_metrics.integration_status.value,
            "overall_health": self.integration_metrics.overall_health.value,
            "services": {name: asdict(status) for name, status in self.service_status.items()},
            "metrics": asdict(self.integration_metrics),
            "agent_subscriptions": dict(self.agent_subscriptions),
            "uptime": time.time() - self.start_time,
            "auto_recovery_enabled": self.auto_recovery,
            "self_audit_enabled": self.self_audit_enabled
        }
    
    def get_metrics(self) -> InfrastructureMetrics:
        """Get infrastructure metrics"""
        return self.integration_metrics 
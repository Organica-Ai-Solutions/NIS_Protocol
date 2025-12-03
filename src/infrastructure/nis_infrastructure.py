#!/usr/bin/env python3
"""
NIS Protocol - Unified Infrastructure Integration
Connects Kafka, Redis, and Zookeeper throughout the entire NIS Protocol

This module provides:
- Centralized infrastructure management
- Event streaming for all NIS components
- Caching for performance optimization
- Real-time telemetry and monitoring
- Message routing for robotics, AI, and automotive systems
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("nis.infrastructure")

# Import infrastructure components
from .message_broker import (
    KafkaMessageBroker, 
    RedisCache, 
    InfrastructureManager,
    MessageType,
    KAFKA_AVAILABLE,
    REDIS_AVAILABLE
)


class NISEventType(Enum):
    """NIS Protocol Event Types for Kafka streaming"""
    # System Events
    SYSTEM_STARTUP = "nis.system.startup"
    SYSTEM_SHUTDOWN = "nis.system.shutdown"
    SYSTEM_HEALTH = "nis.system.health"
    SYSTEM_ERROR = "nis.system.error"
    
    # Chat/AI Events
    CHAT_REQUEST = "nis.chat.request"
    CHAT_RESPONSE = "nis.chat.response"
    AI_INFERENCE = "nis.ai.inference"
    
    # Robotics Events
    ROBOT_COMMAND = "nis.robotics.command"
    ROBOT_TELEMETRY = "nis.robotics.telemetry"
    ROBOT_STATUS = "nis.robotics.status"
    KINEMATICS_COMPUTED = "nis.robotics.kinematics"
    TRAJECTORY_PLANNED = "nis.robotics.trajectory"
    
    # CAN Bus Events
    CAN_MESSAGE = "nis.can.message"
    CAN_ERROR = "nis.can.error"
    CAN_EMERGENCY = "nis.can.emergency"
    
    # OBD-II Events
    OBD_DATA = "nis.obd.data"
    OBD_DIAGNOSTIC = "nis.obd.diagnostic"
    OBD_ALERT = "nis.obd.alert"
    
    # Physics Events
    PHYSICS_VALIDATION = "nis.physics.validation"
    PHYSICS_SIMULATION = "nis.physics.simulation"
    
    # Consciousness Events
    CONSCIOUSNESS_UPDATE = "nis.consciousness.update"
    AGENT_EVOLUTION = "nis.consciousness.evolution"
    
    # Training Events
    TRAINING_STARTED = "nis.training.started"
    TRAINING_COMPLETED = "nis.training.completed"
    MODEL_UPDATED = "nis.training.model_updated"


class CacheNamespace(Enum):
    """Redis cache namespaces"""
    SESSION = "session"
    CONVERSATION = "conversation"
    ROBOT_STATE = "robot_state"
    VEHICLE_STATE = "vehicle_state"
    PHYSICS_CACHE = "physics_cache"
    MODEL_CACHE = "model_cache"
    TELEMETRY = "telemetry"
    METRICS = "metrics"


@dataclass
class NISInfrastructureConfig:
    """Configuration for NIS Infrastructure"""
    kafka_servers: str = "kafka:9092"
    redis_host: str = "redis"
    redis_port: int = 6379
    enable_streaming: bool = True
    enable_caching: bool = True
    cache_ttl_default: int = 3600  # 1 hour
    telemetry_interval: float = 1.0  # seconds
    health_check_interval: float = 30.0  # seconds


class NISInfrastructure:
    """
    Unified Infrastructure Manager for NIS Protocol
    
    Provides centralized access to:
    - Kafka message streaming
    - Redis caching
    - Event publishing/subscribing
    - Telemetry collection
    - Health monitoring
    """
    
    _instance = None
    
    def __new__(cls, config: Optional[NISInfrastructureConfig] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[NISInfrastructureConfig] = None):
        if self._initialized:
            return
        
        self.config = config or NISInfrastructureConfig(
            kafka_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"),
            redis_host=os.getenv("REDIS_HOST", "redis"),
            redis_port=int(os.getenv("REDIS_PORT", "6379"))
        )
        
        self.kafka = KafkaMessageBroker()
        self.redis = RedisCache()
        
        self.is_connected = False
        self.start_time = None
        
        # Event handlers
        self.event_handlers: Dict[NISEventType, List[Callable]] = {}
        
        # Telemetry data
        self.telemetry = {
            'events_published': 0,
            'events_received': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }
        
        # Background tasks
        self._health_task = None
        self._telemetry_task = None
        
        self._initialized = True
        logger.info("NIS Infrastructure initialized")
    
    async def connect(self) -> Dict[str, bool]:
        """Connect to all infrastructure services"""
        results = {}
        
        # Connect Kafka
        try:
            results['kafka'] = await self.kafka.connect()
            if results['kafka']:
                logger.info("✅ Kafka connected")
        except Exception as e:
            logger.error(f"❌ Kafka connection failed: {e}")
            results['kafka'] = False
        
        # Connect Redis
        try:
            results['redis'] = await self.redis.connect()
            if results['redis']:
                logger.info("✅ Redis connected")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            results['redis'] = False
        
        self.is_connected = results.get('kafka', False) or results.get('redis', False)
        self.start_time = time.time()
        
        # Start background tasks
        if self.is_connected:
            self._health_task = asyncio.create_task(self._health_check_loop())
        
        # Publish startup event
        await self.publish_event(NISEventType.SYSTEM_STARTUP, {
            'version': '4.0.1',
            'kafka_connected': results.get('kafka', False),
            'redis_connected': results.get('redis', False),
            'timestamp': time.time()
        })
        
        return results
    
    async def disconnect(self):
        """Disconnect from all services"""
        # Cancel background tasks
        if self._health_task:
            self._health_task.cancel()
        if self._telemetry_task:
            self._telemetry_task.cancel()
        
        # Publish shutdown event
        await self.publish_event(NISEventType.SYSTEM_SHUTDOWN, {
            'uptime': time.time() - self.start_time if self.start_time else 0,
            'events_published': self.telemetry['events_published'],
            'timestamp': time.time()
        })
        
        await self.kafka.disconnect()
        await self.redis.disconnect()
        
        self.is_connected = False
        logger.info("NIS Infrastructure disconnected")
    
    async def publish_event(
        self,
        event_type: NISEventType,
        data: Dict[str, Any],
        key: Optional[str] = None
    ) -> bool:
        """Publish event to Kafka"""
        try:
            enriched_data = {
                **data,
                '_event_type': event_type.value,
                '_timestamp': time.time(),
                '_source': 'nis-protocol'
            }
            
            topic = event_type.value.replace('.', '_')
            success = await self.kafka.publish(topic, enriched_data, key)
            
            if success:
                self.telemetry['events_published'] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Event publish error: {e}")
            self.telemetry['errors'] += 1
            return False
    
    async def subscribe_event(
        self,
        event_type: NISEventType,
        handler: Callable[[Dict[str, Any]], None]
    ):
        """Subscribe to event type"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        
        topic = event_type.value.replace('.', '_')
        await self.kafka.subscribe([topic], self._handle_event)
    
    async def _handle_event(self, data: Dict[str, Any]):
        """Handle incoming event"""
        try:
            event_type_str = data.get('_event_type', '')
            
            for event_type, handlers in self.event_handlers.items():
                if event_type.value == event_type_str:
                    for handler in handlers:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(data)
                        else:
                            handler(data)
            
            self.telemetry['events_received'] += 1
            
        except Exception as e:
            logger.error(f"Event handling error: {e}")
            self.telemetry['errors'] += 1
    
    async def cache_set(
        self,
        namespace: CacheNamespace,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache"""
        full_key = f"{namespace.value}:{key}"
        return await self.redis.set(full_key, value, ttl or self.config.cache_ttl_default)
    
    async def cache_get(
        self,
        namespace: CacheNamespace,
        key: str
    ) -> Optional[Any]:
        """Get value from cache"""
        full_key = f"{namespace.value}:{key}"
        value = await self.redis.get(full_key)
        
        if value is not None:
            self.telemetry['cache_hits'] += 1
        else:
            self.telemetry['cache_misses'] += 1
        
        return value
    
    async def cache_delete(
        self,
        namespace: CacheNamespace,
        key: str
    ) -> bool:
        """Delete value from cache"""
        full_key = f"{namespace.value}:{key}"
        return await self.redis.delete(full_key)
    
    async def publish_telemetry(
        self,
        component: str,
        metrics: Dict[str, Any]
    ):
        """Publish component telemetry"""
        await self.publish_event(NISEventType.SYSTEM_HEALTH, {
            'component': component,
            'metrics': metrics,
            'timestamp': time.time()
        })
        
        # Also cache for quick access
        await self.cache_set(
            CacheNamespace.TELEMETRY,
            component,
            metrics,
            ttl=60  # 1 minute TTL for telemetry
        )
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                health = await self.get_health_status()
                await self.publish_event(NISEventType.SYSTEM_HEALTH, health)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        kafka_stats = self.kafka.get_stats()
        redis_stats = self.redis.get_stats()
        
        return {
            'status': 'healthy' if self.is_connected else 'degraded',
            'uptime_seconds': uptime,
            'infrastructure': {
                'kafka': {
                    'connected': kafka_stats.get('is_connected', False),
                    'available': KAFKA_AVAILABLE,
                    'messages_sent': kafka_stats.get('messages_sent', 0),
                    'messages_received': kafka_stats.get('messages_received', 0),
                    'errors': kafka_stats.get('errors', 0)
                },
                'redis': {
                    'connected': redis_stats.get('is_connected', False),
                    'available': REDIS_AVAILABLE,
                    'hit_rate': redis_stats.get('hit_rate', 0),
                    'operations': redis_stats.get('gets', 0) + redis_stats.get('sets', 0)
                },
                'zookeeper': {
                    'available': kafka_stats.get('is_connected', False),
                    'note': 'Inferred from Kafka connection'
                }
            },
            'telemetry': self.telemetry,
            'timestamp': time.time()
        }
    
    # ========================================================================
    # ROBOTICS INTEGRATION
    # ========================================================================
    
    async def publish_robot_command(
        self,
        robot_id: str,
        command: str,
        parameters: Dict[str, Any]
    ):
        """Publish robot command event"""
        await self.publish_event(NISEventType.ROBOT_COMMAND, {
            'robot_id': robot_id,
            'command': command,
            'parameters': parameters
        }, key=robot_id)
    
    async def publish_robot_telemetry(
        self,
        robot_id: str,
        telemetry: Dict[str, Any]
    ):
        """Publish robot telemetry"""
        await self.publish_event(NISEventType.ROBOT_TELEMETRY, {
            'robot_id': robot_id,
            **telemetry
        }, key=robot_id)
        
        # Cache latest state
        await self.cache_set(
            CacheNamespace.ROBOT_STATE,
            robot_id,
            telemetry,
            ttl=10  # 10 second TTL
        )
    
    async def get_robot_state(self, robot_id: str) -> Optional[Dict[str, Any]]:
        """Get cached robot state"""
        return await self.cache_get(CacheNamespace.ROBOT_STATE, robot_id)
    
    # ========================================================================
    # OBD-II / AUTOMOTIVE INTEGRATION
    # ========================================================================
    
    async def publish_vehicle_data(
        self,
        vehicle_id: str,
        data: Dict[str, Any]
    ):
        """Publish vehicle OBD-II data"""
        await self.publish_event(NISEventType.OBD_DATA, {
            'vehicle_id': vehicle_id,
            **data
        }, key=vehicle_id)
        
        # Cache latest state
        await self.cache_set(
            CacheNamespace.VEHICLE_STATE,
            vehicle_id,
            data,
            ttl=5  # 5 second TTL for vehicle data
        )
    
    async def publish_vehicle_alert(
        self,
        vehicle_id: str,
        alert_type: str,
        message: str,
        details: Dict[str, Any]
    ):
        """Publish vehicle safety alert"""
        await self.publish_event(NISEventType.OBD_ALERT, {
            'vehicle_id': vehicle_id,
            'alert_type': alert_type,
            'message': message,
            'details': details,
            'severity': 'high' if 'critical' in alert_type.lower() else 'medium'
        }, key=vehicle_id)
    
    async def get_vehicle_state(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Get cached vehicle state"""
        return await self.cache_get(CacheNamespace.VEHICLE_STATE, vehicle_id)
    
    # ========================================================================
    # AI/CHAT INTEGRATION
    # ========================================================================
    
    async def publish_chat_event(
        self,
        conversation_id: str,
        event_type: str,
        data: Dict[str, Any]
    ):
        """Publish chat event"""
        event = NISEventType.CHAT_REQUEST if event_type == 'request' else NISEventType.CHAT_RESPONSE
        await self.publish_event(event, {
            'conversation_id': conversation_id,
            **data
        }, key=conversation_id)
    
    async def cache_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]]
    ):
        """Cache conversation history"""
        await self.cache_set(
            CacheNamespace.CONVERSATION,
            conversation_id,
            messages,
            ttl=3600  # 1 hour
        )
    
    async def get_conversation(self, conversation_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached conversation"""
        return await self.cache_get(CacheNamespace.CONVERSATION, conversation_id)
    
    # ========================================================================
    # PHYSICS INTEGRATION
    # ========================================================================
    
    async def publish_physics_validation(
        self,
        validation_id: str,
        result: Dict[str, Any]
    ):
        """Publish physics validation result"""
        await self.publish_event(NISEventType.PHYSICS_VALIDATION, {
            'validation_id': validation_id,
            **result
        })
        
        # Cache result
        await self.cache_set(
            CacheNamespace.PHYSICS_CACHE,
            validation_id,
            result,
            ttl=300  # 5 minutes
        )
    
    async def get_cached_physics(self, validation_id: str) -> Optional[Dict[str, Any]]:
        """Get cached physics validation"""
        return await self.cache_get(CacheNamespace.PHYSICS_CACHE, validation_id)


# Singleton accessor
_infrastructure: Optional[NISInfrastructure] = None


def get_nis_infrastructure() -> NISInfrastructure:
    """Get the NIS Infrastructure singleton"""
    global _infrastructure
    if _infrastructure is None:
        _infrastructure = NISInfrastructure()
    return _infrastructure


async def initialize_infrastructure() -> Dict[str, bool]:
    """Initialize and connect infrastructure"""
    infra = get_nis_infrastructure()
    return await infra.connect()


async def shutdown_infrastructure():
    """Shutdown infrastructure"""
    global _infrastructure
    if _infrastructure:
        await _infrastructure.disconnect()
        _infrastructure = None

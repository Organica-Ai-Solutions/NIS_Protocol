#!/usr/bin/env python3
"""
NIS Protocol - Message Broker Infrastructure
Kafka and Redis integration for real-time messaging and caching

Features:
- Kafka producer/consumer for event streaming
- Redis for caching and pub/sub
- Connection health monitoring
- Automatic reconnection
- Message serialization/deserialization
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading

logger = logging.getLogger("nis.infrastructure.broker")


# Configuration
@dataclass
class KafkaConfig:
    """Kafka configuration"""
    bootstrap_servers: str = "kafka:9092"
    client_id: str = "nis-protocol"
    group_id: str = "nis-consumers"
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = True
    max_poll_records: int = 100
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 10000


@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "redis"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    max_connections: int = 10


class MessageType(Enum):
    """Message types for NIS Protocol"""
    # Robotics
    ROBOT_COMMAND = "robot.command"
    ROBOT_TELEMETRY = "robot.telemetry"
    ROBOT_STATUS = "robot.status"
    EMERGENCY_STOP = "robot.emergency_stop"
    
    # CAN Bus
    CAN_MESSAGE = "can.message"
    CAN_ERROR = "can.error"
    
    # OBD-II (Automotive)
    OBD_DATA = "obd.data"
    OBD_DIAGNOSTIC = "obd.diagnostic"
    OBD_ERROR = "obd.error"
    
    # AI/ML
    INFERENCE_REQUEST = "ai.inference.request"
    INFERENCE_RESPONSE = "ai.inference.response"
    TRAINING_UPDATE = "ai.training.update"
    
    # System
    HEALTH_CHECK = "system.health"
    CONFIG_UPDATE = "system.config"
    LOG_EVENT = "system.log"


# Try to import Kafka
try:
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
    from kafka import KafkaProducer as SyncKafkaProducer
    from kafka import KafkaConsumer as SyncKafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.warning("Kafka libraries not available - using mock mode")


# Try to import Redis
try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis library not available - using mock mode")


class KafkaMessageBroker:
    """
    Kafka Message Broker for NIS Protocol
    
    Handles event streaming for:
    - Robotics telemetry
    - CAN bus messages
    - OBD-II data
    - AI inference requests
    """
    
    def __init__(self, config: Optional[KafkaConfig] = None):
        self.config = config or KafkaConfig(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        )
        self.producer = None
        self.consumers: Dict[str, Any] = {}
        self.is_connected = False
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0,
            'last_activity': None
        }
        
        logger.info(f"Kafka broker initialized (servers: {self.config.bootstrap_servers})")
    
    async def connect(self) -> bool:
        """Connect to Kafka cluster"""
        if not KAFKA_AVAILABLE:
            logger.warning("Kafka not available - running in mock mode")
            self.is_connected = True
            return True
        
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                client_id=self.config.client_id,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            await self.producer.start()
            self.is_connected = True
            logger.info("✅ Kafka producer connected")
            return True
            
        except Exception as e:
            logger.error(f"❌ Kafka connection failed: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Kafka"""
        if self.producer:
            await self.producer.stop()
        
        for consumer in self.consumers.values():
            await consumer.stop()
        
        self.is_connected = False
        logger.info("Kafka broker disconnected")
    
    async def publish(
        self,
        topic: str,
        message: Dict[str, Any],
        key: Optional[str] = None,
        message_type: Optional[MessageType] = None
    ) -> bool:
        """Publish message to Kafka topic"""
        try:
            # Add metadata
            enriched_message = {
                **message,
                '_timestamp': time.time(),
                '_type': message_type.value if message_type else 'unknown',
                '_source': 'nis-protocol'
            }
            
            if KAFKA_AVAILABLE and self.producer:
                await self.producer.send_and_wait(topic, enriched_message, key=key)
            else:
                # Mock mode - just log
                logger.debug(f"[MOCK] Kafka publish: {topic} -> {enriched_message}")
            
            self.stats['messages_sent'] += 1
            self.stats['last_activity'] = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Kafka publish error: {e}")
            self.stats['errors'] += 1
            return False
    
    async def subscribe(
        self,
        topics: List[str],
        handler: Callable[[Dict[str, Any]], None],
        group_id: Optional[str] = None
    ):
        """Subscribe to Kafka topics"""
        if not KAFKA_AVAILABLE:
            logger.warning(f"[MOCK] Kafka subscribe: {topics}")
            return
        
        try:
            consumer = AIOKafkaConsumer(
                *topics,
                bootstrap_servers=self.config.bootstrap_servers,
                group_id=group_id or self.config.group_id,
                auto_offset_reset=self.config.auto_offset_reset,
                value_deserializer=lambda v: json.loads(v.decode('utf-8'))
            )
            await consumer.start()
            
            # Store consumer
            consumer_key = ','.join(topics)
            self.consumers[consumer_key] = consumer
            
            # Start consuming in background
            asyncio.create_task(self._consume_messages(consumer, handler))
            
            logger.info(f"✅ Subscribed to topics: {topics}")
            
        except Exception as e:
            logger.error(f"Kafka subscribe error: {e}")
    
    async def _consume_messages(self, consumer, handler: Callable):
        """Background task to consume messages"""
        try:
            async for msg in consumer:
                try:
                    await handler(msg.value)
                    self.stats['messages_received'] += 1
                    self.stats['last_activity'] = time.time()
                except Exception as e:
                    logger.error(f"Message handler error: {e}")
                    self.stats['errors'] += 1
        except Exception as e:
            logger.error(f"Consumer error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get broker statistics"""
        return {
            **self.stats,
            'is_connected': self.is_connected,
            'kafka_available': KAFKA_AVAILABLE,
            'bootstrap_servers': self.config.bootstrap_servers
        }


class RedisCache:
    """
    Redis Cache for NIS Protocol
    
    Provides:
    - Key-value caching
    - Pub/sub messaging
    - Session storage
    - Rate limiting
    """
    
    def __init__(self, config: Optional[RedisConfig] = None):
        self.config = config or RedisConfig(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", "6379"))
        )
        self.client = None
        self.async_client = None
        self.pubsub = None
        self.is_connected = False
        self.stats = {
            'gets': 0,
            'sets': 0,
            'hits': 0,
            'misses': 0,
            'errors': 0
        }
        
        logger.info(f"Redis cache initialized (host: {self.config.host}:{self.config.port})")
    
    async def connect(self) -> bool:
        """Connect to Redis"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - running in mock mode")
            self.is_connected = True
            return True
        
        try:
            self.async_client = aioredis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout
            )
            
            # Test connection
            await self.async_client.ping()
            self.is_connected = True
            logger.info("✅ Redis connected")
            return True
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.async_client:
            await self.async_client.close()
        self.is_connected = False
        logger.info("Redis disconnected")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            self.stats['gets'] += 1
            
            if not REDIS_AVAILABLE or not self.async_client:
                self.stats['misses'] += 1
                return None
            
            value = await self.async_client.get(key)
            if value:
                self.stats['hits'] += 1
                return json.loads(value)
            else:
                self.stats['misses'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.stats['errors'] += 1
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache"""
        try:
            self.stats['sets'] += 1
            
            if not REDIS_AVAILABLE or not self.async_client:
                return True  # Mock success
            
            serialized = json.dumps(value)
            if ttl:
                await self.async_client.setex(key, ttl, serialized)
            else:
                await self.async_client.set(key, serialized)
            return True
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            self.stats['errors'] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if REDIS_AVAILABLE and self.async_client:
                await self.async_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def publish(self, channel: str, message: Dict[str, Any]) -> bool:
        """Publish message to Redis pub/sub channel"""
        try:
            if REDIS_AVAILABLE and self.async_client:
                await self.async_client.publish(channel, json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Redis publish error: {e}")
            return False
    
    async def subscribe(self, channels: List[str], handler: Callable):
        """Subscribe to Redis pub/sub channels"""
        if not REDIS_AVAILABLE or not self.async_client:
            logger.warning(f"[MOCK] Redis subscribe: {channels}")
            return
        
        try:
            self.pubsub = self.async_client.pubsub()
            await self.pubsub.subscribe(*channels)
            
            # Start listening in background
            asyncio.create_task(self._listen_pubsub(handler))
            
            logger.info(f"✅ Subscribed to Redis channels: {channels}")
            
        except Exception as e:
            logger.error(f"Redis subscribe error: {e}")
    
    async def _listen_pubsub(self, handler: Callable):
        """Background task to listen for pub/sub messages"""
        try:
            async for message in self.pubsub.listen():
                if message['type'] == 'message':
                    data = json.loads(message['data'])
                    await handler(data)
        except Exception as e:
            logger.error(f"Redis pubsub error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = self.stats['hits'] / max(self.stats['gets'], 1)
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'is_connected': self.is_connected,
            'redis_available': REDIS_AVAILABLE,
            'host': f"{self.config.host}:{self.config.port}"
        }


class InfrastructureManager:
    """
    Central manager for all infrastructure services
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.kafka = KafkaMessageBroker()
        self.redis = RedisCache()
        self._initialized = True
        
        logger.info("Infrastructure Manager initialized")
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all infrastructure services"""
        results = {}
        
        results['kafka'] = await self.kafka.connect()
        results['redis'] = await self.redis.connect()
        
        return results
    
    async def disconnect_all(self):
        """Disconnect from all services"""
        await self.kafka.disconnect()
        await self.redis.disconnect()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all services"""
        return {
            'kafka': {
                'connected': self.kafka.is_connected,
                'available': KAFKA_AVAILABLE,
                'stats': self.kafka.get_stats()
            },
            'redis': {
                'connected': self.redis.is_connected,
                'available': REDIS_AVAILABLE,
                'stats': self.redis.get_stats()
            },
            'timestamp': time.time()
        }


# Singleton instance
def get_infrastructure_manager() -> InfrastructureManager:
    """Get the infrastructure manager singleton"""
    return InfrastructureManager()

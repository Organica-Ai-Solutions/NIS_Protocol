#!/usr/bin/env python3
"""
Integration Tests for Infrastructure (Kafka, Redis, Zookeeper)
Tests message streaming, caching, and event handling
"""

import pytest
import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.infrastructure.message_broker import (
    KafkaMessageBroker, RedisCache, 
    KAFKA_AVAILABLE, REDIS_AVAILABLE
)
from src.infrastructure.nis_infrastructure import (
    NISInfrastructure, NISEventType, CacheNamespace,
    get_nis_infrastructure
)


class TestKafkaIntegration:
    """Test Kafka Message Broker Integration"""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup Kafka broker for each test"""
        self.kafka = KafkaMessageBroker()
        yield
        if self.kafka.is_connected:
            await self.kafka.disconnect()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kafka_connection(self):
        """Should connect to Kafka"""
        if not KAFKA_AVAILABLE:
            pytest.skip("Kafka not available")
        
        connected = await self.kafka.connect()
        assert connected is True
        assert self.kafka.is_connected is True
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kafka_publish(self):
        """Should publish message to Kafka"""
        if not KAFKA_AVAILABLE:
            pytest.skip("Kafka not available")
        
        await self.kafka.connect()
        
        success = await self.kafka.publish(
            topic="test_topic",
            message={"test": "data", "timestamp": time.time()},
            key="test_key"
        )
        
        assert success is True
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kafka_statistics(self):
        """Should track Kafka statistics"""
        await self.kafka.connect()
        
        # Publish some messages
        for i in range(5):
            await self.kafka.publish(
                topic="test_stats",
                message={"index": i}
            )
        
        stats = self.kafka.get_stats()
        
        assert 'messages_sent' in stats
        assert stats['messages_sent'] >= 5
        assert 'is_connected' in stats


class TestRedisIntegration:
    """Test Redis Cache Integration"""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup Redis cache for each test"""
        self.redis = RedisCache()
        yield
        if self.redis.is_connected:
            await self.redis.disconnect()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_redis_connection(self):
        """Should connect to Redis"""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis not available")
        
        connected = await self.redis.connect()
        assert connected is True
        assert self.redis.is_connected is True
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_redis_set_get(self):
        """Should set and get values"""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis not available")
        
        await self.redis.connect()
        
        # Set value
        success = await self.redis.set("test_key", {"data": "value"}, ttl=60)
        assert success is True
        
        # Get value
        value = await self.redis.get("test_key")
        assert value is not None
        assert value["data"] == "value"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_redis_ttl(self):
        """Should respect TTL"""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis not available")
        
        await self.redis.connect()
        
        # Set with short TTL
        await self.redis.set("ttl_test", "value", ttl=1)
        
        # Should exist immediately
        value = await self.redis.get("ttl_test")
        assert value is not None
        
        # Wait for expiry
        await asyncio.sleep(1.5)
        
        # Should be gone
        value = await self.redis.get("ttl_test")
        assert value is None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_redis_statistics(self):
        """Should track Redis statistics"""
        await self.redis.connect()
        
        # Do some operations
        await self.redis.set("stats_test", "value")
        await self.redis.get("stats_test")
        await self.redis.get("nonexistent")
        
        stats = self.redis.get_stats()
        
        assert 'gets' in stats
        assert 'sets' in stats
        assert 'hits' in stats
        assert 'misses' in stats


class TestNISInfrastructure:
    """Test Unified NIS Infrastructure"""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup NIS infrastructure for each test"""
        self.infra = NISInfrastructure()
        yield
        if self.infra.is_connected:
            await self.infra.disconnect()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_infrastructure_connection(self):
        """Should connect to all infrastructure services"""
        results = await self.infra.connect()
        
        assert 'kafka' in results
        assert 'redis' in results
        assert self.infra.is_connected is True
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_publish_event(self):
        """Should publish events to Kafka"""
        await self.infra.connect()
        
        success = await self.infra.publish_event(
            NISEventType.SYSTEM_HEALTH,
            {"status": "healthy", "test": True}
        )
        
        assert success is True
        assert self.infra.telemetry['events_published'] > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cache_operations(self):
        """Should cache and retrieve data"""
        await self.infra.connect()
        
        # Cache robot state
        success = await self.infra.cache_set(
            CacheNamespace.ROBOT_STATE,
            "test_robot",
            {"position": [0, 0, 0], "velocity": [0, 0, 0]}
        )
        assert success is True
        
        # Retrieve robot state
        state = await self.infra.cache_get(
            CacheNamespace.ROBOT_STATE,
            "test_robot"
        )
        assert state is not None
        assert state["position"] == [0, 0, 0]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health_status(self):
        """Should report health status"""
        await self.infra.connect()
        
        health = await self.infra.get_health_status()
        
        assert 'status' in health
        assert 'infrastructure' in health
        assert 'telemetry' in health
        assert health['status'] in ['healthy', 'degraded']
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_robot_telemetry(self):
        """Should publish and cache robot telemetry"""
        await self.infra.connect()
        
        # Publish telemetry
        await self.infra.publish_robot_telemetry(
            "test_robot",
            {
                "position": [1.0, 2.0, 3.0],
                "velocity": [0.1, 0.2, 0.3],
                "timestamp": time.time()
            }
        )
        
        # Retrieve cached state
        state = await self.infra.get_robot_state("test_robot")
        assert state is not None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_vehicle_data(self):
        """Should publish and cache vehicle data"""
        await self.infra.connect()
        
        # Publish vehicle data
        await self.infra.publish_vehicle_data(
            "test_vehicle",
            {
                "engine_rpm": 2500,
                "vehicle_speed": 60,
                "coolant_temp": 90
            }
        )
        
        # Retrieve cached state
        state = await self.infra.get_vehicle_state("test_vehicle")
        assert state is not None


class TestEventStreaming:
    """Test Event Streaming Patterns"""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup infrastructure for each test"""
        self.infra = get_nis_infrastructure()
        await self.infra.connect()
        yield
        await self.infra.disconnect()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_robotics_events(self):
        """Should stream robotics events"""
        # Publish robot command
        await self.infra.publish_event(
            NISEventType.ROBOT_COMMAND,
            {
                "robot_id": "arm_1",
                "command": "move",
                "parameters": {"position": [0.5, 0.3, 0.8]}
            }
        )
        
        # Publish robot telemetry
        await self.infra.publish_event(
            NISEventType.ROBOT_TELEMETRY,
            {
                "robot_id": "arm_1",
                "position": [0.5, 0.3, 0.8],
                "velocity": [0, 0, 0]
            }
        )
        
        assert self.infra.telemetry['events_published'] >= 2
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_obd_events(self):
        """Should stream OBD-II events"""
        # Publish OBD data
        await self.infra.publish_event(
            NISEventType.OBD_DATA,
            {
                "vehicle_id": "car_1",
                "engine_rpm": 2500,
                "vehicle_speed": 60
            }
        )
        
        # Publish OBD alert
        await self.infra.publish_event(
            NISEventType.OBD_ALERT,
            {
                "vehicle_id": "car_1",
                "alert_type": "high_temp",
                "message": "Coolant temperature high"
            }
        )
        
        assert self.infra.telemetry['events_published'] >= 2
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_physics_events(self):
        """Should stream physics validation events"""
        await self.infra.publish_event(
            NISEventType.PHYSICS_VALIDATION,
            {
                "validation_id": "val_001",
                "is_valid": True,
                "confidence": 0.95
            }
        )
        
        assert self.infra.telemetry['events_published'] >= 1


class TestCacheNamespaces:
    """Test Cache Namespace Isolation"""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup infrastructure for each test"""
        self.infra = get_nis_infrastructure()
        await self.infra.connect()
        yield
        await self.infra.disconnect()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_namespace_isolation(self):
        """Different namespaces should be isolated"""
        # Set same key in different namespaces
        await self.infra.cache_set(
            CacheNamespace.ROBOT_STATE, "key1", {"type": "robot"}
        )
        await self.infra.cache_set(
            CacheNamespace.VEHICLE_STATE, "key1", {"type": "vehicle"}
        )
        
        # Retrieve from each namespace
        robot = await self.infra.cache_get(CacheNamespace.ROBOT_STATE, "key1")
        vehicle = await self.infra.cache_get(CacheNamespace.VEHICLE_STATE, "key1")
        
        assert robot["type"] == "robot"
        assert vehicle["type"] == "vehicle"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_all_namespaces(self):
        """All namespaces should work"""
        namespaces = [
            CacheNamespace.SESSION,
            CacheNamespace.CONVERSATION,
            CacheNamespace.ROBOT_STATE,
            CacheNamespace.VEHICLE_STATE,
            CacheNamespace.PHYSICS_CACHE,
            CacheNamespace.TELEMETRY
        ]
        
        for ns in namespaces:
            await self.infra.cache_set(ns, "test", {"namespace": ns.value})
            value = await self.infra.cache_get(ns, "test")
            assert value is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])

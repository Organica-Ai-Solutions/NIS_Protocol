#!/usr/bin/env python3
"""
NIS Protocol Test Configuration
Pytest fixtures and configuration for comprehensive testing
"""

import os
import sys
import pytest
import asyncio
from typing import Generator, AsyncGenerator

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set test environment
os.environ.setdefault("NIS_ENV", "test")
os.environ.setdefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")


# ============================================================================
# ASYNC FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# ROBOTICS FIXTURES
# ============================================================================

@pytest.fixture
def robotics_agent():
    """Create a robotics agent for testing"""
    from src.agents.robotics.unified_robotics_agent import UnifiedRoboticsAgent
    agent = UnifiedRoboticsAgent(
        agent_id="test_robotics_agent",
        enable_physics_validation=True,
        enable_redundancy=False  # Disable for faster tests
    )
    return agent


@pytest.fixture
def manipulator_config():
    """Standard 6-DOF manipulator configuration"""
    import numpy as np
    return {
        "robot_id": "test_manipulator",
        "joint_angles": np.array([0.0, 0.5, 1.0, 0.0, 0.5, 0.0]),
        "target_pose": {
            "position": np.array([0.5, 0.3, 0.8]),
            "orientation": np.array([0, 0, 0, 1])
        }
    }


@pytest.fixture
def drone_config():
    """Standard drone configuration"""
    import numpy as np
    return {
        "robot_id": "test_drone",
        "waypoints": [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 0.0, 2.0])
        ],
        "duration": 5.0
    }


# ============================================================================
# PHYSICS FIXTURES
# ============================================================================

@pytest.fixture
def physics_agent():
    """Create a physics agent for testing"""
    from src.agents.physics.unified_physics_agent import UnifiedPhysicsAgent
    agent = UnifiedPhysicsAgent(agent_id="test_physics_agent")
    return agent


@pytest.fixture
def physics_test_data():
    """Standard physics test data"""
    return {
        "mechanics": {
            "velocity": [1.0, 2.0, 3.0],
            "mass": 10.0,
            "force": [5.0, 0.0, 0.0]
        },
        "heat_equation": {
            "thermal_diffusivity": 0.01,
            "domain_length": 1.0,
            "final_time": 0.5
        }
    }


# ============================================================================
# INFRASTRUCTURE FIXTURES
# ============================================================================

@pytest.fixture
async def kafka_broker():
    """Create Kafka broker for testing"""
    from src.infrastructure.message_broker import KafkaMessageBroker
    broker = KafkaMessageBroker()
    connected = await broker.connect()
    yield broker
    await broker.disconnect()


@pytest.fixture
async def redis_cache():
    """Create Redis cache for testing"""
    from src.infrastructure.message_broker import RedisCache
    cache = RedisCache()
    connected = await cache.connect()
    yield cache
    await cache.disconnect()


@pytest.fixture
async def nis_infrastructure():
    """Create NIS infrastructure for testing"""
    from src.infrastructure.nis_infrastructure import get_nis_infrastructure
    infra = get_nis_infrastructure()
    await infra.connect()
    yield infra
    await infra.disconnect()


# ============================================================================
# API FIXTURES
# ============================================================================

@pytest.fixture
def test_client():
    """Create FastAPI test client"""
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create async HTTP client for API testing"""
    import httpx
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        yield client


# ============================================================================
# CAN/OBD FIXTURES
# ============================================================================

@pytest.fixture
def can_protocol():
    """Create CAN protocol for testing"""
    from src.protocols.can_protocol import CANProtocol
    protocol = CANProtocol(force_simulation=True)
    return protocol


@pytest.fixture
def obd_protocol():
    """Create OBD protocol for testing"""
    from src.protocols.obd_protocol import OBDProtocol
    protocol = OBDProtocol(simulation_mode=True)
    return protocol


# ============================================================================
# MOCK DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_chat_request():
    """Sample chat request for testing"""
    return {
        "message": "Test message",
        "user_id": "test_user",
        "conversation_id": None
    }


@pytest.fixture
def sample_robot_command():
    """Sample robot command for testing"""
    return {
        "robot_id": "test_robot",
        "command": "move",
        "parameters": {
            "position": [0.5, 0.3, 0.8],
            "velocity": 0.5
        }
    }


@pytest.fixture
def sample_vehicle_data():
    """Sample OBD-II vehicle data for testing"""
    return {
        "vehicle_id": "test_vehicle",
        "engine_rpm": 2500.0,
        "vehicle_speed": 60.0,
        "coolant_temp": 90.0,
        "throttle_position": 25.0,
        "fuel_level": 75.0,
        "battery_voltage": 14.2
    }


# ============================================================================
# MARKERS
# ============================================================================

def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no external deps)")
    config.addinivalue_line("markers", "integration: Integration tests (requires infrastructure)")
    config.addinivalue_line("markers", "e2e: End-to-end tests (full system)")
    config.addinivalue_line("markers", "slow: Slow tests (>5 seconds)")
    config.addinivalue_line("markers", "hardware: Tests requiring physical hardware")

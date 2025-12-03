#!/usr/bin/env python3
"""
Integration Tests for API Endpoints
Tests all major API endpoints with real requests
"""

import pytest
import httpx
import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


# Base URL for API tests
BASE_URL = os.getenv("NIS_API_URL", "http://localhost:8000")


class TestHealthEndpoints:
    """Test Health and Status Endpoints"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health(self):
        """Health endpoint should return healthy"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_root(self):
        """Root endpoint should return system info"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/")
        
        assert response.status_code == 200
        data = response.json()
        assert "system" in data
        assert "version" in data
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_metrics(self):
        """Metrics endpoint should return Prometheus format"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/metrics")
        
        assert response.status_code == 200
        assert "nis_health_status" in response.text


class TestInfrastructureEndpoints:
    """Test Infrastructure Status Endpoints"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_infrastructure_status(self):
        """Infrastructure status should return all services"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/infrastructure/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "infrastructure" in data
        assert "kafka" in data["infrastructure"]
        assert "redis" in data["infrastructure"]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kafka_status(self):
        """Kafka status endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/infrastructure/kafka")
        
        assert response.status_code == 200
        data = response.json()
        assert "kafka" in data
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_redis_status(self):
        """Redis status endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/infrastructure/redis")
        
        assert response.status_code == 200
        data = response.json()
        assert "redis" in data
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_runner_status(self):
        """Runner status endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/runner/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "runner" in data


class TestRoboticsEndpoints:
    """Test Robotics API Endpoints"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_robotics_capabilities(self):
        """Robotics capabilities endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/robotics/capabilities")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "capabilities" in data
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_forward_kinematics(self):
        """Forward kinematics endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/robotics/forward_kinematics",
                json={
                    "robot_id": "test_arm",
                    "joint_angles": [0.0, 0.5, 1.0, 0.0, 0.5, 0.0],
                    "robot_type": "manipulator"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "result" in data
        assert data["result"]["success"] is True
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_inverse_kinematics(self):
        """Inverse kinematics endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/robotics/inverse_kinematics",
                json={
                    "robot_id": "test_arm",
                    "target_pose": {"position": [0.5, 0.3, 0.8]},
                    "robot_type": "manipulator"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_trajectory_planning(self):
        """Trajectory planning endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/robotics/plan_trajectory",
                json={
                    "robot_id": "test_drone",
                    "waypoints": [[0, 0, 0], [1, 1, 1], [2, 0, 2]],
                    "robot_type": "drone",
                    "duration": 5.0
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


class TestCANEndpoints:
    """Test CAN Protocol Endpoints"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_can_status(self):
        """CAN status endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/robotics/can/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "can_protocol" in data
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_can_safety(self):
        """CAN safety endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/robotics/can/safety")
        
        assert response.status_code == 200
        data = response.json()
        assert "safety_limits" in data


class TestOBDEndpoints:
    """Test OBD-II Endpoints"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_obd_status(self):
        """OBD status endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/robotics/obd/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "obd_protocol" in data
        assert "vehicle_state" in data
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_obd_vehicle(self):
        """OBD vehicle data endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/robotics/obd/vehicle")
        
        assert response.status_code == 200
        data = response.json()
        assert "vehicle" in data
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_obd_dtcs(self):
        """OBD DTCs endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/robotics/obd/dtcs")
        
        assert response.status_code == 200
        data = response.json()
        assert "dtcs" in data
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_obd_safety(self):
        """OBD safety thresholds endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/robotics/obd/safety")
        
        assert response.status_code == 200
        data = response.json()
        assert "safety_thresholds" in data


class TestPhysicsEndpoints:
    """Test Physics API Endpoints"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_physics_capabilities(self):
        """Physics capabilities endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/physics/capabilities")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"
        assert "domains" in data
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_physics_constants(self):
        """Physics constants endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/physics/constants")
        
        assert response.status_code == 200
        data = response.json()
        assert "fundamental_constants" in data
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_physics_validate(self):
        """Physics validation endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/physics/validate",
                json={
                    "physics_data": {
                        "velocity": [1.0, 2.0, 3.0],
                        "mass": 10.0
                    },
                    "domain": "MECHANICS"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "is_valid" in data


class TestChatEndpoints:
    """Test Chat API Endpoints"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_chat_simple(self):
        """Simple chat endpoint"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{BASE_URL}/chat/simple",
                json={
                    "message": "Hello, test",
                    "user_id": "test_user"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["status"] == "success"


class TestBitNetEndpoints:
    """Test BitNet API Endpoints"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_bitnet_status(self):
        """BitNet status endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/bitnet/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestConsciousnessEndpoints:
    """Test Consciousness API Endpoints"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_consciousness_status(self):
        """Consciousness status endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/v4/consciousness/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_dashboard_complete(self):
        """Complete dashboard endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/v4/dashboard/complete")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "dashboard" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])

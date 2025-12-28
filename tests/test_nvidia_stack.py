#!/usr/bin/env python3
"""
Integration Tests for NVIDIA Stack 2025

Tests all NVIDIA components:
- Cosmos (Predict, Transfer, Reason)
- GR00T N1 (Humanoid control)
- Isaac Lab 2.2 (Robot learning)
- Unified NVIDIA API
- All API endpoints
"""

import asyncio
import pytest
import requests
import numpy as np
from typing import Dict, Any
import time


class TestCosmosIntegration:
    """Test Cosmos world foundation models"""
    
    @pytest.mark.asyncio
    async def test_cosmos_data_generator_initialization(self):
        """Test Cosmos data generator initializes correctly"""
        from src.agents.cosmos import get_cosmos_generator
        
        generator = get_cosmos_generator()
        success = await generator.initialize()
        
        assert success is True
        assert generator.initialized is True
        
    @pytest.mark.asyncio
    async def test_generate_training_data(self):
        """Test synthetic data generation"""
        from src.agents.cosmos import get_cosmos_generator
        
        generator = get_cosmos_generator()
        await generator.initialize()
        
        result = await generator.generate_robot_training_data(
            num_samples=10,
            tasks=["pick", "place"]
        )
        
        assert result["success"] is True
        assert result["samples_generated"] > 0
        assert "tasks" in result
        
    @pytest.mark.asyncio
    async def test_data_generation_caching(self):
        """Test that caching works correctly"""
        from src.agents.cosmos import get_cosmos_generator
        
        generator = get_cosmos_generator()
        await generator.initialize()
        
        # First call - should be cache miss
        result1 = await generator.generate_robot_training_data(
            num_samples=5,
            tasks=["test"]
        )
        cache_misses_1 = generator.stats["cache_misses"]
        
        # Second call - should be cache hit
        result2 = await generator.generate_robot_training_data(
            num_samples=5,
            tasks=["test"]
        )
        cache_hits_2 = generator.stats["cache_hits"]
        
        assert cache_hits_2 > 0
        assert result1["samples_generated"] == result2["samples_generated"]
    
    @pytest.mark.asyncio
    async def test_cosmos_reasoner_initialization(self):
        """Test Cosmos reasoner initializes correctly"""
        from src.agents.cosmos import get_cosmos_reasoner
        
        reasoner = get_cosmos_reasoner()
        success = await reasoner.initialize()
        
        assert success is True
        assert reasoner.initialized is True
    
    @pytest.mark.asyncio
    async def test_reasoning_about_task(self):
        """Test vision-language reasoning"""
        from src.agents.cosmos import get_cosmos_reasoner
        
        reasoner = get_cosmos_reasoner()
        await reasoner.initialize()
        
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = await reasoner.reason(
            image=image,
            task="Pick up the red box",
            constraints=["avoid obstacles"]
        )
        
        assert result["success"] is True
        assert "plan" in result
        assert len(result["plan"]) > 0
        assert "safety_check" in result


class TestGR00TIntegration:
    """Test GR00T N1 humanoid control"""
    
    @pytest.mark.asyncio
    async def test_groot_initialization(self):
        """Test GR00T agent initializes correctly"""
        from src.agents.groot import get_groot_agent
        
        agent = get_groot_agent()
        success = await agent.initialize()
        
        assert success is True
        assert agent.initialized is True
    
    @pytest.mark.asyncio
    async def test_task_execution(self):
        """Test humanoid task execution"""
        from src.agents.groot import get_groot_agent
        
        agent = get_groot_agent()
        await agent.initialize()
        
        result = await agent.execute_task(
            task="Walk forward 2 meters"
        )
        
        assert result["success"] is True
        assert "action_sequence" in result
        assert len(result["action_sequence"]) > 0
    
    @pytest.mark.asyncio
    async def test_retry_logic(self):
        """Test that retry logic works"""
        from src.agents.groot import get_groot_agent
        
        agent = get_groot_agent()
        await agent.initialize()
        
        # Execute multiple tasks
        for i in range(3):
            result = await agent.execute_task(
                task=f"Test task {i}"
            )
            assert result["success"] is True
        
        # Check stats
        assert agent.stats["tasks_executed"] == 3
        assert agent.stats["successful_executions"] > 0
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self):
        """Test getting humanoid capabilities"""
        from src.agents.groot import get_groot_agent
        
        agent = get_groot_agent()
        await agent.initialize()
        
        capabilities = await agent.get_capabilities()
        
        assert "robot_type" in capabilities
        assert "supported_tasks" in capabilities
        assert len(capabilities["supported_tasks"]) > 0


class TestIsaacLabIntegration:
    """Test Isaac Lab 2.2 robot learning"""
    
    @pytest.mark.asyncio
    async def test_isaac_lab_initialization(self):
        """Test Isaac Lab trainer initializes correctly"""
        from src.agents.isaac_lab import get_isaac_lab_trainer
        
        trainer = get_isaac_lab_trainer()
        success = await trainer.initialize()
        
        assert success is True
        assert trainer.initialized is True
    
    @pytest.mark.asyncio
    async def test_policy_training(self):
        """Test robot policy training"""
        from src.agents.isaac_lab import get_isaac_lab_trainer
        
        trainer = get_isaac_lab_trainer()
        await trainer.initialize()
        
        result = await trainer.train_policy(
            robot_type="franka_panda",
            task="reach",
            num_iterations=10,
            algorithm="PPO"
        )
        
        assert result["success"] is True
        assert "policy" in result
        assert result["best_reward"] >= 0
    
    @pytest.mark.asyncio
    async def test_policy_export(self):
        """Test policy export"""
        from src.agents.isaac_lab import get_isaac_lab_trainer
        
        trainer = get_isaac_lab_trainer()
        await trainer.initialize()
        
        # Train a policy first
        train_result = await trainer.train_policy(
            robot_type="franka_panda",
            task="reach",
            num_iterations=5
        )
        
        # Export it
        export_result = await trainer.export_policy(
            policy=train_result["policy"],
            format="onnx"
        )
        
        assert export_result["success"] is True
        assert "path" in export_result
    
    @pytest.mark.asyncio
    async def test_get_available_robots(self):
        """Test getting available robot models"""
        from src.agents.isaac_lab import get_isaac_lab_trainer
        
        trainer = get_isaac_lab_trainer()
        robots = trainer.get_available_robots()
        
        assert len(robots) > 0
        assert "franka_panda" in robots
    
    @pytest.mark.asyncio
    async def test_get_available_tasks(self):
        """Test getting available tasks"""
        from src.agents.isaac_lab import get_isaac_lab_trainer
        
        trainer = get_isaac_lab_trainer()
        tasks = trainer.get_available_tasks()
        
        assert len(tasks) > 0
        assert "reach" in tasks


class TestFullStackIntegration:
    """Test full stack integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_cosmos_to_groot_pipeline(self):
        """Test Cosmos reasoning -> GR00T execution"""
        from src.agents.cosmos import get_cosmos_reasoner
        from src.agents.groot import get_groot_agent
        
        # Step 1: Reason about task
        reasoner = get_cosmos_reasoner()
        await reasoner.initialize()
        
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        reasoning_result = await reasoner.reason(
            image=image,
            task="Pick up object",
            constraints=["safe"]
        )
        
        assert reasoning_result["success"] is True
        
        # Step 2: Execute with GR00T
        groot = get_groot_agent()
        await groot.initialize()
        
        execution_result = await groot.execute_task(
            task="Pick up object"
        )
        
        assert execution_result["success"] is True
    
    @pytest.mark.asyncio
    async def test_isaac_lab_to_cosmos_pipeline(self):
        """Test Isaac Lab training -> Cosmos data generation"""
        from src.agents.isaac_lab import get_isaac_lab_trainer
        from src.agents.cosmos import get_cosmos_generator
        
        # Step 1: Generate training data with Cosmos
        generator = get_cosmos_generator()
        await generator.initialize()
        
        data_result = await generator.generate_robot_training_data(
            num_samples=5,
            tasks=["reach"]
        )
        
        assert data_result["success"] is True
        
        # Step 2: Train policy with Isaac Lab
        trainer = get_isaac_lab_trainer()
        await trainer.initialize()
        
        train_result = await trainer.train_policy(
            robot_type="franka_panda",
            task="reach",
            num_iterations=5
        )
        
        assert train_result["success"] is True


class TestEndpointIntegration:
    """Test API endpoints"""
    
    def test_cosmos_endpoints_exist(self):
        """Test that Cosmos endpoints are registered"""
        from routes.cosmos import router
        
        routes = [route.path for route in router.routes]
        
        assert "/generate/training_data" in routes
        assert "/reason" in routes
        assert "/initialize" in routes
    
    def test_humanoid_endpoints_exist(self):
        """Test that humanoid endpoints are registered"""
        from routes.humanoid import router
        
        routes = [route.path for route in router.routes]
        
        assert "/execute_task" in routes
        assert "/capabilities" in routes
        assert "/initialize" in routes
    
    def test_isaac_lab_endpoints_exist(self):
        """Test that Isaac Lab endpoints are registered"""
        from routes.isaac_lab import router
        
        routes = [route.path for route in router.routes]
        
        assert "/train" in routes
        assert "/export" in routes
        assert "/robots" in routes


class TestAPIEndpoints:
    """Test all API endpoints"""
    
    BASE_URL = "http://localhost:8000"
    
    def test_health_endpoint(self):
        """Test health endpoint"""
        r = requests.get(f"{self.BASE_URL}/health", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
    
    def test_cosmos_status_endpoint(self):
        """Test Cosmos status endpoint"""
        r = requests.get(f"{self.BASE_URL}/cosmos/status", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "components" in data
    
    def test_cosmos_initialize_endpoint(self):
        """Test Cosmos initialize endpoint"""
        r = requests.post(f"{self.BASE_URL}/cosmos/initialize", json={}, timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
    
    def test_cosmos_generate_data_endpoint(self):
        """Test Cosmos data generation endpoint"""
        payload = {
            "num_samples": 10,
            "tasks": ["test"],
            "for_bitnet": False
        }
        r = requests.post(f"{self.BASE_URL}/cosmos/generate/training_data", json=payload, timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
    
    def test_cosmos_reason_endpoint(self):
        """Test Cosmos reasoning endpoint"""
        payload = {
            "task": "Test task",
            "constraints": ["safe"]
        }
        r = requests.post(f"{self.BASE_URL}/cosmos/reason", json=payload, timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "plan" in data
    
    def test_humanoid_capabilities_endpoint(self):
        """Test humanoid capabilities endpoint"""
        r = requests.get(f"{self.BASE_URL}/humanoid/capabilities", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "capabilities" in data
    
    def test_humanoid_initialize_endpoint(self):
        """Test humanoid initialize endpoint"""
        r = requests.post(f"{self.BASE_URL}/humanoid/initialize", json={}, timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
    
    def test_humanoid_execute_task_endpoint(self):
        """Test humanoid task execution endpoint"""
        payload = {
            "task": "Walk forward 2 meters"
        }
        r = requests.post(f"{self.BASE_URL}/humanoid/execute_task", json=payload, timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
    
    def test_isaac_lab_robots_endpoint(self):
        """Test Isaac Lab robots endpoint"""
        r = requests.get(f"{self.BASE_URL}/isaac_lab/robots", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "robots" in data
    
    def test_isaac_lab_tasks_endpoint(self):
        """Test Isaac Lab tasks endpoint"""
        r = requests.get(f"{self.BASE_URL}/isaac_lab/tasks", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "tasks" in data
    
    def test_isaac_lab_initialize_endpoint(self):
        """Test Isaac Lab initialize endpoint"""
        r = requests.post(f"{self.BASE_URL}/isaac_lab/initialize", json={}, timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
    
    def test_isaac_lab_train_endpoint(self):
        """Test Isaac Lab training endpoint"""
        payload = {
            "robot_type": "franka_panda",
            "task": "reach",
            "num_iterations": 5,
            "algorithm": "PPO"
        }
        r = requests.post(f"{self.BASE_URL}/isaac_lab/train", json=payload, timeout=15)
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
    
    def test_nvidia_unified_status_endpoint(self):
        """Test unified NVIDIA status endpoint"""
        r = requests.get(f"{self.BASE_URL}/nvidia/status", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "components" in data
    
    def test_nvidia_unified_capabilities_endpoint(self):
        """Test unified NVIDIA capabilities endpoint"""
        r = requests.get(f"{self.BASE_URL}/nvidia/capabilities", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "capabilities" in data
    
    def test_nvidia_unified_initialize_endpoint(self):
        """Test unified NVIDIA initialize endpoint"""
        r = requests.post(f"{self.BASE_URL}/nvidia/initialize", json={}, timeout=15)
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
    
    def test_nvidia_unified_stats_endpoint(self):
        """Test unified NVIDIA stats endpoint"""
        r = requests.get(f"{self.BASE_URL}/nvidia/stats", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "status" in data


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto", "-x"])

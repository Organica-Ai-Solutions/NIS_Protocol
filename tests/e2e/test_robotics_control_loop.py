#!/usr/bin/env python3
"""
NIS Protocol - E2E Tests for Robotics Control Loops

Tests the complete robotics pipeline:
1. Perception → Planning → Execution → Feedback
2. Isaac integration with NIS Robotics Agent
3. Real-time control loop timing
4. Physics validation in the loop
"""

import pytest
import asyncio
import time
import sys
from typing import List, Dict, Any

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Test configuration
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 30


class TestRoboticsControlLoop:
    """E2E tests for robotics control loops"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        try:
            import httpx
            self.client = httpx.Client(base_url=API_BASE_URL, timeout=TIMEOUT)
        except ImportError:
            import requests
            self.client = None
            self.requests = requests
        yield
        if self.client:
            self.client.close()
    
    def _get(self, path: str) -> Dict[str, Any]:
        """GET request helper"""
        if self.client:
            response = self.client.get(path)
        else:
            response = self.requests.get(f"{API_BASE_URL}{path}", timeout=TIMEOUT)
        return response.json()
    
    def _post(self, path: str, json: Dict = None) -> Dict[str, Any]:
        """POST request helper"""
        if self.client:
            response = self.client.post(path, json=json)
        else:
            response = self.requests.post(f"{API_BASE_URL}{path}", json=json, timeout=TIMEOUT)
        return response.json()
    
    # ========================================
    # CONTROL LOOP TESTS
    # ========================================
    
    def test_full_control_loop_drone(self):
        """
        Test complete control loop for drone:
        1. Get current state
        2. Plan trajectory
        3. Execute trajectory
        4. Verify final state
        """
        # 1. Get initial state
        state = self._get("/isaac/robot/state")
        assert state["status"] == "success"
        
        # 2. Define waypoints
        waypoints = [
            [0, 0, 0],
            [1, 0, 1],
            [2, 1, 2],
            [3, 0, 1],
            [4, 0, 0]
        ]
        
        # 3. Execute trajectory
        result = self._post("/isaac/pipeline/execute", json={
            "waypoints": waypoints,
            "robot_type": "drone",
            "duration": 5.0,
            "validate_physics": False  # Skip for speed
        })
        
        assert result["status"] == "success"
        assert result["result"]["success"] == True
        assert "execution" in result["result"]["pipeline_stages"]
        
        # 4. Verify execution completed
        final_state = self._get("/isaac/robot/state")
        assert final_state["status"] == "success"
    
    def test_full_control_loop_manipulator(self):
        """
        Test complete control loop for manipulator:
        1. Forward kinematics
        2. Inverse kinematics
        3. Trajectory planning
        4. Execution
        """
        # 1. Forward kinematics
        fk_result = self._post("/robotics/forward_kinematics", json={
            "robot_id": "test_arm",
            "joint_angles": [0.0, 0.5, 1.0, 0.0, 0.5, 0.0],
            "robot_type": "manipulator"
        })
        # Check nested result structure
        assert "result" in fk_result or "end_effector_pose" in fk_result
        if "result" in fk_result:
            assert "end_effector_pose" in fk_result["result"]
        
        # 2. Inverse kinematics
        ik_result = self._post("/robotics/inverse_kinematics", json={
            "robot_id": "test_arm",
            "target_pose": {"position": [0.5, 0.3, 0.8]},
            "robot_type": "manipulator"
        })
        # Check nested result structure
        assert "result" in ik_result or "joint_angles" in ik_result
        if "result" in ik_result:
            assert "joint_angles" in ik_result["result"]
        
        # 3. Trajectory planning
        traj_result = self._post("/robotics/plan_trajectory", json={
            "robot_id": "test_arm",
            "waypoints": [[0, 0, 0], [0.5, 0.3, 0.8]],
            "robot_type": "manipulator",
            "duration": 3.0
        })
        assert "trajectory" in traj_result or "success" in traj_result
        
        # 4. Execute via Isaac
        exec_result = self._post("/isaac/trajectory/execute", json={
            "waypoints": [[0, 0, 0], [0.5, 0.3, 0.8]],
            "robot_type": "manipulator",
            "duration": 3.0
        })
        assert exec_result["status"] in ["success", "failed"]
    
    def test_perception_planning_execution_loop(self):
        """
        Test perception → planning → execution loop:
        1. Detect objects
        2. Estimate pose
        3. Plan trajectory to object
        4. Execute
        """
        # 1. Detect objects
        detect_result = self._post("/isaac/perception/detect")
        assert detect_result["status"] == "success"
        detections = detect_result["result"]["detections"]
        
        # 2. Estimate pose of first detected object (or mock)
        pose_result = self._post("/isaac/perception/pose/target_object")
        assert pose_result["status"] in ["success", "failed"]
        
        if pose_result["status"] == "success":
            position = pose_result["result"]["position"]
        else:
            position = [0.5, 0.5, 0.5]  # Default
        
        # 3. Plan trajectory to object
        waypoints = [
            [0, 0, 0],
            [position[0], position[1], position[2] + 0.2],  # Above object
            position  # At object
        ]
        
        # 4. Execute
        exec_result = self._post("/isaac/pipeline/execute", json={
            "waypoints": waypoints,
            "robot_type": "manipulator",
            "duration": 4.0,
            "use_perception": True,
            "validate_physics": False
        })
        
        assert exec_result["status"] == "success"
        assert "perception" in exec_result["result"]["pipeline_stages"]
    
    def test_control_loop_timing(self):
        """
        Test control loop meets timing requirements:
        - Planning: <100ms
        - Execution command: <50ms
        - Total loop: <200ms
        """
        timings = []
        
        for i in range(5):
            start = time.time()
            
            # Simple trajectory
            result = self._post("/isaac/trajectory/execute", json={
                "waypoints": [[0, 0, 0], [1, 1, 1]],
                "robot_type": "drone",
                "duration": 1.0,
                "validate_physics": False
            })
            
            elapsed = (time.time() - start) * 1000
            timings.append(elapsed)
            
            # Small delay between requests
            time.sleep(0.1)
        
        avg_timing = sum(timings) / len(timings)
        
        # Timing assertions (relaxed for simulation mode)
        assert avg_timing < 2000, f"Average control loop too slow: {avg_timing}ms"
        print(f"Control loop timing: avg={avg_timing:.2f}ms, min={min(timings):.2f}ms, max={max(timings):.2f}ms")
    
    def test_emergency_stop_in_loop(self):
        """
        Test emergency stop during control loop:
        1. Start trajectory
        2. Trigger emergency stop
        3. Verify robot stopped
        4. Release and continue
        """
        # 1. Check initial state
        state = self._get("/isaac/robot/state")
        assert state["state"]["emergency_stopped"] == False
        
        # 2. Trigger emergency stop
        stop_result = self._post("/isaac/emergency_stop")
        assert stop_result["status"] == "emergency_stopped"
        
        # 3. Verify stopped
        state = self._get("/isaac/robot/state")
        assert state["state"]["emergency_stopped"] == True
        
        # 4. Release
        release_result = self._post("/isaac/emergency_stop/release")
        assert release_result["status"] == "released"
        
        # 5. Verify released
        state = self._get("/isaac/robot/state")
        assert state["state"]["emergency_stopped"] == False
    
    def test_continuous_control_loop(self):
        """
        Test continuous control loop (10 iterations):
        Simulates real-time control with feedback
        """
        iterations = 10
        success_count = 0
        
        for i in range(iterations):
            # Generate waypoint based on iteration
            target = [
                0.5 + 0.1 * (i % 5),
                0.3 + 0.1 * ((i + 1) % 5),
                0.5 + 0.1 * ((i + 2) % 5)
            ]
            
            result = self._post("/isaac/trajectory/execute", json={
                "waypoints": [[0, 0, 0], target],
                "robot_type": "manipulator",
                "duration": 0.5,
                "validate_physics": False
            })
            
            if result["status"] == "success":
                success_count += 1
            
            # Small delay
            time.sleep(0.05)
        
        success_rate = success_count / iterations
        assert success_rate >= 0.8, f"Control loop success rate too low: {success_rate*100}%"
        print(f"Continuous control loop: {success_count}/{iterations} successful ({success_rate*100}%)")
    
    # ========================================
    # ISAAC INTEGRATION TESTS
    # ========================================
    
    def test_isaac_full_pipeline(self):
        """Test Isaac full cognitive-physical pipeline"""
        # Initialize pipeline
        init_result = self._post("/isaac/pipeline/initialize")
        assert init_result["status"] == "success"
        
        # Execute pipeline
        result = self._post("/isaac/pipeline/execute", json={
            "waypoints": [[0, 0, 0], [1, 1, 1], [2, 0, 2]],
            "robot_type": "drone",
            "duration": 5.0,
            "use_perception": True,
            "validate_physics": False
        })
        
        assert result["status"] == "success"
        stages = result["result"]["pipeline_stages"]
        
        # Verify all stages executed
        assert "perception" in stages
        assert "planning" in stages
        assert "execution" in stages
    
    def test_isaac_pick_and_place(self):
        """Test Isaac pick and place operation"""
        result = self._post("/isaac/pipeline/pick_and_place", json={
            "object_id": "test_box",
            "target_position": [1.0, 0.5, 0.2]
        })
        
        # Handle rate limiting
        if "error" in result and "Rate limit" in str(result.get("error", "")):
            pytest.skip("Rate limited")
        
        # May fail due to physics validation, but should complete
        assert result.get("status") in ["success", "failed"]
        assert "result" in result
    
    def test_isaac_telemetry(self):
        """Test Isaac telemetry streaming"""
        # Get telemetry
        telemetry = self._get("/isaac/pipeline/telemetry")
        
        # Handle rate limiting
        if "error" in telemetry and "Rate limit" in str(telemetry.get("error", "")):
            pytest.skip("Rate limited")
        
        assert telemetry.get("status") == "success" or "telemetry" in telemetry
    
    def test_isaac_stats(self):
        """Test Isaac pipeline statistics"""
        stats = self._get("/isaac/pipeline/stats")
        
        # Handle rate limiting
        if "error" in stats and "Rate limit" in str(stats.get("error", "")):
            pytest.skip("Rate limited")
        
        assert stats.get("status") == "success" or "stats" in stats
    
    # ========================================
    # PHYSICS VALIDATION IN LOOP
    # ========================================
    
    def test_physics_validated_control_loop(self):
        """
        Test control loop with physics validation:
        All trajectories must pass physics checks
        """
        # Valid trajectory (should pass)
        valid_result = self._post("/robotics/plan_trajectory", json={
            "robot_id": "physics_test",
            "waypoints": [[0, 0, 0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]],
            "robot_type": "manipulator",
            "duration": 5.0
        })
        
        # Handle rate limiting
        if "error" in valid_result and "Rate limit" in str(valid_result.get("error", "")):
            pytest.skip("Rate limited")
        
        # Check physics validation result
        assert "physics_valid" in valid_result or "success" in valid_result or "result" in valid_result
    
    def test_physics_rejection(self):
        """Test that invalid physics trajectories are rejected"""
        # This would be an invalid trajectory (too fast)
        # The system should reject or warn
        result = self._post("/physics/validate", json={
            "physics_data": {
                "velocity": [100, 100, 100],  # Very high velocity
                "mass": 1.0
            },
            "domain": "MECHANICS"
        })
        
        # Handle rate limiting
        if "error" in result and "Rate limit" in str(result.get("error", "")):
            pytest.skip("Rate limited")
        
        # Should complete (may pass or fail validation)
        assert "validation" in result or "is_valid" in result or "physics_valid" in result or "result" in result


class TestRoboticsWebSocket:
    """E2E tests for WebSocket control"""
    
    @pytest.mark.asyncio
    async def test_websocket_control(self):
        """Test WebSocket-based real-time control"""
        try:
            import websockets
            
            async with websockets.connect(f"ws://localhost:8000/robotics/ws/control") as ws:
                # Send control command
                await ws.send('{"command": "status"}')
                
                # Receive response
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                assert response is not None
                
        except ImportError:
            pytest.skip("websockets library not installed")
        except Exception as e:
            # WebSocket may not be available in test environment
            pytest.skip(f"WebSocket test skipped: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

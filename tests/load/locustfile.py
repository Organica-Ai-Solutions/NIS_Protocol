#!/usr/bin/env python3
"""
NIS Protocol Load Testing with Locust
Comprehensive load testing for all major endpoints

Usage:
    # Install: pip install locust
    # Run: locust -f tests/load/locustfile.py --host=http://localhost:8000
    # Web UI: http://localhost:8089
    
    # Headless mode:
    # locust -f tests/load/locustfile.py --host=http://localhost:8000 --headless -u 100 -r 10 -t 60s
"""

import json
import random
from locust import HttpUser, task, between, tag


class NISProtocolUser(HttpUser):
    """
    Simulates a typical NIS Protocol user
    
    Performs a mix of:
    - Health checks (high frequency)
    - System status queries
    - Robotics operations
    - Physics validations
    - Chat requests (low frequency, expensive)
    """
    
    # Wait 0.5-2 seconds between tasks
    wait_time = between(0.5, 2)
    
    def on_start(self):
        """Called when user starts"""
        # Verify system is healthy
        response = self.client.get("/health")
        if response.status_code != 200:
            raise Exception("System not healthy")
    
    # ========================================
    # HEALTH & SYSTEM (High frequency)
    # ========================================
    
    @task(10)
    @tag("health", "critical")
    def health_check(self):
        """Health check - most frequent"""
        self.client.get("/health")
    
    @task(5)
    @tag("system")
    def system_status(self):
        """System status"""
        self.client.get("/system/status")
    
    @task(3)
    @tag("system")
    def root_endpoint(self):
        """Root endpoint"""
        self.client.get("/")
    
    # ========================================
    # INFRASTRUCTURE (Medium frequency)
    # ========================================
    
    @task(3)
    @tag("infrastructure")
    def infrastructure_status(self):
        """Infrastructure status"""
        self.client.get("/infrastructure/status")
    
    @task(2)
    @tag("infrastructure")
    def kafka_status(self):
        """Kafka status"""
        self.client.get("/infrastructure/kafka")
    
    @task(2)
    @tag("infrastructure")
    def redis_status(self):
        """Redis status"""
        self.client.get("/infrastructure/redis")
    
    # ========================================
    # ROBOTICS (Medium frequency)
    # ========================================
    
    @task(4)
    @tag("robotics")
    def robotics_capabilities(self):
        """Robotics capabilities"""
        self.client.get("/robotics/capabilities")
    
    @task(3)
    @tag("robotics", "compute")
    def forward_kinematics(self):
        """Forward kinematics computation"""
        joint_angles = [random.uniform(-1, 1) for _ in range(6)]
        self.client.post(
            "/robotics/forward_kinematics",
            json={
                "robot_id": f"test_arm_{random.randint(1, 10)}",
                "joint_angles": joint_angles,
                "robot_type": "manipulator"
            }
        )
    
    @task(2)
    @tag("robotics", "compute")
    def inverse_kinematics(self):
        """Inverse kinematics computation"""
        position = [random.uniform(0.3, 0.7) for _ in range(3)]
        self.client.post(
            "/robotics/inverse_kinematics",
            json={
                "robot_id": f"test_arm_{random.randint(1, 10)}",
                "target_pose": {"position": position},
                "robot_type": "manipulator"
            }
        )
    
    @task(2)
    @tag("robotics", "compute")
    def trajectory_planning(self):
        """Trajectory planning"""
        waypoints = [
            [0, 0, 0],
            [random.uniform(0, 2), random.uniform(0, 2), random.uniform(0, 2)],
            [random.uniform(0, 3), random.uniform(0, 3), random.uniform(0, 3)]
        ]
        self.client.post(
            "/robotics/plan_trajectory",
            json={
                "robot_id": f"test_drone_{random.randint(1, 5)}",
                "waypoints": waypoints,
                "robot_type": "drone",
                "duration": random.uniform(3, 10)
            }
        )
    
    # ========================================
    # CAN & OBD (Medium frequency)
    # ========================================
    
    @task(2)
    @tag("can")
    def can_status(self):
        """CAN protocol status"""
        self.client.get("/robotics/can/status")
    
    @task(2)
    @tag("obd")
    def obd_status(self):
        """OBD-II status"""
        self.client.get("/robotics/obd/status")
    
    @task(1)
    @tag("obd")
    def obd_vehicle(self):
        """OBD vehicle data"""
        self.client.get("/robotics/obd/vehicle")
    
    # ========================================
    # PHYSICS (Medium frequency)
    # ========================================
    
    @task(3)
    @tag("physics")
    def physics_capabilities(self):
        """Physics capabilities"""
        self.client.get("/physics/capabilities")
    
    @task(2)
    @tag("physics")
    def physics_constants(self):
        """Physics constants"""
        self.client.get("/physics/constants")
    
    @task(2)
    @tag("physics", "compute")
    def physics_validate(self):
        """Physics validation"""
        self.client.post(
            "/physics/validate",
            json={
                "physics_data": {
                    "velocity": [random.uniform(0, 5) for _ in range(3)],
                    "mass": random.uniform(1, 100)
                },
                "domain": "MECHANICS"
            }
        )
    
    # ========================================
    # OBSERVABILITY (Low frequency)
    # ========================================
    
    @task(1)
    @tag("observability")
    def observability_status(self):
        """Observability status"""
        self.client.get("/observability/status")
    
    @task(1)
    @tag("observability")
    def metrics_json(self):
        """Metrics JSON"""
        self.client.get("/observability/metrics/json")
    
    # ========================================
    # SECURITY (Low frequency)
    # ========================================
    
    @task(1)
    @tag("security")
    def security_status(self):
        """Security status"""
        self.client.get("/security/status")
    
    @task(1)
    @tag("security")
    def security_roles(self):
        """Security roles"""
        self.client.get("/security/roles")
    
    # ========================================
    # CONSCIOUSNESS (Low frequency)
    # ========================================
    
    @task(2)
    @tag("consciousness")
    def consciousness_status(self):
        """Consciousness status"""
        self.client.get("/v4/consciousness/status")
    
    @task(1)
    @tag("consciousness")
    def dashboard_complete(self):
        """Complete dashboard"""
        self.client.get("/v4/dashboard/complete")


class RoboticsHeavyUser(HttpUser):
    """
    Simulates a robotics-focused user
    Heavy on kinematics and trajectory planning
    """
    
    wait_time = between(0.1, 0.5)
    weight = 2  # Less common than regular users
    
    @task(5)
    def forward_kinematics(self):
        """Rapid FK computations"""
        joint_angles = [random.uniform(-1, 1) for _ in range(6)]
        self.client.post(
            "/robotics/forward_kinematics",
            json={
                "robot_id": "rapid_arm",
                "joint_angles": joint_angles,
                "robot_type": "manipulator"
            }
        )
    
    @task(3)
    def inverse_kinematics(self):
        """Rapid IK computations"""
        position = [random.uniform(0.3, 0.7) for _ in range(3)]
        self.client.post(
            "/robotics/inverse_kinematics",
            json={
                "robot_id": "rapid_arm",
                "target_pose": {"position": position},
                "robot_type": "manipulator"
            }
        )
    
    @task(2)
    def trajectory_planning(self):
        """Trajectory planning"""
        waypoints = [[random.uniform(0, 2) for _ in range(3)] for _ in range(5)]
        self.client.post(
            "/robotics/plan_trajectory",
            json={
                "robot_id": "rapid_drone",
                "waypoints": waypoints,
                "robot_type": "drone",
                "duration": 5.0
            }
        )
    
    @task(1)
    def can_status(self):
        """CAN status check"""
        self.client.get("/robotics/can/status")


class MonitoringUser(HttpUser):
    """
    Simulates a monitoring/ops user
    Focuses on health, metrics, and observability
    """
    
    wait_time = between(1, 3)
    weight = 1  # Less common
    
    @task(5)
    def health_check(self):
        """Health monitoring"""
        self.client.get("/health")
    
    @task(3)
    def metrics(self):
        """Metrics collection"""
        self.client.get("/observability/metrics/prometheus")
    
    @task(2)
    def infrastructure(self):
        """Infrastructure monitoring"""
        self.client.get("/infrastructure/status")
    
    @task(2)
    def observability(self):
        """Observability status"""
        self.client.get("/observability/status")
    
    @task(1)
    def traces(self):
        """Recent traces"""
        self.client.get("/observability/traces")
    
    @task(1)
    def security_audit(self):
        """Security audit log"""
        self.client.get("/security/audit-log")

#!/usr/bin/env python3
"""
NIS Protocol API Test Suite
Run comprehensive API tests against a running server
"""

import requests
import json
import time
import sys
import os
from typing import Dict, Any, List, Tuple

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = os.getenv("NIS_API_URL", "http://localhost:8000")

# Test results
passed = 0
failed = 0
results: List[Tuple[str, bool, str]] = []


def test(name: str, condition: bool, message: str = ""):
    """Record test result"""
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}: {message}")
    results.append((name, condition, message))


def test_endpoint(method: str, path: str, expected_status: int = 200, 
                  json_data: Dict = None, check_fields: List[str] = None) -> Dict:
    """Test an API endpoint"""
    url = f"{BASE_URL}{path}"
    time.sleep(0.1)  # Rate limit protection
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=json_data, timeout=30)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        test(f"{method} {path} returns {expected_status}", 
             response.status_code == expected_status,
             f"Got {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if check_fields:
                for field in check_fields:
                    test(f"{path} has '{field}'", field in data, f"Missing {field}")
            return data
        return {}
    except Exception as e:
        test(f"{method} {path}", False, str(e))
        return {}


def main():
    print("=" * 60)
    print("🧪 NIS Protocol API Test Suite")
    print("=" * 60)
    print(f"Target: {BASE_URL}")
    print()
    
    start_time = time.time()
    
    # ========================================
    # HEALTH & SYSTEM TESTS
    # ========================================
    print("\n📋 Health & System Tests")
    print("-" * 40)
    
    test_endpoint("GET", "/health", check_fields=["status", "version"])
    test_endpoint("GET", "/", check_fields=["system", "version"])
    
    # Metrics returns Prometheus format, not JSON
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=10)
        test("GET /metrics returns 200", response.status_code == 200)
        test("/metrics has Prometheus data", "nis_health_status" in response.text)
    except Exception as e:
        test("GET /metrics", False, str(e))
    
    test_endpoint("GET", "/system/status", check_fields=["status"])
    
    # ========================================
    # INFRASTRUCTURE TESTS
    # ========================================
    print("\n🔧 Infrastructure Tests")
    print("-" * 40)
    
    data = test_endpoint("GET", "/infrastructure/status", 
                         check_fields=["status", "infrastructure"])
    if data:
        infra = data.get("infrastructure", {})
        test("Kafka connected", infra.get("kafka", {}).get("connected", False))
        test("Redis connected", infra.get("redis", {}).get("connected", False))
    
    test_endpoint("GET", "/infrastructure/kafka", check_fields=["kafka"])
    test_endpoint("GET", "/infrastructure/redis", check_fields=["redis"])
    test_endpoint("GET", "/runner/status", check_fields=["runner"])
    
    # ========================================
    # ROBOTICS TESTS
    # ========================================
    print("\n🤖 Robotics Tests")
    print("-" * 40)
    
    test_endpoint("GET", "/robotics/capabilities", 
                  check_fields=["status", "capabilities"])
    
    # Forward Kinematics
    fk_data = test_endpoint("POST", "/robotics/forward_kinematics", json_data={
        "robot_id": "test_arm",
        "joint_angles": [0.0, 0.5, 1.0, 0.0, 0.5, 0.0],
        "robot_type": "manipulator"
    }, check_fields=["status", "result"])
    
    if fk_data:
        result = fk_data.get("result", {})
        test("FK success", result.get("success", False))
        test("FK physics valid", result.get("physics_valid", False))
        test("FK has end effector pose", "end_effector_pose" in result)
    
    # Inverse Kinematics
    ik_data = test_endpoint("POST", "/robotics/inverse_kinematics", json_data={
        "robot_id": "test_arm",
        "target_pose": {"position": [0.5, 0.3, 0.8]},
        "robot_type": "manipulator"
    }, check_fields=["status", "result"])
    
    if ik_data:
        result = ik_data.get("result", {})
        test("IK success", result.get("success", False))
        test("IK has joint angles", "joint_angles" in result)
        error = result.get("position_error", 1.0)
        test("IK position error < 1cm", error < 0.01, f"Error: {error}")
    
    # Trajectory Planning
    traj_data = test_endpoint("POST", "/robotics/plan_trajectory", json_data={
        "robot_id": "test_drone",
        "waypoints": [[0, 0, 0], [1, 1, 1], [2, 0, 2]],
        "robot_type": "drone",
        "duration": 5.0
    }, check_fields=["status", "result"])
    
    if traj_data:
        result = traj_data.get("result", {})
        test("Trajectory success", result.get("success", False))
        test("Trajectory physics valid", result.get("physics_valid", False))
        test("Trajectory has points", len(result.get("trajectory", [])) > 0)
    
    # ========================================
    # CAN PROTOCOL TESTS
    # ========================================
    print("\n🔌 CAN Protocol Tests")
    print("-" * 40)
    
    can_data = test_endpoint("GET", "/robotics/can/status", 
                             check_fields=["status", "can_protocol"])
    if can_data:
        protocol = can_data.get("can_protocol", {})
        test("CAN enabled", protocol.get("enabled", False))
    
    test_endpoint("GET", "/robotics/can/safety", check_fields=["safety_limits"])
    
    # ========================================
    # OBD-II TESTS
    # ========================================
    print("\n🚗 OBD-II Tests")
    print("-" * 40)
    
    obd_data = test_endpoint("GET", "/robotics/obd/status", 
                             check_fields=["status", "obd_protocol", "vehicle_state"])
    if obd_data:
        protocol = obd_data.get("obd_protocol", {})
        test("OBD running", protocol.get("is_running", False))
        test("OBD has integration info", "integration" in obd_data)
    
    test_endpoint("GET", "/robotics/obd/vehicle", check_fields=["vehicle"])
    test_endpoint("GET", "/robotics/obd/dtcs", check_fields=["dtcs"])
    test_endpoint("GET", "/robotics/obd/safety", check_fields=["safety_thresholds"])
    
    # ========================================
    # PHYSICS TESTS
    # ========================================
    print("\n⚛️ Physics Tests")
    print("-" * 40)
    
    physics_data = test_endpoint("GET", "/physics/capabilities", 
                                 check_fields=["status", "domains"])
    if physics_data:
        test("Physics active", physics_data.get("status") == "active")
        domains = physics_data.get("domains", [])
        test("Has mechanics domain", "mechanics" in domains)
        test("Has thermodynamics domain", "thermodynamics" in domains)
    
    test_endpoint("GET", "/physics/constants", 
                  check_fields=["fundamental_constants"])
    
    validate_data = test_endpoint("POST", "/physics/validate", json_data={
        "physics_data": {"velocity": [1.0, 2.0, 3.0], "mass": 10.0},
        "domain": "MECHANICS"
    }, check_fields=["is_valid"])
    
    # ========================================
    # BITNET TESTS
    # ========================================
    print("\n🧠 BitNet Tests")
    print("-" * 40)
    
    test_endpoint("GET", "/bitnet/status", check_fields=["status"])
    test_endpoint("GET", "/bitnet/training/status", check_fields=["status"])
    
    # ========================================
    # CONSCIOUSNESS TESTS
    # ========================================
    print("\n🌟 Consciousness Tests")
    print("-" * 40)
    
    consciousness_data = test_endpoint("GET", "/v4/consciousness/status", 
                                       check_fields=["status", "phases"])
    if consciousness_data:
        test("Consciousness operational", 
             consciousness_data.get("status") == "operational")
    
    dashboard_data = test_endpoint("GET", "/v4/dashboard/complete", 
                                   check_fields=["status", "dashboard"])
    if dashboard_data:
        dashboard = dashboard_data.get("dashboard", {})
        test("Dashboard has system health", "system_health" in dashboard)
        test("Dashboard has agents", "agents" in dashboard)
    
    # ========================================
    # SUMMARY
    # ========================================
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"  Total:  {passed + failed}")
    print(f"  Passed: {passed} ✅")
    print(f"  Failed: {failed} ❌")
    print(f"  Time:   {elapsed:.2f}s")
    print()
    
    if failed > 0:
        print("Failed Tests:")
        for name, success, message in results:
            if not success:
                print(f"  ❌ {name}: {message}")
        print()
    
    success_rate = (passed / (passed + failed)) * 100 if (passed + failed) > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 95:
        print("\n🎉 EXCELLENT - All critical tests passed!")
    elif success_rate >= 80:
        print("\n✅ GOOD - Most tests passed")
    elif success_rate >= 60:
        print("\n⚠️ WARNING - Some tests failed")
    else:
        print("\n❌ CRITICAL - Many tests failed")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

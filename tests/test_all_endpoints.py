#!/usr/bin/env python3
import requests
import time
import json
import sys

BASE_URL = "http://localhost:8007"

def check_endpoint(method, path, data=None, expected_status=200, description=""):
    print(f"\n🧪 Testing {description} ({path})...")
    url = f"{BASE_URL}{path}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        if response.status_code == expected_status:
            print(f"✅ SUCCESS: {response.status_code}")
            try:
                print(f"   Response: {json.dumps(response.json(), indent=2)[:300]}...")
            except:
                print(f"   Response: {response.text[:300]}...")
            return True
        else:
            print(f"❌ FAIL: Expected {expected_status}, got {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return False
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

def wait_for_health(retries=60):
    print("⏳ Waiting for server health...")
    for i in range(retries):
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                print("✅ Server is UP!")
                return True
        except:
            pass
        time.sleep(2)
    print("❌ Server timed out.")
    return False

def run_tests():
    if not wait_for_health():
        sys.exit(1)

    tests = [
        {
            "method": "POST",
            "path": "/v4/consciousness/genesis",
            "data": {"capability": "visual_reasoning"},
            "desc": "Genesis (Agent Creation)"
        },
        {
            "method": "POST",
            "path": "/v4/consciousness/plan",
            "data": {"goal_id": "test_plan_1", "high_level_goal": "Scan network security"},
            "desc": "Plan Generation"
        },
        {
            "method": "POST",
            "path": "/v4/consciousness/collective/decide",
            "data": {"problem": "Resource allocation", "local_decision": {"priority": "high"}},
            "desc": "Collective Decision"
        },
        {
            "method": "POST",
            "path": "/v4/consciousness/multipath/start",
            "data": {"problem": "Paradox resolution", "num_paths": 2},
            "desc": "Multipath Reasoning"
        },
        {
            "method": "POST",
            "path": "/v4/consciousness/embodiment/action/execute",
            "data": {"action_type": "calibrate_sensors", "parameters": {"duration": 5}},
            "desc": "Embodied Action"
        },
        {
            "method": "POST",
            "path": "/v4/consciousness/ethics/evaluate",
            "data": {"decision_context": {"action": "shutdown_server", "reason": "maintenance"}},
            "desc": "Ethical Evaluation"
        },
        {
            "method": "POST",
            "path": "/v4/consciousness/debug/record",
            "data": {
                "decision_type": "test",
                "inputs": {"a": 1},
                "output": "b",
                "reasoning": ["test"],
                "confidence": 0.9
            },
            "desc": "Debug Record"
        },
        {
            "method": "POST",
            "path": "/v4/consciousness/marketplace/publish",
            "data": {
                "insight_type": "pattern",
                "content": {"pattern": "A->B"},
                "metadata": {"confidence": 0.8}
            },
            "desc": "Marketplace Publish"
        }
    ]

    failures = 0
    for t in tests:
        if not check_endpoint(t["method"], t["path"], t.get("data"), description=t["desc"]):
            failures += 1

    print(f"\n🏁 Tests Completed. Failures: {failures}")
    if failures > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()

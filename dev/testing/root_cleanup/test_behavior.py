#!/usr/bin/env python3
import requests
import json

# Test agent behavior modification
data = {
    "mode": "parameter_tuning", 
    "behavior_changes": {
        "learning_rate": 0.01,
        "exploration_factor": 0.3
    }
}

try:
    resp = requests.post("http://localhost:8000/agent/behavior/test_agent_001", json=data, timeout=5)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        result = resp.json()
        print(json.dumps(result, indent=2))
    else:
        print(f"Error: {resp.text}")
except Exception as e:
    print(f"Exception: {e}")
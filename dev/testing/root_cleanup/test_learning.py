#!/usr/bin/env python3
import requests
import json

# Test learning process endpoint
print("Testing Learning Process Endpoint...")

data = {
    "operation": "learn_from_data",
    "learning_data": "Neural network training data with input features and target labels for supervised learning",
    "learning_type": "supervised",
    "algorithm": "neural_network"
}

try:
    resp = requests.post("http://localhost:8000/agents/learning/process", json=data, timeout=15)
    print(f"Status Code: {resp.status_code}")
    
    if resp.status_code == 200:
        result = resp.json()
        print("SUCCESS! Response:")
        print(json.dumps(result, indent=2))
    elif resp.status_code == 422:
        print("Parameter validation error:")
        print(resp.json())
    else:
        print(f"Error {resp.status_code}:")
        print(resp.text)
        
except requests.exceptions.Timeout:
    print("Request timed out (>15s)")
except Exception as e:
    print(f"Exception: {e}")

print("\nTest completed.")
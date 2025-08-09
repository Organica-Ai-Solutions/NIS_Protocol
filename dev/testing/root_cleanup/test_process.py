#!/usr/bin/env python3
import requests
import json

# Test general processing endpoint
data = {
    "text": "Quantum entanglement research shows promising applications in computing",
    "context": "scientific research analysis",
    "processing_type": "comprehensive_analysis"
}

try:
    resp = requests.post("http://localhost:8000/process", json=data, timeout=10)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        result = resp.json()
        print(json.dumps(result, indent=2))
    else:
        print(f"Error: {resp.text}")
except Exception as e:
    print(f"Exception: {e}")
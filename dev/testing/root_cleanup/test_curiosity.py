#!/usr/bin/env python3
import requests
import json

# Test curiosity endpoint (this worked before)
print("=== TESTING CURIOSITY ENDPOINT ===")

data = {
    "stimulus": {
        "type": "discovery",
        "content": "Revolutionary breakthrough in quantum error correction achieved",
        "source": "scientific_journal",
        "complexity": "high",
        "novelty": 0.9
    }
}

try:
    resp = requests.post("http://localhost:8000/agents/curiosity/process_stimulus", json=data, timeout=10)
    print(f"Status: {resp.status_code}")
    
    if resp.status_code == 200:
        print("✅ CURIOSITY ENDPOINT WORKING")
        result = resp.json()
        print(json.dumps(result, indent=2)[:500] + "...")
    else:
        print(f"❌ Error: {resp.text[:200]}")
        
except Exception as e:
    print(f"Exception: {e}")

print("\n" + "="*50)

# Test a simple working endpoint for comparison
print("=== TESTING HEALTH ENDPOINT ===")

try:
    resp = requests.get("http://localhost:8000/health", timeout=5)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        health = resp.json()
        print(f"✅ System Status: {health['status']}")
        print(f"✅ Providers: {len(health['provider'])} active")
        print(f"✅ Conversations: {health['conversations_active']}")
    else:
        print(f"❌ Health check failed: {resp.status_code}")
except Exception as e:
    print(f"Health check exception: {e}")

print("Testing complete.")
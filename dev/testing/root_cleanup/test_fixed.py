#!/usr/bin/env python3
import requests
import json
import time

def test_endpoint(method, url, data=None, description=""):
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Method: {method} {url}")
    
    try:
        start_time = time.time()
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        response_time = time.time() - start_time
        
        print(f"Status: {response.status_code}")
        print(f"Time: {response_time:.3f}s")
        
        if response.status_code == 200:
            try:
                response_data = response.json()
                print("Response:")
                print(json.dumps(response_data, indent=2)[:500] + "..." if len(str(response_data)) > 500 else json.dumps(response_data, indent=2))
                return True
            except:
                print("Response (non-JSON):")
                print(response.text[:200] + "..." if len(response.text) > 200 else response.text)
                return True
        else:
            print(f"Error: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"Exception: {str(e)}")
        return False

# Test all endpoints
base_url = "http://localhost"

tests = [
    ("GET", f"{base_url}/metrics", None, "Fixed metrics endpoint"),
    ("POST", f"{base_url}/agents/learning/process", 
     {"operation": "get_params"}, "Learning agent with valid operation"),
    ("POST", f"{base_url}/agents/planning/create_plan", 
     {"goal": "test physics validation"}, "Planning agent"),
    ("POST", f"{base_url}/agents/curiosity/process_stimulus", 
     {"stimulus": "physics experiment"}, "Curiosity agent"),
    ("POST", f"{base_url}/agents/alignment/evaluate_ethics", 
     {"scenario": "AI decision making", "context": "scientific research"}, "Ethics agent"),
]

print("🧪 Testing Fixed NIS Protocol Endpoints")
print("=" * 60)

results = []
for method, url, data, desc in tests:
    success = test_endpoint(method, url, data, desc)
    results.append((desc, success))

print(f"\n{'='*60}")
print("📊 SUMMARY")
print("=" * 60)

working = sum(1 for _, success in results if success)
total = len(results)

for desc, success in results:
    status = "✅ WORKING" if success else "❌ BROKEN"
    print(f"{status}: {desc}")

print(f"\n🎯 Success Rate: {working}/{total} ({working/total*100:.1f}%)")
#!/usr/bin/env python3
"""
Quick NIS Protocol v3.1 Endpoint Test
Test key endpoints to validate our v3.1 implementation
"""

import requests
import json
import time

def test_endpoint(method, endpoint, data=None):
    """Test an endpoint"""
    try:
        url = f"http://localhost:8000{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        print(f"{method} {endpoint}: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS")
            if isinstance(result, dict) and len(str(result)) > 200:
                print(f"   Response preview: {str(result)[:200]}...")
            else:
                print(f"   Response: {result}")
        else:
            print(f"❌ FAILED: {response.text[:100]}")
        print("-" * 50)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("-" * 50)
        return False

def main():
    print("🚀 Quick NIS Protocol v3.1 Endpoint Test")
    print("=" * 60)
    
    tests = [
        # Core endpoints
        ("GET", "/", None),
        ("GET", "/health", None),
        
        # v3.1 Conversational
        ("POST", "/chat", {"message": "Hello v3.1!", "user_id": "tester"}),
        
        # v3.1 Internet & Knowledge
        ("POST", "/internet/search", {"query": "AI consciousness", "max_results": 3}),
        ("GET", "/internet/status", None),
        
        # v3.1 Tools
        ("GET", "/tool/list", None),
        ("POST", "/tool/execute", {"tool_name": "calculator", "parameters": {"expression": "2+2"}}),
        
        # v3.1 Models
        ("GET", "/models", None),
        
        # v3.1 Experimental
        ("POST", "/kan/predict", {"input_data": [1.0, 2.0, 3.0]}),
        ("POST", "/pinn/verify", {"system_state": {"x": 1}, "physical_laws": ["conservation_energy"]}),
    ]
    
    passed = 0
    total = len(tests)
    
    for method, endpoint, data in tests:
        if test_endpoint(method, endpoint, data):
            passed += 1
    
    print(f"🎯 Results: {passed}/{total} endpoints working ({passed/total*100:.1f}%)")
    
    if passed >= total * 0.8:
        print("🎉 EXCELLENT! NIS Protocol v3.1 is working great!")
    elif passed >= total * 0.6:
        print("👍 GOOD! Most v3.1 endpoints are functional!")
    else:
        print("⚠️ Some endpoints need attention.")
    
    return passed >= total * 0.6

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 
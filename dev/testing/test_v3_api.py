#!/usr/bin/env python3
"""
Quick test script for NIS Protocol v3.0 Step 3 API
Tests all the endpoints to ensure the system is working
"""

import requests
import json
import time
from typing import Dict, Any

def test_endpoint(url: str, method: str = "GET", data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Test an API endpoint and return the response"""
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        response.raise_for_status()
        return {
            "success": True,
            "status_code": response.status_code,
            "data": response.json()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "status_code": getattr(e, 'response', {}).get('status_code', 'N/A')
        }

def main():
    """Test all NIS Protocol v3.0 Step 3 endpoints"""
    base_url = "http://localhost:8000"
    
    print("🚀 Testing NIS Protocol v3.0 Step 3 API")
    print("=" * 50)
    
    # Test endpoints
    endpoints = [
        {"url": f"{base_url}/", "method": "GET", "name": "Root"},
        {"url": f"{base_url}/health", "method": "GET", "name": "Health Check"},
        {"url": f"{base_url}/metrics", "method": "GET", "name": "System Metrics"},
        {"url": f"{base_url}/consciousness/status", "method": "GET", "name": "Consciousness Status"},
        {"url": f"{base_url}/infrastructure/status", "method": "GET", "name": "Infrastructure Status"},
        {"url": f"{base_url}/test", "method": "GET", "name": "Test Endpoint"},
    ]
    
    # POST endpoints
    post_endpoints = [
        {
            "url": f"{base_url}/process",
            "method": "POST", 
            "name": "Process Input",
            "data": {"text": "Hello NIS Protocol v3.0!", "generate_speech": False}
        },
        {
            "url": f"{base_url}/chat",
            "method": "POST",
            "name": "Enhanced Chat", 
            "data": {"message": "Test v3.0 chat functionality", "user_id": "test_user"}
        }
    ]
    
    # Test GET endpoints
    print("\n📡 Testing GET Endpoints:")
    print("-" * 30)
    
    results = {}
    for endpoint in endpoints:
        print(f"Testing {endpoint['name']}... ", end="")
        result = test_endpoint(endpoint["url"], endpoint["method"])
        
        if result["success"]:
            print(f"✅ {result['status_code']}")
            results[endpoint["name"]] = "✅ PASS"
        else:
            print(f"❌ {result.get('status_code', 'ERROR')}")
            print(f"   Error: {result['error']}")
            results[endpoint["name"]] = f"❌ FAIL: {result['error']}"
    
    # Test POST endpoints
    print("\n📤 Testing POST Endpoints:")
    print("-" * 30)
    
    for endpoint in post_endpoints:
        print(f"Testing {endpoint['name']}... ", end="")
        result = test_endpoint(endpoint["url"], endpoint["method"], endpoint["data"])
        
        if result["success"]:
            print(f"✅ {result['status_code']}")
            results[endpoint["name"]] = "✅ PASS"
            # Show response preview for POST endpoints
            if "data" in result and "response" in result["data"]:
                response_preview = result["data"]["response"][:100] + "..." if len(result["data"]["response"]) > 100 else result["data"]["response"]
                print(f"   Response: {response_preview}")
        else:
            print(f"❌ {result.get('status_code', 'ERROR')}")
            print(f"   Error: {result['error']}")
            results[endpoint["name"]] = f"❌ FAIL: {result['error']}"
    
    # Summary
    print("\n🎯 Test Results Summary:")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result.startswith("✅"))
    total = len(results)
    
    for name, result in results.items():
        print(f"{name:20} {result}")
    
    print("-" * 50)
    print(f"📊 Results: {passed}/{total} endpoints passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! NIS Protocol v3.0 Step 3 is working perfectly!")
        return True
    else:
        print(f"⚠️ {total - passed} endpoints failed. Check the errors above.")
        return False

if __name__ == "__main__":
    main() 
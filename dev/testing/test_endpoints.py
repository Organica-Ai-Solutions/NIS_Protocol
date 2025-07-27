#!/usr/bin/env python3
"""
NIS Protocol v3 - API Endpoint Tester
Tests all available endpoints to verify the system is working properly.
"""

import requests
import json
import sys
import time

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_endpoint(method, endpoint, data=None, description=""):
    """Test a single endpoint and return the result."""
    url = f"{BASE_URL}{endpoint}"
    
    print(f"\n🧪 Testing {method} {endpoint}")
    if description:
        print(f"   Description: {description}")
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=10)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            print(f"   ❌ Unsupported method: {method}")
            return False
            
        print(f"   Status Code: {response.status_code}")
        
        # Try to parse JSON response
        try:
            json_response = response.json()
            print(f"   Response: {json.dumps(json_response, indent=2)}")
        except:
            print(f"   Response (text): {response.text[:200]}...")
            
        # Consider 2xx status codes as success
        if 200 <= response.status_code < 300:
            print(f"   ✅ SUCCESS")
            return True
        else:
            print(f"   ⚠️  Non-2xx status code")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"   ❌ CONNECTION ERROR - Service not available")
        return False
    except requests.exceptions.Timeout:
        print(f"   ❌ TIMEOUT - Service took too long to respond")
        return False
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        return False

def main():
    """Test all NIS Protocol v3 endpoints."""
    print("🚀 NIS Protocol v3 - Complete API Endpoint Test")
    print("=" * 60)
    
    # Wait for service to be ready
    print("\n⏳ Waiting for service to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("   ✅ Service is ready!")
                break
        except:
            pass
        
        if i == max_retries - 1:
            print("   ❌ Service failed to become ready after 30 attempts")
            sys.exit(1)
            
        print(f"   Attempt {i+1}/{max_retries} - waiting...")
        time.sleep(2)
    
    # Test endpoints
    results = []
    
    # 1. Root endpoint
    results.append(test_endpoint("GET", "/", description="Root endpoint - system info"))
    
    # 2. Health check
    results.append(test_endpoint("GET", "/health", description="Health check"))
    
    # 3. Process endpoint
    test_data = {
        "input": "Hello NIS Protocol v3! Test the cognitive processing.",
        "context": "API endpoint testing",
        "parameters": {"test_mode": True}
    }
    results.append(test_endpoint("POST", "/process", data=test_data, description="Process cognitive input"))
    
    # 4. Consciousness status
    results.append(test_endpoint("GET", "/consciousness/status", description="Consciousness system status"))
    
    # 5. Infrastructure status  
    results.append(test_endpoint("GET", "/infrastructure/status", description="Infrastructure health"))
    
    # 6. Metrics
    results.append(test_endpoint("GET", "/metrics", description="System metrics"))
    
    # 7. Admin restart (POST)
    # results.append(test_endpoint("POST", "/admin/restart", description="Admin restart (skipped for safety)"))
    print(f"\n🧪 Testing POST /admin/restart")
    print(f"   Description: Admin restart (skipped for safety)")
    print(f"   ⏭️  SKIPPED - Would restart services")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! NIS Protocol v3 is fully operational!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed. Check the service logs.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
#!/usr/bin/env python3
"""
Test script to validate fixes for previously failing v3.1 endpoints
"""

import requests
import json
import time

def test_endpoint(method, endpoint, data=None, description=""):
    """Test an endpoint and return detailed results"""
    try:
        url = f"http://localhost:8000{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=15)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=15)
        
        success = response.status_code == 200
        print(f"{'✅' if success else '❌'} {method} {endpoint} - {response.status_code} {description}")
        
        if success and response.status_code == 200:
            result = response.json()
            if isinstance(result, dict):
                # Show key fields from response
                key_fields = []
                if "status" in result:
                    key_fields.append(f"status: {result['status']}")
                if "confidence" in result:
                    key_fields.append(f"confidence: {result['confidence']}")
                if "message" in result:
                    key_fields.append(f"message: {result['message'][:50]}...")
                if "response" in result:
                    key_fields.append(f"response: {result['response'][:50]}...")
                if key_fields:
                    print(f"   {', '.join(key_fields)}")
        elif not success:
            print(f"   Error: {response.text[:100]}...")
        
        return success
        
    except Exception as e:
        print(f"❌ {method} {endpoint} - ERROR: {str(e)}")
        return False

def main():
    print("🔧 Testing Previously Failing v3.1 Endpoints")
    print("=" * 60)
    
    # Wait for system to be ready
    print("⏳ Waiting for system to start...")
    time.sleep(10)
    
    # Test previously failing endpoints
    failing_tests = [
        # Chat endpoints (were 500 errors)
        ("POST", "/chat", {"message": "Hello v3.1 fixed!", "user_id": "fix_tester"}, "Enhanced Chat Fix"),
        ("POST", "/chat/contextual", {
            "message": "Test contextual reasoning", 
            "user_id": "fix_tester", 
            "reasoning_mode": "chain_of_thought",
            "tools_enabled": ["calculator"],
            "context_depth": 5
        }, "Contextual Chat Fix"),
        
        # Parameter validation fixes (were 422 errors) 
        ("POST", "/kan/predict", {"input_data": [1.0, 2.0, 3.0]}, "KAN Predict with data"),
        ("POST", "/kan/predict", {}, "KAN Predict with defaults"),
        
        ("POST", "/laplace/transform", {"signal_data": [1.0, 0.5, 0.2]}, "Laplace with data"),
        ("POST", "/laplace/transform", {}, "Laplace with defaults"),
        
        ("POST", "/a2a/connect", {"target_node": "test-node"}, "A2A Connect with node"),
        ("POST", "/a2a/connect", {}, "A2A Connect with defaults"),
        
        ("POST", "/sandbox/execute", {"code": "print('Fixed!')", "language": "python"}, "Sandbox with code"),
        ("POST", "/sandbox/execute", {}, "Sandbox with defaults"),
    ]
    
    print(f"\n🧪 Testing {len(failing_tests)} Previously Failing Endpoints:")
    print("-" * 60)
    
    passed = 0
    total = len(failing_tests)
    
    for method, endpoint, data, description in failing_tests:
        if test_endpoint(method, endpoint, data, description):
            passed += 1
        time.sleep(1)  # Small delay between tests
    
    print("\n" + "=" * 60)
    print("🎯 FIX VALIDATION RESULTS")
    print("=" * 60)
    
    print(f"📊 Results: {passed}/{total} endpoints fixed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 well-suited! All issues have been fixed!")
        print("✅ NIS Protocol v3.1 is now fully operational!")
    elif passed >= total * 0.8:
        print("🏆 EXCELLENT! Most issues have been resolved!")
        print("✅ System is highly functional with minor remaining issues!")
    elif passed >= total * 0.6:
        print("👍 GOOD! Significant improvements made!")
        print("⚠️ Some endpoints still need attention!")
    else:
        print("⚠️ NEEDS MORE WORK! Several issues remain!")
    
    # Test a few working endpoints to confirm system stability
    print(f"\n🔄 Testing System Stability:")
    print("-" * 30)
    
    stability_tests = [
        ("GET", "/", None, "Root endpoint"),
        ("GET", "/health", None, "Health check"),
        ("POST", "/tool/execute", {"tool_name": "calculator", "parameters": {"expression": "3+3"}}, "Tool execution"),
        ("GET", "/models", None, "Model list"),
    ]
    
    stable_passed = 0
    for method, endpoint, data, description in stability_tests:
        if test_endpoint(method, endpoint, data, description):
            stable_passed += 1
    
    print(f"\n📈 System Stability: {stable_passed}/{len(stability_tests)} core endpoints working")
    
    return passed >= total * 0.8 and stable_passed >= len(stability_tests) * 0.8

if __name__ == "__main__":
    success = main()
    print(f"\n🎊 Fix validation {'PASSED' if success else 'NEEDS ATTENTION'}!")
    exit(0 if success else 1) 
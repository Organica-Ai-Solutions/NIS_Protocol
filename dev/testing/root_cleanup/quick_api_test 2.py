#!/usr/bin/env python3
"""
Quick API Test - Test core endpoints immediately as soon as backend starts
"""

import time
import requests
import sys
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_endpoint(method, endpoint, data=None, timeout=5):
    """Test a single endpoint quickly"""
    url = f"{BASE_URL}{endpoint}"
    start_time = time.time()
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=timeout)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=timeout)
        else:
            return f"❌ Unsupported method: {method}"
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            return f"✅ {method} {endpoint} - {response.status_code} ({response_time:.3f}s)"
        elif response.status_code < 500:
            return f"⚠️ {method} {endpoint} - {response.status_code} ({response_time:.3f}s)"
        else:
            return f"❌ {method} {endpoint} - {response.status_code} ({response_time:.3f}s)"
            
    except requests.exceptions.ConnectionError:
        return f"🔌 {method} {endpoint} - Connection refused (backend not ready)"
    except requests.exceptions.Timeout:
        return f"⏰ {method} {endpoint} - Timeout"
    except Exception as e:
        return f"💥 {method} {endpoint} - Error: {str(e)}"

def main():
    print("🚀 Quick API Test for NIS Protocol v3.2")
    print("=" * 50)
    
    # Wait for backend to start
    print("⏳ Waiting for backend to initialize...")
    
    # Test critical endpoints
    critical_tests = [
        ("GET", "/health"),
        ("GET", "/"),
        ("GET", "/docs"),
        ("GET", "/physics/capabilities"),
        ("GET", "/nvidia/nemo/enterprise/showcase"),
        ("POST", "/chat", {"message": "Hello", "session_id": "test"}),
    ]
    
    max_retries = 12  # 60 seconds total
    retry_count = 0
    
    while retry_count < max_retries:
        print(f"\n🔄 Test attempt {retry_count + 1}/{max_retries}")
        
        results = []
        all_failed = True
        
        for method, endpoint, *data in critical_tests:
            payload = data[0] if data else None
            result = test_endpoint(method, endpoint, payload)
            results.append(result)
            print(f"  {result}")
            
            if "✅" in result or "⚠️" in result:
                all_failed = False
        
        if not all_failed:
            print(f"\n🎉 Backend is responding! Some endpoints are working.")
            break
        
        if retry_count < max_retries - 1:
            print("⏳ Backend not ready yet, waiting 5 seconds...")
            time.sleep(5)
        
        retry_count += 1
    
    if retry_count >= max_retries:
        print(f"\n💔 Backend failed to start after {max_retries * 5} seconds")
        print("💡 Check Docker logs: docker logs nis-backend")
        return False
    
    # Test NVIDIA NeMo showcase endpoints
    print(f"\n🚀 Testing NVIDIA NeMo Integration...")
    nemo_tests = [
        ("GET", "/nvidia/nemo/status"),
        ("GET", "/nvidia/nemo/cosmos/demo"),
        ("GET", "/nvidia/nemo/toolkit/status"),
    ]
    
    for method, endpoint in nemo_tests:
        result = test_endpoint(method, endpoint)
        print(f"  {result}")
    
    print(f"\n✅ Quick test completed at {datetime.now().strftime('%H:%M:%S')}")
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        sys.exit(1)

#!/usr/bin/env python3
"""
Quick NIS Protocol System Check
Validates basic functionality in under 30 seconds
"""

import requests
import time
import json

def check_system():
    base_url = "http://localhost:8000"
    
    print("🔍 QUICK NIS PROTOCOL SYSTEM CHECK")
    print("=" * 40)
    
    # 1. Health Check
    print("1. 🏥 Checking system health...")
    try:
        start = time.time()
        response = requests.get(f"{base_url}/health", timeout=5)
        duration = time.time() - start
        
        if response.status_code == 200:
            print(f"   ✅ Health check OK ({duration*1000:.0f}ms)")
            health_data = response.json()
            print(f"   📊 Status: {health_data.get('status', 'unknown')}")
        else:
            print(f"   ❌ Health check failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False
    
    # 2. API Endpoints
    print("\n2. 🌐 Testing key endpoints...")
    endpoints = [
        ("/", "Root"),
        ("/consciousness/status", "Consciousness"),
        ("/infrastructure/status", "Infrastructure"),
        ("/metrics", "Metrics")
    ]
    
    working_endpoints = 0
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"   ✅ {name} endpoint OK")
                working_endpoints += 1
            else:
                print(f"   ❌ {name} endpoint failed: HTTP {response.status_code}")
        except Exception as e:
            print(f"   ❌ {name} endpoint failed: {e}")
    
    # 3. Basic Processing Test
    print("\n3. 🧠 Testing basic processing...")
    try:
        payload = {
            "text": "What is 2+2?",
            "context": "simple_test",
            "processing_type": "analysis"
        }
        
        start = time.time()
        response = requests.post(f"{base_url}/process", json=payload, timeout=15)
        duration = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Processing works ({duration:.1f}s)")
            print(f"   📝 Response length: {len(str(result))} chars")
            
            # Check for basic response structure
            expected_fields = ["response_text", "confidence", "processing_time"]
            found_fields = [field for field in expected_fields if field in result]
            print(f"   📊 Response fields: {len(found_fields)}/{len(expected_fields)}")
            
        else:
            print(f"   ❌ Processing failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Processing failed: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 40)
    total_score = working_endpoints + 2  # health + processing
    max_score = len(endpoints) + 2
    
    if total_score >= max_score - 1:
        print("🎉 SYSTEM STATUS: EXCELLENT")
        print("   Your NIS Protocol is running properly!")
    elif total_score >= max_score // 2:
        print("⚠️  SYSTEM STATUS: PARTIAL")
        print("   Some components working, check logs for issues")
    else:
        print("❌ SYSTEM STATUS: NEEDS ATTENTION")
        print("   Multiple failures detected")
    
    print(f"📊 Score: {total_score}/{max_score}")
    return total_score >= max_score // 2

if __name__ == "__main__":
    success = check_system()
    exit(0 if success else 1) 
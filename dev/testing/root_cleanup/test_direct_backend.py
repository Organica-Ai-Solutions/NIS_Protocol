#!/usr/bin/env python3
import requests
import json

def test_direct_backend():
    print("Testing direct backend on port 8000...")
    
    try:
        # Test health endpoint directly
        print("\n1. Testing direct health endpoint...")
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Direct backend health endpoint works!")
            print(f"Response: {json.dumps(data, indent=2)[:200]}...")
        else:
            print(f"❌ Direct backend health failed: {response.text}")
    
        # Test metrics endpoint directly  
        print("\n2. Testing direct metrics endpoint...")
        response = requests.get("http://localhost:8000/metrics", timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Direct backend metrics endpoint works!")
            print(f"Response: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ Direct backend metrics failed: {response.text}")
            
        # Test via nginx
        print("\n3. Testing via nginx proxy...")
        response = requests.get("http://localhost/health", timeout=5)
        print(f"Nginx proxy status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Nginx proxy works!")
        else:
            print(f"❌ Nginx proxy failed: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_direct_backend()
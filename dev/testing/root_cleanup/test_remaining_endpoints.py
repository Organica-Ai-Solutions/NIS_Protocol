#!/usr/bin/env python3
"""Test remaining endpoints systematically"""

import requests
import json

def test_endpoint(method, url, data=None, timeout=10):
    """Test an endpoint and return formatted results"""
    try:
        if method.upper() == 'GET':
            response = requests.get(url, timeout=timeout)
        elif method.upper() == 'POST':
            response = requests.post(url, json=data, timeout=timeout)
        
        if response.status_code == 200:
            try:
                return response.json()
            except:
                return response.text[:500] + "..." if len(response.text) > 500 else response.text
        else:
            return f"HTTP {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    base_url = "http://localhost:8000"
    
    # GET endpoints to test
    get_endpoints = [
        "/console",
        "/infrastructure/status", 
        "/metrics",
        "/training/bitnet/status",
        "/training/bitnet/metrics"
    ]
    
    print("🔍 TESTING REMAINING GET ENDPOINTS")
    print("=" * 60)
    
    for endpoint in get_endpoints:
        print(f"\n=== GET {endpoint} ===")
        result = test_endpoint('GET', f"{base_url}{endpoint}")
        if isinstance(result, dict):
            print(json.dumps(result, indent=2))
        else:
            print(result)
    
    # POST endpoints to test
    post_endpoints = [
        {
            "endpoint": "/chat/stream",
            "data": {"message": "Hello", "conversation_id": "test"}
        },
        {
            "endpoint": "/image/edit", 
            "data": {"image_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==", "edit_prompt": "make it blue"}
        },
        {
            "endpoint": "/reasoning/debate",
            "data": {"topic": "Is AI consciousness possible?", "positions": ["yes", "no"]}
        },
        {
            "endpoint": "/research/validate",
            "data": {"claim": "The sky is blue", "evidence_required": "scientific"}
        },
        {
            "endpoint": "/visualization/create",
            "data": {"data_type": "simple", "chart_type": "bar", "title": "Test Chart"}
        },
        {
            "endpoint": "/agent/create",
            "data": {"agent_type": "test", "name": "test_agent"}
        }
    ]
    
    print("\n\n🔍 TESTING REMAINING POST ENDPOINTS")
    print("=" * 60)
    
    for test_case in post_endpoints:
        endpoint = test_case["endpoint"]
        data = test_case["data"]
        print(f"\n=== POST {endpoint} ===")
        result = test_endpoint('POST', f"{base_url}{endpoint}", data)
        if isinstance(result, dict):
            print(json.dumps(result, indent=2))
        else:
            print(result)

if __name__ == "__main__":
    main()
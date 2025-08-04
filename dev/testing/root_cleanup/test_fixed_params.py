#!/usr/bin/env python3
"""Test endpoints with corrected parameters"""

import requests
import json

def test_endpoint(method, url, data=None, timeout=15):
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
    
    # POST endpoints with corrected parameters
    post_endpoints = [
        {
            "endpoint": "/image/edit",
            "data": {
                "image_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==", 
                "prompt": "make it blue",
                "edit_type": "modification"
            }
        },
        {
            "endpoint": "/reasoning/debate",
            "data": {
                "problem": "Is AI consciousness possible?",
                "debate_style": "formal",
                "positions": ["yes", "no"]
            }
        },
        {
            "endpoint": "/visualization/create",
            "data": {
                "data": [1, 2, 3, 4, 5],
                "chart_type": "bar",
                "title": "Test Chart"
            }
        },
        {
            "endpoint": "/agents/alignment/evaluate_ethics",
            "data": {
                "scenario": "An AI is asked to help with medical diagnosis",
                "ethical_frameworks": ["utilitarian", "deontological"]
            }
        },
        {
            "endpoint": "/agents/audit/text",
            "data": {
                "text": "This is a sample text for audit",
                "audit_type": "comprehensive"
            }
        },
        {
            "endpoint": "/agents/curiosity/process_stimulus",
            "data": {
                "stimulus": "A new scientific discovery about quantum computing",
                "stimulus_type": "information"
            }
        },
        {
            "endpoint": "/agents/learning/process",
            "data": {
                "learning_data": "Example learning content",
                "learning_type": "supervised"
            }
        },
        {
            "endpoint": "/agents/planning/create_plan",
            "data": {
                "goal": "Build a machine learning model",
                "constraints": ["time: 1 week", "budget: $1000"]
            }
        },
        {
            "endpoint": "/agents/simulation/run",
            "data": {
                "simulation_type": "physics",
                "parameters": {"gravity": 9.8, "time": 10}
            }
        },
        {
            "endpoint": "/simulation/run",
            "data": {
                "simulation_config": {
                    "type": "physics",
                    "duration": 10,
                    "physics_laws": ["gravity", "conservation_energy"]
                }
            }
        },
        {
            "endpoint": "/nvidia/process",
            "data": {
                "input_data": "sample data for nvidia processing",
                "processing_type": "inference"
            }
        },
        {
            "endpoint": "/process", 
            "data": {
                "input": "sample input data",
                "processing_mode": "standard"
            }
        }
    ]
    
    print("🔍 TESTING ENDPOINTS WITH CORRECTED PARAMETERS")
    print("=" * 70)
    
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
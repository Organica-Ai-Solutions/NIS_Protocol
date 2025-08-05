#!/usr/bin/env python3
import requests
import json
import time

print("=== TESTING NVIDIA PROCESSING ENDPOINT ===")

# Test different NVIDIA processing scenarios
test_cases = [
    {
        "name": "Physics Simulation",
        "data": {
            "prompt": "Accelerate quantum mechanics simulation using NVIDIA GPU",
            "input_data": "Quantum state vectors and Hamiltonian matrices for parallel processing",
            "processing_type": "physics_simulation",
            "model_type": "quantum_physics"
        }
    },
    {
        "name": "Neural Network Inference", 
        "data": {
            "prompt": "Run neural network inference on physics data",
            "input_data": "Tensor data for neural network processing",
            "processing_type": "inference",
            "model_type": "neural_network"
        }
    },
    {
        "name": "Scientific Computing",
        "data": {
            "prompt": "Perform scientific computation with GPU acceleration",
            "input_data": "Large matrix operations for scientific analysis",
            "processing_type": "scientific_computing",
            "model_type": "computational"
        }
    }
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{i}. Testing {test_case['name']}:")
    
    start_time = time.time()
    try:
        resp = requests.post("http://localhost:8000/nvidia/process", 
                           json=test_case["data"], timeout=15)
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"Status: {resp.status_code}")
        print(f"Response Time: {response_time:.4f}s")
        
        if resp.status_code == 200:
            print(f"✅ {test_case['name']} processing successful")
            result = resp.json()
            print(f"Response keys: {list(result.keys())}")
            
            # Check for processing details
            if "processing" in result:
                proc = result["processing"]
                print(f"Processing status: {proc.get('status', 'unknown')}")
                print(f"Provider used: {proc.get('provider_used', 'unknown')}")
                if 'processing_time' in proc:
                    print(f"Internal processing time: {proc['processing_time']}")
                    
        elif resp.status_code == 422:
            print(f"⚠️ Parameter error for {test_case['name']}")
            print(resp.json())
        else:
            print(f"❌ Error {resp.status_code}: {resp.text[:150]}")
            
    except requests.exceptions.Timeout:
        print(f"⏱️ {test_case['name']} timed out (>15s)")
    except Exception as e:
        print(f"❌ Exception: {e}")

print("\nNVIDIA processing testing completed.")
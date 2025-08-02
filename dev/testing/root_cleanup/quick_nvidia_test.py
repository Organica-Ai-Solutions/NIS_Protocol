#!/usr/bin/env python3
"""
Quick NVIDIA API Test - Run this directly from your command prompt
python quick_nvidia_test.py
"""

import time
import os

# Check for required packages
try:
    import requests
    import json
except ImportError:
    print("[ERROR] Required packages not found. Please run install_dependencies.py first.")
    print("Run: python install_dependencies.py")
    import sys
    sys.exit(1)

def load_api_key():
    """Load NVIDIA API key from .env file or environment."""
    try:
        # Try to load from .env file
        with open('.env~', 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if line.startswith('NVIDIA_API_KEY='):
                    return line.split('=', 1)[1].strip()
    except:
        pass
    
    # Try environment variable
    return os.getenv('NVIDIA_API_KEY')

def test_nvidia_api():
    """Quick test of NVIDIA API with Nemotron models."""
    print("\n" + "=" * 60)
    print(" QUICK NVIDIA NEMOTRON TEST")
    print("=" * 60)
    
    # Load API key
    api_key = load_api_key()
    if not api_key:
        print("[ERROR] No NVIDIA API key found!")
        return
    
    print(f"[OK] API key loaded: {api_key[:10]}...")
    
    # Test connection
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    print("\n>> Testing connection...")
    try:
        response = requests.get(
            "https://integrate.api.nvidia.com/v1/models",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            models = response.json()
            print(f"[OK] Connected! Found {len(models.get('data', []))} models")
            
            # Find Nemotron/Llama models
            nemotron_models = []
            for model in models.get('data', []):
                model_id = model.get('id', '').lower()
                if 'nemotron' in model_id or 'llama' in model_id:
                    nemotron_models.append(model.get('id'))
            
            print(f"\n>> Found {len(nemotron_models)} Nemotron/Llama models:")
            for model in nemotron_models[:5]:  # Show first 5
                print(f"   - {model}")
            
            # Test physics reasoning with the best available model
            if nemotron_models:
                print(f"\n>> Testing physics reasoning with {nemotron_models[0]}...")
                
                test_payload = {
                    "model": nemotron_models[0],
                    "messages": [
                        {
                            "role": "user",
                            "content": """Physics Analysis Request:
Temperature: 323.15K, Pressure: 105000Pa, Velocity: 15m/s, Density: 1.2kg/m³
Task: Validate physics consistency and conservation laws. Provide detailed analysis."""
                        }
                    ],
                    "max_tokens": 500,
                    "temperature": 0.1
                }
                
                start_time = time.time()
                response = requests.post(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    headers=headers,
                    json=test_payload,
                    timeout=60
                )
                inference_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    
                    print(f"[OK] Physics reasoning successful!")
                    print(f"Inference time: {inference_time:.2f}s")
                    print(f"Response length: {len(content)} characters")
                    print(f"\n>> Physics Analysis Preview:")
                    print("-" * 40)
                    print(content[:300] + "..." if len(content) > 300 else content)
                    print("-" * 40)
                    
                    # Check for physics accuracy indicators
                    physics_keywords = ['conservation', 'energy', 'momentum', 'pressure', 'temperature', 'physics']
                    found_keywords = sum(1 for keyword in physics_keywords if keyword.lower() in content.lower())
                    
                    accuracy_score = (found_keywords / len(physics_keywords)) * 100
                    print(f"\n>> Physics Accuracy Score: {accuracy_score:.1f}%")
                    
                    if accuracy_score > 80:
                        print("[EXCELLENT] HIGH ACCURACY - Ready for NIS Protocol integration!")
                    elif accuracy_score > 60:
                        print("[GOOD] GOOD ACCURACY - Suitable for physics reasoning")
                    else:
                        print("[MODERATE] MODERATE ACCURACY - May need fine-tuning")
                    
                    # Speed assessment
                    if inference_time < 2.0:
                        print("[FAST] FAST INFERENCE - Excellent speed performance!")
                    elif inference_time < 4.0:
                        print("[GOOD] GOOD SPEED - Suitable for real-time applications")
                    else:
                        print("[MODERATE] MODERATE SPEED - Good for batch processing")
                    
                    print(f"\n>> INTEGRATION RECOMMENDATION:")
                    print(f"   Model: {nemotron_models[0]}")
                    print(f"   Use case: {'Real-time physics validation' if inference_time < 2.0 else 'Batch physics analysis'}")
                    print(f"   Accuracy: {'Production ready' if accuracy_score > 80 else 'Development ready'}")
                    
                else:
                    print(f"[ERROR] Physics test failed: {response.status_code}")
                    print(f"Error: {response.text[:200]}...")
            
        else:
            print(f"[ERROR] Connection failed: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
    
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
    
    print(f"\n[COMPLETE] NVIDIA API test complete!")

if __name__ == "__main__":
    test_nvidia_api()
    
    # Keep console window open if run directly
    if os.name == 'nt':  # Windows
        print("\nPress Enter to exit...")
        input()
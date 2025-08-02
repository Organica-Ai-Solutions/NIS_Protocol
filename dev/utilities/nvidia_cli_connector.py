#!/usr/bin/env python3
"""
NVIDIA CLI Connector for NIS Protocol
Direct connection to NVIDIA's API services and Nemotron models.

Features:
- Connect to NVIDIA API with your API key
- Access Nemotron models (Nano, Super, Ultra)
- Test model availability and performance
- Integrate with NIS Protocol physics validation
"""

import requests
import json
import time
import os
from typing import Dict, List, Optional, Any
from pathlib import Path

class NVIDIAConnector:
    """Connect to NVIDIA's API services and Nemotron models."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('NVIDIA_API_KEY')
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json"
        }
        
        print("🔌 NVIDIA CLI Connector initialized")
        if self.api_key:
            print(f"✅ API key loaded: {self.api_key[:8]}...")
        else:
            print("⚠️ No API key found - limited functionality")
    
    def test_connection(self) -> bool:
        """Test connection to NVIDIA API."""
        try:
            print("\n🔍 Testing NVIDIA API connection...")
            
            if not self.api_key:
                print("❌ No API key provided")
                print("📝 To get an API key:")
                print("   1. Go to https://build.nvidia.com/")
                print("   2. Sign up/login")
                print("   3. Get your free API key")
                print("   4. Set NVIDIA_API_KEY environment variable")
                return False
            
            # Test with models endpoint
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                print("✅ NVIDIA API connection successful!")
                models = response.json()
                print(f"📊 Found {len(models.get('data', []))} available models")
                return True
            else:
                print(f"❌ API connection failed: {response.status_code}")
                print(f"Response: {response.text[:200]}...")
                return False
                
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def list_nemotron_models(self) -> List[Dict[str, Any]]:
        """List available Nemotron models."""
        try:
            print("\n🔍 Searching for Nemotron models...")
            
            if not self.api_key:
                print("⚠️ API key required to list models")
                return []
            
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                all_models = response.json().get('data', [])
                nemotron_models = []
                
                for model in all_models:
                    model_id = model.get('id', '').lower()
                    if 'nemotron' in model_id or 'llama' in model_id:
                        nemotron_models.append({
                            'id': model.get('id'),
                            'name': model.get('name', 'Unknown'),
                            'description': model.get('description', 'No description'),
                            'owned_by': model.get('owned_by', 'Unknown')
                        })
                
                print(f"✅ Found {len(nemotron_models)} Nemotron/Llama models:")
                for model in nemotron_models:
                    print(f"   📦 {model['id']}")
                    print(f"      Name: {model['name']}")
                    print(f"      Owner: {model['owned_by']}")
                
                return nemotron_models
            else:
                print(f"❌ Failed to list models: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []
    
    def test_nemotron_inference(self, model_id: str = "meta/llama-3.3-70b-instruct") -> Dict[str, Any]:
        """Test inference with a Nemotron/Llama model."""
        try:
            print(f"\n🧠 Testing inference with {model_id}...")
            
            if not self.api_key:
                print("⚠️ API key required for inference")
                return {"error": "No API key"}
            
            # Physics reasoning test prompt
            test_prompt = """
You are a physics reasoning AI. Analyze this physics scenario:

Temperature: 323.15 K (50°C)
Pressure: 105000 Pa
Velocity: 15 m/s  
Density: 1.2 kg/m³

Tasks:
1. Check if this state is physically consistent
2. Validate conservation laws (energy, momentum, mass)
3. Calculate key physics metrics
4. Provide reasoning about the physics validity

Respond with detailed physics analysis.
"""
            
            payload = {
                "model": model_id,
                "messages": [
                    {
                        "role": "user", 
                        "content": test_prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.1,
                "stream": False
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            inference_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                print("✅ Inference successful!")
                print(f"⏱️ Inference time: {inference_time:.2f}s")
                print(f"📝 Response length: {len(content)} characters")
                print("\n🔬 Physics Analysis Response:")
                print("=" * 50)
                print(content[:500] + "..." if len(content) > 500 else content)
                print("=" * 50)
                
                return {
                    "success": True,
                    "model": model_id,
                    "inference_time": inference_time,
                    "response_length": len(content),
                    "content": content,
                    "usage": result.get('usage', {})
                }
            else:
                print(f"❌ Inference failed: {response.status_code}")
                print(f"Error: {response.text[:200]}...")
                return {"error": f"HTTP {response.status_code}", "details": response.text}
                
        except Exception as e:
            print(f"❌ Inference test failed: {e}")
            return {"error": str(e)}
    
    def benchmark_models(self) -> Dict[str, Any]:
        """Benchmark available Nemotron models for physics reasoning."""
        try:
            print("\n📊 Benchmarking Nemotron models for physics reasoning...")
            
            # Priority model list (most likely to be available)
            test_models = [
                "meta/llama-3.3-70b-instruct",
                "meta/llama-3.1-70b-instruct", 
                "meta/llama-3.1-8b-instruct",
                "microsoft/phi-3-medium-4k-instruct",
                "nvidia/llama-3.1-nemotron-70b-instruct"
            ]
            
            benchmark_results = {}
            
            for model_id in test_models:
                print(f"\n🧪 Testing {model_id}...")
                result = self.test_nemotron_inference(model_id)
                benchmark_results[model_id] = result
                
                if result.get("success"):
                    print(f"✅ {model_id}: {result['inference_time']:.2f}s")
                else:
                    print(f"❌ {model_id}: {result.get('error', 'Failed')}")
                
                # Small delay between requests
                time.sleep(1)
            
            print("\n📋 BENCHMARK SUMMARY:")
            print("=" * 60)
            for model_id, result in benchmark_results.items():
                if result.get("success"):
                    time_ms = result['inference_time'] * 1000
                    tokens = result.get('usage', {}).get('total_tokens', 'N/A')
                    print(f"✅ {model_id}")
                    print(f"   ⏱️ Time: {time_ms:.0f}ms")
                    print(f"   🔤 Tokens: {tokens}")
                    print(f"   📝 Length: {result['response_length']} chars")
                else:
                    print(f"❌ {model_id}: {result.get('error', 'Failed')}")
                print()
            
            return benchmark_results
            
        except Exception as e:
            print(f"❌ Benchmarking failed: {e}")
            return {"error": str(e)}
    
    def setup_api_key_instructions(self):
        """Provide instructions for setting up NVIDIA API key."""
        print("\n🔑 NVIDIA API KEY SETUP INSTRUCTIONS")
        print("=" * 50)
        print("1. Go to: https://build.nvidia.com/")
        print("2. Sign up or log in with your NVIDIA account")
        print("3. Click 'Get API Key' (free for development)")
        print("4. Copy your API key")
        print("5. Set environment variable:")
        print("   Windows: set NVIDIA_API_KEY=your_key_here")
        print("   Linux/Mac: export NVIDIA_API_KEY=your_key_here")
        print("6. Or create a .env file with: NVIDIA_API_KEY=your_key_here")
        print("\n📝 Free tier includes:")
        print("   - 1,000 requests per month")
        print("   - Access to latest models")
        print("   - Serverless inference")
        print("=" * 50)

def main():
    """Run NVIDIA CLI connector with comprehensive testing."""
    print("🚀 NVIDIA CLI CONNECTOR FOR NIS PROTOCOL")
    print("=" * 60)
    
    # Initialize connector
    connector = NVIDIAConnector()
    
    # Test connection
    if not connector.test_connection():
        connector.setup_api_key_instructions()
        return
    
    # List available models
    models = connector.list_nemotron_models()
    
    # Run benchmarks
    if models:
        print("\n🎯 Running physics reasoning benchmarks...")
        benchmark_results = connector.benchmark_models()
        
        # Save results
        timestamp = int(time.time())
        results_file = f"nvidia_benchmark_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'available_models': models,
                'benchmark_results': benchmark_results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Integration recommendations
        print("\n🔗 INTEGRATION RECOMMENDATIONS:")
        print("=" * 50)
        
        successful_models = [
            model for model, result in benchmark_results.items() 
            if result.get("success")
        ]
        
        if successful_models:
            fastest_model = min(
                successful_models, 
                key=lambda m: benchmark_results[m].get('inference_time', float('inf'))
            )
            
            print(f"🚀 Fastest model: {fastest_model}")
            print(f"   ⏱️ Time: {benchmark_results[fastest_model]['inference_time']:.2f}s")
            print(f"\n📝 Recommended integration:")
            print(f"   - Use {fastest_model} for real-time physics validation")
            print(f"   - Integrate with NIS Protocol Nemotron agents")
            print(f"   - Deploy for production physics reasoning")
        else:
            print("⚠️ No models successfully tested")
            print("   Check API key and network connection")
    
    print("\n✅ NVIDIA CLI connector testing complete!")

if __name__ == "__main__":
    main()
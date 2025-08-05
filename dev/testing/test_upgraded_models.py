#!/usr/bin/env python3
"""
🧪 NIS Protocol v3.2 - Upgraded Models Test Suite
Tests all the latest AI models: GPT-4o, Claude 4 Sonnet, Gemini 2.5 Flash, DeepSeek R1
"""

import requests
import json
import time
import sys

def test_provider(provider_name, expected_model, display_name):
    """Test a specific provider and verify the model"""
    print(f"\n🔧 Testing: {display_name}")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            headers={"Content-Type": "application/json"},
            json={
                "message": f"Hello! Please confirm you are {display_name}",
                "provider": provider_name
            },
            timeout=30
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            actual_provider = data.get("provider", "unknown")
            actual_model = data.get("model", "unknown")
            confidence = data.get("confidence", 0)
            response_length = len(data.get("response", ""))
            
            print(f"✅ SUCCESS ({duration:.2f}s)")
            print(f"   Provider Used: {actual_provider}")
            print(f"   Model Used: {actual_model}")
            print(f"   Expected Model: {expected_model}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Response Length: {response_length}")
            
            # Verify correct provider and model
            provider_match = actual_provider == provider_name
            model_contains_expected = expected_model.lower() in actual_model.lower()
            
            if provider_match and (model_contains_expected or actual_model == expected_model):
                print(f"   ✅ Model correctly matched: {actual_model}")
                return True, duration, actual_model
            else:
                print(f"   ⚠️ Model mismatch - got: {actual_model}")
                return False, duration, actual_model
        else:
            print(f"❌ FAILED - HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False, duration, "error"
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ ERROR ({duration:.2f}s): {e}")
        return False, duration, "error"

def main():
    """Test all upgraded AI models"""
    print("🧪 NIS Protocol v3.2 - Upgraded Models Test Suite")
    print("Testing latest premium AI models with your credits")
    print("=" * 70)
    
    # Check server health first
    try:
        health = requests.get("http://localhost:8000/health", timeout=10)
        if health.status_code == 200:
            health_data = health.json()
            providers = health_data.get("provider", [])
            print(f"✅ Server healthy - Available: {providers}")
        else:
            print(f"❌ Server not healthy - Status: {health.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False
    
    print(f"\n🧪 Testing Upgraded Premium Models")
    print("=" * 70)
    
    # Test configuration: (provider, expected_model, display_name)
    test_cases = [
        ("openai", "gpt-4o", "OpenAI GPT-4o"),
        ("anthropic", "claude-sonnet-4-20250514", "Claude 4 Sonnet"),
        ("google", "gemini-2.5-flash", "Google Gemini 2.5 Flash"),
        ("deepseek", "deepseek-chat", "DeepSeek R1"),
    ]
    
    results = []
    total_passed = 0
    
    for provider, expected_model, display_name in test_cases:
        success, duration, actual_model = test_provider(provider, expected_model, display_name)
        results.append({
            "provider": provider,
            "display_name": display_name,
            "expected": expected_model,
            "actual": actual_model,
            "success": success,
            "duration": duration
        })
        if success:
            total_passed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 UPGRADED MODELS TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Tests Passed: {total_passed}/{len(test_cases)}")
    
    if total_passed == len(test_cases):
        print("\n🎉 ALL PREMIUM MODELS WORKING PERFECTLY!")
        print("✨ Your credits are being put to excellent use:")
        for result in results:
            if result["success"]:
                print(f"   • 🧠 {result['display_name']}: {result['actual']}")
        
        print(f"\n💡 Average Response Time: {sum(r['duration'] for r in results if r['success']) / total_passed:.2f}s")
        print("🚀 Ready for high-performance AI tasks!")
        
    else:
        print(f"\n⚠️ Some models need attention:")
        for result in results:
            if not result["success"]:
                print(f"   • ❌ {result['display_name']}: {result['actual']}")
    
    return total_passed == len(test_cases)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
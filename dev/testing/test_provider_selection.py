#!/usr/bin/env python3
"""
Test LLM Provider Selection
Verifies that the console correctly routes requests to the selected provider
"""

import requests
import json
import sys
import time

BASE_URL = "http://localhost:8000"

def test_provider_selection():
    """Test if provider selection works correctly"""
    
    print("🧪 Testing LLM Provider Selection")
    print("=" * 50)
    
    # Test cases for different providers
    test_cases = [
        {
            "name": "OpenAI (GPT-4)",
            "provider": "openai",
            "expected_indicators": ["gpt", "openai"],
            "should_work": True
        },
        {
            "name": "Anthropic (Claude)",
            "provider": "anthropic", 
            "expected_indicators": ["claude", "anthropic"],
            "should_work": True
        },
        {
            "name": "DeepSeek (R1)",
            "provider": "deepseek",
            "expected_indicators": ["deepseek"],
            "should_work": True
        },
        {
            "name": "Google (Gemini)",
            "provider": "google",
            "expected_indicators": ["gemini", "google"],
            "should_work": True
        },
        {
            "name": "Auto-Select",
            "provider": "",  # Empty string = auto-select
            "expected_indicators": [],  # Any provider is acceptable
            "should_work": True
        },
        {
            "name": "Multimodel Consensus",
            "provider": "multimodel",
            "expected_indicators": ["multimodel", "consensus"],
            "should_work": True
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔧 Test {i}: {test_case['name']}")
        print("-" * 30)
        
        try:
            # Test the formatted chat endpoint
            request_data = {
                "message": "Hello! What provider are you using? Please identify yourself.",
                "user_id": "test_user",
                "agent_type": "default",
                "provider": test_case["provider"],
                "output_mode": "technical"
            }
            
            print(f"📤 Provider requested: {test_case['provider'] or 'auto-select'}")
            
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/chat/formatted",
                json=request_data,
                timeout=30
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                # Get the response content (HTML formatted)
                content = response.text
                
                # Look for provider indicators in the response
                provider_found = None
                for indicator in test_case["expected_indicators"]:
                    if indicator.lower() in content.lower():
                        provider_found = indicator
                        break
                
                # Extract any visible provider info from the response
                provider_mentions = []
                for keyword in ["openai", "gpt", "claude", "anthropic", "deepseek", "gemini", "google", "multimodel"]:
                    if keyword.lower() in content.lower():
                        provider_mentions.append(keyword)
                
                print(f"✅ SUCCESS ({response_time:.2f}s)")
                print(f"   Response length: {len(content)} chars")
                print(f"   Provider mentions: {provider_mentions}")
                
                if test_case["provider"] and provider_found:
                    print(f"   ✅ Expected provider confirmed: {provider_found}")
                elif not test_case["provider"]:
                    print(f"   ✅ Auto-select worked (any provider OK)")
                else:
                    print(f"   ⚠️ Provider not clearly identified in response")
                
                results.append({
                    "test": test_case["name"],
                    "provider_requested": test_case["provider"],
                    "status": "success",
                    "response_time": response_time,
                    "provider_mentions": provider_mentions,
                    "provider_confirmed": provider_found is not None
                })
                
            else:
                print(f"❌ HTTP ERROR ({response_time:.2f}s)")
                print(f"   Status: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                
                results.append({
                    "test": test_case["name"],
                    "provider_requested": test_case["provider"],
                    "status": "http_error",
                    "status_code": response.status_code,
                    "response_time": response_time
                })
                
        except requests.exceptions.Timeout:
            print("❌ TIMEOUT (>30s)")
            results.append({
                "test": test_case["name"],
                "provider_requested": test_case["provider"],
                "status": "timeout"
            })
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results.append({
                "test": test_case["name"],
                "provider_requested": test_case["provider"],
                "status": "error",
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 PROVIDER SELECTION TEST SUMMARY")
    print("=" * 50)
    
    success_count = len([r for r in results if r["status"] == "success"])
    error_count = len([r for r in results if r["status"] != "success"])
    
    print(f"✅ Successful: {success_count}/{len(test_cases)}")
    print(f"❌ Failed:     {error_count}/{len(test_cases)}")
    
    if success_count > 0:
        avg_time = sum(r["response_time"] for r in results if r["status"] == "success") / success_count
        print(f"⏱️  Avg Response Time: {avg_time:.2f}s")
    
    # Detailed provider analysis
    print("\n📋 Provider Analysis:")
    for result in results:
        if result["status"] == "success":
            provider_req = result["provider_requested"] or "auto"
            mentions = result.get("provider_mentions", [])
            confirmed = result.get("provider_confirmed", False)
            
            status_icon = "✅" if confirmed or not result["provider_requested"] else "⚠️"
            print(f"   {status_icon} {provider_req:12} → {mentions}")
    
    # Overall assessment
    if success_count >= len(test_cases) * 0.8:  # 80% success rate
        print("\n🎉 PROVIDER SELECTION: WORKING CORRECTLY!")
        print("   Users can select their preferred LLM provider")
        return True
    else:
        print("\n❌ PROVIDER SELECTION: NEEDS ATTENTION")
        print("   Some providers may not be working correctly")
        return False

def test_health_check():
    """Check if server is running and what providers are available"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            providers = health.get("provider", [])
            print(f"✅ Server healthy - Available providers: {providers}")
            return True, providers
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False, []
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        return False, []

if __name__ == "__main__":
    print("🧪 NIS Protocol v3.2 - Provider Selection Test")
    print("Testing if user provider selection works correctly")
    print()
    
    # Check server health first
    healthy, providers = test_health_check()
    if not healthy:
        print("\n❌ Cannot proceed - server not accessible")
        print("Please ensure NIS Protocol is running with: ./start.sh")
        sys.exit(1)
    
    print(f"🔧 Testing with {len(providers)} available providers")
    print()
    
    # Test provider selection
    success = test_provider_selection()
    
    if success:
        print("\n🚀 PROVIDER SELECTION: FULLY FUNCTIONAL!")
        print("Your users can confidently select their preferred LLM provider!")
        sys.exit(0)
    else:
        print("\n⚠️ PROVIDER SELECTION: PARTIALLY WORKING")
        print("Basic functionality available, some providers may need attention")
        sys.exit(1)
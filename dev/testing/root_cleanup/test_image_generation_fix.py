#!/usr/bin/env python3
"""
Test script to verify image generation fixes
"""

import requests
import json
import sys
import time

def test_image_generation():
    print("🔧 Testing Image Generation Fixes...")
    
    # Test cases
    test_cases = [
        {
            "name": "OpenAI - Dragon (Artistic)",
            "data": {
                "prompt": "dragon",
                "style": "artistic", 
                "provider": "openai",
                "size": "1024x1024"
            }
        },
        {
            "name": "Google - Dragon (Artistic)",
            "data": {
                "prompt": "dragon",
                "style": "artistic",
                "provider": "google", 
                "size": "1024x1024"
            }
        },
        {
            "name": "OpenAI - Neural Network (Technical)",
            "data": {
                "prompt": "neural network diagram",
                "style": "technical",
                "provider": "openai",
                "size": "1024x1024"
            }
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n🎨 Testing: {test_case['name']}")
        
        try:
            start_time = time.time()
            
            # Make request
            response = requests.post(
                "http://localhost:8000/image/generate",
                json=test_case["data"],
                timeout=30
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"   Status Code: {response.status_code}")
            print(f"   Duration: {duration:.2f}s")
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "unknown")
                provider_used = data.get("provider_used", "unknown")
                num_images = data.get("num_images", 0)
                
                print(f"   ✅ Success!")
                print(f"   Provider Used: {provider_used}")
                print(f"   Images Generated: {num_images}")
                print(f"   Real AI: {'✅' if 'mock' not in provider_used.lower() else '❌'}")
                
                # Check if enhanced prompt is appropriate
                enhanced_prompt = data.get("enhanced_prompt", "")
                if test_case["data"]["style"] == "artistic":
                    has_physics = any(term in enhanced_prompt.lower() for term in ['physics', 'conservation', 'equations'])
                    print(f"   Artistic Intent Preserved: {'✅' if not has_physics else '❌'}")
                
                results.append({
                    "test": test_case["name"],
                    "success": True,
                    "provider_used": provider_used,
                    "duration": duration,
                    "real_ai": 'mock' not in provider_used.lower()
                })
                
            else:
                print(f"   ❌ Failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"   Error: {response.text}")
                
                results.append({
                    "test": test_case["name"],
                    "success": False,
                    "error": response.status_code
                })
                
        except requests.exceptions.Timeout:
            print(f"   ❌ Timeout after 30s")
            results.append({
                "test": test_case["name"],
                "success": False,
                "error": "timeout"
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                "test": test_case["name"],
                "success": False,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    successful_tests = sum(1 for r in results if r.get("success", False))
    total_tests = len(results)
    real_ai_tests = sum(1 for r in results if r.get("real_ai", False))
    
    print(f"✅ Successful Tests: {successful_tests}/{total_tests}")
    print(f"🤖 Real AI Tests: {real_ai_tests}/{total_tests}")
    print(f"📈 Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    
    if successful_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Image generation is working!")
        return True
    else:
        print(f"\n⚠️  {total_tests - successful_tests} tests failed. Check the issues above.")
        return False

if __name__ == "__main__":
    success = test_image_generation()
    sys.exit(0 if success else 1)
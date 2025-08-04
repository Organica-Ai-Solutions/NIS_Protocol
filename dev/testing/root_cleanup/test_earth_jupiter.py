#!/usr/bin/env python3
"""
Test Earth from Jupiter image generation to confirm space content works
"""

import requests
import json

print("🌍 EARTH FROM JUPITER TEST")
print("="*30)

BASE_URL = "http://localhost:8000"

def test_earth_jupiter():
    """Test earth from jupiter image generation"""
    
    print("\n🌍 Testing Earth from Jupiter Image")
    print("-" * 30)
    
    try:
        response = requests.post(f"{BASE_URL}/image/generate", json={
            "prompt": "photo of earth from jupiter",
            "style": "artistic",
            "provider": "google"
        }, timeout=15)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            generation = result.get("generation", {})
            
            print(f"Generation Status: {generation.get('status', 'unknown')}")
            print(f"Provider Used: {generation.get('provider_used', 'unknown')}")
            
            if generation.get("images"):
                revised_prompt = generation["images"][0].get("revised_prompt", "")
                print(f"Revised Prompt: {revised_prompt}")
                
                # Check results
                is_creative = "Creative" in revised_prompt or "artistic" in revised_prompt
                no_physics = "Physics Visualization" not in revised_prompt
                
                print(f"✅ Creative Enhancement: {'YES' if is_creative else 'NO'}")
                print(f"✅ No Physics Over-Enhancement: {'YES' if no_physics else 'NO'}")
                
                if is_creative and no_physics:
                    print("🎉 SUCCESS: Earth from Jupiter image generation working correctly!")
                    return True
                else:
                    print("⚠️ ISSUE: Still has problems with enhancement")
                    return False
            else:
                print("❌ No images generated")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    success = test_earth_jupiter()
    
    print(f"\n" + "="*30)
    print(f"FINAL RESULT: {'🎉 SUCCESS' if success else '❌ FAILURE'}")

if __name__ == "__main__":
    main()
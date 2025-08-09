#!/usr/bin/env python3
"""
Final test of all critical fixes
"""

import requests
import json

print("🔧 FINAL CRITICAL FIXES TEST")
print("="*40)

BASE_URL = "http://localhost:8000"

def test_response_formatter():
    """Test response formatter is now working"""
    
    print("\n1. 🛠️ Response Formatter Fix")
    print("-" * 30)
    
    try:
        response = requests.post(f"{BASE_URL}/chat/formatted", json={
            "message": "Test the response formatter after fixes",
            "user_id": "final_test",
            "output_mode": "eli5",
            "audience_level": "beginner"
        }, timeout=30)
        
        success = response.status_code == 200 and "response_formatter' is not defined" not in response.text
        
        print(f"   Status Code: {response.status_code}")
        print(f"   Contains Error: {'❌ YES' if 'response_formatter' in response.text else '✅ NO'}")
        print(f"   Fix Status: {'✅ FIXED' if success else '❌ BROKEN'}")
        
        if success:
            print(f"   Content Preview: {response.text[:100]}...")
        
        return success
        
    except Exception as e:
        print(f"   Exception: {e}")
        return False

def test_image_generation():
    """Test image generation with dragon prompt"""
    
    print(f"\n2. 🎨 Image Generation Fix")
    print("-" * 30)
    
    try:
        response = requests.post(f"{BASE_URL}/image/generate", json={
            "prompt": "A cute dragon playing in clouds",
            "style": "artistic",
            "provider": "openai"  # Test OpenAI specifically since that had the error
        }, timeout=25)
        
        if response.status_code == 200:
            result = response.json()
            generation = result.get("generation", {})
            status = generation.get("status", "unknown")
            
            print(f"   Status Code: {response.status_code}")
            print(f"   Generation Status: {status}")
            
            if status == "success":
                images = generation.get("images", [])
                if images:
                    enhanced_prompt = images[0].get("revised_prompt", "")
                    
                    # Check if prompt enhancement is appropriate (not over-physics)
                    is_over_enhanced = any(phrase in enhanced_prompt for phrase in [
                        "NIS PHYSICS COMPLIANT", "conservation laws", "Physics Visualization"
                    ])
                    
                    is_artistic = "artistic" in enhanced_prompt.lower()
                    
                    print(f"   Images Generated: {len(images)}")
                    print(f"   Over-Enhanced: {'❌ YES' if is_over_enhanced else '✅ NO'}")
                    print(f"   Artistic Style: {'✅ YES' if is_artistic else '❌ NO'}")
                    print(f"   Enhanced Prompt: {enhanced_prompt[:70]}...")
                    
                    return status == "success" and not is_over_enhanced
            else:
                print(f"   Generation Failed: {status}")
                return False
        else:
            print(f"   HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   Exception: {e}")
        return False

def test_basic_chat():
    """Test basic chat functionality"""
    
    print(f"\n3. 💬 Basic Chat Test")
    print("-" * 30)
    
    try:
        response = requests.post(f"{BASE_URL}/chat", json={
            "message": "Tell me about dragons in a fun way",
            "user_id": "final_test",
            "output_mode": "eli5",
            "include_visuals": False
        }, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("response", "")
            
            print(f"   Status Code: {response.status_code}")
            print(f"   Response Length: {len(content)} chars")
            print(f"   Provider: {result.get('provider', 'unknown')}")
            print(f"   Content Preview: {content[:80]}...")
            
            return True
        else:
            print(f"   HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   Exception: {e}")
        return False

def main():
    """Run final tests and generate report"""
    
    print("Running final verification of critical fixes...")
    
    # Test all fixes
    formatter_fixed = test_response_formatter()
    image_fixed = test_image_generation() 
    chat_fixed = test_basic_chat()
    
    # Generate final report
    print(f"\n" + "="*40)
    print("📊 FINAL FIXES REPORT")
    print("="*40)
    
    print(f"\n🛠️ CRITICAL FIXES STATUS:")
    print(f"   1. Response Formatter: {'✅ FIXED' if formatter_fixed else '❌ BROKEN'}")
    print(f"   2. Image Generation: {'✅ FIXED' if image_fixed else '❌ BROKEN'}")
    print(f"   3. Basic Chat: {'✅ WORKING' if chat_fixed else '❌ BROKEN'}")
    
    fixed_count = sum([formatter_fixed, image_fixed, chat_fixed])
    
    print(f"\n📈 OVERALL STATUS:")
    print(f"   Working: {fixed_count}/3 ({fixed_count/3*100:.0f}%)")
    
    if fixed_count == 3:
        print(f"   Result: 🎉 ALL CRITICAL ISSUES RESOLVED!")
        print(f"   Status: ✅ System ready for user testing")
    elif fixed_count >= 2:
        print(f"   Result: 🟡 Most issues resolved")
        print(f"   Status: ⚡ System mostly functional")
    else:
        print(f"   Result: ⚠️ Major issues remain")
        print(f"   Status: 🔧 Needs more work")
    
    if formatter_fixed and image_fixed:
        print(f"\n🎨 SUCCESS: Users can now properly generate artistic images!")
        print(f"🧒 SUCCESS: ELI5 mode and visual modes should work!")
        
    return fixed_count == 3

if __name__ == "__main__":
    main()
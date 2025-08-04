#!/usr/bin/env python3
"""
Test the image generation fixes to ensure artistic content is handled properly
"""

import requests
import json

print("🎨 TESTING IMAGE GENERATION FIXES")
print("="*40)

BASE_URL = "http://localhost:8000"

def test_dragon_image():
    """Test dragon image generation (should be artistic, not physics)"""
    
    print("\n1. 🐉 Testing Dragon Image Generation")
    print("-" * 35)
    
    try:
        response = requests.post(f"{BASE_URL}/image/generate", json={
            "prompt": "A majestic dragon soaring through clouds",
            "style": "artistic",
            "provider": "google"
        }, timeout=25)
        
        if response.status_code == 200:
            result = response.json()
            generation = result.get("generation", {})
            
            if generation.get("status") == "success":
                images = generation.get("images", [])
                if images:
                    revised_prompt = images[0].get("revised_prompt", "")
                    
                    print(f"   ✅ Generation successful")
                    print(f"   📝 Revised prompt: {revised_prompt}")
                    
                    # Check for physics over-enhancement
                    physics_terms = ["Physics Visualization", "conservation laws", "E=mc²", "NIS PHYSICS COMPLIANT"]
                    has_physics = any(term in revised_prompt for term in physics_terms)
                    
                    # Check for artistic enhancement
                    artistic_terms = ["Creative", "artistic", "creative composition"]
                    has_artistic = any(term in revised_prompt for term in artistic_terms)
                    
                    print(f"   🔬 Physics Enhancement: {'❌ FOUND (BAD)' if has_physics else '✅ NONE (GOOD)'}")
                    print(f"   🎨 Artistic Enhancement: {'✅ FOUND (GOOD)' if has_artistic else '❌ NONE (BAD)'}")
                    
                    return not has_physics and has_artistic
                    
            else:
                print(f"   ❌ Generation failed: {generation.get('status', 'unknown')}")
                return False
        else:
            print(f"   ❌ HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def test_earth_jupiter_image():
    """Test earth from jupiter image (should be artistic space, not physics)"""
    
    print(f"\n2. 🌍 Testing Earth from Jupiter Image")
    print("-" * 35)
    
    try:
        response = requests.post(f"{BASE_URL}/image/generate", json={
            "prompt": "photo of earth from jupiter",
            "style": "artistic",
            "provider": "google"
        }, timeout=25)
        
        if response.status_code == 200:
            result = response.json()
            generation = result.get("generation", {})
            
            if generation.get("status") == "success":
                images = generation.get("images", [])
                if images:
                    revised_prompt = images[0].get("revised_prompt", "")
                    
                    print(f"   ✅ Generation successful")
                    print(f"   📝 Revised prompt: {revised_prompt}")
                    
                    # Check for physics over-enhancement
                    physics_terms = ["Physics Visualization", "conservation laws", "∇²φ=0"]
                    has_physics = any(term in revised_prompt for term in physics_terms)
                    
                    # Check for creative enhancement
                    creative_terms = ["Creative", "artistic", "creative composition", "cosmic"]
                    has_creative = any(term in revised_prompt for term in creative_terms)
                    
                    print(f"   🔬 Physics Enhancement: {'❌ FOUND (BAD)' if has_physics else '✅ NONE (GOOD)'}")
                    print(f"   🌌 Creative Enhancement: {'✅ FOUND (GOOD)' if has_creative else '❌ NONE (BAD)'}")
                    
                    return not has_physics and has_creative
                    
            else:
                print(f"   ❌ Generation failed: {generation.get('status', 'unknown')}")
                return False
        else:
            print(f"   ❌ HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def test_technical_image():
    """Test technical image (should have physics enhancement)"""
    
    print(f"\n3. 🔬 Testing Technical Diagram (Should Have Physics)")
    print("-" * 45)
    
    try:
        response = requests.post(f"{BASE_URL}/image/generate", json={
            "prompt": "technical diagram of neural network architecture",
            "style": "scientific",
            "provider": "google"
        }, timeout=25)
        
        if response.status_code == 200:
            result = response.json()
            generation = result.get("generation", {})
            
            if generation.get("status") == "success":
                images = generation.get("images", [])
                if images:
                    revised_prompt = images[0].get("revised_prompt", "")
                    
                    print(f"   ✅ Generation successful")
                    print(f"   📝 Revised prompt: {revised_prompt}")
                    
                    # Check for appropriate technical enhancement
                    technical_terms = ["Physics", "technical", "scientific", "network topology"]
                    has_technical = any(term in revised_prompt for term in technical_terms)
                    
                    print(f"   🔬 Technical Enhancement: {'✅ FOUND (GOOD)' if has_technical else '❌ NONE (BAD)'}")
                    
                    return has_technical
                    
            else:
                print(f"   ❌ Generation failed: {generation.get('status', 'unknown')}")
                return False
        else:
            print(f"   ❌ HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def main():
    """Run all image generation tests"""
    
    print("Testing image generation fixes...")
    
    # Test all scenarios
    dragon_fixed = test_dragon_image()
    space_fixed = test_earth_jupiter_image()
    technical_works = test_technical_image()
    
    # Generate report
    print(f"\n" + "="*40)
    print("📊 IMAGE GENERATION FIXES REPORT")
    print("="*40)
    
    print(f"\n🎨 ARTISTIC CONTENT:")
    print(f"   Dragon Image: {'✅ FIXED' if dragon_fixed else '❌ STILL BROKEN'}")
    print(f"   Space Image: {'✅ FIXED' if space_fixed else '❌ STILL BROKEN'}")
    
    print(f"\n🔬 TECHNICAL CONTENT:")
    print(f"   Neural Network: {'✅ WORKING' if technical_works else '❌ BROKEN'}")
    
    fixed_count = sum([dragon_fixed, space_fixed, technical_works])
    
    print(f"\n📈 OVERALL STATUS:")
    print(f"   Working: {fixed_count}/3 ({fixed_count/3*100:.0f}%)")
    
    if fixed_count == 3:
        print(f"   Result: 🎉 ALL IMAGE GENERATION ISSUES FIXED!")
        print(f"   Status: ✅ Artistic content preserved, technical content enhanced")
    elif fixed_count >= 2:
        print(f"   Result: 🟡 Most issues resolved")
        print(f"   Status: ⚡ Mostly working correctly")
    else:
        print(f"   Result: ⚠️ Issues remain")
        print(f"   Status: 🔧 Needs more work")
    
    return fixed_count == 3

if __name__ == "__main__":
    main()
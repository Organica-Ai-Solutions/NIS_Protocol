#!/usr/bin/env python3
"""
Quick test to verify visual mode fix
"""

import requests
import json

print("🔧 TESTING VISUAL MODE FIX")
print("="*40)

BASE_URL = "http://localhost:8000"

def test_visual_mode_quick():
    """Quick test of visual mode after fix"""
    
    print("\n🎨 Testing Visual Mode:")
    
    try:
        response = requests.post(f"{BASE_URL}/chat", json={
            "message": "Explain neural networks",
            "user_id": "fix_test",
            "output_mode": "visual", 
            "audience_level": "intermediate",
            "include_visuals": True,
            "show_confidence": False
        }, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("response", "")
            
            print(f"✅ Status: 200")
            print(f"📝 Length: {len(content)} chars")
            
            # Check for visual formatting
            visual_indicators = [
                "Visual Summary",
                "Generated Visuals:", 
                "🎨",
                "📊",
                "diagram",
                "Generated successfully"
            ]
            
            found = []
            for indicator in visual_indicators:
                if indicator in content:
                    found.append(indicator)
            
            print(f"🎯 Visual indicators found: {len(found)}/{len(visual_indicators)}")
            if found:
                print(f"   Found: {', '.join(found)}")
            
            # Show key lines
            lines = content.split('\n')
            visual_lines = [line for line in lines if any(word in line for word in ['Visual', 'Generated', '🎨', '📊', 'diagram'])]
            
            if visual_lines:
                print(f"📄 Visual content lines:")
                for line in visual_lines[:5]:
                    print(f"   {line.strip()}")
            else:
                print(f"📄 First 200 chars: {content[:200]}...")
            
            if len(found) >= 2:
                print(f"✅ VISUAL MODE IS NOW WORKING!")
            else:
                print(f"❌ Visual mode still not working properly")
                
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_eli5_mode_quick():
    """Quick test of ELI5 mode for comparison"""
    
    print(f"\n🌟 Testing ELI5 Mode:")
    
    try:
        response = requests.post(f"{BASE_URL}/chat", json={
            "message": "What is AI?",
            "user_id": "fix_test",
            "output_mode": "eli5",
            "audience_level": "beginner",
            "include_visuals": False,
            "show_confidence": False
        }, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("response", "")
            
            eli5_words = ["simple", "like", "basically", "think of", "brain", "computer"]
            found = [w for w in eli5_words if w in content.lower()]
            
            print(f"✅ ELI5 words found: {found}")
            print(f"📄 Preview: {content[:150]}...")
            
        else:
            print(f"❌ ELI5 Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ ELI5 Exception: {e}")

if __name__ == "__main__":
    test_visual_mode_quick()
    test_eli5_mode_quick()
    print(f"\n🎯 Quick fix test completed!")
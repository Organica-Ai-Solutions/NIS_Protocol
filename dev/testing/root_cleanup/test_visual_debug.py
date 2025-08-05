#!/usr/bin/env python3
"""
Debug test for visual mode to identify the exact issue
"""

import requests
import json
import time

print("🔍 VISUAL MODE DEBUG TEST")
print("="*50)

BASE_URL = "http://localhost:8000"

def test_different_modes():
    """Test different output modes to see which formatting is working"""
    
    modes = [
        ("technical", "expert", False, False),
        ("eli5", "beginner", False, False), 
        ("casual", "intermediate", False, False),
        ("visual", "intermediate", True, False),
        ("visual", "intermediate", False, False),  # Visual without include_visuals
    ]
    
    message = "What is a neural network?"
    
    for i, (mode, level, visuals, confidence) in enumerate(modes, 1):
        print(f"\n{i}. Testing: {mode} mode, {level} level, visuals={visuals}, confidence={confidence}")
        
        try:
            response = requests.post(f"{BASE_URL}/chat", json={
                "message": message,
                "user_id": "debug_test",
                "output_mode": mode,
                "audience_level": level,
                "include_visuals": visuals,
                "show_confidence": confidence
            }, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("response", "")
                
                print(f"   ✅ Success ({len(content)} chars)")
                
                # Check for mode-specific formatting
                if mode == "eli5":
                    eli5_words = ["simple", "like", "basically", "think of"]
                    found = [w for w in eli5_words if w in content.lower()]
                    print(f"   🌟 ELI5 words: {found}")
                
                elif mode == "casual":
                    casual_words = ["here's", "basically", "so", "deal"]
                    found = [w for w in casual_words if w in content.lower()]
                    print(f"   💬 Casual words: {found}")
                
                elif mode == "visual":
                    visual_words = ["visual", "diagram", "chart", "generated", "image", "🎨", "📊"]
                    found = [w for w in visual_words if w in content.lower()]
                    print(f"   🎨 Visual words: {found}")
                    
                    # Check for specific visual indicators
                    if "Visual Summary" in content:
                        print(f"   ✅ Has Visual Summary")
                    if "Generated Visuals" in content:
                        print(f"   ✅ Has Generated Visuals section")
                    if "Generated successfully" in content:
                        print(f"   ✅ Images generated successfully")
                
                # Show first 150 chars
                print(f"   📄 Preview: {content[:150]}...")
                
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")

def test_formatted_endpoint():
    """Test the formatted endpoint with visual mode"""
    
    print(f"\n📝 TESTING FORMATTED ENDPOINT WITH VISUAL MODE")
    print("-" * 40)
    
    try:
        response = requests.post(f"{BASE_URL}/chat/formatted", json={
            "message": "Show me how neural networks work",
            "user_id": "debug_test",
            "output_mode": "visual",
            "audience_level": "intermediate",
            "include_visuals": True,
            "show_confidence": False
        }, timeout=60)
        
        if response.status_code == 200:
            content = response.text
            print(f"✅ Formatted endpoint success ({len(content)} chars)")
            
            # Check for HTML and visual elements
            has_html = any(tag in content for tag in ["<div", "<style", "background:"])
            has_visual = any(word in content for word in ["visual", "Visual", "diagram", "image"])
            
            print(f"🖼️ Has HTML: {has_html}")
            print(f"🎨 Has Visual Content: {has_visual}")
            
            # Show relevant lines
            lines = content.split('\n')
            visual_lines = [line for line in lines if any(word in line.lower() for word in ['visual', 'image', 'diagram', 'generated'])]
            
            if visual_lines:
                print(f"📄 Visual Lines Found:")
                for line in visual_lines[:3]:
                    print(f"   {line.strip()}")
            
        else:
            print(f"❌ Formatted endpoint error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Formatted endpoint exception: {e}")

def test_direct_formatter():
    """Test the response formatter directly"""
    
    print(f"\n🔧 TESTING RESPONSE FORMATTER DIRECTLY")
    print("-" * 40)
    
    # Simulate what the chat endpoint should do
    test_data = {
        "content": "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes that process information.",
        "confidence": 0.85,
        "provider": "test",
        "model": "test-model"
    }
    
    try:
        # Import and test the formatter directly
        import sys
        sys.path.insert(0, './src')
        from src.utils.response_formatter import NISResponseFormatter
        
        formatter = NISResponseFormatter()
        
        # Test visual mode
        result = formatter.format_response(
            data=test_data,
            output_mode="visual",
            audience_level="intermediate", 
            include_visuals=True,
            show_confidence=False
        )
        
        print(f"✅ Direct formatter test successful")
        print(f"📊 Result keys: {list(result.keys())}")
        
        formatted_content = result.get("formatted_content", "")
        print(f"📝 Formatted content length: {len(formatted_content)}")
        
        # Check for visual elements
        if "Visual Summary" in formatted_content:
            print(f"✅ Contains Visual Summary")
        if "Generated Visuals" in formatted_content:
            print(f"✅ Contains Generated Visuals")
        if "visual" in formatted_content.lower():
            print(f"✅ Contains visual-related content")
        
        # Show preview
        print(f"📄 Preview: {formatted_content[:200]}...")
        
    except Exception as e:
        print(f"❌ Direct formatter exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting visual mode debug test...")
    
    # Test different output modes
    test_different_modes()
    
    # Test formatted endpoint
    test_formatted_endpoint()
    
    # Test formatter directly
    test_direct_formatter()
    
    print(f"\n🎯 Debug test completed!")
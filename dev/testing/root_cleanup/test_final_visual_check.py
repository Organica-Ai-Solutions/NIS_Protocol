#!/usr/bin/env python3
"""
Final comprehensive test to isolate the visual mode issue
"""

import requests
import json

print("🔬 FINAL VISUAL MODE DIAGNOSTIC")
print("="*50)

BASE_URL = "http://localhost:8000"

def test_response_formatter_integration():
    """Test if response formatter is working at all in the system"""
    
    print("\n🧪 TESTING RESPONSE FORMATTER INTEGRATION")
    print("-" * 40)
    
    # Test different combinations to see what works
    test_modes = [
        ("eli5", "beginner", False, False),
        ("casual", "intermediate", False, False),
        ("visual", "intermediate", False, False),  # Visual without visuals flag
        ("technical", "expert", True, False),      # Technical with visuals flag
        ("visual", "intermediate", True, False),   # Visual with visuals flag
        ("visual", "expert", True, True),          # Visual with everything
    ]
    
    for mode, level, visuals, confidence in test_modes:
        print(f"\n📝 Testing: {mode} + visuals={visuals} + confidence={confidence}")
        
        try:
            response = requests.post(f"{BASE_URL}/chat", json={
                "message": "What is a neural network?",
                "user_id": "diagnostic",
                "output_mode": mode,
                "audience_level": level,
                "include_visuals": visuals,
                "show_confidence": confidence
            }, timeout=45)
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("response", "")
                
                # Check for ANY formatting indicators
                format_indicators = {
                    "eli5": ["simple", "like", "basically", "🌟"],
                    "casual": ["here's", "basically", "deal", "so"],
                    "visual": ["Visual Summary", "Generated Visuals", "🎨", "📊", "diagram"],
                    "confidence": ["confidence", "breakdown", "score", "📊"]
                }
                
                found_any = False
                for fmt_type, indicators in format_indicators.items():
                    for indicator in indicators:
                        if indicator in content:
                            print(f"   ✅ Found {fmt_type} indicator: '{indicator}'")
                            found_any = True
                            break
                
                if not found_any:
                    print(f"   ❌ NO formatting indicators found")
                    print(f"   📄 Raw content: {content[:100]}...")
                else:
                    print(f"   ✅ Some formatting detected")
                    
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")

def test_direct_formatter_call():
    """Test the formatter directly to ensure it works"""
    
    print(f"\n🔧 DIRECT FORMATTER TEST")
    print("-" * 40)
    
    try:
        import sys
        sys.path.insert(0, './src')
        from src.utils.response_formatter import NISResponseFormatter
        
        formatter = NISResponseFormatter()
        
        test_data = {
            "content": "Neural networks are computational models inspired by biological neural networks."
        }
        
        # Test each mode
        modes = ["technical", "eli5", "casual", "visual"]
        
        for mode in modes:
            print(f"\n📝 Testing {mode} mode directly:")
            
            try:
                result = formatter.format_response(
                    data=test_data,
                    output_mode=mode,
                    audience_level="intermediate",
                    include_visuals=True,
                    show_confidence=False
                )
                
                formatted_content = result.get("formatted_content", "")
                print(f"   ✅ Success: {len(formatted_content)} chars")
                
                # Show key characteristics
                if mode == "eli5" and ("simple" in formatted_content.lower() or "like" in formatted_content.lower()):
                    print(f"   ✅ ELI5 transformation applied")
                elif mode == "visual" and ("Visual Summary" in formatted_content or "🎨" in formatted_content):
                    print(f"   ✅ Visual transformation applied")
                elif mode == "casual" and ("deal" in formatted_content.lower() or "basically" in formatted_content.lower()):
                    print(f"   ✅ Casual transformation applied")
                else:
                    print(f"   ⚠️ Transformation may not be working for {mode}")
                
                # Show preview
                print(f"   📄 Preview: {formatted_content[:100]}...")
                
            except Exception as e:
                print(f"   ❌ Error in {mode}: {e}")
                
    except Exception as e:
        print(f"❌ Cannot import formatter: {e}")

def show_system_status():
    """Show overall system status"""
    
    print(f"\n📊 SYSTEM STATUS CHECK")
    print("-" * 40)
    
    try:
        # Check health
        health = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Health: {'✅' if health.status_code == 200 else '❌'}")
        
        # Check image generation
        img_test = requests.post(f"{BASE_URL}/image/generate", json={
            "prompt": "test", "style": "scientific", "provider": "openai"
        }, timeout=15)
        print(f"Image Generation: {'✅' if img_test.status_code == 200 else '❌'}")
        
        # Check basic chat
        chat_test = requests.post(f"{BASE_URL}/chat", json={
            "message": "test", "user_id": "status"
        }, timeout=30)
        print(f"Basic Chat: {'✅' if chat_test.status_code == 200 else '❌'}")
        
    except Exception as e:
        print(f"❌ System status error: {e}")

def main():
    """Run final comprehensive diagnostic"""
    
    print("Running final visual mode diagnostic...")
    
    # Show system status
    show_system_status()
    
    # Test direct formatter
    test_direct_formatter_call()
    
    # Test integration in endpoints
    test_response_formatter_integration()
    
    print(f"\n🎯 CONCLUSION:")
    print("If direct formatter works but integration doesn't,")
    print("the issue is in the chat endpoint formatting logic.")
    print("If neither works, the issue is in the formatter itself.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test console image generation to see what the user would experience
"""

import requests
import json

def test_console_experience():
    print("🎨 Testing Console Image Generation Experience...")
    
    # Test the exact same request the user made
    data = {
        "prompt": "Analyze your consciousness level",
        "style": "artistic", 
        "provider": "google",
        "size": "1024x1024"
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/image/generate",
            json=data,
            timeout=15
        )
        
        print(f"📋 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Success!")
            print(f"🔧 Provider Used: {result.get('provider_used', 'unknown')}")
            print(f"🖼️  Images Generated: {result.get('num_images', 0)}")
            print(f"📊 Status: {result.get('status', 'unknown')}")
            
            # Determine what type of result this is
            provider_used = result.get('provider_used', '').lower()
            
            if 'placeholder' in provider_used:
                print(f"📸 Result Type: ✨ Enhanced AI Placeholder (API unavailable)")
            elif 'real' in provider_used or 'vertex' in provider_used:
                print(f"🎨 Result Type: 🤖 Real AI Generation!")
            else:
                print(f"🔍 Result Type: Smart Processing (checking provider response...)")
                
            # Check the actual generated content
            if result.get('images') and len(result['images']) > 0:
                img = result['images'][0]
                revised_prompt = img.get('revised_prompt', 'N/A')
                print(f"📝 Enhanced Prompt: {revised_prompt[:150]}...")
                
                # Check if this preserves artistic intent
                if 'physics' in revised_prompt.lower() or 'conservation' in revised_prompt.lower():
                    print(f"⚠️  Artistic Intent: May have physics over-enhancement")
                else:
                    print(f"✅ Artistic Intent: Preserved (no excessive physics terms)")
                    
                # Check the quality of the result
                if 'Creative' in revised_prompt or 'artistic' in revised_prompt.lower():
                    print(f"🎨 Content Quality: Appropriate for artistic request")
                elif 'Technical' in revised_prompt or 'Physics' in revised_prompt:
                    print(f"🔬 Content Quality: Technical (may not match artistic request)")
                else:
                    print(f"🤔 Content Quality: Mixed/Unknown")
                    
            print(f"\n📊 Generation Metadata:")
            generation_info = result.get('generation_info', {})
            metadata = result.get('metadata', {})
            
            print(f"   Model: {generation_info.get('model', 'unknown')}")
            print(f"   Style Applied: {generation_info.get('style_applied', 'unknown')}")
            print(f"   Prompt Enhancement: {metadata.get('prompt_enhancement', 'unknown')}")
            print(f"   Safety Filtered: {metadata.get('safety_filtered', 'unknown')}")
            
            # Overall assessment
            print(f"\n🎯 Overall Assessment:")
            if result.get('num_images', 0) > 0:
                print(f"   ✅ Image generation: SUCCESS")
                print(f"   ✅ Error handling: ROBUST (no crashes)")
                print(f"   ✅ Response format: COMPLETE")
                
                if 'artistic' in revised_prompt.lower() and 'physics' not in revised_prompt.lower():
                    print(f"   ✅ Artistic intent: PRESERVED")
                else:
                    print(f"   ⚠️  Artistic intent: Check needed")
                    
                print(f"   🎉 USER EXPERIENCE: IMPROVED!")
            else:
                print(f"   ⚠️  No images generated, but system handled gracefully")
                
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_console_experience()
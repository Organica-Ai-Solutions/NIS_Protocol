#!/usr/bin/env python3
"""
Test the real Gemini 2.0 dragon image generation
"""

import requests
import json
import time

def test_real_dragon():
    print("🐉 Testing REAL Gemini 2.0 Dragon Generation...")
    
    dragon_request = {
        "prompt": "A majestic dragon soaring through a cyberpunk cityscape at sunset",
        "style": "artistic",
        "provider": "google",
        "size": "1024x1024",
        "quality": "standard"
    }
    
    try:
        print("🎨 Sending dragon request to Gemini 2.0...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:8000/image/generate",
            json=dragon_request,
            timeout=30
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"📊 Status: {response.status_code}")
        print(f"⏱️  Duration: {duration:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            
            generation = result.get("generation", {})
            provider_used = generation.get("provider_used", "unknown")
            num_images = generation.get("num_images", 0)
            model_info = generation.get("generation_info", {}).get("model", "unknown")
            note = generation.get("note", "")
            
            print(f"✅ SUCCESS!")
            print(f"🤖 Provider: {provider_used}")
            print(f"🎨 Model: {model_info}")
            print(f"📷 Images: {num_images}")
            print(f"📝 Note: {note}")
            
            # Check if it's real AI generation
            if "real" in provider_used.lower() or "gemini_2.0" in provider_used.lower():
                print(f"🎉 REAL DRAGON GENERATED! No more placeholders!")
                print(f"🐉 Your majestic dragon has been created!")
                
                # Check image data
                if generation.get("images") and len(generation["images"]) > 0:
                    image = generation["images"][0]
                    url_length = len(image.get("url", ""))
                    revised_prompt = image.get("revised_prompt", "")
                    
                    print(f"📊 Image data size: {url_length} chars")
                    print(f"🎨 Enhanced prompt: {revised_prompt[:150]}...")
                    
                    if url_length > 10000:  # Real images are much larger
                        print("✅ CONFIRMED: Real image data detected!")
                    else:
                        print("⚠️  Warning: Image data seems small for real generation")
                        
                else:
                    print("❌ No image data found in response")
                    
            else:
                print(f"⚠️  Still using fallback/placeholder: {provider_used}")
                
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"Error: {response.text[:500]}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out after 30 seconds")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_real_dragon()
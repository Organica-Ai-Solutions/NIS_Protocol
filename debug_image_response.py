#!/usr/bin/env python3
"""
Debug what's actually being returned by image generation
"""

import requests
import json

def debug_response():
    print("🔍 Debugging Image Generation Response...")
    
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
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Print the full response to see what we're getting
            print(f"\n📊 FULL RESPONSE:")
            print(json.dumps(result, indent=2)[:2000])
            print("..." if len(json.dumps(result, indent=2)) > 2000 else "")
            
            # Check both top-level and generation.images array
            images_top = result.get('images', [])
            generation = result.get('generation', {})
            images_gen = generation.get('images', [])
            
            print(f"\n🖼️  IMAGES ARRAYS:")
            print(f"   Top-level images: {len(images_top)}")
            print(f"   Generation images: {len(images_gen)}")
            
            # Use the generation images (correct location)
            images = images_gen if images_gen else images_top
            
            if images:
                for i, img in enumerate(images):
                    print(f"   Image {i+1}:")
                    url_length = len(img.get('url', ''))
                    print(f"     URL length: {url_length}")
                    print(f"     Format: {img.get('format', 'unknown')}")
                    print(f"     Revised prompt: {img.get('revised_prompt', 'N/A')[:100]}...")
                    
                    # Check if it's a real image or placeholder
                    if url_length > 100:
                        print(f"     ✅ Contains image data!")
                    else:
                        print(f"     ⚠️  Minimal/no image data")
            else:
                print("   ❌ No images found in either location")
                
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_response()
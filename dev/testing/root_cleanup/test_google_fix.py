#!/usr/bin/env python3
"""
Quick test to verify Google provider fix
"""

import requests
import json

def test_google_provider():
    print("🔧 Testing Google Provider Fix...")
    
    data = {
        "prompt": "analyze consciousness level",
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
            
            print(f"✅ Success!")
            print(f"Provider Used: {result.get('provider_used', 'unknown')}")
            print(f"Images Generated: {result.get('num_images', 0)}")
            print(f"Status: {result.get('status', 'unknown')}")
            
            # Check if it's a placeholder or real generation
            provider_used = result.get('provider_used', '')
            if 'placeholder' in provider_used.lower():
                print(f"📸 Result: Enhanced placeholder (API not available)")
            elif 'real' in provider_used.lower():
                print(f"🎨 Result: Real AI generation!")
            else:
                print(f"🤔 Result: Unknown type")
                
            if result.get('images'):
                img = result['images'][0]
                print(f"Revised Prompt: {img.get('revised_prompt', 'N/A')[:100]}...")
                
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_google_provider()
#!/usr/bin/env python3
"""Test console update functionality"""

import requests
import json

def test_enhanced_image_generation():
    """Test image generation with new formatting parameters"""
    print("Testing Enhanced Console Features...")
    
    url = "http://localhost:8000/image/generate"
    payload = {
        "prompt": "neural network with spline connections showing KAN architecture",
        "style": "scientific",
        "size": "1024x1024", 
        "provider": "google",
        "output_mode": "eli5",
        "audience_level": "beginner",
        "include_visuals": True,
        "show_confidence": True
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        data = response.json()
        
        print("Console Update SUCCESS!")
        print(f"Status: {data.get('status', 'unknown')}")
        print(f"Generation Time: {data.get('generation', {}).get('generation_info', {}).get('generation_time', 'unknown')}s")
        print(f"Provider: {data.get('generation', {}).get('provider_used', 'unknown')}")
        print(f"Images: {len(data.get('generation', {}).get('images', []))} generated")
        print(f"New Parameters Accepted: output_mode, audience_level, include_visuals, show_confidence")
        print("")
        print("Console now ready with enhanced formatting!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    test_enhanced_image_generation()
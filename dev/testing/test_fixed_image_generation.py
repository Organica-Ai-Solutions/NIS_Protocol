#!/usr/bin/env python3
"""
Test the fixed image generation endpoint
"""
import requests
import json
import base64

def test_image_generation():
    """Test the fixed image generation"""
    print("🧪 Testing Fixed Image Generation")
    print("=" * 50)
    
    # Test endpoint
    url = "http://localhost:8000/image/generate"
    
    # Test data
    test_data = {
        "prompt": "A beautiful sunset over mountains",
        "style": "photorealistic",
        "size": "512x512",
        "provider": "auto",
        "quality": "standard",
        "num_images": 1
    }
    
    try:
        print(f"🔍 Testing with prompt: '{test_data['prompt']}'")
        response = requests.post(url, json=test_data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Image generation working!")
            
            if result.get("generation", {}).get("images"):
                image_data = result["generation"]["images"][0]["url"]
                provider = result["generation"].get("provider_used", "unknown")
                
                print(f"✅ Provider used: {provider}")
                print(f"✅ Image data length: {len(image_data)} chars")
                
                # Save test result
                with open("static/test_generated_image.html", "w") as f:
                    f.write(f"""
<!DOCTYPE html>
<html>
<head><title>NIS Image Generation Test Result</title></head>
<body>
    <h1>🎉 Image Generation Fixed!</h1>
    <p><strong>Test prompt:</strong> {test_data['prompt']}</p>
    <p><strong>Provider:</strong> {provider}</p>
    <img src="{image_data}" alt="Generated image" style="max-width: 500px; border: 1px solid #ccc;">
    <p>✅ Real image generation is now working!</p>
</body>
</html>
""")
                
                print("✅ Test result saved to: static/test_generated_image.html")
                return True
            else:
                print("❌ No image data in response")
                return False
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("   Make sure the application is running on http://localhost:8000")
        return False

if __name__ == "__main__":
    test_image_generation()
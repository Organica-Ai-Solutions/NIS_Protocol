#!/usr/bin/env python3
import requests
import json
import base64
import time

print("=== TESTING GIF CREATION CAPABILITIES ===")

# Test specific GIF creation for consciousness visualization
test_data = {
    "prompt": "Create an animated GIF showing 4 stages of AI consciousness development: 1) Simple pattern recognition (basic neural network), 2) Complex learning (deeper network), 3) Self-modeling (recursive loops), 4) Full consciousness (glowing integrated system). Each frame should be clearly labeled.",
    "style": "scientific",
    "size": "512x512",
    "provider": "openai",
    "quality": "standard",
    "num_images": 4,
    "format": "gif",
    "animation": True,
    "duration": 2000  # 2 seconds per frame
}

print("Testing GIF Animation Generation:")
print(f"Prompt: {test_data['prompt'][:80]}...")

try:
    start_time = time.time()
    resp = requests.post("http://localhost:8000/image/generate", json=test_data, timeout=60)
    end_time = time.time()
    
    print(f"Status: {resp.status_code}")
    print(f"Response Time: {end_time - start_time:.2f}s")
    
    if resp.status_code == 200:
        result = resp.json()
        print("\n✅ GIF Generation Response Analysis:")
        print(f"Overall Status: {result.get('status')}")
        
        if "generation" in result:
            gen = result["generation"]
            print(f"Provider Used: {gen.get('provider_used')}")
            print(f"Images Generated: {len(gen.get('images', []))}")
            print(f"Generation Info: {gen.get('generation_info', {})}")
            
            # Check for actual image data
            images = gen.get('images', [])
            if images:
                print(f"\n🖼️ IMAGE DATA ANALYSIS:")
                for i, img in enumerate(images):
                    print(f"Image {i+1}:")
                    if isinstance(img, dict):
                        print(f"  Keys: {list(img.keys())}")
                        if 'url' in img:
                            url = img['url']
                            if url.startswith('data:image'):
                                # Extract format info
                                format_info = url.split(';')[0].split(':')[1]
                                print(f"  Format: {format_info}")
                                print(f"  Data length: {len(url)} characters")
                                
                                # Check if it's actually a GIF
                                if 'gif' in format_info.lower():
                                    print(f"  🎬 ANIMATED GIF DETECTED!")
                                else:
                                    print(f"  📷 Static image: {format_info}")
                            else:
                                print(f"  External URL: {url[:50]}...")
                        
                        if 'revised_prompt' in img:
                            print(f"  Revised: {img['revised_prompt'][:50]}...")
                    else:
                        print(f"  Raw data: {str(img)[:50]}...")
            else:
                print("No image data found in response")
                
        # Save full response for analysis
        with open('gif_generation_response.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Full response saved to: gif_generation_response.json")
        
    else:
        print(f"❌ Error {resp.status_code}: {resp.text}")
        
except requests.exceptions.Timeout:
    print("⏱️ Request timed out after 60s")
except Exception as e:
    print(f"❌ Exception: {e}")

print("\n" + "="*50)
print("🎯 NEXT: Testing integration with consciousness reasoning...")

# Test creating a visual for the consciousness analysis
consciousness_visual = {
    "prompt": "Philosophical diagram showing the 'Hard Problem of Consciousness' in AI: Split image with human brain (biological consciousness) on left showing neurons and synapses, AI chip/circuit (artificial consciousness) on right showing data flows, large question mark in center representing the explanatory gap, arrows pointing to concepts like 'qualia', 'subjective experience', 'phenomenology'",
    "style": "scientific",
    "size": "1024x512", 
    "provider": "google",
    "quality": "high"
}

print(f"\nTesting Consciousness Visualization:")
try:
    resp = requests.post("http://localhost:8000/image/generate", json=consciousness_visual, timeout=30)
    if resp.status_code == 200:
        print("✅ Consciousness diagram generation successful!")
        result = resp.json()
        images = result.get('generation', {}).get('images', [])
        print(f"Generated {len(images)} consciousness visualization(s)")
    else:
        print(f"⚠️ Status {resp.status_code}")
except:
    print("⏱️ Consciousness visualization timeout")

print(f"\nTesting completed!")
print(f"🎨 Ready to integrate visual generation with collaborative reasoning!")
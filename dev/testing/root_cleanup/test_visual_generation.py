#!/usr/bin/env python3
import requests
import json
import time

print("=== TESTING VISUAL GENERATION CAPABILITIES ===")

# Test different visual generation scenarios
test_cases = [
    {
        "name": "Philosophical Concept Visualization",
        "data": {
            "prompt": "Visual representation of consciousness layers in AI: sensory input layer, processing layer, awareness layer, and decision layer, shown as interconnected neural networks with flowing data",
            "style": "scientific",
            "size": "1024x1024",
            "provider": "google",
            "quality": "standard",
            "num_images": 1
        },
        "timeout": 30
    },
    {
        "name": "Animated Sequence Request", 
        "data": {
            "prompt": "Create an animated sequence showing the evolution of consciousness in AI systems: from simple pattern recognition to self-awareness, 4 frames showing progression",
            "style": "scientific",
            "size": "512x512", 
            "provider": "openai",
            "format": "gif",
            "animation": True,
            "num_images": 4
        },
        "timeout": 45
    },
    {
        "name": "Diagram Generation",
        "data": {
            "prompt": "Technical diagram illustrating the Hard Problem of Consciousness: brain neural networks on left, question mark in center, AI neural networks on right, with arrows showing the explanatory gap",
            "style": "technical",
            "size": "1024x512",
            "provider": "google",
            "quality": "high"
        },
        "timeout": 25
    },
    {
        "name": "Philosophy Mind Map",
        "data": {
            "prompt": "Mind map visualization of AI consciousness philosophy: central node 'AI Consciousness' connected to branches for Ethics, Metaphysics, Epistemology, Functionalism, each with sub-concepts",
            "style": "diagram", 
            "size": "1024x1024",
            "provider": "openai"
        },
        "timeout": 20
    }
]

results = []

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{i}. Testing {test_case['name']}:")
    print(f"   Prompt: {test_case['data']['prompt'][:60]}...")
    
    start_time = time.time()
    try:
        resp = requests.post("http://localhost:8000/image/generate", 
                           json=test_case["data"], 
                           timeout=test_case["timeout"])
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"   Status: {resp.status_code}")
        print(f"   Response Time: {response_time:.2f}s")
        
        if resp.status_code == 200:
            print(f"   ✅ {test_case['name']} generation successful")
            result = resp.json()
            
            if "generation" in result:
                gen = result["generation"]
                print(f"   Provider: {gen.get('provider_used', 'unknown')}")
                print(f"   Images: {len(gen.get('images', []))}")
                print(f"   Generation time: {gen.get('generation_info', {}).get('generation_time', 'unknown')}")
                
                # Check if any animation/GIF support
                if 'format' in gen or 'animation' in gen:
                    print(f"   🎬 Animation support detected!")
                    
            results.append({
                "test": test_case['name'],
                "status": "success",
                "time": response_time,
                "details": result
            })
            
        elif resp.status_code == 422:
            print(f"   ⚠️ Parameter error")
            error = resp.json()
            print(f"   Error details: {error}")
            results.append({
                "test": test_case['name'], 
                "status": "parameter_error",
                "error": error
            })
            
        else:
            print(f"   ❌ Error {resp.status_code}")
            print(f"   Response: {resp.text[:100]}")
            results.append({
                "test": test_case['name'],
                "status": "error", 
                "code": resp.status_code
            })
            
    except requests.exceptions.Timeout:
        print(f"   ⏱️ Timeout after {test_case['timeout']}s")
        results.append({
            "test": test_case['name'],
            "status": "timeout"
        })
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        results.append({
            "test": test_case['name'],
            "status": "exception",
            "error": str(e)
        })

print(f"\n{'='*60}")
print("VISUAL GENERATION TESTING SUMMARY")
print(f"{'='*60}")

successful = sum(1 for r in results if r["status"] == "success")
total = len(results)

print(f"Successful generations: {successful}/{total}")

for result in results:
    status_icon = "✅" if result["status"] == "success" else "❌" if result["status"] == "error" else "⏱️" if result["status"] == "timeout" else "⚠️"
    print(f"{status_icon} {result['test']}: {result['status']}")

if successful > 0:
    print(f"\n🎨 Visual generation capabilities confirmed!")
    print("Ready to enhance collaborative reasoning with visual outputs!")
else:
    print(f"\n⚠️ Visual generation needs investigation")

print("\nTesting completed.")
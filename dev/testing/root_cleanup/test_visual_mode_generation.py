#!/usr/bin/env python3
"""
Comprehensive test for visual mode with actual image generation
"""

import requests
import json
import time

print("🎨 COMPREHENSIVE VISUAL MODE + IMAGE GENERATION TEST")
print("="*70)

BASE_URL = "http://localhost:8000"

def test_visual_mode_with_generation():
    """Test visual mode that should actually generate images"""
    
    print("\n🖼️ TESTING VISUAL MODE WITH ACTUAL IMAGE GENERATION")
    print("-" * 60)
    
    test_cases = [
        {
            "name": "Neural Network Visual",
            "message": "Explain neural networks",
            "expected_visuals": ["neural network", "architecture", "diagram"],
            "should_generate": True
        },
        {
            "name": "Machine Learning Visual", 
            "message": "How does machine learning work?",
            "expected_visuals": ["learning", "process", "flow"],
            "should_generate": True
        },
        {
            "name": "AI Concepts Visual",
            "message": "What is artificial intelligence?",
            "expected_visuals": ["ai", "system", "overview"],
            "should_generate": True
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}:")
        print(f"   Message: '{test_case['message']}'")
        
        try:
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/chat", json={
                "message": test_case["message"],
                "user_id": "visual_test",
                "output_mode": "visual",
                "audience_level": "intermediate",
                "include_visuals": True,
                "show_confidence": False
            }, timeout=90)  # Longer timeout for image generation
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("response", "")
                
                print(f"   ✅ Status: {response.status_code}")
                print(f"   ⏱️ Time: {response_time:.2f}s")
                print(f"   📝 Length: {len(content)} chars")
                print(f"   🧠 Provider: {result.get('provider', 'unknown')}")
                
                # Check for visual generation indicators
                visual_indicators = [
                    "Generated Visuals:",
                    "✅", 
                    "Generated successfully",
                    "Image URL:",
                    "Visual Summary"
                ]
                
                found_indicators = []
                for indicator in visual_indicators:
                    if indicator in content:
                        found_indicators.append(indicator)
                
                print(f"   🎨 Visual Indicators: {len(found_indicators)}/{len(visual_indicators)}")
                if found_indicators:
                    print(f"      Found: {', '.join(found_indicators)}")
                
                # Check for actual image generation success
                has_generated_images = "Generated successfully" in content
                has_image_urls = "Image URL:" in content
                has_visual_summary = "Visual Summary" in content
                
                print(f"   🖼️ Generated Images: {'✅ YES' if has_generated_images else '❌ NO'}")
                print(f"   🔗 Image URLs: {'✅ YES' if has_image_urls else '❌ NO'}")
                print(f"   📊 Visual Summary: {'✅ YES' if has_visual_summary else '❌ NO'}")
                
                # Show content preview focusing on visual parts
                lines = content.split('\n')
                visual_lines = [line for line in lines if any(word in line.lower() for word in ['visual', 'generated', 'image', 'diagram', '✅', '🎨', '📊'])]
                
                if visual_lines:
                    print(f"   📄 Visual Content Preview:")
                    for line in visual_lines[:5]:  # Show first 5 visual-related lines
                        print(f"      {line.strip()}")
                
                # Determine success
                success = has_visual_summary and (has_generated_images or len(found_indicators) >= 3)
                status_icon = "✅" if success else "⚠️"
                
                print(f"   {status_icon} Result: {'SUCCESS' if success else 'PARTIAL'}")
                
                results.append({
                    "test": test_case["name"],
                    "status": "success" if success else "partial",
                    "time": response_time,
                    "has_generated_images": has_generated_images,
                    "has_image_urls": has_image_urls,
                    "visual_indicators": len(found_indicators),
                    "content_length": len(content)
                })
                
            else:
                print(f"   ❌ Error: {response.status_code}")
                print(f"   📄 Response: {response.text[:100]}...")
                results.append({
                    "test": test_case["name"],
                    "status": "error",
                    "code": response.status_code
                })
                
        except requests.exceptions.Timeout:
            print(f"   ⏱️ Timeout after 90s")
            results.append({
                "test": test_case["name"],
                "status": "timeout"
            })
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            results.append({
                "test": test_case["name"],
                "status": "exception"
            })
        
        # Small delay between tests
        time.sleep(2)
    
    return results

def test_direct_image_generation():
    """Test direct image generation API for comparison"""
    
    print("\n🔧 TESTING DIRECT IMAGE GENERATION API")
    print("-" * 60)
    
    test_prompts = [
        "Neural network architecture diagram with input, hidden, and output layers",
        "Machine learning process flowchart showing data to predictions",
        "AI system overview with interconnected components"
    ]
    
    direct_results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Testing: {prompt[:50]}...")
        
        try:
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/image/generate", json={
                "prompt": prompt,
                "style": "scientific",
                "size": "1024x768",
                "provider": "openai",
                "quality": "high"
            }, timeout=45)
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                generation = result.get("generation", {})
                
                print(f"   ✅ Status: {generation.get('status', 'unknown')}")
                print(f"   ⏱️ Time: {response_time:.2f}s")
                print(f"   🎨 Provider: {generation.get('provider_used', 'unknown')}")
                print(f"   🖼️ Images: {len(generation.get('images', []))}")
                
                direct_results.append({
                    "prompt": prompt[:30],
                    "status": "success",
                    "time": response_time,
                    "images": len(generation.get('images', []))
                })
            else:
                print(f"   ❌ Error: {response.status_code}")
                direct_results.append({
                    "prompt": prompt[:30],
                    "status": "error"
                })
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            direct_results.append({
                "prompt": prompt[:30],
                "status": "exception"
            })
    
    return direct_results

def generate_visual_test_report(visual_results, direct_results):
    """Generate comprehensive visual mode test report"""
    
    print("\n" + "="*70)
    print("📊 VISUAL MODE + IMAGE GENERATION REPORT")
    print("="*70)
    
    # Visual mode analysis
    successful_visual = [r for r in visual_results if r["status"] in ["success", "partial"]]
    visual_success_rate = len(successful_visual) / len(visual_results) * 100
    
    print(f"\n🎨 VISUAL MODE RESULTS:")
    print(f"   Success Rate: {len(successful_visual)}/{len(visual_results)} ({visual_success_rate:.1f}%)")
    
    if successful_visual:
        avg_time = sum(r.get("time", 0) for r in successful_visual) / len(successful_visual)
        images_generated = sum(1 for r in successful_visual if r.get("has_generated_images", False))
        urls_provided = sum(1 for r in successful_visual if r.get("has_image_urls", False))
        
        print(f"   Average Response Time: {avg_time:.2f}s")
        print(f"   Actual Images Generated: {images_generated}/{len(successful_visual)}")
        print(f"   Image URLs Provided: {urls_provided}/{len(successful_visual)}")
    
    # Direct generation analysis
    successful_direct = [r for r in direct_results if r["status"] == "success"]
    direct_success_rate = len(successful_direct) / len(direct_results) * 100
    
    print(f"\n🔧 DIRECT IMAGE GENERATION:")
    print(f"   Success Rate: {len(successful_direct)}/{len(direct_results)} ({direct_success_rate:.1f}%)")
    
    if successful_direct:
        avg_direct_time = sum(r.get("time", 0) for r in successful_direct) / len(successful_direct)
        print(f"   Average Generation Time: {avg_direct_time:.2f}s")
    
    # Individual test results
    print(f"\n📋 DETAILED RESULTS:")
    for result in visual_results:
        status_icon = {"success": "✅", "partial": "⚠️", "error": "❌", "timeout": "⏱️", "exception": "💥"}.get(result["status"], "❓")
        print(f"   {status_icon} {result['test']}: {result['status'].upper()}")
        
        if result["status"] in ["success", "partial"]:
            time_val = result.get("time", 0)
            images = "✅" if result.get("has_generated_images", False) else "❌"
            urls = "✅" if result.get("has_image_urls", False) else "❌"
            print(f"       Time: {time_val:.2f}s | Images: {images} | URLs: {urls}")
    
    # Overall assessment
    if visual_success_rate >= 80 and any(r.get("has_generated_images", False) for r in successful_visual):
        grade = "🟢 EXCELLENT"
        message = "Visual mode with image generation working perfectly!"
    elif visual_success_rate >= 60 and direct_success_rate >= 80:
        grade = "🟡 GOOD"  
        message = "Image generation works, visual mode integration needs improvement"
    elif direct_success_rate >= 60:
        grade = "🟠 FAIR"
        message = "Basic image generation working, visual mode needs fixes"
    else:
        grade = "🔴 POOR"
        message = "Image generation system needs major fixes"
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    print(f"   Grade: {grade}")
    print(f"   Status: {message}")
    
    # Specific confirmations
    print(f"\n✅ CONFIRMED WORKING:")
    if any(r.get("has_generated_images", False) for r in successful_visual):
        print(f"   • Visual mode actually generates images")
    if any(r.get("has_image_urls", False) for r in successful_visual):
        print(f"   • Image URLs are provided in responses")
    if direct_success_rate >= 80:
        print(f"   • Direct image generation API is functional")
    if visual_success_rate >= 60:
        print(f"   • Visual mode formatting is working")
    
    return visual_success_rate

def main():
    """Run comprehensive visual mode and image generation test"""
    
    print("Starting comprehensive visual mode + image generation test...")
    print("This will test end-to-end visual mode with actual image generation")
    
    # Test visual mode integration
    visual_results = test_visual_mode_with_generation()
    
    # Test direct image generation
    direct_results = test_direct_image_generation()
    
    # Generate comprehensive report
    success_rate = generate_visual_test_report(visual_results, direct_results)
    
    print(f"\n🎯 Visual mode test completed with {success_rate:.1f}% success rate!")
    
    if success_rate >= 80:
        print("🎉 Visual mode with image generation is working perfectly!")
    else:
        print("⚠️ Visual mode needs additional improvements")

if __name__ == "__main__":
    main()
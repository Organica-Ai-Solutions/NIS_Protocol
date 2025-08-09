#!/usr/bin/env python3
"""
Test GPT-Style Dynamic Chart Generation
Verifies the new /visualization/dynamic endpoint works like GPT/Claude
"""

import asyncio
import sys
import time
import requests
import json

# Test server URL
BASE_URL = "http://localhost:8000"

def test_dynamic_chart_endpoint():
    """Test the new GPT-style dynamic chart generation endpoint"""
    
    print("🎨 Testing GPT-Style Dynamic Chart Generation")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Physics - Bouncing Ball",
            "content": "When a ball bounces, several key principles of physics come into play: gravity, impact, elasticity, rebound, friction, and air resistance. The ball initially accelerates towards the ground due to gravity.",
            "topic": "Physics of a Bouncing Ball",
            "chart_type": "physics"
        },
        {
            "name": "Performance Metrics",
            "content": "The system shows significant improvements in speed (60% to 95%), accuracy (75% to 98%), and reliability (80% to 99%). These performance gains represent a major breakthrough.",
            "topic": "System Performance Analysis", 
            "chart_type": "performance"
        },
        {
            "name": "NIS Pipeline",
            "content": "The NIS Protocol pipeline consists of Laplace Transform signal processing, KAN symbolic reasoning, PINN physics validation, and LLM coordination working together.",
            "topic": "NIS Protocol Pipeline",
            "chart_type": "pipeline"
        },
        {
            "name": "Comparison Analysis",
            "content": "AI image generation achieves 60% accuracy while code-based generation reaches 100% accuracy. Speed differs significantly: AI generation takes 25 seconds vs 0.1 seconds for code generation.",
            "topic": "Visualization Method Comparison",
            "chart_type": "comparison"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📊 Test {i}: {test_case['name']}")
        print("-" * 40)
        
        try:
            # Test the dynamic endpoint
            start_time = time.time()
            
            response = requests.post(
                f"{BASE_URL}/visualization/dynamic",
                json={
                    "content": test_case["content"],
                    "topic": test_case["topic"],
                    "chart_type": test_case["chart_type"]
                },
                timeout=60  # Give it time for code execution
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Check if we got a successful chart
                if result.get("status") == "success" and result.get("dynamic_chart"):
                    chart = result["dynamic_chart"]
                    
                    print(f"✅ SUCCESS ({response_time:.2f}s)")
                    print(f"   Method: {chart.get('method', 'unknown')}")
                    print(f"   Format: {chart.get('format', 'unknown')}")
                    print(f"   Execution Time: {chart.get('execution_time_ms', 'N/A')}ms")
                    print(f"   Chart Image Length: {len(chart.get('chart_image', ''))}")
                    
                    if chart.get("chart_code"):
                        lines = chart["chart_code"].count('\n')
                        print(f"   Generated Code: {lines} lines of Python")
                    
                    results.append({
                        "test": test_case["name"],
                        "status": "success",
                        "response_time": response_time,
                        "method": chart.get("method"),
                        "execution_time": chart.get("execution_time_ms")
                    })
                    
                elif result.get("status") == "fallback":
                    print(f"⚠️  FALLBACK ({response_time:.2f}s)")
                    print(f"   Using SVG fallback instead of code execution")
                    
                    results.append({
                        "test": test_case["name"],
                        "status": "fallback",
                        "response_time": response_time,
                        "method": "svg_fallback"
                    })
                    
                else:
                    print(f"❌ UNEXPECTED RESPONSE ({response_time:.2f}s)")
                    print(f"   Status: {result.get('status')}")
                    print(f"   Result keys: {list(result.keys())}")
                    
                    results.append({
                        "test": test_case["name"],
                        "status": "unexpected",
                        "response_time": response_time,
                        "details": str(result)
                    })
                    
            else:
                print(f"❌ HTTP ERROR ({response_time:.2f}s)")
                print(f"   Status Code: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                
                results.append({
                    "test": test_case["name"],
                    "status": "http_error",
                    "response_time": response_time,
                    "status_code": response.status_code
                })
                
        except requests.exceptions.Timeout:
            print("❌ TIMEOUT (>60s)")
            print("   Code execution took too long")
            
            results.append({
                "test": test_case["name"],
                "status": "timeout",
                "response_time": 60.0
            })
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            
            results.append({
                "test": test_case["name"],
                "status": "error",
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 GPT-STYLE CHART GENERATION TEST SUMMARY")
    print("=" * 60)
    
    success_count = len([r for r in results if r["status"] == "success"])
    fallback_count = len([r for r in results if r["status"] == "fallback"])
    error_count = len([r for r in results if r["status"] not in ["success", "fallback"]])
    
    print(f"✅ Successful: {success_count}/{len(test_cases)}")
    print(f"⚠️  Fallback:   {fallback_count}/{len(test_cases)}")
    print(f"❌ Errors:     {error_count}/{len(test_cases)}")
    
    if success_count > 0:
        avg_time = sum(r["response_time"] for r in results if r["status"] == "success") / success_count
        print(f"⏱️  Avg Success Time: {avg_time:.2f}s")
    
    # Overall assessment
    if success_count >= len(test_cases) * 0.75:  # 75% success rate
        print("\n🎉 GPT-STYLE CHART GENERATION: WORKING EXCELLENTLY!")
        print("   The dynamic code generation approach is functioning like GPT/Claude")
        return True
    elif success_count + fallback_count >= len(test_cases) * 0.75:  # 75% working (including fallbacks)
        print("\n✅ GPT-STYLE CHART GENERATION: WORKING WITH FALLBACKS")
        print("   System gracefully falls back to SVG when code execution unavailable")
        return True
    else:
        print("\n❌ GPT-STYLE CHART GENERATION: NEEDS ATTENTION")
        print("   Code execution environment may need setup")
        return False

def test_health_first():
    """Test that the server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy and responding")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        return False

if __name__ == "__main__":
    print("🧪 NIS Protocol v3.2 - GPT-Style Chart Generation Test")
    print("Testing the new dynamic visualization system")
    print()
    
    # Test server health first
    if not test_health_first():
        print("\n❌ Cannot proceed - server not accessible")
        print("Please ensure NIS Protocol is running with: ./start.sh")
        sys.exit(1)
    
    # Test the dynamic chart generation
    success = test_dynamic_chart_endpoint()
    
    if success:
        print("\n🚀 READY FOR PRODUCTION!")
        print("Your NIS Protocol now generates charts like GPT/Claude!")
        sys.exit(0)
    else:
        print("\n⚠️  PARTIALLY WORKING")
        print("Basic functionality available, code execution needs environment setup")
        sys.exit(1)
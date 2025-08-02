#!/usr/bin/env python3
"""
Test script with CORRECT parameter structures for NIS Protocol endpoints
"""
import requests
import json
import time

def test_endpoint(method, url, data=None, description=""):
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Method: {method} {url}")
    if data:
        print(f"Data: {json.dumps(data, indent=2)}")
    
    try:
        start_time = time.time()
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        response_time = time.time() - start_time
        
        print(f"Status: {response.status_code}")
        print(f"Time: {response_time:.3f}s")
        
        if response.status_code in [200, 201, 202]:
            try:
                response_data = response.json()
                print("✅ SUCCESS - Response:")
                print(json.dumps(response_data, indent=2)[:800] + "..." if len(str(response_data)) > 800 else json.dumps(response_data, indent=2))
                return True
            except:
                print("✅ SUCCESS - Response (non-JSON):")
                print(response.text[:200] + "..." if len(response.text) > 200 else response.text)
                return True
        else:
            print(f"❌ FAILED - Error: {response.text[:400]}")
            return False
            
    except Exception as e:
        print(f"💥 EXCEPTION: {str(e)}")
        return False

def main():
    base_url = "http://localhost"
    
    print("🔧 Testing FIXED Parameter Structures")
    print("="*60)
    
    # Test 1: Fixed Curiosity endpoint - stimulus as dictionary
    curiosity_success = test_endpoint(
        "POST", 
        f"{base_url}/agents/curiosity/process_stimulus",
        {
            "stimulus": {
                "type": "physics_experiment",
                "content": "energy conservation test",
                "domain": "physics",
                "complexity": "medium"
            },
            "context": {
                "environment": "laboratory",
                "urgency": "normal"
            }
        },
        "Curiosity agent with CORRECT stimulus structure"
    )
    
    # Test 2: Fixed Ethics endpoint - action as dictionary with required field
    ethics_success = test_endpoint(
        "POST",
        f"{base_url}/agents/alignment/evaluate_ethics", 
        {
            "action": {
                "type": "ai_decision",
                "description": "AI system making autonomous physics calculations",
                "context": "scientific research",
                "scope": "data_analysis"
            },
            "context": {
                "domain": "scientific_research",
                "stakeholders": ["researchers", "public"],
                "urgency": "normal"
            }
        },
        "Ethics agent with CORRECT action structure"
    )
    
    # Test 3: Test async chat endpoint
    async_chat_success = test_endpoint(
        "POST",
        f"{base_url}/chat/stream",
        {
            "message": "Test the async chat interface with physics validation",
            "user_id": "test_user",
            "conversation_id": "test_conv_001"
        },
        "Async chat stream endpoint"
    )
    
    # Test 4: Working complex simulation endpoint
    simulation_complex_success = test_endpoint(
        "POST",
        f"{base_url}/agents/simulation/run",
        {
            "scenario_id": "physics_test_001",
            "scenario_type": "physics", 
            "parameters": {
                "mass": 1.0,
                "height": 10.0,
                "gravity": 9.8,
                "simulation_type": "falling_object"
            }
        },
        "Complex simulation endpoint"
    )
    
    # Test 5: NEW Simple simulation endpoint (just enabled)
    simulation_simple_success = test_endpoint(
        "POST",
        f"{base_url}/simulation/run",
        {
            "concept": "energy conservation in a falling ball"
        },
        "NEW Simple simulation endpoint (FIXED & ENABLED)"
    )
    
    # Summary
    results = [
        ("Curiosity (fixed)", curiosity_success),
        ("Ethics (fixed)", ethics_success), 
        ("Async Chat", async_chat_success),
        ("Complex Simulation", simulation_complex_success),
        ("Simple Simulation (NEW)", simulation_simple_success)
    ]
    
    print(f"\n{'='*60}")
    print("📊 FIXED ENDPOINTS SUMMARY")
    print("="*60)
    
    working = sum(1 for _, success in results if success)
    total = len(results)
    
    for desc, success in results:
        status = "✅ WORKING" if success else "❌ STILL BROKEN"
        print(f"{status}: {desc}")
    
    print(f"\n🎯 Fix Success Rate: {working}/{total} ({working/total*100:.1f}%)")
    
    if working == total:
        print("🎉 ALL ENDPOINTS NOW WORKING - DEMO READY!")
    elif working >= total * 0.75:
        print("🚀 MOSTLY WORKING - GOOD FOR DEMOS!")
    else:
        print("⚠️  STILL NEEDS WORK")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
NIS Protocol v3 - Comprehensive Endpoint Testing
Test all endpoints to see what's actually working vs empty/placeholder responses
"""

import requests
import json
import time
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import sys

@dataclass
class EndpointTest:
    method: str
    url: str
    data: Dict[str, Any] = None
    headers: Dict[str, str] = None
    expected_status: int = 200
    description: str = ""

class NISEndpointTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def test_endpoint(self, test: EndpointTest) -> Dict[str, Any]:
        """Test a single endpoint and return detailed results"""
        print(f"🧪 Testing {test.method} {test.url} - {test.description}")
        
        try:
            start_time = time.time()
            
            if test.method == "GET":
                response = self.session.get(test.url, timeout=30)
            elif test.method == "POST":
                response = self.session.post(test.url, json=test.data, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {test.method}")
            
            response_time = time.time() - start_time
            
            # Try to parse JSON
            try:
                response_data = response.json()
            except:
                response_data = {"raw_text": response.text[:500]}
            
            result = {
                "endpoint": test.url,
                "method": test.method,
                "description": test.description,
                "status_code": response.status_code,
                "response_time": round(response_time, 3),
                "expected_status": test.expected_status,
                "success": response.status_code == test.expected_status,
                "response_data": response_data,
                "has_meaningful_content": self._check_meaningful_content(response_data),
                "error": None
            }
            
            # Print result
            if result["success"] and result["has_meaningful_content"]:
                print(f"   ✅ SUCCESS - {response.status_code} in {response_time:.3f}s")
            elif result["success"]:
                print(f"   ⚠️  EMPTY - {response.status_code} in {response_time:.3f}s (placeholder response)")
            else:
                print(f"   ❌ FAILED - {response.status_code} in {response_time:.3f}s")
                
        except Exception as e:
            result = {
                "endpoint": test.url,
                "method": test.method,
                "description": test.description,
                "status_code": None,
                "response_time": None,
                "expected_status": test.expected_status,
                "success": False,
                "response_data": None,
                "has_meaningful_content": False,
                "error": str(e)
            }
            print(f"   💥 ERROR - {str(e)}")
        
        self.results.append(result)
        return result
    
    def _check_meaningful_content(self, response_data: Any) -> bool:
        """Check if response has meaningful content vs placeholder"""
        if not response_data:
            return False
        
        if isinstance(response_data, dict):
            # Check for common placeholder indicators
            response_str = json.dumps(response_data).lower()
            placeholders = [
                "placeholder", "todo", "not implemented", "coming soon",
                "mock", "fake", "dummy", "test", "empty",
                "null", "none", "{}", "[]"
            ]
            
            # If response is very short or contains placeholders
            if len(response_str) < 20 or any(p in response_str for p in placeholders):
                return False
                
            # Check for actual data structures
            meaningful_keys = [
                "status", "response", "result", "data", "payload",
                "confidence", "physics_compliance", "reasoning",
                "agents", "system", "health", "real_ai"
            ]
            
            return any(key in response_data for key in meaningful_keys)
        
        return True
    
    def run_comprehensive_test(self):
        """Run all endpoint tests"""
        print("🚀 NIS Protocol v3 - Comprehensive Endpoint Testing")
        print("=" * 60)
        
        # Define all tests
        tests = [
            # Core System Endpoints
            EndpointTest("GET", f"{self.base_url}/", description="System status and info"),
            EndpointTest("GET", f"{self.base_url}/health", description="Health check"),
            
            # Chat and Core Functionality  
            EndpointTest("POST", f"{self.base_url}/chat", 
                        data={"message": "Test physics validation with a simple pendulum"},
                        description="Main chat endpoint with physics test"),
            
            EndpointTest("POST", f"{self.base_url}/chat", 
                        data={"message": "What is 2+2?"},
                        description="Simple math question"),
            
            # Physics and Simulation
            EndpointTest("POST", f"{self.base_url}/simulation/run",
                        data={"concept": "conservation of energy in a falling object"},
                        description="Physics simulation endpoint"),
            
            # Agent Endpoints
            EndpointTest("POST", f"{self.base_url}/agents/learning/process",
                        data={"operation": "status"},
                        description="Learning agent"),
            
            EndpointTest("POST", f"{self.base_url}/agents/reasoning/analyze",
                        data={"query": "energy conservation", "domain": "physics"},
                        description="Reasoning agent"),
            
            EndpointTest("POST", f"{self.base_url}/agents/physics/validate",
                        data={"scenario": "A ball falls from height h with mass m"},
                        description="Physics validation agent"),
            
            EndpointTest("POST", f"{self.base_url}/agents/memory/store",
                        data={"key": "test", "data": {"test": "data"}},
                        description="Memory storage agent"),
            
            EndpointTest("POST", f"{self.base_url}/agents/simulation/run",
                        data={"scenario": "simple pendulum", "parameters": {"length": 1.0, "mass": 1.0}},
                        description="Simulation agent"),
            
            # Tool and Integration Endpoints
            EndpointTest("GET", f"{self.base_url}/tools/list", description="Available tools"),
            EndpointTest("GET", f"{self.base_url}/models/status", description="Model status"),
            EndpointTest("GET", f"{self.base_url}/integrations/status", description="Integration status"),
        ]
        
        # Run all tests
        for i, test in enumerate(tests, 1):
            print(f"\\n[{i}/{len(tests)}]", end=" ")
            self.test_endpoint(test)
            time.sleep(0.5)  # Brief pause between tests
        
        # Generate summary
        self.print_summary()
        self.save_results()
    
    def print_summary(self):
        """Print test summary"""
        print("\\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total = len(self.results)
        successful = sum(1 for r in self.results if r["success"])
        meaningful = sum(1 for r in self.results if r["success"] and r["has_meaningful_content"])
        empty = sum(1 for r in self.results if r["success"] and not r["has_meaningful_content"])
        failed = sum(1 for r in self.results if not r["success"])
        
        print(f"📈 Total Endpoints Tested: {total}")
        print(f"✅ Successful Responses: {successful}")
        print(f"🎯 With Meaningful Content: {meaningful}")
        print(f"⚠️  Empty/Placeholder Responses: {empty}")
        print(f"❌ Failed Responses: {failed}")
        print(f"📊 Success Rate: {successful/total*100:.1f}%")
        print(f"🔥 Production Ready: {meaningful/total*100:.1f}%")
        
        print("\\n🎯 DEMO-READY ENDPOINTS:")
        for result in self.results:
            if result["success"] and result["has_meaningful_content"]:
                print(f"   ✅ {result['method']} {result['endpoint']} - {result['description']}")
        
        print("\\n⚠️  NEEDS WORK (Empty/Placeholder):")
        for result in self.results:
            if result["success"] and not result["has_meaningful_content"]:
                print(f"   🔧 {result['method']} {result['endpoint']} - {result['description']}")
        
        print("\\n❌ BROKEN ENDPOINTS:")
        for result in self.results:
            if not result["success"]:
                error_msg = result["error"] or f"Status {result['status_code']}"
                print(f"   💥 {result['method']} {result['endpoint']} - {error_msg}")
    
    def save_results(self):
        """Save detailed results to file"""
        timestamp = int(time.time())
        filename = f"endpoint_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "test_summary": {
                    "total": len(self.results),
                    "successful": sum(1 for r in self.results if r["success"]),
                    "meaningful": sum(1 for r in self.results if r["success"] and r["has_meaningful_content"]),
                    "empty": sum(1 for r in self.results if r["success"] and not r["has_meaningful_content"]),
                    "failed": sum(1 for r in self.results if not r["success"])
                },
                "detailed_results": self.results
            }, f, indent=2)
        
        print(f"\\n💾 Detailed results saved to: {filename}")

if __name__ == "__main__":
    # Check if system is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print("🟢 System is running, starting tests...")
    except:
        print("🔴 System not responding on localhost:8000")
        print("💡 Run './rebuild_and_test.sh' first to start the system")
        sys.exit(1)
    
    # Run tests
    tester = NISEndpointTester()
    tester.run_comprehensive_test()
    
    print("\\n🎯 NEXT STEPS:")
    print("1. Fix empty/placeholder endpoints")
    print("2. Ensure physics validation actually works")
    print("3. Test with real physics scenarios")
    print("4. Document working features for demos")
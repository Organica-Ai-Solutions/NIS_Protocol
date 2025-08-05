#!/usr/bin/env python3
"""
Comprehensive NIS Protocol v3 Endpoint Testing Suite
Tests all available endpoints and reports status
"""

import requests
import json
import time
from typing import Dict, Any, List

class EndpointTester:
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.results = []
        
    def test_endpoint(self, method: str, endpoint: str, data: Dict = None, expected_status: int = 200) -> Dict[str, Any]:
        """Test a single endpoint and return results"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=10)
            elif method == "PUT":
                response = requests.put(url, json=data, timeout=10)
            elif method == "DELETE":
                response = requests.delete(url, timeout=10)
            else:
                return {"endpoint": endpoint, "status": "UNKNOWN_METHOD", "error": f"Unknown method: {method}"}
            
            result = {
                "endpoint": endpoint,
                "method": method,
                "status_code": response.status_code,
                "status": "PASS" if response.status_code == expected_status else "FAIL",
                "response_time": response.elapsed.total_seconds(),
                "content_length": len(response.text),
                "content_type": response.headers.get("content-type", "unknown")
            }
            
            try:
                result["response"] = response.json()
            except:
                result["response"] = response.text[:200] + "..." if len(response.text) > 200 else response.text
                
            return result
            
        except Exception as e:
            return {
                "endpoint": endpoint,
                "method": method,
                "status": "ERROR",
                "error": str(e)
            }
    
    def run_all_tests(self):
        """Run comprehensive tests for all NIS v3 endpoints"""
        print("🚀 Starting NIS Protocol v3 Endpoint Testing Suite")
        print("=" * 60)
        
        # Test cases: (method, endpoint, data, expected_status)
        test_cases = [
            # Basic endpoints
            ("GET", "/", None, 200),
            ("GET", "/health", None, 200),
            
            # Agent management
            ("GET", "/agents", None, 200),
            ("POST", "/agent/create", {"agent_type": "consciousness", "config": {}}, 200),
            
            # Chat endpoints  
            ("POST", "/chat", {"message": "Hello NIS Protocol"}, 200),
            ("POST", "/chat", {"message": "Test NIS pipeline", "agent_type": "consciousness"}, 200),
            
            # New endpoints from recent development
            ("GET", "/consciousness/status", None, 200),
            ("GET", "/infrastructure/status", None, 200),
            ("GET", "/metrics", None, 200),
            ("POST", "/process", {"input": "test data"}, 200),
            
            # Memory system
            ("POST", "/memory/store", {"content": "test memory", "metadata": {}}, 200),
            ("POST", "/memory/query", {"query": "test query"}, 200),
            
            # Tool execution
            ("POST", "/tool/execute", {"tool_name": "test_tool", "parameters": {}}, 200),
            
            # Protocol integrations
            ("GET", "/protocol/status", None, 200),
            
            # Dashboard
            ("GET", "/dashboard/metrics", None, 200),
        ]
        
        for method, endpoint, data, expected_status in test_cases:
            print(f"Testing {method} {endpoint}...")
            result = self.test_endpoint(method, endpoint, data, expected_status)
            self.results.append(result)
            
            status_icon = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⚠️"
            print(f"  {status_icon} {result['status']} - {result.get('status_code', 'N/A')}")
            
            # Brief pause between requests
            time.sleep(0.1)
        
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📊 ENDPOINT TEST RESULTS SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed = len([r for r in self.results if r["status"] == "PASS"])
        failed = len([r for r in self.results if r["status"] == "FAIL"])
        errors = len([r for r in self.results if r["status"] == "ERROR"])
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Errors: {errors}")
        print(f"Success Rate: {(passed/total_tests)*100:.1f}%")
        
        # Detailed results
        print("\n📋 DETAILED RESULTS:")
        print("-" * 60)
        
        for result in self.results:
            status_icon = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⚠️"
            endpoint = result["endpoint"]
            method = result.get("method", "")
            status = result["status"]
            
            print(f"{status_icon} {method:6} {endpoint:25} - {status}")
            
            if result["status"] == "ERROR":
                print(f"     Error: {result.get('error', 'Unknown error')}")
            elif result["status"] == "FAIL":
                print(f"     Expected: 200, Got: {result.get('status_code', 'N/A')}")
        
        # Save results to file
        with open("endpoint_test_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: endpoint_test_results.json")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if errors > 0:
            print("- Check server connectivity and ensure NIS v3 is running")
        if failed > 0:
            print("- Review failed endpoints for implementation issues")
        if passed == total_tests:
            print("- All endpoints working! System ready for production use")

if __name__ == "__main__":
    tester = EndpointTester()
    tester.run_all_tests() 
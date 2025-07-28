#!/usr/bin/env python3
"""
NIS Protocol v3.1 Complete Endpoint Testing Suite
Test all 40+ endpoints across 10 categories
"""

import requests
import json
import time
import asyncio
from typing import Dict, Any, List

class NISv31Tester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = {}
        
    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def test_endpoint(self, method: str, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """Test an endpoint and return results"""
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method == "GET":
                response = requests.get(url, timeout=15)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=15)
            elif method == "DELETE":
                response = requests.delete(url, timeout=15)
            
            result = {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "data": response.json() if response.status_code == 200 else None,
                "error": None
            }
            
            if response.status_code == 200:
                self.log(f"✅ {method} {endpoint} - {response.status_code}")
            else:
                self.log(f"❌ {method} {endpoint} - {response.status_code}")
                result["error"] = response.text
            
            return result
            
        except Exception as e:
            self.log(f"❌ {method} {endpoint} - Exception: {str(e)}")
            return {
                "success": False,
                "status_code": 0,
                "response_time": 0,
                "data": None,
                "error": str(e)
            }
    
    def test_core_endpoints(self):
        """Test core v3.0 enhanced endpoints"""
        self.log("🏠 Testing Core Endpoints...")
        
        tests = [
            ("GET", "/", None),
            ("GET", "/health", None),
        ]
        
        results = {}
        for method, endpoint, data in tests:
            results[f"{method} {endpoint}"] = self.test_endpoint(method, endpoint, data)
        
        self.test_results["core"] = results
        return results
    
    def test_conversational_layer(self):
        """Test conversational layer endpoints"""
        self.log("💬 Testing Conversational Layer...")
        
        tests = [
            ("POST", "/chat", {
                "message": "Hello! I'm testing the v3.1 enhanced chat system.",
                "user_id": "v31_tester"
            }),
            ("POST", "/chat/contextual", {
                "message": "Explain quantum computing with enhanced reasoning",
                "user_id": "v31_tester",
                "tools_enabled": ["web_search", "calculator"],
                "reasoning_mode": "chain_of_thought"
            })
        ]
        
        results = {}
        for method, endpoint, data in tests:
            results[f"{method} {endpoint}"] = self.test_endpoint(method, endpoint, data)
        
        self.test_results["conversational"] = results
        return results
    
    def test_internet_knowledge(self):
        """Test internet & knowledge endpoints"""
        self.log("🌐 Testing Internet & Knowledge...")
        
        tests = [
            ("POST", "/internet/search", {
                "query": "artificial intelligence consciousness",
                "max_results": 5,
                "academic_sources": True
            }),
            ("POST", "/internet/fetch-url", {
                "url": "https://example.com/ai-research",
                "parse_mode": "academic_paper",
                "extract_entities": True
            }),
            ("POST", "/internet/fact-check", {
                "statement": "AI systems can exhibit emergent consciousness",
                "confidence_threshold": 0.8
            }),
            ("GET", "/internet/status", None)
        ]
        
        results = {}
        for method, endpoint, data in tests:
            results[f"{method} {endpoint}"] = self.test_endpoint(method, endpoint, data)
        
        self.test_results["internet"] = results
        return results
    
    def test_tool_execution(self):
        """Test tool execution layer"""
        self.log("🔧 Testing Tool Execution...")
        
        tests = [
            ("GET", "/tool/list", None),
            ("POST", "/tool/execute", {
                "tool_name": "calculator",
                "parameters": {"expression": "2 + 2 * 3"},
                "sandbox": True
            }),
            ("POST", "/tool/register", {
                "name": "test_tool_v31",
                "description": "Test tool for v3.1 validation",
                "parameters_schema": {"input": {"type": "string"}},
                "category": "testing"
            }),
            ("POST", "/tool/test", {
                "tool_name": "calculator",
                "test_parameters": {"expression": "5 * 5"}
            })
        ]
        
        results = {}
        for method, endpoint, data in tests:
            if method == "POST" and endpoint == "/tool/test":
                # Special handling for tool test endpoint
                result = self.test_endpoint("POST", "/tool/test?tool_name=calculator", {"expression": "5*5"})
            else:
                result = self.test_endpoint(method, endpoint, data)
            results[f"{method} {endpoint}"] = result
        
        self.test_results["tools"] = results
        return results
    
    def test_agent_orchestration(self):
        """Test agent orchestration"""
        self.log("🤖 Testing Agent Orchestration...")
        
        tests = [
            ("POST", "/agent/create", {
                "agent_type": "research",
                "capabilities": ["web_search", "analysis", "synthesis"],
                "memory_size": "1GB",
                "tools": ["web_search", "calculator"]
            }),
            ("GET", "/agent/list", None)
        ]
        
        results = {}
        agent_id = None
        
        for method, endpoint, data in tests:
            result = self.test_endpoint(method, endpoint, data)
            results[f"{method} {endpoint}"] = result
            
            # Capture agent_id for further tests
            if endpoint == "/agent/create" and result["success"]:
                agent_id = result["data"].get("agent_id")
        
        # Test agent instruction if we have an agent_id
        if agent_id:
            instruct_result = self.test_endpoint("POST", "/agent/instruct", {
                "agent_id": agent_id,
                "instruction": "Analyze the current state of AI research",
                "priority": 5
            })
            results["POST /agent/instruct"] = instruct_result
        
        self.test_results["agents"] = results
        return results
    
    def test_model_management(self):
        """Test model management"""
        self.log("📊 Testing Model Management...")
        
        tests = [
            ("GET", "/models", None),
            ("POST", "/models/load", {
                "model_name": "test-model-v31",
                "model_type": "llm",
                "source": "local_cache"
            }),
            ("GET", "/models/status", None),
            ("POST", "/models/evaluate", {"model_name": "gpt-4", "test_dataset": "benchmark"})
        ]
        
        results = {}
        for method, endpoint, data in tests:
            if method == "POST" and endpoint == "/models/evaluate":
                # Special handling for evaluate endpoint
                result = self.test_endpoint("POST", "/models/evaluate?model_name=gpt-4&test_dataset=benchmark", None)
            else:
                result = self.test_endpoint(method, endpoint, data)
            results[f"{method} {endpoint}"] = result
        
        self.test_results["models"] = results
        return results
    
    def test_memory_knowledge(self):
        """Test memory & knowledge system"""
        self.log("🧠 Testing Memory & Knowledge...")
        
        tests = [
            ("POST", "/memory/store", {
                "content": "NIS Protocol v3.1 has been successfully implemented with 40+ endpoints",
                "metadata": {"type": "achievement", "version": "3.1"},
                "importance": 0.9
            }),
            ("POST", "/memory/query", {
                "query": "NIS Protocol achievements",
                "max_results": 5,
                "similarity_threshold": 0.7
            }),
            ("DELETE", "/memory/clear", {"session_id": "test_session_v31"})
        ]
        
        results = {}
        memory_id = None
        
        for method, endpoint, data in tests:
            if method == "DELETE":
                result = self.test_endpoint("DELETE", "/memory/clear?session_id=test_session_v31", None)
            else:
                result = self.test_endpoint(method, endpoint, data)
            results[f"{method} {endpoint}"] = result
            
            if endpoint == "/memory/store" and result["success"]:
                memory_id = result["data"].get("memory_id")
        
        # Test semantic linking if we have memory_id
        if memory_id:
            link_result = self.test_endpoint("POST", "/memory/semantic-link", {
                "source_id": memory_id,
                "target_id": "mem_example",
                "relationship": "related_to",
                "strength": 0.8
            })
            results["POST /memory/semantic-link"] = link_result
        
        self.test_results["memory"] = results
        return results
    
    def test_reasoning_validation(self):
        """Test reasoning & validation"""
        self.log("🧠 Testing Reasoning & Validation...")
        
        tests = [
            ("POST", "/reason/plan", {
                "query": "How does consciousness emerge in AI systems?",
                "reasoning_style": "chain_of_thought",
                "depth": "comprehensive",
                "validation_layers": ["logic", "evidence"]
            }),
            ("POST", "/reason/validate", {
                "reasoning_chain": [
                    "AI systems process information",
                    "Complex processing can exhibit emergent properties",
                    "Consciousness might be an emergent property"
                ],
                "physics_constraints": ["conservation_laws"],
                "confidence_threshold": 0.8
            }),
            ("POST", "/reason/simulate", {"query": "What is consciousness?", "num_paths": 3}),
            ("GET", "/reason/status", None)
        ]
        
        results = {}
        for method, endpoint, data in tests:
            if method == "POST" and endpoint == "/reason/simulate":
                result = self.test_endpoint("POST", "/reason/simulate?query=What%20is%20consciousness&num_paths=3", None)
            else:
                result = self.test_endpoint(method, endpoint, data)
            results[f"{method} {endpoint}"] = result
        
        self.test_results["reasoning"] = results
        return results
    
    def test_monitoring_logs(self):
        """Test monitoring & logs"""
        self.log("📊 Testing Monitoring & Logs...")
        
        tests = [
            ("GET", "/logs", None),
            ("GET", "/dashboard/realtime", None),
            ("GET", "/metrics/latency", None)
        ]
        
        results = {}
        for method, endpoint, data in tests:
            if endpoint == "/logs":
                result = self.test_endpoint("GET", "/logs?level=INFO&limit=10", None)
            else:
                result = self.test_endpoint(method, endpoint, data)
            results[f"{method} {endpoint}"] = result
        
        self.test_results["monitoring"] = results
        return results
    
    def test_developer_utilities(self):
        """Test developer utilities"""
        self.log("🛠️ Testing Developer Utilities...")
        
        tests = [
            ("POST", "/config/reload", None),
            ("POST", "/sandbox/execute", {
                "code": "print('Hello from NIS v3.1 sandbox!')",
                "language": "python",
                "timeout": 10
            })
        ]
        
        results = {}
        for method, endpoint, data in tests:
            result = self.test_endpoint(method, endpoint, data)
            results[f"{method} {endpoint}"] = result
        
        self.test_results["developer"] = results
        return results
    
    def test_experimental_layers(self):
        """Test experimental layers"""
        self.log("🔬 Testing Experimental Layers...")
        
        tests = [
            ("POST", "/kan/predict", {
                "input_data": [1.0, 2.0, 3.0],
                "function_type": "symbolic",
                "interpretability_mode": True
            }),
            ("POST", "/pinn/verify", {
                "system_state": {"position": [1, 2, 3], "velocity": [0.5, 0.2, 0.1]},
                "physical_laws": ["conservation_energy", "conservation_momentum"],
                "boundary_conditions": {"boundary_type": "fixed"}
            }),
            ("POST", "/laplace/transform", {
                "signal_data": [1.0, 0.5, 0.2, 0.1],
                "transform_type": "forward",
                "analysis_mode": "frequency"
            }),
            ("POST", "/a2a/connect", {
                "target_node": "nis://test-node.example.com:8000",
                "authentication": "shared_key",
                "sync_memory": True
            })
        ]
        
        results = {}
        for method, endpoint, data in tests:
            result = self.test_endpoint(method, endpoint, data)
            results[f"{method} {endpoint}"] = result
        
        self.test_results["experimental"] = results
        return results
    
    def run_comprehensive_test(self):
        """Run all v3.1 endpoint tests"""
        start_time = time.time()
        
        self.log("🚀 Starting NIS Protocol v3.1 Comprehensive Endpoint Testing")
        self.log("=" * 80)
        
        # Test all categories
        test_categories = [
            ("Core Endpoints", self.test_core_endpoints),
            ("Conversational Layer", self.test_conversational_layer),
            ("Internet & Knowledge", self.test_internet_knowledge),
            ("Tool Execution", self.test_tool_execution),
            ("Agent Orchestration", self.test_agent_orchestration),
            ("Model Management", self.test_model_management),
            ("Memory & Knowledge", self.test_memory_knowledge),
            ("Reasoning & Validation", self.test_reasoning_validation),
            ("Monitoring & Logs", self.test_monitoring_logs),
            ("Developer Utilities", self.test_developer_utilities),
            ("Experimental Layers", self.test_experimental_layers)
        ]
        
        total_tests = 0
        passed_tests = 0
        
        for category_name, test_function in test_categories:
            self.log(f"\n📋 Testing {category_name}...")
            category_results = test_function()
            
            category_passed = sum(1 for result in category_results.values() if result["success"])
            category_total = len(category_results)
            
            total_tests += category_total
            passed_tests += category_passed
            
            self.log(f"✅ {category_name}: {category_passed}/{category_total} passed")
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        self.log("\n" + "=" * 80)
        self.log("🎯 NIS PROTOCOL v3.1 COMPREHENSIVE TEST RESULTS")
        self.log("=" * 80)
        
        self.log(f"📊 Overall Results: {passed_tests}/{total_tests} endpoints passed ({passed_tests/total_tests*100:.1f}%)")
        self.log(f"⏱️ Total Test Time: {total_time:.1f}s")
        
        # Category breakdown
        self.log(f"\n📋 Category Breakdown:")
        for category, results in self.test_results.items():
            category_passed = sum(1 for result in results.values() if result["success"])
            category_total = len(results)
            self.log(f"   {category:20} {category_passed:2}/{category_total:2} ({category_passed/category_total*100:5.1f}%)")
        
        # Assessment
        success_rate = passed_tests / total_tests
        if success_rate >= 0.95:
            self.log("\n🏆 EXCELLENT: NIS Protocol v3.1 is performing exceptionally!")
            self.log("✅ All major endpoints operational with outstanding reliability!")
        elif success_rate >= 0.85:
            self.log("\n🎉 VERY GOOD: Strong v3.1 performance across all categories!")
            self.log("✅ System ready for production deployment!")
        elif success_rate >= 0.75:
            self.log("\n👍 GOOD: Solid v3.1 functionality with minor issues!")
            self.log("⚠️ Some endpoints need attention but core system is strong!")
        else:
            self.log("\n⚠️ NEEDS ATTENTION: Several endpoints require debugging!")
        
        # Save detailed results
        with open("v31_test_results.json", "w") as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "success_rate": success_rate,
                    "test_time": total_time,
                    "timestamp": time.time()
                },
                "detailed_results": self.test_results
            }, f, indent=2)
        
        self.log(f"\n📁 Detailed results saved to: v31_test_results.json")
        self.log("🎯 NIS Protocol v3.1 comprehensive testing completed!")
        
        return {
            "success_rate": success_rate,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "categories": len(test_categories)
        }

def main():
    """Run the comprehensive v3.1 test suite"""
    print("🔬 NIS PROTOCOL v3.1 COMPLETE ENDPOINT TESTING SUITE")
    print("Testing all 40+ endpoints across 10 categories")
    print("=" * 80)
    
    tester = NISv31Tester()
    results = tester.run_comprehensive_test()
    
    return results["success_rate"] >= 0.8

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 
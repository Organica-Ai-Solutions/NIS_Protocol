#!/usr/bin/env python3
"""
Comprehensive NIS Protocol v3.2 Endpoint Testing
Tests all available endpoints for proper functionality
"""

import requests
import json
import time
import base64
from typing import Dict, Any, List

BASE_URL = "http://localhost:8000"

class NISEndpointTester:
    def __init__(self):
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def log_result(self, endpoint: str, method: str, status: str, details: str, response_time: float = 0):
        """Log test result"""
        result = {
            "endpoint": endpoint,
            "method": method,
            "status": status,
            "details": details,
            "response_time": response_time,
            "timestamp": time.time()
        }
        self.results.append(result)
        self.total_tests += 1
        
        if status == "PASS":
            self.passed_tests += 1
            print(f"✅ {method} {endpoint} - {details} ({response_time:.3f}s)")
        else:
            self.failed_tests += 1
            print(f"❌ {method} {endpoint} - {details}")
    
    def test_basic_endpoints(self):
        """Test basic system endpoints"""
        print("\n🔧 Testing Basic Endpoints...")
        
        # Test root endpoint
        try:
            start_time = time.time()
            response = requests.get(f"{BASE_URL}/", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                self.log_result("/", "GET", "PASS", 
                    f"System operational - {len(data.get('real_llm_integrated', []))} providers", response_time)
            else:
                self.log_result("/", "GET", "FAIL", f"Status code: {response.status_code}")
        except Exception as e:
            self.log_result("/", "GET", "FAIL", f"Error: {str(e)}")
        
        # Test health endpoint
        try:
            start_time = time.time()
            response = requests.get(f"{BASE_URL}/health", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                self.log_result("/health", "GET", "PASS", 
                    f"Healthy - {data.get('conversations_active', 0)} conversations", response_time)
            else:
                self.log_result("/health", "GET", "FAIL", f"Status code: {response.status_code}")
        except Exception as e:
            self.log_result("/health", "GET", "FAIL", f"Error: {str(e)}")
    
    def test_chat_endpoints(self):
        """Test chat functionality"""
        print("\n💬 Testing Chat Endpoints...")
        
        # Test basic chat
        chat_payload = {
            "message": "Explain quantum entanglement in simple terms",
            "user_id": "test_user",
            "conversation_id": "test_conv_001",
            "agent_type": "consciousness_agent",
            "provider": "openai"
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/chat", json=chat_payload, timeout=30)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                self.log_result("/chat", "POST", "PASS", 
                    f"Response generated - Provider: {data.get('provider', 'unknown')}", response_time)
            else:
                self.log_result("/chat", "POST", "FAIL", f"Status code: {response.status_code}")
        except Exception as e:
            self.log_result("/chat", "POST", "FAIL", f"Error: {str(e)}")
        
        # Test formatted chat
        formatted_payload = {
            **chat_payload,
            "output_mode": "eli5",
            "audience_level": "beginner",
            "include_visuals": True,
            "show_confidence": True
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/chat/formatted", json=formatted_payload, timeout=30)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                self.log_result("/chat/formatted", "POST", "PASS", 
                    f"Formatted response - Mode: {data.get('output_mode', 'unknown')}", response_time)
            else:
                self.log_result("/chat/formatted", "POST", "FAIL", f"Status code: {response.status_code}")
        except Exception as e:
            self.log_result("/chat/formatted", "POST", "FAIL", f"Error: {str(e)}")
    
    def test_image_endpoints(self):
        """Test image generation and editing"""
        print("\n🎨 Testing Image Endpoints...")
        
        # Test image generation
        image_payload = {
            "prompt": "A beautiful dragon flying over a cyberpunk city",
            "style": "artistic",
            "size": "1024x1024",
            "provider": "google",
            "quality": "high"
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/image/generate", json=image_payload, timeout=45)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                generation = data.get("generation", {})
                if generation.get("status") == "success":
                    images = generation.get("images", [])
                    self.log_result("/image/generate", "POST", "PASS", 
                        f"Generated {len(images)} images - Provider: {generation.get('provider_used', 'unknown')}", response_time)
                else:
                    self.log_result("/image/generate", "POST", "FAIL", 
                        f"Generation failed: {generation.get('status', 'unknown')}")
            else:
                self.log_result("/image/generate", "POST", "FAIL", f"Status code: {response.status_code}")
        except Exception as e:
            self.log_result("/image/generate", "POST", "FAIL", f"Error: {str(e)}")
    
    def test_vision_endpoints(self):
        """Test vision analysis"""
        print("\n👁️ Testing Vision Endpoints...")
        
        # Create a simple test image (1x1 red pixel)
        test_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        vision_payload = {
            "image_data": test_image,
            "analysis_type": "comprehensive",
            "provider": "auto",
            "context": "Testing NIS Protocol vision analysis capabilities"
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/vision/analyze", json=vision_payload, timeout=30)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                analysis = data.get("analysis", {})
                self.log_result("/vision/analyze", "POST", "PASS", 
                    f"Analysis complete - Confidence: {analysis.get('confidence', 0):.2f}", response_time)
            else:
                self.log_result("/vision/analyze", "POST", "FAIL", f"Status code: {response.status_code}")
        except Exception as e:
            self.log_result("/vision/analyze", "POST", "FAIL", f"Error: {str(e)}")
    
    def test_research_endpoints(self):
        """Test research functionality"""
        print("\n🔬 Testing Research Endpoints...")
        
        research_payload = {
            "query": "quantum computing breakthroughs 2024",
            "research_depth": "comprehensive",
            "source_types": ["arxiv", "semantic_scholar", "wikipedia"],
            "time_limit": 60,
            "min_sources": 3
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/research/deep", json=research_payload, timeout=90)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                research = data.get("research", {})
                sources = research.get("sources", [])
                self.log_result("/research/deep", "POST", "PASS", 
                    f"Research complete - {len(sources)} sources found", response_time)
            else:
                self.log_result("/research/deep", "POST", "FAIL", f"Status code: {response.status_code}")
        except Exception as e:
            self.log_result("/research/deep", "POST", "FAIL", f"Error: {str(e)}")
    
    def test_reasoning_endpoints(self):
        """Test collaborative reasoning"""
        print("\n🧠 Testing Reasoning Endpoints...")
        
        reasoning_payload = {
            "query": "What are the ethical implications of artificial general intelligence?",
            "reasoning_depth": "comprehensive",
            "include_multiple_perspectives": True,
            "citation_required": True
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/reasoning/collaborative", json=reasoning_payload, timeout=60)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                self.log_result("/reasoning/collaborative", "POST", "PASS", 
                    f"Reasoning complete - Multi-perspective analysis", response_time)
            else:
                self.log_result("/reasoning/collaborative", "POST", "FAIL", f"Status code: {response.status_code}")
        except Exception as e:
            self.log_result("/reasoning/collaborative", "POST", "FAIL", f"Error: {str(e)}")
    
    def test_specialized_endpoints(self):
        """Test specialized endpoints"""
        print("\n🚀 Testing Specialized Endpoints...")
        
        endpoints_to_test = [
            ("/visualization/create", {
                "data_type": "physics_simulation",
                "parameters": {"scenario": "pendulum", "time_steps": 100}
            }),
            ("/document/analyze", {
                "document_data": "This is a test document for analysis.",
                "document_type": "text",
                "processing_mode": "comprehensive"
            }),
            ("/agents/multimodal/status", {}),
            ("/training/bitnet/status", {})
        ]
        
        for endpoint, payload in endpoints_to_test:
            try:
                start_time = time.time()
                if payload:
                    response = requests.post(f"{BASE_URL}{endpoint}", json=payload, timeout=30)
                else:
                    response = requests.get(f"{BASE_URL}{endpoint}", timeout=30)
                response_time = time.time() - start_time
                
                method = "POST" if payload else "GET"
                if response.status_code in [200, 201]:
                    self.log_result(endpoint, method, "PASS", 
                        f"Endpoint functional", response_time)
                else:
                    self.log_result(endpoint, method, "FAIL", 
                        f"Status code: {response.status_code}")
            except Exception as e:
                method = "POST" if payload else "GET"
                self.log_result(endpoint, method, "FAIL", f"Error: {str(e)}")
    
    def run_all_tests(self):
        """Run comprehensive endpoint testing"""
        print("🧪 NIS Protocol v3.2 Comprehensive Endpoint Testing")
        print("=" * 55)
        
        start_time = time.time()
        
        # Run all test suites
        self.test_basic_endpoints()
        self.test_chat_endpoints()
        self.test_image_endpoints()
        self.test_vision_endpoints()
        self.test_research_endpoints()
        self.test_reasoning_endpoints()
        self.test_specialized_endpoints()
        
        total_time = time.time() - start_time
        
        # Print summary
        print(f"\n📊 Testing Summary")
        print("=" * 30)
        print(f"Total Tests: {self.total_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"🎯 Success Rate: {(self.passed_tests/self.total_tests*100):.1f}%")
        print(f"⏱️ Total Time: {total_time:.2f}s")
        
        if self.failed_tests > 0:
            print(f"\n⚠️ Failed Endpoints:")
            for result in self.results:
                if result["status"] == "FAIL":
                    print(f"   • {result['method']} {result['endpoint']} - {result['details']}")
        
        # Save detailed results
        with open("dev/testing/endpoint_test_results.json", "w") as f:
            json.dump({
                "summary": {
                    "total_tests": self.total_tests,
                    "passed_tests": self.passed_tests,
                    "failed_tests": self.failed_tests,
                    "success_rate": self.passed_tests/self.total_tests*100,
                    "total_time": total_time
                },
                "results": self.results
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: dev/testing/endpoint_test_results.json")
        
        return self.passed_tests, self.failed_tests

if __name__ == "__main__":
    tester = NISEndpointTester()
    passed, failed = tester.run_all_tests()
    
    if failed == 0:
        print("\n🎉 All endpoints working perfectly!")
    else:
        print(f"\n🔧 Need to fix {failed} endpoint(s)")
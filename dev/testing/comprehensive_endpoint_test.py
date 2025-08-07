#!/usr/bin/env python3
"""
Comprehensive Endpoint Testing Suite
===================================

This script tests all NIS Protocol v3 endpoints including the new Enhanced Memory System endpoints.
It can run with or without a live server for testing purposes.
"""

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the FastAPI app and dependencies
try:
    from main import app
    from fastapi.testclient import TestClient
    from src.chat.enhanced_memory_chat import EnhancedChatMemory, ChatMemoryConfig
    from src.agents.memory.enhanced_memory_agent import EnhancedMemoryAgent
    print("✅ Successfully imported test components")
except ImportError as e:
    print(f"❌ Failed to import components: {e}")
    sys.exit(1)

class EndpointTester:
    """Comprehensive endpoint testing suite."""
    
    def __init__(self, use_live_server: bool = False):
        self.use_live_server = use_live_server
        if use_live_server:
            self.base_url = "http://localhost:8000"
            import requests
            self.client = requests
        else:
            # Use FastAPI test client for direct testing
            self.client = TestClient(app)
            self.base_url = ""
        
        self.test_results = []
        self.conversation_id = None
        self.user_id = "test_user_endpoint_suite"
    
    def log_test(self, test_name: str, status: str, details: str = ""):
        """Log test results."""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": time.time()
        }
        self.test_results.append(result)
        
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_icon} {test_name}: {status}")
        if details:
            print(f"   {details}")
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        print("\n🧪 Testing Health Endpoint")
        try:
            if self.use_live_server:
                response = self.client.get(f"{self.base_url}/health")
                data = response.json()
            else:
                response = self.client.get("/health")
                data = response.json()
            
            if response.status_code == 200:
                self.log_test("Health Check", "PASS", f"Status: {data.get('status', 'unknown')}")
                return True
            else:
                self.log_test("Health Check", "FAIL", f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Health Check", "FAIL", str(e))
            return False
    
    def test_basic_chat_endpoint(self):
        """Test basic chat functionality."""
        print("\n🧪 Testing Basic Chat Endpoint")
        try:
            chat_data = {
                "message": "What is the NIS Protocol v3 architecture?",
                "user_id": self.user_id,
                "provider": "openai",
                "agent_type": "general"
            }
            
            if self.use_live_server:
                response = self.client.post(f"{self.base_url}/chat", json=chat_data)
                data = response.json()
            else:
                response = self.client.post("/chat", json=chat_data)
                data = response.json()
            
            if response.status_code == 200:
                content = data.get('content', '')
                self.conversation_id = data.get('conversation_id')
                self.log_test("Basic Chat", "PASS", f"Response length: {len(content)}, Conv ID: {self.conversation_id}")
                return True
            else:
                self.log_test("Basic Chat", "FAIL", f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Basic Chat", "FAIL", str(e))
            return False
    
    def test_formatted_chat_endpoint(self):
        """Test formatted chat endpoint."""
        print("\n🧪 Testing Formatted Chat Endpoint")
        try:
            chat_data = {
                "message": "How does the KAN reasoning layer work in detail?",
                "user_id": self.user_id,
                "conversation_id": self.conversation_id,
                "provider": "openai",
                "agent_type": "general"
            }
            
            if self.use_live_server:
                response = self.client.post(f"{self.base_url}/chat/formatted", json=chat_data)
            else:
                response = self.client.post("/chat/formatted", json=chat_data)
            
            if response.status_code == 200:
                content = response.text if hasattr(response, 'text') else str(response.content)
                self.log_test("Formatted Chat", "PASS", f"HTML response length: {len(content)}")
                return True
            else:
                self.log_test("Formatted Chat", "FAIL", f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Formatted Chat", "FAIL", str(e))
            return False
    
    def test_memory_stats_endpoint(self):
        """Test memory statistics endpoint."""
        print("\n🧪 Testing Memory Stats Endpoint")
        try:
            if self.use_live_server:
                response = self.client.get(f"{self.base_url}/memory/stats")
                data = response.json()
            else:
                response = self.client.get("/memory/stats")
                data = response.json()
            
            if response.status_code == 200:
                stats = data.get('stats', {})
                enhanced_enabled = stats.get('enhanced_memory_enabled', False)
                total_messages = stats.get('total_messages', 0)
                self.log_test("Memory Stats", "PASS", f"Enhanced: {enhanced_enabled}, Messages: {total_messages}")
                return True
            else:
                self.log_test("Memory Stats", "FAIL", f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Memory Stats", "FAIL", str(e))
            return False
    
    def test_conversation_search_endpoint(self):
        """Test conversation search endpoint."""
        print("\n🧪 Testing Conversation Search Endpoint")
        try:
            params = {
                "query": "architecture",
                "user_id": self.user_id,
                "limit": 5
            }
            
            if self.use_live_server:
                response = self.client.get(f"{self.base_url}/memory/conversations", params=params)
                data = response.json()
            else:
                response = self.client.get("/memory/conversations", params=params)
                data = response.json()
            
            if response.status_code == 200:
                conversations = data.get('conversations', [])
                self.log_test("Conversation Search", "PASS", f"Found {len(conversations)} conversations")
                return True
            else:
                self.log_test("Conversation Search", "FAIL", f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Conversation Search", "FAIL", str(e))
            return False
    
    def test_conversation_details_endpoint(self):
        """Test conversation details endpoint."""
        print("\n🧪 Testing Conversation Details Endpoint")
        if not self.conversation_id:
            self.log_test("Conversation Details", "SKIP", "No conversation ID available")
            return False
        
        try:
            if self.use_live_server:
                response = self.client.get(f"{self.base_url}/memory/conversation/{self.conversation_id}")
                data = response.json()
            else:
                response = self.client.get(f"/memory/conversation/{self.conversation_id}")
                data = response.json()
            
            if response.status_code == 200:
                message_count = data.get('message_count', 0)
                summary = data.get('summary', '')
                self.log_test("Conversation Details", "PASS", f"Messages: {message_count}, Summary: {summary[:50]}...")
                return True
            else:
                self.log_test("Conversation Details", "FAIL", f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Conversation Details", "FAIL", str(e))
            return False
    
    def test_topics_endpoint(self):
        """Test topics discovery endpoint."""
        print("\n🧪 Testing Topics Endpoint")
        try:
            if self.use_live_server:
                response = self.client.get(f"{self.base_url}/memory/topics")
                data = response.json()
            else:
                response = self.client.get("/memory/topics")
                data = response.json()
            
            if response.status_code == 200:
                topics = data.get('topics', [])
                total_topics = data.get('total_topics', 0)
                self.log_test("Topics Discovery", "PASS", f"Found {len(topics)} topics, Total: {total_topics}")
                return True
            else:
                self.log_test("Topics Discovery", "FAIL", f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Topics Discovery", "FAIL", str(e))
            return False
    
    def test_context_preview_endpoint(self):
        """Test context preview endpoint."""
        print("\n🧪 Testing Context Preview Endpoint")
        if not self.conversation_id:
            self.log_test("Context Preview", "SKIP", "No conversation ID available")
            return False
        
        try:
            params = {
                "message": "Tell me more about the physics validation layer"
            }
            
            if self.use_live_server:
                response = self.client.get(f"{self.base_url}/memory/conversation/{self.conversation_id}/context", params=params)
                data = response.json()
            else:
                response = self.client.get(f"/memory/conversation/{self.conversation_id}/context", params=params)
                data = response.json()
            
            if response.status_code == 200:
                context_count = data.get('context_count', 0)
                self.log_test("Context Preview", "PASS", f"Context messages: {context_count}")
                return True
            else:
                self.log_test("Context Preview", "FAIL", f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Context Preview", "FAIL", str(e))
            return False
    
    def test_deep_conversation(self):
        """Test deep conversation with memory continuity."""
        print("\n🧪 Testing Deep Conversation with Memory")
        
        conversation_topics = [
            "Explain the NIS Protocol v3 signal processing pipeline in detail",
            "How does the Laplace transform layer specifically handle time-domain signals?", 
            "What role does the KAN layer play in making the processing interpretable?",
            "How does the PINN layer validate the physics constraints?",
            "Can you connect all these layers and explain how they work together?",
            "What are the key advantages of this architecture over traditional approaches?"
        ]
        
        conversation_results = []
        
        for i, topic in enumerate(conversation_topics):
            try:
                chat_data = {
                    "message": topic,
                    "user_id": self.user_id,
                    "conversation_id": self.conversation_id,
                    "provider": "openai",
                    "agent_type": "general"
                }
                
                if self.use_live_server:
                    response = self.client.post(f"{self.base_url}/chat", json=chat_data)
                    data = response.json()
                else:
                    response = self.client.post("/chat", json=chat_data)
                    data = response.json()
                
                if response.status_code == 200:
                    content = data.get('content', '')
                    self.conversation_id = data.get('conversation_id')
                    conversation_results.append({
                        "topic": topic,
                        "response_length": len(content),
                        "success": True
                    })
                    print(f"   ✅ Deep conversation {i+1}/6: {len(content)} chars")
                else:
                    conversation_results.append({
                        "topic": topic,
                        "error": f"Status {response.status_code}",
                        "success": False
                    })
                    print(f"   ❌ Deep conversation {i+1}/6: Failed")
                
                # Small delay between messages
                time.sleep(1)
                
            except Exception as e:
                conversation_results.append({
                    "topic": topic,
                    "error": str(e),
                    "success": False
                })
                print(f"   ❌ Deep conversation {i+1}/6: {str(e)}")
        
        successful_topics = sum(1 for r in conversation_results if r.get('success', False))
        self.log_test("Deep Conversation", "PASS" if successful_topics >= 4 else "PARTIAL", 
                     f"Completed {successful_topics}/6 topics successfully")
        
        return successful_topics >= 4
    
    def test_all_endpoints(self):
        """Run comprehensive test suite."""
        print("🚀 NIS Protocol v3 Comprehensive Endpoint Testing")
        print("=" * 60)
        
        # Core endpoints
        tests = [
            self.test_health_endpoint,
            self.test_basic_chat_endpoint,
            self.test_formatted_chat_endpoint,
            self.test_memory_stats_endpoint,
            self.test_conversation_search_endpoint,
            self.test_conversation_details_endpoint,
            self.test_topics_endpoint,
            self.test_context_preview_endpoint,
            self.test_deep_conversation
        ]
        
        passed = 0
        total = len(tests)
        
        for test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                print(f"❌ Test {test_func.__name__} crashed: {e}")
        
        print(f"\n📊 Test Results: {passed}/{total} tests passed")
        
        # Print detailed results
        print("\n📋 Detailed Results:")
        for result in self.test_results:
            status_icon = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⚠️"
            print(f"{status_icon} {result['test']}: {result['status']}")
            if result["details"]:
                print(f"   └─ {result['details']}")
        
        return passed, total
    
    def generate_test_report(self):
        """Generate a detailed test report."""
        report = {
            "test_suite": "NIS Protocol v3 Comprehensive Endpoint Testing",
            "timestamp": time.time(),
            "results": self.test_results,
            "summary": {
                "total_tests": len(self.test_results),
                "passed": sum(1 for r in self.test_results if r["status"] == "PASS"),
                "failed": sum(1 for r in self.test_results if r["status"] == "FAIL"),
                "skipped": sum(1 for r in self.test_results if r["status"] == "SKIP")
            }
        }
        
        return report

def main():
    """Run the comprehensive endpoint test suite."""
    print("🧪 Starting Comprehensive Endpoint Testing...")
    
    # Test with FastAPI test client (doesn't require live server)
    print("\n🔧 Testing with FastAPI Test Client (Direct)")
    tester = EndpointTester(use_live_server=False)
    passed, total = tester.test_all_endpoints()
    
    # Generate test report
    report = tester.generate_test_report()
    
    # Save test report
    with open("dev/testing/endpoint_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n🎉 Testing Complete! {passed}/{total} tests passed")
    print(f"📄 Detailed report saved to: dev/testing/endpoint_test_report.json")
    
    if passed == total:
        print("✨ All endpoints are working perfectly!")
    elif passed >= total * 0.8:
        print("👍 Most endpoints working well with minor issues")
    else:
        print("⚠️  Some endpoints need attention")

if __name__ == "__main__":
    main()
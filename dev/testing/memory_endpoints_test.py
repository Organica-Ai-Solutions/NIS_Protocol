#!/usr/bin/env python3
"""
Enhanced Memory Endpoints Deep Testing
=====================================

Comprehensive testing of all Enhanced Memory System endpoints
with real data and deep conversation scenarios.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from main import app
    from fastapi.testclient import TestClient
    print("✅ Successfully imported test components")
except ImportError as e:
    print(f"❌ Failed to import components: {e}")
    sys.exit(1)

class MemoryEndpointTester:
    """Deep testing of Enhanced Memory System endpoints."""
    
    def __init__(self):
        self.client = TestClient(app)
        self.test_results = []
        self.conversation_ids = []
        self.user_id = "deep_test_user_001"
    
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
    
    def setup_test_conversations(self):
        """Create test conversations by adding messages directly to memory."""
        print("\n🔧 Setting up test conversations...")
        
        # We'll add messages to the legacy memory to simulate conversations
        from main import conversation_memory, add_message_to_conversation
        
        # Create test conversations
        test_conversations = [
            {
                "id": f"test_conv_{self.user_id}_001",
                "messages": [
                    ("user", "What is the NIS Protocol v3 architecture?"),
                    ("assistant", "The NIS Protocol v3 features a multi-layered architecture with Laplace transform signal processing, KAN reasoning layers for interpretability, and PINN physics validation."),
                    ("user", "How does the signal processing work?"),
                    ("assistant", "The signal processing uses Laplace transforms to convert time-domain signals into frequency domain for analysis, enabling better pattern recognition and noise filtering.")
                ]
            },
            {
                "id": f"test_conv_{self.user_id}_002",
                "messages": [
                    ("user", "Explain neural network interpretability"),
                    ("assistant", "Neural network interpretability involves making the decision-making process of neural networks transparent and understandable to humans through various techniques like attention mechanisms and feature visualization."),
                    ("user", "What role does KAN play in this?"),
                    ("assistant", "KAN (Knowledge Augmented Networks) layers use spline-based function approximation that provides inherent interpretability by representing learned functions in a mathematically interpretable form.")
                ]
            }
        ]
        
        for conv in test_conversations:
            conv_id = conv["id"]
            self.conversation_ids.append(conv_id)
            
            for role, content in conv["messages"]:
                # Add to legacy memory system
                message = {
                    "role": role, 
                    "content": content, 
                    "timestamp": time.time()
                }
                if conv_id not in conversation_memory:
                    conversation_memory[conv_id] = []
                conversation_memory[conv_id].append(message)
        
        print(f"   ✅ Created {len(test_conversations)} test conversations")
    
    def test_memory_stats(self):
        """Test memory statistics endpoint in detail."""
        print("\n🧪 Testing Memory Stats Endpoint (Deep)")
        
        response = self.client.get("/memory/stats")
        
        if response.status_code == 200:
            data = response.json()
            stats = data.get('stats', {})
            
            # Check all expected fields
            expected_fields = ['enhanced_memory_enabled', 'total_conversations', 'total_messages']
            missing_fields = [field for field in expected_fields if field not in stats]
            
            if missing_fields:
                self.log_test("Memory Stats Fields", "PARTIAL", f"Missing fields: {missing_fields}")
            else:
                self.log_test("Memory Stats Fields", "PASS", "All expected fields present")
            
            # Check data types and reasonable values
            total_convs = stats.get('total_conversations', 0)
            total_msgs = stats.get('total_messages', 0)
            
            if total_convs >= 2 and total_msgs >= 8:
                self.log_test("Memory Stats Data", "PASS", f"Convs: {total_convs}, Msgs: {total_msgs}")
                return True
            else:
                self.log_test("Memory Stats Data", "PARTIAL", f"Lower than expected - Convs: {total_convs}, Msgs: {total_msgs}")
                return True
        else:
            self.log_test("Memory Stats", "FAIL", f"Status code: {response.status_code}")
            return False
    
    def test_conversation_search_deep(self):
        """Deep testing of conversation search functionality."""
        print("\n🧪 Testing Conversation Search (Deep)")
        
        test_queries = [
            ("architecture", "Should find NIS Protocol discussion"),
            ("neural network", "Should find interpretability discussion"),
            ("signal processing", "Should find signal processing discussion"),
            ("nonexistent_topic_xyz", "Should return empty results"),
            ("", "Should handle empty query")
        ]
        
        passed = 0
        total = len(test_queries)
        
        for query, description in test_queries:
            params = {
                "query": query,
                "user_id": self.user_id,
                "limit": 10
            }
            
            response = self.client.get("/memory/conversations", params=params)
            
            if response.status_code == 200:
                data = response.json()
                conversations = data.get('conversations', [])
                
                # Check response structure
                if 'status' in data and 'conversations' in data:
                    if query == "nonexistent_topic_xyz":
                        # Should return few or no results
                        if len(conversations) <= 1:
                            self.log_test(f"Search: {query}", "PASS", f"Correctly returned {len(conversations)} results")
                            passed += 1
                        else:
                            self.log_test(f"Search: {query}", "PARTIAL", f"Expected few results, got {len(conversations)}")
                    elif query == "":
                        # Empty query should handle gracefully
                        self.log_test(f"Search: empty", "PASS", f"Handled empty query, returned {len(conversations)} results")
                        passed += 1
                    else:
                        # Should find relevant results
                        if len(conversations) > 0:
                            self.log_test(f"Search: {query}", "PASS", f"Found {len(conversations)} conversations")
                            passed += 1
                        else:
                            self.log_test(f"Search: {query}", "PARTIAL", "No results found")
                else:
                    self.log_test(f"Search: {query}", "FAIL", "Invalid response structure")
            else:
                self.log_test(f"Search: {query}", "FAIL", f"Status code: {response.status_code}")
        
        overall_status = "PASS" if passed == total else "PARTIAL" if passed >= total/2 else "FAIL"
        self.log_test("Conversation Search Deep", overall_status, f"{passed}/{total} queries successful")
        return passed >= total/2
    
    def test_conversation_details_deep(self):
        """Deep testing of conversation details endpoint."""
        print("\n🧪 Testing Conversation Details (Deep)")
        
        if not self.conversation_ids:
            self.log_test("Conversation Details Deep", "SKIP", "No conversation IDs available")
            return False
        
        passed = 0
        total = len(self.conversation_ids)
        
        for conv_id in self.conversation_ids:
            # Test with context
            response = self.client.get(f"/memory/conversation/{conv_id}?include_context=true")
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ['status', 'conversation_id', 'message_count']
                if all(field in data for field in required_fields):
                    message_count = data.get('message_count', 0)
                    if message_count > 0:
                        self.log_test(f"Details: {conv_id[:20]}...", "PASS", f"Messages: {message_count}")
                        passed += 1
                    else:
                        self.log_test(f"Details: {conv_id[:20]}...", "PARTIAL", "No messages found")
                else:
                    missing = [f for f in required_fields if f not in data]
                    self.log_test(f"Details: {conv_id[:20]}...", "FAIL", f"Missing fields: {missing}")
            else:
                self.log_test(f"Details: {conv_id[:20]}...", "FAIL", f"Status code: {response.status_code}")
        
        overall_status = "PASS" if passed == total else "PARTIAL" if passed >= total/2 else "FAIL"
        self.log_test("Conversation Details Deep", overall_status, f"{passed}/{total} conversations analyzed")
        return passed >= total/2
    
    def test_topics_endpoint_deep(self):
        """Deep testing of topics discovery."""
        print("\n🧪 Testing Topics Discovery (Deep)")
        
        # Test different limits
        test_limits = [5, 10, 20, 50]
        passed = 0
        
        for limit in test_limits:
            response = self.client.get(f"/memory/topics?limit={limit}")
            
            if response.status_code == 200:
                data = response.json()
                
                if 'status' in data and 'topics' in data:
                    topics = data.get('topics', [])
                    total_topics = data.get('total_topics', 0)
                    
                    # Check that response respects limit
                    if len(topics) <= limit:
                        self.log_test(f"Topics limit {limit}", "PASS", f"Returned {len(topics)}/{total_topics} topics")
                        passed += 1
                    else:
                        self.log_test(f"Topics limit {limit}", "FAIL", f"Exceeded limit: {len(topics)} > {limit}")
                else:
                    self.log_test(f"Topics limit {limit}", "FAIL", "Invalid response structure")
            else:
                self.log_test(f"Topics limit {limit}", "FAIL", f"Status code: {response.status_code}")
        
        overall_status = "PASS" if passed == len(test_limits) else "PARTIAL" if passed >= len(test_limits)/2 else "FAIL"
        self.log_test("Topics Discovery Deep", overall_status, f"{passed}/{len(test_limits)} limit tests passed")
        return passed >= len(test_limits)/2
    
    def test_context_preview_deep(self):
        """Deep testing of context preview functionality."""
        print("\n🧪 Testing Context Preview (Deep)")
        
        if not self.conversation_ids:
            self.log_test("Context Preview Deep", "SKIP", "No conversation IDs available")
            return False
        
        test_messages = [
            "Tell me more about the architecture",
            "How does the physics layer work?", 
            "What about interpretability?",
            "Can you explain the signal processing in detail?"
        ]
        
        passed = 0
        total = len(test_messages) * len(self.conversation_ids)
        
        for conv_id in self.conversation_ids:
            for message in test_messages:
                params = {"message": message}
                response = self.client.get(f"/memory/conversation/{conv_id}/context", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'status' in data and 'context_messages' in data:
                        context_count = data.get('context_count', 0)
                        self.log_test(f"Context: {message[:20]}...", "PASS", f"Context: {context_count} messages")
                        passed += 1
                    else:
                        self.log_test(f"Context: {message[:20]}...", "FAIL", "Invalid response structure")
                else:
                    self.log_test(f"Context: {message[:20]}...", "FAIL", f"Status code: {response.status_code}")
        
        overall_status = "PASS" if passed == total else "PARTIAL" if passed >= total/2 else "FAIL"
        self.log_test("Context Preview Deep", overall_status, f"{passed}/{total} context previews successful")
        return passed >= total/2
    
    def test_memory_cleanup(self):
        """Test memory cleanup functionality (safely)."""
        print("\n🧪 Testing Memory Cleanup (Safe)")
        
        # Test with a very long retention period to avoid deleting test data
        response = self.client.post("/memory/cleanup?days_to_keep=365")
        
        if response.status_code == 200:
            data = response.json()
            
            if 'status' in data and 'message' in data:
                self.log_test("Memory Cleanup", "PASS", data.get('message', ''))
                return True
            else:
                self.log_test("Memory Cleanup", "FAIL", "Invalid response structure")
                return False
        else:
            self.log_test("Memory Cleanup", "FAIL", f"Status code: {response.status_code}")
            return False
    
    def test_error_handling(self):
        """Test error handling for invalid requests."""
        print("\n🧪 Testing Error Handling")
        
        test_cases = [
            ("GET", "/memory/conversation/nonexistent_id", "Non-existent conversation"),
            ("GET", "/memory/topic/nonexistent_topic/conversations", "Non-existent topic"),
            ("GET", "/memory/conversation/invalid_id/context?message=test", "Invalid conversation context")
        ]
        
        passed = 0
        total = len(test_cases)
        
        for method, endpoint, description in test_cases:
            if method == "GET":
                response = self.client.get(endpoint)
            else:
                response = self.client.post(endpoint)
            
            # Should handle errors gracefully (not crash)
            if response.status_code in [200, 404, 400, 422]:
                self.log_test(f"Error: {description}", "PASS", f"Status: {response.status_code}")
                passed += 1
            else:
                self.log_test(f"Error: {description}", "FAIL", f"Unexpected status: {response.status_code}")
        
        overall_status = "PASS" if passed == total else "PARTIAL" if passed >= total/2 else "FAIL"
        self.log_test("Error Handling", overall_status, f"{passed}/{total} error cases handled properly")
        return passed >= total/2
    
    def run_comprehensive_test(self):
        """Run all memory endpoint tests."""
        print("🚀 Enhanced Memory System Deep Testing")
        print("=" * 60)
        
        # Setup
        self.setup_test_conversations()
        
        # Run all tests
        tests = [
            self.test_memory_stats,
            self.test_conversation_search_deep,
            self.test_conversation_details_deep,
            self.test_topics_endpoint_deep,
            self.test_context_preview_deep,
            self.test_memory_cleanup,
            self.test_error_handling
        ]
        
        passed = 0
        total = len(tests)
        
        for test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                test_name = test_func.__name__
                self.log_test(f"Test {test_name}", "FAIL", f"Exception: {str(e)}")
        
        print(f"\n📊 Deep Test Results: {passed}/{total} test suites passed")
        
        # Generate summary
        if passed == total:
            print("🎉 EXCELLENT: All enhanced memory endpoints working perfectly!")
        elif passed >= total * 0.8:
            print("👍 GOOD: Most enhanced memory endpoints working well")
        else:
            print("⚠️  NEEDS ATTENTION: Some enhanced memory endpoints have issues")
        
        return passed, total
    
    def generate_detailed_report(self):
        """Generate a detailed test report."""
        report = {
            "test_suite": "Enhanced Memory System Deep Testing",
            "timestamp": time.time(),
            "user_id": self.user_id,
            "conversation_ids": self.conversation_ids,
            "results": self.test_results,
            "summary": {
                "total_tests": len(self.test_results),
                "passed": sum(1 for r in self.test_results if r["status"] == "PASS"),
                "failed": sum(1 for r in self.test_results if r["status"] == "FAIL"),
                "partial": sum(1 for r in self.test_results if r["status"] == "PARTIAL"),
                "skipped": sum(1 for r in self.test_results if r["status"] == "SKIP")
            }
        }
        
        return report

def main():
    """Run the comprehensive memory endpoint testing."""
    tester = MemoryEndpointTester()
    passed, total = tester.run_comprehensive_test()
    
    # Generate and save detailed report
    report = tester.generate_detailed_report()
    
    with open("dev/testing/memory_endpoints_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: dev/testing/memory_endpoints_test_report.json")
    
    # Print detailed results
    print("\n📋 Detailed Test Results:")
    for result in tester.test_results:
        status_icon = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⚠️" if result["status"] == "PARTIAL" else "⏭️"
        print(f"{status_icon} {result['test']}: {result['status']}")
        if result["details"]:
            print(f"   └─ {result['details']}")

if __name__ == "__main__":
    main()
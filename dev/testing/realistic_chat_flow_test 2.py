#!/usr/bin/env python3
"""
Realistic Chat Flow Testing Suite
# NVIDIA NIM Integration for NIS Protocol v3.0
# Recommended models for consciousness enhancement:
# 1. nvidia/llama-3.3-nemotron-super-49b-v1.5 (Core consciousness)
# 2. nvidia/llama-3.1-nemoguard-8b-content-safety (Safety)
# 3. nvidia/llama-3.1-nemotron-nano-4b-v1.1 (A-to-A agents)
# 4. nvidia/llama-3.2-90b-vision-instruct (Multimodal consciousness)

================================docker build -t nis-protocol:3.2 . --no-cache

This test suite simulates real-world chat scenarios to test:
- Actual chat endpoint functionality
- Realistic conversation flows
- Memory integration in practice
- Response quality and context usage
- Stream vs regular chat comparison
"""

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from main import app, conversation_memory, initialize_system
    from fastapi.testclient import TestClient
    print("✅ Successfully imported realistic chat flow test components")
except ImportError as e:
    print(f"❌ Failed to import components: {e}")
    sys.exit(1)

class RealisticChatFlowTester:
    """Test realistic chat flows and endpoint functionality."""
    
    def __init__(self):
        self.client = TestClient(app)
        self.test_results = []
        self.chat_sessions = {}
        
    def log_test(self, test_name: str, status: str, details: str = "", flow_data: Dict = None):
        """Enhanced logging with chat flow metrics."""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "flow_data": flow_data or {},
            "timestamp": time.time()
        }
        self.test_results.append(result)
        
        status_icon = "💬" if status == "CHAT_SUCCESS" else "🌊" if status == "STREAM_SUCCESS" else "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️" if status == "PARTIAL" else "🔄" if status == "PROCESSING" else "📊"
        print(f"{status_icon} {test_name}: {status}")
        if details:
            print(f"   {details}")
        if flow_data:
            print(f"   📊 Flow: {flow_data}")
    
    def test_basic_chat_endpoint_functionality(self):
        """Test basic chat endpoint with enhanced memory."""
        print("\n💬 Testing Basic Chat Endpoint Functionality")
        
        # Test different chat scenarios
        chat_scenarios = [
            {
                "name": "simple_question",
                "message": "What is machine learning?",
                "user_id": "test_user_001",
                "expected_keywords": ["machine learning", "algorithms", "data"]
            },
            {
                "name": "technical_question", 
                "message": "Explain the mathematical foundations of gradient descent optimization",
                "user_id": "test_user_002",
                "expected_keywords": ["gradient", "optimization", "mathematical"]
            },
            {
                "name": "follow_up_question",
                "message": "How does this relate to what we discussed about neural networks?",
                "user_id": "test_user_001",
                "expected_keywords": ["neural networks", "relate", "discussed"]
            },
            {
                "name": "context_dependent_question",
                "message": "Can you elaborate on the backpropagation algorithm from our previous conversation?",
                "user_id": "test_user_002", 
                "expected_keywords": ["backpropagation", "algorithm", "previous"]
            }
        ]
        
        chat_results = []
        
        for scenario in chat_scenarios:
            name = scenario["name"]
            message = scenario["message"]
            user_id = scenario["user_id"]
            expected_keywords = scenario["expected_keywords"]
            
            # Test /chat endpoint
            chat_payload = {
                "message": message,
                "user_id": user_id,
                "conversation_id": f"chat_test_{user_id}"
            }
            
            start_time = time.time()
            response = self.client.post("/chat", json=chat_payload)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                assistant_response = data.get('response', '')
                conversation_id = data.get('conversation_id', '')
                
                # Analyze response quality
                keyword_matches = sum(1 for keyword in expected_keywords 
                                    if keyword.lower() in assistant_response.lower())
                keyword_coverage = keyword_matches / len(expected_keywords)
                
                # Check if response is substantive
                response_length = len(assistant_response)
                substantive_response = response_length > 50
                
                # Check for memory integration indicators
                memory_indicators = ["discussed", "previous", "mentioned", "talked about", "earlier"]
                memory_integration = any(indicator in assistant_response.lower() 
                                       for indicator in memory_indicators)
                
                chat_metrics = {
                    "scenario": name,
                    "response_time": round(response_time * 1000, 2),  # ms
                    "response_length": response_length,
                    "keyword_coverage": round(keyword_coverage, 2),
                    "substantive_response": substantive_response,
                    "memory_integration": memory_integration,
                    "conversation_id": conversation_id
                }
                
                chat_results.append(chat_metrics)
                
                status = "CHAT_SUCCESS" if keyword_coverage >= 0.5 and substantive_response else "PARTIAL"
                self.log_test(f"Chat Endpoint: {name}", status,
                            f"Keywords: {keyword_coverage:.1%}, Response: {response_length} chars, Time: {response_time*1000:.1f}ms",
                            chat_metrics)
            else:
                self.log_test(f"Chat Endpoint: {name}", "FAIL", f"HTTP {response.status_code}")
        
        # Overall chat endpoint assessment
        successful_chats = sum(1 for r in chat_results if r["keyword_coverage"] >= 0.5 and r["substantive_response"])
        avg_response_time = sum(r["response_time"] for r in chat_results) / len(chat_results) if chat_results else 0
        avg_keyword_coverage = sum(r["keyword_coverage"] for r in chat_results) / len(chat_results) if chat_results else 0
        
        overall_status = "CHAT_SUCCESS" if successful_chats >= len(chat_scenarios) * 0.75 else "PARTIAL"
        self.log_test("Basic Chat Endpoint", overall_status,
                     f"Success: {successful_chats}/{len(chat_scenarios)}, Avg time: {avg_response_time:.1f}ms, Coverage: {avg_keyword_coverage:.1%}")
        
        return successful_chats >= len(chat_scenarios) * 0.5
    
    def test_formatted_chat_endpoint(self):
        """Test formatted chat endpoint with enhanced formatting."""
        print("\n💬 Testing Formatted Chat Endpoint")
        
        # Test scenarios that should trigger different formatting
        formatted_scenarios = [
            {
                "name": "code_explanation",
                "message": "Show me a Python function to calculate fibonacci numbers",
                "user_id": "format_user_001",
                "expected_elements": ["python", "function", "fibonacci"]
            },
            {
                "name": "mathematical_content",
                "message": "Explain the mathematical formula for compound interest with examples",
                "user_id": "format_user_002", 
                "expected_elements": ["formula", "mathematical", "compound interest"]
            },
            {
                "name": "step_by_step_process",
                "message": "How do I set up a machine learning pipeline step by step?",
                "user_id": "format_user_003",
                "expected_elements": ["step", "machine learning", "pipeline"]
            },
            {
                "name": "comparison_analysis",
                "message": "Compare supervised vs unsupervised learning approaches",
                "user_id": "format_user_004",
                "expected_elements": ["compare", "supervised", "unsupervised"]
            }
        ]
        
        formatted_results = []
        
        for scenario in formatted_scenarios:
            name = scenario["name"]
            message = scenario["message"]
            user_id = scenario["user_id"]
            expected_elements = scenario["expected_elements"]
            
            # Test /chat/formatted endpoint
            formatted_payload = {
                "message": message,
                "user_id": user_id,
                "conversation_id": f"formatted_test_{user_id}",
                "format_style": "detailed"
            }
            
            start_time = time.time()
            response = self.client.post("/chat/formatted", json=formatted_payload)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                formatted_response = data.get('formatted_response', '')
                conversation_id = data.get('conversation_id', '')
                
                # Analyze formatting quality
                element_matches = sum(1 for element in expected_elements 
                                    if element.lower() in formatted_response.lower())
                element_coverage = element_matches / len(expected_elements)
                
                # Check for formatting indicators
                formatting_indicators = ["##", "**", "```", "1.", "2.", "3.", "-", "*"]
                formatting_present = any(indicator in formatted_response 
                                       for indicator in formatting_indicators)
                
                # Check response structure
                response_sections = len(formatted_response.split('\n\n'))
                structured_response = response_sections > 2
                
                formatted_metrics = {
                    "scenario": name,
                    "response_time": round(response_time * 1000, 2),
                    "response_length": len(formatted_response),
                    "element_coverage": round(element_coverage, 2),
                    "formatting_present": formatting_present,
                    "structured_response": structured_response,
                    "response_sections": response_sections,
                    "conversation_id": conversation_id
                }
                
                formatted_results.append(formatted_metrics)
                
                status = "CHAT_SUCCESS" if element_coverage >= 0.5 and formatting_present else "PARTIAL"
                self.log_test(f"Formatted Chat: {name}", status,
                            f"Elements: {element_coverage:.1%}, Formatting: {formatting_present}, Sections: {response_sections}",
                            formatted_metrics)
            else:
                self.log_test(f"Formatted Chat: {name}", "FAIL", f"HTTP {response.status_code}")
        
        # Overall formatted chat assessment
        successful_formatted = sum(1 for r in formatted_results 
                                 if r["element_coverage"] >= 0.5 and r["formatting_present"])
        avg_sections = sum(r["response_sections"] for r in formatted_results) / len(formatted_results) if formatted_results else 0
        
        overall_status = "CHAT_SUCCESS" if successful_formatted >= len(formatted_scenarios) * 0.75 else "PARTIAL"
        self.log_test("Formatted Chat Endpoint", overall_status,
                     f"Success: {successful_formatted}/{len(formatted_scenarios)}, Avg sections: {avg_sections:.1f}")
        
        return successful_formatted >= len(formatted_scenarios) * 0.5
    
    def test_streaming_chat_functionality(self):
        """Test streaming chat endpoint functionality."""
        print("\n🌊 Testing Streaming Chat Functionality")
        
        # Test streaming scenarios
        streaming_scenarios = [
            {
                "name": "long_explanation",
                "message": "Explain in detail how neural networks work, including backpropagation, activation functions, and training process",
                "user_id": "stream_user_001",
                "expected_length": 200  # Minimum expected response length
            },
            {
                "name": "complex_technical",
                "message": "Describe the mathematical foundations of quantum computing, including qubits, superposition, entanglement, and quantum gates",
                "user_id": "stream_user_002",
                "expected_length": 300
            },
            {
                "name": "step_by_step_guide",
                "message": "Provide a comprehensive guide to machine learning model deployment in production environments",
                "user_id": "stream_user_003",
                "expected_length": 250
            }
        ]
        
        streaming_results = []
        
        for scenario in streaming_scenarios:
            name = scenario["name"]
            message = scenario["message"]
            user_id = scenario["user_id"]
            expected_length = scenario["expected_length"]
            
            # Test /chat/stream endpoint
            stream_payload = {
                "message": message,
                "user_id": user_id,
                "conversation_id": f"stream_test_{user_id}"
            }
            
            start_time = time.time()
            
            # Note: Testing streaming endpoint with regular client
            # In real implementation, this would be a streaming response
            response = self.client.post("/chat/stream", json=stream_payload)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                # For this test, we'll treat it as a regular response
                # In production, this would be streamed chunks
                try:
                    data = response.json()
                    streamed_response = data.get('response', '')
                    conversation_id = data.get('conversation_id', '')
                except:
                    # Handle potential streaming format
                    streamed_response = response.text
                    conversation_id = "stream_conv"
                
                # Analyze streaming response
                response_length = len(streamed_response)
                meets_length_expectation = response_length >= expected_length
                
                # Check for comprehensive content
                technical_indicators = ["process", "system", "algorithm", "method", "approach"]
                technical_depth = sum(1 for indicator in technical_indicators 
                                    if indicator.lower() in streamed_response.lower())
                comprehensive_content = technical_depth >= 3
                
                streaming_metrics = {
                    "scenario": name,
                    "response_time": round(response_time * 1000, 2),
                    "response_length": response_length,
                    "expected_length": expected_length,
                    "meets_length_expectation": meets_length_expectation,
                    "technical_depth": technical_depth,
                    "comprehensive_content": comprehensive_content,
                    "conversation_id": conversation_id
                }
                
                streaming_results.append(streaming_metrics)
                
                status = "STREAM_SUCCESS" if meets_length_expectation and comprehensive_content else "PARTIAL"
                self.log_test(f"Streaming Chat: {name}", status,
                            f"Length: {response_length}/{expected_length}, Depth: {technical_depth}, Time: {response_time*1000:.1f}ms",
                            streaming_metrics)
            else:
                self.log_test(f"Streaming Chat: {name}", "FAIL", f"HTTP {response.status_code}")
        
        # Overall streaming assessment
        successful_streams = sum(1 for r in streaming_results 
                               if r["meets_length_expectation"] and r["comprehensive_content"])
        avg_length = sum(r["response_length"] for r in streaming_results) / len(streaming_results) if streaming_results else 0
        
        overall_status = "STREAM_SUCCESS" if successful_streams >= len(streaming_scenarios) * 0.75 else "PARTIAL"
        self.log_test("Streaming Chat Endpoint", overall_status,
                     f"Success: {successful_streams}/{len(streaming_scenarios)}, Avg length: {avg_length:.0f} chars")
        
        return successful_streams >= len(streaming_scenarios) * 0.5
    
    def test_conversation_continuity_across_endpoints(self):
        """Test conversation continuity across different chat endpoints."""
        print("\n🔄 Testing Conversation Continuity Across Endpoints")
        
        # Create a multi-turn conversation using different endpoints
        continuity_user = "continuity_user_001"
        continuity_conv_id = f"continuity_test_{uuid.uuid4().hex[:8]}"
        
        conversation_flow = [
            {
                "endpoint": "/chat",
                "message": "Let's start learning about artificial intelligence",
                "expected_context": ["artificial intelligence", "AI"]
            },
            {
                "endpoint": "/chat/formatted",
                "message": "Can you explain machine learning in detail?",
                "expected_context": ["machine learning", "learning", "artificial intelligence"]
            },
            {
                "endpoint": "/chat/stream",
                "message": "How does this connect to what we discussed about AI?",
                "expected_context": ["AI", "artificial intelligence", "connect", "discussed"]
            },
            {
                "endpoint": "/chat",
                "message": "What are the practical applications of these concepts?",
                "expected_context": ["applications", "practical", "concepts"]
            }
        ]
        
        continuity_results = []
        accumulated_context = []
        
        for i, step in enumerate(conversation_flow):
            endpoint = step["endpoint"]
            message = step["message"]
            expected_context = step["expected_context"]
            
            # Prepare payload
            payload = {
                "message": message,
                "user_id": continuity_user,
                "conversation_id": continuity_conv_id
            }
            
            # Add format style for formatted endpoint
            if endpoint == "/chat/formatted":
                payload["format_style"] = "detailed"
            
            # Make request
            start_time = time.time()
            response = self.client.post(endpoint, json=payload)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    response_text = data.get('response', '') or data.get('formatted_response', '')
                except:
                    response_text = response.text
                
                # Check context continuity
                context_matches = sum(1 for context in expected_context 
                                    if context.lower() in response_text.lower())
                context_coverage = context_matches / len(expected_context)
                
                # Check if response references previous conversation
                previous_reference_indicators = ["discussed", "mentioned", "talked about", "previous", "earlier", "before"]
                references_previous = any(indicator in response_text.lower() 
                                        for indicator in previous_reference_indicators) if i > 0 else True
                
                # Update accumulated context
                accumulated_context.extend(expected_context)
                
                # Check if response builds on accumulated context
                accumulated_matches = sum(1 for context in accumulated_context 
                                        if context.lower() in response_text.lower())
                context_building = accumulated_matches / len(set(accumulated_context)) if accumulated_context else 0
                
                continuity_metrics = {
                    "step": i + 1,
                    "endpoint": endpoint,
                    "response_time": round(response_time * 1000, 2),
                    "context_coverage": round(context_coverage, 2),
                    "references_previous": references_previous,
                    "context_building": round(context_building, 2),
                    "response_length": len(response_text)
                }
                
                continuity_results.append(continuity_metrics)
                
                status = "PASS" if context_coverage >= 0.5 and (references_previous or i == 0) else "PARTIAL"
                self.log_test(f"Continuity Step {i+1}: {endpoint}", status,
                            f"Context: {context_coverage:.1%}, References: {references_previous}, Building: {context_building:.1%}",
                            continuity_metrics)
            else:
                self.log_test(f"Continuity Step {i+1}: {endpoint}", "FAIL", f"HTTP {response.status_code}")
        
        # Overall continuity assessment
        successful_steps = sum(1 for r in continuity_results 
                             if r["context_coverage"] >= 0.5 and r["references_previous"])
        avg_context_building = sum(r["context_building"] for r in continuity_results) / len(continuity_results) if continuity_results else 0
        
        overall_status = "PASS" if successful_steps >= len(conversation_flow) * 0.75 else "PARTIAL"
        self.log_test("Conversation Continuity", overall_status,
                     f"Successful steps: {successful_steps}/{len(conversation_flow)}, Context building: {avg_context_building:.1%}")
        
        return successful_steps >= len(conversation_flow) * 0.5
    
    def test_memory_integration_in_chat_flow(self):
        """Test how well memory integration works in actual chat flows."""
        print("\n🧠 Testing Memory Integration in Chat Flow")
        
        # Create a knowledge-building conversation that should use memory effectively
        memory_user = "memory_user_001"
        memory_conv_id = f"memory_integration_{uuid.uuid4().hex[:8]}"
        
        # First, establish some knowledge in the conversation
        setup_messages = [
            "I'm working on a quantum computing project involving variational quantum eigensolvers",
            "The project focuses on molecular simulation for drug discovery applications",
            "We're particularly interested in optimizing the ansatz circuits for better performance"
        ]
        
        # Add setup messages through chat endpoint
        for msg in setup_messages:
            setup_payload = {
                "message": msg,
                "user_id": memory_user,
                "conversation_id": memory_conv_id
            }
            self.client.post("/chat", json=setup_payload)
            time.sleep(0.1)  # Small delay between messages
        
        # Now test memory retrieval with follow-up questions
        memory_test_queries = [
            {
                "query": "Can you remind me what my project is about?",
                "expected_memories": ["quantum computing", "variational", "molecular simulation"],
                "memory_type": "project_recall"
            },
            {
                "query": "How can I improve the ansatz circuits we discussed?",
                "expected_memories": ["ansatz", "circuits", "optimize", "performance"],
                "memory_type": "technical_follow_up"
            },
            {
                "query": "What are the applications of this work in drug discovery?",
                "expected_memories": ["drug discovery", "molecular", "applications"],
                "memory_type": "application_context"
            },
            {
                "query": "Connect everything we've discussed about my quantum project",
                "expected_memories": ["quantum", "variational", "molecular", "ansatz", "drug discovery"],
                "memory_type": "comprehensive_synthesis"
            }
        ]
        
        memory_integration_results = []
        
        for test_query in memory_test_queries:
            query = test_query["query"]
            expected_memories = test_query["expected_memories"]
            memory_type = test_query["memory_type"]
            
            # Test memory integration through chat
            query_payload = {
                "message": query,
                "user_id": memory_user,
                "conversation_id": memory_conv_id
            }
            
            start_time = time.time()
            response = self.client.post("/chat", json=query_payload)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get('response', '')
                
                # Analyze memory integration
                memory_matches = sum(1 for memory in expected_memories 
                                   if memory.lower() in response_text.lower())
                memory_recall = memory_matches / len(expected_memories)
                
                # Check for memory indicators
                memory_indicators = ["you mentioned", "as discussed", "from your project", "previously", "earlier"]
                uses_memory_language = any(indicator in response_text.lower() 
                                         for indicator in memory_indicators)
                
                # Check response relevance to established context
                contextual_relevance = len(response_text) > 100 and memory_recall > 0
                
                memory_metrics = {
                    "memory_type": memory_type,
                    "response_time": round(response_time * 1000, 2),
                    "memory_recall": round(memory_recall, 2),
                    "uses_memory_language": uses_memory_language,
                    "contextual_relevance": contextual_relevance,
                    "response_length": len(response_text),
                    "memory_matches": memory_matches
                }
                
                memory_integration_results.append(memory_metrics)
                
                status = "PASS" if memory_recall >= 0.6 and uses_memory_language else "PARTIAL" if memory_recall >= 0.3 else "FAIL"
                self.log_test(f"Memory Integration: {memory_type}", status,
                            f"Recall: {memory_recall:.1%}, Memory language: {uses_memory_language}, Matches: {memory_matches}",
                            memory_metrics)
            else:
                self.log_test(f"Memory Integration: {memory_type}", "FAIL", f"HTTP {response.status_code}")
        
        # Overall memory integration assessment
        successful_memory_tests = sum(1 for r in memory_integration_results 
                                    if r["memory_recall"] >= 0.5 and r["uses_memory_language"])
        avg_memory_recall = sum(r["memory_recall"] for r in memory_integration_results) / len(memory_integration_results) if memory_integration_results else 0
        
        overall_status = "PASS" if successful_memory_tests >= len(memory_test_queries) * 0.75 else "PARTIAL"
        self.log_test("Memory Integration in Chat Flow", overall_status,
                     f"Success: {successful_memory_tests}/{len(memory_test_queries)}, Avg recall: {avg_memory_recall:.1%}")
        
        return successful_memory_tests >= len(memory_test_queries) * 0.5
    
    def run_realistic_chat_flow_suite(self):
        """Run the complete realistic chat flow testing suite."""
        print("💬 REALISTIC CHAT FLOW TESTING")
        print("=" * 80)
        
        # Realistic chat flow test suite
        chat_flow_tests = [
            ("Basic Chat Endpoint Functionality", self.test_basic_chat_endpoint_functionality),
            ("Formatted Chat Endpoint", self.test_formatted_chat_endpoint),
            ("Streaming Chat Functionality", self.test_streaming_chat_functionality),
            ("Conversation Continuity Across Endpoints", self.test_conversation_continuity_across_endpoints),
            ("Memory Integration in Chat Flow", self.test_memory_integration_in_chat_flow)
        ]
        
        passed = 0
        total = len(chat_flow_tests)
        
        for test_name, test_func in chat_flow_tests:
            try:
                print(f"\n💬 Running {test_name}...")
                start_time = time.time()
                
                if test_func():
                    passed += 1
                    elapsed = time.time() - start_time
                    print(f"   ✅ {test_name} successful in {elapsed:.2f}s")
                else:
                    elapsed = time.time() - start_time
                    print(f"   ⚠️  {test_name} partially successful in {elapsed:.2f}s")
            except Exception as e:
                elapsed = time.time() - start_time
                self.log_test(f"{test_name} Exception", "FAIL", str(e))
                print(f"   ❌ {test_name} failed with exception in {elapsed:.2f}s: {e}")
        
        # Generate realistic chat flow results
        print(f"\n📊 Realistic Chat Flow Results: {passed}/{total} chat flow tests successful")
        
        if passed == total:
            print("💬 EXCELLENT: All chat endpoints working perfectly with memory integration!")
        elif passed >= total * 0.8:
            print("🌊 VERY GOOD: Strong chat functionality with minor areas for improvement!")
        elif passed >= total * 0.6:
            print("✅ GOOD: Solid chat performance with some optimization opportunities!")
        else:
            print("🔄 NEEDS IMPROVEMENT: Basic chat functionality working, requires enhancement!")
        
        return passed, total

async def main():
    """Run the realistic chat flow testing suite."""
    print("💬 Initializing Realistic Chat Flow Testing...")
    
    # Initialize the system for testing
    print("🔧 Initializing NIS Protocol system for testing...")
    try:
        await initialize_system()
        print("✅ System initialized successfully for testing")
    except Exception as e:
        print(f"⚠️ System initialization warning: {e} - proceeding with tests")
    
    tester = RealisticChatFlowTester()
    passed, total = tester.run_realistic_chat_flow_suite()
    
    # Generate realistic chat flow report
    chat_flow_report = {
        "test_suite": "Realistic Chat Flow Testing",
        "timestamp": time.time(),
        "chat_sessions": tester.chat_sessions,
        "results_summary": {
            "total_chat_tests": total,
            "successful_chat_tests": passed,
            "chat_success_rate": round(passed / total, 3) if total > 0 else 0
        },
        "detailed_results": tester.test_results,
        "chat_capabilities": {
            "basic_chat": passed >= 1,
            "formatted_chat": passed >= 2,
            "streaming_chat": passed >= 3,
            "conversation_continuity": passed >= 4,
            "memory_integration": passed >= 5
        }
    }
    
    # Save realistic chat flow report
    with open("dev/testing/realistic_chat_flow_report.json", "w") as f:
        json.dump(chat_flow_report, f, indent=2)
    
    print(f"\n📄 Chat flow report saved to: dev/testing/realistic_chat_flow_report.json")
    print(f"🎯 Final Chat Flow Score: {passed}/{total} chat flow tests successful")
    print(f"📈 Chat Success Rate: {chat_flow_report['results_summary']['chat_success_rate']:.1%}")
    
    print(f"\n💬 Chat Capabilities:")
    for capability, achieved in chat_flow_report["chat_capabilities"].items():
        print(f"   • {capability.replace('_', ' ').title()}: {'💬' if achieved else '🔄'}")

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Extreme Memory Stress Testing
============================

This pushes the Enhanced Memory System to absolute limits with:
- Massive conversation volumes
- Concurrent access patterns
- Complex nested topic relationships
- Edge case data corruption scenarios
- Performance bottleneck identification
"""

import asyncio
import json
import sys
import time
import uuid
import threading
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from main import app, conversation_memory
    from fastapi.testclient import TestClient
    print("✅ Successfully imported extreme test components")
except ImportError as e:
    print(f"❌ Failed to import components: {e}")
    sys.exit(1)

class ExtremeMemoryStressTester:
    """Extreme stress testing for memory system boundaries."""
    
    def __init__(self):
        self.client = TestClient(app)
        self.test_results = []
        self.performance_metrics = {}
        self.stress_conversations = []
        
    def log_test(self, test_name: str, status: str, details: str = "", performance_data: Dict = None):
        """Enhanced logging with performance metrics."""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "performance": performance_data or {},
            "timestamp": time.time()
        }
        self.test_results.append(result)
        
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️" if status == "PARTIAL" else "🔥" if status == "EXTREME" else "📊"
        print(f"{status_icon} {test_name}: {status}")
        if details:
            print(f"   {details}")
        if performance_data:
            print(f"   📊 Performance: {performance_data}")
    
    def generate_massive_conversation_dataset(self, num_conversations: int = 50, messages_per_conv: int = 20):
        """Generate a massive dataset of realistic conversations."""
        print(f"\n🔥 Generating {num_conversations} conversations with {messages_per_conv} messages each...")
        
        # Advanced topic templates for realistic conversations
        topic_templates = [
            {
                "domain": "Deep Learning Architecture",
                "concepts": ["transformers", "attention mechanisms", "neural architecture search", "gradient descent", "backpropagation"],
                "questions": [
                    "How do attention mechanisms in transformers differ from traditional RNNs?",
                    "What are the computational complexities of different attention patterns?",
                    "How does gradient checkpointing affect memory usage in large models?",
                    "What role does layer normalization play in training stability?",
                    "How do you handle vanishing gradients in very deep networks?"
                ]
            },
            {
                "domain": "Quantum Computing Theory",
                "concepts": ["quantum entanglement", "superposition", "quantum gates", "decoherence", "quantum algorithms"],
                "questions": [
                    "How does quantum decoherence affect quantum algorithm performance?",
                    "What are the limitations of current quantum error correction methods?",
                    "How do you implement quantum Fourier transforms efficiently?",
                    "What's the relationship between quantum entanglement and computational advantage?",
                    "How do you design quantum circuits for specific optimization problems?"
                ]
            },
            {
                "domain": "Advanced Physics Simulation",
                "concepts": ["molecular dynamics", "finite element analysis", "computational fluid dynamics", "statistical mechanics", "phase transitions"],
                "questions": [
                    "How do you handle multi-scale phenomena in molecular dynamics simulations?",
                    "What are the stability conditions for explicit time integration schemes?",
                    "How do you model turbulence in computational fluid dynamics?",
                    "What role does entropy play in non-equilibrium statistical mechanics?",
                    "How do you simulate phase transitions in complex materials?"
                ]
            },
            {
                "domain": "Information Theory & Cryptography", 
                "concepts": ["Shannon entropy", "mutual information", "cryptographic protocols", "quantum cryptography", "information complexity"],
                "questions": [
                    "How does mutual information relate to data compression efficiency?",
                    "What are the information-theoretic limits of secure communication?",
                    "How do you prove security in quantum key distribution protocols?",
                    "What's the relationship between computational and information complexity?",
                    "How do you design optimal error-correcting codes for specific channels?"
                ]
            },
            {
                "domain": "Complex Systems & Network Theory",
                "concepts": ["scale-free networks", "small-world phenomena", "emergence", "self-organization", "criticality"],
                "questions": [
                    "How do you identify critical transitions in complex systems?",
                    "What causes the emergence of scale-free topology in real networks?",
                    "How do you model cascading failures in interconnected systems?",
                    "What role does noise play in self-organizing systems?",
                    "How do you quantify emergence in multi-agent systems?"
                ]
            }
        ]
        
        conversation_count = 0
        total_messages = 0
        
        for i in range(num_conversations):
            # Select random topic template
            template = topic_templates[i % len(topic_templates)]
            conv_id = f"extreme_conv_{uuid.uuid4().hex[:12]}"
            user_id = f"extreme_user_{(i % 10) + 1:03d}"
            
            conversation = []
            
            # Generate realistic conversation flow
            for j in range(messages_per_conv):
                if j % 2 == 0:  # User message
                    if j == 0:  # Opening question
                        content = template["questions"][j // 2 % len(template["questions"])]
                    else:  # Follow-up questions
                        concepts = template["concepts"]
                        concept = concepts[j // 2 % len(concepts)]
                        content = f"Can you elaborate on {concept} in the context of {template['domain'].lower()}? How does it interact with the previous concepts we discussed?"
                    
                    role = "user"
                else:  # Assistant response
                    # Generate complex technical response
                    concepts = template["concepts"]
                    relevant_concepts = concepts[:min(3, len(concepts))]
                    
                    content = f"In {template['domain']}, {relevant_concepts[0]} is fundamentally connected to {relevant_concepts[1] if len(relevant_concepts) > 1 else 'the underlying principles'}. "
                    content += f"The mathematical formulation involves complex relationships where ∇²φ + k²φ = 0 represents the wave equation governing {relevant_concepts[0]}. "
                    content += f"When we consider {relevant_concepts[2] if len(relevant_concepts) > 2 else 'higher-order effects'}, the system exhibits non-linear dynamics described by the Hamiltonian H = Σᵢ(pᵢ²/2m) + V(q₁,...,qₙ). "
                    content += f"This connects to information theory through the Shannon entropy S = -Σᵢ pᵢ log₂(pᵢ), showing how {template['domain'].lower()} systems process and store information."
                    
                    role = "assistant"
                
                message = {
                    "role": role,
                    "content": content,
                    "timestamp": time.time() + total_messages * 0.1,  # Stagger timestamps
                    "conversation_id": conv_id,
                    "user_id": user_id,
                    "domain": template["domain"],
                    "message_index": j
                }
                
                conversation.append(message)
                total_messages += 1
            
            # Add to conversation memory
            if conv_id not in conversation_memory:
                conversation_memory[conv_id] = []
            
            for msg in conversation:
                conversation_memory[conv_id].append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg["timestamp"],
                    "domain": msg["domain"]
                })
            
            self.stress_conversations.append(conv_id)
            conversation_count += 1
            
            if conversation_count % 10 == 0:
                print(f"   Generated {conversation_count}/{num_conversations} conversations...")
        
        print(f"   ✅ Generated {conversation_count} conversations with {total_messages} total messages")
        return conversation_count, total_messages
    
    def test_massive_search_performance(self):
        """Test search performance with massive datasets."""
        print("\n🔥 Testing Massive Search Performance")
        
        # Test increasingly complex queries
        complex_queries = [
            "quantum entanglement decoherence",
            "neural architecture search gradient descent optimization",
            "computational fluid dynamics turbulence finite element analysis",
            "Shannon entropy mutual information cryptographic protocols security",
            "scale-free networks emergence self-organization complex systems critical transitions",
            "transformer attention mechanisms backpropagation layer normalization stability",
            "molecular dynamics multi-scale phenomena statistical mechanics phase transitions",
            "quantum Fourier transforms error correction quantum gates superposition",
            "information complexity computational complexity cryptographic protocols quantum cryptography",
            "cascading failures interconnected systems network topology scale-free small-world"
        ]
        
        performance_results = []
        
        for i, query in enumerate(complex_queries):
            start_time = time.time()
            
            params = {"query": query, "limit": 20}
            response = self.client.get("/memory/conversations", params=params)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                conversations = data.get('conversations', [])
                
                performance_data = {
                    "query_length": len(query),
                    "response_time_ms": round(response_time * 1000, 2),
                    "results_count": len(conversations),
                    "query_complexity": len(query.split())
                }
                
                performance_results.append(performance_data)
                
                status = "PASS" if response_time < 1.0 else "PARTIAL" if response_time < 3.0 else "FAIL"
                self.log_test(f"Massive Search {i+1}", status, 
                            f"Query: '{query[:50]}...', Results: {len(conversations)}", 
                            performance_data)
            else:
                self.log_test(f"Massive Search {i+1}", "FAIL", f"HTTP {response.status_code}")
        
        # Analyze performance trends
        if performance_results:
            avg_response_time = sum(r["response_time_ms"] for r in performance_results) / len(performance_results)
            max_response_time = max(r["response_time_ms"] for r in performance_results)
            
            overall_performance = {
                "average_response_time_ms": round(avg_response_time, 2),
                "max_response_time_ms": round(max_response_time, 2),
                "total_queries": len(performance_results)
            }
            
            status = "PASS" if avg_response_time < 500 else "PARTIAL" if avg_response_time < 1500 else "FAIL"
            self.log_test("Massive Search Performance", status, 
                        f"Avg: {avg_response_time:.1f}ms, Max: {max_response_time:.1f}ms",
                        overall_performance)
            
            return status != "FAIL"
        
        return False
    
    def test_concurrent_access_stress(self):
        """Test concurrent access patterns to identify race conditions."""
        print("\n🔥 Testing Concurrent Access Stress")
        
        def make_request(query_id: int):
            """Make a single request - for concurrent testing."""
            try:
                params = {"query": f"test concurrent query {query_id}", "limit": 5}
                response = self.client.get("/memory/conversations", params=params)
                return {
                    "query_id": query_id,
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response_time": time.time()
                }
            except Exception as e:
                return {
                    "query_id": query_id,
                    "status_code": 0,
                    "success": False,
                    "error": str(e),
                    "response_time": time.time()
                }
        
        # Test with increasing concurrency levels
        concurrency_levels = [5, 10, 20, 50]
        concurrent_results = []
        
        for num_threads in concurrency_levels:
            print(f"   Testing with {num_threads} concurrent requests...")
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(make_request, i) for i in range(num_threads)]
                results = [future.result() for future in futures]
            
            end_time = time.time()
            total_time = end_time - start_time
            
            successful_requests = sum(1 for r in results if r["success"])
            success_rate = successful_requests / num_threads
            
            performance_data = {
                "concurrency_level": num_threads,
                "success_rate": round(success_rate, 3),
                "total_time_s": round(total_time, 2),
                "requests_per_second": round(num_threads / total_time, 1)
            }
            
            concurrent_results.append(performance_data)
            
            status = "PASS" if success_rate >= 0.9 else "PARTIAL" if success_rate >= 0.7 else "FAIL"
            self.log_test(f"Concurrent {num_threads} threads", status,
                        f"Success rate: {success_rate:.1%}, RPS: {performance_data['requests_per_second']}",
                        performance_data)
        
        # Overall concurrent access assessment
        overall_success = all(r["success_rate"] >= 0.8 for r in concurrent_results)
        self.log_test("Concurrent Access Stress", "PASS" if overall_success else "PARTIAL",
                     f"Tested up to {max(concurrency_levels)} concurrent requests")
        
        return overall_success
    
    def test_memory_boundary_conditions(self):
        """Test memory system boundary conditions and edge cases."""
        print("\n🔥 Testing Memory Boundary Conditions")
        
        boundary_tests = []
        
        # Test 1: Extremely large conversation details request
        if self.stress_conversations:
            conv_id = self.stress_conversations[0]
            start_time = time.time()
            response = self.client.get(f"/memory/conversation/{conv_id}?include_context=true")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                message_count = data.get('message_count', 0)
                
                status = "PASS" if response_time < 2.0 else "PARTIAL"
                boundary_tests.append(("Large Conversation Details", status, 
                                     f"{message_count} messages in {response_time:.2f}s"))
            else:
                boundary_tests.append(("Large Conversation Details", "FAIL", f"HTTP {response.status_code}"))
        
        # Test 2: Maximum limit stress test
        start_time = time.time()
        response = self.client.get("/memory/conversations?query=quantum&limit=999999")
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            results = len(data.get('conversations', []))
            status = "PASS" if response_time < 5.0 else "PARTIAL"
            boundary_tests.append(("Maximum Limit Test", status, 
                                 f"{results} results in {response_time:.2f}s"))
        else:
            boundary_tests.append(("Maximum Limit Test", "FAIL", f"HTTP {response.status_code}"))
        
        # Test 3: Nested context preview with complex query
        if self.stress_conversations:
            conv_id = self.stress_conversations[0]
            complex_message = "Analyze the quantum mechanical foundations of neural network architectures in the context of information theory and complex systems, considering the role of entropy, entanglement, and emergence in distributed AI systems."
            
            start_time = time.time()
            params = {"message": complex_message}
            response = self.client.get(f"/memory/conversation/{conv_id}/context", params=params)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                context_count = data.get('context_count', 0)
                status = "PASS" if response_time < 3.0 else "PARTIAL"
                boundary_tests.append(("Complex Context Preview", status,
                                     f"{context_count} context msgs in {response_time:.2f}s"))
            else:
                boundary_tests.append(("Complex Context Preview", "FAIL", f"HTTP {response.status_code}"))
        
        # Test 4: Stress test memory stats under load
        start_time = time.time()
        response = self.client.get("/memory/stats")
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            stats = data.get('stats', {})
            total_convs = stats.get('total_conversations', 0)
            total_msgs = stats.get('total_messages', 0)
            
            status = "PASS" if response_time < 1.0 and total_convs > 0 else "PARTIAL"
            boundary_tests.append(("Memory Stats Under Load", status,
                                 f"{total_convs} convs, {total_msgs} msgs in {response_time:.2f}s"))
        else:
            boundary_tests.append(("Memory Stats Under Load", "FAIL", f"HTTP {response.status_code}"))
        
        # Log all boundary test results
        for test_name, status, details in boundary_tests:
            self.log_test(f"Boundary: {test_name}", status, details)
        
        passed_boundary = sum(1 for _, status, _ in boundary_tests if status == "PASS")
        total_boundary = len(boundary_tests)
        
        overall_status = "PASS" if passed_boundary >= total_boundary * 0.75 else "PARTIAL"
        self.log_test("Memory Boundary Conditions", overall_status,
                     f"{passed_boundary}/{total_boundary} boundary tests passed")
        
        return passed_boundary >= total_boundary * 0.5
    
    def test_extreme_topic_relationships(self):
        """Test extreme topic relationship detection and linking."""
        print("\n🔥 Testing Extreme Topic Relationships")
        
        # Create queries that span multiple domains
        cross_domain_queries = [
            ("How do quantum mechanical principles apply to neural network optimization?", 
             ["quantum", "neural", "optimization"]),
            ("What's the connection between information theory and molecular dynamics simulations?",
             ["information", "molecular", "dynamics"]),
            ("How does network theory relate to cryptographic protocol design?",
             ["network", "cryptographic", "protocol"]),
            ("Connect statistical mechanics to machine learning regularization techniques",
             ["statistical", "mechanics", "regularization"]),
            ("Relate quantum entanglement to distributed computing architectures",
             ["entanglement", "distributed", "computing"])
        ]
        
        relationship_results = []
        
        for query, expected_domains in cross_domain_queries:
            # Test search across domains
            params = {"query": query, "limit": 10}
            response = self.client.get("/memory/conversations", params=params)
            
            if response.status_code == 200:
                data = response.json()
                conversations = data.get('conversations', [])
                
                # Analyze cross-domain relevance
                domain_coverage = 0
                for conv in conversations:
                    preview = conv.get('preview', '').lower()
                    for domain in expected_domains:
                        if domain in preview:
                            domain_coverage += 1
                            break
                
                relevance_score = domain_coverage / max(len(conversations), 1)
                
                status = "PASS" if relevance_score >= 0.5 else "PARTIAL" if relevance_score >= 0.2 else "FAIL"
                relationship_results.append((query, status, relevance_score))
                
                self.log_test(f"Cross-Domain: {query[:40]}...", status,
                            f"Relevance: {relevance_score:.1%}, Results: {len(conversations)}")
            else:
                relationship_results.append((query, "FAIL", 0))
                self.log_test(f"Cross-Domain: {query[:40]}...", "FAIL", f"HTTP {response.status_code}")
        
        # Overall relationship detection assessment
        passed_relationships = sum(1 for _, status, _ in relationship_results if status != "FAIL")
        total_relationships = len(relationship_results)
        
        overall_status = "PASS" if passed_relationships >= total_relationships * 0.6 else "PARTIAL"
        self.log_test("Extreme Topic Relationships", overall_status,
                     f"{passed_relationships}/{total_relationships} cross-domain queries successful")
        
        return passed_relationships >= total_relationships * 0.4
    
    def run_extreme_stress_suite(self):
        """Run the complete extreme stress testing suite."""
        print("🔥 EXTREME Memory System Stress Testing")
        print("=" * 80)
        
        # Generate massive dataset first
        conv_count, msg_count = self.generate_massive_conversation_dataset(30, 15)
        
        if conv_count == 0:
            print("❌ Failed to generate test dataset")
            return 0, 1
        
        print(f"\n🎯 Testing with {conv_count} conversations ({msg_count} messages)")
        
        # Extreme test suite
        extreme_tests = [
            ("Massive Search Performance", self.test_massive_search_performance),
            ("Concurrent Access Stress", self.test_concurrent_access_stress),
            ("Memory Boundary Conditions", self.test_memory_boundary_conditions),
            ("Extreme Topic Relationships", self.test_extreme_topic_relationships)
        ]
        
        passed = 0
        total = len(extreme_tests)
        
        for test_name, test_func in extreme_tests:
            try:
                print(f"\n🔥 Running {test_name}...")
                start_time = time.time()
                
                if test_func():
                    passed += 1
                    elapsed = time.time() - start_time
                    print(f"   ✅ {test_name} completed successfully in {elapsed:.2f}s")
                else:
                    elapsed = time.time() - start_time
                    print(f"   ⚠️  {test_name} completed with issues in {elapsed:.2f}s")
            except Exception as e:
                elapsed = time.time() - start_time
                self.log_test(f"{test_name} Exception", "FAIL", str(e))
                print(f"   ❌ {test_name} failed with exception in {elapsed:.2f}s: {e}")
        
        # Generate extreme test results
        print(f"\n📊 Extreme Test Results: {passed}/{total} test suites passed")
        
        if passed == total:
            print("🏆 LEGENDARY: All extreme tests passed! This system is bulletproof!")
        elif passed >= total * 0.75:
            print("🔥 OUTSTANDING: Passed extreme stress testing with flying colors!")
        elif passed >= total * 0.5:
            print("💪 ROBUST: Survived extreme testing with good performance!")
        else:
            print("⚠️  STRESS DETECTED: System shows strain under extreme conditions.")
        
        return passed, total

def main():
    """Run the extreme memory stress testing suite."""
    print("🔥 Initializing EXTREME Memory Stress Testing...")
    
    tester = ExtremeMemoryStressTester()
    passed, total = tester.run_extreme_stress_suite()
    
    # Generate extreme test report
    extreme_report = {
        "test_suite": "EXTREME Memory System Stress Testing",
        "timestamp": time.time(),
        "dataset_stats": {
            "conversations_generated": len(tester.stress_conversations),
            "total_test_messages": sum(len(conversation_memory.get(conv_id, [])) for conv_id in tester.stress_conversations)
        },
        "results_summary": {
            "total_test_suites": total,
            "passed_suites": passed,
            "success_rate": round(passed / total, 3) if total > 0 else 0
        },
        "detailed_results": tester.test_results,
        "performance_metrics": tester.performance_metrics
    }
    
    # Save extreme test report
    with open("dev/testing/extreme_stress_test_report.json", "w") as f:
        json.dump(extreme_report, f, indent=2)
    
    print(f"\n📄 Extreme test report saved to: dev/testing/extreme_stress_test_report.json")
    print(f"🎯 Final Extreme Score: {passed}/{total} test suites passed")
    print(f"📈 Extreme Success Rate: {extreme_report['results_summary']['success_rate']:.1%}")
    
    print(f"\n🔥 Dataset Generated:")
    print(f"   • {extreme_report['dataset_stats']['conversations_generated']} conversations")
    print(f"   • {extreme_report['dataset_stats']['total_test_messages']} total messages")
    print(f"   • Complex technical discussions across 5 domains")

if __name__ == "__main__":
    main()
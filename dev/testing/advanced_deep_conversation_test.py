#!/usr/bin/env python3
"""
Advanced Deep Conversation Testing Suite
=======================================

This suite pushes the Enhanced Memory System to its limits with:
- Complex multi-session conversations
- Deep technical discussions
- Cross-topic connections
- Memory stress testing
- Edge cases and boundary conditions
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
    from main import app, conversation_memory
    from fastapi.testclient import TestClient
    print("✅ Successfully imported advanced test components")
except ImportError as e:
    print(f"❌ Failed to import components: {e}")
    sys.exit(1)

class AdvancedDeepTester:
    """Advanced testing suite for deep conversations and memory stress testing."""
    
    def __init__(self):
        self.client = TestClient(app)
        self.test_results = []
        self.conversation_sessions = {}
        self.user_ids = [
            "deep_researcher_001",
            "technical_expert_002", 
            "curious_student_003"
        ]
        self.advanced_topics = []
    
    def log_test(self, test_name: str, status: str, details: str = "", metadata: Dict = None):
        """Enhanced logging with metadata."""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        self.test_results.append(result)
        
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️" if status == "PARTIAL" else "🔄" if status == "PROCESSING" else "📊"
        print(f"{status_icon} {test_name}: {status}")
        if details:
            print(f"   {details}")
    
    def create_complex_conversation_scenarios(self):
        """Create complex multi-session conversation scenarios."""
        print("\n🔬 Creating Complex Conversation Scenarios...")
        
        # Scenario 1: Deep Technical Architecture Discussion
        tech_conv_id = f"tech_deep_{self.user_ids[0]}_{uuid.uuid4().hex[:8]}"
        tech_messages = [
            ("user", "I need a comprehensive explanation of the NIS Protocol v3 architecture, specifically focusing on the mathematical foundations of the Laplace→KAN→PINN pipeline"),
            ("assistant", "The NIS Protocol v3 implements a sophisticated three-layer mathematical processing pipeline. The Laplace transform layer converts time-domain signals f(t) into the s-domain F(s) = ∫₀^∞ f(t)e^(-st)dt, enabling frequency analysis and noise filtering. This feeds into KAN (Kolmogorov-Arnold Networks) layers that use learnable spline-based activation functions instead of fixed activations, providing inherent interpretability through mathematical function approximation."),
            ("user", "Elaborate on how the spline-based activation functions in KAN layers differ from traditional ReLU or sigmoid activations, and why this matters for interpretability"),
            ("assistant", "Traditional activations like ReLU (max(0,x)) or sigmoid (1/(1+e^(-x))) are fixed functions applied element-wise. KAN layers instead use learnable univariate splines φ(x) = Σᵢ cᵢ Bᵢ(x) where Bᵢ are B-spline basis functions and cᵢ are learnable coefficients. This allows each neuron to learn arbitrary smooth functions rather than being constrained to linear combinations of fixed activations."),
            ("user", "Now connect this to the PINN layer - how does physics-informed neural networking validate and correct the KAN reasoning outputs?"),
            ("assistant", "The PINN (Physics-Informed Neural Networks) layer incorporates physical laws as soft constraints by adding physics-based loss terms to the optimization. For NIS Protocol, this means the KAN outputs must satisfy conservation laws, thermodynamic principles, and causal relationships. The PINN loss function becomes L_total = L_data + λ_physics * L_physics where L_physics = ||∇·F - ρ||² + ||∂u/∂t + u·∇u + ∇p||² enforcing continuity and momentum conservation."),
            ("user", "What happens when the KAN reasoning conflicts with physics constraints? How does the auto-correction mechanism work?"),
            ("assistant", "When conflicts arise, the PINN layer uses a gradient-based correction mechanism. It computes ∂L_physics/∂θ_KAN and propagates corrections back to KAN parameters. If physics violations exceed threshold τ, the system triggers auto-correction: θ_KAN ← θ_KAN - α∇L_physics, where α is adaptively adjusted. For severe violations, the system can reject KAN outputs entirely and fall back to physics-compliant approximations.")
        ]
        
        # Scenario 2: Cross-Domain Knowledge Integration
        knowledge_conv_id = f"knowledge_{self.user_ids[1]}_{uuid.uuid4().hex[:8]}"
        knowledge_messages = [
            ("user", "How does the NIS Protocol's memory system relate to biological neural plasticity and synaptic consolidation?"),
            ("assistant", "The NIS enhanced memory system mirrors several biological mechanisms: like Long-Term Potentiation (LTP), frequently accessed conversation topics increase in importance score through repeated activation. The semantic search resembles how the hippocampus creates associative links between related memories. Memory consolidation in NIS occurs periodically, similar to how sleep consolidates memories from hippocampus to neocortex."),
            ("user", "Can you connect this to information theory - what's the relationship between semantic similarity thresholds and Shannon entropy in conversation space?"),
            ("assistant", "Excellent connection! The semantic similarity threshold acts as an information filter. In conversation space, high-entropy (diverse) topics require lower similarity thresholds to maintain coherence, while low-entropy (focused) discussions can use higher thresholds. The Shannon entropy H(C) = -Σp(topic_i)log₂p(topic_i) of conversation C guides adaptive threshold adjustment: threshold = θ_base + β*H(C)."),
            ("user", "How would you extend this to handle quantum information processing concepts in future NIS versions?"),
            ("assistant", "Fascinating direction! Quantum-enhanced NIS could leverage superposition for parallel conversation context exploration, entanglement for instantaneous cross-conversation correlation updates, and quantum memory for exponentially larger context spaces. The semantic similarity could become quantum amplitude: |ψ⟩ = Σᵢαᵢ|context_i⟩ where quantum interference enhances relevant contexts and destructively interferes with irrelevant ones.")
        ]
        
        # Scenario 3: Rapid Context Switching and Memory Stress
        stress_conv_id = f"stress_{self.user_ids[2]}_{uuid.uuid4().hex[:8]}"
        stress_messages = [
            ("user", "Switch between these topics rapidly: 1) Explain transformer attention mechanisms"),
            ("assistant", "Transformer attention uses scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V where Q,K,V are query, key, value matrices. Multi-head attention runs h parallel attention functions: MultiHead(Q,K,V) = Concat(head₁,...,head_h)W^O where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)."),
            ("user", "2) Now discuss quantum entanglement in Bell states"),
            ("assistant", "Bell states are maximally entangled two-qubit states: |Φ⁺⟩ = (|00⟩ + |11⟩)/√2, |Φ⁻⟩ = (|00⟩ - |11⟩)/√2, |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2, |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2. They violate Bell's inequality: S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')| > 2 for quantum systems."),
            ("user", "3) Back to transformers - how do they relate to the NIS KAN layers?"),
            ("assistant", "Brilliant connection! Both use learned attention/activation patterns, but KAN's spline-based functions are more interpretable than transformer's dense attention matrices. We could design KAN-formers: replacing transformer feedforward layers with KAN layers for inherent interpretability while maintaining attention mechanisms for sequence processing."),
            ("user", "4) Connect quantum entanglement to distributed AI systems"),
            ("assistant", "Quantum-inspired distributed AI could use entanglement-like correlations for instant parameter synchronization across nodes. If node A updates parameters θ_A, entangled node B instantly reflects this: |ψ_system⟩ = α|θ_A, θ_B⟩ + β|θ_A', θ_B'⟩. This could eliminate communication overhead in federated learning.")
        ]
        
        # Store scenarios
        self.conversation_sessions = {
            "technical_deep": (tech_conv_id, tech_messages),
            "knowledge_integration": (knowledge_conv_id, knowledge_messages),
            "context_switching": (stress_conv_id, stress_messages)
        }
        
        print(f"   ✅ Created {len(self.conversation_sessions)} complex scenarios")
        return True
    
    def populate_conversation_memory(self):
        """Populate conversation memory with complex scenarios."""
        print("\n🔄 Populating Conversation Memory...")
        
        total_messages = 0
        for scenario_name, (conv_id, messages) in self.conversation_sessions.items():
            if conv_id not in conversation_memory:
                conversation_memory[conv_id] = []
            
            for role, content in messages:
                message = {
                    "role": role,
                    "content": content,
                    "timestamp": time.time() + total_messages,  # Stagger timestamps
                    "scenario": scenario_name
                }
                conversation_memory[conv_id].append(message)
                total_messages += 1
        
        print(f"   ✅ Added {total_messages} complex messages across {len(self.conversation_sessions)} conversations")
        return True
    
    def test_complex_semantic_search(self):
        """Test semantic search with complex technical queries."""
        print("\n🧪 Testing Complex Semantic Search")
        
        complex_queries = [
            ("Laplace transform mathematical foundations", "Should find technical architecture discussion"),
            ("spline-based activation functions vs ReLU", "Should find KAN layer explanations"),
            ("physics-informed neural networks PINN validation", "Should find physics constraint discussions"),
            ("quantum entanglement Bell states violation", "Should find quantum mechanics content"),
            ("transformer attention mechanisms scaled dot-product", "Should find transformer explanations"),
            ("biological neural plasticity synaptic consolidation", "Should find bio-AI connections"),
            ("Shannon entropy information theory conversation space", "Should find information theory content"),
            ("quantum information processing superposition entanglement", "Should find quantum AI concepts"),
            ("federated learning parameter synchronization distributed", "Should find distributed AI content"),
            ("nonexistent_quantum_spline_transformer_biology", "Should return limited results")
        ]
        
        passed = 0
        total = len(complex_queries)
        
        for query, expected_description in complex_queries:
            params = {
                "query": query,
                "user_id": self.user_ids[0],
                "limit": 10
            }
            
            response = self.client.get("/memory/conversations", params=params)
            
            if response.status_code == 200:
                data = response.json()
                conversations = data.get('conversations', [])
                
                # Analyze result quality
                if "nonexistent" in query:
                    # Should return few results
                    if len(conversations) <= 1:
                        self.log_test(f"Complex Search: {query[:30]}...", "PASS", 
                                    f"Correctly filtered - {len(conversations)} results")
                        passed += 1
                    else:
                        self.log_test(f"Complex Search: {query[:30]}...", "PARTIAL",
                                    f"Expected few results, got {len(conversations)}")
                else:
                    # Should find relevant technical content
                    if len(conversations) > 0:
                        # Check if results seem relevant
                        relevant_count = 0
                        for conv in conversations:
                            preview = conv.get('preview', '').lower()
                            if any(term in preview for term in query.lower().split()[:3]):
                                relevant_count += 1
                        
                        if relevant_count > 0:
                            self.log_test(f"Complex Search: {query[:30]}...", "PASS",
                                        f"Found {len(conversations)} convs, {relevant_count} relevant")
                            passed += 1
                        else:
                            self.log_test(f"Complex Search: {query[:30]}...", "PARTIAL",
                                        f"Found {len(conversations)} convs but relevance unclear")
                    else:
                        self.log_test(f"Complex Search: {query[:30]}...", "FAIL",
                                    "No results for technical query")
            else:
                self.log_test(f"Complex Search: {query[:30]}...", "FAIL",
                            f"HTTP {response.status_code}")
        
        success_rate = passed / total
        overall_status = "PASS" if success_rate >= 0.8 else "PARTIAL" if success_rate >= 0.6 else "FAIL"
        self.log_test("Complex Semantic Search", overall_status,
                     f"{passed}/{total} queries successful ({success_rate:.1%})")
        
        return success_rate >= 0.6
    
    def test_cross_conversation_context_linking(self):
        """Test context linking between related conversations."""
        print("\n🧪 Testing Cross-Conversation Context Linking")
        
        # Test queries that should link concepts across conversations
        linking_tests = [
            ("How do KAN layers connect to quantum computing?", ["KAN", "quantum", "spline"]),
            ("Relate transformer attention to physics constraints", ["attention", "physics", "transformer"]),
            ("Connect biological plasticity to information theory", ["biological", "plasticity", "Shannon"]),
            ("How does entanglement apply to distributed AI?", ["entanglement", "distributed", "quantum"]),
            ("Integrate PINN validation with attention mechanisms", ["PINN", "validation", "attention"])
        ]
        
        passed = 0
        total = len(linking_tests)
        
        for test_query, expected_concepts in linking_tests:
            # Use context preview to see what context would be pulled
            # Test with each conversation
            context_found = False
            for scenario_name, (conv_id, _) in self.conversation_sessions.items():
                params = {"message": test_query}
                response = self.client.get(f"/memory/conversation/{conv_id}/context", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    context_messages = data.get('context_messages', [])
                    
                    # Check if context includes cross-conversation content
                    concepts_found = []
                    for msg in context_messages:
                        content = msg.get('content', '').lower()
                        source = msg.get('source', '')
                        
                        for concept in expected_concepts:
                            if concept.lower() in content:
                                concepts_found.append(concept)
                        
                        # Check if we have cross-conversation context
                        if source == "semantic_context":
                            context_found = True
                    
                    unique_concepts = list(set(concepts_found))
                    if len(unique_concepts) >= 2:  # Found multiple related concepts
                        self.log_test(f"Context Link: {test_query[:40]}...", "PASS",
                                    f"Found {len(unique_concepts)} concepts: {unique_concepts}")
                        passed += 1
                        break
            
            if not context_found:
                self.log_test(f"Context Link: {test_query[:40]}...", "PARTIAL",
                            "No cross-conversation context detected")
        
        success_rate = passed / total
        overall_status = "PASS" if success_rate >= 0.6 else "PARTIAL" if success_rate >= 0.4 else "FAIL"
        self.log_test("Cross-Conversation Linking", overall_status,
                     f"{passed}/{total} linking tests successful ({success_rate:.1%})")
        
        return success_rate >= 0.4
    
    def test_memory_stress_scenarios(self):
        """Stress test the memory system with edge cases."""
        print("\n🧪 Testing Memory Stress Scenarios")
        
        stress_tests = []
        
        # Test 1: Very long query
        very_long_query = "quantum " * 100 + "transformer " * 100 + "physics " * 100
        params = {"query": very_long_query, "limit": 5}
        response = self.client.get("/memory/conversations", params=params)
        
        if response.status_code == 200:
            stress_tests.append(("Very Long Query", "PASS", "Handled gracefully"))
        else:
            stress_tests.append(("Very Long Query", "FAIL", f"HTTP {response.status_code}"))
        
        # Test 2: Special characters and edge cases
        special_queries = [
            "SELECT * FROM conversations",  # SQL injection attempt
            "<script>alert('test')</script>",  # XSS attempt
            "../../etc/passwd",  # Path traversal attempt
            "∫∂∇∞≠±×÷√∑∏",  # Mathematical symbols
            "🧠🔬⚡🚀💡🎯",  # Emojis
            "",  # Empty query
            " ",  # Whitespace only
            "a",  # Single character
        ]
        
        special_passed = 0
        for query in special_queries:
            params = {"query": query, "limit": 3}
            response = self.client.get("/memory/conversations", params=params)
            
            if response.status_code in [200, 400, 422]:  # Acceptable responses
                special_passed += 1
        
        special_rate = special_passed / len(special_queries)
        status = "PASS" if special_rate >= 0.8 else "PARTIAL" if special_rate >= 0.6 else "FAIL"
        stress_tests.append(("Special Characters", status, f"{special_passed}/{len(special_queries)} handled"))
        
        # Test 3: Rapid consecutive requests
        rapid_passed = 0
        rapid_total = 20
        start_time = time.time()
        
        for i in range(rapid_total):
            params = {"query": f"test query {i}", "limit": 2}
            response = self.client.get("/memory/conversations", params=params)
            if response.status_code == 200:
                rapid_passed += 1
        
        elapsed = time.time() - start_time
        rapid_rate = rapid_passed / rapid_total
        rapid_status = "PASS" if rapid_rate >= 0.9 else "PARTIAL" if rapid_rate >= 0.7 else "FAIL"
        stress_tests.append(("Rapid Requests", rapid_status, 
                           f"{rapid_passed}/{rapid_total} in {elapsed:.2f}s"))
        
        # Test 4: Large limit values
        large_limits = [100, 1000, 10000]
        limit_passed = 0
        
        for limit in large_limits:
            params = {"query": "test", "limit": limit}
            response = self.client.get("/memory/conversations", params=params)
            if response.status_code == 200:
                data = response.json()
                conversations = data.get('conversations', [])
                # Should handle gracefully, not crash
                limit_passed += 1
        
        limit_rate = limit_passed / len(large_limits)
        limit_status = "PASS" if limit_rate >= 0.8 else "PARTIAL" if limit_rate >= 0.6 else "FAIL"
        stress_tests.append(("Large Limits", limit_status, f"{limit_passed}/{len(large_limits)} handled"))
        
        # Log all stress test results
        for test_name, status, details in stress_tests:
            self.log_test(f"Stress: {test_name}", status, details)
        
        # Overall stress test evaluation
        passed_stress = sum(1 for _, status, _ in stress_tests if status == "PASS")
        total_stress = len(stress_tests)
        
        overall_status = "PASS" if passed_stress >= total_stress * 0.75 else "PARTIAL"
        self.log_test("Memory Stress Testing", overall_status,
                     f"{passed_stress}/{total_stress} stress tests passed")
        
        return passed_stress >= total_stress * 0.5
    
    def test_advanced_memory_analytics(self):
        """Test advanced memory analytics and insights."""
        print("\n🧪 Testing Advanced Memory Analytics")
        
        analytics_tests = []
        
        # Test detailed memory stats
        response = self.client.get("/memory/stats")
        if response.status_code == 200:
            data = response.json()
            stats = data.get('stats', {})
            
            # Analyze the depth of statistics provided
            expected_metrics = ['total_conversations', 'total_messages', 'enhanced_memory_enabled']
            found_metrics = [metric for metric in expected_metrics if metric in stats]
            
            if len(found_metrics) == len(expected_metrics):
                analytics_tests.append(("Stats Completeness", "PASS", f"All {len(expected_metrics)} metrics"))
            else:
                analytics_tests.append(("Stats Completeness", "PARTIAL", f"{len(found_metrics)}/{len(expected_metrics)} metrics"))
        else:
            analytics_tests.append(("Stats Completeness", "FAIL", f"HTTP {response.status_code}"))
        
        # Test topics discovery depth
        response = self.client.get("/memory/topics?limit=50")
        if response.status_code == 200:
            data = response.json()
            topics = data.get('topics', [])
            total_topics = data.get('total_topics', 0)
            
            # Even if no topics auto-generated yet, endpoint should work
            analytics_tests.append(("Topics Discovery", "PASS", f"{len(topics)} topics available"))
        else:
            analytics_tests.append(("Topics Discovery", "FAIL", f"HTTP {response.status_code}"))
        
        # Test conversation analysis depth
        if self.conversation_sessions:
            conv_id = list(self.conversation_sessions.values())[0][0]
            response = self.client.get(f"/memory/conversation/{conv_id}?include_context=true")
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['message_count', 'summary']
                found_fields = [field for field in required_fields if field in data]
                
                if len(found_fields) == len(required_fields):
                    analytics_tests.append(("Conversation Analysis", "PASS", "Complete analysis"))
                else:
                    analytics_tests.append(("Conversation Analysis", "PARTIAL", 
                                          f"{len(found_fields)}/{len(required_fields)} fields"))
            else:
                analytics_tests.append(("Conversation Analysis", "FAIL", f"HTTP {response.status_code}"))
        
        # Log analytics results
        for test_name, status, details in analytics_tests:
            self.log_test(f"Analytics: {test_name}", status, details)
        
        passed_analytics = sum(1 for _, status, _ in analytics_tests if status == "PASS")
        total_analytics = len(analytics_tests)
        
        overall_status = "PASS" if passed_analytics >= total_analytics * 0.8 else "PARTIAL"
        self.log_test("Advanced Memory Analytics", overall_status,
                     f"{passed_analytics}/{total_analytics} analytics tests passed")
        
        return passed_analytics >= total_analytics * 0.6
    
    def run_advanced_test_suite(self):
        """Run the complete advanced testing suite."""
        print("🚀 Advanced Deep Conversation & Memory Stress Testing")
        print("=" * 70)
        
        # Setup
        if not self.create_complex_conversation_scenarios():
            print("❌ Failed to create scenarios")
            return 0, 1
        
        if not self.populate_conversation_memory():
            print("❌ Failed to populate memory")
            return 0, 1
        
        # Advanced test suite
        advanced_tests = [
            ("Complex Semantic Search", self.test_complex_semantic_search),
            ("Cross-Conversation Linking", self.test_cross_conversation_context_linking), 
            ("Memory Stress Testing", self.test_memory_stress_scenarios),
            ("Advanced Analytics", self.test_advanced_memory_analytics)
        ]
        
        passed = 0
        total = len(advanced_tests)
        
        for test_name, test_func in advanced_tests:
            try:
                print(f"\n🔬 Running {test_name}...")
                if test_func():
                    passed += 1
                    print(f"   ✅ {test_name} completed successfully")
                else:
                    print(f"   ⚠️  {test_name} completed with issues")
            except Exception as e:
                self.log_test(f"{test_name} Exception", "FAIL", str(e))
                print(f"   ❌ {test_name} failed with exception: {e}")
        
        # Generate comprehensive results
        print(f"\n📊 Advanced Test Results: {passed}/{total} test suites passed")
        
        if passed == total:
            print("🏆 OUTSTANDING: All advanced tests passed! The memory system is highly robust.")
        elif passed >= total * 0.75:
            print("🎯 EXCELLENT: Most advanced tests passed with flying colors!")
        elif passed >= total * 0.5:
            print("👍 GOOD: Advanced tests show solid performance with room for optimization.")
        else:
            print("⚠️  NEEDS WORK: Advanced tests reveal areas needing improvement.")
        
        return passed, total
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive test report with insights."""
        
        # Analyze test results for insights
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["status"] == "PASS")
        failed_tests = sum(1 for r in self.test_results if r["status"] == "FAIL")
        partial_tests = sum(1 for r in self.test_results if r["status"] == "PARTIAL")
        
        # Calculate performance metrics
        overall_success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        report = {
            "test_suite": "Advanced Deep Conversation & Memory Stress Testing",
            "timestamp": time.time(),
            "test_environment": {
                "conversation_scenarios": len(self.conversation_sessions),
                "user_profiles": len(self.user_ids),
                "total_test_messages": sum(len(msgs) for _, msgs in self.conversation_sessions.values())
            },
            "results_summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "partial": partial_tests,
                "success_rate": round(overall_success_rate, 3)
            },
            "performance_insights": {
                "semantic_search_robustness": "High" if overall_success_rate >= 0.8 else "Medium" if overall_success_rate >= 0.6 else "Low",
                "stress_test_resilience": "Tested with edge cases, special characters, and rapid requests",
                "cross_conversation_linking": "Evaluated complex topic connections across conversations",
                "memory_analytics_depth": "Comprehensive analytics and reporting capabilities tested"
            },
            "detailed_results": self.test_results,
            "recommendations": self._generate_recommendations(overall_success_rate)
        }
        
        return report
    
    def _generate_recommendations(self, success_rate: float) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if success_rate >= 0.9:
            recommendations.append("Excellent performance! Consider adding more advanced features.")
            recommendations.append("System is ready for production use with complex scenarios.")
        elif success_rate >= 0.7:
            recommendations.append("Good performance with minor areas for improvement.")
            recommendations.append("Consider optimizing semantic search thresholds.")
        elif success_rate >= 0.5:
            recommendations.append("Moderate performance - focus on improving search relevance.")
            recommendations.append("Consider enhancing cross-conversation context linking.")
        else:
            recommendations.append("Significant improvements needed in core functionality.")
            recommendations.append("Review and optimize memory storage and retrieval mechanisms.")
        
        # Add specific recommendations based on test results
        failed_tests = [r for r in self.test_results if r["status"] == "FAIL"]
        if failed_tests:
            recommendations.append(f"Address {len(failed_tests)} failed test cases.")
        
        partial_tests = [r for r in self.test_results if r["status"] == "PARTIAL"]
        if len(partial_tests) > 3:
            recommendations.append("Investigate partial test results for optimization opportunities.")
        
        return recommendations

def main():
    """Run the advanced deep conversation testing suite."""
    print("🔬 Initializing Advanced Deep Testing Suite...")
    
    tester = AdvancedDeepTester()
    passed, total = tester.run_advanced_test_suite()
    
    # Generate comprehensive report
    report = tester.generate_comprehensive_report()
    
    # Save detailed report
    with open("dev/testing/advanced_deep_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Comprehensive report saved to: dev/testing/advanced_deep_test_report.json")
    
    # Print key insights
    print("\n📋 Key Insights:")
    for insight_key, insight_value in report["performance_insights"].items():
        print(f"   • {insight_key.replace('_', ' ').title()}: {insight_value}")
    
    print("\n💡 Recommendations:")
    for rec in report["recommendations"]:
        print(f"   • {rec}")
    
    print(f"\n🎯 Final Score: {passed}/{total} advanced test suites passed")
    print(f"📈 Success Rate: {report['results_summary']['success_rate']:.1%}")

if __name__ == "__main__":
    main()
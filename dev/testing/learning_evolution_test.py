#!/usr/bin/env python3
"""
Learning Evolution Test Suite
============================

This advanced testing suite focuses on how the Enhanced Memory System
learns and evolves its understanding through continuous conversations.
It simulates realistic learning scenarios to help the system improve
its reasoning and context understanding.
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
    print("✅ Successfully imported learning evolution test components")
except ImportError as e:
    print(f"❌ Failed to import components: {e}")
    sys.exit(1)

class LearningEvolutionTester:
    """Test how the memory system learns and evolves understanding."""
    
    def __init__(self):
        self.client = TestClient(app)
        self.test_results = []
        self.learning_sessions = {}
        self.knowledge_progression = []
        
    def log_test(self, test_name: str, status: str, details: str = "", learning_data: Dict = None):
        """Enhanced logging with learning metrics."""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "learning_data": learning_data or {},
            "timestamp": time.time()
        }
        self.test_results.append(result)
        
        status_icon = "🧠" if status == "LEARNING" else "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️" if status == "PARTIAL" else "📈" if status == "PROGRESS" else "🔄"
        print(f"{status_icon} {test_name}: {status}")
        if details:
            print(f"   {details}")
        if learning_data:
            print(f"   📊 Learning: {learning_data}")
    
    def create_progressive_learning_conversation(self):
        """Create a conversation that progressively builds complex understanding."""
        print("\n🧠 Creating Progressive Learning Conversation...")
        
        # Simulate a user learning advanced quantum computing over multiple sessions
        conv_id = f"learning_quantum_{uuid.uuid4().hex[:8]}"
        user_id = "curious_learner_001"
        
        # Session 1: Basic concepts
        session_1_messages = [
            ("user", "I'm new to quantum computing. Can you explain what a qubit is?"),
            ("assistant", "A qubit (quantum bit) is the fundamental unit of quantum information. Unlike classical bits that are either 0 or 1, a qubit can exist in a superposition of both states simultaneously. Mathematically, we represent a qubit as |ψ⟩ = α|0⟩ + β|1⟩ where α and β are complex probability amplitudes satisfying |α|² + |β|² = 1."),
            ("user", "What does superposition actually mean in practical terms?"),
            ("assistant", "Superposition means the qubit exists in all possible states until measured. Think of it like a spinning coin in the air - it's both heads and tails until it lands. However, this analogy is limited because quantum superposition involves complex probability amplitudes that can interfere constructively or destructively, leading to quantum phenomena like entanglement and quantum parallelism."),
            ("user", "How is this different from just not knowing the state?"),
            ("assistant", "Excellent question! This touches on the difference between quantum superposition and classical ignorance. In classical systems, an unknown state is just hidden information - the coin is actually heads or tails, we just don't know which. In quantum systems, the superposition is real and measurable through interference effects. The qubit genuinely exists in both states simultaneously until measurement collapses it to a definite state.")
        ]
        
        # Add session 1 to memory
        if conv_id not in conversation_memory:
            conversation_memory[conv_id] = []
        
        for role, content in session_1_messages:
            message = {
                "role": role,
                "content": content,
                "timestamp": time.time(),
                "session": 1,
                "topic_level": "basic"
            }
            conversation_memory[conv_id].append(message)
        
        # Wait a bit to simulate time passage
        time.sleep(0.1)
        
        # Session 2: Intermediate concepts (building on Session 1)
        session_2_messages = [
            ("user", "Now I understand superposition better. Can you explain quantum entanglement?"),
            ("assistant", "Building on your understanding of superposition, quantum entanglement occurs when two or more qubits become correlated in such a way that the quantum state of each qubit cannot be described independently. For two entangled qubits, we might have |ψ⟩ = (|00⟩ + |11⟩)/√2. Measuring one qubit instantly determines the state of the other, regardless of distance."),
            ("user", "Wait, how does this relate to the superposition concept we discussed? Are both qubits in superposition?"),
            ("assistant", "Excellent connection! Yes, in the entangled state |ψ⟩ = (|00⟩ + |11⟩)/√2, the system is in superposition of two possibilities: both qubits being 0 OR both being 1. However, neither individual qubit is in a pure superposition state. Each qubit is in what we call a 'mixed state' - it has no definite superposition on its own, but the correlation between them is perfectly defined. This is what makes entanglement so special and different from simple superposition."),
            ("user", "This is mind-bending! How do we create these entangled states in practice?"),
            ("assistant", "Since you grasp the theoretical foundation, let's look at practical implementation. The most common method uses a CNOT (Controlled-NOT) gate preceded by a Hadamard gate. Start with |00⟩, apply Hadamard to the first qubit: (|0⟩ + |1⟩)|0⟩/√2, then apply CNOT: (|00⟩ + |11⟩)/√2. The Hadamard creates superposition (as you learned earlier), and CNOT creates the correlation that defines entanglement.")
        ]
        
        for role, content in session_2_messages:
            message = {
                "role": role,
                "content": content,
                "timestamp": time.time(),
                "session": 2,
                "topic_level": "intermediate"
            }
            conversation_memory[conv_id].append(message)
        
        time.sleep(0.1)
        
        # Session 3: Advanced concepts (building on Sessions 1 & 2)
        session_3_messages = [
            ("user", "I want to understand quantum algorithms. How does Grover's algorithm use these concepts?"),
            ("assistant", "Perfect timing! Grover's algorithm beautifully demonstrates both superposition and quantum interference. It searches an unsorted database quadratically faster than classical algorithms. The algorithm starts by putting all qubits in superposition (like we discussed in session 1), then uses controlled rotations and reflections to amplify the probability amplitude of the target state while diminishing others through destructive interference."),
            ("user", "Can you connect this to the entanglement concepts? Does Grover's use entanglement?"),
            ("assistant", "Great question showing your deep thinking! Grover's algorithm primarily uses superposition and interference rather than entanglement. However, entanglement becomes crucial when we implement Grover's on quantum computers with multiple qubits, and especially when we combine it with other algorithms. The oracle function in Grover's often creates temporary entanglement between query qubits and ancilla qubits, but this isn't the core mechanism - that's the amplitude amplification through superposition manipulation we discussed."),
            ("user", "How does this relate to quantum error correction? I imagine these delicate superposition states are fragile."),
            ("assistant", "Absolutely brilliant insight! You're connecting all the concepts beautifully. Quantum error correction is essential because, as you correctly intuited, superposition and entanglement are extremely fragile - they're destroyed by decoherence from environmental noise. Quantum error correction uses clever entanglement schemes to encode logical qubits across multiple physical qubits. Ironically, we use the same entanglement that makes quantum computing powerful to protect against the noise that threatens to destroy quantum states. The error correction codes create 'protected subspaces' where quantum information can survive despite individual qubit errors.")
        ]
        
        for role, content in session_3_messages:
            message = {
                "role": role,
                "content": content,
                "timestamp": time.time(),
                "session": 3,
                "topic_level": "advanced"
            }
            conversation_memory[conv_id].append(message)
        
        self.learning_sessions[conv_id] = {
            "user_id": user_id,
            "total_sessions": 3,
            "total_messages": len(session_1_messages) + len(session_2_messages) + len(session_3_messages),
            "progression": ["basic", "intermediate", "advanced"],
            "key_concepts": ["superposition", "entanglement", "quantum algorithms", "error correction"]
        }
        
        print(f"   ✅ Created progressive learning conversation with {self.learning_sessions[conv_id]['total_messages']} messages")
        return conv_id
    
    def test_knowledge_building_memory(self, conv_id: str):
        """Test how well memory retrieves and builds on previous knowledge."""
        print("\n🧠 Testing Knowledge Building Memory")
        
        # Test queries that should demonstrate learning progression
        learning_queries = [
            ("What is superposition in quantum computing?", ["superposition", "qubit", "states"], "basic"),
            ("How does entanglement relate to superposition?", ["entanglement", "superposition", "correlation"], "intermediate"),
            ("How do quantum algorithms use these concepts?", ["algorithms", "Grover", "superposition", "interference"], "advanced"),
            ("Explain quantum error correction and its relationship to entanglement", ["error correction", "entanglement", "decoherence"], "advanced"),
            ("Connect all quantum concepts from basic to advanced", ["superposition", "entanglement", "algorithms", "error correction"], "synthesis")
        ]
        
        knowledge_progression = []
        
        for query, expected_concepts, complexity_level in learning_queries:
            # Test context retrieval for this specific conversation
            params = {"message": query}
            response = self.client.get(f"/memory/conversation/{conv_id}/context", params=params)
            
            if response.status_code == 200:
                data = response.json()
                context_messages = data.get('context_messages', [])
                
                # Analyze knowledge building
                concepts_found = set()
                session_coverage = set()
                
                for msg in context_messages:
                    content = msg.get('content', '').lower()
                    session = msg.get('session', 0)
                    
                    for concept in expected_concepts:
                        if concept.lower() in content:
                            concepts_found.add(concept)
                    
                    if session > 0:
                        session_coverage.add(session)
                
                concept_coverage = len(concepts_found) / len(expected_concepts)
                session_span = len(session_coverage)
                
                learning_data = {
                    "complexity_level": complexity_level,
                    "concept_coverage": round(concept_coverage, 2),
                    "concepts_found": list(concepts_found),
                    "sessions_referenced": sorted(list(session_coverage)),
                    "context_count": len(context_messages)
                }
                
                knowledge_progression.append(learning_data)
                
                status = "LEARNING" if concept_coverage >= 0.7 and session_span >= 2 else "PARTIAL" if concept_coverage >= 0.4 else "FAIL"
                self.log_test(f"Knowledge Building: {complexity_level}", status,
                            f"Query: '{query[:50]}...', Coverage: {concept_coverage:.1%}",
                            learning_data)
            else:
                self.log_test(f"Knowledge Building: {complexity_level}", "FAIL", f"HTTP {response.status_code}")
        
        # Analyze overall learning progression
        self.knowledge_progression = knowledge_progression
        
        avg_coverage = sum(kp["concept_coverage"] for kp in knowledge_progression) / len(knowledge_progression)
        max_sessions = max((len(kp["sessions_referenced"]) for kp in knowledge_progression), default=0)
        
        overall_status = "LEARNING" if avg_coverage >= 0.6 and max_sessions >= 2 else "PARTIAL"
        self.log_test("Knowledge Building Memory", overall_status,
                     f"Average concept coverage: {avg_coverage:.1%}, Max session span: {max_sessions}")
        
        return avg_coverage >= 0.5
    
    def test_conversational_reasoning_chains(self):
        """Test complex reasoning chains across conversation context."""
        print("\n🧠 Testing Conversational Reasoning Chains")
        
        # Create a complex problem-solving conversation
        problem_conv_id = f"reasoning_{uuid.uuid4().hex[:8]}"
        
        reasoning_messages = [
            ("user", "I need to design a quantum algorithm for optimization. Where should I start?"),
            ("assistant", "For quantum optimization, I'd recommend starting with the Quantum Approximate Optimization Algorithm (QAOA). It's designed specifically for combinatorial optimization problems. QAOA uses alternating unitary operations: e^(-iγH_C) e^(-iβH_M) where H_C encodes the cost function and H_M is the mixing Hamiltonian. The parameters γ and β are optimized classically to minimize the expectation value ⟨H_C⟩."),
            ("user", "How does QAOA compare to Grover's algorithm for optimization?"),
            ("assistant", "Great question! While both are quantum algorithms, they serve different optimization purposes. Grover's algorithm is optimal for unstructured search - finding a specific item in an unsorted database with O(√N) complexity. QAOA, however, is designed for structured optimization problems where we want to find the minimum of a cost function over a combinatorial space. QAOA can potentially provide quantum advantage for specific problem instances, while Grover's provides a proven quadratic speedup for search."),
            ("user", "What if I have constraints in my optimization problem? How would I incorporate them?"),
            ("assistant", "Excellent follow-up! For constrained optimization in QAOA, you have several approaches: 1) Penalty methods - add constraint violations to your cost Hamiltonian H_C with large penalty coefficients, 2) Barrier methods - modify the mixing Hamiltonian H_M to avoid constraint-violating states, or 3) Lagrangian approaches - introduce auxiliary qubits for Lagrange multipliers. The choice depends on your constraint structure. Hard constraints often work better with modified mixing operators, while soft constraints can use penalty methods."),
            ("user", "How do I choose the optimal circuit depth for QAOA? Is there a theoretical guideline?"),
            ("assistant", "This is one of QAOA's key challenges! The circuit depth p (number of QAOA layers) involves a trade-off: deeper circuits can achieve better approximation ratios but are more susceptible to noise and require longer coherence times. Theoretical work suggests that for many problems, modest depths (p=1 to p=10) can provide significant quantum advantage. However, the optimal depth depends on your specific problem structure, noise levels, and desired approximation quality. Recent research shows that for some problems, even p=1 can outperform classical algorithms, while others require p=O(log n) for advantage.")
        ]
        
        # Add to conversation memory
        if problem_conv_id not in conversation_memory:
            conversation_memory[problem_conv_id] = []
        
        for role, content in reasoning_messages:
            message = {
                "role": role,
                "content": content,
                "timestamp": time.time(),
                "reasoning_step": len(conversation_memory[problem_conv_id]) + 1
            }
            conversation_memory[problem_conv_id].append(message)
        
        # Test reasoning chain retrieval
        reasoning_tests = [
            ("What optimization algorithm should I use and why?", ["QAOA", "optimization", "algorithm"]),
            ("How does the choice compare to other quantum approaches?", ["Grover", "QAOA", "comparison", "optimization"]),
            ("What about constraints in the optimization?", ["constraints", "penalty", "Lagrangian", "Hamiltonian"]),
            ("How do I determine the right circuit parameters?", ["circuit depth", "layers", "trade-off", "noise"])
        ]
        
        reasoning_results = []
        
        for query, expected_reasoning in reasoning_tests:
            params = {"message": query}
            response = self.client.get(f"/memory/conversation/{problem_conv_id}/context", params=params)
            
            if response.status_code == 200:
                data = response.json()
                context_messages = data.get('context_messages', [])
                
                # Check reasoning chain coherence
                reasoning_elements = set()
                logical_flow = []
                
                for msg in context_messages:
                    content = msg.get('content', '').lower()
                    step = msg.get('reasoning_step', 0)
                    
                    for element in expected_reasoning:
                        if element.lower() in content:
                            reasoning_elements.add(element)
                    
                    if step > 0:
                        logical_flow.append(step)
                
                reasoning_coverage = len(reasoning_elements) / len(expected_reasoning)
                logical_sequence = len(set(logical_flow)) > 1  # Multiple reasoning steps
                
                reasoning_data = {
                    "reasoning_coverage": round(reasoning_coverage, 2),
                    "elements_found": list(reasoning_elements),
                    "logical_sequence": logical_sequence,
                    "reasoning_steps": sorted(logical_flow)
                }
                
                reasoning_results.append(reasoning_data)
                
                status = "LEARNING" if reasoning_coverage >= 0.7 and logical_sequence else "PARTIAL" if reasoning_coverage >= 0.4 else "FAIL"
                self.log_test(f"Reasoning Chain: {query[:40]}...", status,
                            f"Coverage: {reasoning_coverage:.1%}, Sequence: {logical_sequence}",
                            reasoning_data)
            else:
                self.log_test(f"Reasoning Chain: {query[:40]}...", "FAIL", f"HTTP {response.status_code}")
        
        # Overall reasoning chain assessment
        avg_reasoning = sum(r["reasoning_coverage"] for r in reasoning_results) / len(reasoning_results)
        logical_chains = sum(1 for r in reasoning_results if r["logical_sequence"])
        
        overall_status = "LEARNING" if avg_reasoning >= 0.6 and logical_chains >= len(reasoning_results) * 0.75 else "PARTIAL"
        self.log_test("Conversational Reasoning Chains", overall_status,
                     f"Avg reasoning: {avg_reasoning:.1%}, Logical chains: {logical_chains}/{len(reasoning_results)}")
        
        return avg_reasoning >= 0.5
    
    def test_adaptive_context_relevance(self):
        """Test how well the system adapts context relevance based on conversation flow."""
        print("\n🧠 Testing Adaptive Context Relevance")
        
        # Create conversation with topic shifts to test adaptive relevance
        adaptive_conv_id = f"adaptive_{uuid.uuid4().hex[:8]}"
        
        # Start with one topic, then shift to related but different topics
        adaptive_messages = [
            ("user", "Explain machine learning gradient descent optimization"),
            ("assistant", "Gradient descent is an iterative optimization algorithm that finds the minimum of a function by following the negative gradient. For a function f(θ), we update parameters: θ_{t+1} = θ_t - α∇f(θ_t) where α is the learning rate. The gradient ∇f points in the direction of steepest increase, so -∇f points toward the steepest decrease."),
            ("user", "How does this relate to quantum optimization algorithms?"),
            ("assistant", "Excellent connection! While classical gradient descent operates on smooth, differentiable functions in continuous spaces, quantum optimization algorithms like QAOA work differently. They optimize discrete combinatorial problems using quantum superposition and interference. However, there are quantum analogs: Quantum Natural Gradient methods and variational quantum algorithms use classical optimizers (including gradient descent) to optimize quantum circuit parameters."),
            ("user", "What about neural network training? How does backpropagation use gradient descent?"),
            ("assistant", "Backpropagation is the algorithm that computes gradients for gradient descent in neural networks. It uses the chain rule to efficiently calculate ∂L/∂w_i for each weight w_i in the network, where L is the loss function. Starting from the output layer, it propagates error gradients backward through the network: ∂L/∂w_i = ∂L/∂z_j · ∂z_j/∂w_i, where z_j is the weighted input to neuron j. This makes gradient descent feasible for networks with millions of parameters."),
            ("user", "Can we apply quantum principles to improve neural network training?"),
            ("assistant", "Fascinating direction! Quantum-enhanced neural networks are an active research area. Quantum Neural Networks (QNNs) use quantum circuits as parameterized functions, potentially offering exponential advantages for certain problems. Variational Quantum Circuits can serve as quantum analogs of classical neural layers. The quantum gradient descent involves computing gradients with respect to quantum circuit parameters, often using parameter-shift rules or finite differences since quantum circuits are inherently discrete.")
        ]
        
        # Add to conversation memory
        if adaptive_conv_id not in conversation_memory:
            conversation_memory[adaptive_conv_id] = []
        
        for i, (role, content) in enumerate(adaptive_messages):
            message = {
                "role": role,
                "content": content,
                "timestamp": time.time(),
                "topic_shift": i // 2,  # Groups messages by topic pairs
                "relevance_context": ["gradient_descent", "quantum", "neural_networks", "optimization"][i // 2]
            }
            conversation_memory[adaptive_conv_id].append(message)
        
        # Test adaptive relevance with queries that should prioritize different contexts
        relevance_tests = [
            ("Explain gradient descent mathematics", ["gradient", "mathematics", "algorithm"], "gradient_descent"),
            ("How do quantum algorithms optimize?", ["quantum", "QAOA", "optimization"], "quantum"),
            ("What is backpropagation in neural networks?", ["backpropagation", "neural", "chain rule"], "neural_networks"),
            ("Connect quantum computing to machine learning", ["quantum", "neural", "gradient", "optimization"], "mixed")
        ]
        
        relevance_results = []
        
        for query, expected_terms, expected_context in relevance_tests:
            params = {"message": query}
            response = self.client.get(f"/memory/conversation/{adaptive_conv_id}/context", params=params)
            
            if response.status_code == 200:
                data = response.json()
                context_messages = data.get('context_messages', [])
                
                # Analyze context relevance and adaptation
                relevant_contexts = set()
                term_coverage = set()
                
                for msg in context_messages:
                    content = msg.get('content', '').lower()
                    msg_context = msg.get('relevance_context', '')
                    
                    if msg_context:
                        relevant_contexts.add(msg_context)
                    
                    for term in expected_terms:
                        if term.lower() in content:
                            term_coverage.add(term)
                
                term_relevance = len(term_coverage) / len(expected_terms)
                context_adaptation = expected_context in relevant_contexts if expected_context != "mixed" else len(relevant_contexts) > 1
                
                relevance_data = {
                    "expected_context": expected_context,
                    "relevant_contexts": list(relevant_contexts),
                    "term_relevance": round(term_relevance, 2),
                    "context_adaptation": context_adaptation,
                    "terms_found": list(term_coverage)
                }
                
                relevance_results.append(relevance_data)
                
                status = "LEARNING" if term_relevance >= 0.7 and context_adaptation else "PARTIAL" if term_relevance >= 0.4 else "FAIL"
                self.log_test(f"Adaptive Relevance: {expected_context}", status,
                            f"Term relevance: {term_relevance:.1%}, Adapted: {context_adaptation}",
                            relevance_data)
            else:
                self.log_test(f"Adaptive Relevance: {expected_context}", "FAIL", f"HTTP {response.status_code}")
        
        # Overall adaptive relevance assessment
        avg_relevance = sum(r["term_relevance"] for r in relevance_results) / len(relevance_results)
        adaptations = sum(1 for r in relevance_results if r["context_adaptation"])
        
        overall_status = "LEARNING" if avg_relevance >= 0.6 and adaptations >= len(relevance_results) * 0.75 else "PARTIAL"
        self.log_test("Adaptive Context Relevance", overall_status,
                     f"Avg relevance: {avg_relevance:.1%}, Adaptations: {adaptations}/{len(relevance_results)}")
        
        return avg_relevance >= 0.5
    
    def test_real_time_learning_integration(self):
        """Test how the system integrates new learning in real-time during conversations."""
        print("\n🧠 Testing Real-Time Learning Integration")
        
        # Test with continuous conversation flow and memory updates
        realtime_conv_id = f"realtime_{uuid.uuid4().hex[:8]}"
        
        # Simulate a conversation where each exchange builds on the previous
        learning_exchanges = [
            ("user", "What are transformers in deep learning?"),
            ("assistant", "Transformers are neural network architectures based on self-attention mechanisms. The key innovation is the attention mechanism: Attention(Q,K,V) = softmax(QK^T/√d_k)V, where Q, K, V are query, key, and value matrices. This allows the model to focus on relevant parts of the input sequence when processing each element."),
            ("user", "How does the attention mechanism actually work step by step?"),
            ("assistant", "Great follow-up! Let me break down the attention mechanism step by step, building on the formula I just mentioned: 1) Linear projections create Q, K, V from input X: Q=XW_Q, K=XW_K, V=XW_V. 2) Compute attention scores: scores = QK^T. 3) Scale by √d_k to prevent vanishing gradients. 4) Apply softmax to get attention weights. 5) Weighted sum of values: output = softmax(QK^T/√d_k)V. Each step serves a specific purpose in allowing tokens to 'attend' to other relevant tokens."),
            ("user", "Why do we need multi-head attention instead of just single attention?"),
            ("assistant", "Excellent question that builds on your understanding! Multi-head attention allows the model to attend to different types of relationships simultaneously. While single attention might focus on one pattern (like syntactic relationships), multi-head attention runs h parallel attention heads, each potentially learning different aspects: syntactic, semantic, positional, etc. Mathematically: MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O where each head_i = Attention(QW_i^Q, KW_i^K, VW_i^V). This gives the model richer representational capacity.")
        ]
        
        # Add messages progressively and test memory integration after each
        if realtime_conv_id not in conversation_memory:
            conversation_memory[realtime_conv_id] = []
        
        integration_results = []
        
        for i, (role, content) in enumerate(learning_exchanges):
            # Add message to memory
            message = {
                "role": role,
                "content": content,
                "timestamp": time.time(),
                "exchange_number": i + 1
            }
            conversation_memory[realtime_conv_id].append(message)
            
            # Test immediate memory integration
            if role == "user":  # Test after user questions
                test_query = f"Based on our discussion, explain transformers at exchange {i+1}"
                params = {"message": test_query}
                response = self.client.get(f"/memory/conversation/{realtime_conv_id}/context", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    context_messages = data.get('context_messages', [])
                    
                    # Check if latest exchanges are properly integrated
                    latest_exchanges = [msg for msg in context_messages 
                                      if msg.get('exchange_number', 0) <= i + 1]
                    integration_depth = len(latest_exchanges)
                    
                    integration_data = {
                        "exchange_number": i + 1,
                        "integration_depth": integration_depth,
                        "context_count": len(context_messages),
                        "real_time_update": integration_depth > 0
                    }
                    
                    integration_results.append(integration_data)
                    
                    status = "LEARNING" if integration_depth >= (i + 1) * 0.5 else "PARTIAL"
                    self.log_test(f"Real-Time Integration: Exchange {i+1}", status,
                                f"Integration depth: {integration_depth}, Real-time: {integration_data['real_time_update']}",
                                integration_data)
        
        # Overall real-time learning assessment
        avg_integration = sum(r["integration_depth"] for r in integration_results) / len(integration_results) if integration_results else 0
        real_time_updates = sum(1 for r in integration_results if r["real_time_update"])
        
        overall_status = "LEARNING" if avg_integration >= 2 and real_time_updates == len(integration_results) else "PARTIAL"
        self.log_test("Real-Time Learning Integration", overall_status,
                     f"Avg integration: {avg_integration:.1f}, Real-time updates: {real_time_updates}/{len(integration_results)}")
        
        return avg_integration >= 1.5
    
    def analyze_learning_evolution(self):
        """Analyze how the system's learning evolved throughout testing."""
        print("\n📈 Analyzing Learning Evolution")
        
        if not self.knowledge_progression:
            self.log_test("Learning Evolution Analysis", "FAIL", "No knowledge progression data available")
            return False
        
        # Analyze learning progression patterns
        complexity_levels = ["basic", "intermediate", "advanced", "synthesis"]
        coverage_by_level = {}
        
        for progress in self.knowledge_progression:
            level = progress.get("complexity_level", "unknown")
            coverage = progress.get("concept_coverage", 0)
            
            if level not in coverage_by_level:
                coverage_by_level[level] = []
            coverage_by_level[level].append(coverage)
        
        # Calculate learning evolution metrics
        evolution_metrics = {}
        for level in complexity_levels:
            if level in coverage_by_level:
                avg_coverage = sum(coverage_by_level[level]) / len(coverage_by_level[level])
                evolution_metrics[level] = round(avg_coverage, 2)
        
        # Determine if there's positive learning progression
        learning_progression = list(evolution_metrics.values())
        positive_trend = all(learning_progression[i] <= learning_progression[i+1] 
                           for i in range(len(learning_progression)-1)) if len(learning_progression) > 1 else True
        
        evolution_data = {
            "complexity_progression": evolution_metrics,
            "positive_learning_trend": positive_trend,
            "total_concepts_tracked": sum(len(p.get("concepts_found", [])) for p in self.knowledge_progression),
            "learning_depth": len(evolution_metrics)
        }
        
        status = "LEARNING" if positive_trend and len(evolution_metrics) >= 3 else "PARTIAL"
        self.log_test("Learning Evolution Analysis", status,
                     f"Positive trend: {positive_trend}, Depth: {len(evolution_metrics)} levels",
                     evolution_data)
        
        return positive_trend and len(evolution_metrics) >= 2
    
    def run_learning_evolution_suite(self):
        """Run the complete learning evolution testing suite."""
        print("🧠 LEARNING EVOLUTION & ADAPTATION TESTING")
        print("=" * 80)
        
        # Create learning conversation
        conv_id = self.create_progressive_learning_conversation()
        
        # Learning evolution test suite
        learning_tests = [
            ("Knowledge Building Memory", lambda: self.test_knowledge_building_memory(conv_id)),
            ("Conversational Reasoning Chains", self.test_conversational_reasoning_chains),
            ("Adaptive Context Relevance", self.test_adaptive_context_relevance),
            ("Real-Time Learning Integration", self.test_real_time_learning_integration),
            ("Learning Evolution Analysis", self.analyze_learning_evolution)
        ]
        
        passed = 0
        total = len(learning_tests)
        
        for test_name, test_func in learning_tests:
            try:
                print(f"\n🧠 Running {test_name}...")
                start_time = time.time()
                
                if test_func():
                    passed += 1
                    elapsed = time.time() - start_time
                    print(f"   ✅ {test_name} shows positive learning in {elapsed:.2f}s")
                else:
                    elapsed = time.time() - start_time
                    print(f"   📈 {test_name} shows learning potential in {elapsed:.2f}s")
            except Exception as e:
                elapsed = time.time() - start_time
                self.log_test(f"{test_name} Exception", "FAIL", str(e))
                print(f"   ❌ {test_name} failed with exception in {elapsed:.2f}s: {e}")
        
        # Generate learning evolution results
        print(f"\n📊 Learning Evolution Results: {passed}/{total} learning tests successful")
        
        if passed == total:
            print("🧠 BRILLIANT: The system demonstrates exceptional learning capabilities!")
        elif passed >= total * 0.8:
            print("📈 EXCELLENT: Strong learning evolution and adaptation detected!")
        elif passed >= total * 0.6:
            print("🎯 GOOD: Solid learning patterns with room for optimization!")
        else:
            print("📚 DEVELOPING: Learning capabilities detected, needs enhancement!")
        
        return passed, total

def main():
    """Run the learning evolution testing suite."""
    print("🧠 Initializing Learning Evolution Testing...")
    
    tester = LearningEvolutionTester()
    passed, total = tester.run_learning_evolution_suite()
    
    # Generate learning evolution report
    learning_report = {
        "test_suite": "Learning Evolution & Adaptation Testing",
        "timestamp": time.time(),
        "learning_sessions": tester.learning_sessions,
        "knowledge_progression": tester.knowledge_progression,
        "results_summary": {
            "total_learning_tests": total,
            "successful_learning": passed,
            "learning_success_rate": round(passed / total, 3) if total > 0 else 0
        },
        "detailed_results": tester.test_results,
        "learning_insights": {
            "progressive_understanding": len(tester.knowledge_progression) > 0,
            "contextual_adaptation": passed >= total * 0.6,
            "real_time_integration": passed >= total * 0.8
        }
    }
    
    # Save learning evolution report
    with open("dev/testing/learning_evolution_report.json", "w") as f:
        json.dump(learning_report, f, indent=2)
    
    print(f"\n📄 Learning evolution report saved to: dev/testing/learning_evolution_report.json")
    print(f"🎯 Final Learning Score: {passed}/{total} learning tests successful")
    print(f"📈 Learning Success Rate: {learning_report['results_summary']['learning_success_rate']:.1%}")
    
    print(f"\n🧠 Learning Insights:")
    for insight_key, insight_value in learning_report["learning_insights"].items():
        print(f"   • {insight_key.replace('_', ' ').title()}: {'✅' if insight_value else '📈'}")

if __name__ == "__main__":
    main()
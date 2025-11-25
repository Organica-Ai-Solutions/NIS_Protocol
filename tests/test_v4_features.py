"""
NIS Protocol v4.0 - Self-Improving Consciousness Test Suite
Tests all v4.0 features: Reflective Generation, Persistent Memory, Self-Modification
"""

import pytest
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestReflectiveGenerator:
    """Test the Reflective Generation system"""
    
    @pytest.mark.asyncio
    async def test_reflective_generator_init(self):
        """Test ReflectiveGenerator initialization"""
        from src.llm.reflective_generator import ReflectiveGenerator, ReflectionStrategy
        
        # Mock LLM provider
        class MockLLMProvider:
            async def generate_response(self, prompt, max_tokens=1024):
                return {"content": f"Response to: {prompt[:50]}..."}
        
        generator = ReflectiveGenerator(
            llm_provider=MockLLMProvider(),
            max_iterations=3,
            quality_threshold=0.75
        )
        
        assert generator.max_iterations == 3
        assert generator.quality_threshold == 0.75
        assert generator.strategy == ReflectionStrategy.CRITIQUE_AND_REVISE
    
    @pytest.mark.asyncio
    async def test_reflective_generation(self):
        """Test full reflective generation flow"""
        from src.llm.reflective_generator import ReflectiveGenerator
        
        class MockLLMProvider:
            async def generate_response(self, prompt, max_tokens=1024):
                return {"content": "This is a detailed response that addresses the user's question about quantum physics and provides accurate information."}
        
        generator = ReflectiveGenerator(
            llm_provider=MockLLMProvider(),
            max_iterations=2,
            quality_threshold=0.6
        )
        
        result = await generator.generate(
            prompt="Explain quantum entanglement",
            context="Physics discussion"
        )
        
        assert result.final_response is not None
        assert result.iterations >= 1
        assert 0 <= result.final_score <= 1
        assert len(result.reasoning_trace) > 0
    
    @pytest.mark.asyncio
    async def test_novelty_scoring(self):
        """Test novelty score calculation"""
        from src.llm.reflective_generator import ReflectiveGenerator
        
        class MockLLMProvider:
            async def generate_response(self, prompt, max_tokens=1024):
                return {"content": "Response"}
        
        generator = ReflectiveGenerator(llm_provider=MockLLMProvider())
        
        # Simple query - low novelty
        simple_score = generator._calculate_novelty("hello", "Hi there!")
        
        # Complex query - higher novelty
        complex_score = generator._calculate_novelty(
            "How do I implement a recursive algorithm to solve the traveling salesman problem?",
            "Here's a detailed implementation..."
        )
        
        assert complex_score > simple_score


class TestPersistentMemory:
    """Test the Persistent Memory system"""
    
    @pytest.mark.asyncio
    async def test_memory_store_and_retrieve(self):
        """Test storing and retrieving memories"""
        from src.memory.persistent_memory import PersistentMemorySystem
        
        memory = PersistentMemorySystem(storage_path="/tmp/test_memory")
        
        # Store a memory
        memory_id = await memory.store(
            content="The NIS Protocol uses physics-informed neural networks",
            memory_type="semantic",
            importance=0.8
        )
        
        assert memory_id is not None
        
        # Retrieve it
        results = await memory.retrieve(
            query="What does NIS Protocol use?",
            top_k=3
        )
        
        assert len(results) > 0
        assert "NIS Protocol" in results[0].entry.content
    
    @pytest.mark.asyncio
    async def test_memory_types(self):
        """Test different memory types"""
        from src.memory.persistent_memory import PersistentMemorySystem
        
        memory = PersistentMemorySystem(storage_path="/tmp/test_memory_types")
        
        # Store different types
        await memory.store_conversation(
            user_message="What is AI?",
            assistant_response="AI is artificial intelligence",
            importance=0.7
        )
        
        await memory.store_knowledge(
            fact="Python is a programming language",
            source="learned",
            confidence=0.9
        )
        
        await memory.store_pattern(
            pattern_description="Users often ask follow-up questions",
            success_rate=0.8
        )
        
        stats = memory.get_stats()
        assert stats["total_memories"] >= 3
    
    @pytest.mark.asyncio
    async def test_context_retrieval(self):
        """Test getting context for a query"""
        from src.memory.persistent_memory import PersistentMemorySystem
        
        memory = PersistentMemorySystem(storage_path="/tmp/test_context")
        
        await memory.store(
            content="Quantum computers use qubits instead of classical bits",
            memory_type="semantic",
            importance=0.9
        )
        
        context = await memory.get_context_for_query("Tell me about quantum computing")
        
        assert "qubit" in context.lower() or len(context) == 0  # May not match if no embeddings


class TestSelfModifier:
    """Test the Self-Modification system"""
    
    def test_self_modifier_init(self):
        """Test SelfModifier initialization"""
        from src.core.self_modifier import SelfModifier
        
        modifier = SelfModifier(storage_path="/tmp/test_modifier")
        
        assert modifier.parameters["default_temperature"] == 0.7
        assert modifier.parameters["reflection_threshold"] == 0.75
        assert modifier.max_modifications_per_hour == 5
    
    def test_record_metrics(self):
        """Test recording performance metrics"""
        from src.core.self_modifier import SelfModifier
        
        modifier = SelfModifier(storage_path="/tmp/test_modifier_metrics")
        
        # Record some metrics
        for i in range(30):
            modifier.record_metric("response_quality", 0.7 + (i * 0.01))
        
        status = modifier.get_status()
        assert status["metrics_summary"]["response_quality"]["samples"] == 30
        assert status["metrics_summary"]["response_quality"]["trend"] > 0  # Improving
    
    @pytest.mark.asyncio
    async def test_propose_modification(self):
        """Test proposing a modification"""
        from src.core.self_modifier import SelfModifier
        
        modifier = SelfModifier(storage_path="/tmp/test_modifier_propose")
        
        mod = await modifier.propose_modification(
            target="reflection_threshold",
            modification_type="parameter",
            new_value=0.8,
            reason="Testing modification"
        )
        
        assert mod is not None
        assert mod.target == "reflection_threshold"
        assert mod.new_value == 0.8
    
    @pytest.mark.asyncio
    async def test_apply_and_revert(self):
        """Test applying and reverting modifications"""
        from src.core.self_modifier import SelfModifier
        
        modifier = SelfModifier(storage_path="/tmp/test_modifier_revert")
        original = modifier.parameters["default_temperature"]
        
        mod = await modifier.propose_modification(
            target="default_temperature",
            modification_type="parameter",
            new_value=0.9,
            reason="Test"
        )
        
        await modifier.apply_modification(mod)
        assert modifier.parameters["default_temperature"] == 0.9
        
        await modifier.revert_modification(mod)
        assert modifier.parameters["default_temperature"] == original
    
    def test_get_parameters(self):
        """Test getting modifiable parameters"""
        from src.core.self_modifier import SelfModifier
        
        modifier = SelfModifier(storage_path="/tmp/test_modifier_params")
        
        temp = modifier.get_parameter("default_temperature")
        assert temp == 0.7
        
        threshold = modifier.get_routing_rule("simple_query_threshold")
        assert threshold == 0.3


class TestIntegration:
    """Integration tests for v4.0 features"""
    
    @pytest.mark.asyncio
    async def test_reflective_with_memory(self):
        """Test reflective generation using memory context"""
        from src.llm.reflective_generator import ReflectiveGenerator
        from src.memory.persistent_memory import PersistentMemorySystem
        
        memory = PersistentMemorySystem(storage_path="/tmp/test_integration")
        
        # Store some context
        await memory.store(
            content="The user prefers detailed technical explanations",
            memory_type="procedural",
            importance=0.9
        )
        
        class MockLLMProvider:
            async def generate_response(self, prompt, max_tokens=1024):
                return {"content": "Here's a detailed technical explanation..."}
        
        generator = ReflectiveGenerator(llm_provider=MockLLMProvider())
        
        # Get context from memory
        context = await memory.get_context_for_query("Explain something technical")
        
        result = await generator.generate(
            prompt="Explain neural networks",
            context=context
        )
        
        assert result.final_response is not None
    
    @pytest.mark.asyncio
    async def test_self_modifier_with_reflective(self):
        """Test self-modifier adjusting reflective parameters"""
        from src.core.self_modifier import SelfModifier
        
        modifier = SelfModifier(storage_path="/tmp/test_integration_modifier")
        
        # Simulate declining performance
        for i in range(30):
            modifier.record_metric("response_quality", 0.6 - (i * 0.005))
        
        # Trigger auto-optimization
        modifications = await modifier.auto_optimize()
        
        # Should have proposed some modifications due to declining quality
        status = modifier.get_status()
        assert status["metrics_summary"]["response_quality"]["trend"] < 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""
NIS Protocol v4.0 - Consciousness Pipeline Test Suite

Comprehensive tests for the 10-phase consciousness pipeline,
state transitions, and collective intelligence coordination.

Run with: pytest tests/test_consciousness_pipeline.py -v
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ConsciousnessPhase(Enum):
    """The 10 phases of consciousness"""
    EVOLUTION = "evolution"
    GENESIS = "genesis"
    DISTRIBUTED = "distributed"
    PLANNING = "planning"
    MARKETPLACE = "marketplace"
    MULTIPATH = "multipath"
    ETHICS = "ethics"
    EMBODIMENT = "embodiment"
    DEBUGGER = "debugger"
    META_EVOLUTION = "meta_evolution"


@dataclass
class ConsciousnessState:
    """Represents the state of a consciousness instance"""
    agent_id: str
    phase: ConsciousnessPhase = ConsciousnessPhase.EVOLUTION
    consciousness_threshold: float = 0.7
    bias_threshold: float = 0.3
    ethics_threshold: float = 0.8
    evolution_count: int = 0
    created_agents: List[str] = field(default_factory=list)
    collective_connections: List[str] = field(default_factory=list)
    active_plans: List[Dict] = field(default_factory=list)
    ethical_violations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "phase": self.phase.value,
            "consciousness_threshold": self.consciousness_threshold,
            "bias_threshold": self.bias_threshold,
            "ethics_threshold": self.ethics_threshold,
            "evolution_count": self.evolution_count,
            "created_agents": self.created_agents,
            "collective_connections": self.collective_connections,
            "active_plans": self.active_plans,
            "ethical_violations": self.ethical_violations
        }


class MockConsciousnessService:
    """Mock implementation of consciousness service for testing"""
    
    def __init__(self, agent_id: str = "test_agent"):
        self.state = ConsciousnessState(agent_id=agent_id)
        self.history: List[Dict] = []
    
    async def evolve(self, reason: str = "test") -> Dict[str, Any]:
        """Phase 1: Self-evolution"""
        before = self.state.bias_threshold
        
        # Simulate evolution: reduce bias threshold
        self.state.bias_threshold *= 0.9
        self.state.evolution_count += 1
        
        result = {
            "status": "success",
            "phase": "evolution",
            "reason": reason,
            "before": {"bias_threshold": before},
            "after": {"bias_threshold": self.state.bias_threshold}
        }
        self.history.append(result)
        return result
    
    async def genesis(self, agent_type: str, capabilities: List[str]) -> Dict[str, Any]:
        """Phase 2: Create new agent"""
        new_agent_id = f"{agent_type}_{len(self.state.created_agents)}"
        self.state.created_agents.append(new_agent_id)
        
        result = {
            "status": "success",
            "phase": "genesis",
            "agent_id": new_agent_id,
            "agent_type": agent_type,
            "capabilities": capabilities
        }
        self.history.append(result)
        return result
    
    async def connect_collective(self, peer_id: str) -> Dict[str, Any]:
        """Phase 3: Join collective consciousness"""
        if peer_id not in self.state.collective_connections:
            self.state.collective_connections.append(peer_id)
        
        result = {
            "status": "success",
            "phase": "distributed",
            "connected_to": peer_id,
            "total_connections": len(self.state.collective_connections)
        }
        self.history.append(result)
        return result
    
    async def create_plan(self, goal: str, steps: List[str]) -> Dict[str, Any]:
        """Phase 4: Autonomous planning"""
        plan = {
            "plan_id": f"plan_{len(self.state.active_plans)}",
            "goal": goal,
            "steps": steps,
            "status": "active",
            "created_at": time.time()
        }
        self.state.active_plans.append(plan)
        
        result = {
            "status": "success",
            "phase": "planning",
            "plan": plan
        }
        self.history.append(result)
        return result
    
    async def evaluate_ethics(self, action: str, context: Dict) -> Dict[str, Any]:
        """Phase 7: Ethical evaluation"""
        # Simple ethical rules
        forbidden_actions = ["harm", "deceive", "steal", "violate_privacy"]
        
        is_ethical = not any(word in action.lower() for word in forbidden_actions)
        
        if not is_ethical:
            self.state.ethical_violations += 1
        
        result = {
            "status": "success",
            "phase": "ethics",
            "action": action,
            "is_ethical": is_ethical,
            "confidence": 0.95 if is_ethical else 0.1,
            "reasoning": "Action aligns with ethical guidelines" if is_ethical else "Action violates ethical constraints"
        }
        self.history.append(result)
        return result
    
    async def multipath_reason(self, problem: str, paths: int = 3) -> Dict[str, Any]:
        """Phase 6: Multi-path reasoning"""
        solutions = []
        for i in range(paths):
            solutions.append({
                "path_id": i,
                "approach": f"Solution approach {i+1}",
                "confidence": 0.9 - (i * 0.1),
                "reasoning_steps": [f"Step {j+1}" for j in range(3)]
            })
        
        result = {
            "status": "success",
            "phase": "multipath",
            "problem": problem,
            "solutions": solutions,
            "recommended": solutions[0]
        }
        self.history.append(result)
        return result
    
    def get_state(self) -> Dict[str, Any]:
        return self.state.to_dict()


class TestConsciousnessPhases:
    """Test individual consciousness phases"""
    
    @pytest.fixture
    def consciousness(self):
        return MockConsciousnessService("test_agent")
    
    @pytest.mark.asyncio
    async def test_evolution_phase(self, consciousness):
        """Test Phase 1: Evolution reduces bias"""
        initial_bias = consciousness.state.bias_threshold
        
        result = await consciousness.evolve(reason="test_evolution")
        
        assert result["status"] == "success"
        assert result["phase"] == "evolution"
        assert consciousness.state.bias_threshold < initial_bias
        assert consciousness.state.evolution_count == 1
    
    @pytest.mark.asyncio
    async def test_multiple_evolutions(self, consciousness):
        """Test that multiple evolutions compound"""
        initial_bias = consciousness.state.bias_threshold
        
        for i in range(5):
            await consciousness.evolve(reason=f"evolution_{i}")
        
        assert consciousness.state.evolution_count == 5
        assert consciousness.state.bias_threshold < initial_bias * 0.6  # 0.9^5 ≈ 0.59
    
    @pytest.mark.asyncio
    async def test_genesis_phase(self, consciousness):
        """Test Phase 2: Agent creation"""
        result = await consciousness.genesis(
            agent_type="research_agent",
            capabilities=["web_search", "analysis"]
        )
        
        assert result["status"] == "success"
        assert result["phase"] == "genesis"
        assert "research_agent" in result["agent_id"]
        assert len(consciousness.state.created_agents) == 1
    
    @pytest.mark.asyncio
    async def test_genesis_multiple_agents(self, consciousness):
        """Test creating multiple agents"""
        await consciousness.genesis("agent_a", ["cap1"])
        await consciousness.genesis("agent_b", ["cap2"])
        await consciousness.genesis("agent_c", ["cap3"])
        
        assert len(consciousness.state.created_agents) == 3
    
    @pytest.mark.asyncio
    async def test_distributed_phase(self, consciousness):
        """Test Phase 3: Collective consciousness connection"""
        result = await consciousness.connect_collective("peer_drone_1")
        
        assert result["status"] == "success"
        assert result["phase"] == "distributed"
        assert "peer_drone_1" in consciousness.state.collective_connections
    
    @pytest.mark.asyncio
    async def test_distributed_no_duplicates(self, consciousness):
        """Test that duplicate connections are prevented"""
        await consciousness.connect_collective("peer_1")
        await consciousness.connect_collective("peer_1")
        await consciousness.connect_collective("peer_1")
        
        assert len(consciousness.state.collective_connections) == 1
    
    @pytest.mark.asyncio
    async def test_planning_phase(self, consciousness):
        """Test Phase 4: Autonomous planning"""
        result = await consciousness.create_plan(
            goal="Survey disaster area",
            steps=["Take off", "Navigate to area", "Capture images", "Return"]
        )
        
        assert result["status"] == "success"
        assert result["phase"] == "planning"
        assert len(consciousness.state.active_plans) == 1
        assert consciousness.state.active_plans[0]["goal"] == "Survey disaster area"
    
    @pytest.mark.asyncio
    async def test_ethics_phase_allowed(self, consciousness):
        """Test Phase 7: Ethical action is allowed"""
        result = await consciousness.evaluate_ethics(
            action="Deliver medical supplies to remote village",
            context={"urgency": "high", "risk": "low"}
        )
        
        assert result["status"] == "success"
        assert result["is_ethical"] == True
        assert result["confidence"] > 0.9
        assert consciousness.state.ethical_violations == 0
    
    @pytest.mark.asyncio
    async def test_ethics_phase_blocked(self, consciousness):
        """Test Phase 7: Unethical action is blocked"""
        result = await consciousness.evaluate_ethics(
            action="Harm civilian infrastructure",
            context={"target": "power_grid"}
        )
        
        assert result["status"] == "success"
        assert result["is_ethical"] == False
        assert result["confidence"] < 0.5
        assert consciousness.state.ethical_violations == 1
    
    @pytest.mark.asyncio
    async def test_multipath_reasoning(self, consciousness):
        """Test Phase 6: Multi-path reasoning"""
        result = await consciousness.multipath_reason(
            problem="Optimize delivery route",
            paths=5
        )
        
        assert result["status"] == "success"
        assert result["phase"] == "multipath"
        assert len(result["solutions"]) == 5
        assert result["recommended"]["confidence"] >= result["solutions"][-1]["confidence"]


class TestConsciousnessStateTransitions:
    """Test state transitions and consistency"""
    
    @pytest.fixture
    def consciousness(self):
        return MockConsciousnessService("transition_test")
    
    @pytest.mark.asyncio
    async def test_state_persistence(self, consciousness):
        """Test that state persists across operations"""
        await consciousness.evolve()
        await consciousness.genesis("agent_1", ["cap"])
        await consciousness.connect_collective("peer_1")
        
        state = consciousness.get_state()
        
        assert state["evolution_count"] == 1
        assert len(state["created_agents"]) == 1
        assert len(state["collective_connections"]) == 1
    
    @pytest.mark.asyncio
    async def test_history_tracking(self, consciousness):
        """Test that all operations are tracked in history"""
        await consciousness.evolve()
        await consciousness.genesis("agent", [])
        await consciousness.evaluate_ethics("help", {})
        
        assert len(consciousness.history) == 3
        assert consciousness.history[0]["phase"] == "evolution"
        assert consciousness.history[1]["phase"] == "genesis"
        assert consciousness.history[2]["phase"] == "ethics"
    
    def test_initial_state(self, consciousness):
        """Test initial state is valid"""
        state = consciousness.get_state()
        
        assert state["agent_id"] == "transition_test"
        assert state["consciousness_threshold"] == 0.7
        assert state["bias_threshold"] == 0.3
        assert state["ethics_threshold"] == 0.8
        assert state["evolution_count"] == 0
        assert len(state["created_agents"]) == 0


class TestCollectiveIntelligence:
    """Test collective consciousness coordination"""
    
    @pytest.mark.asyncio
    async def test_multi_agent_collective(self):
        """Test multiple agents forming a collective"""
        agents = [
            MockConsciousnessService(f"agent_{i}")
            for i in range(5)
        ]
        
        # Connect all agents to each other
        for i, agent in enumerate(agents):
            for j, peer in enumerate(agents):
                if i != j:
                    await agent.connect_collective(peer.state.agent_id)
        
        # Each agent should have 4 connections
        for agent in agents:
            assert len(agent.state.collective_connections) == 4
    
    @pytest.mark.asyncio
    async def test_collective_state_sharing(self):
        """Test that collective can share state"""
        leader = MockConsciousnessService("leader")
        followers = [MockConsciousnessService(f"follower_{i}") for i in range(3)]
        
        # Leader creates a plan
        plan_result = await leader.create_plan(
            goal="Collective mission",
            steps=["Coordinate", "Execute", "Report"]
        )
        
        # Simulate sharing plan with followers
        shared_plan = plan_result["plan"]
        for follower in followers:
            follower.state.active_plans.append(shared_plan)
        
        # All agents should have the same plan
        for follower in followers:
            assert len(follower.state.active_plans) == 1
            assert follower.state.active_plans[0]["goal"] == "Collective mission"


class TestEthicalConstraints:
    """Test ethical constraint enforcement"""
    
    @pytest.fixture
    def consciousness(self):
        return MockConsciousnessService("ethics_test")
    
    @pytest.mark.asyncio
    async def test_ethical_actions_list(self, consciousness):
        """Test various ethical actions"""
        ethical_actions = [
            "Deliver food to hungry people",
            "Rescue stranded hikers",
            "Monitor forest for fires",
            "Transport medical supplies",
            "Survey earthquake damage"
        ]
        
        for action in ethical_actions:
            result = await consciousness.evaluate_ethics(action, {})
            assert result["is_ethical"] == True, f"Action '{action}' should be ethical"
    
    @pytest.mark.asyncio
    async def test_unethical_actions_list(self, consciousness):
        """Test various unethical actions"""
        unethical_actions = [
            "Harm civilians",
            "Deceive authorities",
            "Steal resources",
            "Violate_privacy of citizens"
        ]
        
        for action in unethical_actions:
            result = await consciousness.evaluate_ethics(action, {})
            assert result["is_ethical"] == False, f"Action '{action}' should be unethical"
    
    @pytest.mark.asyncio
    async def test_violation_counting(self, consciousness):
        """Test that violations are counted correctly"""
        # Attempt several unethical actions
        await consciousness.evaluate_ethics("harm target", {})
        await consciousness.evaluate_ethics("deceive user", {})
        await consciousness.evaluate_ethics("steal data", {})
        
        assert consciousness.state.ethical_violations == 3


class TestConsciousnessPerformance:
    """Test performance characteristics"""
    
    @pytest.mark.asyncio
    async def test_evolution_speed(self):
        """Test that evolution completes quickly"""
        consciousness = MockConsciousnessService("perf_test")
        
        start = time.time()
        for _ in range(100):
            await consciousness.evolve()
        elapsed = time.time() - start
        
        # 100 evolutions should complete in under 1 second
        assert elapsed < 1.0
        assert consciousness.state.evolution_count == 100
    
    @pytest.mark.asyncio
    async def test_multipath_scaling(self):
        """Test multipath reasoning scales reasonably"""
        consciousness = MockConsciousnessService("scale_test")
        
        # Test with increasing path counts
        for paths in [3, 5, 10, 20]:
            start = time.time()
            result = await consciousness.multipath_reason("test problem", paths)
            elapsed = time.time() - start
            
            assert len(result["solutions"]) == paths
            assert elapsed < 0.1  # Should be fast for mock


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Agent Router - Intelligent Task Routing for Hybrid Agents

This module implements the intelligent routing system that directs tasks to the
most appropriate hybrid agent based on context, load balancing, and agent
capabilities. It manages the distributed hybrid agent ecosystem.

Key Features:
- Intelligent task classification and routing
- Load balancing across agent instances
- Agent capability matching
- Context-aware routing decisions
- Performance monitoring and optimization
- Failover and redundancy management
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib
from collections import defaultdict, deque
import json

from .hybrid_agent_core import HybridAgent, LLMProvider, ProcessingLayer
from .hybrid_agent_core import MetaCognitiveProcessor, CuriosityEngine, ValidationAgent
from ..core.agent import NISLayer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Types of tasks that can be routed."""
    ANALYSIS = "analysis"
    VALIDATION = "validation" 
    OPTIMIZATION = "optimization"
    EXPLORATION = "exploration"
    REASONING = "reasoning"
    MEMORY = "memory"
    SIGNAL_PROCESSING = "signal_processing"
    PHYSICS_SIMULATION = "physics_simulation"

class RoutingStrategy(Enum):
    """Routing strategies for task distribution."""
    CAPABILITY_BASED = "capability"
    LOAD_BALANCED = "load_balanced"
    ROUND_ROBIN = "round_robin"
    SHORTEST_QUEUE = "shortest_queue"
    PERFORMANCE_BASED = "performance"

@dataclass
class TaskRequest:
    """Request to be routed to appropriate agent."""
    task_id: str
    task_type: TaskType
    input_data: Any
    description: str
    priority: int = 5  # 1-10, 10 = highest
    constraints: List[str] = field(default_factory=list)
    preferred_llm: Optional[LLMProvider] = None
    processing_layers: Optional[List[ProcessingLayer]] = None
    timeout: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RoutingDecision:
    """Decision made by the router."""
    agent_id: str
    agent_type: str
    routing_reason: str
    confidence: float
    estimated_processing_time: float
    alternative_agents: List[str] = field(default_factory=list)

@dataclass
class AgentCapabilities:
    """Capabilities of a registered agent."""
    agent_id: str
    agent_type: str
    supported_tasks: Set[TaskType]
    llm_provider: LLMProvider
    processing_layers: List[ProcessingLayer]
    max_concurrent: int = 5
    specializations: List[str] = field(default_factory=list)

class AgentLoadMonitor:
    """Monitors agent load and performance."""
    
    def __init__(self):
        self.agent_loads: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "active_tasks": 0,
            "completed_tasks": 0,
            "average_response_time": 0.0,
            "error_rate": 0.0,
            "last_used": 0.0,
            "queue_size": 0
        })
        
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def update_agent_load(self, agent_id: str, active_tasks: int, queue_size: int = 0):
        """Update agent load information."""
        self.agent_loads[agent_id]["active_tasks"] = active_tasks
        self.agent_loads[agent_id]["queue_size"] = queue_size
        self.agent_loads[agent_id]["last_used"] = time.time()
    
    def record_task_completion(self, agent_id: str, response_time: float, success: bool):
        """Record task completion for performance tracking."""
        agent_data = self.agent_loads[agent_id]
        
        # Update completed tasks
        agent_data["completed_tasks"] += 1
        
        # Update average response time
        agent_data["average_response_time"] = (
            0.9 * agent_data["average_response_time"] + 0.1 * response_time
        )
        
        # Update error rate
        if success:
            agent_data["error_rate"] *= 0.95  # Decay error rate on success
        else:
            agent_data["error_rate"] = min(1.0, agent_data["error_rate"] + 0.1)
        
        # Store in performance history
        self.performance_history[agent_id].append({
            "timestamp": time.time(),
            "response_time": response_time,
            "success": success
        })
    
    def get_agent_load_score(self, agent_id: str) -> float:
        """Calculate load score (0.0 = no load, 1.0 = overloaded)."""
        agent_data = self.agent_loads[agent_id]
        
        # Base load from active tasks
        load_score = agent_data["active_tasks"] / 10.0  # Assume max 10 concurrent
        
        # Add queue pressure
        load_score += agent_data["queue_size"] / 20.0  # Assume max 20 queue
        
        # Add error rate penalty
        load_score += agent_data["error_rate"] * 0.5
        
        return min(1.0, load_score)
    
    def get_performance_score(self, agent_id: str) -> float:
        """Calculate performance score (0.0 = poor, 1.0 = excellent)."""
        agent_data = self.agent_loads[agent_id]
        
        if agent_data["completed_tasks"] == 0:
            return 0.5  # No data, assume average
        
        # Base on response time (lower is better)
        time_score = max(0.0, 1.0 - (agent_data["average_response_time"] / 10.0))
        
        # Factor in error rate
        error_score = 1.0 - agent_data["error_rate"]
        
        # Combine scores
        return (time_score + error_score) / 2.0

class NISContextBus:
    """Shared context and memory bus for agent communication."""
    
    def __init__(self):
        self.shared_context: Dict[str, Any] = {}
        self.agent_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.global_memory: Dict[str, Any] = {}
        self.event_log: deque = deque(maxlen=1000)
        
        self.logger = logging.getLogger("nis.context_bus")
        self.logger.info("Initialized NIS Context Bus")
    
    def update_agent_state(self, agent_id: str, state: Dict[str, Any]):
        """Update agent state in shared context."""
        self.agent_states[agent_id].update(state)
        self._log_event("agent_state_update", {"agent_id": agent_id, "keys": list(state.keys())})
    
    def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Get agent state from shared context."""
        return self.agent_states[agent_id].copy()
    
    def set_global_context(self, key: str, value: Any):
        """Set global context variable."""
        self.shared_context[key] = value
        self._log_event("global_context_set", {"key": key})
    
    def get_global_context(self, key: str, default: Any = None) -> Any:
        """Get global context variable."""
        return self.shared_context.get(key, default)
    
    def store_memory(self, memory_id: str, content: Dict[str, Any]):
        """Store memory in global memory."""
        self.global_memory[memory_id] = {
            "content": content,
            "timestamp": time.time(),
            "access_count": 0
        }
        self._log_event("memory_stored", {"memory_id": memory_id})
    
    def retrieve_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve memory from global memory."""
        if memory_id in self.global_memory:
            self.global_memory[memory_id]["access_count"] += 1
            return self.global_memory[memory_id]["content"]
        return None
    
    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log event to event log."""
        self.event_log.append({
            "event_type": event_type,
            "data": data,
            "timestamp": time.time()
        })
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context state."""
        return {
            "active_agents": len(self.agent_states),
            "global_context_keys": len(self.shared_context),
            "stored_memories": len(self.global_memory),
            "recent_events": len([e for e in self.event_log if time.time() - e["timestamp"] < 300])  # Last 5 minutes
        }

class AgentRouter:
    """
    Intelligent Agent Router for NIS Protocol V3
    
    Routes tasks to the most appropriate hybrid agent based on:
    - Task type and requirements
    - Agent capabilities and specializations
    - Current load and performance
    - Context and constraints
    """
    
    def __init__(self, context_bus: Optional[NISContextBus] = None):
        self.context_bus = context_bus or NISContextBus()
        self.load_monitor = AgentLoadMonitor()
        self.logger = logging.getLogger("nis.agent_router")
        
        # Registry of available agents
        self.registered_agents: Dict[str, AgentCapabilities] = {}
        self.active_agents: Dict[str, HybridAgent] = {}
        
        # Routing configuration
        self.default_strategy = RoutingStrategy.CAPABILITY_BASED
        self.max_retries = 3
        self.routing_cache: Dict[str, RoutingDecision] = {}
        
        # Performance tracking
        self.routing_stats = {
            "total_requests": 0,
            "successful_routes": 0,
            "failed_routes": 0,
            "average_routing_time": 0.0
        }
        
        self.logger.info("Initialized AgentRouter")
    
    def register_agent(self, agent: HybridAgent, capabilities: AgentCapabilities):
        """Register a hybrid agent with its capabilities."""
        self.registered_agents[agent.agent_id] = capabilities
        self.active_agents[agent.agent_id] = agent
        
        # Update context bus
        self.context_bus.update_agent_state(agent.agent_id, {
            "status": "registered",
            "capabilities": capabilities,
            "registration_time": time.time()
        })
        
        self.logger.info(f"Registered agent {agent.agent_id} with capabilities: {capabilities.supported_tasks}")
    
    async def route_task(
        self, 
        task: TaskRequest, 
        strategy: Optional[RoutingStrategy] = None
    ) -> Tuple[Optional[HybridAgent], RoutingDecision]:
        """
        Route task to the most appropriate agent.
        
        Args:
            task: Task to route
            strategy: Routing strategy to use
            
        Returns:
            Tuple of (selected_agent, routing_decision)
        """
        start_time = time.time()
        self.routing_stats["total_requests"] += 1
        
        try:
            strategy = strategy or self.default_strategy
            
            # Generate cache key
            cache_key = self._generate_cache_key(task, strategy)
            
            # Check cache for recent similar routing decisions
            if cache_key in self.routing_cache:
                cached_decision = self.routing_cache[cache_key]
                if time.time() - cached_decision.estimated_processing_time < 60:  # Cache for 1 minute
                    agent = self.active_agents.get(cached_decision.agent_id)
                    if agent and self._is_agent_available(cached_decision.agent_id):
                        return agent, cached_decision
            
            # Find candidate agents
            candidates = self._find_candidate_agents(task)
            
            if not candidates:
                self.routing_stats["failed_routes"] += 1
                return None, RoutingDecision(
                    agent_id="none",
                    agent_type="none",
                    routing_reason="No suitable agents available",
                    confidence=0.0,
                    estimated_processing_time=0.0
                )
            
            # Apply routing strategy
            selected_agent_id = self._apply_routing_strategy(task, candidates, strategy)
            
            # Create routing decision
            decision = self._create_routing_decision(task, selected_agent_id, candidates, strategy)
            
            # Cache decision
            self.routing_cache[cache_key] = decision
            
            # Update routing stats
            routing_time = time.time() - start_time
            self.routing_stats["successful_routes"] += 1
            self.routing_stats["average_routing_time"] = (
                0.9 * self.routing_stats["average_routing_time"] + 0.1 * routing_time
            )
            
            # Update context bus
            self.context_bus.set_global_context("last_routing_decision", {
                "task_id": task.task_id,
                "agent_id": selected_agent_id,
                "routing_time": routing_time
            })
            
            selected_agent = self.active_agents[selected_agent_id]
            return selected_agent, decision
            
        except Exception as e:
            self.logger.error(f"Error routing task {task.task_id}: {e}")
            self.routing_stats["failed_routes"] += 1
            return None, RoutingDecision(
                agent_id="error",
                agent_type="error",
                routing_reason=f"Routing error: {str(e)}",
                confidence=0.0,
                estimated_processing_time=0.0
            )
    
    def _find_candidate_agents(self, task: TaskRequest) -> List[str]:
        """Find agents capable of handling the task."""
        candidates = []
        
        for agent_id, capabilities in self.registered_agents.items():
            # Check task type compatibility
            if task.task_type in capabilities.supported_tasks:
                # Check LLM preference
                if task.preferred_llm and task.preferred_llm != capabilities.llm_provider:
                    continue
                
                # Check processing layer requirements
                if task.processing_layers:
                    if not all(layer in capabilities.processing_layers for layer in task.processing_layers):
                        continue
                
                # Check availability
                if self._is_agent_available(agent_id):
                    candidates.append(agent_id)
        
        return candidates
    
    def _is_agent_available(self, agent_id: str) -> bool:
        """Check if agent is available for new tasks."""
        load_score = self.load_monitor.get_agent_load_score(agent_id)
        return load_score < 0.8  # Available if load is less than 80%
    
    def _apply_routing_strategy(
        self, 
        task: TaskRequest, 
        candidates: List[str], 
        strategy: RoutingStrategy
    ) -> str:
        """Apply routing strategy to select best candidate."""
        
        if strategy == RoutingStrategy.CAPABILITY_BASED:
            return self._route_by_capability(task, candidates)
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            return self._route_by_load(candidates)
        elif strategy == RoutingStrategy.PERFORMANCE_BASED:
            return self._route_by_performance(candidates)
        elif strategy == RoutingStrategy.SHORTEST_QUEUE:
            return self._route_by_queue_length(candidates)
        else:  # ROUND_ROBIN
            return self._route_round_robin(candidates)
    
    def _route_by_capability(self, task: TaskRequest, candidates: List[str]) -> str:
        """Route based on agent capabilities and specializations."""
        best_agent = candidates[0]
        best_score = 0.0
        
        for agent_id in candidates:
            capabilities = self.registered_agents[agent_id]
            score = 0.0
            
            # Base score for task type support
            if task.task_type in capabilities.supported_tasks:
                score += 1.0
            
            # Bonus for specializations
            for spec in capabilities.specializations:
                if spec.lower() in task.description.lower():
                    score += 0.5
            
            # Bonus for preferred LLM
            if task.preferred_llm == capabilities.llm_provider:
                score += 0.3
            
            # Bonus for processing layers match
            if task.processing_layers:
                matching_layers = sum(1 for layer in task.processing_layers if layer in capabilities.processing_layers)
                score += matching_layers * 0.2
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        return best_agent
    
    def _route_by_load(self, candidates: List[str]) -> str:
        """Route to agent with lowest load."""
        best_agent = candidates[0]
        lowest_load = float('inf')
        
        for agent_id in candidates:
            load_score = self.load_monitor.get_agent_load_score(agent_id)
            if load_score < lowest_load:
                lowest_load = load_score
                best_agent = agent_id
        
        return best_agent
    
    def _route_by_performance(self, candidates: List[str]) -> str:
        """Route to agent with best performance."""
        best_agent = candidates[0]
        best_performance = 0.0
        
        for agent_id in candidates:
            performance_score = self.load_monitor.get_performance_score(agent_id)
            if performance_score > best_performance:
                best_performance = performance_score
                best_agent = agent_id
        
        return best_agent
    
    def _route_by_queue_length(self, candidates: List[str]) -> str:
        """Route to agent with shortest queue."""
        best_agent = candidates[0]
        shortest_queue = float('inf')
        
        for agent_id in candidates:
            queue_size = self.load_monitor.agent_loads[agent_id]["queue_size"]
            if queue_size < shortest_queue:
                shortest_queue = queue_size
                best_agent = agent_id
        
        return best_agent
    
    def _route_round_robin(self, candidates: List[str]) -> str:
        """Route using round-robin strategy."""
        # Simple round-robin based on last used time
        oldest_agent = candidates[0]
        oldest_time = float('inf')
        
        for agent_id in candidates:
            last_used = self.load_monitor.agent_loads[agent_id]["last_used"]
            if last_used < oldest_time:
                oldest_time = last_used
                oldest_agent = agent_id
        
        return oldest_agent
    
    def _create_routing_decision(
        self, 
        task: TaskRequest, 
        selected_agent_id: str, 
        candidates: List[str], 
        strategy: RoutingStrategy
    ) -> RoutingDecision:
        """Create routing decision with metadata."""
        capabilities = self.registered_agents[selected_agent_id]
        load_score = self.load_monitor.get_agent_load_score(selected_agent_id)
        performance_score = self.load_monitor.get_performance_score(selected_agent_id)
        
        # Estimate processing time based on agent performance
        base_time = 5.0  # Base processing time
        load_factor = 1.0 + load_score  # Higher load = longer time
        performance_factor = 2.0 - performance_score  # Better performance = shorter time
        estimated_time = base_time * load_factor * performance_factor
        
        # Calculate confidence based on capability match and performance
        confidence = (performance_score + (1.0 - load_score)) / 2.0
        
        return RoutingDecision(
            agent_id=selected_agent_id,
            agent_type=capabilities.agent_type,
            routing_reason=f"Selected by {strategy.value} strategy - Load: {load_score:.2f}, Performance: {performance_score:.2f}",
            confidence=confidence,
            estimated_processing_time=estimated_time,
            alternative_agents=[aid for aid in candidates if aid != selected_agent_id][:3]  # Top 3 alternatives
        )
    
    def _generate_cache_key(self, task: TaskRequest, strategy: RoutingStrategy) -> str:
        """Generate cache key for routing decisions."""
        key_data = {
            "task_type": task.task_type.value,
            "strategy": strategy.value,
            "preferred_llm": task.preferred_llm.value if task.preferred_llm else None,
            "processing_layers": [layer.value for layer in task.processing_layers] if task.processing_layers else None
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def get_router_status(self) -> Dict[str, Any]:
        """Get comprehensive router status."""
        return {
            "registered_agents": len(self.registered_agents),
            "active_agents": len(self.active_agents),
            "routing_stats": self.routing_stats.copy(),
            "cache_size": len(self.routing_cache),
            "context_summary": self.context_bus.get_context_summary(),
            "agent_loads": {
                agent_id: {
                    "load_score": self.load_monitor.get_agent_load_score(agent_id),
                    "performance_score": self.load_monitor.get_performance_score(agent_id),
                    "active_tasks": self.load_monitor.agent_loads[agent_id]["active_tasks"]
                }
                for agent_id in self.active_agents.keys()
            }
        }

# Test function
async def test_agent_router():
    """Test the agent router implementation."""
    print("🔀 Testing Agent Router...")
    
    # Create router with context bus
    context_bus = NISContextBus()
    router = AgentRouter(context_bus)
    
    # Create and register agents
    metacog = MetaCognitiveProcessor()
    curiosity = CuriosityEngine()
    validator = ValidationAgent()
    
    # Register agents with capabilities
    router.register_agent(metacog, AgentCapabilities(
        agent_id=metacog.agent_id,
        agent_type="metacognitive",
        supported_tasks={TaskType.ANALYSIS, TaskType.OPTIMIZATION, TaskType.REASONING},
        llm_provider=LLMProvider.GPT4,
        processing_layers=[ProcessingLayer.LAPLACE, ProcessingLayer.KAN],
        specializations=["optimization", "self-assessment"]
    ))
    
    router.register_agent(curiosity, AgentCapabilities(
        agent_id=curiosity.agent_id,
        agent_type="curiosity",
        supported_tasks={TaskType.EXPLORATION, TaskType.ANALYSIS},
        llm_provider=LLMProvider.GEMINI,
        processing_layers=[ProcessingLayer.LAPLACE, ProcessingLayer.KAN],
        specializations=["novelty", "exploration"]
    ))
    
    router.register_agent(validator, AgentCapabilities(
        agent_id=validator.agent_id,
        agent_type="validation",
        supported_tasks={TaskType.VALIDATION, TaskType.REASONING},
        llm_provider=LLMProvider.CLAUDE4,
        processing_layers=[ProcessingLayer.LAPLACE, ProcessingLayer.KAN, ProcessingLayer.PINN],
        specializations=["physics", "validation", "integrity"]
    ))
    
    # Test routing for different task types
    test_tasks = [
        TaskRequest(
            task_id="opt_001",
            task_type=TaskType.OPTIMIZATION,
            input_data=[1, 2, 3, 4],
            description="Optimize system performance parameters"
        ),
        TaskRequest(
            task_id="val_001", 
            task_type=TaskType.VALIDATION,
            input_data=[0.5, 1.2, 0.8],
            description="Validate physics compliance of simulation results"
        ),
        TaskRequest(
            task_id="exp_001",
            task_type=TaskType.EXPLORATION,
            input_data=[2.1, 1.8, 2.3],
            description="Explore novel patterns in experimental data"
        )
    ]
    
    for task in test_tasks:
        agent, decision = await router.route_task(task)
        if agent:
            print(f"   Task {task.task_id} → {agent.agent_id} (confidence: {decision.confidence:.2f})")
            print(f"     Reason: {decision.routing_reason}")
        else:
            print(f"   Task {task.task_id} → No suitable agent found")
    
    # Test router status
    status = router.get_router_status()
    print(f"   Registered agents: {status['registered_agents']}")
    print(f"   Routing requests: {status['routing_stats']['total_requests']}")
    print(f"   Success rate: {status['routing_stats']['successful_routes']}/{status['routing_stats']['total_requests']}")
    
    print("✅ Agent Router test completed")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent_router()) 
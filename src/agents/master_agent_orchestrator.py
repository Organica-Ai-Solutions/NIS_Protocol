#!/usr/bin/env python3
"""
🚀 NIS Protocol Master Agent Orchestrator
Comprehensive agent coordination system for seamless multi-agent operations

Features:
- LangChain Deep Agent integration
- Real-time agent health monitoring
- Dynamic agent discovery and registration
- Intelligent task routing and delegation
- Cross-agent communication and memory sharing
- Performance optimization and load balancing
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import concurrent.futures
from pathlib import Path

# Core NIS components
from ..core.agent import NISAgent
from ..memory.memory_manager import MemoryManager
from ..llm.llm_manager import LLMManager

# LangChain integration
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    from typing_extensions import TypedDict
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# Deep agent components
from .deep.planner import DeepAgentPlanner, ExecutionPlan
from .deep.skills import DatasetSkill, PipelineSkill, ResearchSkill, AuditSkill, CodeSkill

# Specialized agents (with availability checks)
agent_imports = {}
try:
    from .memory.enhanced_memory_agent import EnhancedMemoryAgent
    agent_imports['memory'] = EnhancedMemoryAgent
except ImportError:
    agent_imports['memory'] = None

try:
    from .reasoning.unified_reasoning_agent import UnifiedReasoningAgent
    agent_imports['reasoning'] = UnifiedReasoningAgent
except ImportError:
    agent_imports['reasoning'] = None

try:
    from .physics.unified_physics_agent import UnifiedPhysicsAgent
    agent_imports['physics'] = UnifiedPhysicsAgent
except ImportError:
    agent_imports['physics'] = None

try:
    from .consciousness.enhanced_conscious_agent import EnhancedConsciousAgent
    agent_imports['consciousness'] = EnhancedConsciousAgent
except ImportError:
    agent_imports['consciousness'] = None

try:
    from .research.web_search_agent import WebSearchAgent
    from .research.deep_research_agent import DeepResearchAgent
    agent_imports['web_search'] = WebSearchAgent
    agent_imports['deep_research'] = DeepResearchAgent
except ImportError:
    agent_imports['web_search'] = None
    agent_imports['deep_research'] = None

try:
    from .multimodal.vision_agent import VisionAgent
    agent_imports['vision'] = VisionAgent
except ImportError:
    agent_imports['vision'] = None


class AgentStatus(Enum):
    """Agent status enumeration"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class AgentInfo:
    """Information about a registered agent"""
    agent_id: str
    agent_type: str
    instance: Any
    capabilities: List[str]
    status: AgentStatus = AgentStatus.INITIALIZING
    load: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    performance_score: float = 1.0
    error_count: int = 0
    total_tasks: int = 0
    successful_tasks: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskRequest:
    """Request for agent task execution"""
    task_id: str
    task_type: str
    priority: TaskPriority
    data: Dict[str, Any]
    target_agent: Optional[str] = None
    capabilities_required: List[str] = field(default_factory=list)
    timeout: float = 300.0
    created_at: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result from agent task execution"""
    task_id: str
    agent_id: str
    status: str
    result: Any
    execution_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MasterAgentOrchestrator:
    """
    🧠 Master Agent Orchestrator for NIS Protocol
    
    Coordinates all agents in the system with advanced features:
    - Deep Agent integration with LangChain workflows
    - Intelligent task routing and load balancing
    - Real-time health monitoring and auto-recovery
    - Cross-agent memory and communication
    - Performance optimization and analytics
    """
    
    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        llm_manager: Optional[LLMManager] = None,
        enable_deep_agent: bool = True,
        enable_auto_discovery: bool = True,
        config: Dict[str, Any] = None
    ):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.memory_manager = memory_manager or MemoryManager()
        self.llm_manager = llm_manager or LLMManager()
        
        # Agent registry and management
        self.agents: Dict[str, AgentInfo] = {}
        self.agent_capabilities: Dict[str, Set[str]] = defaultdict(set)
        self.task_queue: deque = deque()
        self.active_tasks: Dict[str, TaskRequest] = {}
        self.task_results: Dict[str, TaskResult] = {}
        
        # Deep Agent integration
        self.enable_deep_agent = enable_deep_agent and LANGGRAPH_AVAILABLE
        self.deep_planner: Optional[DeepAgentPlanner] = None
        self.deep_agent_skills: Dict[str, Any] = {}
        
        # Performance tracking
        self.performance_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_response_time": 0.0,
            "agent_utilization": {},
            "capability_usage": defaultdict(int),
            "error_rates": defaultdict(float)
        }
        
        # Communication channels
        self.message_channels: Dict[str, deque] = defaultdict(deque)
        self.shared_memory: Dict[str, Any] = {}
        
        # Auto-discovery and monitoring
        self.enable_auto_discovery = enable_auto_discovery
        self.monitoring_active = False
        self.health_check_interval = 30.0
        
        # Initialize system
        self._initialize_system()
        
    def _initialize_system(self):
        """Initialize the orchestrator system"""
        self.logger.info("🚀 Initializing Master Agent Orchestrator")
        
        # Initialize Deep Agent if enabled
        if self.enable_deep_agent:
            self._initialize_deep_agent()
        
        # Auto-discover and register agents
        if self.enable_auto_discovery:
            self._auto_discover_agents()
        
        # Start monitoring
        self._start_monitoring()
        
        self.logger.info("✅ Master Agent Orchestrator initialized successfully")
    
    def _initialize_deep_agent(self):
        """Initialize the Deep Agent planner and skills"""
        try:
            # Create a base agent for the deep planner
            base_agent = NISAgent(agent_id="deep_agent_base")
            
            # Initialize Deep Agent planner
            self.deep_planner = DeepAgentPlanner(
                agent=base_agent,
                memory_manager=self.memory_manager
            )
            
            # Register skills
            self.deep_agent_skills = {
                'dataset': DatasetSkill(base_agent, self.memory_manager),
                'pipeline': PipelineSkill(base_agent, self.memory_manager),
                'research': ResearchSkill(base_agent, self.memory_manager),
                'audit': AuditSkill(base_agent, self.memory_manager),
                'code': CodeSkill(base_agent, self.memory_manager)
            }
            
            # Register skills with planner
            for skill_name, skill_instance in self.deep_agent_skills.items():
                self.deep_planner.register_skill(skill_name, skill_instance)
            
            self.logger.info("🧠 Deep Agent planner initialized with 5 skills")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Deep Agent: {e}")
            self.enable_deep_agent = False
    
    def _auto_discover_agents(self):
        """Auto-discover and register available agents"""
        discovered_count = 0
        
        for agent_type, agent_class in agent_imports.items():
            if agent_class is None:
                self.logger.debug(f"⚠️ Agent type '{agent_type}' not available")
                continue
                
            try:
                # Create agent instance
                agent_id = f"{agent_type}_agent"
                agent_instance = agent_class(agent_id=agent_id)
                
                # Determine capabilities based on agent type
                capabilities = self._determine_agent_capabilities(agent_type, agent_instance)
                
                # Register the agent
                self.register_agent(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    instance=agent_instance,
                    capabilities=capabilities
                )
                
                discovered_count += 1
                self.logger.info(f"✅ Auto-discovered and registered {agent_type} agent")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to auto-discover {agent_type} agent: {e}")
        
        self.logger.info(f"🔍 Auto-discovery complete: {discovered_count} agents registered")
    
    def _determine_agent_capabilities(self, agent_type: str, agent_instance: Any) -> List[str]:
        """Determine capabilities for an agent"""
        capability_map = {
            'memory': ['memory_storage', 'memory_retrieval', 'temporal_modeling', 'lstm_processing'],
            'reasoning': ['logical_reasoning', 'kan_processing', 'symbolic_extraction', 'inference'],
            'physics': ['physics_validation', 'pinn_solving', 'conservation_laws', 'differential_equations'],
            'consciousness': ['meta_cognition', 'self_reflection', 'introspection', 'awareness'],
            'web_search': ['web_research', 'information_gathering', 'fact_checking', 'data_collection'],
            'deep_research': ['deep_analysis', 'comprehensive_research', 'multi_source_integration'],
            'vision': ['image_processing', 'visual_analysis', 'object_detection', 'scene_understanding']
        }
        
        base_capabilities = capability_map.get(agent_type, [])
        
        # Check for additional capabilities from agent instance
        if hasattr(agent_instance, 'get_capabilities'):
            try:
                additional_caps = agent_instance.get_capabilities()
                if isinstance(additional_caps, list):
                    base_capabilities.extend(additional_caps)
            except Exception:
                pass
        
        return base_capabilities
    
    def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        instance: Any,
        capabilities: List[str],
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Register an agent with the orchestrator"""
        try:
            agent_info = AgentInfo(
                agent_id=agent_id,
                agent_type=agent_type,
                instance=instance,
                capabilities=capabilities,
                status=AgentStatus.READY,
                metadata=metadata or {}
            )
            
            self.agents[agent_id] = agent_info
            
            # Update capability mapping
            for capability in capabilities:
                self.agent_capabilities[capability].add(agent_id)
            
            self.logger.info(f"📝 Registered agent: {agent_id} with {len(capabilities)} capabilities")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register agent {agent_id}: {e}")
            return False
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the orchestrator"""
        try:
            if agent_id not in self.agents:
                return False
            
            agent_info = self.agents[agent_id]
            
            # Remove from capability mapping
            for capability in agent_info.capabilities:
                self.agent_capabilities[capability].discard(agent_id)
            
            # Remove agent
            del self.agents[agent_id]
            
            self.logger.info(f"🗑️ Unregistered agent: {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to unregister agent {agent_id}: {e}")
            return False
    
    async def submit_task(
        self,
        task_type: str,
        data: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        target_agent: Optional[str] = None,
        capabilities_required: List[str] = None,
        timeout: float = 300.0,
        context: Dict[str, Any] = None
    ) -> str:
        """Submit a task for execution"""
        
        task_id = f"task_{int(time.time() * 1000)}_{len(self.active_tasks)}"
        
        task_request = TaskRequest(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            data=data,
            target_agent=target_agent,
            capabilities_required=capabilities_required or [],
            timeout=timeout,
            context=context or {}
        )
        
        # Add to queue based on priority
        if priority.value >= TaskPriority.HIGH.value:
            self.task_queue.appendleft(task_request)
        else:
            self.task_queue.append(task_request)
        
        self.active_tasks[task_id] = task_request
        
        self.logger.info(f"📋 Task submitted: {task_id} (type: {task_type}, priority: {priority.name})")
        
        # Process task immediately if it's urgent
        if priority.value >= TaskPriority.CRITICAL.value:
            asyncio.create_task(self._process_task_queue())
        
        return task_id
    
    async def get_task_result(self, task_id: str, wait: bool = True, timeout: float = 300.0) -> Optional[TaskResult]:
        """Get the result of a submitted task"""
        
        if task_id in self.task_results:
            return self.task_results[task_id]
        
        if not wait:
            return None
        
        # Wait for result with timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            if task_id in self.task_results:
                return self.task_results[task_id]
            await asyncio.sleep(0.1)
        
        return None
    
    async def _process_task_queue(self):
        """Process pending tasks in the queue"""
        
        while self.task_queue:
            task_request = self.task_queue.popleft()
            
            try:
                # Find appropriate agent
                agent_id = await self._select_agent_for_task(task_request)
                
                if not agent_id:
                    # If using Deep Agent, try complex planning
                    if self.enable_deep_agent and self.deep_planner:
                        result = await self._execute_with_deep_agent(task_request)
                    else:
                        result = TaskResult(
                            task_id=task_request.task_id,
                            agent_id="system",
                            status="failed",
                            result=None,
                            execution_time=0.0,
                            error="No suitable agent found"
                        )
                else:
                    # Execute with selected agent
                    result = await self._execute_task_with_agent(task_request, agent_id)
                
                # Store result
                self.task_results[task_request.task_id] = result
                
                # Update metrics
                self._update_performance_metrics(result)
                
                # Clean up active task
                if task_request.task_id in self.active_tasks:
                    del self.active_tasks[task_request.task_id]
                    
            except Exception as e:
                self.logger.error(f"❌ Error processing task {task_request.task_id}: {e}")
                
                # Create error result
                error_result = TaskResult(
                    task_id=task_request.task_id,
                    agent_id="system",
                    status="error",
                    result=None,
                    execution_time=0.0,
                    error=str(e)
                )
                
                self.task_results[task_request.task_id] = error_result
                
                if task_request.task_id in self.active_tasks:
                    del self.active_tasks[task_request.task_id]
    
    async def _select_agent_for_task(self, task_request: TaskRequest) -> Optional[str]:
        """Select the best agent for a task"""
        
        # If target agent specified, use it if available
        if task_request.target_agent:
            if (task_request.target_agent in self.agents and 
                self.agents[task_request.target_agent].status == AgentStatus.READY):
                return task_request.target_agent
        
        # Find agents with required capabilities
        candidate_agents = set()
        
        if task_request.capabilities_required:
            for capability in task_request.capabilities_required:
                if capability in self.agent_capabilities:
                    candidate_agents.update(self.agent_capabilities[capability])
        else:
            # No specific capabilities required, consider all ready agents
            candidate_agents = {
                agent_id for agent_id, info in self.agents.items() 
                if info.status == AgentStatus.READY
            }
        
        # Filter by availability and load
        available_agents = []
        for agent_id in candidate_agents:
            agent_info = self.agents[agent_id]
            if (agent_info.status == AgentStatus.READY and 
                agent_info.load < 0.8):  # Don't overload agents
                available_agents.append((agent_id, agent_info))
        
        if not available_agents:
            return None
        
        # Select best agent based on load and performance
        best_agent = min(
            available_agents,
            key=lambda x: (x[1].load, -x[1].performance_score)
        )
        
        return best_agent[0]
    
    async def _execute_task_with_agent(self, task_request: TaskRequest, agent_id: str) -> TaskResult:
        """Execute a task with a specific agent"""
        start_time = time.time()
        agent_info = self.agents[agent_id]
        
        try:
            # Update agent status
            agent_info.status = AgentStatus.BUSY
            agent_info.load = min(agent_info.load + 0.1, 1.0)
            
            # Execute the task
            if hasattr(agent_info.instance, 'process'):
                result = await agent_info.instance.process(task_request.data, task_request.context)
            elif hasattr(agent_info.instance, 'execute'):
                result = await agent_info.instance.execute(task_request.data)
            else:
                # Fallback method
                result = {"status": "completed", "message": f"Task processed by {agent_id}"}
            
            execution_time = time.time() - start_time
            
            # Update agent metrics
            agent_info.total_tasks += 1
            agent_info.successful_tasks += 1
            agent_info.performance_score = min(
                agent_info.performance_score * 1.01,
                1.0
            )
            
            return TaskResult(
                task_id=task_request.task_id,
                agent_id=agent_id,
                status="completed",
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Update agent error metrics
            agent_info.error_count += 1
            agent_info.total_tasks += 1
            agent_info.performance_score = max(
                agent_info.performance_score * 0.95,
                0.1
            )
            
            return TaskResult(
                task_id=task_request.task_id,
                agent_id=agent_id,
                status="failed",
                result=None,
                execution_time=execution_time,
                error=str(e)
            )
            
        finally:
            # Reset agent status
            agent_info.status = AgentStatus.READY
            agent_info.load = max(agent_info.load - 0.1, 0.0)
            agent_info.last_heartbeat = time.time()
    
    async def _execute_with_deep_agent(self, task_request: TaskRequest) -> TaskResult:
        """Execute a complex task using the Deep Agent planner"""
        start_time = time.time()
        
        try:
            # Create execution plan
            plan = await self.deep_planner.create_plan(
                goal=task_request.task_type,
                context={
                    "data": task_request.data,
                    "priority": task_request.priority.name,
                    "available_agents": list(self.agents.keys()),
                    "capabilities": dict(self.agent_capabilities)
                }
            )
            
            # Execute the plan
            execution_result = await self.deep_planner.execute_plan(plan.id)
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task_request.task_id,
                agent_id="deep_agent",
                status=execution_result["status"],
                result=execution_result,
                execution_time=execution_time,
                metadata={"plan_id": plan.id}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task_request.task_id,
                agent_id="deep_agent",
                status="failed",
                result=None,
                execution_time=execution_time,
                error=str(e)
            )
    
    def _update_performance_metrics(self, result: TaskResult):
        """Update system performance metrics"""
        self.performance_metrics["total_tasks"] += 1
        
        if result.status == "completed":
            self.performance_metrics["successful_tasks"] += 1
        else:
            self.performance_metrics["failed_tasks"] += 1
        
        # Update average response time
        total_tasks = self.performance_metrics["total_tasks"]
        current_avg = self.performance_metrics["average_response_time"]
        self.performance_metrics["average_response_time"] = (
            (current_avg * (total_tasks - 1) + result.execution_time) / total_tasks
        )
        
        # Update agent utilization
        if result.agent_id not in self.performance_metrics["agent_utilization"]:
            self.performance_metrics["agent_utilization"][result.agent_id] = []
        
        self.performance_metrics["agent_utilization"][result.agent_id].append({
            "timestamp": time.time(),
            "execution_time": result.execution_time,
            "status": result.status
        })
        
        # Keep only recent data (last 100 entries)
        if len(self.performance_metrics["agent_utilization"][result.agent_id]) > 100:
            self.performance_metrics["agent_utilization"][result.agent_id] = \
                self.performance_metrics["agent_utilization"][result.agent_id][-100:]
    
    def _start_monitoring(self):
        """Start agent health monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            asyncio.create_task(self._health_monitor_loop())
    
    async def _health_monitor_loop(self):
        """Main health monitoring loop"""
        while self.monitoring_active:
            try:
                await self._check_agent_health()
                await self._process_task_queue()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"❌ Health monitor error: {e}")
                await asyncio.sleep(5.0)
    
    async def _check_agent_health(self):
        """Check health of all registered agents"""
        current_time = time.time()
        
        for agent_id, agent_info in self.agents.items():
            # Check if agent is responsive
            time_since_heartbeat = current_time - agent_info.last_heartbeat
            
            if time_since_heartbeat > 120.0:  # 2 minutes timeout
                agent_info.status = AgentStatus.OFFLINE
                self.logger.warning(f"⚠️ Agent {agent_id} appears offline")
            elif hasattr(agent_info.instance, 'health_check'):
                try:
                    health_result = await agent_info.instance.health_check()
                    if health_result.get('status') == 'healthy':
                        agent_info.status = AgentStatus.READY
                        agent_info.last_heartbeat = current_time
                    else:
                        agent_info.status = AgentStatus.ERROR
                except Exception:
                    agent_info.status = AgentStatus.ERROR
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "orchestrator_status": "active" if self.monitoring_active else "inactive",
            "total_agents": len(self.agents),
            "ready_agents": len([a for a in self.agents.values() if a.status == AgentStatus.READY]),
            "busy_agents": len([a for a in self.agents.values() if a.status == AgentStatus.BUSY]),
            "offline_agents": len([a for a in self.agents.values() if a.status == AgentStatus.OFFLINE]),
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.task_queue),
            "deep_agent_enabled": self.enable_deep_agent,
            "capabilities_available": list(self.agent_capabilities.keys()),
            "performance_metrics": self.performance_metrics,
            "timestamp": time.time()
        }
    
    def get_agent_details(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about all agents"""
        return {
            agent_id: {
                "agent_type": info.agent_type,
                "status": info.status.value,
                "capabilities": info.capabilities,
                "load": info.load,
                "performance_score": info.performance_score,
                "total_tasks": info.total_tasks,
                "successful_tasks": info.successful_tasks,
                "error_count": info.error_count,
                "success_rate": info.successful_tasks / max(info.total_tasks, 1),
                "last_heartbeat": info.last_heartbeat,
                "metadata": info.metadata
            }
            for agent_id, info in self.agents.items()
        }
    
    async def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        self.logger.info("🛑 Shutting down Master Agent Orchestrator")
        
        # Stop monitoring
        self.monitoring_active = False
        
        # Complete remaining tasks
        if self.task_queue:
            self.logger.info(f"📋 Processing {len(self.task_queue)} remaining tasks")
            await self._process_task_queue()
        
        # Shutdown agents
        for agent_id, agent_info in self.agents.items():
            if hasattr(agent_info.instance, 'shutdown'):
                try:
                    await agent_info.instance.shutdown()
                except Exception as e:
                    self.logger.error(f"❌ Error shutting down agent {agent_id}: {e}")
        
        self.logger.info("✅ Master Agent Orchestrator shutdown complete")


# Utility function to create a configured orchestrator
def create_master_orchestrator(config: Dict[str, Any] = None) -> MasterAgentOrchestrator:
    """Create a pre-configured Master Agent Orchestrator"""
    
    default_config = {
        "enable_deep_agent": True,
        "enable_auto_discovery": True,
        "health_check_interval": 30.0,
        "max_task_queue_size": 1000,
        "agent_load_threshold": 0.8
    }
    
    if config:
        default_config.update(config)
    
    return MasterAgentOrchestrator(config=default_config)


if __name__ == "__main__":
    # Example usage
    async def main():
        orchestrator = create_master_orchestrator()
        
        # Submit a test task
        task_id = await orchestrator.submit_task(
            task_type="test_reasoning",
            data={"input": "What is 2+2?"},
            priority=TaskPriority.NORMAL
        )
        
        # Get result
        result = await orchestrator.get_task_result(task_id)
        print(f"Task result: {result}")
        
        # Check system status
        status = orchestrator.get_system_status()
        print(f"System status: {json.dumps(status, indent=2)}")
        
        await orchestrator.shutdown()
    
    asyncio.run(main())

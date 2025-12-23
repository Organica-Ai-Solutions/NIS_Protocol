#!/usr/bin/env python3
"""
NIS Protocol Intelligent Agent Orchestrator
Brain-like agent coordination with smart activation and real-time monitoring

This system simulates human brain structure with:
- Always-active core agents (like autonomic nervous system)
- Context-activated specialized agents (like focused attention)
- Real-time state management and performance monitoring
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque

from .state_manager import nis_state_manager, StateEventType, emit_state_event

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent status types"""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    PROCESSING = "processing"
    IDLE = "idle"
    ERROR = "error"
    SLEEPING = "sleeping"

class AgentType(Enum):
    """Agent classification types"""
    CORE = "core"              # Always active (brain stem)
    SPECIALIZED = "specialized" # Context activated (cerebral cortex)
    PROTOCOL = "protocol"      # External communication (nervous system)
    LEARNING = "learning"      # Adaptive functions (hippocampus)
    MONITORING = "monitoring"  # System oversight (prefrontal cortex)

class ActivationTrigger(Enum):
    """How agents get activated"""
    ALWAYS = "always"
    CONTEXT = "context" 
    ON_DEMAND = "on_demand"
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"

class AgentAction(Enum):
    """Constrained action space - Cursor pattern for agent operations"""
    # Sensing
    READ_SENSOR = "read_sensor"
    QUERY_STATE = "query_state"
    
    # Memory
    QUERY_MEMORY = "query_memory"
    STORE_MEMORY = "store_memory"
    
    # Planning
    CREATE_PLAN = "create_plan"
    EXECUTE_PLAN = "execute_plan"
    
    # Tools
    RUN_TOOL = "run_tool"
    CALL_LLM = "call_llm"
    
    # Deployment
    DEPLOY_EDGE = "deploy_edge"
    UPDATE_CONFIG = "update_config"
    
    # Policy
    SET_POLICY = "set_policy"
    CHECK_POLICY = "check_policy"

@dataclass
class AgentMetrics:
    """Performance metrics for an agent"""
    activation_count: int = 0
    total_processing_time: float = 0.0
    average_response_time: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    last_active: Optional[float] = None
    resource_usage: Dict[str, float] = None
    health_check_count: int = 0  # Track number of health checks performed
    
    def __post_init__(self):
        if self.resource_usage is None:
            self.resource_usage = {"cpu": 0.0, "memory": 0.0, "tokens": 0}

@dataclass
class ActionDefinition:
    """Defines a constrained action with validation rules (Cursor pattern)"""
    action_type: AgentAction
    agent_id: str
    parameters: Dict[str, Any]
    requires_approval: bool = False
    rollback_enabled: bool = True
    timeout_seconds: float = 30.0
    audit_id: Optional[str] = None

@dataclass
class ActionResult:
    """Result of an action execution (Cursor pattern)"""
    action_id: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    audit_trail: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.audit_trail is None:
            self.audit_trail = []

@dataclass
class AgentDefinition:
    """Complete agent definition"""
    agent_id: str
    name: str
    agent_type: AgentType
    activation_trigger: ActivationTrigger
    status: AgentStatus = AgentStatus.INACTIVE
    context_keywords: List[str] = None
    dependencies: List[str] = None
    max_concurrent: int = 1
    timeout_seconds: float = 120.0  # Increased from 30 to 120 seconds for proper initialization
    priority: int = 5  # 1-10, higher = more important
    description: str = ""
    
    def __post_init__(self):
        if self.context_keywords is None:
            self.context_keywords = []
        if self.dependencies is None:
            self.dependencies = []

class NISAgentOrchestrator:
    """
    🧠 Intelligent Agent Orchestrator
    
    Simulates human brain architecture:
    - Core agents (brain stem): Always active, handle basic functions
    - Specialized agents (cortex): Activated by context/need
    - Protocol agents (nervous system): Handle external communication
    - Learning agents (hippocampus): Adapt and improve
    - Monitoring agents (prefrontal cortex): Oversight and control
    """
    
    def __init__(self):
        self.agents: Dict[str, AgentDefinition] = {}
        self.agent_instances: Dict[str, Any] = {}
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.active_agents: Set[str] = set()
        self.processing_queue: deque = deque()
        
        # Performance tracking
        self.orchestrator_metrics = {
            "total_activations": 0,
            "concurrent_agents": 0,
            "queue_size": 0,
            "average_activation_time": 0.0,
            "error_rate": 0.0
        }
        
        # Context analysis
        self.context_analyzer = ContextAnalyzer()
        self.dependency_resolver = DependencyResolver()
        
        # Initialize the brain structure
        self._initialize_brain_structure()
        
        logger.info("🧠 NIS Agent Orchestrator initialized")
    
    def _initialize_brain_structure(self):
        """Initialize the human brain-like agent structure"""
        
        # 🧠 CORE AGENTS (Always Active - Optimized with clear namespacing)
        self.register_agent(AgentDefinition(
            agent_id="laplace_signal_processor",
            name="Laplace Signal Processing Agent",
            agent_type=AgentType.CORE,
            activation_trigger=ActivationTrigger.ALWAYS,
            context_keywords=["input", "signal", "data", "preprocessing", "transform", "frequency"],
            priority=10,
            description="Processes incoming signals using optimized Laplace transform operations with token-efficient responses"
        ))
        
        self.register_agent(AgentDefinition(
            agent_id="kan_reasoning_engine",
            name="KAN Reasoning Engine",
            agent_type=AgentType.CORE,
            activation_trigger=ActivationTrigger.ALWAYS,
            dependencies=["laplace_signal_processor"],
            context_keywords=["analyze", "reason", "think", "symbolic", "interpret", "extract"],
            priority=10,
            description="Performs symbolic reasoning using KAN networks with interpretable function extraction"
        ))
        
        self.register_agent(AgentDefinition(
            agent_id="physics_validator",
            name="Physics Validation Agent (PINN)",
            agent_type=AgentType.CORE,
            activation_trigger=ActivationTrigger.ALWAYS,
            dependencies=["kan_reasoning_engine"],
            context_keywords=["physics", "validate", "constraints", "laws", "conservation", "correction"],
            priority=9,
            description="Validates outputs against physics laws with auto-correction capabilities using PINN networks"
        ))
        
        self.register_agent(AgentDefinition(
            agent_id="consciousness",
            name="Consciousness Agent",
            agent_type=AgentType.CORE,
            activation_trigger=ActivationTrigger.ALWAYS,
            context_keywords=["awareness", "meta", "self", "consciousness"],
            priority=8,
            description="Self-awareness and meta-cognitive processing"
        ))
        
        self.register_agent(AgentDefinition(
            agent_id="memory",
            name="Memory Agent",
            agent_type=AgentType.CORE,
            activation_trigger=ActivationTrigger.ALWAYS,
            context_keywords=["remember", "store", "recall", "memory"],
            priority=8,
            description="Memory storage and retrieval"
        ))
        
        self.register_agent(AgentDefinition(
            agent_id="coordination",
            name="Meta Coordination Agent",
            agent_type=AgentType.CORE,
            activation_trigger=ActivationTrigger.ALWAYS,
            context_keywords=["coordinate", "orchestrate", "manage"],
            priority=9,
            description="Meta-level coordination and oversight"
        ))
        
        # 🎯 SPECIALIZED AGENTS (Context Activated - Consolidated per research principles)
        self.register_agent(AgentDefinition(
            agent_id="multimodal_analysis_engine",
            name="Consolidated Multimodal Analysis Engine",
            agent_type=AgentType.SPECIALIZED,
            activation_trigger=ActivationTrigger.CONTEXT,
            context_keywords=["image", "visual", "photo", "picture", "video", "analyze", "see", "document", "pdf", "text", "file"],
            max_concurrent=2,
            priority=7,
            description="Consolidated vision and document analysis with optimized response formats and token efficiency"
        ))
        
        self.register_agent(AgentDefinition(
            agent_id="research_and_search_engine",
            name="Consolidated Research and Search Engine", 
            agent_type=AgentType.SPECIALIZED,
            activation_trigger=ActivationTrigger.CONTEXT,
            context_keywords=["search", "web", "internet", "research", "find", "lookup", "study", "investigate", "analyze"],
            max_concurrent=2,
            priority=6,
            description="Consolidated web search and deep research with semantic analysis and document processing capabilities"
        ))
        
        self.register_agent(AgentDefinition(
            agent_id="nvidia_simulation",
            name="NVIDIA Physics Simulation",
            agent_type=AgentType.SPECIALIZED,
            activation_trigger=ActivationTrigger.CONTEXT,
            context_keywords=["simulation", "physics", "modeling", "nvidia", "compute"],
            dependencies=["physics_validation"],
            max_concurrent=1,
            priority=7,
            description="Advanced physics simulation using NVIDIA NeMo"
        ))
        
        # 🌐 PROTOCOL AGENTS (External Communication - Nervous System)
        self.register_agent(AgentDefinition(
            agent_id="a2a_protocol",
            name="Agent-to-Agent Protocol",
            agent_type=AgentType.PROTOCOL,
            activation_trigger=ActivationTrigger.EVENT_DRIVEN,
            context_keywords=["agent", "communicate", "protocol", "a2a"],
            priority=5,
            description="Agent-to-agent communication protocol"
        ))
        
        self.register_agent(AgentDefinition(
            agent_id="mcp_protocol",
            name="Model Context Protocol",
            agent_type=AgentType.PROTOCOL,
            activation_trigger=ActivationTrigger.EVENT_DRIVEN,
            context_keywords=["model", "context", "mcp", "protocol"],
            priority=5,
            description="Model context protocol integration"
        ))
        
        # 🧠 LEARNING AGENTS (Adaptive Functions - Hippocampus)
        self.register_agent(AgentDefinition(
            agent_id="learning",
            name="Learning Agent",
            agent_type=AgentType.LEARNING,
            activation_trigger=ActivationTrigger.SCHEDULED,
            context_keywords=["learn", "adapt", "improve", "train"],
            priority=4,
            description="Continuous learning and adaptation"
        ))
        
        self.register_agent(AgentDefinition(
            agent_id="bitnet_training",
            name="BitNet Training Agent",
            agent_type=AgentType.LEARNING,
            activation_trigger=ActivationTrigger.ON_DEMAND,
            context_keywords=["bitnet", "training", "optimize", "neural"],
            max_concurrent=1,
            priority=3,
            description="BitNet neural network training"
        ))
        
        logger.info(f"🧠 Initialized {len(self.agents)} agents in brain structure")
    
    def register_agent(self, agent_def: AgentDefinition) -> bool:
        """Register a new agent in the orchestrator"""
        try:
            self.agents[agent_def.agent_id] = agent_def
            self.agent_metrics[agent_def.agent_id] = AgentMetrics()
            
            logger.info(f"📝 Registered agent: {agent_def.agent_id} ({agent_def.agent_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_def.agent_id}: {e}")
            return False
    
    async def start_orchestrator(self):
        """Start the agent orchestrator system"""
        logger.info("🚀 Starting NIS Agent Orchestrator...")
        
        # Start always-active core agents
        await self._activate_core_agents()
        
        # Start background monitoring
        asyncio.create_task(self._monitor_agents())
        asyncio.create_task(self._process_queue())
        asyncio.create_task(self._update_system_state())
        
        logger.info("✅ Agent Orchestrator started successfully")
    
    async def _activate_core_agents(self):
        """Activate all core agents that should always be running"""
        core_agents = [
            agent_id for agent_id, agent in self.agents.items()
            if agent.activation_trigger == ActivationTrigger.ALWAYS
        ]
        
        for agent_id in core_agents:
            await self.activate_agent(agent_id, context="system_startup")
        
        logger.info(f"🧠 Activated {len(core_agents)} core agents")
    
    async def activate_agent(self, agent_id: str, context: str = "", force: bool = False) -> bool:
        """Activate a specific agent"""
        if agent_id not in self.agents:
            logger.warning(f"Unknown agent: {agent_id}")
            return False
        
        agent = self.agents[agent_id]
        
        # Check if already active and at capacity
        active_count = sum(1 for aid in self.active_agents if aid.startswith(agent_id))
        if active_count >= agent.max_concurrent and not force:
            logger.debug(f"Agent {agent_id} at max capacity ({active_count}/{agent.max_concurrent})")
            return False
        
        try:
            start_time = time.time()
            
            # Check dependencies
            if not await self._check_dependencies(agent_id):
                logger.warning(f"Dependencies not met for agent {agent_id}")
                return False
            
            # Update status
            agent.status = AgentStatus.INITIALIZING
            
            # Create unique instance ID for concurrent agents
            instance_id = f"{agent_id}_{len(self.active_agents)}_{int(time.time())}"
            
            # Simulate agent activation (replace with actual agent instantiation)
            await self._simulate_agent_activation(agent_id, instance_id, context)
            
            # Mark as active
            self.active_agents.add(instance_id)
            agent.status = AgentStatus.ACTIVE
            
            # Update metrics
            metrics = self.agent_metrics[agent_id]
            metrics.activation_count += 1
            metrics.last_active = time.time()
            
            activation_time = time.time() - start_time
            self.orchestrator_metrics["total_activations"] += 1
            self.orchestrator_metrics["concurrent_agents"] = len(self.active_agents)
            
            # Emit state event
            await emit_state_event(
                StateEventType.AGENT_STATUS_CHANGE,
                {
                    "agent_id": agent_id,
                    "instance_id": instance_id,
                    "status": "activated",
                    "context": context,
                    "activation_time": activation_time,
                    "agent_type": agent.agent_type.value
                }
            )
            
            logger.info(f"✅ Activated agent: {agent_id} (instance: {instance_id})")
            return True
            
        except Exception as e:
            agent.status = AgentStatus.ERROR
            logger.error(f"Failed to activate agent {agent_id}: {e}")
            return False
    
    async def deactivate_agent(self, instance_id: str) -> bool:
        """Deactivate a specific agent instance"""
        if instance_id not in self.active_agents:
            return False
        
        try:
            # Extract agent_id from instance_id
            agent_id = instance_id.split('_')[0]
            
            # Remove from active set
            self.active_agents.remove(instance_id)
            
            # Update agent status
            if agent_id in self.agents:
                active_count = sum(1 for aid in self.active_agents if aid.startswith(agent_id))
                if active_count == 0:
                    self.agents[agent_id].status = AgentStatus.IDLE
            
            # Update metrics
            self.orchestrator_metrics["concurrent_agents"] = len(self.active_agents)
            
            # Emit state event
            await emit_state_event(
                StateEventType.AGENT_STATUS_CHANGE,
                {
                    "agent_id": agent_id,
                    "instance_id": instance_id,
                    "status": "deactivated"
                }
            )
            
            logger.info(f"🔴 Deactivated agent instance: {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deactivate agent instance {instance_id}: {e}")
            return False
    
    async def process_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request through the agent orchestrator"""
        request_id = f"req_{int(time.time())}_{len(self.processing_queue)}"
        
        try:
            # Analyze context to determine which agents to activate
            context_analysis = await self.context_analyzer.analyze(input_data)
            
            # Determine required agents
            required_agents = await self._determine_required_agents(context_analysis)
            
            # Activate agents if needed
            activated_agents = []
            for agent_id in required_agents:
                if await self.activate_agent(agent_id, context=context_analysis.get("primary_context", "")):
                    activated_agents.append(agent_id)
            
            # Process through the pipeline
            result = await self._process_through_pipeline(input_data, activated_agents, request_id)
            
            return {
                "request_id": request_id,
                "result": result,
                "activated_agents": activated_agents,
                "context_analysis": context_analysis,
                "processing_time": time.time() - float(request_id.split('_')[1])
            }
            
        except Exception as e:
            logger.error(f"Failed to process request {request_id}: {e}")
            return {"request_id": request_id, "error": str(e)}
    
    async def _determine_required_agents(self, context_analysis: Dict[str, Any]) -> List[str]:
        """Determine which agents are needed based on context"""
        required_agents = []
        
        # Always include core agents
        core_agents = [
            agent_id for agent_id, agent in self.agents.items()
            if agent.activation_trigger == ActivationTrigger.ALWAYS
        ]
        required_agents.extend(core_agents)
        
        # Analyze context for specialized agents
        context_text = context_analysis.get("text", "").lower()
        keywords = context_analysis.get("keywords", [])
        
        for agent_id, agent in self.agents.items():
            if agent.activation_trigger == ActivationTrigger.CONTEXT:
                # Check if any agent keywords match the context
                if any(keyword in context_text or keyword in keywords 
                       for keyword in agent.context_keywords):
                    required_agents.append(agent_id)
        
        return list(set(required_agents))  # Remove duplicates
    
    async def _simulate_agent_activation(self, agent_id: str, instance_id: str, context: str):
        """Simulate agent activation (replace with actual agent instantiation)"""
        # This is where you would actually instantiate and configure the real agent
        await asyncio.sleep(0.2)  # Simulate initialization time (slightly longer)

        # Store agent instance with proper status tracking
        self.agent_instances[instance_id] = {
            "agent_id": agent_id,
            "instance_id": instance_id,
            "context": context,
            "created_at": time.time(),
            "status": "active",
            "last_health_check": time.time(),
            "health_check_count": 0
        }

        # Update agent metrics to show it's healthy
        if agent_id in self.agent_metrics:
            metrics = self.agent_metrics[agent_id]
            metrics.last_active = time.time()
            metrics.health_check_count += 1
    
    async def _check_dependencies(self, agent_id: str) -> bool:
        """Check if agent dependencies are satisfied"""
        agent = self.agents[agent_id]
        
        for dep_id in agent.dependencies:
            # Check if dependency is active
            dep_active = any(aid.startswith(dep_id) for aid in self.active_agents)
            if not dep_active:
                logger.debug(f"Dependency {dep_id} not active for agent {agent_id}")
                return False
        
        return True
    
    async def _process_through_pipeline(self, input_data: Dict[str, Any], agents: List[str], request_id: str) -> Dict[str, Any]:
        """Process data through the agent pipeline"""
        # This is a simplified pipeline - implement actual agent processing
        result = {"input": input_data, "processed_by": agents, "request_id": request_id}
        
        # Simulate processing time
        await asyncio.sleep(0.2)
        
        return result
    
    async def _monitor_agents(self):
        """Background task to monitor agent health and performance"""
        while True:
            try:
                # Check agent health
                for instance_id in list(self.active_agents):
                    if await self._check_agent_health(instance_id):
                        continue
                    else:
                        await self.deactivate_agent(instance_id)
                
                # Update performance metrics
                await self._calculate_performance_metrics()
                
                await asyncio.sleep(15)  # Monitor every 15 seconds (reduced frequency)
                
            except Exception as e:
                logger.error(f"Agent monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _check_agent_health(self, instance_id: str) -> bool:
        """Check if an agent instance is healthy"""
        if instance_id not in self.agent_instances:
            return False

        instance = self.agent_instances[instance_id]

        # Update health check timestamp
        instance["last_health_check"] = time.time()
        instance["health_check_count"] += 1

        # Check if agent has been running too long (basic timeout)
        agent_id = instance["agent_id"]
        if agent_id in self.agents:
            timeout = self.agents[agent_id].timeout_seconds
            if time.time() - instance["created_at"] > timeout:
                logger.warning(f"Agent {instance_id} timed out after {timeout} seconds")
                return False

        # Check if agent is currently processing (extend timeout if active)
        if instance.get("is_processing", False):
            # Extend timeout for active agents
            extended_timeout = timeout * 2
            if time.time() - instance["created_at"] > extended_timeout:
                logger.warning(f"Active agent {instance_id} exceeded extended timeout")
                return False

        # Agent is healthy
        return True
    
    async def _calculate_performance_metrics(self):
        """Calculate and update performance metrics"""
        try:
            total_agents = len(self.agents)
            active_agents = len(self.active_agents)
            
            # Update orchestrator metrics
            self.orchestrator_metrics.update({
                "concurrent_agents": active_agents,
                "queue_size": len(self.processing_queue),
                "agent_utilization": active_agents / max(total_agents, 1)
            })
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
    
    async def _process_queue(self):
        """Background task to process the request queue"""
        while True:
            try:
                if self.processing_queue:
                    # Process next request in queue
                    request = self.processing_queue.popleft()
                    await self.process_request(request)
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(1)
    
    async def _update_system_state(self):
        """Background task to update system state"""
        while True:
            try:
                # Prepare agent status for state update
                agent_status = {}
                for agent_id, agent in self.agents.items():
                    active_instances = [aid for aid in self.active_agents if aid.startswith(agent_id)]
                    
                    agent_status[agent_id] = {
                        "status": agent.status.value,
                        "type": agent.agent_type.value,
                        "active_instances": len(active_instances),
                        "max_concurrent": agent.max_concurrent,
                        "priority": agent.priority,
                        "metrics": asdict(self.agent_metrics[agent_id])
                    }
                
                # Update global state
                from .state_manager import update_system_state
                await update_system_state({
                    "active_agents": agent_status,
                    "orchestrator_metrics": self.orchestrator_metrics,
                    "total_agents": len(self.agents),
                    "active_instances": len(self.active_agents)
                })
                
                await asyncio.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"State update error: {e}")
                await asyncio.sleep(2)
    
    def get_agent_status(self, agent_id: str = None) -> Dict[str, Any]:
        """Get status of specific agent or all agents"""
        if agent_id:
            if agent_id not in self.agents:
                return {}
            
            agent = self.agents[agent_id]
            metrics = self.agent_metrics[agent_id]
            active_instances = [aid for aid in self.active_agents if aid.startswith(agent_id)]
            
            return {
                "agent_id": agent_id,
                "status": agent.status.value if hasattr(agent.status, 'value') else str(agent.status),
                "type": agent.agent_type.value if hasattr(agent.agent_type, 'value') else str(agent.agent_type),
                "active_instances": len(active_instances),
                "metrics": asdict(metrics),
                "definition": asdict(agent)
            }
        else:
            # Return all agents - fix recursive call
            agents_status = {}
            for agent_id in self.agents.keys():
                if agent_id in self.agents:
                    agents_status[agent_id] = self.get_agent_status(agent_id)
            return agents_status
    
    async def execute_agent_action(
        self,
        action: ActionDefinition,
        context_pack: Dict[str, Any]
    ) -> ActionResult:
        """
        Agent execution loop with validation and rollback
        
        Flow: propose → validate → apply → verify → (rollback if needed)
        
        This is the core reliability pattern - every action is:
        1. Validated before execution
        2. Applied with monitoring
        3. Verified after execution
        4. Rolled back if verification fails
        """
        action_id = f"action_{int(time.time())}_{action.agent_id}"
        start_time = time.time()
        audit_trail = []
        
        try:
            # STEP 1: PROPOSE (log the action plan)
            audit_trail.append({
                "step": "propose",
                "timestamp": time.time(),
                "action": action.action_type.value,
                "agent_id": action.agent_id,
                "parameters": action.parameters
            })
            
            logger.info(f"🎯 Executing action: {action.action_type.value} for agent: {action.agent_id}")
            
            # STEP 2: VALIDATE (check constraints BEFORE execution)
            validation = await self._validate_action(action, context_pack)
            if not validation["valid"]:
                logger.warning(f"❌ Validation failed: {validation['reason']}")
                return ActionResult(
                    action_id=action_id,
                    success=False,
                    output=None,
                    error=f"Validation failed: {validation['reason']}",
                    execution_time=time.time() - start_time,
                    audit_trail=audit_trail
                )
            
            audit_trail.append({
                "step": "validate",
                "timestamp": time.time(),
                "result": "passed"
            })
            
            # STEP 3: APPLY (execute the action)
            result = await self._apply_action(action, context_pack)
            
            audit_trail.append({
                "step": "apply",
                "timestamp": time.time(),
                "result": "executed"
            })
            
            # STEP 4: VERIFY (check post-conditions)
            verification = await self._verify_result(action, result, context_pack)
            
            if not verification["valid"]:
                logger.warning(f"⚠️ Verification failed: {verification['reason']}")
                
                # STEP 5: ROLLBACK (auto-repair if possible)
                if action.rollback_enabled:
                    await self._rollback_action(action, result)
                    audit_trail.append({
                        "step": "rollback",
                        "timestamp": time.time(),
                        "reason": verification["reason"]
                    })
                
                return ActionResult(
                    action_id=action_id,
                    success=False,
                    output=result,
                    error=f"Verification failed: {verification['reason']}",
                    execution_time=time.time() - start_time,
                    audit_trail=audit_trail
                )
            
            audit_trail.append({
                "step": "verify",
                "timestamp": time.time(),
                "result": "passed"
            })
            
            # SUCCESS
            logger.info(f"✅ Action completed: {action_id}")
            return ActionResult(
                action_id=action_id,
                success=True,
                output=result,
                execution_time=time.time() - start_time,
                audit_trail=audit_trail
            )
            
        except Exception as e:
            logger.error(f"❌ Action execution failed: {e}")
            
            # Auto-rollback on exception
            if action.rollback_enabled:
                try:
                    await self._rollback_action(action, None)
                    audit_trail.append({
                        "step": "rollback",
                        "timestamp": time.time(),
                        "reason": f"Exception: {str(e)}"
                    })
                except Exception as rollback_error:
                    logger.error(f"Rollback failed: {rollback_error}")
            
            return ActionResult(
                action_id=action_id,
                success=False,
                output=None,
                error=str(e),
                execution_time=time.time() - start_time,
                audit_trail=audit_trail
            )
    
    async def _validate_action(
        self,
        action: ActionDefinition,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate action BEFORE execution"""
        # Check if agent exists
        if action.agent_id not in self.agents:
            return {"valid": False, "reason": f"Agent {action.agent_id} not found"}
        
        # Check if action is allowed for this agent
        allowed_tools = context.get("tools", [])
        if action.action_type.value.upper() not in allowed_tools:
            return {"valid": False, "reason": f"Action {action.action_type.value} not allowed for agent {action.agent_id}"}
        
        # Check token budget
        if context.get("token_budget", 0) <= 0:
            return {"valid": False, "reason": "Token budget exceeded"}
        
        # Check timeout
        if action.timeout_seconds <= 0:
            return {"valid": False, "reason": "Invalid timeout"}
        
        return {"valid": True}
    
    async def _apply_action(
        self,
        action: ActionDefinition,
        context: Dict[str, Any]
    ) -> Any:
        """Execute the actual action"""
        # Route to appropriate handler based on action type
        action_handlers = {
            AgentAction.QUERY_STATE: self._handle_query_state,
            AgentAction.QUERY_MEMORY: self._handle_query_memory,
            AgentAction.STORE_MEMORY: self._handle_store_memory,
            AgentAction.CALL_LLM: self._handle_call_llm,
            AgentAction.RUN_TOOL: self._handle_run_tool,
            AgentAction.CREATE_PLAN: self._handle_create_plan,
        }
        
        handler = action_handlers.get(action.action_type)
        if handler:
            return await handler(action, context)
        else:
            # Default handler for unimplemented actions
            return {"status": "not_implemented", "action": action.action_type.value}
    
    async def _verify_result(
        self,
        action: ActionDefinition,
        result: Any,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify result meets post-conditions"""
        # Basic verification - check result is not None
        if result is None:
            return {"valid": False, "reason": "Result is None"}
        
        # Check if result has expected structure
        if isinstance(result, dict) and "error" in result:
            return {"valid": False, "reason": f"Action returned error: {result.get('error')}"}
        
        return {"valid": True}
    
    async def _rollback_action(
        self,
        action: ActionDefinition,
        result: Any
    ) -> None:
        """Rollback action if it failed"""
        logger.info(f"🔄 Rolling back action: {action.action_type.value}")
        # TODO: Implement actual rollback logic per action type
        # For now, just log the rollback
        pass
    
    # Action handlers (stubs for now - will be implemented as needed)
    async def _handle_query_state(self, action: ActionDefinition, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle QUERY_STATE action"""
        return {"state": context.get("state", {})}
    
    async def _handle_query_memory(self, action: ActionDefinition, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle QUERY_MEMORY action"""
        return {"memory": context.get("memory", [])}
    
    async def _handle_store_memory(self, action: ActionDefinition, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle STORE_MEMORY action"""
        return {"status": "stored", "memory_id": f"mem_{int(time.time())}"}
    
    async def _handle_call_llm(self, action: ActionDefinition, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle CALL_LLM action"""
        return {"status": "llm_called", "response": "LLM response placeholder"}
    
    async def _handle_run_tool(self, action: ActionDefinition, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle RUN_TOOL action"""
        return {"status": "tool_executed", "tool": action.parameters.get("tool_name", "unknown")}
    
    async def _handle_create_plan(self, action: ActionDefinition, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle CREATE_PLAN action"""
        return {"status": "plan_created", "plan_id": f"plan_{int(time.time())}"}

class ContextAnalyzer:
    """Analyzes input context to determine required agents (Cursor pattern enhanced)"""
    
    async def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze input to determine context and requirements"""
        try:
            text = str(input_data.get("text", "")).lower()
            
            # Extract keywords
            keywords = []
            if "image" in text or "photo" in text or "picture" in text:
                keywords.extend(["visual", "image", "analysis"])
            if "document" in text or "pdf" in text or "file" in text:
                keywords.extend(["document", "text", "analysis"])
            if "search" in text or "research" in text or "find" in text:
                keywords.extend(["web", "search", "research"])
            
            # Determine primary context
            primary_context = "general"
            if "visual" in keywords:
                primary_context = "visual_analysis"
            elif "document" in keywords:
                primary_context = "document_analysis"
            elif "search" in keywords:
                primary_context = "web_research"
            
            return {
                "text": text,
                "keywords": keywords,
                "primary_context": primary_context,
                "complexity": len(text.split()),
                "requires_vision": "visual" in keywords,
                "requires_search": "search" in keywords,
                "requires_documents": "document" in keywords
            }
            
        except Exception as e:
            logger.error(f"Context analysis error: {e}")
            return {"text": "", "keywords": [], "primary_context": "general"}
    
    def build_context_pack(
        self,
        agent_id: str,
        request_data: Dict[str, Any],
        agent_state: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4000
    ) -> Dict[str, Any]:
        """
        Build just-in-time context pack (Cursor's secret sauce)
        
        Returns ONLY what this agent needs:
        - Relevant state (not entire system state)
        - Relevant memory (semantic search, limited)
        - Allowed tools (policy-based)
        - Active policies
        - NO noise
        """
        context_pack = {
            "agent_id": agent_id,
            "timestamp": time.time(),
            "request": request_data,
            "state": self._get_relevant_state(agent_id, agent_state or {}),
            "memory": self._get_relevant_memory(agent_id, request_data),
            "tools": self._get_allowed_tools(agent_id),
            "policies": self._get_active_policies(agent_id),
            "token_budget": max_tokens
        }
        
        return context_pack
    
    def _get_relevant_state(self, agent_id: str, full_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get only state relevant to this agent (scoped, not entire state)"""
        # Filter state by agent scope - only return what this agent needs
        relevant_keys = {
            "laplace_signal_processor": ["input_signals", "preprocessing_config"],
            "kan_reasoning_engine": ["reasoning_context", "symbolic_state"],
            "physics_validator": ["physics_constraints", "validation_rules"],
            "multimodal_analysis_engine": ["vision_config", "document_state"],
            "research_and_search_engine": ["search_history", "research_context"],
        }
        
        agent_keys = relevant_keys.get(agent_id, [])
        return {k: full_state.get(k) for k in agent_keys if k in full_state}
    
    def _get_relevant_memory(self, agent_id: str, request: Dict[str, Any]) -> List[Dict]:
        """Get only memories relevant to this request (semantic search with limit)"""
        # TODO: Integrate with actual memory system
        # For now, return empty list - will be implemented when memory system is integrated
        return []
    
    def _get_allowed_tools(self, agent_id: str) -> List[str]:
        """Get tools this agent is allowed to use (policy-based)"""
        # Define tool access per agent type
        tool_access = {
            "laplace_signal_processor": ["READ_SENSOR", "QUERY_STATE"],
            "kan_reasoning_engine": ["CALL_LLM", "QUERY_MEMORY", "RUN_TOOL"],
            "physics_validator": ["QUERY_STATE", "RUN_TOOL"],
            "multimodal_analysis_engine": ["CALL_LLM", "RUN_TOOL", "QUERY_MEMORY"],
            "research_and_search_engine": ["CALL_LLM", "RUN_TOOL", "QUERY_MEMORY"],
            "consciousness": ["CALL_LLM", "QUERY_MEMORY", "STORE_MEMORY", "CREATE_PLAN"],
            "memory": ["QUERY_MEMORY", "STORE_MEMORY"],
            "coordination": ["CREATE_PLAN", "EXECUTE_PLAN", "QUERY_STATE"],
        }
        
        return tool_access.get(agent_id, ["QUERY_STATE"])
    
    def _get_active_policies(self, agent_id: str) -> List[Dict]:
        """Get policies that apply to this agent"""
        # TODO: Integrate with actual policy engine
        # For now, return basic policies
        return [
            {"rule": "All actions must be auditable", "level": "critical"},
            {"rule": "Respect token budgets", "level": "high"},
            {"rule": "No unauthorized data access", "level": "critical"}
        ]

class DependencyResolver:
    """Resolves agent dependencies and activation order"""
    
    def resolve_activation_order(self, agents: Dict[str, AgentDefinition], target_agents: List[str]) -> List[str]:
        """Determine the order to activate agents based on dependencies"""
        resolved = []
        remaining = target_agents.copy()
        
        while remaining:
            # Find agents with no unresolved dependencies
            ready = []
            for agent_id in remaining:
                if agent_id not in agents:
                    continue
                
                deps = agents[agent_id].dependencies
                if all(dep in resolved for dep in deps):
                    ready.append(agent_id)
            
            if not ready:
                # Circular dependency or missing dependency
                logger.warning(f"Cannot resolve dependencies for: {remaining}")
                resolved.extend(remaining)
                break
            
            # Add ready agents to resolved list
            for agent_id in ready:
                resolved.append(agent_id)
                remaining.remove(agent_id)
        
        return resolved

# Global orchestrator instance
nis_agent_orchestrator = NISAgentOrchestrator()

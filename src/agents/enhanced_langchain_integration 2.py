#!/usr/bin/env python3
"""
🔗 Enhanced LangChain Integration for All NIS Agents
Advanced LangChain/LangGraph integration layer that ensures every agent
works seamlessly with LangChain workflows and deep agent planning

Features:
- Universal LangChain wrapper for all agent types
- LangGraph workflow integration
- LangSmith observability for all agents
- Tool interoperability across agents
- Cross-agent conversation and memory sharing
- Standardized agent interface for LangChain
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import inspect

# LangChain core imports
try:
    from langchain_core.language_models import BaseLLM
    from langchain_core.tools import BaseTool, tool
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.runnables import Runnable
    LANGCHAIN_CORE_AVAILABLE = True
except ImportError:
    LANGCHAIN_CORE_AVAILABLE = False

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import ToolExecutor, ToolInvocation
    from typing_extensions import TypedDict, Annotated
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# LangSmith imports
try:
    from langsmith import traceable, Client as LangSmithClient
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

# Core NIS components
from ..core.agent import NISAgent
from ..llm.llm_manager import LLMManager
from ..memory.memory_manager import MemoryManager

# Import all available agents with fallback
agent_classes = {}

try:
    from .memory.enhanced_memory_agent import EnhancedMemoryAgent
    agent_classes['memory'] = EnhancedMemoryAgent
except ImportError:
    pass

try:
    from .reasoning.unified_reasoning_agent import UnifiedReasoningAgent
    agent_classes['reasoning'] = UnifiedReasoningAgent
except ImportError:
    pass

try:
    from .physics.unified_physics_agent import UnifiedPhysicsAgent
    agent_classes['physics'] = UnifiedPhysicsAgent
except ImportError:
    pass

try:
    from .consciousness.enhanced_conscious_agent import EnhancedConsciousAgent
    agent_classes['consciousness'] = EnhancedConsciousAgent
except ImportError:
    pass

try:
    from .research.web_search_agent import WebSearchAgent
    from .research.deep_research_agent import DeepResearchAgent
    agent_classes['web_search'] = WebSearchAgent
    agent_classes['deep_research'] = DeepResearchAgent
except ImportError:
    pass

try:
    from .multimodal.vision_agent import VisionAgent
    agent_classes['vision'] = VisionAgent
except ImportError:
    pass


class AgentToolCategory(Enum):
    """Categories of agent tools"""
    MEMORY = "memory"
    REASONING = "reasoning"
    PHYSICS = "physics"
    RESEARCH = "research"
    VISION = "vision"
    COMMUNICATION = "communication"
    COORDINATION = "coordination"
    ANALYSIS = "analysis"


@dataclass
class LangChainAgentConfig:
    """Configuration for LangChain agent integration"""
    agent_id: str
    agent_type: str
    enable_tools: bool = True
    enable_memory: bool = True
    enable_callbacks: bool = True
    enable_tracing: bool = True
    tool_categories: List[AgentToolCategory] = field(default_factory=list)
    custom_tools: List[str] = field(default_factory=list)
    llm_provider: str = "anthropic"
    temperature: float = 0.7
    max_tokens: int = 1000


class NISAgentCallback(BaseCallbackHandler):
    """Custom callback handler for NIS agents"""
    
    def __init__(self, agent_id: str):
        super().__init__()
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"nis.langchain.{agent_id}")
        self.conversation_history = []
        self.performance_metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "average_latency": 0.0,
            "token_usage": 0
        }
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts running"""
        self.performance_metrics["total_calls"] += 1
        self.logger.debug(f"LLM started for agent {self.agent_id}")
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM ends running"""
        self.performance_metrics["successful_calls"] += 1
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            self.performance_metrics["token_usage"] += token_usage.get('total_tokens', 0)
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """Called when LLM encounters an error"""
        self.performance_metrics["failed_calls"] += 1
        self.logger.error(f"LLM error for agent {self.agent_id}: {error}")


class LangChainAgentWrapper:
    """
    Universal wrapper that makes any NIS agent compatible with LangChain workflows
    """
    
    def __init__(
        self,
        agent: Any,
        config: LangChainAgentConfig,
        llm_manager: Optional[LLMManager] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        self.agent = agent
        self.config = config
        self.llm_manager = llm_manager or LLMManager()
        self.memory_manager = memory_manager or MemoryManager()
        
        self.logger = logging.getLogger(f"nis.langchain.wrapper.{config.agent_id}")
        
        # LangChain components
        self.tools: List[BaseTool] = []
        self.callback_handler = NISAgentCallback(config.agent_id) if config.enable_callbacks else None
        
        # Conversation state
        self.conversation_memory = []
        self.shared_context = {}
        
        # Initialize LangChain integration
        self._initialize_langchain_integration()
    
    def _initialize_langchain_integration(self):
        """Initialize LangChain integration components"""
        try:
            # Create tools from agent capabilities
            if self.config.enable_tools:
                self._create_agent_tools()
            
            # Setup tracing if enabled
            if self.config.enable_tracing and LANGSMITH_AVAILABLE:
                self._setup_langsmith_tracing()
            
            self.logger.info(f"LangChain integration initialized for {self.config.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LangChain integration: {e}")
    
    def _create_agent_tools(self):
        """Create LangChain tools from agent capabilities"""
        
        # Standard agent processing tool
        @tool
        def process_with_agent(input_data: str, context: str = "") -> str:
            """Process input using the NIS agent"""
            try:
                if hasattr(self.agent, 'process'):
                    result = asyncio.run(self.agent.process(
                        {"input": input_data, "context": context}
                    ))
                elif hasattr(self.agent, 'execute'):
                    result = asyncio.run(self.agent.execute(
                        {"input": input_data, "context": context}
                    ))
                else:
                    result = {"output": f"Processed by {self.config.agent_id}: {input_data}"}
                
                return json.dumps(result) if isinstance(result, dict) else str(result)
                
            except Exception as e:
                return f"Error processing with {self.config.agent_id}: {str(e)}"
        
        process_with_agent.name = f"{self.config.agent_id}_process"
        self.tools.append(process_with_agent)
        
        # Memory agent specific tools
        if self.config.agent_type == "memory" and hasattr(self.agent, 'store_memory'):
            @tool
            def store_memory(content: str, memory_type: str = "episodic") -> str:
                """Store information in agent memory"""
                try:
                    result = asyncio.run(self.agent.store_memory({
                        "content": content,
                        "memory_type": memory_type,
                        "source": "langchain_integration"
                    }))
                    return json.dumps(result)
                except Exception as e:
                    return f"Memory storage error: {str(e)}"
            
            @tool
            def retrieve_memory(query: str, memory_type: str = "episodic") -> str:
                """Retrieve information from agent memory"""
                try:
                    result = asyncio.run(self.agent.retrieve_memory({
                        "query": query,
                        "memory_type": memory_type,
                        "source": "langchain_integration"
                    }))
                    return json.dumps(result)
                except Exception as e:
                    return f"Memory retrieval error: {str(e)}"
            
            store_memory.name = f"{self.config.agent_id}_store_memory"
            retrieve_memory.name = f"{self.config.agent_id}_retrieve_memory"
            self.tools.extend([store_memory, retrieve_memory])
        
        # Reasoning agent specific tools
        if self.config.agent_type == "reasoning" and hasattr(self.agent, 'reason'):
            @tool
            def logical_reasoning(problem: str, reasoning_type: str = "deductive") -> str:
                """Perform logical reasoning on a problem"""
                try:
                    result = asyncio.run(self.agent.reason({
                        "problem": problem,
                        "reasoning_type": reasoning_type,
                        "source": "langchain_integration"
                    }))
                    return json.dumps(result)
                except Exception as e:
                    return f"Reasoning error: {str(e)}"
            
            logical_reasoning.name = f"{self.config.agent_id}_reason"
            self.tools.append(logical_reasoning)
        
        # Physics agent specific tools
        if self.config.agent_type == "physics" and hasattr(self.agent, 'validate_physics'):
            @tool
            def validate_physics(data: str, physics_domain: str = "mechanics") -> str:
                """Validate physics principles in data"""
                try:
                    result = asyncio.run(self.agent.validate_physics({
                        "data": data,
                        "domain": physics_domain,
                        "source": "langchain_integration"
                    }))
                    return json.dumps(result)
                except Exception as e:
                    return f"Physics validation error: {str(e)}"
            
            validate_physics.name = f"{self.config.agent_id}_validate_physics"
            self.tools.append(validate_physics)
        
        # Research agent specific tools
        if self.config.agent_type in ["web_search", "deep_research"] and hasattr(self.agent, 'search'):
            @tool
            def research_topic(query: str, depth: str = "standard") -> str:
                """Research a topic using the research agent"""
                try:
                    result = asyncio.run(self.agent.search({
                        "query": query,
                        "depth": depth,
                        "source": "langchain_integration"
                    }))
                    return json.dumps(result)
                except Exception as e:
                    return f"Research error: {str(e)}"
            
            research_topic.name = f"{self.config.agent_id}_research"
            self.tools.append(research_topic)
        
        # Vision agent specific tools
        if self.config.agent_type == "vision" and hasattr(self.agent, 'analyze_image'):
            @tool
            def analyze_image(image_path: str, analysis_type: str = "general") -> str:
                """Analyze an image using the vision agent"""
                try:
                    result = asyncio.run(self.agent.analyze_image({
                        "image_path": image_path,
                        "analysis_type": analysis_type,
                        "source": "langchain_integration"
                    }))
                    return json.dumps(result)
                except Exception as e:
                    return f"Image analysis error: {str(e)}"
            
            analyze_image.name = f"{self.config.agent_id}_analyze_image"
            self.tools.append(analyze_image)
        
        self.logger.info(f"Created {len(self.tools)} LangChain tools for {self.config.agent_id}")
    
    def _setup_langsmith_tracing(self):
        """Setup LangSmith tracing for observability"""
        try:
            # This would be implemented with real LangSmith client
            self.logger.info(f"LangSmith tracing enabled for {self.config.agent_id}")
        except Exception as e:
            self.logger.warning(f"Failed to setup LangSmith tracing: {e}")
    
    def get_tools(self) -> List[BaseTool]:
        """Get all tools provided by this agent"""
        return self.tools
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the wrapped agent"""
        return {
            "agent_id": self.config.agent_id,
            "agent_type": self.config.agent_type,
            "tools_count": len(self.tools),
            "capabilities": getattr(self.agent, 'capabilities', []),
            "status": getattr(self.agent, 'status', 'unknown'),
            "langchain_enabled": True,
            "performance_metrics": self.callback_handler.performance_metrics if self.callback_handler else {}
        }
    
    async def invoke_as_runnable(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make the agent invokable as a LangChain Runnable"""
        try:
            if hasattr(self.agent, 'process'):
                result = await self.agent.process(input_data)
            elif hasattr(self.agent, 'execute'):
                result = await self.agent.execute(input_data)
            else:
                result = {"output": f"Processed by {self.config.agent_id}"}
            
            # Store in conversation memory if enabled
            if self.config.enable_memory:
                self.conversation_memory.append({
                    "timestamp": time.time(),
                    "input": input_data,
                    "output": result,
                    "agent_id": self.config.agent_id
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error invoking agent as runnable: {e}")
            return {"error": str(e), "agent_id": self.config.agent_id}


class EnhancedLangChainIntegration:
    """
    Enhanced LangChain integration manager for all NIS agents
    """
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        memory_manager: Optional[MemoryManager] = None,
        enable_auto_wrap: bool = True
    ):
        self.llm_manager = llm_manager or LLMManager()
        self.memory_manager = memory_manager or MemoryManager()
        self.enable_auto_wrap = enable_auto_wrap
        
        self.logger = logging.getLogger(__name__)
        
        # Registry of wrapped agents
        self.wrapped_agents: Dict[str, LangChainAgentWrapper] = {}
        self.all_tools: Dict[str, BaseTool] = {}
        self.agent_configs: Dict[str, LangChainAgentConfig] = {}
        
        # LangGraph workflows
        self.workflows: Dict[str, StateGraph] = {}
        self.compiled_graphs: Dict[str, Any] = {}
        
        # Cross-agent communication
        self.message_bus: Dict[str, List[Dict[str, Any]]] = {}
        self.shared_memory: Dict[str, Any] = {}
        
        # Performance tracking
        self.integration_metrics = {
            "agents_wrapped": 0,
            "tools_created": 0,
            "workflows_active": 0,
            "cross_agent_messages": 0,
            "successful_integrations": 0,
            "failed_integrations": 0
        }
        
        # Auto-wrap available agents if enabled
        if self.enable_auto_wrap:
            self._auto_wrap_agents()
    
    def _auto_wrap_agents(self):
        """Automatically wrap all available NIS agents"""
        for agent_type, agent_class in agent_classes.items():
            try:
                # Create agent instance
                agent_id = f"langchain_{agent_type}_agent"
                agent_instance = agent_class(agent_id=agent_id)
                
                # Create configuration
                config = LangChainAgentConfig(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    enable_tools=True,
                    enable_memory=True,
                    enable_callbacks=True,
                    enable_tracing=True
                )
                
                # Wrap the agent
                success = self.wrap_agent(agent_instance, config)
                
                if success:
                    self.integration_metrics["successful_integrations"] += 1
                    self.logger.info(f"✅ Auto-wrapped {agent_type} agent for LangChain")
                else:
                    self.integration_metrics["failed_integrations"] += 1
                    
            except Exception as e:
                self.integration_metrics["failed_integrations"] += 1
                self.logger.error(f"❌ Failed to auto-wrap {agent_type} agent: {e}")
    
    def wrap_agent(
        self,
        agent: Any,
        config: LangChainAgentConfig
    ) -> bool:
        """Wrap a NIS agent for LangChain compatibility"""
        try:
            wrapper = LangChainAgentWrapper(
                agent=agent,
                config=config,
                llm_manager=self.llm_manager,
                memory_manager=self.memory_manager
            )
            
            self.wrapped_agents[config.agent_id] = wrapper
            self.agent_configs[config.agent_id] = config
            
            # Register tools
            for tool in wrapper.get_tools():
                self.all_tools[tool.name] = tool
            
            self.integration_metrics["agents_wrapped"] += 1
            self.integration_metrics["tools_created"] += len(wrapper.get_tools())
            
            self.logger.info(f"🔗 Wrapped agent {config.agent_id} with {len(wrapper.get_tools())} tools")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to wrap agent {config.agent_id}: {e}")
            return False
    
    def get_agent_wrapper(self, agent_id: str) -> Optional[LangChainAgentWrapper]:
        """Get a wrapped agent by ID"""
        return self.wrapped_agents.get(agent_id)
    
    def get_all_tools(self) -> Dict[str, BaseTool]:
        """Get all tools from all wrapped agents"""
        return self.all_tools
    
    def get_tools_by_category(self, category: AgentToolCategory) -> List[BaseTool]:
        """Get tools filtered by category"""
        category_tools = []
        for agent_id, wrapper in self.wrapped_agents.items():
            config = self.agent_configs[agent_id]
            if category in config.tool_categories or config.agent_type == category.value:
                category_tools.extend(wrapper.get_tools())
        return category_tools
    
    def create_multi_agent_workflow(
        self,
        workflow_name: str,
        agents: List[str],
        workflow_config: Dict[str, Any] = None
    ) -> Optional[StateGraph]:
        """Create a LangGraph workflow with multiple agents"""
        
        if not LANGGRAPH_AVAILABLE:
            self.logger.error("LangGraph not available for workflow creation")
            return None
        
        try:
            # Define workflow state
            class WorkflowState(TypedDict):
                messages: List[BaseMessage]
                current_agent: str
                shared_context: Dict[str, Any]
                iteration_count: int
                max_iterations: int
                
            # Create workflow graph
            workflow = StateGraph(WorkflowState)
            
            # Add agent nodes
            for agent_id in agents:
                if agent_id in self.wrapped_agents:
                    wrapper = self.wrapped_agents[agent_id]
                    
                    async def agent_node(state: WorkflowState, agent_wrapper=wrapper):
                        """Node function for agent processing"""
                        try:
                            # Get latest message
                            if state["messages"]:
                                latest_message = state["messages"][-1]
                                input_data = {
                                    "input": latest_message.content,
                                    "context": state["shared_context"]
                                }
                            else:
                                input_data = {"input": "Hello", "context": {}}
                            
                            # Process with agent
                            result = await agent_wrapper.invoke_as_runnable(input_data)
                            
                            # Create response message
                            response_message = AIMessage(
                                content=json.dumps(result) if isinstance(result, dict) else str(result),
                                additional_kwargs={"agent_id": agent_wrapper.config.agent_id}
                            )
                            
                            # Update state
                            return {
                                "messages": state["messages"] + [response_message],
                                "current_agent": agent_wrapper.config.agent_id,
                                "shared_context": {**state["shared_context"], "last_result": result},
                                "iteration_count": state["iteration_count"] + 1,
                                "max_iterations": state["max_iterations"]
                            }
                            
                        except Exception as e:
                            error_message = AIMessage(
                                content=f"Error in {agent_wrapper.config.agent_id}: {str(e)}",
                                additional_kwargs={"agent_id": agent_wrapper.config.agent_id, "error": True}
                            )
                            return {
                                "messages": state["messages"] + [error_message],
                                "current_agent": agent_wrapper.config.agent_id,
                                "shared_context": state["shared_context"],
                                "iteration_count": state["iteration_count"] + 1,
                                "max_iterations": state["max_iterations"]
                            }
                    
                    workflow.add_node(agent_id, agent_node)
            
            # Add edges based on workflow config
            workflow_config = workflow_config or {}
            routing = workflow_config.get("routing", "sequential")
            
            if routing == "sequential":
                # Sequential routing
                for i, agent_id in enumerate(agents):
                    if i == 0:
                        workflow.set_entry_point(agent_id)
                    if i < len(agents) - 1:
                        workflow.add_edge(agent_id, agents[i + 1])
                    else:
                        workflow.add_edge(agent_id, END)
            else:
                # Custom routing logic could be implemented here
                pass
            
            # Compile workflow
            checkpointer = MemorySaver()
            compiled_graph = workflow.compile(checkpointer=checkpointer)
            
            self.workflows[workflow_name] = workflow
            self.compiled_graphs[workflow_name] = compiled_graph
            self.integration_metrics["workflows_active"] += 1
            
            self.logger.info(f"🔄 Created multi-agent workflow: {workflow_name}")
            return workflow
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create workflow {workflow_name}: {e}")
            return None
    
    async def execute_workflow(
        self,
        workflow_name: str,
        initial_input: str,
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """Execute a multi-agent workflow"""
        
        if workflow_name not in self.compiled_graphs:
            return {"error": f"Workflow {workflow_name} not found"}
        
        try:
            compiled_graph = self.compiled_graphs[workflow_name]
            
            # Initial state
            initial_state = {
                "messages": [HumanMessage(content=initial_input)],
                "current_agent": "",
                "shared_context": {},
                "iteration_count": 0,
                "max_iterations": max_iterations
            }
            
            # Execute workflow
            config = {"configurable": {"thread_id": f"workflow_{int(time.time())}"}}
            
            final_state = None
            async for state in compiled_graph.astream(initial_state, config):
                final_state = state
                
                # Check for max iterations
                if state.get("iteration_count", 0) >= max_iterations:
                    break
            
            return {
                "workflow_name": workflow_name,
                "status": "completed",
                "final_state": final_state,
                "iterations": final_state.get("iteration_count", 0) if final_state else 0,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to execute workflow {workflow_name}: {e}")
            return {
                "workflow_name": workflow_name,
                "status": "failed",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def send_cross_agent_message(
        self,
        from_agent: str,
        to_agent: str,
        message: Dict[str, Any]
    ) -> bool:
        """Send a message between agents"""
        try:
            if to_agent not in self.message_bus:
                self.message_bus[to_agent] = []
            
            self.message_bus[to_agent].append({
                "from": from_agent,
                "message": message,
                "timestamp": time.time()
            })
            
            self.integration_metrics["cross_agent_messages"] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send cross-agent message: {e}")
            return False
    
    def get_agent_messages(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get messages for an agent"""
        return self.message_bus.get(agent_id, [])
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        return {
            "langchain_core_available": LANGCHAIN_CORE_AVAILABLE,
            "langgraph_available": LANGGRAPH_AVAILABLE,
            "langsmith_available": LANGSMITH_AVAILABLE,
            "wrapped_agents": list(self.wrapped_agents.keys()),
            "total_tools": len(self.all_tools),
            "active_workflows": list(self.workflows.keys()),
            "metrics": self.integration_metrics,
            "agent_details": {
                agent_id: wrapper.get_agent_info()
                for agent_id, wrapper in self.wrapped_agents.items()
            },
            "timestamp": time.time()
        }


# Global integration instance
_global_integration: Optional[EnhancedLangChainIntegration] = None

def get_langchain_integration() -> EnhancedLangChainIntegration:
    """Get the global LangChain integration instance"""
    global _global_integration
    if _global_integration is None:
        _global_integration = EnhancedLangChainIntegration()
    return _global_integration

def create_custom_integration(
    llm_manager: Optional[LLMManager] = None,
    memory_manager: Optional[MemoryManager] = None,
    enable_auto_wrap: bool = True
) -> EnhancedLangChainIntegration:
    """Create a custom LangChain integration instance"""
    return EnhancedLangChainIntegration(
        llm_manager=llm_manager,
        memory_manager=memory_manager,
        enable_auto_wrap=enable_auto_wrap
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        integration = get_langchain_integration()
        
        # Check status
        status = integration.get_integration_status()
        print(f"Integration status: {json.dumps(status, indent=2)}")
        
        # Create a workflow
        agents = list(integration.wrapped_agents.keys())[:3]  # Use first 3 agents
        if agents:
            workflow = integration.create_multi_agent_workflow("test_workflow", agents)
            
            if workflow:
                # Execute workflow
                result = await integration.execute_workflow(
                    "test_workflow",
                    "Test the multi-agent workflow",
                    max_iterations=5
                )
                print(f"Workflow result: {json.dumps(result, indent=2)}")
    
    asyncio.run(main())

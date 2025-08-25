"""
MCP Server for NIS Protocol

Main MCP server implementation that bridges Deep Agents with mcp-ui.
Handles tool registration, request routing, and UI resource generation.
"""

import json
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..agents.deep import DeepAgentPlanner, DatasetSkill, PipelineSkill, ResearchSkill, AuditSkill, CodeSkill
from ..core.agent import NISAgent
from ..memory.memory_manager import MemoryManager
from .schemas import ToolSchemas
from .ui_resources import UIResourceGenerator
from .intent_validator import IntentValidator


@dataclass
class MCPRequest:
    """Represents an MCP tool request."""
    tool_name: str
    parameters: Dict[str, Any]
    request_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MCPResponse:
    """Represents an MCP tool response."""
    request_id: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    ui_resource: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MCPToolHandler(ABC):
    """Abstract base class for MCP tool handlers."""
    
    @abstractmethod
    async def handle(self, request: MCPRequest) -> MCPResponse:
        """Handle an MCP tool request."""
        pass
        
    @abstractmethod
    def get_tool_names(self) -> List[str]:
        """Get list of tool names this handler supports."""
        pass


class DeepAgentToolHandler(MCPToolHandler):
    """
    Tool handler that routes MCP requests to Deep Agent skills.
    
    Maps MCP tools to Deep Agent skills and generates appropriate UI resources.
    """
    
    def __init__(self, planner: DeepAgentPlanner, ui_generator: UIResourceGenerator):
        self.planner = planner
        self.ui_generator = ui_generator
        self.skill_mapping = {
            "dataset": "dataset",
            "pipeline": "pipeline", 
            "research": "research",
            "audit": "audit",
            "code": "code"
        }
        
    async def handle(self, request: MCPRequest) -> MCPResponse:
        """Route MCP request to appropriate Deep Agent skill."""
        try:
            # Parse tool name (e.g., "dataset.search" -> skill="dataset", action="search")
            if "." not in request.tool_name:
                return MCPResponse(
                    request_id=request.request_id,
                    success=False,
                    error=f"Invalid tool name format: {request.tool_name}"
                )
                
            skill_name, action = request.tool_name.split(".", 1)
            
            if skill_name not in self.skill_mapping:
                return MCPResponse(
                    request_id=request.request_id,
                    success=False,
                    error=f"Unknown skill: {skill_name}"
                )
                
            # Get the skill instance
            skill = self.planner.skills.get(skill_name)
            if not skill:
                return MCPResponse(
                    request_id=request.request_id,
                    success=False,
                    error=f"Skill not registered: {skill_name}"
                )
                
            # Execute the skill action
            result = await skill.execute(action, request.parameters)
            
            if not result.get("success", False):
                return MCPResponse(
                    request_id=request.request_id,
                    success=False,
                    error=result.get("error", {}).get("message", "Skill execution failed"),
                    data=result
                )
                
            # Generate UI resource for the result
            ui_resource = await self._generate_ui_resource(
                skill_name, action, request.parameters, result["data"]
            )
            
            return MCPResponse(
                request_id=request.request_id,
                success=True,
                data=result["data"],
                ui_resource=ui_resource,
                metadata={
                    "skill": skill_name,
                    "action": action,
                    "execution_time": result.get("timestamp"),
                    "tool_name": request.tool_name
                }
            )
            
        except Exception as e:
            logging.error(f"Error handling MCP request {request.request_id}: {str(e)}")
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error=f"Internal server error: {str(e)}"
            )
            
    def get_tool_names(self) -> List[str]:
        """Get all supported tool names."""
        tools = []
        for skill_name in self.skill_mapping.keys():
            skill = self.planner.skills.get(skill_name)
            if skill:
                actions = skill.get_available_actions()
                for action in actions:
                    tools.append(f"{skill_name}.{action}")
        return tools
        
    async def _generate_ui_resource(self, skill: str, action: str, parameters: Dict[str, Any], data: Any) -> Dict[str, Any]:
        """Generate appropriate UI resource for the skill result."""
        try:
            if skill == "dataset":
                if action == "search":
                    return self.ui_generator.create_data_grid(
                        data.get("items", []),
                        title="Dataset Search Results",
                        searchable=True,
                        pagination=True
                    )
                elif action == "preview":
                    return self.ui_generator.create_tabbed_viewer({
                        "Schema": data.get("schema", {}),
                        "Sample": data.get("sample_data", []),
                        "Statistics": data.get("statistics", {})
                    })
                    
            elif skill == "pipeline":
                if action == "run":
                    return self.ui_generator.create_progress_monitor(
                        data.get("run_id"),
                        data.get("status", "unknown"),
                        data.get("progress", 0)
                    )
                elif action == "status":
                    return self.ui_generator.create_pipeline_status(data)
                    
            elif skill == "research":
                if action == "plan":
                    return self.ui_generator.create_research_plan_tree(data)
                elif action == "search":
                    return self.ui_generator.create_research_results(data.get("results", []))
                    
            elif skill == "audit":
                if action == "view":
                    return self.ui_generator.create_audit_timeline(data.get("timeline", []))
                elif action == "analyze":
                    return self.ui_generator.create_analysis_dashboard(data)
                    
            elif skill == "code":
                if action == "edit":
                    return self.ui_generator.create_diff_viewer(data.get("diff", []))
                elif action == "review":
                    return self.ui_generator.create_code_review_panel(data)
                    
            # Fallback to generic data viewer
            return self.ui_generator.create_data_viewer(data)
            
        except Exception as e:
            logging.warning(f"Failed to generate UI resource for {skill}.{action}: {str(e)}")
            return self.ui_generator.create_data_viewer(data)


class MCPServer:
    """
    Main MCP Server for NIS Protocol.
    
    Orchestrates Deep Agents, tool handling, and UI resource generation
    to provide a rich, interactive MCP experience.
    """
    
    def __init__(self, agent: NISAgent, memory_manager: MemoryManager, config: Dict[str, Any] = None):
        self.agent = agent
        self.memory = memory_manager
        self.config = config or {}
        
        # Initialize components
        self.schemas = ToolSchemas()
        self.ui_generator = UIResourceGenerator()
        self.intent_validator = IntentValidator()
        
        # Initialize Deep Agent system
        self.planner = DeepAgentPlanner(agent, memory_manager)
        self._setup_skills()
        
        # Initialize handlers
        self.tool_handler = DeepAgentToolHandler(self.planner, self.ui_generator)
        
        # Server state
        self.active_sessions = {}
        self.request_handlers: Dict[str, Callable] = {}
        self.middleware_stack = []
        
        logging.info("MCP Server initialized with Deep Agents integration")
        
    def _setup_skills(self):
        """Initialize and register Deep Agent skills."""
        skills = [
            ("dataset", DatasetSkill(self.agent, self.memory)),
            ("pipeline", PipelineSkill(self.agent, self.memory)),
            ("research", ResearchSkill(self.agent, self.memory)),
            ("audit", AuditSkill(self.agent, self.memory)),
            ("code", CodeSkill(self.agent, self.memory))
        ]
        
        for name, skill in skills:
            self.planner.register_skill(name, skill)
            
        logging.info(f"Registered {len(skills)} Deep Agent skills")
        
    async def start_server(self, host: str = "localhost", port: int = 8000):
        """Start the MCP server."""
        # This would typically start an HTTP server or other transport
        # For now, we'll set up the infrastructure
        logging.info(f"MCP Server starting on {host}:{port}")
        
        # Register tool endpoints
        await self._register_tools()
        
        # Setup intent handlers  
        await self._setup_intent_handlers()
        
        logging.info("MCP Server started successfully")
        
    async def _register_tools(self):
        """Register all MCP tools with their schemas."""
        tools = self.schemas.get_mcp_tool_definitions()
        
        for tool in tools:
            tool_name = tool["name"]
            self.request_handlers[tool_name] = self._handle_tool_request
            
        logging.info(f"Registered {len(tools)} MCP tools")
        
    async def _setup_intent_handlers(self):
        """Setup handlers for UI intents from mcp-ui."""
        intent_handlers = {
            "tool": self._handle_tool_intent,
            "intent": self._handle_generic_intent, 
            "prompt": self._handle_prompt_intent,
            "notify": self._handle_notify_intent,
            "link": self._handle_link_intent
        }
        
        for intent_type, handler in intent_handlers.items():
            self.intent_validator.register_handler(intent_type, handler)
            
        logging.info(f"Registered {len(intent_handlers)} intent handlers")
        
    async def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main request handler for MCP requests.
        
        Supports both tool calls and UI intents.
        """
        try:
            request_type = request_data.get("type", "tool")
            
            if request_type == "tool":
                return await self._handle_tool_request(request_data)
            elif request_type == "intent":
                return await self._handle_intent_request(request_data)
            else:
                return {
                    "success": False,
                    "error": f"Unknown request type: {request_type}"
                }
                
        except Exception as e:
            logging.error(f"Error handling request: {str(e)}")
            return {
                "success": False,
                "error": f"Server error: {str(e)}"
            }
            
    async def _handle_tool_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool requests."""
        try:
            # Parse request
            tool_name = request_data.get("tool_name") or request_data.get("function", {}).get("name")
            parameters = request_data.get("parameters", {})
            request_id = request_data.get("request_id", f"req_{int(time.time() * 1000)}")
            
            if not tool_name:
                return {
                    "success": False,
                    "error": "Missing tool_name in request"
                }
                
            # Validate tool exists
            if tool_name not in self.request_handlers:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}"
                }
                
            # Validate input parameters
            is_valid, errors = self.schemas.validate_tool_input(tool_name, parameters)
            if not is_valid:
                return {
                    "success": False,
                    "error": f"Invalid parameters: {'; '.join(errors)}"
                }
                
            # Create MCP request
            mcp_request = MCPRequest(
                tool_name=tool_name,
                parameters=parameters,
                request_id=request_id,
                user_id=request_data.get("user_id"),
                session_id=request_data.get("session_id"),
                metadata=request_data.get("metadata", {})
            )
            
            # Handle request
            response = await self.tool_handler.handle(mcp_request)
            
            # Format response
            result = {
                "request_id": response.request_id,
                "success": response.success,
                "data": response.data,
                "metadata": response.metadata
            }
            
            if response.ui_resource:
                result["ui_resource"] = response.ui_resource
                
            if response.error:
                result["error"] = response.error
                
            return result
            
        except Exception as e:
            logging.error(f"Error in tool request handler: {str(e)}")
            return {
                "success": False,
                "error": f"Tool request failed: {str(e)}"
            }
            
    async def _handle_intent_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle UI intent requests from mcp-ui."""
        try:
            intent_type = request_data.get("intent_type")
            payload = request_data.get("payload", {})
            message_id = request_data.get("message_id")
            
            if not intent_type:
                return {
                    "success": False,
                    "error": "Missing intent_type in request"
                }
                
            # Validate intent
            is_valid, errors = self.intent_validator.validate_intent(intent_type, payload)
            if not is_valid:
                return {
                    "success": False,
                    "error": f"Invalid intent: {'; '.join(errors)}"
                }
                
            # Handle intent
            result = await self.intent_validator.handle_intent(intent_type, payload, message_id)
            
            return {
                "success": True,
                "intent_type": intent_type,
                "message_id": message_id,
                "result": result
            }
            
        except Exception as e:
            logging.error(f"Error in intent request handler: {str(e)}")
            return {
                "success": False,
                "error": f"Intent request failed: {str(e)}"
            }
            
    async def _handle_tool_intent(self, payload: Dict[str, Any], message_id: str = None) -> Dict[str, Any]:
        """Handle tool execution intents from UI."""
        tool_name = payload.get("toolName")
        params = payload.get("params", {})
        
        if not tool_name:
            return {"error": "Missing toolName in payload"}
            
        # Execute tool request
        request_data = {
            "tool_name": tool_name,
            "parameters": params,
            "request_id": f"intent_{message_id}_{int(time.time() * 1000)}",
            "metadata": {"source": "ui_intent", "message_id": message_id}
        }
        
        return await self._handle_tool_request(request_data)
        
    async def _handle_generic_intent(self, payload: Dict[str, Any], message_id: str = None) -> Dict[str, Any]:
        """Handle generic intents from UI."""
        intent = payload.get("intent")
        params = payload.get("params", {})
        
        # Route based on intent
        if intent == "refresh":
            return {"message": "UI refreshed", "timestamp": time.time()}
        elif intent == "navigate":
            return {"message": f"Navigation to {params.get('url', 'unknown')}", "params": params}
        else:
            return {"message": f"Handled intent: {intent}", "params": params}
            
    async def _handle_prompt_intent(self, payload: Dict[str, Any], message_id: str = None) -> Dict[str, Any]:
        """Handle prompt intents from UI."""
        prompt = payload.get("prompt")
        
        if not prompt:
            return {"error": "Missing prompt in payload"}
            
        # Send prompt to agent
        response = await self.agent.process_request({
            "action": "process_prompt",
            "data": {"prompt": prompt, "source": "ui_intent"},
            "metadata": {"message_id": message_id}
        })
        
        return {"response": response, "prompt": prompt}
        
    async def _handle_notify_intent(self, payload: Dict[str, Any], message_id: str = None) -> Dict[str, Any]:
        """Handle notification intents from UI."""
        message = payload.get("message", "")
        
        logging.info(f"UI notification: {message}")
        
        return {"acknowledged": True, "message": message}
        
    async def _handle_link_intent(self, payload: Dict[str, Any], message_id: str = None) -> Dict[str, Any]:
        """Handle link click intents from UI.""" 
        url = payload.get("url")
        
        if not url:
            return {"error": "Missing url in payload"}
            
        # Log the link click - could implement navigation logic here
        logging.info(f"UI link clicked: {url}")
        
        return {"url": url, "action": "link_clicked"}
        
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information and capabilities."""
        tools = self.tool_handler.get_tool_names()
        
        return {
            "name": "NIS Protocol MCP Server",
            "version": "1.0.0",
            "description": "Deep Agents + mcp-ui integration for NIS Protocol",
            "capabilities": {
                "tools": len(tools),
                "deep_agents": True,
                "ui_resources": True,
                "intent_handling": True
            },
            "tools": tools,
            "skills": list(self.planner.skills.keys()),
            "ui_components": self.ui_generator.get_supported_components()
        }

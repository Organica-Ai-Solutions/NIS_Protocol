"""
MCP Integration Module for NIS Protocol

Integrates the MCP + Deep Agents + mcp-ui stack into the main NIS Protocol application.
Provides easy setup and configuration for production use.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from ..core.agent import NISAgent
from ..memory.memory_manager import MemoryManager
from ..core.registry import NISRegistry
from .server import MCPServer
from .demo import MCPDemo


class MCPIntegration:
    """
    Main integration class for MCP functionality in NIS Protocol.
    
    Handles initialization, configuration, and lifecycle management
    of the MCP server and Deep Agents integration.
    """
    
    def __init__(self, nis_config: Dict[str, Any] = None):
        self.nis_config = nis_config or {}
        self.mcp_server: Optional[MCPServer] = None
        self.agent: Optional[Agent] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.is_initialized = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self, agent: NISAgent = None, memory_manager: MemoryManager = None):
        """
        Initialize the MCP integration with NIS Protocol components.
        
        Args:
            agent: Existing NIS Agent instance (optional)
            memory_manager: Existing memory manager (optional)
        """
        self.logger.info("Initializing MCP integration...")
        
        try:
            # Use provided components or create new ones
            if agent:
                self.agent = agent
            else:
                self.agent = await self._create_agent()
                
            if memory_manager:
                self.memory_manager = memory_manager
            else:
                self.memory_manager = await self._create_memory_manager()
            
            # Create MCP server
            mcp_config = self.nis_config.get("mcp", {})
            self.mcp_server = MCPServer(self.agent, self.memory_manager, mcp_config)
            
            # Start the server
            host = mcp_config.get("host", "localhost")
            port = mcp_config.get("port", 8000)
            await self.mcp_server.start_server(host, port)
            
            self.is_initialized = True
            self.logger.info(f"MCP integration initialized on {host}:{port}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP integration: {str(e)}")
            raise
            
    async def _create_agent(self) -> NISAgent:
        """Create a new Agent instance for MCP."""
        agent_config = self.nis_config.get("agent", {})
        agent = NISAgent(agent_id="mcp_agent")
        # NISAgent doesn't have initialize method, it's ready after construction
        return agent
        
    async def _create_memory_manager(self) -> MemoryManager:
        """Create a new MemoryManager instance for MCP."""
        memory_config = self.nis_config.get("memory", {})
        # Create appropriate storage backend based on config
        backend_type = memory_config.get("backend", "in_memory")
        
        if backend_type == "sqlite":
            # SQLite backend not implemented yet, fall back to in-memory
            from ..memory.memory_manager import InMemoryStorage
            storage_backend = InMemoryStorage()
        else:
            # Default to in-memory storage
            from ..memory.memory_manager import InMemoryStorage
            storage_backend = InMemoryStorage()
        
        memory_manager = MemoryManager(storage_backend=storage_backend)
        return memory_manager
        
    async def handle_mcp_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an MCP request through the integrated server.
        
        Args:
            request: MCP request data
            
        Returns:
            MCP response with data and optional UI resources
        """
        if not self.is_initialized:
            raise RuntimeError("MCP integration not initialized")
            
        return await self.mcp_server.handle_request(request)
        
    def get_tool_registry(self) -> Dict[str, Any]:
        """Get the registry of available MCP tools."""
        if not self.is_initialized:
            return {}
            
        return self.mcp_server.schemas.get_all_tools()
        
    def get_server_info(self) -> Dict[str, Any]:
        """Get information about the MCP server capabilities."""
        if not self.is_initialized:
            return {"status": "not_initialized"}
            
        info = self.mcp_server.get_server_info()
        info["integration_status"] = "active"
        return info
        
    async def create_execution_plan(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a Deep Agent execution plan for a complex goal.
        
        Args:
            goal: High-level goal to achieve
            context: Additional context for planning
            
        Returns:
            Execution plan with steps and dependencies
        """
        if not self.is_initialized:
            raise RuntimeError("MCP integration not initialized")
            
        plan = await self.mcp_server.planner.create_plan(goal, context)
        return {
            "plan_id": plan.id,
            "goal": plan.goal,
            "steps": [
                {
                    "id": step.id,
                    "skill": step.skill,
                    "action": step.action,
                    "description": step.description,
                    "status": step.status.value,
                    "dependencies": step.dependencies
                }
                for step in plan.steps
            ],
            "created_at": plan.created_at,
            "metadata": plan.metadata
        }
        
    async def execute_plan(self, plan_id: str) -> Dict[str, Any]:
        """
        Execute a Deep Agent plan.
        
        Args:
            plan_id: ID of the plan to execute
            
        Returns:
            Execution results and status
        """
        if not self.is_initialized:
            raise RuntimeError("MCP integration not initialized")
            
        return await self.mcp_server.planner.execute_plan(plan_id)
        
    async def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """Get the status of a Deep Agent plan."""
        if not self.is_initialized:
            raise RuntimeError("MCP integration not initialized")
            
        return self.mcp_server.planner.get_plan_status(plan_id)
        
    def generate_ui_resource(self, data_type: str, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Generate a UI resource for given data.
        
        Args:
            data_type: Type of UI resource (grid, tabs, progress, etc.)
            data: Data to display
            **kwargs: Additional parameters for UI generation
            
        Returns:
            UI resource suitable for mcp-ui rendering
        """
        if not self.is_initialized:
            raise RuntimeError("MCP integration not initialized")
            
        ui_gen = self.mcp_server.ui_generator
        
        if data_type == "grid":
            return ui_gen.create_data_grid(data, **kwargs)
        elif data_type == "tabs":
            return ui_gen.create_tabbed_viewer(data, **kwargs)
        elif data_type == "progress":
            return ui_gen.create_progress_monitor(data, **kwargs)
        elif data_type == "timeline":
            return ui_gen.create_audit_timeline(data, **kwargs)
        elif data_type == "diff":
            return ui_gen.create_diff_viewer(data, **kwargs)
        else:
            return ui_gen.create_data_viewer(data, **kwargs)
            
    async def run_demo(self) -> Dict[str, Any]:
        """Run the MCP integration demo."""
        if not self.is_initialized:
            await self.initialize()
            
        demo = MCPDemo()
        demo.server = self.mcp_server  # Use our initialized server
        return await demo.run_complete_demo()
        
    async def shutdown(self):
        """Shutdown the MCP integration."""
        self.logger.info("Shutting down MCP integration...")
        
        # Add shutdown logic here (stop server, cleanup resources, etc.)
        if self.memory_manager:
            await self.memory_manager.cleanup()
            
        self.is_initialized = False
        self.logger.info("MCP integration shutdown complete")
        
    def export_configuration(self) -> Dict[str, Any]:
        """Export the current MCP configuration."""
        return {
            "mcp_server": {
                "tools": list(self.get_tool_registry().keys()) if self.is_initialized else [],
                "capabilities": self.get_server_info().get("capabilities", {}) if self.is_initialized else {}
            },
            "deep_agents": {
                "skills": list(self.mcp_server.planner.skills.keys()) if self.is_initialized else [],
                "active_plans": len(self.mcp_server.planner.active_plans) if self.is_initialized else 0
            },
            "ui_resources": {
                "supported_components": self.mcp_server.ui_generator.get_supported_components() if self.is_initialized else []
            },
            "configuration": self.nis_config
        }


# Convenience functions for easy integration

async def setup_mcp_integration(nis_config: Dict[str, Any] = None, 
                               agent: NISAgent = None, 
                               memory_manager: MemoryManager = None) -> MCPIntegration:
    """
    Setup MCP integration with NIS Protocol.
    
    Args:
        nis_config: NIS Protocol configuration
        agent: Existing agent instance (optional)
        memory_manager: Existing memory manager (optional)
        
    Returns:
        Configured MCPIntegration instance
    """
    integration = MCPIntegration(nis_config)
    await integration.initialize(agent, memory_manager)
    return integration


def create_mcp_endpoint_handler(integration: MCPIntegration):
    """
    Create an endpoint handler for web frameworks.
    
    Args:
        integration: Configured MCPIntegration instance
        
    Returns:
        Async handler function for MCP requests
    """
    async def handle_mcp_endpoint(request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP endpoint request."""
        try:
            return await integration.handle_mcp_request(request_data)
        except Exception as e:
            return {
                "success": False,
                "error": f"MCP request failed: {str(e)}"
            }
    
    return handle_mcp_endpoint


def add_mcp_routes(app, integration: MCPIntegration, base_path: str = "/mcp"):
    """
    Add MCP routes to a web application (framework-agnostic).
    
    Args:
        app: Web application instance
        integration: Configured MCPIntegration instance  
        base_path: Base path for MCP routes
    """
    # This is a generic template - adapt for specific frameworks
    
    # Example for FastAPI:
    # @app.post(f"{base_path}/tools")
    # async def mcp_tools(request: dict):
    #     return await integration.handle_mcp_request(request)
    
    # @app.get(f"{base_path}/info")
    # async def mcp_info():
    #     return integration.get_server_info()
    
    # @app.post(f"{base_path}/plans")
    # async def create_plan(request: dict):
    #     return await integration.create_execution_plan(
    #         request["goal"], request.get("context", {})
    #     )
    
    logging.info(f"MCP routes would be added to {base_path} (framework-specific implementation needed)")


# Example configuration
EXAMPLE_CONFIG = {
    "mcp": {
        "host": "localhost",
        "port": 8000,
        "enable_ui": True,
        "security": {
            "validate_intents": True,
            "sandbox_ui": True
        }
    },
    "agent": {
        "provider": "anthropic",  # Real provider for production
        "model": "claude-3-sonnet-20241022",
        "max_tokens": 4000
    },
    "memory": {
        "backend": "sqlite",
        "connection_string": "data/mcp_memory.db",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
    }
}

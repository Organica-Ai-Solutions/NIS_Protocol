"""
NIS Protocol - Local MCP Server
================================
A Model Context Protocol (MCP) server that exposes NIS Protocol capabilities
as MCP tools and resources.

Run standalone:
    python -m src.protocols.mcp_server

Or import and run programmatically:
    from src.protocols.mcp_server import NISMCPServer
    server = NISMCPServer()
    server.run()

MCP Specification: https://modelcontextprotocol.io/
"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# FastAPI for HTTP server
try:
    from fastapi import FastAPI, HTTPException, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPMessageType(str, Enum):
    """MCP JSON-RPC message types"""
    INITIALIZE = "initialize"
    INITIALIZED = "initialized"
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"
    PING = "ping"
    PONG = "pong"


@dataclass
class MCPTool:
    """MCP Tool definition"""
    name: str
    description: str
    inputSchema: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MCPResource:
    """MCP Resource definition"""
    uri: str
    name: str
    description: str
    mimeType: str = "application/json"
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MCPPrompt:
    """MCP Prompt template"""
    name: str
    description: str
    arguments: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class NISMCPServer:
    """
    NIS Protocol MCP Server
    
    Exposes NIS Protocol capabilities via Model Context Protocol:
    - Tools: Physics, Robotics, Research, Code Execution
    - Resources: Agent status, System info, Conversation history
    - Prompts: Analysis templates, Research templates
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 3000):
        self.host = host
        self.port = port
        self.server_info = {
            "name": "nis-protocol-mcp",
            "version": "1.0.0",
            "protocolVersion": "2024-11-05"
        }
        self.capabilities = {
            "tools": {},
            "resources": {},
            "prompts": {}
        }
        
        # Initialize tools, resources, and prompts
        self._init_tools()
        self._init_resources()
        self._init_prompts()
        
        # Session tracking
        self.sessions: Dict[str, Dict] = {}
        
        if HAS_FASTAPI:
            self.app = self._create_app()
        else:
            self.app = None
            logger.warning("FastAPI not available - MCP server will run in limited mode")
    
    def _init_tools(self):
        """Initialize available MCP tools"""
        self.tools: List[MCPTool] = [
            # Physics Tools
            MCPTool(
                name="nis.physics.solve",
                description="Solve physics equations (heat, wave, Laplace, Poisson)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "equation_type": {"type": "string", "enum": ["heat", "wave", "laplace", "poisson"]},
                        "parameters": {"type": "object"}
                    },
                    "required": ["equation_type"]
                }
            ),
            MCPTool(
                name="nis.physics.validate",
                description="Validate physics claims against known laws",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "domain": {"type": "string"}
                    },
                    "required": ["claim"]
                }
            ),
            
            # Robotics Tools
            MCPTool(
                name="nis.robotics.forward_kinematics",
                description="Compute robot end-effector position from joint angles",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "robot_type": {"type": "string", "enum": ["drone", "manipulator", "humanoid", "ground_vehicle"]},
                        "joint_angles": {"type": "array", "items": {"type": "number"}}
                    },
                    "required": ["robot_type", "joint_angles"]
                }
            ),
            MCPTool(
                name="nis.robotics.inverse_kinematics",
                description="Compute joint angles to reach target position",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "robot_type": {"type": "string"},
                        "target_position": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}}
                    },
                    "required": ["robot_type", "target_position"]
                }
            ),
            
            # Research Tools
            MCPTool(
                name="nis.research.deep",
                description="Perform deep research on a topic",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "depth": {"type": "string", "enum": ["quick", "standard", "deep"]}
                    },
                    "required": ["topic"]
                }
            ),
            MCPTool(
                name="nis.research.validate_claim",
                description="Validate a research claim",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "domain": {"type": "string"}
                    },
                    "required": ["claim"]
                }
            ),
            
            # Code Execution Tools
            MCPTool(
                name="nis.code.execute",
                description="Execute Python code in sandboxed environment",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "language": {"type": "string", "default": "python"}
                    },
                    "required": ["code"]
                }
            ),
            
            # Agent Tools
            MCPTool(
                name="nis.agent.run",
                description="Run a NIS Protocol agent",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_name": {"type": "string"},
                        "input": {"type": "string"}
                    },
                    "required": ["agent_name", "input"]
                }
            ),
            
            # Chat Tools
            MCPTool(
                name="nis.chat",
                description="Send a message to NIS Protocol LLM",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "provider": {"type": "string", "enum": ["anthropic", "deepseek", "google", "nvidia", "kimi"]}
                    },
                    "required": ["message"]
                }
            )
        ]
    
    def _init_resources(self):
        """Initialize available MCP resources"""
        self.resources: List[MCPResource] = [
            MCPResource(
                uri="nis://agents/status",
                name="Agent Status",
                description="Current status of all NIS Protocol agents"
            ),
            MCPResource(
                uri="nis://system/health",
                name="System Health",
                description="NIS Protocol system health information"
            ),
            MCPResource(
                uri="nis://models/available",
                name="Available Models",
                description="List of available LLM models and providers"
            ),
            MCPResource(
                uri="nis://physics/constants",
                name="Physics Constants",
                description="Physical constants database"
            ),
            MCPResource(
                uri="nis://robotics/capabilities",
                name="Robotics Capabilities",
                description="Supported robot types and capabilities"
            )
        ]
    
    def _init_prompts(self):
        """Initialize available MCP prompts"""
        self.prompts: List[MCPPrompt] = [
            MCPPrompt(
                name="physics_analysis",
                description="Template for physics-informed analysis",
                arguments=[
                    {"name": "topic", "description": "Physics topic to analyze", "required": True},
                    {"name": "depth", "description": "Analysis depth", "required": False}
                ]
            ),
            MCPPrompt(
                name="research_synthesis",
                description="Template for research synthesis",
                arguments=[
                    {"name": "topic", "description": "Research topic", "required": True},
                    {"name": "sources", "description": "Number of sources", "required": False}
                ]
            ),
            MCPPrompt(
                name="code_review",
                description="Template for code review",
                arguments=[
                    {"name": "code", "description": "Code to review", "required": True},
                    {"name": "language", "description": "Programming language", "required": False}
                ]
            )
        ]
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application"""
        app = FastAPI(
            title="NIS Protocol MCP Server",
            description="Model Context Protocol server for NIS Protocol",
            version="1.0.0"
        )
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # JSON-RPC endpoint
        @app.post("/")
        async def handle_jsonrpc(request: dict):
            return await self._handle_jsonrpc(request)
        
        # REST endpoints for compatibility
        @app.get("/health")
        async def health():
            return {"status": "healthy", "server": self.server_info}
        
        @app.get("/tools")
        async def list_tools():
            return {"tools": [t.to_dict() for t in self.tools]}
        
        @app.get("/resources")
        async def list_resources():
            return {"resources": [r.to_dict() for r in self.resources]}
        
        @app.get("/prompts")
        async def list_prompts():
            return {"prompts": [p.to_dict() for p in self.prompts]}
        
        @app.post("/tools/{tool_name}")
        async def call_tool(tool_name: str, arguments: dict = {}):
            return await self._execute_tool(tool_name, arguments)
        
        # WebSocket for streaming
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = {"websocket": websocket, "created": time.time()}
            
            try:
                while True:
                    data = await websocket.receive_json()
                    response = await self._handle_jsonrpc(data)
                    await websocket.send_json(response)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                del self.sessions[session_id]
        
        return app
    
    async def _handle_jsonrpc(self, request: dict) -> dict:
        """Handle JSON-RPC 2.0 request"""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            if method == MCPMessageType.INITIALIZE:
                result = {
                    "protocolVersion": self.server_info["protocolVersion"],
                    "serverInfo": self.server_info,
                    "capabilities": self.capabilities
                }
            
            elif method == MCPMessageType.TOOLS_LIST:
                result = {"tools": [t.to_dict() for t in self.tools]}
            
            elif method == MCPMessageType.TOOLS_CALL:
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})
                result = await self._execute_tool(tool_name, arguments)
            
            elif method == MCPMessageType.RESOURCES_LIST:
                result = {"resources": [r.to_dict() for r in self.resources]}
            
            elif method == MCPMessageType.RESOURCES_READ:
                uri = params.get("uri", "")
                result = await self._read_resource(uri)
            
            elif method == MCPMessageType.PROMPTS_LIST:
                result = {"prompts": [p.to_dict() for p in self.prompts]}
            
            elif method == MCPMessageType.PROMPTS_GET:
                name = params.get("name", "")
                result = await self._get_prompt(name, params.get("arguments", {}))
            
            elif method == MCPMessageType.PING:
                result = {"type": "pong"}
            
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"}
                }
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"JSON-RPC error: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32000, "message": str(e)}
            }
    
    async def _execute_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute an MCP tool by calling NIS Protocol API"""
        import aiohttp
        
        nis_base_url = os.getenv("NIS_API_URL", "http://localhost:8000")
        
        # Map MCP tools to NIS Protocol endpoints
        tool_mapping = {
            "nis.physics.solve": ("POST", "/physics/solve"),
            "nis.physics.validate": ("POST", "/physics/validate"),
            "nis.robotics.forward_kinematics": ("POST", "/robotics/forward_kinematics"),
            "nis.robotics.inverse_kinematics": ("POST", "/robotics/inverse_kinematics"),
            "nis.research.deep": ("POST", "/research/deep"),
            "nis.research.validate_claim": ("POST", "/research/validate-claim"),
            "nis.code.execute": ("POST", "/execute"),
            "nis.agent.run": ("POST", "/runs"),
            "nis.chat": ("POST", "/chat")
        }
        
        if tool_name not in tool_mapping:
            return {"error": f"Unknown tool: {tool_name}"}
        
        method, endpoint = tool_mapping[tool_name]
        url = f"{nis_base_url}{endpoint}"
        
        try:
            async with aiohttp.ClientSession() as session:
                if method == "POST":
                    async with session.post(url, json=arguments, timeout=30) as response:
                        result = await response.json()
                else:
                    async with session.get(url, timeout=30) as response:
                        result = await response.json()
                
                return {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                    "isError": False
                }
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "isError": True
            }
    
    async def _read_resource(self, uri: str) -> dict:
        """Read an MCP resource"""
        import aiohttp
        
        nis_base_url = os.getenv("NIS_API_URL", "http://localhost:8000")
        
        # Map resource URIs to NIS Protocol endpoints
        resource_mapping = {
            "nis://agents/status": "/agents/status",
            "nis://system/health": "/health",
            "nis://models/available": "/models",
            "nis://physics/constants": "/physics/constants",
            "nis://robotics/capabilities": "/robotics/capabilities"
        }
        
        if uri not in resource_mapping:
            return {"error": f"Unknown resource: {uri}"}
        
        endpoint = resource_mapping[uri]
        url = f"{nis_base_url}{endpoint}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    result = await response.json()
                    return {
                        "contents": [{
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(result, indent=2)
                        }]
                    }
        except Exception as e:
            logger.error(f"Resource read error: {e}")
            return {"error": str(e)}
    
    async def _get_prompt(self, name: str, arguments: dict) -> dict:
        """Get a prompt template with arguments filled in"""
        prompts_content = {
            "physics_analysis": """Analyze the following physics topic using NIS Protocol's physics-informed approach:

Topic: {topic}
Depth: {depth}

Please provide:
1. Fundamental principles involved
2. Mathematical formulation
3. Physical constraints and validation
4. Practical applications""",
            
            "research_synthesis": """Synthesize research on the following topic:

Topic: {topic}
Sources to analyze: {sources}

Please provide:
1. Key findings from literature
2. Current state of research
3. Open questions and gaps
4. Future directions""",
            
            "code_review": """Review the following code:

Language: {language}
```
{code}
```

Please analyze:
1. Code quality and style
2. Potential bugs or issues
3. Performance considerations
4. Suggested improvements"""
        }
        
        if name not in prompts_content:
            return {"error": f"Unknown prompt: {name}"}
        
        template = prompts_content[name]
        
        # Fill in arguments with defaults
        filled = template.format(
            topic=arguments.get("topic", "[topic]"),
            depth=arguments.get("depth", "standard"),
            sources=arguments.get("sources", "5"),
            language=arguments.get("language", "python"),
            code=arguments.get("code", "[code]")
        )
        
        return {
            "description": f"Prompt: {name}",
            "messages": [{"role": "user", "content": {"type": "text", "text": filled}}]
        }
    
    def run(self):
        """Run the MCP server"""
        if not HAS_FASTAPI:
            logger.error("FastAPI required to run MCP server. Install with: pip install fastapi uvicorn aiohttp")
            return
        
        logger.info(f"🚀 Starting NIS Protocol MCP Server on {self.host}:{self.port}")
        logger.info(f"📚 Tools: {len(self.tools)}, Resources: {len(self.resources)}, Prompts: {len(self.prompts)}")
        
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NIS Protocol MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=3000, help="Port to listen on")
    args = parser.parse_args()
    
    server = NISMCPServer(host=args.host, port=args.port)
    server.run()

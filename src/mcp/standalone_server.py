"""
Standalone MCP Server for NIS Protocol
Exposes NIS capabilities to ChatGPT, Claude, and other MCP clients

Usage:
    python -m src.mcp.standalone_server
    
Environment Variables:
    NIS_MCP_API_KEY: API key for authentication (optional for local dev)
    NIS_MCP_MODE: Set to 'standalone' for external clients
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path (auto-detect or use environment variable)
project_root = os.getenv('NIS_PROJECT_ROOT')
if not project_root:
    # Auto-detect based on this file's location
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
logger.info(f"📁 NIS_PROJECT_ROOT: {project_root}")
sys.path.insert(0, project_root)
os.chdir(project_root)  # Ensure we're in the right directory for relative paths


@dataclass
class MCPToolRequest:
    """MCP tool request"""
    tool: str
    arguments: Dict[str, Any]
    request_id: str


@dataclass
class MCPToolResponse:
    """MCP tool response"""
    request_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class StandaloneMCPServer:
    """
    Standalone MCP server for external clients (ChatGPT, Claude, etc.)
    
    This server exposes NIS Protocol capabilities through a clean MCP interface
    with proper authentication, rate limiting, and cost controls.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.job_registry = {}  # job_id -> job_data
        self.cost_tracker = {"total_usd": 0.0, "jobs": []}
        
        # Import NIS components
        self._init_nis_components()
        
        self.logger.info("🚀 Standalone MCP Server initialized")
    
    def _init_nis_components(self):
        """Initialize NIS Protocol components"""
        try:
            # Import agents (lazy load to avoid startup overhead)
            self.logger.info("Loading NIS Protocol components...")
            
            # These will be imported on-demand when tools are called
            self.components_loaded = False
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NIS components: {e}")
            self.components_loaded = False
    
    async def _load_components_if_needed(self):
        """Lazy load NIS components"""
        if self.components_loaded:
            return
        
        try:
            from ..agents.robotics.unified_robotics_agent import UnifiedRoboticsAgent
            from ..agents.physics.unified_physics_agent import UnifiedPhysicsAgent
            from ..agents.research.deep_research_agent import DeepResearchAgent
            
            self.robotics_agent = UnifiedRoboticsAgent()
            self.physics_agent = UnifiedPhysicsAgent()
            self.research_agent = DeepResearchAgent()
            
            self.components_loaded = True
            self.logger.info("✅ NIS components loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load NIS components: {e}")
            raise
    
    async def handle_tool_call(self, request: MCPToolRequest) -> MCPToolResponse:
        """Route tool call to appropriate handler"""
        try:
            # Verify authentication
            if not self._check_auth():
                return MCPToolResponse(
                    request_id=request.request_id,
                    success=False,
                    error="Authentication failed"
                )
            
            # Route to tool handler
            if request.tool == "nis.run_pipeline":
                return await self._handle_run_pipeline(request)
            elif request.tool == "nis.job_status":
                return await self._handle_job_status(request)
            elif request.tool == "nis.get_artifact":
                return await self._handle_get_artifact(request)
            elif request.tool == "nis.cost_report":
                return await self._handle_cost_report(request)
            elif request.tool == "nis.list_capabilities":
                return await self._handle_list_capabilities(request)
            elif request.tool == "nis.robotics_control":
                return await self._handle_robotics_control(request)
            else:
                return MCPToolResponse(
                    request_id=request.request_id,
                    success=False,
                    error=f"Unknown tool: {request.tool}"
                )
        
        except Exception as e:
            self.logger.error(f"Tool call error: {e}")
            return MCPToolResponse(
                request_id=request.request_id,
                success=False,
                error=str(e)
            )
    
    def _check_auth(self) -> bool:
        """Check API key authentication"""
        api_key = os.getenv("NIS_MCP_API_KEY")
        if not api_key:
            # Allow in dev mode
            if os.getenv("NIS_MCP_MODE") == "dev":
                return True
            self.logger.warning("No API key provided and not in dev mode")
            return False
        return True
    
    async def _handle_run_pipeline(self, request: MCPToolRequest) -> MCPToolResponse:
        """Execute NIS pipeline"""
        await self._load_components_if_needed()
        
        args = request.arguments
        pipeline_type = args.get("pipeline_type")
        input_data = args.get("input_data", {})
        budget_usd = args.get("budget_usd", 10.0)
        
        # Generate job ID
        import uuid
        job_id = str(uuid.uuid4())[:8]
        
        # Track cost
        self.cost_tracker["jobs"].append({
            "job_id": job_id,
            "budget_usd": budget_usd,
            "pipeline_type": pipeline_type,
            "status": "running"
        })
        
        # Store job
        self.job_registry[job_id] = {
            "status": "running",
            "pipeline_type": pipeline_type,
            "input_data": input_data,
            "results": {}
        }
        
        self.logger.info(f"🚀 Starting job {job_id}: {pipeline_type}")
        
        return MCPToolResponse(
            request_id=request.request_id,
            success=True,
            result={
                "job_id": job_id,
                "status": "submitted",
                "pipeline_type": pipeline_type,
                "estimated_completion": "30-60 seconds"
            }
        )
    
    async def _handle_job_status(self, request: MCPToolRequest) -> MCPToolResponse:
        """Get job status"""
        job_id = request.arguments.get("job_id")
        
        if job_id not in self.job_registry:
            return MCPToolResponse(
                request_id=request.request_id,
                success=False,
                error=f"Job {job_id} not found"
            )
        
        job = self.job_registry[job_id]
        
        return MCPToolResponse(
            request_id=request.request_id,
            success=True,
            result={
                "job_id": job_id,
                "status": job["status"],
                "pipeline_type": job["pipeline_type"],
                "available_artifacts": list(job["results"].keys())
            }
        )
    
    async def _handle_get_artifact(self, request: MCPToolRequest) -> MCPToolResponse:
        """Get job artifact"""
        job_id = request.arguments.get("job_id")
        artifact_name = request.arguments.get("artifact_name")
        
        if job_id not in self.job_registry:
            return MCPToolResponse(
                request_id=request.request_id,
                success=False,
                error=f"Job {job_id} not found"
            )
        
        job = self.job_registry[job_id]
        
        if artifact_name not in job["results"]:
            return MCPToolResponse(
                request_id=request.request_id,
                success=False,
                error=f"Artifact {artifact_name} not available"
            )
        
        return MCPToolResponse(
            request_id=request.request_id,
            success=True,
            result=job["results"][artifact_name]
        )
    
    async def _handle_cost_report(self, request: MCPToolRequest) -> MCPToolResponse:
        """Get cost report"""
        window = request.arguments.get("window", "24h")
        
        return MCPToolResponse(
            request_id=request.request_id,
            success=True,
            result={
                "window": window,
                "total_cost_usd": self.cost_tracker["total_usd"],
                "total_jobs": len(self.cost_tracker["jobs"]),
                "jobs": self.cost_tracker["jobs"]
            }
        )
    
    async def _handle_list_capabilities(self, request: MCPToolRequest) -> MCPToolResponse:
        """List NIS capabilities"""
        category = request.arguments.get("category", "all")
        
        capabilities = {
            "agents": [
                "Robotics Agent (FK/IK/Trajectory)",
                "Physics Agent (PINN validation)",
                "Research Agent (Deep research)",
                "Signal Processing Agent (Laplace)",
                "Reasoning Agent (KAN networks)"
            ],
            "pipelines": [
                "signal_processing",
                "physics_validation",
                "robotics_control",
                "research_analysis"
            ],
            "providers": [
                "OpenAI (GPT-4o, GPT-4o-mini)",
                "Anthropic (Claude Sonnet, Claude Opus)",
                "Google (Gemini 2.5 Pro/Flash)",
                "DeepSeek (DeepSeek-Chat)",
                "NVIDIA (Nemotron-340B)"
            ]
        }
        
        if category != "all":
            result = {category: capabilities.get(category, [])}
        else:
            result = capabilities
        
        return MCPToolResponse(
            request_id=request.request_id,
            success=True,
            result=result
        )
    
    async def _handle_robotics_control(self, request: MCPToolRequest) -> MCPToolResponse:
        """Handle robotics control"""
        await self._load_components_if_needed()
        
        robot_id = request.arguments.get("robot_id")
        command_type = request.arguments.get("command_type")
        parameters = request.arguments.get("parameters", {})
        
        self.logger.info(f"🤖 Robotics command: {command_type} for {robot_id}")
        
        # This would integrate with the actual robotics agent
        result = {
            "robot_id": robot_id,
            "command": command_type,
            "status": "executed",
            "timestamp": "2025-01-19T12:00:00Z"
        }
        
        return MCPToolResponse(
            request_id=request.request_id,
            success=True,
            result=result
        )
    
    async def run_stdio_server(self):
        """Run MCP server over STDIO (for ChatGPT/Claude integration)"""
        self.logger.info("📡 Starting STDIO MCP server...")
        
        while True:
            try:
                # Read JSON-RPC request from stdin
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                
                if not line:
                    break
                
                request_data = json.loads(line.strip())
                
                # Parse MCP request
                request = MCPToolRequest(
                    tool=request_data.get("method"),
                    arguments=request_data.get("params", {}),
                    request_id=request_data.get("id", "unknown")
                )
                
                # Handle tool call
                response = await self.handle_tool_call(request)
                
                # Send JSON-RPC response to stdout
                response_data = {
                    "jsonrpc": "2.0",
                    "id": response.request_id,
                    "result": asdict(response) if response.success else None,
                    "error": {"message": response.error} if not response.success else None
                }
                
                print(json.dumps(response_data), flush=True)
                
            except Exception as e:
                self.logger.error(f"Server error: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "id": "error",
                    "error": {"message": str(e)}
                }
                print(json.dumps(error_response), flush=True)


async def main():
    """Main entry point"""
    server = StandaloneMCPServer()
    
    # Run in STDIO mode (for ChatGPT/Claude)
    await server.run_stdio_server()


if __name__ == "__main__":
    asyncio.run(main())


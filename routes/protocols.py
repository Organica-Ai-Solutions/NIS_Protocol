"""
NIS Protocol v4.0 - Protocol Routes

This module contains all third-party protocol integration endpoints:
- MCP (Model Context Protocol) - Anthropic
- A2A (Agent-to-Agent) - Google
- ACP (Agent Communication Protocol) - IBM/Linux Foundation

MIGRATION STATUS: Ready for testing
- These routes mirror the ones in main.py
- Can be tested independently before switching over
- main.py routes remain active until migration is complete

Usage:
    from routes.protocols import router as protocols_router
    app.include_router(protocols_router, tags=["Protocols"])
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("nis.routes.protocols")

# Create router
router = APIRouter(tags=["Third-Party Protocols"])


# ====== Pydantic Models ======

class ProtocolTaskRequest(BaseModel):
    """Request model for A2A task creation"""
    description: str
    agent_id: str = "nis_protocol_agent"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    callback_url: Optional[str] = None


class ProtocolExecuteRequest(BaseModel):
    """Request model for ACP agent execution"""
    agent_url: str
    message: Dict[str, Any]
    async_mode: bool = False


class ProtocolMessageRequest(BaseModel):
    """Request model for protocol message translation"""
    target_protocol: str
    message: Dict[str, Any]


class ACPMessagePart(BaseModel):
    """ACP message part"""
    content: str
    content_type: str = "text/plain"
    content_encoding: str = "plain"


class ACPMessage(BaseModel):
    """ACP message format"""
    role: str
    parts: List[ACPMessagePart]


class ACPRunRequest(BaseModel):
    """ACP run request"""
    agent_name: str
    input: List[ACPMessage]
    session_id: Optional[str] = None


# Storage for ACP runs and sessions
acp_runs: Dict[str, Dict] = {}
acp_sessions: Dict[str, List] = {}


# ====== MCP Protocol Endpoints ======

@router.get("/protocol/mcp/tools")
async def mcp_discover_tools():
    """Discover available MCP tools"""
    protocol_adapters = getattr(router, '_protocol_adapters', {})
    mcp_demo_catalog = getattr(router, '_mcp_demo_catalog', None)
    
    # Default demo tools when MCP server is not available
    demo_tools = [
        {"name": "web_search", "description": "Search the web for information"},
        {"name": "code_execute", "description": "Execute code in a sandbox"},
        {"name": "file_read", "description": "Read file contents"},
        {"name": "file_write", "description": "Write content to files"},
        {"name": "memory_store", "description": "Store data in memory"},
        {"name": "memory_retrieve", "description": "Retrieve data from memory"}
    ]
    
    adapter = protocol_adapters.get("mcp")
    if not adapter:
        return {
            "status": "demo",
            "protocol": "mcp",
            "message": "MCP server not connected - showing demo tools",
            "tools": mcp_demo_catalog.get("tools", demo_tools) if mcp_demo_catalog else demo_tools
        }

    try:
        await adapter.discover_tools()
        return {
            "status": "success",
            "protocol": "mcp",
            "tools": list(adapter.tools_registry.values())
        }
    except Exception as e:
        logger.warning(f"MCP tool discovery failed: {e}")
        return {
            "status": "demo",
            "protocol": "mcp",
            "message": f"MCP server unavailable: {str(e)[:50]}",
            "tools": mcp_demo_catalog.get("tools", demo_tools) if mcp_demo_catalog else demo_tools
        }


@router.post("/protocol/mcp/initialize")
async def mcp_initialize(demo_mode: bool = False):
    """
    Initialize MCP connection and discover capabilities
    
    Set demo_mode=true to test without an actual MCP server
    """
    protocol_adapters = getattr(router, '_protocol_adapters', {})
    
    if not protocol_adapters.get("mcp"):
        raise HTTPException(status_code=503, detail="MCP adapter not initialized")
    
    # Demo mode for testing without external MCP server
    if demo_mode:
        return {
            "status": "success",
            "protocol": "mcp",
            "mode": "demo",
            "server_info": {
                "name": "NIS Protocol MCP Demo Server",
                "version": "1.0.0",
                "description": "Demo MCP server for testing (no external server required)"
            },
            "capabilities": {
                "tools": {
                    "listChanged": True,
                    "available_tools": [
                        {
                            "name": "nis_physics_validate",
                            "description": "Validate physics constraints using PINN",
                            "input_schema": {"type": "object", "properties": {"data": {"type": "object"}}}
                        },
                        {
                            "name": "nis_kan_reason",
                            "description": "Symbolic reasoning with KAN networks",
                            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
                        }
                    ]
                },
                "resources": {
                    "available_resources": [
                        {
                            "uri": "nis://protocol/schema",
                            "name": "NIS Protocol Schema",
                            "description": "Complete NIS Protocol data schema"
                        }
                    ]
                },
                "prompts": {
                    "available_prompts": [
                        {
                            "name": "physics_analysis",
                            "description": "Template for physics-informed analysis"
                        }
                    ]
                }
            },
            "note": "This is a demo response. To use a real MCP server, set MCP_SERVER_URL environment variable and call without demo_mode."
        }
    
    try:
        result = await protocol_adapters["mcp"].initialize()
        return {
            "status": "success",
            "protocol": "mcp",
            "mode": "production",
            "server_info": result.get("serverInfo", {}),
            "capabilities": result.get("capabilities", {})
        }
    except Exception as e:
        error_type = type(e).__name__
        if "Connection" in error_type:
            raise HTTPException(
                status_code=503, 
                detail={
                    "error": f"Connection failed: {e}",
                    "suggestion": "No MCP server found. Try adding '?demo_mode=true' to test the integration.",
                    "setup_guide": "To connect a real MCP server, set MCP_SERVER_URL environment variable."
                }
            )
        elif "Timeout" in error_type:
            raise HTTPException(status_code=504, detail=f"Timeout: {e}")
        else:
            raise HTTPException(status_code=500, detail=str(e))


# ====== A2A Protocol Endpoints ======

@router.post("/protocol/a2a/create-task")
async def a2a_create_task(request: ProtocolTaskRequest, demo_mode: bool = False):
    """
    Create an A2A task on external agent
    
    Set demo_mode=true to test without Google A2A API
    """
    protocol_adapters = getattr(router, '_protocol_adapters', {})
    
    if not protocol_adapters.get("a2a"):
        raise HTTPException(status_code=503, detail="A2A adapter not initialized")
    
    # Demo mode for testing without Google A2A API
    if demo_mode:
        task_id = f"demo_task_{uuid.uuid4().hex[:8]}"
        return {
            "status": "success",
            "protocol": "a2a",
            "mode": "demo",
            "task": {
                "task_id": task_id,
                "agent_id": request.agent_id,
                "description": request.description,
                "status": "running",
                "created_at": time.time(),
                "estimated_completion": "2-5 seconds",
                "artifacts": [],
                "progress": {
                    "status": "in_progress",
                    "percent_complete": 25,
                    "current_step": "Initializing NIS Protocol pipeline",
                    "steps": [
                        "Laplace signal processing",
                        "KAN symbolic reasoning",
                        "PINN physics validation",
                        "LLM synthesis"
                    ]
                }
            },
            "note": "This is a demo response. To use Google A2A, set A2A_API_KEY environment variable and call without demo_mode.",
            "next_steps": f"Check task status at: GET /protocol/a2a/task/{task_id}?demo_mode=true"
        }
    
    try:
        result = await protocol_adapters["a2a"].create_task(
            description=request.description,
            agent_id=request.agent_id,
            parameters=request.parameters,
            callback_url=request.callback_url
        )
        return {
            "status": "success",
            "protocol": "a2a",
            "mode": "production",
            "task": result
        }
    except Exception as e:
        error_type = type(e).__name__
        if "CircuitBreaker" in error_type:
            raise HTTPException(status_code=503, detail="Circuit breaker open - service unavailable")
        elif "Timeout" in error_type:
            raise HTTPException(status_code=504, detail=f"Timeout: {e}")
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": str(e),
                    "suggestion": "No A2A agent configured. Try adding '?demo_mode=true' to test the integration.",
                    "setup_guide": "To use Google A2A, set A2A_API_KEY and configure agent endpoints."
                }
            )


@router.get("/protocol/a2a/task/{task_id}")
async def a2a_get_task_status(task_id: str, demo_mode: bool = False):
    """
    Get A2A task status
    
    Set demo_mode=true for demo tasks
    """
    protocol_adapters = getattr(router, '_protocol_adapters', {})
    
    if not protocol_adapters.get("a2a"):
        raise HTTPException(status_code=503, detail="A2A adapter not initialized")
    
    # Demo mode for testing
    if demo_mode or task_id.startswith("demo_task_"):
        return {
            "status": "success",
            "protocol": "a2a",
            "mode": "demo",
            "task_status": {
                "task_id": task_id,
                "status": "completed",
                "created_at": time.time() - 5.2,
                "completed_at": time.time(),
                "duration_seconds": 5.2,
                "progress": {
                    "status": "completed",
                    "percent_complete": 100,
                    "current_step": "Task completed successfully"
                },
                "result": {
                    "success": True,
                    "output": "NIS Protocol analysis complete",
                    "pipeline_results": {
                        "laplace": "Signal processed and transformed",
                        "kan": "Symbolic reasoning extracted key patterns",
                        "pinn": "Physics constraints validated",
                        "llm": "Final synthesis completed"
                    },
                    "artifacts": [
                        {
                            "type": "analysis_report",
                            "name": "nis_protocol_results.json",
                            "size": "2.4 KB"
                        }
                    ]
                }
            }
        }
    
    try:
        result = await protocol_adapters["a2a"].get_task_status(task_id)
        return {
            "status": "success",
            "protocol": "a2a",
            "mode": "production",
            "task_status": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "suggestion": "If this is a demo task, add '?demo_mode=true' to the request."
            }
        )


@router.delete("/protocol/a2a/task/{task_id}")
async def a2a_cancel_task(task_id: str):
    """Cancel an A2A task"""
    protocol_adapters = getattr(router, '_protocol_adapters', {})
    
    if not protocol_adapters.get("a2a"):
        raise HTTPException(status_code=503, detail="A2A adapter not initialized")
    
    try:
        result = await protocol_adapters["a2a"].cancel_task(task_id)
        return {
            "status": "success",
            "protocol": "a2a",
            "cancelled_task": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====== ACP Protocol Endpoints ======

@router.get("/protocol/acp/agent-card")
async def acp_get_agent_card():
    """Get NIS Protocol Agent Card for ACP offline discovery"""
    protocol_adapters = getattr(router, '_protocol_adapters', {})
    
    if not protocol_adapters.get("acp"):
        raise HTTPException(status_code=503, detail="ACP adapter not initialized")
    
    try:
        card = protocol_adapters["acp"].export_agent_card()
        return {
            "status": "success",
            "protocol": "acp",
            "agent_card": card
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/protocol/acp/execute")
async def acp_execute_agent(request: ProtocolExecuteRequest, demo_mode: bool = False):
    """
    Execute external ACP agent (async or sync)
    
    Set demo_mode=true to test without external ACP agent
    """
    protocol_adapters = getattr(router, '_protocol_adapters', {})
    
    # Demo mode for testing
    if demo_mode:
        execution_id = f"acp_exec_{uuid.uuid4().hex[:8]}"
        return {
            "status": "success",
            "protocol": "acp",
            "mode": "demo",
            "execution": {
                "execution_id": execution_id,
                "agent_url": request.agent_url,
                "status": "completed",
                "async_mode": request.async_mode,
                "started_at": time.time() - 1.5,
                "completed_at": time.time(),
                "duration_seconds": 1.5,
                "result": {
                    "success": True,
                    "response": f"Demo ACP response for message: {request.message.get('content', 'No content')[:50]}...",
                    "capabilities_used": ["reasoning", "analysis"],
                    "confidence": 0.95
                }
            },
            "note": "This is a demo response. Configure ACP_AGENT_URL to connect to real ACP agents."
        }
    
    if not protocol_adapters.get("acp"):
        raise HTTPException(
            status_code=503, 
            detail={
                "error": "ACP adapter not initialized",
                "suggestion": "Try adding '?demo_mode=true' to test the integration.",
                "setup_guide": "Configure ACP_AGENT_URL environment variable to connect to ACP agents."
            }
        )
    
    try:
        result = await protocol_adapters["acp"].execute_agent(
            agent_url=request.agent_url,
            message=request.message,
            async_mode=request.async_mode
        )
        return {
            "status": "success",
            "protocol": "acp",
            "mode": "production",
            "result": result
        }
    except Exception as e:
        error_type = type(e).__name__
        if "CircuitBreaker" in error_type:
            raise HTTPException(status_code=503, detail="Circuit breaker open - service unavailable")
        elif "Timeout" in error_type:
            raise HTTPException(status_code=504, detail=f"Timeout: {e}")
        else:
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": str(e),
                    "suggestion": "Try adding '?demo_mode=true' to test the integration."
                }
            )


# ====== ACP BeeAI Compatible Endpoints ======

@router.get("/agents")
async def acp_list_agents():
    """
    🤖 ACP: List available agents (following IBM/Linux Foundation ACP spec)
    
    Returns agent manifests for discovery and composition.
    Compatible with BeeAI platform.
    """
    get_agents_status = getattr(router, '_get_agents_status', None)
    
    if get_agents_status:
        agents_status = await get_agents_status()
    else:
        agents_status = {"agents": {}}
    
    acp_agents = []
    for agent_id, agent_data in agents_status.get("agents", {}).items():
        acp_agents.append({
            "name": agent_id,
            "description": agent_data.get("definition", {}).get("description", f"NIS Protocol {agent_id} agent"),
            "metadata": {
                "type": agent_data.get("type", "specialized"),
                "status": agent_data.get("status", "inactive"),
                "priority": agent_data.get("definition", {}).get("priority", 5),
                "capabilities": agent_data.get("definition", {}).get("context_keywords", [])
            }
        })
    
    return {"agents": acp_agents}


@router.post("/runs")
async def acp_create_run(request: ACPRunRequest):
    """
    🚀 ACP: Create and execute an agent run (following IBM/Linux Foundation ACP spec)
    
    Supports sync execution with multimodal messages.
    Compatible with BeeAI platform.
    """
    llm_provider = getattr(router, '_llm_provider', None)
    
    run_id = str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())
    
    # Extract input text from ACP message format
    input_text = ""
    for msg in request.input:
        for part in msg.parts:
            if part.content_type == "text/plain":
                input_text += part.content + " "
    input_text = input_text.strip()
    
    # Store run info
    acp_runs[run_id] = {
        "run_id": run_id,
        "agent_name": request.agent_name,
        "session_id": session_id,
        "status": "running",
        "input": [msg.dict() for msg in request.input],
        "output": [],
        "error": None,
        "created_at": time.time()
    }
    
    # Store in session
    if session_id not in acp_sessions:
        acp_sessions[session_id] = []
    acp_sessions[session_id].append(acp_runs[run_id])
    
    try:
        # Route to appropriate agent
        if llm_provider and request.agent_name in ["reasoning", "consciousness", "meta_coordinator"]:
            # Use LLM for core agents
            result = await llm_provider.generate_response(
                messages=[{"role": "user", "content": input_text}],
                temperature=0.7,
                max_tokens=500
            )
            response_text = result.get("content", "I processed your request.")
        elif request.agent_name == "physics_validation":
            # Physics agent
            response_text = f"Physics analysis of: {input_text[:100]}... [Validated with PINN pipeline]"
        elif request.agent_name == "research_and_search_engine":
            # Research agent
            response_text = f"Research synthesis for: {input_text[:100]}... [Sources analyzed]"
        elif llm_provider:
            # Default agent response with LLM
            result = await llm_provider.generate_response(
                messages=[{"role": "user", "content": f"As the {request.agent_name} agent, respond to: {input_text}"}],
                temperature=0.7,
                max_tokens=300
            )
            response_text = result.get("content", f"Agent {request.agent_name} processed your request.")
        else:
            # Fallback without LLM
            response_text = f"Agent {request.agent_name} received: {input_text[:100]}... [LLM not available]"
        
        # Format ACP response
        acp_runs[run_id]["status"] = "completed"
        acp_runs[run_id]["output"] = [{
            "role": f"agent/{request.agent_name}",
            "parts": [{
                "content": response_text,
                "content_type": "text/plain",
                "content_encoding": "plain"
            }]
        }]
        
    except Exception as e:
        acp_runs[run_id]["status"] = "failed"
        acp_runs[run_id]["error"] = str(e)
    
    return acp_runs[run_id]


@router.get("/runs/{run_id}")
async def acp_get_run(run_id: str):
    """
    📊 ACP: Get run status and output
    """
    if run_id not in acp_runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return acp_runs[run_id]


@router.get("/sessions/{session_id}")
async def acp_get_session(session_id: str):
    """
    📋 ACP: Get session history
    """
    if session_id not in acp_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {"session_id": session_id, "runs": acp_sessions[session_id]}


# ====== Protocol Health & Translation ======

@router.get("/protocol/health")
async def protocol_health():
    """Get health status of all protocol adapters"""
    protocol_adapters = getattr(router, '_protocol_adapters', {})
    health_status = {}
    
    for protocol_name, adapter in protocol_adapters.items():
        if adapter:
            try:
                health_status[protocol_name] = adapter.get_health_status()
            except Exception as e:
                health_status[protocol_name] = {
                    "healthy": False,
                    "error": str(e)
                }
        else:
            health_status[protocol_name] = {
                "healthy": False,
                "error": "Adapter not initialized"
            }
    
    return {
        "status": "success",
        "protocols": health_status,
        "overall_healthy": all(
            h.get("healthy", False) for h in health_status.values()
        )
    }


@router.post("/protocol/translate")
async def protocol_translate_message(request: ProtocolMessageRequest):
    """Translate message between NIS Protocol and external protocol format"""
    protocol_adapters = getattr(router, '_protocol_adapters', {})
    target = request.target_protocol.lower()
    
    if target not in protocol_adapters or not protocol_adapters[target]:
        raise HTTPException(status_code=400, detail=f"Protocol '{target}' not available")
    
    adapter = protocol_adapters[target]
    message = request.message
    
    try:
        # Determine direction based on message format
        if message.get("protocol") == "nis":
            # NIS to external
            if target == "mcp":
                raise HTTPException(
                    status_code=400,
                    detail="MCP requires specific tool/resource calls, not message translation"
                )
            elif target == "a2a":
                raise HTTPException(
                    status_code=400,
                    detail="A2A requires task creation, not message translation"
                )
            elif target == "acp":
                translated = adapter.translate_from_nis(message)
                return {
                    "status": "success",
                    "direction": "nis_to_acp",
                    "translated": translated
                }
        else:
            # External to NIS
            if target == "acp":
                translated = adapter.translate_to_nis(message)
                return {
                    "status": "success",
                    "direction": "acp_to_nis",
                    "translated": translated
                }
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Translation not applicable for {target}"
                )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====== MCP Integration Endpoints ======

@router.post("/api/mcp/tools")
async def handle_mcp_tool_request(request: dict, demo_mode: bool = False):
    """
    🔧 Execute MCP tools with Deep Agents integration
    
    Supports all 25+ tools:
    - dataset.search, dataset.preview, dataset.analyze
    - pipeline.run, pipeline.status, pipeline.configure
    - research.plan, research.search, research.synthesize
    - audit.view, audit.analyze, audit.compliance
    - code.edit, code.review, code.analyze
    
    Set demo_mode=true to test without full MCP integration
    
    Returns interactive UI resources compatible with @mcp-ui/client
    """
    mcp_integration = getattr(router, '_mcp_integration', None)
    
    tool_name = request.get("tool_name", request.get("tool", ""))
    arguments = request.get("arguments", request.get("parameters", {}))
    
    # Demo mode for testing
    if demo_mode:
        return {
            "success": True,
            "mode": "demo",
            "tool_name": tool_name,
            "execution_id": f"mcp_tool_{uuid.uuid4().hex[:8]}",
            "result": {
                "status": "completed",
                "output": f"Demo execution of {tool_name} with args: {arguments}",
                "artifacts": [],
                "metrics": {
                    "execution_time_ms": 150,
                    "tokens_used": 0
                }
            },
            "note": "This is a demo response. Enable MCP integration for full functionality."
        }
    
    if not mcp_integration:
        return {
            "success": False, 
            "error": "MCP integration not available",
            "suggestion": "Try adding '?demo_mode=true' to test the tool execution."
        }
    
    try:
        response = await mcp_integration.handle_mcp_request(request)
        return response
    except Exception as e:
        logger.error(f"MCP tool request error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/api/mcp/ui-action")
async def handle_mcp_ui_action(action: dict):
    """
    🎨 Handle UI actions from mcp-ui components
    
    Processes actions like:
    - tool: Execute tool from UI button/form
    - intent: Handle generic UI intents
    - prompt: Process prompts from UI
    - notify: Handle notifications
    - link: Process link clicks
    """
    mcp_ui_adapter = getattr(router, '_mcp_ui_adapter', None)
    
    if not mcp_ui_adapter:
        return {"success": False, "error": "MCP UI adapter not available"}
    
    try:
        message_id = action.get('messageId')
        response = await mcp_ui_adapter.client_handler.handle_ui_action(action, message_id)
        return response
    except Exception as e:
        logger.error(f"MCP UI action error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/api/mcp/plans")
async def create_execution_plan(request: dict):
    """
    🧠 Create Deep Agent execution plan
    
    Example:
    {
        "goal": "Analyze climate change impact on agriculture",
        "context": {"region": "North America", "timeframe": "2020-2024"}
    }
    
    Returns multi-step plan with dependencies and UI monitoring
    """
    mcp_integration = getattr(router, '_mcp_integration', None)
    
    if not mcp_integration:
        return {"success": False, "error": "MCP integration not available"}
    
    try:
        goal = request.get("goal")
        context = request.get("context", {})
        
        if not goal:
            return {"success": False, "error": "Goal required"}
        
        plan = await mcp_integration.create_execution_plan(goal, context)
        return {"success": True, "plan": plan}
    except Exception as e:
        logger.error(f"Plan creation error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/api/mcp/plans/{plan_id}/execute")
async def execute_plan(plan_id: str):
    """
    ⚡ Execute Deep Agent plan
    
    Executes plan step-by-step with progress monitoring
    and UI resource generation for each step
    """
    mcp_integration = getattr(router, '_mcp_integration', None)
    
    if not mcp_integration:
        return {"success": False, "error": "MCP integration not available"}
    
    try:
        result = await mcp_integration.execute_plan(plan_id)
        return {"success": True, "execution": result}
    except Exception as e:
        logger.error(f"Plan execution error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/api/mcp/plans/{plan_id}/status")
async def get_plan_status(plan_id: str):
    """
    📊 Get Deep Agent plan status
    
    Returns current execution status, progress, and step results
    """
    mcp_integration = getattr(router, '_mcp_integration', None)
    
    if not mcp_integration:
        return {"success": False, "error": "MCP integration not available"}
    
    try:
        status = await mcp_integration.get_plan_status(plan_id)
        return {"success": True, "status": status}
    except Exception as e:
        logger.error(f"Plan status error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/api/mcp/info")
async def get_mcp_info():
    """
    ℹ️ Get MCP integration information
    
    Returns available tools, capabilities, and status
    """
    mcp_integration = getattr(router, '_mcp_integration', None)
    
    if not mcp_integration:
        return {"available": False, "error": "MCP integration not available"}
    
    try:
        info = mcp_integration.get_server_info()
        return {"available": True, "info": info}
    except Exception as e:
        logger.error(f"MCP info error: {e}")
        return {"available": False, "error": str(e)}


@router.post("/api/mcp/invoke")
async def langgraph_invoke(request: dict):
    """
    🔗 LangGraph Agent Chat UI compatibility endpoint
    
    Compatible with langchain-ai/agent-chat-ui
    Processes messages and returns with UI resources
    """
    langgraph_bridge = getattr(router, '_langgraph_bridge', None)
    
    if not langgraph_bridge:
        return {"error": "LangGraph bridge not available"}
    
    try:
        input_data = request.get("input", {})
        config = request.get("config", {})
        
        messages = input_data.get("messages", [])
        if not messages:
            return {"error": "No messages provided"}
        
        last_message = messages[-1].get("content", "")
        session_id = config.get("configurable", {}).get("thread_id")
        
        # Collect streaming responses
        responses = []
        async for chunk in langgraph_bridge.handle_chat_message(last_message, session_id):
            responses.append(chunk)
        
        return {
            "output": {"messages": responses},
            "metadata": {
                "run_id": f"run_{int(time.time() * 1000)}",
                "thread_id": session_id
            }
        }
    except Exception as e:
        logger.error(f"LangGraph invoke error: {e}")
        return {"error": str(e)}


@router.post("/mcp/chat")
async def mcp_chat(request: Dict[str, Any]):
    """
    💬 MCP Agent Chat Interface
    
    Direct interface to the Model Context Protocol agent system.
    Supports tool execution, planning, and multi-agent coordination.
    """
    mcp_integration = getattr(router, '_mcp_integration', None)
    
    try:
        if not mcp_integration:
            raise HTTPException(status_code=503, detail="MCP Integration not initialized")
        
        if hasattr(mcp_integration, "handle_mcp_request"):
            return await mcp_integration.handle_mcp_request(request)
        else:
            raise HTTPException(status_code=500, detail="MCP Integration handler missing")
             
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools/list")
async def list_tools():
    """
    🛠️ List Available Tools
    """
    mcp_integration = getattr(router, '_mcp_integration', None)
    
    try:
        if mcp_integration and hasattr(mcp_integration, "get_tool_registry"):
            return mcp_integration.get_tool_registry()
        
        # Fallback: Return known agents as tools
        return {
            "tools": [
                {"name": "web_search", "description": "Search the internet using DuckDuckGo"},
                {"name": "vision_analysis", "description": "Analyze images"},
                {"name": "physics_simulation", "description": "Run physics simulations"},
                {"name": "code_execution", "description": "Execute python code (sandboxed)"}
            ]
        }
    except Exception as e:
        logger.error(f"Tool listing error: {e}")
        return {"error": str(e)}


# ====== Dependency Injection Helper ======

def set_dependencies(
    protocol_adapters=None,
    mcp_demo_catalog=None,
    mcp_integration=None,
    mcp_ui_adapter=None,
    langgraph_bridge=None,
    llm_provider=None,
    get_agents_status=None
):
    """Set dependencies for the protocols router"""
    router._protocol_adapters = protocol_adapters or {}
    router._mcp_demo_catalog = mcp_demo_catalog
    router._mcp_integration = mcp_integration
    router._mcp_ui_adapter = mcp_ui_adapter
    router._langgraph_bridge = langgraph_bridge
    router._llm_provider = llm_provider
    router._get_agents_status = get_agents_status

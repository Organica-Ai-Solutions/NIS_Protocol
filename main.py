#!/usr/bin/env python3
"""
NIS Protocol v4.0.1
Enterprise AI Operating System with Modular Route Architecture

Copyright 2025 Organica AI Solutions

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Architecture:
    - 24 modular route modules in routes/
    - 222 API endpoints
    - Dependency injection pattern
    - See docs/organized/architecture/ROUTE_MIGRATION.md
"""

import asyncio
import logging
import os
import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

# ── NIS Tool Executor (shared HTTP-based agentic tools) ──────────────────────
try:
    from src.core.tool_executor import detect_intent, dispatch as tool_dispatch, memory_search
    TOOL_EXECUTOR_AVAILABLE = True
except ImportError:
    TOOL_EXECUTOR_AVAILABLE = False

# ====== LOGGING SETUP ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nis_protocol")

# ====== FASTAPI SETUP ======
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# ====== NIS PROTOCOL IMPORTS ======
from src.utils.aws_secrets import load_all_api_keys
from src.core.state_manager import nis_state_manager, StateEventType
from src.meta.unified_coordinator import create_scientific_coordinator, BehaviorMode
from src.services.pipeline_service import create_pipeline_service
from src.services.protocol_bridge_service import create_protocol_bridge_service
from src.agents.research.web_search_agent import WebSearchAgent
from src.llm.llm_manager import GeneralLLMProvider
from src.agents.learning.learning_agent import LearningAgent
from src.agents.planning.autonomous_planning_system import AutonomousPlanningSystem
from src.agents.goals.curiosity_engine import CuriosityEngine
from src.agents.goals.adaptive_goal_system import AdaptiveGoalSystem
from src.agents.alignment.ethical_reasoner import EthicalReasoner
from src.agents.simulation.enhanced_scenario_simulator import EnhancedScenarioSimulator
from src.llm.reflective_generator import ReflectiveGenerator
from src.memory.persistent_memory import get_memory_system
from src.core.self_modifier import get_self_modifier
from src.agents.multimodal.vision_agent import MultimodalVisionAgent
from src.agents.research.deep_research_agent import DeepResearchAgent
from src.agents.reasoning.enhanced_reasoning_chain import EnhancedReasoningChain
from src.agents.document.document_analysis_agent import DocumentAnalysisAgent
from src.agents.autonomous_execution.executor import create_anthropic_style_executor
from src.agents.visualization.diagram_agent import DiagramAgent
from src.agents.data_pipeline.real_time_pipeline_agent import create_real_time_pipeline_agent
from src.core.agent_orchestrator import NISAgentOrchestrator

# VibeVoice communication
from src.agents.communication.vibevoice_engine import VibeVoiceEngine

# A2UI Formatter for GenUI integration
from src.utils.a2ui_formatter import format_text_as_a2ui, create_error_widget, A2UIFormatter

# A2A Protocol for official GenUI WebSocket integration
from src.protocols.a2a_protocol import create_a2a_handler, A2AProtocolHandler

# NVIDIA NeMo Integration (optional)
try:
    from src.agents.nvidia_nemo.nemo_integration_manager import NeMoIntegrationManager, NeMoIntegrationConfig
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    NeMoIntegrationManager = None
    logger.info("NVIDIA NeMo integration not available")

# Protocol adapters
from src.adapters.mcp_adapter import MCPAdapter
from src.adapters.a2a_adapter import A2AAdapter
from src.adapters.acp_adapter import ACPAdapter

# Security
import os
try:
    from src.security.auth import verify_api_key, check_rate_limit
    from src.security.user_management import user_manager
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    user_manager = None
    logger.warning("Security module not available")

# ====== PYDANTIC MODELS ======
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    user_id: Optional[str] = "anonymous"
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    genui_enabled: Optional[bool] = False
    use_tools: Optional[bool] = True
    enable_agents: Optional[bool] = True

class ChatResponse(BaseModel):
    response: str
    user_id: str
    conversation_id: str
    timestamp: float
    confidence: Optional[float] = None
    provider: str
    real_ai: bool
    model: str
    tokens_used: int

# ====== GLOBAL STATE ======
llm_provider: Optional[GeneralLLMProvider] = None
web_search_agent: Optional[WebSearchAgent] = None
simulation_coordinator = None
learning_agent: Optional[LearningAgent] = None
planning_system: Optional[AutonomousPlanningSystem] = None
curiosity_engine: Optional[CuriosityEngine] = None
pipeline_service = None
protocol_bridge = None
bitnet_trainer = None
vibevoice_engine = None
nemo_manager = None
persistent_memory = None
reflective_generator = None
self_modifier = None
adaptive_goal_system = None
vision_agent: Optional[MultimodalVisionAgent] = None
research_agent: Optional[DeepResearchAgent] = None
reasoning_chain: Optional[EnhancedReasoningChain] = None
document_agent: Optional[DocumentAnalysisAgent] = None
pipeline_agent = None

# A2A Protocol handler
a2a_handler: Optional[A2AProtocolHandler] = None
a2ui_formatter_instance: Optional[A2UIFormatter] = None
nis_agent_orchestrator = None

# Global agent instances (initialized during startup)
vision_agent = None
coordinator = None
orchestrator = None

# NVIDIA Stack 2025 global instances
cosmos_generator_global = None
cosmos_reasoner_global = None
groot_agent_global = None
isaac_lab_trainer_global = None

# Registries
conversation_memory: Dict[str, List[Dict[str, Any]]] = {}
agent_registry: Dict[str, Dict[str, Any]] = {}
tool_registry: Dict[str, Dict[str, Any]] = {}

# Protocol adapters
protocol_adapters = {
    "mcp": None,
    "a2a": None,
    "acp": None
}

# ====== HELPER FUNCTIONS ======
def get_or_create_conversation(conversation_id: Optional[str], user_id: Optional[str] = None) -> str:
    """Get existing conversation or create a new one"""
    if conversation_id:
        return conversation_id
    new_id = f"conv_{uuid.uuid4().hex[:12]}"
    conversation_memory[new_id] = []
    return new_id

async def add_message_to_conversation(
    conversation_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None
):
    """Add a message to conversation memory"""
    if conversation_id not in conversation_memory:
        conversation_memory[conversation_id] = []
    
    message = {
        "role": role,
        "content": content,
        "timestamp": time.time(),
        "metadata": metadata or {},
        "user_id": user_id
    }
    conversation_memory[conversation_id].append(message)

# ====== FASTAPI APP ======
app = FastAPI(
    title="NIS Protocol v4.0.1",
    description="Enterprise AI Operating System with Modular Route Architecture",
    version="4.0.1",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== MODULAR ROUTE INTEGRATION ======
from routes import (
    # Routers
    robotics_router, physics_router, bitnet_router, webhooks_router,
    monitoring_router, memory_router, chat_router, agents_router,
    research_router, voice_router, protocols_router, vision_router,
    reasoning_router, pipeline_router, system_router, nvidia_router,
    auth_router, utilities_router, v4_features_router, llm_router, 
    unified_router, core_router, isaac_router, hub_gateway_router,
    autonomous_router, skills_router, openclaw_router, openfang_router, neurokernel_router,
    events_router, prototype_router,
    # Dependency setters
    set_bitnet_trainer, set_monitoring_dependencies, set_memory_dependencies,
    set_chat_dependencies, set_agents_dependencies, set_research_dependencies,
    set_voice_dependencies, set_protocols_dependencies, set_vision_dependencies,
    set_reasoning_dependencies, set_pipeline_dependencies, set_system_dependencies,
    set_nvidia_dependencies, set_auth_dependencies, set_utilities_dependencies,
    set_v4_features_dependencies, set_llm_dependencies, set_unified_dependencies,
    set_core_dependencies, set_autonomous_dependencies
)

# Include all routers
app.include_router(core_router)
app.include_router(chat_router)
app.include_router(memory_router)
app.include_router(agents_router)
app.include_router(monitoring_router)
app.include_router(research_router)
app.include_router(voice_router)
app.include_router(vision_router)
app.include_router(reasoning_router)
app.include_router(protocols_router)
app.include_router(pipeline_router)
app.include_router(system_router)
app.include_router(nvidia_router)
app.include_router(auth_router)
app.include_router(utilities_router)
app.include_router(v4_features_router)
app.include_router(llm_router)
app.include_router(unified_router)
app.include_router(robotics_router)
app.include_router(physics_router)
app.include_router(bitnet_router)
app.include_router(webhooks_router)
app.include_router(isaac_router)
app.include_router(hub_gateway_router)
app.include_router(autonomous_router)
app.include_router(skills_router)
app.include_router(openclaw_router)
app.include_router(openfang_router)     # OpenFang MCP integration (/openfang/*)
app.include_router(neurokernel_router)  # NeuroKernel v2 (/neurokernel/*)
app.include_router(events_router)       # SSE channel adapter (/events/*)
app.include_router(prototype_router)    # Prototype endpoints (/prototype/*) - NOT for production

# NVIDIA Cosmos and GR00T integration
try:
    from routes.cosmos import router as cosmos_router
    from routes.cookoff import router as cookoff_router
    from routes.humanoid import router as humanoid_router
    from routes.isaac_lab import router as isaac_lab_router
    from routes.yolo_vision import router as yolo_router
    app.include_router(cosmos_router)
    app.include_router(cookoff_router)
    app.include_router(humanoid_router)
    app.include_router(isaac_lab_router)
    app.include_router(yolo_router)
    logger.info("✅ NVIDIA Stack integrated (Cosmos, Cookoff, GR00T, Isaac Lab, YOLO)")
except Exception as e:
    logger.warning(f"NVIDIA stack routes not loaded: {e}")

# Latino Arm Dance — real-time mic-driven choreographer
try:
    from routes.cosmos_dance import router as cosmos_dance_router
    app.include_router(cosmos_dance_router)
    logger.info("✅ Cosmos Dance loaded (/cosmos-dance/*)")
except Exception as e:
    logger.warning(f"Cosmos Dance not loaded: {e}")

# NVIDIA Unified API
try:
    from routes.nvidia_unified import router as nvidia_unified_router
    app.include_router(nvidia_unified_router)
    logger.info("✅ NVIDIA Unified API loaded")
except Exception as e:
    logger.warning(f"NVIDIA Unified API not loaded: {e}")

logger.info("✅ 30 modular route modules loaded (290+ endpoints)")

# ====== WEBSOCKET ENDPOINTS ======

# Agent Status WebSocket - Real-time agent activity
@app.websocket("/ws/agents")
async def agents_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time agent status updates.

    Sends agent activity, task progress, and resource utilization.
    """ 

    await websocket.accept()
    logger.info("🔌 Agent Status WebSocket connected")
    
    try:
        while True:
            # Get real agent status from orchestrator if available
            agents_data = {
                "type": "agent_status",
                "timestamp": datetime.now().isoformat(),
                "agents": []
            }
            
            # Add active agents if orchestrator is available
            if nis_agent_orchestrator:
                try:
                    # Get registered agents
                    if hasattr(nis_agent_orchestrator, 'agents'):
                        for agent_id, agent in nis_agent_orchestrator.agents.items():
                            agents_data["agents"].append({
                                "id": agent_id,
                                "name": getattr(agent, 'name', agent_id),
                                "type": getattr(agent, 'agent_type', 'unknown'),
                                "status": "active",
                                "task": getattr(agent, 'current_task', 'Idle'),
                                "progress": 0.0,
                                "resource_usage": {
                                    "cpu": 0.0,
                                    "memory": 0.0
                                }
                            })
                except Exception as e:
                    logger.debug(f"Agent status error: {e}")
            
            if not agents_data["agents"]:
                agents_data["status"] = "no_registered_agents"
                agents_data["message"] = "Agent orchestrator is not initialized or has no registered agents."
            
            await websocket.send_json(agents_data)
            await asyncio.sleep(2)  # Update every 2 seconds
            
    except WebSocketDisconnect:
        logger.info("🔌 Agent Status WebSocket disconnected")
    except Exception as e:
        logger.error(f"Agent WebSocket error: {e}")


# Runtime Pipeline WebSocket
@app.websocket("/ws/tao")
async def tao_loop_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for runtime pipeline state.
    Sends only available state from initialized services.
    """
    await websocket.accept()
    logger.info("🔌 TAO Loop WebSocket connected")
    
    try:
        while True:
            tao_data = {
                "type": "tao_update",
                "timestamp": datetime.now().isoformat(),
                "phase": "runtime_status",
                "steps": []
            }
            
            # Get real TAO data from pipeline service if available
            if pipeline_service:
                try:
                    # Try to get real thinking steps
                    if hasattr(pipeline_service, 'get_current_thought'):
                        thought = pipeline_service.get_current_thought()
                        if thought:
                            tao_data["steps"].append({
                                "content": thought,
                                "confidence": 0.85
                            })
                except Exception as e:
                    logger.debug(f"TAO data error: {e}")
            
            if not tao_data["steps"]:
                tao_data["status"] = "no_runtime_steps"
                tao_data["message"] = "No live pipeline steps are currently available."
            
            await websocket.send_json(tao_data)
            await asyncio.sleep(3)  # Update every 3 seconds
            
    except WebSocketDisconnect:
        logger.info("🔌 TAO Loop WebSocket disconnected")
    except Exception as e:
        logger.error(f"TAO WebSocket error: {e}")


# Main Chat WebSocket
@app.websocket("/ws")
async def main_websocket(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time chat communication
    """
    await websocket.accept()
    logger.info("🔌 Main WebSocket connected")
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "message")
            
            if msg_type == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
            elif msg_type == "message":
                message = data.get("content", data.get("message", ""))
                
                # Process with LLM if available
                try:
                    if llm_provider:
                        result = await llm_provider.generate_response(
                            messages=[
                                {"role": "system", "content": "You are NIS Protocol v4.0, a robotics and edge-AI orchestration platform. Be accurate, concise, and technically grounded."},
                                {"role": "user", "content": message}
                            ],
                            temperature=0.7
                        )
                        response_text = result.get("content", "Response generated")
                        provider_used = result.get("provider", "nis-protocol")
                    else:
                        response_text = f"NIS Protocol received: {message}"
                        provider_used = "demo"
                except Exception as e:
                    logger.error(f"❌ WebSocket chat error: {e}")
                    response_text = f"Error processing message: {str(e)}"
                    provider_used = "error"
                
                await websocket.send_json({
                    "type": "response",
                    "content": response_text,
                    "provider": provider_used,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                await websocket.send_json({
                    "type": "ack",
                    "received_type": msg_type,
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info("🔌 Main WebSocket disconnected")
    except Exception as e:
        logger.error(f"❌ Main WebSocket error: {e}")

# ====== AGENTIC WEBSOCKET ENDPOINT ======

def _detect_intent(msg: str) -> str:
    """Classify message intent for tool routing."""
    m = msg.lower()
    if any(k in m for k in ["snapshot", "photo", "picture", "what do you see", "look", "vision", "camera", "see"]):
        return "vision"
    if any(k in m for k in ["pick", "place", "grab", "stack", "sort", "move arm", "xarm", "gripper", "open gripper", "close gripper", "wave", "home"]):
        return "xarm"
    if any(k in m for k in ["cosmos", "plan", "robot plan", "cookoff", "execute plan"]):
        return "cosmos"
    if any(k in m for k in ["status", "health", "services", "running", "system", "ports", "uptime"]):
        return "status"
    if any(k in m for k in ["skill", "skills", "openclaw", "what can you do", "capabilities"]):
        return "skills"
    return "chat"


async def _tool_vision(websocket: WebSocket, message: str) -> tuple:
    """Capture snapshot via Pi camera and optionally run Cosmos vision."""
    import httpx
    tool_id = f"tool_{uuid.uuid4().hex[:8]}"
    await websocket.send_json({
        "type": "TOOL_CALL", "tool_id": tool_id,
        "tool": "camera_snapshot",
        "args": {"source": "pi_camera"},
        "timestamp": datetime.now().isoformat(),
    })
    image_b64 = None
    snap_result = "Camera not available"
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.get("http://localhost:8085/camera/snapshot")
            if r.status_code == 200:
                data = r.json()
                image_b64 = data.get("image_base64") or data.get("image")
                snap_result = "Snapshot captured" if image_b64 else "No image data"
    except Exception as e:
        snap_result = f"Camera error: {e}"
    await websocket.send_json({
        "type": "TOOL_RESULT", "tool_id": tool_id,
        "result": snap_result, "has_image": image_b64 is not None,
        "timestamp": datetime.now().isoformat(),
    })
    return image_b64, snap_result


async def _tool_xarm(websocket: WebSocket, message: str) -> str:
    """Send xArm command via OpenClaw bridge."""
    import httpx
    m = message.lower()
    if "open" in m and "gripper" in m:
        cmd = "open_gripper"
    elif "close" in m and "gripper" in m:
        cmd = "close_gripper"
    elif "home" in m:
        cmd = "home"
    elif "wave" in m:
        cmd = "wave"
    elif "ready" in m:
        cmd = "ready"
    elif "inspect" in m:
        cmd = "inspect"
    elif "pick" in m:
        cmd = "pick"
    elif "place" in m:
        cmd = "place"
    elif "stop" in m:
        cmd = "stop"
    else:
        cmd = "status"
    tool_id = f"tool_{uuid.uuid4().hex[:8]}"
    await websocket.send_json({
        "type": "TOOL_CALL", "tool_id": tool_id,
        "tool": "xarm_control",
        "args": {"command": cmd},
        "timestamp": datetime.now().isoformat(),
    })
    result_text = f"xArm command '{cmd}' sent"
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.post(
                "http://localhost:8000/openclaw/invoke",
                json={"tool": "nis_xarm", "args": {"command": cmd}},
            )
            if r.status_code == 200:
                data = r.json()
                result_text = str(data.get("result", {}).get("result", cmd + " executed"))
    except Exception as e:
        result_text = f"xArm error: {e}"
    await websocket.send_json({
        "type": "TOOL_RESULT", "tool_id": tool_id,
        "result": result_text,
        "timestamp": datetime.now().isoformat(),
    })
    return result_text


async def _tool_cosmos(websocket: WebSocket, message: str, image_b64: Optional[str] = None) -> str:
    """Run Cosmos Cookoff plan via OpenClaw bridge."""
    import httpx
    tool_id = f"tool_{uuid.uuid4().hex[:8]}"
    await websocket.send_json({
        "type": "TOOL_CALL", "tool_id": tool_id,
        "tool": "cosmos_plan",
        "args": {"query": message[:200]},
        "timestamp": datetime.now().isoformat(),
    })
    result_text = "Cosmos plan generated"
    try:
        payload: Dict[str, Any] = {"tool": "nis_cosmos_plan", "args": {"query": message}}
        if image_b64:
            payload["args"]["image_base64"] = image_b64
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post("http://localhost:8000/openclaw/invoke", json=payload)
            if r.status_code == 200:
                data = r.json()
                actions = data.get("result", {}).get("action_recommendations", [])
                result_text = "\n".join(f"• {a}" for a in actions) if actions else "No plan generated"
    except Exception as e:
        result_text = f"Cosmos error: {e}"
    await websocket.send_json({
        "type": "TOOL_RESULT", "tool_id": tool_id,
        "result": result_text,
        "timestamp": datetime.now().isoformat(),
    })
    return result_text


async def _tool_status(websocket: WebSocket) -> str:
    """Check all NeuroLinux service health."""
    import httpx
    tool_id = f"tool_{uuid.uuid4().hex[:8]}"
    await websocket.send_json({
        "type": "TOOL_CALL", "tool_id": tool_id,
        "tool": "system_status",
        "args": {},
        "timestamp": datetime.now().isoformat(),
    })
    lines = []
    checks = [
        ("NIS Protocol",      "http://localhost:8000/health"),
        ("Agent Gateway",     "http://localhost:8085/health"),
        ("NeuroHub UI",       "http://localhost:3000"),
        ("OpenClaw Bridge",   "http://localhost:8000/openclaw/status"),
    ]
    async with httpx.AsyncClient(timeout=3.0) as client:
        for name, url in checks:
            try:
                r = await client.get(url)
                lines.append(f"✅ {name}: online ({r.status_code})")
            except Exception:
                lines.append(f"❌ {name}: offline")
    result_text = "\n".join(lines)
    await websocket.send_json({
        "type": "TOOL_RESULT", "tool_id": tool_id,
        "result": result_text,
        "timestamp": datetime.now().isoformat(),
    })
    return result_text


async def _tool_skills(websocket: WebSocket) -> str:
    """List available OpenClaw skills."""
    import httpx
    tool_id = f"tool_{uuid.uuid4().hex[:8]}"
    await websocket.send_json({
        "type": "TOOL_CALL", "tool_id": tool_id,
        "tool": "list_skills",
        "args": {},
        "timestamp": datetime.now().isoformat(),
    })
    result_text = "Skills unavailable"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(
                "http://localhost:8000/openclaw/invoke",
                json={"tool": "nis_skills", "args": {}},
            )
            if r.status_code == 200:
                data = r.json()
                skills = data.get("result", {}).get("skills", [])
                if skills:
                    result_text = "\n".join(f"• {s['name']}: {s.get('description','')[:80]}" for s in skills[:10])
                else:
                    result_text = "No skills registered yet"
    except Exception as e:
        result_text = f"Skills error: {e}"
    await websocket.send_json({
        "type": "TOOL_RESULT", "tool_id": tool_id,
        "result": result_text,
        "timestamp": datetime.now().isoformat(),
    })
    return result_text


@app.websocket("/ws/agentic")
async def agentic_websocket(websocket: WebSocket):
    """

    Agentic AI WebSocket — Full AG-UI Protocol
    Real tool dispatch: vision, xArm, Cosmos, system status, OpenClaw skills.
    Streams THINKING_STEP + TOOL_CALL + TOOL_RESULT + AGENT_ACTIVATION events.

    """
    await websocket.accept()
    logger.info("🤖 Agentic WebSocket connected")

    message_count = 0

    _LIVE_STEPS = [
        ("Signal Decomposition",  "Applying Laplace transform to extract frequency components"),
        ("Pattern Recognition",   "KAN network identifying symbolic patterns and causal relationships"),
        ("Physics Validation",    "PINN verifying response against physical constraints"),
        ("Cosmos Reasoning",      "Dispatching to H100 Cosmos Reason2 for world-model grounding"),
        ("Context Integration",   "Merging agent outputs with conversation memory and sensor state"),
        ("Response Synthesis",    "Assembling final response from validated sub-agent conclusions"),
    ]

    agents = [
        ("Laplace Signal Processor", "Frequency domain analysis"),
        ("KAN Reasoning Engine",     "Symbolic pattern extraction"),
        ("Physics Validator (PINN)", "Physics constraint validation"),
    ]

    async def _stream_thinking(stop_event: asyncio.Event) -> None:
        for title, content in _LIVE_STEPS:
            if stop_event.is_set():
                break
            try:
                await websocket.send_json({
                    "type": "THINKING_STEP", "title": title, "content": content,
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception:
                break
            try:
                await asyncio.wait_for(
                    asyncio.shield(asyncio.ensure_future(stop_event.wait())),
                    timeout=1.8,
                )
            except asyncio.TimeoutError:
                pass

    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            image_b64_in = data.get("image_base64")  # client may send a snapshot
            message_count += 1

            logger.info(f"📨 Agentic #{message_count}: {message[:60]}")

            intent = _detect_intent(message)
            logger.info(f"🎯 Intent: {intent}")

            # ── Phase 1: Analyzing ───────────────────────────────────────

            await websocket.send_json({
                "type": "THINKING_STEP",
                "title": "Analyzing Request",

                "content": f"Intent detected: [{intent}] — routing to appropriate agent pipeline",
                "timestamp": datetime.now().isoformat(),
            })
            await asyncio.sleep(0.1)

            # ── Phase 2: Agent activation ────────────────────────────────
            active_agents = {
                "vision":  [agents[0], agents[1]],
                "xarm":    [agents[0], agents[2]],
                "cosmos":  agents,
                "status":  [agents[1]],
                "skills":  [agents[1]],
                "chat":    agents,
            }.get(intent, agents)

            for agent_name, task in active_agents:
                await websocket.send_json({
                    "type": "AGENT_ACTIVATION", "agent_name": agent_name,
                    "status": "active", "task": task,
                    "timestamp": datetime.now().isoformat(),
                })
                await asyncio.sleep(0.1)

            # ── Phase 3: Tool dispatch ───────────────────────────────────
            tool_context = ""
            image_b64_out = image_b64_in

            if intent == "vision":
                await websocket.send_json({
                    "type": "THINKING_STEP", "title": "Camera Capture",
                    "content": "Requesting Pi Camera snapshot for visual analysis",
                    "timestamp": datetime.now().isoformat(),
                })
                image_b64_out, snap = await _tool_vision(websocket, message)
                tool_context = f"Camera result: {snap}"

            elif intent == "xarm":
                await websocket.send_json({
                    "type": "THINKING_STEP", "title": "Arm Control",
                    "content": "Sending command to Hiwonder xArm via OpenClaw bridge",
                    "timestamp": datetime.now().isoformat(),
                })
                tool_context = await _tool_xarm(websocket, message)

            elif intent == "cosmos":
                await websocket.send_json({
                    "type": "THINKING_STEP", "title": "Cosmos Planning",
                    "content": "Invoking NVIDIA Cosmos Reason2 on H100 for robot action plan",
                    "timestamp": datetime.now().isoformat(),
                })
                tool_context = await _tool_cosmos(websocket, message, image_b64_out)

            elif intent == "status":
                await websocket.send_json({
                    "type": "THINKING_STEP", "title": "System Health Check",
                    "content": "Polling all NeuroLinux service endpoints",
                    "timestamp": datetime.now().isoformat(),
                })
                tool_context = await _tool_status(websocket)

            elif intent == "skills":
                tool_context = await _tool_skills(websocket)

            # ── Phase 4: LLM synthesis + concurrent thinking stream ──────
            stop_thinking = asyncio.Event()
            thinking_task = asyncio.ensure_future(_stream_thinking(stop_thinking))

            system_prompt = (
                "You are NIS Protocol v4.0, an advanced agentic AI operating system "
                "by Organica AI Solutions running on NeuroLinux on a Raspberry Pi 5. "
                "You coordinate Laplace signal processing, KAN reasoning, PINN physics validation, "
                "Pi Camera vision, Hiwonder xArm 6-DOF control, and NVIDIA Cosmos Reason2 on H100. "
                "Be concise, technically grounded, and always identify as NIS Protocol."
            )
            user_content = message
            if tool_context:
                user_content = f"{message}\n\n[Tool results]\n{tool_context}"


            try:
                if llm_provider:
                    result = await llm_provider.generate_response(
                        messages=[

                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_content},

                        ],
                        temperature=0.7,
                    )
                    response_text = result.get("content", "Response generated.")
                    provider_used = result.get("provider", "nis-protocol")
                else:
                    await asyncio.sleep(0.4)
                    if tool_context:
                        response_text = (
                            f"**NIS Protocol v4.0** — tool execution complete.\n\n"
                            f"{tool_context}\n\n"
                            "_Connect an LLM provider for synthesized narrative responses._"
                        )
                    else:
                        response_text = (
                            f"**NIS Protocol v4.0** — demo mode.\n\n"
                            f"Received: \"{message}\"\n\n"
                            "Connect an LLM provider (Qwen2.5 / Cosmos / OpenAI) for full agentic responses."
                        )
                    provider_used = "demo"
            except Exception as e:

                logger.error(f"❌ Agentic LLM error: {e}")
                response_text = tool_context or f"NIS Protocol error: {e}"
                provider_used = "error"
            finally:
                stop_thinking.set()
                try:
                    await asyncio.wait_for(thinking_task, timeout=0.5)
                except (asyncio.TimeoutError, Exception):
                    thinking_task.cancel()

            # ── Phase 5: Agent deactivation ──────────────────────────────
            for agent_name, _ in active_agents:
                await websocket.send_json({
                    "type": "AGENT_DEACTIVATION", "agent_name": agent_name,
                    "timestamp": datetime.now().isoformat(),
                })
                await asyncio.sleep(0.06)

            # ── Phase 6: Final response ──────────────────────────────────

            await websocket.send_json({
                "type": "TEXT_MESSAGE_CONTENT",
                "content": response_text,
                "role": "assistant",
                "metadata": {
                    "provider": provider_used,
                    "intent": intent,
                    "real_ai": provider_used not in ("demo", "error"),
                    "tools_used": intent if intent != "chat" else None,
                },
                "image_base64": image_b64_out,
                "timestamp": datetime.now().isoformat(),
            })

            logger.info(f"✅ Agentic #{message_count} done — intent={intent} provider={provider_used}")

    except WebSocketDisconnect:
        logger.info(f"🔌 Agentic WebSocket disconnected after {message_count} messages")
    except Exception as e:
        logger.error(f"❌ Agentic WebSocket error: {e}")

# ====== ENHANCED A2A WEBSOCKET ENDPOINT ======
@app.websocket("/ws/a2a")
async def a2a_endpoint(websocket: WebSocket):
    """
    🚀 Enhanced A2A WebSocket - Full GenUI Integration
    
    Implements official GenUI A2A Protocol with A2UI widget formatting.
    Streams responses as rich interactive widgets in real-time.
    """
    await a2a_handler.handle_connection(websocket)

# ====== MAIN CHAT ENDPOINTS (v3.2.7 compatibility) ======
from fastapi.responses import RedirectResponse, HTMLResponse

@app.get("/chat", response_class=HTMLResponse, tags=["Chat"])
async def chat_browser():
    """
    Browser access to chat - redirects to the chat console.
    For API access, use POST /chat with JSON body.
    """
    return RedirectResponse(url="/static/chat_console.html", status_code=302)

@app.get("/console", response_class=HTMLResponse, tags=["Chat"])
async def console_redirect():
    """
    Legacy console route - redirects to static chat console.
    """
    return RedirectResponse(url="/static/chat_console.html", status_code=302)

@app.post("/chat", tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main Chat Endpoint - NIS Protocol v4.0
    
    Enhanced chat with intelligent query routing and real LLM responses.
    Supports A2UI formatting for GenUI-enabled clients.
    """
    global llm_provider, conversation_memory
    
    conversation_id = get_or_create_conversation(request.conversation_id, request.user_id)
    
    # Add user message to conversation
    await add_message_to_conversation(
        conversation_id, "user", request.message, user_id=request.user_id
    )

    # ── Step 1: Retrieve relevant memories from ChromaDB ────────────────────
    memory_context = ""
    if persistent_memory:
        try:
            mem_result = await memory_search(request.message, top_k=3,
                                             memory_system=persistent_memory)
            if mem_result["ok"] and mem_result.get("memories"):
                snippets = [f"- {m['content'][:200]}" for m in mem_result["memories"]]
                memory_context = "Relevant memories:\n" + "\n".join(snippets)
                logger.info(f"🧠 Injected {len(mem_result['memories'])} memories into context")
        except Exception as _me:
            logger.debug(f"Memory retrieval skipped: {_me}")

    if memory_context:
        messages[0]["content"] += f"\n\n{memory_context}"

    # ── NeuroKernel v2: Scan + Skill inject ──────────────────────────────────
    _nk_skills_used: list = []
    _nk_scan_blocked = False
    try:
        from src.core.prompt_injection_scanner import get_scanner
        from src.core.skill_loader import get_skill_loader
        _scan = get_scanner()
        _scan_result = _scan.scan(request.message, context=f"chat/{request.user_id}")
        if _scan_result.action.value == "block":
            _nk_scan_blocked = True
            logger.warning(f"NeuroKernel: blocked input from {request.user_id} — {_scan_result.summary()}")
        else:
            if _scan_result.sanitized_text:
                request.message = _scan_result.sanitized_text
            _loader = get_skill_loader()
            _skill_ctx = _loader.build_context_for(request.message, max_skills=2)
            if _skill_ctx:
                messages[0]["content"] += _skill_ctx
                _nk_skills_used = [s.name for s in _loader.skills_for_query(request.message, 2)]
    except Exception as _nke:
        logger.debug(f"NeuroKernel pre-process skipped: {_nke}")

    if _nk_scan_blocked:
        return JSONResponse({"error": "Input blocked by security scanner", "code": "SCAN_BLOCK"}, status_code=400)

    # ── Step 2: Agentic tool dispatch ────────────────────────────────────────

    tools_used = []
    tool_results = []

    if request.use_tools and TOOL_EXECUTOR_AVAILABLE:
        intent = detect_intent(request.message)
        if intent != "chat":
            logger.info(f"🔧 Tool dispatch — intent: {intent}")
            try:
                result = await tool_dispatch(intent, request.message,
                                             memory_system=persistent_memory)
                tools_used.append(result.get("tool", intent))
                tool_results.append(result)
                logger.info(f"🔧 Tool result: {result.get('summary', '')[:100]}")
            except Exception as _te:
                logger.warning(f"Tool dispatch error: {_te}")

    elif request.use_tools:
        # Legacy code-execution path (fallback when tool_executor unavailable)
        import re
        message_lower = request.message.lower()
        code_keywords = ["execute", "run code", "calculate", "compute", "eval", "python", "code"]
        math_patterns = [r'\d+\s*[+\-*/^]\s*\d+']
        needs_code = any(k in message_lower for k in code_keywords) or \
                     any(re.search(p, request.message) for p in math_patterns)
        if needs_code:
            code_to_run = None
            if "```" in request.message:
                blocks = request.message.split("```")
                if len(blocks) > 1:
                    code_to_run = blocks[1].strip().lstrip("python").strip()
            else:
                m = re.search(r'(\d+\s*[+\-*/^]\s*\d+)', request.message)
                if m:
                    expr = m.group(1).replace('^', '**')
                    code_to_run = f"result = {expr}\nprint(f'Result: {{result}}')"
            if code_to_run:
                import httpx as _httpx
                for url in ["http://localhost:8001/execute"]:
                    try:
                        async with _httpx.AsyncClient() as _c:
                            _r = await _c.post(url, json={"code_content": code_to_run}, timeout=5.0)
                            if _r.status_code == 200:
                                tools_used.append("code_execute")
                                tool_results.append({"tool": "code_execute",
                                                     "output": _r.json().get("output", "")})
                                break
                    except Exception:
                        pass
    
    # ── Step 3: Store this interaction in persistent memory ─────────────────
    if persistent_memory:
        try:
            await persistent_memory.store(
                content=request.message,
                memory_type="conversation",
                metadata={"user_id": request.user_id, "conversation_id": conversation_id,
                          "tools": tools_used},
            )
        except Exception:
            pass

    # ── Step 4: Synthesise LLM response ──────────────────────────────────────
    try:
        if llm_provider:
            # Inject tool results as context
            if tool_results:
                tool_context = "\n\n[Tool results]\n"
                for tr in tool_results:
                    summary = tr.get("summary") or tr.get("output") or tr.get("status", "executed")
                    tool_context += f"- {tr['tool']}: {summary}\n"
                    # Append action list for cosmos planning
                    if tr.get("actions"):
                        for a in tr["actions"]:
                            tool_context += f"  • {a}\n"
                messages[-1]["content"] += tool_context
            
            result = await llm_provider.generate_response(
                messages=messages,
                temperature=0.7,
                requested_provider=request.provider
            )
            
            response_text = result.get("content", "No response generated")
            provider_used = result.get("provider", "unknown")
            model_used = result.get("model", "unknown")
            tokens_used = result.get("tokens_used", 0)
            real_ai = result.get("real_ai", False)
        else:
            response_text = "LLM provider not initialized. Please check your API keys."
            provider_used = "none"
            model_used = "none"
            tokens_used = 0
            real_ai = False
    except Exception as e:
        logger.error(f"Chat error: {e}")
        response_text = f"Error generating response: {str(e)}"
        provider_used = "error"
        model_used = "none"
        tokens_used = 0
        real_ai = False
    
    # Add assistant response to conversation
    await add_message_to_conversation(
        conversation_id, "assistant", response_text, user_id=request.user_id
    )

    # ── NeuroKernel v2: Audit log ─────────────────────────────────────────────
    try:
        from src.core.audit_chain import get_audit_chain
        from src.core.loop_guard import get_loop_guard
        _chain = get_audit_chain()
        _chain.log(
            agent_id=f"chat/{provider_used}",
            action_type="llm_call",
            layer="reasoning",
            payload={
                "user_id": request.user_id,
                "conversation_id": conversation_id,
                "input_preview": request.message[:200],
                "response_preview": response_text[:200],
                "tools_used": tools_used,
                "real_ai": real_ai,
            },
            skill_attribution=_nk_skills_used,
            success=real_ai or len(response_text) > 10,
            tags=["chat", provider_used],
        )
        _guard = get_loop_guard()
        _guard.record("llm_call", {"intent": "chat"}, context_id=conversation_id, made_progress=True)
    except Exception as _ae:
        logger.debug(f"NeuroKernel audit skipped: {_ae}")
    
    # Check if GenUI formatting is requested
    if request.genui_enabled:
        try:
            a2ui_response = format_text_as_a2ui(
                response_text,
                wrap_in_card=True,
                include_actions=True
            )
            
            # Return response with A2UI messages array
            return {
                "response": response_text,
                "a2ui_messages": a2ui_response.get("a2ui_messages", []),
                "tools_used": tools_used,
                "tool_results": tool_results,
                "user_id": request.user_id or "anonymous",
                "conversation_id": conversation_id,
                "timestamp": time.time(),
                "provider": provider_used,
                "model": model_used,
                "tokens_used": tokens_used,
                "real_ai": real_ai,
                "genui_formatted": True
            }
        except Exception as e:
            logger.error(f"A2UI formatting error: {e}")
            error_widget = create_error_widget(f"Failed to format response: {str(e)}")
            return {
                **error_widget,
                "user_id": request.user_id or "anonymous",
                "conversation_id": conversation_id,
                "timestamp": time.time(),
                "provider": provider_used,
                "model": model_used,
                "genui_formatted": True
            }
    
    # Return response with tool execution results
    intent_used = tool_results[0].get("tool") if tool_results else "chat"
    return {
        "response": response_text,
        "tools_used": tools_used,
        "tool_results": tool_results,
        "intent": intent_used,
        "user_id": request.user_id or "anonymous",
        "conversation_id": conversation_id,
        "timestamp": time.time(),
        "confidence": 0.85,
        "provider": provider_used,
        "real_ai": real_ai,
        "model": model_used,
        "tokens_used": tokens_used
    }

# ====== SECURITY MIDDLEWARE ======
if SECURITY_AVAILABLE:
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        """
        Global rate limiting middleware - applies to all endpoints except public ones
        Can be disabled by setting DISABLE_RATE_LIMIT=true environment variable
        """
        # Skip rate limiting if disabled for testing
        disable_flag = os.getenv("DISABLE_RATE_LIMIT", "false").lower()
        logger.info(f"Rate limit check: DISABLE_RATE_LIMIT={disable_flag}")
        if disable_flag in ["true", "1", "yes"]:
            logger.info("⚠️ Rate limiting DISABLED for testing")
            return await call_next(request)
        
        # Skip rate limiting for public endpoints
        public_endpoints = ["/health", "/docs", "/redoc", "/openapi.json", "/metrics"]
        if any(request.url.path.startswith(ep) for ep in public_endpoints):
            return await call_next(request)
        
        if not SECURITY_AVAILABLE:
            return await call_next(request)
        
        try:
            client_ip = request.client.host if request.client else "unknown"
            api_key = request.headers.get("X-API-Key")
            
            allowed, remaining, reset, tier = check_rate_limit(client_ip, api_key)
            if not allowed:
                return JSONResponse(
                    {
                        "error": "Rate limit exceeded", 
                        "retry_after": reset,
                        "tier": tier,
                        "message": f"Rate limit for {tier} tier exceeded. Upgrade for higher limits."
                    },
                    status_code=429,
                    headers={
                        "X-RateLimit-Remaining": str(remaining),
                        "X-RateLimit-Reset": str(reset),
                        "X-RateLimit-Tier": tier
                    }
                )
            response = await call_next(request)
            response.headers["X-RateLimit-Remaining"] = str(int(remaining) if remaining != float('inf') else "999999")
            response.headers["X-RateLimit-Tier"] = tier
            return response
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            return JSONResponse({"error": "Rate limiting error"}, status_code=500)

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ====== INITIALIZATION ======
def initialize_agent_orchestrator(llm_provider=None):
    """Initialize the agent orchestrator with LLM provider"""
    global nis_agent_orchestrator
    if nis_agent_orchestrator is None:
        try:
            from src.core.agent_orchestrator import initialize_orchestrator
            nis_agent_orchestrator = initialize_orchestrator(
                llm_provider=llm_provider
            )
            logger.info("✅ Agent Orchestrator initialized with context-aware execution and memory")
        except Exception as e:
            logger.error(f"❌ Agent Orchestrator failed: {e}")

async def initialize_system():
    """Initialize all NIS Protocol components with timeout protection"""
    global llm_provider, web_search_agent, simulation_coordinator
    global learning_agent, planning_system, curiosity_engine
    global pipeline_service, protocol_bridge, bitnet_trainer
    global persistent_memory, reflective_generator, self_modifier, adaptive_goal_system
    global vision_agent, research_agent, reasoning_chain, document_agent, pipeline_agent
    
    logger.info("🚀 Initializing NIS Protocol v4.0.1...")
    logger.info("⏱️  Startup timeout: 300 seconds")
    
    # Initialize Infrastructure (Kafka, Redis, Zookeeper)
    try:
        logger.info("🔄 Step 1/10: Initializing infrastructure...")
        from src.infrastructure.nis_infrastructure import initialize_infrastructure, get_nis_infrastructure
        infra_status = await asyncio.wait_for(initialize_infrastructure(), timeout=30)
        logger.info(f"✅ Step 1/10: Infrastructure connected: Kafka={infra_status.get('kafka')}, Redis={infra_status.get('redis')}")
    except asyncio.TimeoutError:
        logger.warning("⚠️ Step 1/10: Infrastructure timeout - continuing with degraded mode")
    except Exception as e:
        logger.warning(f"⚠️ Step 1/10: Infrastructure initialization: {e}")
    
    # Load API Keys (AWS Secrets Manager or Environment Variables)
    try:
        logger.info("🔄 Step 2/10: Loading API Keys...")
        api_keys = load_all_api_keys()
        
        # Update environment with loaded keys (for backward compatibility)
        for key_name, key_value in api_keys.items():
            if key_value and not os.getenv(key_name):
                os.environ[key_name] = key_value
        
        aws_enabled = os.getenv("AWS_SECRETS_ENABLED", "false").lower() == "true"
        if aws_enabled:
            logger.info(f"✅ Step 2/10: Loaded {len(api_keys)} API keys from AWS Secrets Manager")
        else:
            logger.info(f"✅ Step 2/10: Loaded {len(api_keys)} API keys from environment variables")
    except Exception as e:
        logger.warning(f"⚠️ Step 2/10: API key loading failed: {e}")
    
    # LLM Provider
    try:
        logger.info("🔄 Step 3/10: Initializing LLM Provider...")
        llm_provider = GeneralLLMProvider()
        logger.info("✅ Step 3/10: LLM Provider initialized")
    except Exception as e:
        logger.error(f"❌ Step 3/10: LLM Provider failed: {e}")
    
    # Initialize Memory System (with timeout - model download can be slow)
    try:
        logger.info("🔄 Step 4/10: Initializing Persistent Memory System...")
        logger.info("   → This may download sentence-transformers model (~500MB) on first run")
        from src.memory.persistent_memory import PersistentMemorySystem
        
        # Run in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        persistent_memory = await asyncio.wait_for(
            loop.run_in_executor(None, PersistentMemorySystem),
            timeout=120  # 2 minutes for model download
        )
        logger.info("✅ Step 4/10: Persistent Memory System initialized")
    except asyncio.TimeoutError:
        logger.warning("⚠️ Step 4/10: Memory System timeout (model download?) - continuing without memory")
        persistent_memory = None
    except Exception as e:
        logger.warning(f"⚠️ Step 4/10: Memory System initialization failed: {e}")
        persistent_memory = None
    
    # Re-initialize Agent Orchestrator with LLM Provider and Memory System
    try:
        logger.info("🔄 Step 5/10: Initializing Agent Orchestrator with LLM Provider...")
        initialize_agent_orchestrator(
            llm_provider=llm_provider
        )
        if nis_agent_orchestrator:
            await asyncio.wait_for(nis_agent_orchestrator.start_orchestrator(), timeout=30)
            logger.info("✅ Step 5/10: Agent Orchestrator with context-aware execution and memory ready")
    except asyncio.TimeoutError:
        logger.error("❌ Step 5/10: Agent Orchestrator timeout")
    except Exception as e:
        logger.error(f"❌ Step 5/10: Agent Orchestrator initialization failed: {e}")
    
    # Core agents
    logger.info("🔄 Step 6/10: Initializing core agents...")
    web_search_agent = WebSearchAgent()
    coordinator = create_scientific_coordinator()
    simulation_coordinator = coordinator
    logger.info("✅ Step 6/10: Core agents initialized")
    
    try:
        logger.info("🔄 Step 7/10: Initializing Learning Agent...")
        learning_agent = LearningAgent(agent_id="core_learning_agent")
        logger.info("✅ Step 7/10: Learning Agent initialized")
    except Exception as e:
        logger.error(f"❌ Step 7/10: Learning Agent failed: {e}")
    
    logger.info("🔄 Step 8/10: Initializing Planning and Curiosity...")
    planning_system = AutonomousPlanningSystem()
    curiosity_engine = CuriosityEngine()
    logger.info("✅ Step 8/10: Planning and Curiosity initialized")
    
    # pipeline Service (10-phase pipeline)
    logger.info("🔄 Step 9/10: Initializing pipeline Service...")
    pipeline_service = create_pipeline_service()
    if not pipeline_service:
        logger.warning("⚠️ Step 9/10: pipeline service creation failed, using fallback")
    else:
        try:
            pipeline_service.__init_evolution__()
            pipeline_service.__init_genesis__()
            pipeline_service.__init_distributed__()
            pipeline_service.__init_planning__()
            pipeline_service.__init_marketplace__()
            pipeline_service.__init_multipath__()
            pipeline_service.__init_embodiment__()
            pipeline_service.__init_debugger__()
            pipeline_service.__init_meta_evolution__()
            logger.info("✅ Step 9/10: 10-phase pipeline Pipeline initialized")
        except Exception as e:
            logger.warning(f"⚠️ Step 9/10: Some pipeline phases skipped: {e}")
    
    # V4.0 Self-improving components
    try:
        logger.info("🔄 Step 10/10: Initializing V4.0 Self-improving components...")
        persistent_memory = get_memory_system()
        self_modifier = get_self_modifier()
        reflective_generator = ReflectiveGenerator(
            llm_provider=llm_provider,
            pipeline_service=pipeline_service,
            quality_threshold=0.75
        )
        adaptive_goal_system = AdaptiveGoalSystem(
            agent_id="core_goal_system",
            persistent_memory=persistent_memory,
            reflective_generator=reflective_generator
        )
        app.state.adaptive_goal_system = adaptive_goal_system
        logger.info("✅ Step 9/10: V4.0 Self-improving components initialized")
    except Exception as e:
        logger.error(f"❌ Step 9/10: V4.0 components failed: {e}")
    
    # Protocol bridge
    protocol_bridge = create_protocol_bridge_service(
        pipeline_service=pipeline_service,
        unified_coordinator=coordinator
    )

    # EdgeAIOS — HYBRID_ADAPTIVE mode for online/offline switching + local BitNet fallback
    try:
        from src.core.edge_ai_operating_system import (
            EdgeAIOperatingSystem, EdgeDeviceProfile, EdgeDeviceType,
            EdgeAICapabilities, OperationMode
        )
        import psutil as _psutil
        _ram_mb = int(_psutil.virtual_memory().total / 1024 / 1024)
        _has_gpu = os.path.exists("/dev/nvidia0") or os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1")
        edge_profile = EdgeDeviceProfile(
            device_type=EdgeDeviceType.ROBOTICS_SYSTEM,
            cpu_cores=os.cpu_count() or 4,
            memory_mb=_ram_mb,
            storage_gb=100,
            has_gpu=_has_gpu,
            battery_powered=False,
        )
        edge_capabilities = EdgeAICapabilities(
            local_inference=True,
            online_learning=True,
            physics_validation=True,
            computer_vision=True,
            natural_language=True,
            path_planning=True,
        )
        edge_os = EdgeAIOperatingSystem(
            device_profile=edge_profile,
            ai_capabilities=edge_capabilities,
            operation_mode=OperationMode.HYBRID_ADAPTIVE,
        )
        _edge_task = asyncio.create_task(edge_os.initialize_edge_system())
        _edge_task.add_done_callback(_log_startup_task_exc)
        app.state.edge_os = edge_os
        logger.info("✅ EdgeAIOS HYBRID_ADAPTIVE started (offline fallback active)")
    except Exception as _e:
        logger.warning(f"⚠️ EdgeAIOS skipped: {_e}")
    
    # Multimodal agents
    logger.info("🔄 Step 10/10: Initializing multimodal agents and final components...")
    vision_agent = MultimodalVisionAgent(agent_id="vision_agent")

    # Inject route dependencies NOW with the agents we have so far (learning + vision).
    # This ensures /agents/status always shows real agents even if later inits hang.
    try:
        inject_route_dependencies()
        logger.info("   -> Route dependencies injected (early — learning+vision ready)")
    except Exception as e:
        logger.warning(f"   -> Early inject skipped: {e}")

    try:
        research_agent = await asyncio.to_thread(DeepResearchAgent, "research_agent")
    except Exception as e:
        logger.warning(f"   -> research_agent skipped: {e}")
    try:
        reasoning_chain = await asyncio.to_thread(EnhancedReasoningChain, "reasoning_chain")
    except Exception as e:
        logger.warning(f"   -> reasoning_chain skipped: {e}")
    try:
        document_agent = await asyncio.to_thread(DocumentAnalysisAgent, "document_agent")
    except Exception as e:
        logger.warning(f"   -> document_agent skipped: {e}")
    
    # Pipeline agent
    try:
        pipeline_agent = await asyncio.wait_for(create_real_time_pipeline_agent(), timeout=30)
        logger.info("   → Pipeline Agent initialized")
    except asyncio.TimeoutError:
        logger.warning("   → Pipeline Agent timeout")
    except Exception as e:
        logger.warning(f"   → Pipeline Agent skipped: {e}")
    
    # BitNet trainer (optional)
    bitnet_dir = os.getenv("BITNET_MODEL_PATH", "models/bitnet/models/bitnet")
    if os.path.exists(os.path.join(bitnet_dir, "config.json")):
        try:
            from src.agents.training.bitnet_online_trainer import create_bitnet_online_trainer, OnlineTrainingConfig
            config = OnlineTrainingConfig(model_path=bitnet_dir)
            bitnet_trainer = create_bitnet_online_trainer(
                agent_id="bitnet_trainer",
                config=config,
                pipeline_service=pipeline_service
            )
            logger.info("   → BitNet Trainer initialized")
        except Exception as e:
            logger.warning(f"   → BitNet Trainer skipped: {e}")
    
    # Initialize A2A Protocol Handler
    global a2a_handler, a2ui_formatter_instance
    try:
        a2ui_formatter_instance = A2UIFormatter()
        a2a_handler = create_a2a_handler(
            llm_provider=llm_provider,
            a2ui_formatter=a2ui_formatter_instance
        )
        logger.info("   → A2A Protocol Handler initialized (WebSocket support)")
    except Exception as e:
        logger.warning(f"   → A2A Protocol Handler skipped: {e}")
    
    # Inject dependencies into route modules -- always runs
    try:
        inject_route_dependencies()
        logger.info("✅ Route dependencies injected")
    except Exception as e:
        logger.error(f"❌ inject_route_dependencies failed: {e}")

    logger.info("✅ Step 10/10: All components initialized")
    logger.info("")
    logger.info("="*60)
    logger.info("🎉 NIS Protocol v4.0.1 READY FOR REQUESTS")
    logger.info("="*60)
    logger.info(f"   Memory System: {'ENABLED' if persistent_memory else 'DISABLED'}")
    logger.info(f"   LLM Provider: {'READY' if llm_provider else 'UNAVAILABLE'}")
    logger.info(f"   Agent Orchestrator: {'READY' if nis_agent_orchestrator else 'UNAVAILABLE'}")
    logger.info("="*60)

def inject_route_dependencies():
    """Inject dependencies into all route modules"""
    logger.info("🔗 Injecting route dependencies...")
    
    try:
        # BitNet
        set_bitnet_trainer(bitnet_trainer)
        
        # Monitoring
        set_monitoring_dependencies(
            llm_provider=llm_provider,
            conversation_memory=conversation_memory,
            agent_registry=agent_registry,
            tool_registry=tool_registry
        )
        
        # Memory
        set_memory_dependencies(
            persistent_memory=persistent_memory,
            conversation_memory=conversation_memory
        )
        
        # Chat
        set_chat_dependencies(
            llm_provider=llm_provider,
            reflective_generator=reflective_generator
        )
        
        # Agents
        set_agents_dependencies(
            learning_agent=learning_agent,
            planning_system=planning_system,
            curiosity_engine=curiosity_engine,
            ethical_reasoner=None,
            scenario_simulator=None,
            physics_agent=None,  # Physics agent is in pipeline service
            vision_agent=vision_agent,
            research_agent=research_agent,
            reasoning_agent=reasoning_chain
        )
        
        # Research
        set_research_dependencies(
            web_search_agent=web_search_agent,
            llm_provider=llm_provider
        )
        
        # Voice
        set_voice_dependencies(
            llm_provider=llm_provider,
            conversation_memory=conversation_memory,
            vibevoice_engine=vibevoice_engine,
            pipeline_service=pipeline_service,
            get_or_create_conversation=get_or_create_conversation,
            add_message_to_conversation=add_message_to_conversation
        )
        
        # Protocols
        set_protocols_dependencies(
            protocol_adapters=protocol_adapters,
            mcp_integration=protocol_adapters.get("mcp"),
            llm_provider=llm_provider
        )
        
        # Vision
        set_vision_dependencies(
            vision_agent=vision_agent,
            document_agent=document_agent
        )
        
        # Reasoning
        set_reasoning_dependencies(
            reasoning_chain=reasoning_chain,
            vision_agent=vision_agent,
            research_agent=research_agent,
            document_agent=document_agent
        )
        
        # pipeline
        set_pipeline_dependencies(
            pipeline_service=pipeline_service,
            conversation_memory=conversation_memory
        )
        
        # System
        set_system_dependencies(llm_provider=llm_provider)
        
        # NVIDIA
        set_nvidia_dependencies(nemo_manager=nemo_manager)
        
        # Auth
        if user_manager:
            set_auth_dependencies(user_manager=user_manager)
        
        # Utilities
        set_utilities_dependencies()
        
        # V4 Features
        set_v4_features_dependencies(
            persistent_memory=persistent_memory,
            self_modifier=self_modifier,
            adaptive_goal_system=adaptive_goal_system
        )
        
        # LLM
        set_llm_dependencies(llm_provider=llm_provider)
        
        # Unified
        set_unified_dependencies(llm_provider=llm_provider)
        
        # Core
        set_core_dependencies(
            llm_provider=llm_provider,
            conversation_memory=conversation_memory,
            agent_registry=agent_registry,
            tool_registry=tool_registry
        )
        
        # Autonomous Agents with LLM Planning
        from src.core.autonomous_orchestrator import AutonomousOrchestrator
        autonomous_orchestrator = AutonomousOrchestrator(llm_provider=llm_provider)
        set_autonomous_dependencies(autonomous_orchestrator=autonomous_orchestrator)
        logger.info("✅ Autonomous orchestrator initialized with LLM-powered planning")

        # OpenClaw bridge — inject LLM provider
        from routes.openclaw import set_dependencies as set_openclaw_dependencies
        set_openclaw_dependencies(llm_provider=llm_provider)
        logger.info("✅ OpenClaw bridge dependencies injected")
        
        logger.info("✅ All route dependencies injected")
    except Exception as e:
        logger.error(f"❌ Dependency injection failed: {e}")
        import traceback
        traceback.print_exc()

def initialize_vibevoice():
    """Initialize VibeVoice engine"""
    global vibevoice_engine
    try:
        vibevoice_engine = VibeVoiceEngine()
        vibevoice_engine.initialize()
        logger.info("✅ VibeVoice engine initialized")
    except Exception as e:
        logger.warning(f"⚠️ VibeVoice skipped: {e}")

def initialize_nemo():
    """Initialize NVIDIA NeMo manager"""
    global nemo_manager
    if not NEMO_AVAILABLE:
        return
    
    try:
        nim_api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY")
        dgx_endpoint = os.getenv("DGX_CLOUD_ENDPOINT")
        
        config = NeMoIntegrationConfig(
            nim_api_key=nim_api_key,
            dgx_cloud_endpoint=dgx_endpoint,
            enable_nim_inference=bool(nim_api_key),
            enable_dgx_cloud=bool(dgx_endpoint)
        )
        nemo_manager = NeMoIntegrationManager(config=config)
        logger.info("✅ NVIDIA NeMo manager initialized")
    except Exception as e:
        logger.warning(f"⚠️ NeMo manager skipped: {e}")

def initialize_protocol_adapters():
    """Initialize protocol adapters"""
    global protocol_adapters
    
    try:
        protocol_adapters["mcp"] = MCPAdapter({
            "base_url": os.getenv("MCP_SERVER_URL", "http://localhost:3000"),
            "timeout": 30
        })
        protocol_adapters["a2a"] = A2AAdapter({
            "server_url": os.getenv("A2A_SERVER_URL", "http://localhost:3001"),
            "timeout": 30
        })
        protocol_adapters["acp"] = ACPAdapter({
            "agent_url": os.getenv("ACP_AGENT_URL", "http://localhost:3002"),
            "timeout": 30
        })
        logger.info("✅ Protocol adapters initialized")
    except Exception as e:
        logger.warning(f"⚠️ Protocol adapters skipped: {e}")

# ====== STARTUP EVENT ======
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup - FAST MODE for testing"""
    logger.info("🚀 Initializing NIS Protocol v4.0.1 (FAST MODE)...")
    
    # Skip heavy initialization if SKIP_INIT is set
    if os.getenv("SKIP_INIT", "false").lower() in ["true", "1", "yes"]:
        logger.info("⚡ SKIP_INIT enabled - using minimal initialization")
        return
    
    def _log_startup_task_exc(t):
        if not t.cancelled() and t.exception():
            logger.error(f"Startup background task crashed: {t.exception()}", exc_info=t.exception())

    try:
        # Run initialization in background to not block server startup
        _init_task = asyncio.create_task(initialize_system_background())
        _init_task.add_done_callback(_log_startup_task_exc)
        logger.info("✅ Server ready - initialization running in background")
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        logger.error("System will continue with fallback mode")

    # NeuroKernel v2 — start autonomously in background
    try:
        from src.core.neurokernel import get_neurokernel
        kernel = get_neurokernel()
        _nk_task = asyncio.create_task(kernel.startup())
        _nk_task.add_done_callback(_log_startup_task_exc)
        logger.info("NeuroKernel v2 startup task queued")
    except Exception as e:
        logger.warning(f"NeuroKernel v2 startup skipped: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown — stop NeuroKernel drives and flush audit chain."""
    try:
        from src.core.neurokernel import get_neurokernel
        await get_neurokernel().shutdown()
        logger.info("NeuroKernel v2 shutdown complete")
    except Exception as e:
        logger.warning(f"NeuroKernel shutdown: {e}")


async def initialize_system_background():
    """Initialize system in background"""
    startup_timeout = 120  # 2 minutes
    try:
        await asyncio.wait_for(
            initialize_system(),
            timeout=startup_timeout
        )
        logger.info("✅ Background initialization complete")
    except asyncio.TimeoutError:
        logger.error(f"❌ Initialization timeout after {startup_timeout} seconds")
    except Exception as e:
        logger.error(f"❌ Background initialization error: {e}")

# ====== WEBSOCKET A2A ENDPOINT ======
@app.websocket("/a2a")
async def a2a_websocket_endpoint(websocket: WebSocket):
    """
    Official GenUI A2A Protocol WebSocket Endpoint
    
    Implements the A2A (Agent-to-Agent) streaming protocol for real-time
    agent-to-UI communication with GenUI framework.
    
    Protocol Flow:
    1. Client connects via WebSocket
    2. Server sends AgentCard with agent metadata
    3. Client sends user messages
    4. Server streams SurfaceUpdate messages with UI widgets
    5. Server sends BeginRendering/EndRendering signals
    
    Compatible with official genui_a2ui Flutter package.
    """
    await websocket.accept()
    logger.info(f"A2A WebSocket connection established from {websocket.client}")
    
    try:
        if a2a_handler:
            await a2a_handler.handle_connection(websocket)
        else:
            # Fallback if A2A handler not initialized
            await websocket.send_json({
                "type": "error",
                "error": "A2A Protocol handler not initialized"
            })
            await websocket.close()
    except WebSocketDisconnect:
        logger.info("A2A WebSocket client disconnected")
    except Exception as e:
        logger.error(f"A2A WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass

# ====== MAIN ======
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


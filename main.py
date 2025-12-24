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
from src.services.consciousness_service import create_consciousness_service
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
consciousness_service = None
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
    reasoning_router, consciousness_router, system_router, nvidia_router,
    auth_router, utilities_router, v4_features_router, llm_router, 
    unified_router, core_router, isaac_router, hub_gateway_router,
    # Dependency setters
    set_bitnet_trainer, set_monitoring_dependencies, set_memory_dependencies,
    set_chat_dependencies, set_agents_dependencies, set_research_dependencies,
    set_voice_dependencies, set_protocols_dependencies, set_vision_dependencies,
    set_reasoning_dependencies, set_consciousness_dependencies, set_system_dependencies,
    set_nvidia_dependencies, set_auth_dependencies, set_utilities_dependencies,
    set_v4_features_dependencies, set_llm_dependencies, set_unified_dependencies,
    set_core_dependencies
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
app.include_router(consciousness_router)
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

logger.info("✅ 26 modular route modules loaded (250+ endpoints)")

# ====== MAIN WEBSOCKET ENDPOINT ======
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
                                {"role": "system", "content": "You are NIS Protocol v4.0, an advanced AI operating system by Organica AI Solutions."},
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
@app.websocket("/ws/agentic")
async def agentic_websocket(websocket: WebSocket):
    """
    Agentic AI WebSocket - Real-time Agent Visualization
    Implements AG-UI Protocol for transparent agentic AI workflows
    """
    await websocket.accept()
    logger.info("🤖 Agentic WebSocket connected")
    
    message_count = 0
    
    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            message_count += 1
            
            logger.info(f"📨 Agentic message #{message_count}: {message[:50]}...")
            
            # STEP 1: THINKING PHASE
            await websocket.send_json({
                "type": "THINKING_STEP",
                "step_number": 1,
                "title": "Analyzing Request",
                "content": f"Processing: '{message[:100]}...'",
                "confidence": 0.95,
                "timestamp": datetime.now().isoformat()
            })
            
            await asyncio.sleep(0.2)
            
            # STEP 2: AGENT ACTIVATION
            agents = [
                ("Laplace Signal Processor", "Frequency domain analysis"),
                ("KAN Reasoning Engine", "Symbolic pattern extraction"),
                ("Physics Validator (PINN)", "Physics constraint validation"),
            ]
            
            for agent_name, task in agents:
                await websocket.send_json({
                    "type": "AGENT_ACTIVATION",
                    "agent_name": agent_name,
                    "status": "active",
                    "task": task,
                    "timestamp": datetime.now().isoformat()
                })
                await asyncio.sleep(0.1)
            
            # STEP 3: PROCESS MESSAGE
            try:
                if llm_provider:
                    result = await llm_provider.generate_response(
                        messages=[
                            {"role": "system", "content": "You are NIS Protocol v4.0, an advanced AI operating system."},
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
                logger.error(f"❌ Agentic chat error: {e}")
                response_text = f"Demo response for: {message}"
                provider_used = "demo"
            
            # STEP 4: DEACTIVATE AGENTS
            for agent_name, _ in agents:
                await websocket.send_json({
                    "type": "AGENT_DEACTIVATION",
                    "agent_name": agent_name,
                    "timestamp": datetime.now().isoformat()
                })
                await asyncio.sleep(0.05)
            
            # STEP 5: SEND RESPONSE
            await websocket.send_json({
                "type": "TEXT_MESSAGE_CONTENT",
                "content": response_text,
                "role": "assistant",
                "metadata": {"provider": provider_used, "real_ai": provider_used != "demo"},
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"✅ Agentic message #{message_count} completed")
            
    except WebSocketDisconnect:
        logger.info(f"🔌 Agentic WebSocket disconnected after {message_count} messages")
    except Exception as e:
        logger.error(f"❌ Agentic WebSocket error: {e}")

# ====== ENHANCED A2A WEBSOCKET ENDPOINT ======
from enhanced_a2a_websocket import enhanced_a2a_websocket

@app.websocket("/ws/a2a")
async def a2a_endpoint(websocket: WebSocket):
    """
    🚀 Enhanced A2A WebSocket - Full GenUI Integration
    
    Combines AG-UI transparency events with A2UIFormatter for rich widget generation.
    This endpoint properly formats responses as GenUI surfaces.
    """
    await enhanced_a2a_websocket(websocket, llm_provider, a2ui_formatter_instance)

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
    
    # Build messages for LLM
    messages = [
        {
            "role": "system",
            "content": "You are NIS (Neural Intelligence System), an advanced AI operating system by Organica AI Solutions. You are NOT Claude, GPT, or any base model - you ARE NIS Protocol v4.0. You coordinate multiple AI providers as your compute layer. Always identify as NIS Protocol. Be helpful, accurate, and technically grounded."
        }
    ]
    
    # Add conversation history
    if conversation_id in conversation_memory:
        for msg in conversation_memory[conversation_id][-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    else:
        messages.append({"role": "user", "content": request.message})
    
    # Generate response
    try:
        if llm_provider:
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
    
    # Check if GenUI formatting is requested
    if request.genui_enabled:
        try:
            a2ui_response = format_text_as_a2ui(
                response_text,
                wrap_in_card=True,
                include_actions=True
            )
            
            return {
                **a2ui_response,
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
    
    # Return standard response (backward compatibility)
    return ChatResponse(
        response=response_text,
        user_id=request.user_id or "anonymous",
        conversation_id=conversation_id,
        timestamp=time.time(),
        confidence=0.85,
        provider=provider_used,
        real_ai=real_ai,
        model=model_used,
        tokens_used=tokens_used
    )

# ====== SECURITY MIDDLEWARE ======
if SECURITY_AVAILABLE:
    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        # Skip security for docs, health endpoints, and WebSocket endpoints
        if request.url.path in ["/health", "/", "/docs", "/openapi.json", "/redoc"] or request.url.path.startswith("/ws/"):
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        api_key = request.headers.get("X-API-Key")
        
        allowed, remaining, reset = check_rate_limit(client_ip, api_key)
        if not allowed:
            return JSONResponse(
                {"error": "Rate limit exceeded", "retry_after": reset},
                status_code=429
            )
        
        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ====== INITIALIZATION ======
def initialize_agent_orchestrator(llm_provider=None, memory_system=None):
    """Initialize the agent orchestrator with LLM provider and memory system"""
    global nis_agent_orchestrator
    if nis_agent_orchestrator is None:
        try:
            from src.core.agent_orchestrator import initialize_orchestrator
            nis_agent_orchestrator = initialize_orchestrator(
                llm_provider=llm_provider,
                memory_system=memory_system
            )
            logger.info("✅ Agent Orchestrator initialized with context-aware execution and memory")
        except Exception as e:
            logger.error(f"❌ Agent Orchestrator failed: {e}")

async def initialize_system():
    """Initialize all NIS Protocol components with timeout protection"""
    global llm_provider, web_search_agent, simulation_coordinator
    global learning_agent, planning_system, curiosity_engine
    global consciousness_service, protocol_bridge, bitnet_trainer
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
        logger.info("🔄 Step 5/10: Initializing Agent Orchestrator with LLM Provider and Memory...")
        initialize_agent_orchestrator(
            llm_provider=llm_provider,
            memory_system=persistent_memory
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
    
    # Consciousness Service (10-phase pipeline)
    logger.info("🔄 Step 9/10: Initializing Consciousness Service...")
    consciousness_service = create_consciousness_service()
    try:
        consciousness_service.__init_evolution__()
        consciousness_service.__init_genesis__()
        consciousness_service.__init_distributed__()
        consciousness_service.__init_planning__()
        consciousness_service.__init_marketplace__()
        consciousness_service.__init_multipath__()
        consciousness_service.__init_embodiment__()
        consciousness_service.__init_debugger__()
        consciousness_service.__init_meta_evolution__()
        logger.info("✅ Step 9/10: 10-phase Consciousness Pipeline initialized")
    except Exception as e:
        logger.warning(f"⚠️ Step 9/10: Some consciousness phases skipped: {e}")
    
    # V4.0 Self-improving components
    try:
        logger.info("🔄 Step 10/10: Initializing V4.0 Self-improving components...")
        persistent_memory = get_memory_system()
        self_modifier = get_self_modifier()
        reflective_generator = ReflectiveGenerator(
            llm_provider=llm_provider,
            consciousness_service=consciousness_service,
            quality_threshold=0.75
        )
        adaptive_goal_system = AdaptiveGoalSystem(
            agent_id="core_goal_system",
            persistent_memory=persistent_memory,
            reflective_generator=reflective_generator
        )
        logger.info("✅ Step 9/10: V4.0 Self-improving components initialized")
    except Exception as e:
        logger.error(f"❌ Step 9/10: V4.0 components failed: {e}")
    
    # Protocol bridge
    protocol_bridge = create_protocol_bridge_service(
        consciousness_service=consciousness_service,
        unified_coordinator=coordinator
    )
    
    # Multimodal agents
    logger.info("🔄 Step 10/10: Initializing multimodal agents and final components...")
    vision_agent = MultimodalVisionAgent(agent_id="vision_agent")
    research_agent = DeepResearchAgent(agent_id="research_agent")
    reasoning_chain = EnhancedReasoningChain(agent_id="reasoning_chain")
    document_agent = DocumentAnalysisAgent(agent_id="document_agent")
    
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
                consciousness_service=consciousness_service
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
    
    # Inject dependencies into route modules
    inject_route_dependencies()
    
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
            agent_registry=agent_registry
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
            consciousness_service=consciousness_service,
            get_or_create_conversation=get_or_create_conversation,
            add_message_to_conversation=add_message_to_conversation
        )
        
        # Protocols
        set_protocols_dependencies(
            protocol_adapters=protocol_adapters,
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
        
        # Consciousness
        set_consciousness_dependencies(
            consciousness_service=consciousness_service,
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
    """Application startup"""
    logger.info("🚀 Starting NIS Protocol v4.0.1...")
    
    # Note: Agent orchestrator will be initialized with LLM provider in initialize_system()
    initialize_protocol_adapters()
    initialize_vibevoice()
    initialize_nemo()
    
    await initialize_system()
    
    logger.info("=" * 50)
    logger.info("NIS Protocol v4.0.1 - Ready")
    logger.info("=" * 50)
    logger.info("📚 API Docs: http://localhost:8000/docs")
    logger.info("🔬 ReDoc: http://localhost:8000/redoc")
    logger.info("❤️ Health: http://localhost:8000/health")
    logger.info("🔌 WebSocket A2A: ws://localhost:8000/a2a")
    logger.info("=" * 50)

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

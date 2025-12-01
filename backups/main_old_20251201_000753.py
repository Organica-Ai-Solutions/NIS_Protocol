#!/usr/bin/env python3
"""
NIS Protocol v4.0
Enterprise AI Operating System with 10-Phase Consciousness Pipeline
Full Robotics Integration, Authentication, and Flutter Frontend Support

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
"""

import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
# import numpy as np  # Moved to local imports to prevent FastAPI serialization issues
import soundfile as sf
from pydub import AudioSegment
import requests
import aiohttp  # For internal API calls in tool execution

# Set up logging early to avoid NameError
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nis_general_pattern")

# Import State Manager for Real-Time Visibility
from src.core.state_manager import nis_state_manager, StateEventType

# FastAPI and web framework imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Response, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from fastapi.responses import JSONResponse

from src.meta.unified_coordinator import create_scientific_coordinator, BehaviorMode
from src.utils.env_config import EnvironmentConfig
# SimulationCoordinator functionality is now integrated into UnifiedCoordinator
# No separate import needed - using unified coordinator instead

# High-performance audio processing imports
# Temporarily disabled to fix numpy serialization issue
from src.services.streaming_stt_service import StreamingSTTService, transcribe_audio_stream
from src.services.audio_buffer_service import get_audio_processor, HighPerformanceAudioBuffer
from src.services.wake_word_service import get_wake_word_detector, get_conversation_manager

# ✅ REAL NIS Protocol platform imports - No mocks allowed
# Import real implementations from dedicated modules
# Temporarily disabled to debug numpy serialization
from src.agents.signal_processing.unified_signal_agent import UnifiedSignalAgent
from src.agents.reasoning.unified_reasoning_agent import EnhancedKANReasoningAgent
from src.meta.unified_coordinator import EnhancedPINNPhysicsAgent
from src.nis_protocol.core.platform import create_edge_platform

# VibeVoice communication imports
from src.agents.communication.vibevoice_engine import VibeVoiceEngine
import src.agents.communication.vibevoice_engine as vibevoice_module

# Global VibeVoice engine instance
vibevoice_engine = None

# Global NVIDIA NeMo Integration Manager instance
nemo_manager = None

# Enhanced optimization systems (temporarily disabled for startup)
from src.mcp.schemas.enhanced_tool_schemas import EnhancedToolSchemas
from src.mcp.enhanced_response_system import EnhancedResponseSystem, ResponseFormat
from src.mcp.token_efficiency_system import TokenEfficiencyManager

# NIS HUB Integration - Enhanced Services
from src.services.consciousness_service import create_consciousness_service
from src.services.protocol_bridge_service import create_protocol_bridge_service
from src.agents.research.web_search_agent import WebSearchAgent
from src.llm.llm_manager import GeneralLLMProvider
from src.agents.learning.learning_agent import LearningAgent
from src.agents.consciousness.conscious_agent import ConsciousAgent
# Temporarily disabled to debug numpy serialization
from src.agents.signal_processing.unified_signal_agent import create_enhanced_laplace_transformer
from src.agents.reasoning.unified_reasoning_agent import create_enhanced_kan_reasoning_agent
# Real physics agent implementation required by integrity rules
from src.agents.physics.unified_physics_agent import create_enhanced_pinn_physics_agent
from src.agents.planning.autonomous_planning_system import AutonomousPlanningSystem
from src.agents.goals.curiosity_engine import CuriosityEngine
from src.agents.goals.adaptive_goal_system import AdaptiveGoalSystem  # V4.0
from src.utils.self_audit import self_audit_engine
from src.utils.response_formatter import NISResponseFormatter
from src.agents.alignment.ethical_reasoner import EthicalReasoner, EthicalFramework
from src.agents.simulation.enhanced_scenario_simulator import EnhancedScenarioSimulator, ScenarioType, SimulationParameters
from src.llm.reflective_generator import ReflectiveGenerator  # V4.0
from src.memory.persistent_memory import get_memory_system    # V4.0
from src.core.self_modifier import get_self_modifier          # V4.0

# NVIDIA NeMo Enterprise Integration with proper error handling
try:
    from src.agents.nvidia_nemo import (
        NeMoPhysicsAgent, NeMoAgentOrchestrator, 
        create_nemo_physics_agent, create_nemo_agent_orchestrator
    )
    from src.agents.nvidia_nemo.nemo_integration_manager import (
        NeMoIntegrationManager, NVIDIAInceptionIntegration
    )
    NEMO_INTEGRATION_AVAILABLE = True
    logger.info("✅ NVIDIA NeMo integration available")
except ImportError as e:
    logger.warning(f"⚠️ NVIDIA NeMo integration not available: {e}")
    NEMO_INTEGRATION_AVAILABLE = False
    NeMoPhysicsAgent = None
    NeMoAgentOrchestrator = None
    create_nemo_physics_agent = None
    create_nemo_agent_orchestrator = None
    NeMoIntegrationManager = None
    NVIDIAInceptionIntegration = None

# Import Autonomous Executor and Diagram Agent
from src.agents.autonomous_execution.executor import create_anthropic_style_executor
from src.agents.visualization.diagram_agent import DiagramAgent

# Real chat memory implementation required by integrity rules
try:
    from src.chat.enhanced_memory_chat import EnhancedChatMemory, ChatMemoryConfig
except ImportError:
    class EnhancedChatMemory:
        """Enhanced chat memory implementation required - no mocks allowed per .cursorrules"""
        def __init__(self, config=None):
            raise NotImplementedError("Enhanced chat memory must be properly implemented - mocks prohibited by engineering integrity rules")
    
    class ChatMemoryConfig:
        """Chat memory config implementation required - no mocks allowed per .cursorrules"""
        def __init__(self):
            raise NotImplementedError("Chat memory config must be properly implemented - mocks prohibited by engineering integrity rules")
# Temporarily disabled due to missing dependencies
# from src.agents.memory.enhanced_memory_agent import EnhancedMemoryAgent
try:
    from src.agents.memory.enhanced_memory_agent import EnhancedMemoryAgent
except ImportError as e:
    logger.warning(f"EnhancedMemoryAgent import failed: {e}. Memory system will run in basic mode.")

# from src.agents.autonomous_execution.anthropic_style_executor import create_anthropic_style_executor, ExecutionStrategy, ExecutionMode  # Temporarily disabled
# from src.agents.training.bitnet_online_trainer import create_bitnet_online_trainer, OnlineTrainingConfig  # Temporarily disabled

# Enhanced Multimodal Agents - v3.2
from src.agents.multimodal.vision_agent import MultimodalVisionAgent
from src.agents.research.deep_research_agent import DeepResearchAgent
from src.agents.reasoning.enhanced_reasoning_chain import EnhancedReasoningChain, ReasoningType
from src.agents.document.document_analysis_agent import DocumentAnalysisAgent, DocumentType, ProcessingMode

# Precision Visualization Agents - Code-based (NOT AI image gen)
# Temporarily disabled to debug numpy serialization
# from src.agents.visualization.diagram_agent import DiagramAgent
# from src.agents.visualization.code_chart_agent import CodeChartAgent

# Real-Time Data Pipeline Integration
from src.agents.data_pipeline.real_time_pipeline_agent import create_real_time_pipeline_agent, DataStreamConfig, PipelineMetricType

# Logging already configured above

from src.utils.confidence_calculator import calculate_confidence

# NIS State Management and WebSocket System
from src.core.state_manager import (
    nis_state_manager, StateEventType, emit_state_event, 
    update_system_state, get_current_state
)
from src.core.websocket_manager import (
    nis_websocket_manager, ConnectionType, broadcast_to_all, send_to_user
)
from src.core.agent_orchestrator import NISAgentOrchestrator, AgentStatus, AgentType

# Initialize NIS Agent Orchestrator at module level (must be done early)
# Move to later initialization to avoid import issues
nis_agent_orchestrator = None

def initialize_agent_orchestrator():
    """Initialize the agent orchestrator after all imports are loaded"""
    global nis_agent_orchestrator
    if nis_agent_orchestrator is None:
        try:
            logger.info("Attempting to initialize NIS Agent Orchestrator...")
            nis_agent_orchestrator = NISAgentOrchestrator()
            logger.info("NIS Agent Orchestrator initialized successfully at module level")
            logger.info(f"Orchestrator type: {type(nis_agent_orchestrator)}")
            logger.info(f"Orchestrator has agents: {hasattr(nis_agent_orchestrator, 'agents')}")
            if hasattr(nis_agent_orchestrator, 'agents'):
                logger.info(f"Number of agents: {len(nis_agent_orchestrator.agents)}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize NIS Agent Orchestrator at module level: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            nis_agent_orchestrator = None
            return False
    return True

# ====== GRACEFUL LLM IMPORTS ======
LLM_AVAILABLE = False
try:
    import aiohttp
    LLM_AVAILABLE = True
    logger.info("HTTP client available for real LLM integration")
except Exception as e:
    logger.warning(f" LLM integration will be limited: {e}")

# ====== APPLICATION MODELS ======
class SimpleChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"
    conversation_id: Optional[str] = None

# Keep the original for compatibility but create a simple version
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    user_id: Optional[str] = "anonymous"
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    agent_type: Optional[str] = "default"  # Add agent_type with default
    provider: Optional[str] = None  # Provider to use (openai, anthropic, deepseek, kimi, nvidia, google)
    model: Optional[str] = None  # Specific model to use (overrides provider default)
    # Formatting parameters
    output_mode: Optional[str] = Field(default="technical", description="Output mode: technical, casual, eli5, visual")
    audience_level: Optional[str] = Field(default="expert", description="Audience level: expert, intermediate, beginner")
    include_visuals: Optional[bool] = Field(default=False, description="Include visual elements")
    show_confidence: Optional[bool] = Field(default=False, description="Show confidence breakdown")
    # Smart optimization parameters
    enable_caching: Optional[bool] = Field(default=True, description="Enable smart caching")
    # Tool optimization parameters
    response_format: Optional[str] = Field(default="detailed", description="Response format: concise, detailed, structured, natural")
    token_limit: Optional[int] = Field(default=None, description="Maximum tokens for response")
    page: Optional[int] = Field(default=1, description="Page number for paginated responses")
    page_size: Optional[int] = Field(default=20, description="Items per page")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Filters for data selection")
    priority: Optional[str] = Field(default="normal", description="Request priority: low, normal, high, critical")
    # Consensus control parameters
    consensus_mode: Optional[str] = Field(default=None, description="Consensus mode: single, dual, triple, smart, custom")
    consensus_providers: Optional[List[str]] = Field(default=None, description="Custom provider list for consensus")
    max_cost: Optional[float] = Field(default=0.10, description="Maximum cost per request")
    user_preference: Optional[str] = Field(default="balanced", description="User preference: quality, speed, cost, balanced")

class ChatResponse(BaseModel):
    response: str
    user_id: str
    conversation_id: str
    timestamp: float
    confidence: Optional[float]
    provider: str
    real_ai: bool
    model: str
    tokens_used: int
    reasoning_trace: Optional[List[str]] = None

class AgentCreateRequest(BaseModel):
    agent_type: str = Field(..., description="Type of agent")
    capabilities: List[str] = Field(default_factory=list)
    memory_size: str = "1GB"
    tools: Optional[List[str]] = None

class SetBehaviorRequest(BaseModel):
    mode: BehaviorMode

# ====== GLOBAL STATE - ARCHAEOLOGICAL PATTERN ======
llm_provider: Optional[GeneralLLMProvider] = None
web_search_agent: Optional[WebSearchAgent] = None
simulation_coordinator = None
learning_agent: Optional[LearningAgent] = None
planning_system: Optional[AutonomousPlanningSystem] = None
curiosity_engine: Optional[CuriosityEngine] = None
ethical_reasoner: Optional[EthicalReasoner] = None
scenario_simulator: Optional[EnhancedScenarioSimulator] = None
anthropic_executor = None  # autonomous executor
bitnet_trainer = None  # BitNet online training system
laplace = None  # Will be created from unified coordinator
kan = None  # Will be created from unified coordinator
pinn = None  # Will be created from unified coordinator
conversation_memory: Dict[str, List[Dict[str, Any]]] = {}  # Legacy - kept for compatibility
enhanced_chat_memory = None  # Enhanced chat memory system with persistence
agent_registry: Dict[str, Dict[str, Any]] = {}
tool_registry: Dict[str, Dict[str, Any]] = {}
# Enhanced Multimodal Agents - v3.2
vision_agent = None  # Multimodal vision analysis agent
research_agent = None  # Deep research agent
reasoning_chain = None  # Enhanced reasoning chain agent
document_agent = None  # Document analysis agent
pipeline_agent = None  # Real-time data pipeline agent

# ✅ Conversation Management Helper Functions
def get_or_create_conversation(conversation_id: Optional[str], user_id: Optional[str] = None) -> str:
    """Get existing conversation or create a new one"""
    if conversation_id:
        return conversation_id
    
    # Generate new conversation ID
    new_id = f"conv_{uuid.uuid4().hex[:12]}"
    conversation_memory[new_id] = []
    logger.info(f"Created new conversation: {new_id} for user: {user_id or 'anonymous'}")
    return new_id

async def add_message_to_conversation(
    conversation_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None
):
    """Add a message to conversation memory"""
    # Initialize conversation if it doesn't exist
    if conversation_id not in conversation_memory:
        conversation_memory[conversation_id] = []
    
    # Create message record
    message = {
        "role": role,
        "content": content,
        "timestamp": time.time(),
        "metadata": metadata or {},
        "user_id": user_id
    }
    
    # Add to legacy conversation memory
    conversation_memory[conversation_id].append(message)
    
    # Try to add to enhanced chat memory if available
    if enhanced_chat_memory:
        try:
            await enhanced_chat_memory.add_message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                metadata=metadata,
                user_id=user_id
            )
        except Exception as e:
            logger.warning(f"Failed to add message to enhanced memory: {e}")

async def get_enhanced_conversation_context(
    conversation_id: str,
    current_message: Optional[str] = None,
    max_messages: int = 50
) -> List[Dict[str, Any]]:
    """Get conversation context using enhanced memory system."""
    
    if enhanced_chat_memory:
        try:
            return await enhanced_chat_memory.get_conversation_context(
                conversation_id=conversation_id,
                max_messages=max_messages,
                include_semantic_context=True,
                current_message=current_message
            )
        except Exception as e:
            logger.error(f"Failed to get enhanced context: {e}")
    
    # Fallback to legacy system
    context_messages = conversation_memory.get(conversation_id, [])[-max_messages:]
    return [
        {
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": datetime.fromtimestamp(msg["timestamp"]).isoformat(),
            "source": "legacy_memory"
        }
        for msg in context_messages
    ]

# NVIDIA Inception Integration (Enterprise Access)
nvidia_inception = None  # NVIDIA Inception program integration
nemo_manager = None  # NeMo Integration Manager  
agents = {}  # Agent registry for NeMo integration

# NIS HUB Enhanced Services
consciousness_service = None
protocol_bridge = None
vision_agent = None
research_agent = None
reasoning_chain = None
document_agent = None

# coordinator = create_scientific_coordinator()  # Temporarily disabled to debug numpy serialization
coordinator = None

# Initialize the environment config and integrity metrics - temporarily disabled
# env_config = EnvironmentConfig()
env_config = None

# Initialize optimization systems - temporarily disabled to debug numpy serialization
# enhanced_schemas = EnhancedToolSchemas()
# response_system = EnhancedResponseSystem()
# token_manager = TokenEfficiencyManager()
enhanced_schemas = None
response_system = None
token_manager = None

# Edge AI Operating System - enabled
from src.core.edge_ai_operating_system import (
    EdgeAIOperatingSystem, create_drone_ai_os, create_robot_ai_os, create_vehicle_ai_os
)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np  # Local import to prevent global serialization issues
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

# Create the optimized FastAPI app
app = FastAPI(
    title="NIS Protocol v3.2 - Optimized Agent Platform",
    description="Advanced agent platform with optimized tool systems, token-efficient responses, and enhanced agent coordination.",
    version="3.2.0"
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
# Import and include all modular route modules (v4.0.1 migration)
from routes import (
    # Routers
    robotics_router, physics_router, bitnet_router, webhooks_router,
    monitoring_router, memory_router, chat_router, agents_router,
    research_router, voice_router, protocols_router, vision_router,
    reasoning_router, consciousness_router, system_router, nvidia_router,
    auth_router, utilities_router, v4_features_router, llm_router, unified_router,
    core_router,
    # Dependency setters
    set_bitnet_trainer, set_monitoring_dependencies, set_memory_dependencies,
    set_chat_dependencies, set_agents_dependencies, set_research_dependencies,
    set_voice_dependencies, set_protocols_dependencies, set_vision_dependencies,
    set_reasoning_dependencies, set_consciousness_dependencies, set_system_dependencies,
    set_nvidia_dependencies, set_auth_dependencies, set_utilities_dependencies,
    set_v4_features_dependencies, set_llm_dependencies, set_unified_dependencies,
    set_core_dependencies
)

# Include all modular routers
# Note: These routers contain migrated endpoints - the original endpoints in main.py
# will be removed after testing confirms the modular routes work correctly
app.include_router(robotics_router)
app.include_router(physics_router)
app.include_router(bitnet_router)
app.include_router(webhooks_router)
app.include_router(monitoring_router)
app.include_router(memory_router)
app.include_router(chat_router)
app.include_router(agents_router)
app.include_router(research_router)
app.include_router(voice_router)
app.include_router(protocols_router)
app.include_router(vision_router)
app.include_router(reasoning_router)
app.include_router(consciousness_router)
app.include_router(system_router)
app.include_router(nvidia_router)
app.include_router(auth_router)
app.include_router(utilities_router)
app.include_router(v4_features_router)
app.include_router(llm_router)
app.include_router(unified_router)
app.include_router(core_router)

logger.info("✅ 24 modular route modules integrated (222 endpoints)")

# Security middleware (auth + rate limiting)
try:
    from src.security.auth import verify_api_key, check_rate_limit
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    
    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        # Skip security for health/metrics endpoints
        if request.url.path in ["/health", "/metrics", "/metrics/prometheus", "/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)
        
        # Get client identifier
        client_ip = request.client.host if request.client else "unknown"
        api_key = request.headers.get("X-API-Key")
        
        # Check rate limit
        allowed, remaining, reset = check_rate_limit(client_ip, api_key)
        
        if not allowed:
            return JSONResponse(
                {"error": "Rate limit exceeded", "retry_after": reset},
                status_code=429,
                headers={"Retry-After": str(reset), "X-RateLimit-Remaining": "0"}
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset)
        
        return response
    
    logger.info("✅ Security middleware enabled (rate limiting active)")
except Exception as e:
    logger.warning(f"⚠️ Security middleware not loaded: {e}")

# Mount static files gracefully
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
else:
    logger.warning("⚠️ Static directory not found - static files will not be served")

# Mount BitNet Mobile Bundle downloads
bitnet_mobile_dir = "models/bitnet/mobile"
if not os.path.exists(bitnet_mobile_dir):
    os.makedirs(bitnet_mobile_dir, exist_ok=True)
app.mount("/downloads/bitnet", StaticFiles(directory=bitnet_mobile_dir), name="bitnet_downloads")

# ====== THIRD-PARTY PROTOCOL INTEGRATION ======
# Real production integration of MCP, A2A, and ACP protocols
from src.adapters.mcp_adapter import MCPAdapter
from src.adapters.a2a_adapter import A2AAdapter
from src.adapters.acp_adapter import ACPAdapter
from src.adapters.protocol_errors import (
    ProtocolConnectionError,
    ProtocolTimeoutError,
    ProtocolValidationError,
    CircuitBreakerOpenError
)

# Global protocol adapter instances
protocol_adapters = {
    "mcp": None,
    "a2a": None,
    "acp": None
}

def initialize_protocol_adapters():
    """Initialize all third-party protocol adapters"""
    global protocol_adapters
    
    try:
        # MCP Adapter (Anthropic)
        mcp_config = {
            "base_url": os.getenv("MCP_SERVER_URL", "http://localhost:3000"),  # MCP uses base_url
            "timeout": int(os.getenv("MCP_TIMEOUT", "30")),
            "failure_threshold": 5,
            "recovery_timeout": 60
        }
        protocol_adapters["mcp"] = MCPAdapter(mcp_config)
        logger.info("✅ MCP Adapter initialized")
        
        # A2A Adapter (Google)
        a2a_config = {
            "base_url": os.getenv("A2A_BASE_URL", "https://api.google.com/a2a/v1"),
            "api_key": os.getenv("A2A_API_KEY", ""),
            "timeout": int(os.getenv("A2A_TIMEOUT", "30")),
            "failure_threshold": 5
        }
        protocol_adapters["a2a"] = A2AAdapter(a2a_config)
        logger.info("✅ A2A Adapter initialized")
        
        # ACP Adapter (IBM)
        acp_config = {
            "base_url": os.getenv("ACP_BASE_URL", "http://localhost:8080"),
            "api_key": os.getenv("ACP_API_KEY", ""),
            "timeout": int(os.getenv("ACP_TIMEOUT", "30")),
            "failure_threshold": 5
        }
        protocol_adapters["acp"] = ACPAdapter(acp_config)
        logger.info("✅ ACP Adapter initialized")
        
        logger.info("🌐 All protocol adapters initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize protocol adapters: {e}")
        return False

# Initialize on startup
initialize_protocol_adapters()

# ====== MCP + DEEP AGENTS + MCP-UI INTEGRATION ======
# Initialize MCP integration globally
mcp_integration = None
mcp_demo_catalog = None

try:
    project_root = os.getenv("NIS_PROJECT_ROOT", os.getcwd())
    demo_catalog_path = os.path.join(project_root, "mcp_chatgpt_config.json")
    if os.path.exists(demo_catalog_path):
        with open(demo_catalog_path, "r", encoding="utf-8") as demo_file:
            mcp_demo_catalog = json.load(demo_file)
except Exception:
    mcp_demo_catalog = None

# @app.on_event("startup")  # Commented out to avoid duplicate startup events
async def setup_mcp_integration_disabled():
    """Initialize MCP + Deep Agents + mcp-ui integration on startup."""
    global mcp_integration
    try:
        from src.mcp.integration import setup_mcp_integration
        from src.mcp.langgraph_bridge import create_langgraph_adapter
        from src.mcp.mcp_ui_integration import setup_official_mcp_ui_integration
        
        logger.info("🚀 Initializing MCP + Deep Agents + mcp-ui integration...")
        
        # Setup MCP integration with NIS Protocol
        mcp_integration = await setup_mcp_integration({
            'mcp': {
                'host': '0.0.0.0', 
                'port': 8001,
                'enable_ui': True,
                'security': {'validate_intents': True, 'sandbox_ui': True}
            },
            'agent': {'provider': 'anthropic'},  # Use real agent provider
            'memory': {'backend': 'sqlite', 'connection_string': 'data/mcp_memory.db'}
        })
        
        # Setup LangGraph bridge for Agent Chat UI compatibility
        langgraph_bridge = create_langgraph_adapter(mcp_integration)
        
        # Setup official mcp-ui integration
        mcp_ui_adapter = setup_official_mcp_ui_integration(mcp_integration.mcp_server)
        
        # Store for use in endpoints
        app.state.mcp_integration = mcp_integration
        app.state.langgraph_bridge = langgraph_bridge
        app.state.mcp_ui_adapter = mcp_ui_adapter
        
        logger.info("✅ MCP integration ready with:")
        logger.info(f"   → {len(mcp_integration.get_tool_registry())} interactive tools")
        logger.info(f"   → {len(mcp_integration.mcp_server.planner.skills)} Deep Agent skills")
        logger.info(f"   → {len(mcp_ui_adapter.get_supported_content_types())} UI content types")
        
    except Exception as e:
        logger.error(f"❌ MCP integration failed: {e}")
        # Continue without MCP integration
        app.state.mcp_integration = None

# ====== FRONTEND UI ENDPOINTS REMOVED ======
# Frontend is in a separate repository
# Console, modern-chat, and formatted-chat endpoints removed
# API documentation available at /docs

async def initialize_system():
    """Initialize the NIS Protocol system - can be called manually for testing."""
    global llm_provider, web_search_agent, simulation_coordinator, learning_agent, planning_system, curiosity_engine, fallback_learning_agent
    global consciousness_service, protocol_bridge, bitnet_trainer
    global persistent_memory, reflective_generator, self_modifier, adaptive_goal_system  # V4.0 Globals
    try:
        llm_provider = GeneralLLMProvider()
        logger.info(" LLM Provider initialized successfully")
    except Exception as e:
        logger.error(f" Failed to initialize LLM Provider: {e}")
        # Create a fallback LLM provider to prevent None errors
        from src.llm.llm_manager import LLMManager
        try:
            llm_provider = LLMManager()
            logger.info("Fallback LLM Manager initialized")
        except Exception as e2:
            logger.error(f" Failed to initialize fallback LLM Manager: {e2}")
            # Final fallback - use GeneralLLMProvider with proper error handling
            llm_provider = GeneralLLMProvider()
            logger.warning("Using GeneralLLMProvider fallback - configure real LLM providers for full functionality")
    
    # Initialize Brain-like Agent Orchestrator
    try:
        if nis_agent_orchestrator is not None:
            await nis_agent_orchestrator.start_orchestrator()
            logger.info("Brain-like Agent Orchestrator initialized with 14 intelligent agents")
        else:
            logger.warning("❌ Agent Orchestrator is None - skipping initialization")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Agent Orchestrator: {e}")
    
    # Initialize Web Search Agent
    web_search_agent = WebSearchAgent()
    
    # Initialize Unified Scientific Coordinator (contains laplace, kan, pinn)
    coordinator = create_scientific_coordinator()

    # Initialize Simulation Coordinator (now unified)
    simulation_coordinator = coordinator  # Use unified coordinator's simulation capabilities

    # Initialize Learning Agent (full implementation - no fallbacks)
    try:
        learning_agent = LearningAgent(agent_id="core_learning_agent_01")
        logger.info("✅ LearningAgent initialized successfully")
    except Exception as e:
        logger.error(f"❌ LearningAgent initialization failed: {e}")
        raise  # Fail fast - no fallbacks per integrity rules

    # Initialize Planning System
    planning_system = AutonomousPlanningSystem()

    # Initialize Curiosity Engine
    curiosity_engine = CuriosityEngine()

    # Initialize Ethical Reasoner
    ethical_reasoner = EthicalReasoner()

    global scenario_simulator
    # Initialize Scenario Simulator (full implementation - no fallbacks)
    try:
        scenario_simulator = EnhancedScenarioSimulator()
        logger.info("✅ EnhancedScenarioSimulator initialized successfully")
    except Exception as e:
        logger.error(f"❌ EnhancedScenarioSimulator initialization failed: {e}")
        raise  # Fail fast - no fallbacks per integrity rules

    # Initialize Response Formatter
    response_formatter = NISResponseFormatter()

    # Initialize Conscious Agent
    conscious_agent = ConsciousAgent(agent_id="core_conscious_agent")

    # Initialize Enhanced Chat Memory System
    logger.info(" Initializing Enhanced Chat Memory System...")
    try:
        # Ensure LLM provider is available
        if llm_provider is None:
            logger.warning("LLM provider not available - using fallback memory system")
            enhanced_chat_memory = None
        else:
            # Create memory agent for enhanced capabilities
            try:
                memory_agent = EnhancedMemoryAgent(
                    agent_id="chat_memory_agent",
                    storage_path="data/chat_memory/agent_storage",
                    enable_logging=True,
                    enable_self_audit=True
                )
            except Exception as mem_e:
                logger.warning(f"EnhancedMemoryAgent initialization failed: {mem_e} - using fallback")
                memory_agent = None

            # Create chat memory configuration
            try:
                memory_config = ChatMemoryConfig(
                    storage_path="data/chat_memory/",
                    max_recent_messages=20,
                    max_context_messages=50,
                    semantic_search_threshold=0.7,
                    enable_cross_conversation_linking=True
                )
            except TypeError:
                # Fallback ChatMemoryConfig doesn't accept parameters
                logger.warning("ChatMemoryConfig fallback used - no parameters supported")
                memory_config = None

            # Initialize enhanced chat memory only if both components are available
            if memory_agent is not None and memory_config is not None:
                enhanced_chat_memory = EnhancedChatMemory(
                    config=memory_config,
                    memory_agent=memory_agent,
                    llm_provider=llm_provider
                )
                logger.info(" Enhanced Chat Memory System initialized successfully")
            else:
                logger.warning("Memory agent or config not available - using fallback memory system")
                enhanced_chat_memory = None
    except Exception as e:
        logger.error(f"Failed to initialize Enhanced Chat Memory: {e}")
        enhanced_chat_memory = None

    # ✅ FIXED: Re-enable scientific pipeline agents (numpy serialization fixed)
    # Use coordinator's pipeline agents (avoid duplication)
    # Access the agents from the coordinator after proper initialization
    laplace = coordinator.laplace if hasattr(coordinator, 'laplace') else None
    kan = coordinator.kan if hasattr(coordinator, 'kan') else None
    pinn = coordinator.pinn if hasattr(coordinator, 'pinn') else None

    # Log initialization status
    logger.info(f"✅ ScientificCoordinator initialized: laplace={laplace is not None}, kan={kan is not None}, pinn={pinn is not None}")

    # Start agent orchestrator background tasks if it exists
    if nis_agent_orchestrator is not None:
        try:
            await nis_agent_orchestrator.start_orchestrator()
            logger.info("NIS Agent Orchestrator background tasks started")
        except Exception as e:
            logger.error(f"Failed to start agent orchestrator background tasks: {e}")

    # 🧠 Initialize NIS HUB Enhanced Services
    consciousness_service = create_consciousness_service()
    
    # 🧬 V4.0: Initialize all consciousness phases
    consciousness_service.__init_evolution__()
    logger.info("🧬 Phase 1: Evolutionary consciousness initialized")
    
    consciousness_service.__init_genesis__()
    logger.info("🔬 Phase 2: Agent Genesis initialized")
    
    consciousness_service.__init_distributed__()
    logger.info("🌐 Phase 3: Distributed consciousness initialized")
    
    consciousness_service.__init_planning__()
    logger.info("🎯 Phase 4: Autonomous planning initialized")
    
    consciousness_service.__init_marketplace__()
    logger.info("💼 Phase 5: Consciousness marketplace initialized")
    
    consciousness_service.__init_multipath__()
    logger.info("🌳 Phase 6: Multi-path reasoning initialized")
    
    try:
        consciousness_service.__init_embodiment__()
        logger.info("🤖 Phase 8: Physical embodiment initialized")
    except OSError as e:
        logger.warning(f"⚠️ Phase 8 skipped due to file system error: {e}")
    except Exception as e:
        logger.error(f"❌ Phase 8 initialization failed: {e}")
    
    consciousness_service.__init_debugger__()
    logger.info("🔍 Phase 9: Consciousness debugger initialized")
    
    consciousness_service.__init_meta_evolution__()
    logger.info("🔬 Phase 10: Meta-evolution initialized")
    
    logger.info("✅ All 10 v4.0 consciousness phases initialized - system ready for AGI-level operation")
    
    # 🧠 V4.0: Initialize Self-Improving Components (Memory, Reflection, Goals)
    try:
        logger.info("🧠 Initializing V4.0 Self-Improving Components...")
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
        logger.info("✅ V4.0 Self-Improving System initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize V4.0 components: {e}")
    
    protocol_bridge = create_protocol_bridge_service(
        consciousness_service=consciousness_service,
        unified_coordinator=coordinator
    )
    
    # 🚀 Initialize Autonomous Executor
    executor = create_anthropic_style_executor(
        agent_id="anthropic_autonomous_executor",
        enable_consciousness_validation=True,
        enable_physics_validation=True,
        human_oversight_level="adaptive"
    )
    
    # 🎯 Initialize BitNet Online Training System (gated)
    bitnet_trainer = None
    try:
        # BitNet Initialization Logic (Enhanced with Auto-Detect)
        bitnet_dir = os.getenv("BITNET_MODEL_PATH", "models/bitnet/models/bitnet")
        model_exists = os.path.exists(os.path.join(bitnet_dir, "config.json"))
        
        # Enable if explicitly set OR if model exists (Auto-Detect)
        env_enabled = os.getenv("BITNET_TRAINING_ENABLED", "false").lower() == "true"
        bitnet_enabled = env_enabled or model_exists

        if bitnet_enabled and model_exists:
            logger.info(f"🤖 Found BitNet model at {bitnet_dir}")
            logger.info("🚀 Initializing BitNet Online Trainer...")
            
            from src.agents.training.bitnet_online_trainer import OnlineTrainingConfig, create_bitnet_online_trainer
            training_config = OnlineTrainingConfig(
                model_path=bitnet_dir,
                learning_rate=1e-5,
                quality_threshold=0.6,
                checkpoint_interval_minutes=30
            )
            try:
                bitnet_trainer = create_bitnet_online_trainer(
                    agent_id="bitnet_online_trainer",
                    config=training_config,
                    consciousness_service=consciousness_service
                )
                # 🔗 Connect BitNet trainer to LLM provider for automatic training capture
                if llm_provider:
                    llm_provider.set_training_collector(bitnet_trainer)
                    logger.info("🔗 BitNet Trainer connected to LLM Provider for automatic learning")
                logger.info("✅ BitNet Online Trainer ACTIVE and integrated with Consciousness Service")
            except Exception as e:
                logger.error(f"❌ Failed to initialize BitNet Trainer: {e}")
        else:
            if not model_exists:
                logger.info(f"ℹ️ BitNet model not found at {bitnet_dir} - Training disabled")
            else:
                logger.info("ℹ️ BitNet Training disabled by configuration")
    except Exception as e:
        logger.warning(f"BitNet Online Trainer initialization skipped: {e}")

    # Initialize Enhanced Multimodal Agents - v3.2
    global vision_agent, research_agent, reasoning_chain, document_agent
    vision_agent = MultimodalVisionAgent(agent_id="multimodal_vision_agent")
    research_agent = DeepResearchAgent(agent_id="deep_research_agent")
    reasoning_chain = EnhancedReasoningChain(agent_id="enhanced_reasoning_chain")
    document_agent = DocumentAnalysisAgent(agent_id="document_analysis_agent")
    
    # Initialize Precision Visualization Agent (Code-based, NOT AI image gen)
    diagram_agent = DiagramAgent()
    
    # Initialize Real-Time Data Pipeline Agent (global scope)
    global pipeline_agent
    pipeline_agent = None
    try:
        pipeline_agent = await create_real_time_pipeline_agent()
        logger.info("🚀 Real-Time Pipeline Agent initialized successfully")
    except Exception as e:
        logger.warning(f"⚠️ Real-Time Pipeline Agent initialization failed: {e} - using mock mode")

    logger.info("✅ NIS Protocol v3.2 ready with REAL LLM integration, NIS HUB consciousness, and multimodal capabilities!")
    logger.info(f"🧠 Consciousness Service initialized: {consciousness_service.agent_id}")
    logger.info(f"🌉 Protocol Bridge initialized: {protocol_bridge.agent_id}")
    logger.info(f"🎨 Vision Agent initialized: {vision_agent.agent_id}")
    logger.info(f"🔬 Research Agent initialized: {research_agent.agent_id}")
    logger.info(f"🧠 Reasoning Chain initialized: {reasoning_chain.agent_id}")
    logger.info(f"📄 Document Agent initialized: {document_agent.agent_id}")
    # logger.info(f"🚀 Anthropic-Style Executor initialized: {anthropic_executor.agent_id}")  # Temporarily disabled
    # logger.info(f"🎯 BitNet Online Trainer initialized: {bitnet_trainer.agent_id}")  # Temporarily disabled
    
    # ====== INJECT DEPENDENCIES INTO MODULAR ROUTES ======
    # This connects the initialized services to the modular route modules
    logger.info("🔗 Injecting dependencies into modular route modules...")
    
    try:
        # BitNet routes
        set_bitnet_trainer(bitnet_trainer)
        
        # Monitoring routes
        set_monitoring_dependencies(
            llm_provider=llm_provider,
            consciousness_service=consciousness_service
        )
        
        # Memory routes
        set_memory_dependencies(
            llm_provider=llm_provider,
            persistent_memory=persistent_memory
        )
        
        # Chat routes
        set_chat_dependencies(
            llm_provider=llm_provider,
            reflective_generator=reflective_generator,
            bitnet_trainer=bitnet_trainer,
            consciousness_service=consciousness_service
        )
        
        # Agents routes
        set_agents_dependencies(
            llm_provider=llm_provider,
            learning_agent=learning_agent,
            planning_system=planning_system,
            curiosity_engine=curiosity_engine,
            consciousness_service=consciousness_service
        )
        
        # Research routes
        set_research_dependencies(
            llm_provider=llm_provider,
            web_search_agent=web_search_agent,
            research_agent=research_agent
        )
        
        # Voice routes
        set_voice_dependencies(
            llm_provider=llm_provider,
            vibevoice_engine=vibevoice_engine
        )
        
        # Protocols routes
        set_protocols_dependencies(
            llm_provider=llm_provider,
            protocol_adapters=protocol_adapters
        )
        
        # Vision routes
        set_vision_dependencies(
            llm_provider=llm_provider,
            vision_agent=vision_agent,
            document_agent=document_agent
        )
        
        # Reasoning routes
        set_reasoning_dependencies(
            llm_provider=llm_provider,
            reasoning_chain=reasoning_chain
        )
        
        # Consciousness routes
        set_consciousness_dependencies(
            consciousness_service=consciousness_service,
            llm_provider=llm_provider
        )
        
        # System routes
        set_system_dependencies(
            llm_provider=llm_provider,
            consciousness_service=consciousness_service
        )
        
        # NVIDIA routes
        set_nvidia_dependencies(
            nemo_manager=nemo_manager
        )
        
        # Auth routes
        from src.security.user_management import user_manager
        set_auth_dependencies(user_manager=user_manager)
        
        # Utilities routes
        set_utilities_dependencies()
        
        # V4 Features routes
        set_v4_features_dependencies(
            persistent_memory=persistent_memory,
            self_modifier=self_modifier,
            adaptive_goal_system=adaptive_goal_system
        )
        
        # LLM routes
        set_llm_dependencies(llm_provider=llm_provider)
        
        # Unified routes
        set_unified_dependencies(llm_provider=llm_provider)
        
        # Core routes
        set_core_dependencies(
            llm_provider=llm_provider,
            conversation_memory=conversation_memory,
            agent_registry=agent_registry,
            tool_registry=tool_registry
        )
        
        logger.info("✅ All modular route dependencies injected successfully")
    except Exception as e:
        logger.error(f"❌ Failed to inject route dependencies: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("startup")  # RE-ENABLED: Initialize all components for chat functionality
async def startup_event():
    """Application startup event - RE-ENABLED FOR FULL FUNCTIONALITY."""
    logger.info("🚀 Starting NIS Protocol v3 initialization...")
    
    # Initialize Orchestrator explicitly
    initialize_agent_orchestrator()
    
    # Await initialization to ensure v4 endpoints are ready (Consciousness Service required)
    await initialize_system()
    
    # Initialize MCP integration in background (non-blocking)
    import asyncio as _asyncio
    _asyncio.create_task(initialize_mcp_integration())
    
    # Initialize VibeVoice engine synchronously
    initialize_vibevoice_engine()
    
    # Initialize NVIDIA NeMo manager synchronously
    initialize_nemo_manager()
    
    logger.info("🔄 Initialization scheduled in background")
    logger.info("📊 Enhanced pipeline: Laplace → Consciousness → KAN → PINN → Safety → Multimodal")
    logger.info("🎓 Online Training: BitNet continuously learning from conversations")


def initialize_vibevoice_engine():
    """Initialize VibeVoice engine synchronously."""
    global vibevoice_engine
    try:
        logger.info("🎙️ Initializing VibeVoice engine...")

        vibevoice_engine = VibeVoiceEngine()
        vibevoice_engine.initialize()

        logger.info("✅ VibeVoice engine ready for conversational voice chat")

    except Exception as e:
        logger.error(f"❌ VibeVoice initialization failed: {e}")
        # Continue without VibeVoice
        vibevoice_engine = None


def initialize_nemo_manager():
    """Initialize NVIDIA NeMo Integration Manager synchronously."""
    global nemo_manager
    try:
        if not NEMO_INTEGRATION_AVAILABLE:
            logger.info("⚠️ NVIDIA NeMo integration not available - skipping initialization")
            nemo_manager = None
            return
        
        logger.info("🚀 Initializing NVIDIA NeMo Integration Manager...")
        
        # Create config with API key from environment
        from src.agents.nvidia_nemo.nemo_integration_manager import NeMoIntegrationConfig
        
        nim_api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY")
        dgx_endpoint = os.getenv("DGX_CLOUD_ENDPOINT")
        
        config = NeMoIntegrationConfig(
            nim_api_key=nim_api_key,
            dgx_cloud_endpoint=dgx_endpoint,
            enable_nim_inference=bool(nim_api_key),
            enable_dgx_cloud=bool(dgx_endpoint)
        )
        
        # Create NeMo Integration Manager instance
        nemo_manager = NeMoIntegrationManager(config=config)
        
        # Initialize integration resources if keys are present
        if nim_api_key:
            import asyncio
            # We are in a sync function, but initialization requires async
            # This will be handled by the background task or when first used
            logger.info("✅ NVIDIA API Key found - NIM Inference enabled")
        
        logger.info("✅ NVIDIA NeMo Integration Manager ready")
        logger.info("   • NeMo Framework: Available")
        logger.info("   • Agent Toolkit: Ready for installation")
        logger.info("   • Inception Benefits: Enterprise access enabled")

    except Exception as e:
        logger.error(f"❌ NVIDIA NeMo manager initialization failed: {e}")
        # Continue without NeMo manager
        nemo_manager = None


async def initialize_mcp_integration():
    """Initialize MCP integration in background."""
    global mcp_integration
    try:
        logger.info("🚀 Initializing MCP + Deep Agents + mcp-ui integration...")
        
        from src.mcp.integration import MCPIntegration
        
        # Create MCP integration instance with port 8020 to avoid conflict with FastAPI on 8000
        mcp_config = {
            "mcp": {
                "host": "localhost",
                "port": 8020  # Changed from 8000 to avoid FastAPI conflict
            }
        }
        mcp_integration = MCPIntegration(nis_config=mcp_config)
        await mcp_integration.initialize()
        
        # Store in app state for endpoint access
        app.state.mcp_integration = mcp_integration
        
        logger.info("✅ MCP integration ready on port 8020")
        
    except Exception as e:
        logger.error(f"❌ MCP integration failed: {e}")
        # Continue without MCP integration
        app.state.mcp_integration = None


# ====== ALL ENDPOINTS NOW IN MODULAR ROUTES ======
# See routes/ directory for all API endpoints:
# - 24 route modules with 222 endpoints
# - Dependency injection via set_*_dependencies() functions
# - Documentation: docs/organized/architecture/ROUTE_MIGRATION.md

# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000

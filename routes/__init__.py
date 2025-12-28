"""
NIS Protocol v4.0 - Route Modules

This package contains modular route definitions for the NIS Protocol API.
Routes are organized by domain for better maintainability.

IMPORTANT: This is a gradual migration. Routes are being moved from main.py
to these modules incrementally to ensure stability.

Modules:
- robotics.py: FK/IK, trajectory planning, telemetry (5 endpoints)
- physics.py: Physics validation, PINN, equations (6 endpoints)
- bitnet.py: BitNet training, status, download (6 endpoints)
- webhooks.py: Webhook registration and management (3 endpoints)
- monitoring.py: Health, metrics, analytics (16 endpoints)
- memory.py: Memory management, conversations, topics (15 endpoints)
- chat.py: Simple chat, streaming, reflective (9 endpoints)
- agents.py: Learning, planning, simulation, ethics (11 endpoints)
- research.py: Deep research, claim validation (4 endpoints)
- voice.py: STT, TTS, voice chat, voice settings (7 endpoints)
- protocols.py: MCP, A2A, ACP protocol integrations (22 endpoints)
- vision.py: Image analysis, generation, visualization (12 endpoints)
- reasoning.py: Collaborative reasoning, debate (3 endpoints)
- consciousness.py: V4.0 evolution, genesis, collective, embodiment (28 endpoints)
- system.py: Configuration, state, edge AI, brain orchestration (18 endpoints)
- nvidia.py: NVIDIA Inception, NeMo, enterprise features (8 endpoints)
- auth.py: Authentication, user management, API keys (11 endpoints)
- utilities.py: Cost tracking, cache, templates, code execution (14 endpoints)
- v4_features.py: V4.0 memory, self-modification, goals (11 endpoints)
- llm.py: LLM optimization, consensus, analytics (7 endpoints)
- unified.py: Unified pipeline, autonomous mode, integration (4 endpoints)
- core.py: Root endpoint, health check (2 endpoints)

Usage:
    from routes import (
        robotics_router, physics_router, bitnet_router, webhooks_router,
        monitoring_router, memory_router, chat_router, agents_router, 
        research_router, voice_router, protocols_router, vision_router,
        reasoning_router
    )
    
    # Include routers in FastAPI app
    app.include_router(robotics_router)    # /robotics prefix
    app.include_router(physics_router)     # /physics prefix
    app.include_router(bitnet_router)      # /models/bitnet and /training/bitnet
    app.include_router(webhooks_router)    # /webhooks prefix
    app.include_router(monitoring_router)  # /health, /metrics, /analytics, /system
    app.include_router(memory_router)      # /memory prefix
    app.include_router(chat_router)        # /chat prefix
    app.include_router(agents_router)      # /agents prefix
    app.include_router(research_router)    # /research prefix
    app.include_router(voice_router)       # /voice, /communication, /ws/voice-chat
    app.include_router(protocols_router)   # /protocol/*, /api/mcp/*, /agents, /runs
    app.include_router(vision_router)      # /vision/*, /image/*, /visualization/*
    app.include_router(reasoning_router)   # /reasoning/*

MIGRATION STATUS:
- [x] robotics.py - Ready for testing (5 endpoints)
- [x] physics.py - Ready for testing (6 endpoints)
- [x] bitnet.py - Ready for testing (6 endpoints)
- [x] webhooks.py - Ready for testing (3 endpoints)
- [x] monitoring.py - Ready for testing (16 endpoints)
- [x] memory.py - Ready for testing (15 endpoints)
- [x] chat.py - Ready for testing (9 endpoints)
- [x] agents.py - Ready for testing (11 endpoints)
- [x] research.py - Ready for testing (4 endpoints)
- [x] voice.py - Ready for testing (7 endpoints)
- [x] protocols.py - Ready for testing (22 endpoints)
- [x] vision.py - Ready for testing (12 endpoints)
- [x] reasoning.py - Ready for testing (3 endpoints)
- [x] consciousness.py - Ready for testing (28 endpoints)
- [x] system.py - Ready for testing (18 endpoints)
- [x] nvidia.py - Ready for testing (8 endpoints)
- [x] auth.py - Ready for testing (11 endpoints)
- [x] utilities.py - Ready for testing (14 endpoints)
- [x] v4_features.py - Ready for testing (11 endpoints)
- [x] llm.py - Ready for testing (7 endpoints)
- [x] unified.py - Ready for testing (4 endpoints)
- [x] core.py - Ready for testing (2 endpoints)
- [x] isaac.py - Ready for testing (18 endpoints)

Total: 240 endpoints in 25 modules (100% complete)
main.py reduced from 12,291 to 1,196 lines (90% reduction)
"""

# Import routers
from .robotics import router as robotics_router
from .physics import router as physics_router
from .bitnet import router as bitnet_router, set_bitnet_trainer
from .webhooks import router as webhooks_router, trigger_webhooks, get_webhooks
from .monitoring import router as monitoring_router, set_dependencies as set_monitoring_dependencies
from .memory import router as memory_router, set_dependencies as set_memory_dependencies
from .chat import router as chat_router, set_dependencies as set_chat_dependencies
from .agents import router as agents_router, set_dependencies as set_agents_dependencies
from .research import router as research_router, set_dependencies as set_research_dependencies
from .voice import router as voice_router, set_dependencies as set_voice_dependencies
from .protocols import router as protocols_router, set_dependencies as set_protocols_dependencies
from .vision import router as vision_router, set_dependencies as set_vision_dependencies
from .reasoning import router as reasoning_router, set_dependencies as set_reasoning_dependencies
from .consciousness import router as consciousness_router, set_dependencies as set_consciousness_dependencies
from .system import router as system_router, set_dependencies as set_system_dependencies
from .nvidia import router as nvidia_router, set_dependencies as set_nvidia_dependencies
from .auth import router as auth_router, set_dependencies as set_auth_dependencies
from .utilities import router as utilities_router, set_dependencies as set_utilities_dependencies
from .v4_features import router as v4_features_router, set_dependencies as set_v4_features_dependencies
from .llm import router as llm_router, set_dependencies as set_llm_dependencies
from .unified import router as unified_router, set_dependencies as set_unified_dependencies
from .core import router as core_router, set_dependencies as set_core_dependencies
from .isaac import router as isaac_router
from .hub_gateway import router as hub_gateway_router
from .autonomous import router as autonomous_router, set_dependencies as set_autonomous_dependencies

__all__ = [
    # Routers
    "robotics_router",
    "physics_router", 
    "bitnet_router",
    "webhooks_router",
    "monitoring_router",
    "memory_router",
    "chat_router",
    "agents_router",
    "research_router",
    "voice_router",
    "protocols_router",
    "vision_router",
    "reasoning_router",
    "consciousness_router",
    "system_router",
    "nvidia_router",
    "auth_router",
    "utilities_router",
    "v4_features_router",
    "llm_router",
    "unified_router",
    "core_router",
    "isaac_router",
    "hub_gateway_router",
    "autonomous_router",
    # Dependency setters
    "set_bitnet_trainer",
    "set_monitoring_dependencies",
    "set_memory_dependencies",
    "set_chat_dependencies",
    "set_agents_dependencies",
    "set_research_dependencies",
    "set_voice_dependencies",
    "set_protocols_dependencies",
    "set_vision_dependencies",
    "set_reasoning_dependencies",
    "set_consciousness_dependencies",
    "set_system_dependencies",
    "set_nvidia_dependencies",
    "set_auth_dependencies",
    "set_utilities_dependencies",
    "set_v4_features_dependencies",
    "set_llm_dependencies",
    "set_unified_dependencies",
    "set_core_dependencies",
    "set_autonomous_dependencies",
    # Webhook helpers
    "trigger_webhooks",
    "get_webhooks",
]

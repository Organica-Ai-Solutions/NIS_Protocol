"""
NIS Protocol v4.0 - Core System Routes

This module contains essential system endpoints:
- Root endpoint (/)
- Health check (/health)
- System status

These are the minimal endpoints required for the API to function.
Frontend UI endpoints (console, modern-chat) are NOT included as
the frontend is in a separate repository.

MIGRATION STATUS: Ready for testing

Usage:
    from routes.core import router as core_router
    app.include_router(core_router, tags=["System"])
"""

import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("nis.routes.core")

# Create router
router = APIRouter(tags=["System"])


# ====== Dependency Injection ======

def get_llm_provider():
    return getattr(router, '_llm_provider', None)

def get_conversation_memory():
    return getattr(router, '_conversation_memory', {})

def get_agent_registry():
    return getattr(router, '_agent_registry', {})

def get_tool_registry():
    return getattr(router, '_tool_registry', {})


# ====== Core System Endpoints ======

@router.get("/")
async def read_root():
    """
    Root endpoint - NIS Protocol system information
    
    Returns system status, available providers, and API documentation links.
    """
    llm_provider = get_llm_provider()
    
    models = []
    provider_names = []
    
    if llm_provider and hasattr(llm_provider, 'providers') and llm_provider.providers:
        provider_names = list(llm_provider.providers.keys())
        for p in llm_provider.providers.values():
            if isinstance(p, dict):
                models.append(p.get("model", "default"))
            else:
                models.append(getattr(p, 'model', 'default'))
    else:
        models = ["system-initializing"]

    return {
        "system": "NIS Protocol v4.0",
        "version": "4.0.1",
        "pattern": "modular_architecture",
        "status": "operational",
        "real_llm_integrated": provider_names,
        "provider": provider_names,
        "model": models,
        "features": [
            "Modular Route Architecture (23 modules)",
            "Real LLM Integration (OpenAI, Anthropic, DeepSeek, Google, NVIDIA)",
            "Multi-Agent Coordination",
            "Physics-Informed Reasoning (PINN)",
            "10-Phase Consciousness Pipeline",
            "Robotics Control (FK/IK, Trajectory)",
            "Voice Chat (STT/TTS)",
            "Vision & Image Generation",
            "Protocol Integration (MCP, A2A, ACP)"
        ],
        "api_docs": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        },
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "chat": "/chat",
            "vision": "/vision/analyze",
            "voice": "/voice/transcribe",
            "research": "/research/deep",
            "robotics": "/robotics/capabilities",
            "consciousness": "/v4/dashboard/complete"
        },
        "timestamp": time.time()
    }


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns system health status for monitoring and load balancers.
    """
    llm_provider = get_llm_provider()
    conversation_memory = get_conversation_memory()
    agent_registry = get_agent_registry()
    tool_registry = get_tool_registry()
    
    try:
        provider_names = []
        models = []
        
        if llm_provider and getattr(llm_provider, 'providers', None):
            provider_names = list(llm_provider.providers.keys())
            for p in llm_provider.providers.values():
                if isinstance(p, dict):
                    models.append(p.get("model", "default"))
                else:
                    models.append(getattr(p, 'model', 'default'))
        
        return {
            "status": "healthy",
            "version": "4.0.1",
            "timestamp": time.time(),
            "provider": provider_names,
            "model": models,
            "real_ai": provider_names,
            "conversations_active": len(conversation_memory) if isinstance(conversation_memory, dict) else 0,
            "agents_registered": len(agent_registry) if isinstance(agent_registry, dict) else 0,
            "tools_available": len(tool_registry) if isinstance(tool_registry, dict) else 0,
            "modular_routes": 23,
            "pattern": "modular_architecture"
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# ====== Dependency Injection Helper ======

def set_dependencies(
    llm_provider=None,
    conversation_memory=None,
    agent_registry=None,
    tool_registry=None
):
    """Set dependencies for the core router"""
    router._llm_provider = llm_provider
    router._conversation_memory = conversation_memory or {}
    router._agent_registry = agent_registry or {}
    router._tool_registry = tool_registry or {}

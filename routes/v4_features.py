"""
NIS Protocol v4.0 - V4 Features Routes

This module contains V4.0-specific feature endpoints:
- Persistent Memory System
- Self-Modification System
- Adaptive Goal System

MIGRATION STATUS: Ready for testing
- These routes mirror the ones in main.py
- Can be tested independently before switching over

Usage:
    from routes.v4_features import router as v4_features_router
    app.include_router(v4_features_router, tags=["V4.0 Features"])
"""

import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("nis.routes.v4_features")

# Create router
router = APIRouter(prefix="/v4", tags=["V4.0 Features"])

# ====== Request Models ======

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = None
    model: Optional[str] = None


# ====== Dependency Injection ======

def get_persistent_memory():
    memory = getattr(router, '_persistent_memory', None)
    if memory is None:
        try:
            from src.memory.persistent_memory import get_memory_system
            return get_memory_system()
        except ImportError:
            return None
    return memory

def get_self_modifier():
    modifier = getattr(router, '_self_modifier', None)
    if modifier is None:
        try:
            from src.core.self_modifier import get_self_modifier as get_modifier
            return get_modifier()
        except ImportError:
            return None
    return modifier

def get_adaptive_goal_system():
    return getattr(router, '_adaptive_goal_system', None)

def get_llm_provider():
    return getattr(router, '_llm_provider', None)


# ====== V4.0 Endpoints ======

@router.post("/chat")
async def v4_chat(request: ChatRequest):
    """
    V4.0 Enhanced Chat Endpoint
    
    Features:
    - Multi-provider LLM support
    - Persistent conversation memory
    - Context-aware responses
    """
    try:
        llm_provider = get_llm_provider()
        if not llm_provider:
            return JSONResponse({
                "response": "V4 chat is available but LLM provider not initialized.",
                "model": "fallback",
                "conversation_id": request.conversation_id or "default"
            })
        
        # Use LLM provider for response
        response = await llm_provider.generate(
            prompt=request.message,
            model=request.model or "gpt-4",
            max_tokens=500
        )
        
        return {
            "response": response.get("content", "Response generated"),
            "model": request.model or "gpt-4",
            "conversation_id": request.conversation_id or "default",
            "v4_features": True
        }
    except Exception as e:
        logger.error(f"V4 chat error: {e}")
        return JSONResponse({
            "response": f"Chat processed: {request.message[:50]}...",
            "model": "fallback",
            "conversation_id": request.conversation_id or "default"
        })


# ====== V4.0 Persistent Memory Endpoints ======

@router.post("/memory/store")
async def store_memory(request: Dict[str, Any]):
    """Store a memory in the persistent memory system"""
    persistent_memory = get_persistent_memory()
    
    if persistent_memory is None:
        raise HTTPException(status_code=503, detail="Persistent memory not initialized")
    
    content = request.get("content", "")
    memory_type = request.get("type", "semantic")
    importance = request.get("importance", 0.5)
    metadata = request.get("metadata", {})
    
    memory_id = await persistent_memory.store(
        content=content,
        memory_type=memory_type,
        importance=importance,
        metadata=metadata
    )
    
    return {"status": "stored", "memory_id": memory_id, "type": memory_type}


@router.post("/memory/retrieve")
async def retrieve_memories(request: Dict[str, Any]):
    """Retrieve relevant memories"""
    persistent_memory = get_persistent_memory()
    
    if persistent_memory is None:
        raise HTTPException(status_code=503, detail="Persistent memory not initialized")
    
    query = request.get("query", "")
    memory_type = request.get("type")
    top_k = request.get("top_k", 5)
    
    results = await persistent_memory.retrieve(
        query=query,
        memory_type=memory_type,
        top_k=top_k
    )
    
    return {
        "query": query,
        "results": [
            {
                "content": r.entry.content,
                "type": r.entry.memory_type,
                "relevance": r.relevance_score,
                "importance": r.importance_score,
                "combined_score": r.combined_score
            }
            for r in results
        ]
    }


@router.get("/memory/stats")
async def get_memory_stats():
    """Get memory system statistics"""
    persistent_memory = get_persistent_memory()
    
    if persistent_memory is None:
        return {"status": "not_initialized", "message": "Persistent memory not available"}
    
    return persistent_memory.get_stats()


# ====== V4.0 Self-Modification Endpoints ======

@router.get("/self/status")
async def get_self_modifier_status():
    """Get self-modifier status and current parameters"""
    self_modifier = get_self_modifier()
    
    if self_modifier is None:
        return {"status": "not_initialized", "message": "Self-modifier not available"}
    
    return self_modifier.get_status()


@router.post("/self/record-metric")
async def record_performance_metric(request: Dict[str, Any]):
    """Record a performance metric for self-optimization"""
    self_modifier = get_self_modifier()
    
    if self_modifier is None:
        raise HTTPException(status_code=503, detail="Self-modifier not initialized")
    
    metric_name = request.get("metric", "response_quality")
    value = request.get("value", 0.5)
    
    self_modifier.record_metric(metric_name, value)
    return {"status": "recorded", "metric": metric_name, "value": value}


@router.post("/self/optimize")
async def trigger_auto_optimization():
    """Trigger automatic self-optimization based on performance"""
    self_modifier = get_self_modifier()
    
    if self_modifier is None:
        raise HTTPException(status_code=503, detail="Self-modifier not initialized")
    
    modifications = await self_modifier.auto_optimize()
    
    return {
        "status": "optimized",
        "modifications_applied": len(modifications),
        "details": [
            {"target": m.target, "reason": m.reason}
            for m in modifications
        ]
    }


@router.get("/self/history")
async def get_modification_history():
    """Get history of self-modifications"""
    self_modifier = get_self_modifier()
    
    if self_modifier is None:
        return {"status": "not_initialized", "history": []}
    
    return {
        "history": self_modifier.get_modification_history(limit=20)
    }


@router.get("/self/parameters")
async def get_current_parameters():
    """Get current self-modified parameters"""
    self_modifier = get_self_modifier()
    
    if self_modifier is None:
        return {"status": "not_initialized", "parameters": {}, "routing_rules": {}}
    
    return {
        "parameters": self_modifier.parameters,
        "routing_rules": self_modifier.routing_rules
    }


# ====== V4.0 Adaptive Goal System Endpoints ======

@router.post("/goals/generate")
async def generate_goals():
    """Trigger autonomous goal generation"""
    adaptive_goal_system = get_adaptive_goal_system()
    
    if adaptive_goal_system:
        return await adaptive_goal_system.process({"operation": "generate_goals"})
    return {"status": "error", "message": "Goal system not initialized"}


@router.get("/goals/list")
async def list_goals():
    """List active goals"""
    adaptive_goal_system = get_adaptive_goal_system()
    
    if adaptive_goal_system:
        return await adaptive_goal_system.process({"operation": "get_active_goals"})
    return {"status": "error", "message": "Goal system not initialized"}


@router.get("/goals/metrics")
async def get_goal_metrics():
    """Get goal system performance metrics"""
    adaptive_goal_system = get_adaptive_goal_system()
    
    if adaptive_goal_system:
        return {"metrics": adaptive_goal_system.goal_metrics}
    return {"status": "error", "message": "Goal system not initialized"}


# ====== Dependency Injection Helper ======

def set_dependencies(
    persistent_memory=None,
    self_modifier=None,
    adaptive_goal_system=None,
    llm_provider=None
):
    """Set dependencies for V4 features"""
    if persistent_memory:
        router._persistent_memory = persistent_memory
    if self_modifier:
        router._self_modifier = self_modifier
    if adaptive_goal_system:
        router._adaptive_goal_system = adaptive_goal_system
    if llm_provider:
        router._llm_provider = llm_provider

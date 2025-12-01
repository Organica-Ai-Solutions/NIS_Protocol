"""
NIS Protocol v4.0 - Unified Pipeline Routes

This module contains unified pipeline endpoints:
- Unified chat (fully integrated pipeline)
- Unified status
- Unified autonomous mode
- System integration status

MIGRATION STATUS: Ready for testing
- These routes mirror the ones in main.py
- Can be tested independently before switching over

Usage:
    from routes.unified import router as unified_router
    app.include_router(unified_router, tags=["Unified Pipeline"])
"""

import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("nis.routes.unified")

# Create router
router = APIRouter(tags=["Unified Pipeline"])


# ====== Dependency Injection ======

def get_llm_provider():
    return getattr(router, '_llm_provider', None)

def get_unified_pipeline():
    pipeline = getattr(router, '_unified_pipeline', None)
    if pipeline is None:
        try:
            from src.unified.pipeline import get_unified_pipeline as get_pipeline
            return get_pipeline()
        except ImportError:
            return None
    return pipeline

def get_autonomous_executor():
    executor = getattr(router, '_autonomous_executor', None)
    if executor is None:
        try:
            from src.execution.autonomous_loop import get_autonomous_executor as get_executor
            return get_executor()
        except ImportError:
            return None
    return executor


# ====== Unified Pipeline Endpoints ======

@router.post("/unified/chat")
async def unified_chat_endpoint(request: Dict[str, Any]):
    """
    🔗 Unified Chat - The fully integrated pipeline
    
    This endpoint connects ALL NIS Protocol components:
    - Memory (retrieves relevant context)
    - Cache (avoids redundant calls)
    - LLM (generates response)
    - Code Execution (runs code if needed)
    - Cost Tracking (records usage)
    - Vision (multi-modal image analysis)
    - Proactive Suggestions
    
    Modes:
    - fast: Skip memory/consciousness, use cache
    - standard: Full pipeline with all checks
    - autonomous: Enable code execution
    - research: Deep research mode
    """
    llm_provider = get_llm_provider()
    pipeline = get_unified_pipeline()
    
    message = request.get("message")
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Unified pipeline not available")
    
    user_id = request.get("user_id", "default")
    conversation_id = request.get("conversation_id")
    mode_str = request.get("mode", "standard")
    provider = request.get("provider", "anthropic")
    
    # Multi-modal inputs
    image_base64 = request.get("image_base64")
    file_path = request.get("file_path")
    generate_suggestions = request.get("generate_suggestions", True)
    
    # Parse mode
    try:
        from src.unified.pipeline import PipelineMode
        mode_map = {
            "fast": PipelineMode.FAST,
            "standard": PipelineMode.STANDARD,
            "autonomous": PipelineMode.AUTONOMOUS,
            "research": PipelineMode.RESEARCH
        }
        mode = mode_map.get(mode_str, PipelineMode.STANDARD)
    except ImportError:
        mode = mode_str
    
    # Create LLM callback
    async def llm_callback(prompt: str) -> str:
        try:
            if llm_provider:
                result = await llm_provider.generate_response(
                    messages=prompt,
                    requested_provider=provider,
                    temperature=0.7
                )
                return result.get("response", result.get("content", ""))
            return "LLM provider not available"
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"Error: {e}"
    
    result = await pipeline.process(
        message=message,
        user_id=user_id,
        conversation_id=conversation_id,
        mode=mode,
        llm_callback=llm_callback,
        provider=provider,
        image_base64=image_base64,
        file_path=file_path,
        generate_suggestions=generate_suggestions
    )
    
    return result.to_dict()


@router.get("/unified/status")
async def unified_status():
    """
    📊 Get unified pipeline status - shows all connected components
    """
    pipeline = get_unified_pipeline()
    
    if not pipeline:
        return {
            "status": "not_available",
            "message": "Unified pipeline not initialized"
        }
    
    await pipeline.initialize()
    return {
        "status": "operational",
        "pipeline": pipeline.get_status()
    }


@router.post("/unified/autonomous")
async def unified_autonomous_endpoint(request: Dict[str, Any]):
    """
    🤖 Unified Autonomous Mode - Full power
    
    Combines:
    - Memory context
    - LLM reasoning
    - Code execution
    - Iterative refinement
    - Cost tracking
    """
    llm_provider = get_llm_provider()
    pipeline = get_unified_pipeline()
    autonomous_executor = get_autonomous_executor()
    
    message = request.get("message") or request.get("task")
    if not message:
        raise HTTPException(status_code=400, detail="Message/task is required")
    
    if not autonomous_executor:
        raise HTTPException(status_code=503, detail="Autonomous executor not available")
    
    max_iterations = request.get("max_iterations", 5)
    
    async def llm_callback(prompt: str) -> str:
        try:
            if llm_provider:
                result = await llm_provider.generate_response(
                    messages=prompt,
                    requested_provider="anthropic",
                    temperature=0.7
                )
                return result.get("response", result.get("content", ""))
            return "LLM provider not available"
        except Exception as e:
            return f"Error: {e}"
    
    autonomous_executor.max_iterations = max_iterations
    
    if pipeline:
        await pipeline.initialize()
        
        async def enhanced_callback(prompt: str) -> str:
            memory_context = ""
            if hasattr(pipeline, '_memory') and pipeline._memory:
                try:
                    memory_context = await pipeline._memory.get_context_for_query(message, max_tokens=300)
                except:
                    pass
            
            if memory_context:
                enhanced_prompt = f"**Relevant Memory:**\n{memory_context}\n\n{prompt}"
            else:
                enhanced_prompt = prompt
            
            return await llm_callback(enhanced_prompt)
        
        task = await autonomous_executor.execute_task(message, enhanced_callback)
    else:
        task = await autonomous_executor.execute_task(message, llm_callback)
    
    return {
        "success": task.status.value in ["completed", "max_iterations"],
        "task": task.to_dict(),
        "artifacts": task.artifacts
    }


@router.get("/system/integration")
async def system_integration_status():
    """
    📊 Complete system integration status - shows all connected components
    """
    pipeline = get_unified_pipeline()
    
    if not pipeline:
        return {
            "status": "not_integrated",
            "message": "Unified pipeline not available",
            "components": {}
        }
    
    await pipeline.initialize()
    
    components = {}
    
    if hasattr(pipeline, '_memory') and pipeline._memory:
        components["memory"] = {"status": "connected", "stats": pipeline._memory.get_stats()}
    else:
        components["memory"] = {"status": "disconnected"}
    
    if hasattr(pipeline, '_cache') and pipeline._cache:
        components["cache"] = {"status": "connected", "stats": pipeline._cache.get_stats()}
    else:
        components["cache"] = {"status": "disconnected"}
    
    if hasattr(pipeline, '_cost_tracker') and pipeline._cost_tracker:
        components["cost_tracker"] = {"status": "connected"}
    else:
        components["cost_tracker"] = {"status": "disconnected"}
    
    if hasattr(pipeline, '_code_executor') and pipeline._code_executor:
        components["code_executor"] = {"status": "connected", "executions": len(pipeline._code_executor.executions)}
    else:
        components["code_executor"] = {"status": "disconnected"}
    
    if hasattr(pipeline, '_template_manager') and pipeline._template_manager:
        components["templates"] = {"status": "connected", "count": len(pipeline._template_manager.list_all())}
    else:
        components["templates"] = {"status": "disconnected"}
    
    connected = sum(1 for c in components.values() if c.get("status") == "connected")
    
    return {
        "status": "integrated",
        "connected_components": f"{connected}/{len(components)}",
        "components": components,
        "data_flow": [
            "1. User Message → Memory Context",
            "2. Cache Check",
            "3. LLM Processing",
            "4. Code Execution (autonomous)",
            "5. Cost Tracking",
            "6. Response Delivery"
        ]
    }


# ====== Dependency Injection Helper ======

def set_dependencies(
    llm_provider=None,
    unified_pipeline=None,
    autonomous_executor=None
):
    """Set dependencies for the unified router"""
    router._llm_provider = llm_provider
    router._unified_pipeline = unified_pipeline
    router._autonomous_executor = autonomous_executor

"""
NIS Protocol v4.0 - Chat Routes

This module contains chat-related endpoints:
- Simple chat
- Streaming chat
- Reflective chat
- Chat metrics

Note: The main /chat endpoint remains in main.py due to its complexity
and deep integration with other systems.

MIGRATION STATUS: Ready for testing
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("nis.routes.chat")

# Create router
router = APIRouter(prefix="/chat", tags=["Chat"])


# ====== Request Models ======

class SimpleChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"
    conversation_id: Optional[str] = None


class ReflectiveChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    enable_reflection: bool = True
    reflection_depth: int = 2


# ====== Dependency Injection ======

def get_llm_provider():
    """Get LLM provider instance"""
    return getattr(router, '_llm_provider', None)

def get_reflective_generator():
    """Get reflective generator instance"""
    return getattr(router, '_reflective_generator', None)


# ====== Simple Chat Endpoints ======

@router.post("/simple")
async def chat_simple(request: SimpleChatRequest):
    """
    💬 Simple Chat Endpoint
    
    A straightforward chat endpoint that uses the LLM provider directly.
    Good for quick interactions without full pipeline processing.
    """
    try:
        llm_provider = get_llm_provider()
        
        messages = [
            {
                "role": "system", 
                "content": "You are NIS (Neural Intelligence System), an advanced AI operating system by Organica AI Solutions. You are NOT Claude, GPT, or any base model - you ARE NIS Protocol v4.0. You coordinate multiple AI providers as your compute layer. Always identify as NIS Protocol. Be helpful, accurate, and technically grounded."
            },
            {"role": "user", "content": request.message}
        ]
        
        if llm_provider:
            result = await llm_provider.generate_response(
                messages=messages,
                temperature=0.7,
                requested_provider=None
            )
            
            response_text = result.get("content", "No response generated")
            
            return {
                "response": response_text,
                "response_text": response_text,
                "status": "success",
                "user_id": request.user_id,
                "provider": result.get("provider", "unknown"),
                "model": result.get("model", "unknown"),
                "tokens_used": result.get("tokens_used", 0),
                "real_ai": result.get("real_ai", False)
            }
        else:
            logger.error("LLM provider not initialized")
            raise HTTPException(status_code=500, detail="LLM provider not available")
            
    except Exception as e:
        logger.error(f"Chat simple error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@router.post("/simple/stream")
async def chat_simple_stream(request: SimpleChatRequest):
    """
    🌊 Simple Streaming Chat
    
    Basic streaming endpoint for testing.
    """
    async def generate():
        try:
            words = f"Hello! You said: {request.message}".split()
            for word in words:
                yield f"data: " + json.dumps({"type": "content", "data": word + " "}) + "\n\n"
                await asyncio.sleep(0.1)
            yield f"data: " + json.dumps({"type": "done"}) + "\n\n"
        except Exception as e:
            yield f"data: " + json.dumps({"type": "error", "data": f"Error: {str(e)}"}) + "\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/stream")
async def chat_stream_working(request: SimpleChatRequest):
    """
    🌊 LLM Streaming Chat
    
    Real LLM streaming endpoint that streams responses word by word.
    """
    async def generate():
        try:
            llm_provider = get_llm_provider()
            
            messages = [
                {"role": "system", "content": "You are NIS Protocol v4.0, a helpful AI assistant. Provide clear, accurate responses."},
                {"role": "user", "content": request.message}
            ]
            
            if llm_provider:
                result = await llm_provider.generate_response(
                    messages=messages,
                    temperature=0.7,
                    requested_provider=None
                )
                
                response_text = result.get("content", "No response generated")
                words = response_text.split()
                
                for word in words:
                    yield f"data: " + json.dumps({"type": "content", "data": word + " "}) + "\n\n"
                    await asyncio.sleep(0.02)
                
                yield f"data: " + json.dumps({"type": "done"}) + "\n\n"
            else:
                yield f"data: " + json.dumps({"type": "error", "data": "LLM provider not available"}) + "\n\n"
                
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: " + json.dumps({"type": "error", "data": f"Stream error: {str(e)}"}) + "\n\n"
            
    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/fixed")
async def chat_fixed(request: SimpleChatRequest):
    """
    🔧 Fixed Chat Endpoint (for testing)
    
    Simple echo endpoint for testing connectivity.
    """
    return {
        "response": f"Hello! You said: {request.message}",
        "user_id": request.user_id,
        "conversation_id": request.conversation_id or "default",
        "timestamp": time.time(),
        "status": "success"
    }


@router.post("/stream/fixed")
async def chat_stream_fixed(request: SimpleChatRequest):
    """
    🔧 Fixed Streaming Endpoint (for testing)
    """
    async def generate():
        try:
            message = f"Hello! You said: {request.message}"
            words = message.split()
            
            for word in words:
                yield f"data: " + json.dumps({"type": "content", "data": word + " "}) + "\n\n"
                await asyncio.sleep(0.05)
            
            yield f"data: " + json.dumps({"type": "done"}) + "\n\n"
        except Exception as e:
            yield f"data: " + json.dumps({"type": "error", "data": str(e)}) + "\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


# ====== Reflective Chat ======

@router.post("/reflective")
async def chat_reflective(request: ReflectiveChatRequest):
    """
    🪞 Reflective Chat (v4.0)
    
    Chat with self-reflection capabilities for improved reasoning.
    The AI reflects on its response before finalizing.
    """
    try:
        llm_provider = get_llm_provider()
        reflective_generator = get_reflective_generator()
        
        if not llm_provider:
            raise HTTPException(status_code=503, detail="LLM provider not available")
        
        # Use reflective generator if available
        if reflective_generator and request.enable_reflection:
            result = await reflective_generator.generate_with_reflection(
                prompt=request.message,
                reflection_depth=request.reflection_depth
            )
            
            return {
                "status": "success",
                "response": result.get("final_response", ""),
                "reflections": result.get("reflections", []),
                "reflection_count": len(result.get("reflections", [])),
                "improved": result.get("improved", False),
                "conversation_id": request.conversation_id,
                "timestamp": time.time()
            }
        else:
            # Fallback to simple chat
            messages = [
                {"role": "system", "content": "You are NIS Protocol v4.0. Be helpful and accurate."},
                {"role": "user", "content": request.message}
            ]
            
            result = await llm_provider.generate_response(messages=messages, temperature=0.7)
            
            return {
                "status": "success",
                "response": result.get("content", ""),
                "reflections": [],
                "reflection_count": 0,
                "improved": False,
                "conversation_id": request.conversation_id,
                "timestamp": time.time()
            }
            
    except Exception as e:
        logger.error(f"Reflective chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reflective/metrics")
async def get_reflective_metrics():
    """
    📊 Get Reflective Chat Metrics
    
    Returns metrics about reflective chat usage and improvements.
    """
    try:
        reflective_generator = get_reflective_generator()
        
        if reflective_generator:
            metrics = reflective_generator.get_metrics()
            return {
                "status": "success",
                "metrics": metrics,
                "timestamp": time.time()
            }
        else:
            return {
                "status": "success",
                "metrics": {
                    "total_reflections": 0,
                    "improvement_rate": 0.0,
                    "avg_reflection_depth": 0
                },
                "reflective_generator_available": False,
                "timestamp": time.time()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====== Optimized Chat ======

@router.post("/optimized")
async def chat_optimized(request: SimpleChatRequest):
    """
    ⚡ Optimized Chat Endpoint
    
    Uses query routing to optimize response generation.
    """
    try:
        llm_provider = get_llm_provider()
        
        if not llm_provider:
            raise HTTPException(status_code=503, detail="LLM provider not available")
        
        messages = [
            {"role": "system", "content": "You are NIS Protocol v4.0. Provide concise, accurate responses."},
            {"role": "user", "content": request.message}
        ]
        
        result = await llm_provider.generate_response(
            messages=messages,
            temperature=0.5,  # Lower temperature for more focused responses
            requested_provider=None
        )
        
        return {
            "status": "success",
            "response": result.get("content", ""),
            "provider": result.get("provider", "unknown"),
            "model": result.get("model", "unknown"),
            "tokens_used": result.get("tokens_used", 0),
            "optimized": True,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Optimized chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====== Dependency Injection Helper ======

def set_dependencies(llm_provider=None, reflective_generator=None):
    """Set dependencies for the chat router"""
    router._llm_provider = llm_provider
    router._reflective_generator = reflective_generator

"""
NIS Protocol v4.0 - Reasoning Routes

This module contains reasoning-related endpoints:
- Collaborative reasoning (multi-model)
- Debate reasoning
- Multimodal agent status

MIGRATION STATUS: Ready for testing
- These routes mirror the ones in main.py
- Can be tested independently before switching over
- main.py routes remain active until migration is complete

Usage:
    from routes.reasoning import router as reasoning_router
    app.include_router(reasoning_router, tags=["Reasoning"])
"""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("nis.routes.reasoning")

# Create router
router = APIRouter(tags=["Reasoning"])


# ====== Request Models ======

class ReasoningRequest(BaseModel):
    """Request model for collaborative reasoning"""
    problem: str = Field(..., description="The problem to reason about")
    reasoning_type: Optional[str] = Field(default=None, description="Type: analytical, creative, mathematical, scientific, ethical")
    depth: int = Field(default=2, ge=1, le=5, description="Reasoning depth (1-5)")
    require_consensus: bool = Field(default=True, description="Require model consensus")
    max_iterations: int = Field(default=3, ge=1, le=10, description="Maximum reasoning iterations")


class DebateRequest(BaseModel):
    """Request model for debate reasoning"""
    problem: str = Field(..., description="The topic/problem to debate")
    positions: Optional[List[str]] = Field(default=None, description="Initial positions (auto-generated if not provided)")
    rounds: int = Field(default=3, ge=1, le=5, description="Number of debate rounds")


# ====== Dependency Injection ======

def get_reasoning_chain():
    return getattr(router, '_reasoning_chain', None)

def get_vision_agent():
    return getattr(router, '_vision_agent', None)

def get_research_agent():
    return getattr(router, '_research_agent', None)

def get_document_agent():
    return getattr(router, '_document_agent', None)


# ====== Endpoints ======

@router.post("/reasoning/collaborative")
async def collaborative_reasoning(request: ReasoningRequest):
    """
    🧠 Perform collaborative reasoning with multiple models
    
    Features:
    - Chain-of-thought reasoning across multiple models
    - Model specialization for different problem types
    - Cross-validation and error checking
    - Metacognitive reasoning about reasoning quality
    """
    try:
        reasoning_chain = get_reasoning_chain()
        
        if not reasoning_chain:
            raise HTTPException(status_code=500, detail="Reasoning chain not initialized")
        
        # Convert string reasoning type to enum if provided
        reasoning_type = None
        if request.reasoning_type:
            try:
                from src.agents.reasoning.enhanced_reasoning_chain import ReasoningType
                reasoning_type = ReasoningType(request.reasoning_type)
            except (ImportError, ValueError) as e:
                logger.warning(f"Could not convert reasoning type: {e}")
        
        result = await reasoning_chain.collaborative_reasoning(
            problem=request.problem,
            reasoning_type=reasoning_type,
            depth=request.depth,
            require_consensus=request.require_consensus,
            max_iterations=request.max_iterations
        )
        
        return {
            "status": "success",
            "reasoning": result,
            "agent_id": reasoning_chain.agent_id,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Collaborative reasoning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Collaborative reasoning failed: {str(e)}")


@router.post("/reasoning/debate")
async def debate_reasoning(request: DebateRequest):
    """
    🗣️ Conduct structured debate between models to reach better conclusions
    
    Capabilities:
    - Multi-model debate and discussion
    - Position generation and refinement
    - Consensus building through argumentation
    - Disagreement analysis and resolution
    """
    try:
        reasoning_chain = get_reasoning_chain()
        
        if not reasoning_chain:
            raise HTTPException(status_code=500, detail="Reasoning chain not initialized")
        
        result = await reasoning_chain.debate_reasoning(
            problem=request.problem,
            positions=request.positions,
            rounds=request.rounds
        )
        
        return {
            "status": "success",
            "debate": result,
            "agent_id": reasoning_chain.agent_id,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Debate reasoning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Debate reasoning failed: {str(e)}")


@router.get("/agents/multimodal/status")
async def get_multimodal_status():
    """
    📋 Get status of all multimodal agents
    
    Returns detailed status information for:
    - Vision Agent capabilities and providers
    - Research Agent sources and tools
    - Document Analysis Agent processing capabilities
    - Enhanced Reasoning Chain model coordination
    - Current system performance metrics
    """
    try:
        vision_agent = get_vision_agent()
        research_agent = get_research_agent()
        reasoning_chain = get_reasoning_chain()
        document_agent = get_document_agent()
        
        vision_status = await vision_agent.get_status() if vision_agent else {"status": "not_initialized"}
        research_status = await research_agent.get_status() if research_agent else {"status": "not_initialized"}
        reasoning_status = await reasoning_chain.get_status() if reasoning_chain else {"status": "not_initialized"}
        document_status = await document_agent.get_status() if document_agent else {"status": "not_initialized"}
        
        return {
            "status": "operational",
            "version": "3.2",
            "multimodal_capabilities": {
                "vision": vision_status,
                "research": research_status,
                "reasoning": reasoning_status,
                "document": document_status
            },
            "enhanced_features": [
                "Image analysis with physics focus",
                "Academic paper research",
                "Claim validation with evidence",
                "Scientific visualization generation",
                "Multi-source fact checking",
                "Knowledge graph construction",
                "Advanced document processing",
                "Multi-model collaborative reasoning",
                "Structured debate and consensus building",
                "PDF extraction and analysis",
                "Citation and reference analysis"
            ],
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Multimodal status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


# ====== Dependency Injection Helper ======

def set_dependencies(
    reasoning_chain=None,
    vision_agent=None,
    research_agent=None,
    document_agent=None
):
    """Set dependencies for the reasoning router"""
    router._reasoning_chain = reasoning_chain
    router._vision_agent = vision_agent
    router._research_agent = research_agent
    router._document_agent = document_agent

"""
NIS Protocol v4.0 - Humanoid Robot Routes (GR00T N1)

This module contains humanoid robot control endpoints using NVIDIA Isaac GR00T N1:
- High-level task execution
- Natural language control
- Whole-body motion planning
- Real-time humanoid control

Usage:
    from routes.humanoid import router as humanoid_router
    app.include_router(humanoid_router, tags=["Humanoid"])
"""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("nis.routes.humanoid")

# Create router
router = APIRouter(prefix="/humanoid", tags=["Humanoid Robots (GR00T)"])


# ====== Request Models ======

class TaskExecutionRequest(BaseModel):
    task: str = Field(..., description="Natural language task description")
    visual_context: Optional[str] = Field(default=None, description="Base64 encoded image")
    timeout: float = Field(default=30.0, description="Execution timeout in seconds")


class MotionRequest(BaseModel):
    motion_type: str = Field(..., description="Type of motion (walk, reach, grasp, etc.)")
    parameters: Dict[str, Any] = Field(default={}, description="Motion parameters")
    duration: float = Field(default=3.0, description="Motion duration in seconds")


# ====== Task Execution Endpoints ======

@router.post("/execute_task")
async def execute_humanoid_task(request: TaskExecutionRequest):
    """
    🤖 Execute High-Level Humanoid Task
    
    Uses GR00T N1 foundation model to execute complex humanoid tasks:
    - Natural language understanding
    - Vision-based perception
    - Whole-body motion planning
    - Real-time execution
    
    Example: "Walk to the table and pick up the cup"
    """
    try:
        import numpy as np
        from src.agents.groot import get_groot_agent
        
        agent = get_groot_agent()
        
        if not agent.initialized:
            await agent.initialize()
        
        # Mock visual input if not provided
        visual_input = None
        if request.visual_context:
            # In production, decode base64 image
            visual_input = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = await agent.execute_task(
            task=request.task,
            visual_input=visual_input
        )
        
        return {
            "status": "success" if result.get("success") else "failed",
            "action_sequence": result.get("action_sequence", []),
            "execution_time": result.get("execution_time", 0.0),
            "confidence": result.get("confidence", 0.0),
            "reasoning": result.get("reasoning", ""),
            "fallback": result.get("fallback", False),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Task execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/motion/execute")
async def execute_motion(request: MotionRequest):
    """
    🏃 Execute Specific Motion
    
    Execute a specific humanoid motion:
    - walk_forward, walk_backward
    - turn_left, turn_right
    - reach, grasp, release
    - stand, sit, crouch
    """
    try:
        from src.agents.groot import get_groot_agent
        
        agent = get_groot_agent()
        
        if not agent.initialized:
            await agent.initialize()
        
        # Convert motion type to task description
        task = f"Execute {request.motion_type} motion"
        if request.parameters:
            task += f" with parameters: {request.parameters}"
        
        result = await agent.execute_task(task=task)
        
        return {
            "status": "success" if result.get("success") else "failed",
            "motion_type": request.motion_type,
            "executed_actions": result.get("action_sequence", []),
            "duration": request.duration,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Motion execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====== Status and Capabilities ======

@router.get("/capabilities")
async def get_humanoid_capabilities():
    """
    📋 Get Humanoid Capabilities
    
    Returns the capabilities of the GR00T-powered humanoid system.
    """
    try:
        from src.agents.groot import get_groot_agent
        
        agent = get_groot_agent()
        
        if not agent.initialized:
            await agent.initialize()
        
        capabilities = await agent.get_capabilities()
        
        return {
            "status": "active",
            "capabilities": capabilities,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Capabilities error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_humanoid_stats():
    """
    📊 Get Humanoid Execution Statistics
    
    Returns statistics about humanoid task execution.
    """
    try:
        from src.agents.groot import get_groot_agent
        
        agent = get_groot_agent()
        stats = agent.get_stats()
        
        return {
            "status": "active",
            "stats": stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/initialize")
async def initialize_humanoid():
    """
    🔧 Initialize Humanoid System
    
    Initializes the GR00T N1 humanoid control system.
    """
    try:
        from src.agents.groot import get_groot_agent
        
        agent = get_groot_agent()
        success = await agent.initialize()
        
        return {
            "status": "success" if success else "failed",
            "message": "GR00T N1 humanoid system initialized",
            "model_available": agent._model is not None,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====== Demo Scenarios ======

@router.post("/demo/pick_and_place")
async def demo_pick_and_place():
    """
    🎬 Demo: Pick and Place
    
    Demonstrates humanoid pick and place capability.
    """
    try:
        from src.agents.groot import get_groot_agent
        
        agent = get_groot_agent()
        
        if not agent.initialized:
            await agent.initialize()
        
        # Execute pick and place sequence
        pick_result = await agent.execute_task(
            task="Walk to the object and pick it up"
        )
        
        if pick_result.get("success"):
            place_result = await agent.execute_task(
                task="Walk to the target location and place the object"
            )
            
            return {
                "status": "success",
                "pick_phase": pick_result,
                "place_phase": place_result,
                "total_time": (
                    pick_result.get("execution_time", 0) +
                    place_result.get("execution_time", 0)
                ),
                "timestamp": time.time()
            }
        else:
            return {
                "status": "failed",
                "error": "Pick phase failed",
                "pick_phase": pick_result,
                "timestamp": time.time()
            }
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/demo/navigation")
async def demo_navigation():
    """
    🎬 Demo: Navigation
    
    Demonstrates humanoid navigation capability.
    """
    try:
        from src.agents.groot import get_groot_agent
        
        agent = get_groot_agent()
        
        if not agent.initialized:
            await agent.initialize()
        
        result = await agent.execute_task(
            task="Walk forward 3 meters, turn right, then walk forward 2 meters"
        )
        
        return {
            "status": "success" if result.get("success") else "failed",
            "navigation_result": result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

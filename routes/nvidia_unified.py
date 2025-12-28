"""
Unified NVIDIA Stack Routes

Single endpoint for accessing all NVIDIA capabilities through
the integrated NIS Protocol interface.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("nis.routes.nvidia_unified")

# Create router
router = APIRouter(prefix="/nvidia", tags=["NVIDIA Stack (Unified)"])


# ====== Request Models ======

class FullPipelineRequest(BaseModel):
    goal: str = Field(..., description="High-level goal description")
    robot_type: str = Field(default="humanoid", description="Robot type (humanoid, manipulator, etc.)")
    constraints: List[str] = Field(default=[], description="Constraints and requirements")


# ====== Unified Endpoints ======

@router.get("/status")
async def get_nvidia_status():
    """
    📊 Get NVIDIA Stack Status
    
    Returns status of all NVIDIA components integrated into NIS Protocol.
    """
    try:
        from src.core.nvidia_integration import get_nvidia_integration
        
        nvidia = get_nvidia_integration()
        status = nvidia.get_status()
        capabilities = nvidia.get_capabilities()
        
        return {
            "status": "active" if status["initialized"] else "initializing",
            "components": status["components"],
            "capabilities": capabilities,
            "stats": status["stats"],
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Status error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }


@router.post("/initialize")
async def initialize_nvidia_stack():
    """
    🔧 Initialize NVIDIA Stack
    
    Initializes all NVIDIA components within NIS Protocol.
    """
    try:
        from src.core.nvidia_integration import initialize_nvidia_stack
        
        nvidia = await initialize_nvidia_stack()
        status = nvidia.get_status()
        
        return {
            "status": "success",
            "message": "NVIDIA Stack fully integrated into NIS Protocol",
            "components_initialized": status["stats"]["components_initialized"],
            "components": status["components"],
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute")
async def execute_full_pipeline(request: FullPipelineRequest):
    """
    🚀 Execute Full NVIDIA Pipeline
    
    Executes complete pipeline through NIS Protocol:
    1. Reason about goal (Cosmos)
    2. Plan execution strategy
    3. Train/validate if needed (Isaac Lab + NIS Physics)
    4. Execute on robot (GR00T/Isaac)
    
    This is the main entry point for NVIDIA stack operations.
    """
    try:
        from src.core.nvidia_integration import get_nvidia_integration
        
        nvidia = get_nvidia_integration()
        
        if not nvidia.initialized:
            await nvidia.initialize()
        
        logger.info(f"Executing NVIDIA pipeline for: {request.goal}")
        
        result = await nvidia.execute_full_pipeline(
            goal=request.goal,
            robot_type=request.robot_type
        )
        
        return {
            "status": "success" if result.get("success") else "failed",
            "goal": request.goal,
            "robot_type": request.robot_type,
            "stages": result.get("stages", {}),
            "result": result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Pipeline execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities")
async def get_nvidia_capabilities():
    """
    📋 Get NVIDIA Stack Capabilities
    
    Returns all available capabilities from the integrated NVIDIA stack.
    """
    try:
        from src.core.nvidia_integration import get_nvidia_integration
        
        nvidia = get_nvidia_integration()
        capabilities = nvidia.get_capabilities()
        
        return {
            "status": "success",
            "capabilities": capabilities["capabilities"],
            "available_features": capabilities["available_features"],
            "total_features": capabilities["total_features"],
            "description": {
                "data_generation": "Cosmos - Synthetic training data generation",
                "reasoning": "Cosmos Reason - Vision-language reasoning",
                "humanoid_control": "GR00T N1 - Humanoid robot control",
                "robot_learning": "Isaac Lab 2.2 - GPU-accelerated RL training",
                "simulation": "Isaac Sim/ROS - Physics simulation"
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Capabilities error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_nvidia_stats():
    """
    📈 Get NVIDIA Stack Statistics
    
    Returns usage statistics for all NVIDIA components.
    """
    try:
        from src.core.nvidia_integration import get_nvidia_integration
        
        nvidia = get_nvidia_integration()
        status = nvidia.get_status()
        
        # Get individual component stats
        from src.agents.cosmos import get_cosmos_generator, get_cosmos_reasoner
        from src.agents.groot import get_groot_agent
        from src.agents.isaac_lab import get_isaac_lab_trainer
        
        component_stats = {
            "cosmos_generator": get_cosmos_generator().get_stats(),
            "cosmos_reasoner": get_cosmos_reasoner().get_stats(),
            "groot_agent": get_groot_agent().get_stats(),
            "isaac_lab_trainer": get_isaac_lab_trainer().get_stats()
        }
        
        return {
            "status": "success",
            "overall_stats": status["stats"],
            "component_stats": component_stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

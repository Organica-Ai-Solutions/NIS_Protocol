"""
NIS Protocol v4.0 - Isaac Lab Routes

This module contains Isaac Lab 2.2 integration endpoints:
- Robot policy training (RL/IL)
- Multi-robot environments
- Policy export and deployment
- Integration with NIS physics validation

Usage:
    from routes.isaac_lab import router as isaac_lab_router
    app.include_router(isaac_lab_router, tags=["Isaac Lab"])
"""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("nis.routes.isaac_lab")

# Create router
router = APIRouter(prefix="/isaac_lab", tags=["Isaac Lab 2.2"])


# ====== Request Models ======

class TrainingRequest(BaseModel):
    robot_type: str = Field(..., description="Robot model to train")
    task: str = Field(..., description="Task to train on")
    num_iterations: int = Field(default=1000, description="Training iterations")
    algorithm: str = Field(default="PPO", description="RL algorithm (PPO, SAC, TD3)")
    num_envs: int = Field(default=4096, description="Parallel environments")


class PolicyExportRequest(BaseModel):
    policy_id: str = Field(..., description="Policy identifier")
    format: str = Field(default="onnx", description="Export format")
    path: Optional[str] = Field(default=None, description="Export path")


class ValidationRequest(BaseModel):
    policy_id: str = Field(..., description="Policy to validate")
    test_scenarios: List[Dict[str, Any]] = Field(default=[], description="Test scenarios")


# ====== Training Endpoints ======

@router.post("/train")
async def train_policy(request: TrainingRequest):
    """
    🎓 Train Robot Policy
    
    Train a robot policy using Isaac Lab 2.2:
    - GPU-accelerated (4096+ parallel environments)
    - Multiple RL algorithms (PPO, SAC, TD3)
    - 16+ robot models available
    - 30+ tasks available
    
    Example: Train Franka Panda for pick-and-place
    """
    try:
        from src.agents.isaac_lab import get_isaac_lab_trainer
        
        trainer = get_isaac_lab_trainer()
        
        if not trainer.initialized:
            await trainer.initialize()
        
        logger.info(f"Training {request.robot_type} on {request.task}")
        
        result = await trainer.train_policy(
            robot_type=request.robot_type,
            task=request.task,
            num_iterations=request.num_iterations,
            algorithm=request.algorithm
        )
        
        return {
            "status": "success" if result.get("success") else "failed",
            "policy": result.get("policy", {}),
            "best_reward": result.get("best_reward", 0),
            "training_time": result.get("training_time", 0),
            "episodes": result.get("episodes", 0),
            "algorithm": result.get("algorithm"),
            "fallback": result.get("fallback", False),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_policy(request: PolicyExportRequest):
    """
    📦 Export Trained Policy
    
    Export policy for deployment:
    - ONNX format (cross-platform)
    - TorchScript (PyTorch)
    - Ready for edge deployment
    """
    try:
        from src.agents.isaac_lab import get_isaac_lab_trainer
        
        trainer = get_isaac_lab_trainer()
        
        # Get policy from cache
        policy = trainer._policies.get(request.policy_id)
        
        if not policy:
            raise HTTPException(status_code=404, detail="Policy not found")
        
        result = await trainer.export_policy(
            policy=policy,
            format=request.format,
            path=request.path
        )
        
        return {
            "status": "success" if result.get("success") else "failed",
            "format": result.get("format"),
            "path": result.get("path"),
            "size_mb": result.get("size_mb", 0),
            "message": result.get("message"),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate")
async def validate_policy(request: ValidationRequest):
    """
    ✅ Validate Policy with NIS Physics
    
    Validate trained policy using NIS physics validation:
    - PINN physics checks
    - Safety validation
    - Performance metrics
    """
    try:
        from src.agents.isaac_lab import get_isaac_lab_trainer
        
        trainer = get_isaac_lab_trainer()
        
        # Get policy
        policy = trainer._policies.get(request.policy_id)
        
        if not policy:
            raise HTTPException(status_code=404, detail="Policy not found")
        
        result = await trainer.validate_policy_with_nis(
            policy=policy,
            test_scenarios=request.test_scenarios
        )
        
        return {
            "status": "success" if result.get("success") else "failed",
            "validation_results": result.get("validation_results", []),
            "success_rate": result.get("success_rate", 0),
            "physics_validated": result.get("physics_validated", False),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====== Information Endpoints ======

@router.get("/robots")
async def get_available_robots():
    """
    🤖 Get Available Robot Models
    
    Returns list of robot models available for training.
    """
    try:
        from src.agents.isaac_lab import get_isaac_lab_trainer
        
        trainer = get_isaac_lab_trainer()
        robots = trainer.get_available_robots()
        
        return {
            "status": "success",
            "robots": robots,
            "count": len(robots),
            "categories": {
                "manipulators": 4,
                "quadrupeds": 4,
                "humanoids": 3,
                "mobile_manipulators": 2
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks")
async def get_available_tasks():
    """
    📋 Get Available Training Tasks
    
    Returns list of tasks available for training.
    """
    try:
        from src.agents.isaac_lab import get_isaac_lab_trainer
        
        trainer = get_isaac_lab_trainer()
        tasks = trainer.get_available_tasks()
        
        return {
            "status": "success",
            "tasks": tasks,
            "count": len(tasks),
            "categories": {
                "manipulation": 4,
                "locomotion": 3,
                "complex": 3
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_training_stats():
    """
    📊 Get Training Statistics
    
    Returns statistics about Isaac Lab training sessions.
    """
    try:
        from src.agents.isaac_lab import get_isaac_lab_trainer
        
        trainer = get_isaac_lab_trainer()
        stats = trainer.get_stats()
        
        return {
            "status": "active",
            "stats": stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/initialize")
async def initialize_isaac_lab():
    """
    🔧 Initialize Isaac Lab
    
    Initializes the Isaac Lab 2.2 training system.
    """
    try:
        from src.agents.isaac_lab import get_isaac_lab_trainer
        
        trainer = get_isaac_lab_trainer()
        success = await trainer.initialize()
        
        return {
            "status": "success" if success else "failed",
            "message": "Isaac Lab 2.2 initialized",
            "available_robots": len(trainer.get_available_robots()),
            "available_tasks": len(trainer.get_available_tasks()),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

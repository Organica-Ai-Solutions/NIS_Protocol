"""
NIS Protocol v4.0 - BitNet Routes

This module contains all BitNet training and model endpoints:
- Training status
- Force training
- Training metrics
- Model download
- Training data export
- Data persistence

MIGRATION STATUS: Ready for testing
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("nis.routes.bitnet")

# Create router - note: we use empty prefix since paths vary between /models/bitnet and /training/bitnet
router = APIRouter(tags=["BitNet"])


# ====== Request/Response Models ======

class ForceTrainingRequest(BaseModel):
    reason: str = Field(default="Manual training trigger", description="Reason for forcing training")


class TrainingStatusResponse(BaseModel):
    is_training: bool
    training_available: bool
    total_examples: int
    unused_examples: int
    offline_readiness_score: float
    metrics: Dict[str, Any]
    config: Dict[str, Any]
    supports_offline: bool
    download_url: Optional[str] = None
    download_checksum: Optional[str] = None
    download_size_mb: Optional[float] = None
    version: Optional[str] = None
    model_variant: Optional[str] = None
    lora_available: bool = False
    last_updated: Optional[str] = None


# ====== Helper to get BitNet trainer ======
# Note: In actual use, this will be injected from main.py

def get_bitnet_trainer():
    """Get the global bitnet_trainer instance - must be set by main app"""
    # This will be set when router is included
    return getattr(router, '_bitnet_trainer', None)


def set_bitnet_trainer(trainer):
    """Set the bitnet_trainer instance for this router"""
    router._bitnet_trainer = trainer


# ====== Endpoints ======

@router.get("/bitnet/status")
async def get_bitnet_status_simple():
    """
    🎯 Get BitNet Status (Simple endpoint)
    
    Quick status check for BitNet model availability.
    """
    bitnet_trainer = get_bitnet_trainer()
    
    if not bitnet_trainer:
        return {
            "status": "disabled",
            "training_available": False,
            "offline_ready": False,
            "message": "BitNet trainer not initialized"
        }
    
    try:
        status = await bitnet_trainer.get_training_status()
        return {
            "status": "active",
            "training_available": status["training_available"],
            "is_training": status["is_training"],
            "total_examples": status["total_examples"],
            "offline_readiness_score": status["metrics"].get("offline_readiness_score", 0.0),
            "offline_ready": status["metrics"].get("offline_readiness_score", 0.0) >= 0.5
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "training_available": False
        }


@router.get("/training/bitnet/status")
async def get_bitnet_training_status():
    """
    🎯 Get BitNet Training Status
    
    Quick training status check.
    """
    bitnet_trainer = get_bitnet_trainer()
    
    if not bitnet_trainer:
        return {
            "status": "disabled",
            "is_training": False,
            "training_available": False,
            "message": "BitNet trainer not initialized"
        }
    
    try:
        status = await bitnet_trainer.get_training_status()
        return {
            "status": "active",
            "is_training": status["is_training"],
            "training_available": status["training_available"],
            "total_examples": status["total_examples"],
            "unused_examples": status["unused_examples"],
            "last_training_time": status["metrics"].get("last_training_time"),
            "next_training_time": status["metrics"].get("next_training_time")
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "is_training": False
        }


@router.get("/models/bitnet/status", response_model=TrainingStatusResponse)
async def get_bitnet_model_status(request: Request):
    """
    🎯 Get BitNet Online Training Status
    
    Monitor the real-time training status of BitNet models including:
    - Current training activity
    - Training examples collected
    - Offline readiness score
    - Training metrics and configuration
    """
    bitnet_trainer = get_bitnet_trainer()
    
    if not bitnet_trainer:
        logger.info("BitNet trainer not initialized, returning mock status")
        return TrainingStatusResponse(
            is_training=False,
            training_available=False,
            total_examples=0,
            unused_examples=0,
            offline_readiness_score=0.0,
            metrics={
                "offline_readiness_score": 0.0,
                "total_training_sessions": 0,
                "last_training_time": None,
                "model_version": "mock_v1.0",
                "training_quality_avg": 0.0
            },
            config={
                "model_path": "models/bitnet/models/bitnet",
                "learning_rate": 1e-5,
                "training_interval_seconds": 300.0,
                "min_examples_before_training": 5,
                "quality_threshold": 0.6,
                "status": "disabled"
            },
            supports_offline=os.path.exists(os.path.join("models", "bitnet", "models", "bitnet")),
            download_url=None,
            download_checksum=None,
            download_size_mb=None,
            version="mock_v1.0",
            model_variant="b1.58-2B",
            lora_available=False,
            last_updated=None
        )
    
    try:
        status = await bitnet_trainer.get_training_status()

        mobile_bundle_config = status.get("mobile_bundle", {})
        bitnet_dir = status["config"].get("model_path", "models/bitnet/models/bitnet")
        bundle_path = mobile_bundle_config.get("path")
        bundle_size = None
        if bundle_path and os.path.exists(bundle_path):
            bundle_size = round(os.path.getsize(bundle_path) / (1024 * 1024), 2)

        download_url = mobile_bundle_config.get("download_url")
        if not download_url and bundle_path and os.path.exists(bundle_path):
             bundle_filename = os.path.basename(bundle_path)
             base_url = str(request.base_url).rstrip('/')
             download_url = f"{base_url}/downloads/bitnet/{bundle_filename}"

        return TrainingStatusResponse(
            is_training=status["is_training"],
            training_available=status["training_available"],
            total_examples=status["total_examples"],
            unused_examples=status["unused_examples"],
            offline_readiness_score=status["metrics"].get("offline_readiness_score", 0.0),
            metrics=status["metrics"],
            config=status["config"],
            supports_offline=os.path.exists(bitnet_dir),
            download_url=download_url,
            download_checksum=mobile_bundle_config.get("checksum"),
            download_size_mb=mobile_bundle_config.get("size_mb", bundle_size),
            version=mobile_bundle_config.get("version"),
            model_variant=mobile_bundle_config.get("variant"),
            lora_available=mobile_bundle_config.get("lora_available", False),
            last_updated=status.get("metrics", {}).get("last_training_time")
        )
        
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {str(e)}")


@router.post("/training/bitnet/force")
async def force_bitnet_training(request: ForceTrainingRequest):
    """
    🚀 Force BitNet Training Session
    
    Manually trigger an immediate BitNet training session with current examples.
    Useful for testing and immediate model improvement.
    """
    bitnet_trainer = get_bitnet_trainer()
    
    if not bitnet_trainer:
        logger.info("BitNet trainer not initialized, returning mock training response")
        return JSONResponse(content={
            "status": "disabled",
            "message": "BitNet training is currently disabled",
            "training_triggered": False,
            "reason": request.reason,
            "timestamp": datetime.now().isoformat(),
            "mock_response": True
        }, status_code=200)
    
    try:
        logger.info(f"🎯 Manual training session requested: {request.reason}")
        
        result = await bitnet_trainer.force_training_session()
        
        return JSONResponse(content={
            "success": result["success"],
            "message": result["message"],
            "reason": request.reason,
            "timestamp": datetime.now().isoformat(),
            "metrics": result.get("metrics", {}),
            "training_triggered": result["success"]
        }, status_code=200 if result["success"] else 400)
        
    except Exception as e:
        logger.error(f"Error forcing training session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to force training: {str(e)}")


@router.get("/training/bitnet/metrics")
async def get_detailed_training_metrics():
    """
    📊 Get Detailed BitNet Training Metrics
    
    Comprehensive training metrics including:
    - Training history and performance
    - Model improvement scores
    - Quality assessment statistics
    - Offline readiness analysis
    """
    bitnet_trainer = get_bitnet_trainer()
    
    try:
        if not bitnet_trainer:
            logger.info("BitNet trainer not initialized, returning mock metrics")
            return {
                "status": "success",
                "training_available": False,
                "training_metrics": {
                    "offline_readiness_score": 0.0,
                    "total_training_sessions": 0,
                    "last_training_session": None,
                    "average_quality_score": 0.0,
                    "total_model_updates": 0
                },
                "efficiency_metrics": {
                    "examples_per_session": 0.0,
                    "training_frequency_minutes": 30.0,
                    "quality_threshold": 0.7
                },
                "offline_readiness": {
                    "score": 0.0,
                    "status": "Initializing",
                    "estimated_ready": False,
                    "recommendations": ["BitNet training not available in this deployment"]
                }
            }
    
        status = await bitnet_trainer.get_training_status()
        
        metrics = status["metrics"].copy()
        total_sessions = metrics.get("total_training_sessions", 0)
        total_examples = status["total_examples"]
        efficiency = total_examples / max(total_sessions, 1)
        
        readiness_score = metrics.get("offline_readiness_score", 0.0)
        readiness_status = "Ready" if readiness_score >= 0.8 else "Training" if readiness_score >= 0.5 else "Initializing"
        
        return {
            "status": "success",
            "training_available": True,
            "training_metrics": metrics,
            "efficiency_metrics": {
                "examples_per_session": efficiency,
                "training_frequency_minutes": status["config"]["training_interval_seconds"] / 60,
                "quality_threshold": status["config"]["quality_threshold"]
            },
            "offline_readiness": {
                "score": readiness_score,
                "status": readiness_status,
                "estimated_ready": readiness_score >= 0.8,
                "recommendations": [
                    "Continue conversation interactions" if readiness_score < 0.5 else "Ready for offline deployment",
                    f"Need {max(0, int((0.8 - readiness_score) * 500))} more quality examples" if readiness_score < 0.8 else "Training complete"
                ]
            },
            "system_info": {
                "training_available": status["training_available"],
                "total_examples": total_examples,
                "unused_examples": status["unused_examples"],
                "last_update": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        import traceback
        logger.error(f"Error getting detailed metrics: {e}")
        return {
            "status": "error",
            "error": str(e),
            "training_available": False,
            "traceback": traceback.format_exc()
        }


@router.get("/models/bitnet/download")
async def download_bitnet_model():
    """
    📥 Download Trained BitNet Model
    
    Returns information about the downloadable BitNet model bundle.
    The model can be used for offline local inference.
    """
    bitnet_trainer = get_bitnet_trainer()
    
    try:
        if bitnet_trainer:
            status = await bitnet_trainer.get_training_status()
            mobile_bundle = status.get("mobile_bundle", {})
            
            if mobile_bundle.get("path") and os.path.exists(mobile_bundle["path"]):
                return {
                    "status": "available",
                    "download_url": f"/downloads/bitnet/{os.path.basename(mobile_bundle['path'])}",
                    "filename": os.path.basename(mobile_bundle["path"]),
                    "size_mb": mobile_bundle.get("size_mb"),
                    "checksum": mobile_bundle.get("checksum"),
                    "version": mobile_bundle.get("version"),
                    "variant": mobile_bundle.get("variant"),
                    "training_examples": status.get("total_examples", 0),
                    "quality_score": status.get("metrics", {}).get("average_quality_score", 0),
                    "offline_ready": status.get("metrics", {}).get("offline_readiness_score", 0) >= 0.5
                }
        
        # Check for any existing bundles
        bundle_dir = "models/bitnet/mobile"
        if os.path.exists(bundle_dir):
            bundles = [f for f in os.listdir(bundle_dir) if f.endswith('.zip')]
            if bundles:
                latest = sorted(bundles)[-1]
                bundle_path = os.path.join(bundle_dir, latest)
                size_mb = round(os.path.getsize(bundle_path) / (1024 * 1024), 2)
                return {
                    "status": "available",
                    "download_url": f"/downloads/bitnet/{latest}",
                    "filename": latest,
                    "size_mb": size_mb,
                    "note": "Pre-existing bundle (may not include latest training)"
                }
        
        return {
            "status": "not_ready",
            "message": "BitNet model bundle not yet available",
            "training_examples": bitnet_trainer.training_metrics.get("total_examples_collected", 0) if bitnet_trainer else 0,
            "recommendation": "Continue training to generate downloadable model"
        }
        
    except Exception as e:
        logger.error(f"Error getting BitNet download info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/bitnet/export")
async def export_bitnet_training_data():
    """
    📤 Export BitNet Training Data
    
    Export all training examples as JSON for backup or transfer.
    """
    bitnet_trainer = get_bitnet_trainer()
    
    try:
        training_data_path = "data/bitnet_training/training_examples.json"
        
        if os.path.exists(training_data_path):
            with open(training_data_path, 'r') as f:
                examples = json.load(f)
            
            return {
                "status": "success",
                "total_examples": len(examples),
                "export_path": training_data_path,
                "domains": _categorize_training_examples(examples),
                "download_url": "/static/bitnet_training_export.json"
            }
        
        if bitnet_trainer:
            examples = [
                {
                    "prompt": ex.prompt,
                    "response": ex.response,
                    "quality_score": ex.quality_score
                }
                for ex in bitnet_trainer.training_examples
            ]
            return {
                "status": "success",
                "total_examples": len(examples),
                "note": "Data from memory (not yet persisted)"
            }
        
        return {"status": "no_data", "message": "No training data available"}
        
    except Exception as e:
        logger.error(f"Error exporting training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _categorize_training_examples(examples: list) -> dict:
    """Categorize training examples by domain"""
    categories = {
        "robotics": 0,
        "can_bus": 0,
        "physics": 0,
        "ai_ml": 0,
        "general": 0
    }
    
    robotics_keywords = ["robot", "kinematic", "trajectory", "manipulator", "servo", "actuator"]
    can_keywords = ["can bus", "can protocol", "ecu", "automotive", "j1939", "canopen"]
    physics_keywords = ["physics", "force", "energy", "momentum", "newton", "thermodynamic"]
    ai_keywords = ["neural", "machine learning", "deep learning", "ai", "model", "training"]
    
    for ex in examples:
        prompt_lower = ex.get("prompt", "").lower()
        if any(kw in prompt_lower for kw in robotics_keywords):
            categories["robotics"] += 1
        elif any(kw in prompt_lower for kw in can_keywords):
            categories["can_bus"] += 1
        elif any(kw in prompt_lower for kw in physics_keywords):
            categories["physics"] += 1
        elif any(kw in prompt_lower for kw in ai_keywords):
            categories["ai_ml"] += 1
        else:
            categories["general"] += 1
    
    return categories


@router.post("/models/bitnet/persist")
async def persist_bitnet_training():
    """
    💾 Persist BitNet Training Data
    
    Force save all training data to disk.
    """
    bitnet_trainer = get_bitnet_trainer()
    
    try:
        if bitnet_trainer:
            bitnet_trainer._persist_training_data()
            return {
                "status": "success",
                "message": "Training data persisted",
                "examples_saved": len(bitnet_trainer.training_examples),
                "path": str(bitnet_trainer.training_data_path)
            }
        return {"status": "error", "message": "BitNet trainer not initialized"}
    except Exception as e:
        logger.error(f"Error persisting training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

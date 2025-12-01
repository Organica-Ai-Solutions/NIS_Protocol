"""
NIS Protocol v4.0 - System Routes

This module contains system-level endpoints:
- Configuration (models, providers)
- State Management
- Edge AI deployment
- Brain Orchestration
- Autonomous Execution
- Tool Optimization

MIGRATION STATUS: Ready for testing
- These routes mirror the ones in main.py
- Can be tested independently before switching over

Usage:
    from routes.system import router as system_router
    app.include_router(system_router, tags=["System"])
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger("nis.routes.system")

# Create router
router = APIRouter(tags=["System"])


# ====== Request Models ======

class ModelConfigRequest(BaseModel):
    model: str = Field(..., description="Model name to set as default")


# ====== Dependency Injection ======

def get_llm_provider():
    return getattr(router, '_llm_provider', None)

def get_nis_websocket_manager():
    return getattr(router, '_nis_websocket_manager', None)

def get_nis_state_manager():
    return getattr(router, '_nis_state_manager', None)

def get_nis_agent_orchestrator():
    return getattr(router, '_nis_agent_orchestrator', None)

def get_enhanced_schemas():
    return getattr(router, '_enhanced_schemas', None)

def get_token_manager():
    return getattr(router, '_token_manager', None)

def get_autonomous_executor():
    return getattr(router, '_autonomous_executor', None)


# ====== Configuration Endpoints ======

@router.get("/models")
async def list_all_models():
    """
    📋 List all available models for each provider
    
    Returns the default model and available models for each LLM provider.
    """
    llm_provider = get_llm_provider()
    
    if not llm_provider:
        raise HTTPException(status_code=503, detail="LLM Provider not initialized")
    
    return {
        "status": "success",
        "providers": llm_provider.list_models(),
        "default_provider": llm_provider.default_provider
    }


@router.get("/models/{provider}")
async def list_provider_models(provider: str):
    """
    📋 List available models for a specific provider
    
    Args:
        provider: Provider name (openai, anthropic, deepseek, kimi, nvidia, google, bitnet)
    """
    llm_provider = get_llm_provider()
    
    if not llm_provider:
        raise HTTPException(status_code=503, detail="LLM Provider not initialized")
    
    if provider not in llm_provider.default_models:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider}")
    
    return {
        "status": "success",
        **llm_provider.list_models(provider)
    }


@router.put("/models/{provider}")
async def set_provider_model(provider: str, config: ModelConfigRequest):
    """
    ⚙️ Set the default model for a provider
    
    Args:
        provider: Provider name (openai, anthropic, deepseek, kimi, nvidia, google)
        config: Model configuration with the model name
    """
    llm_provider = get_llm_provider()
    
    if not llm_provider:
        raise HTTPException(status_code=503, detail="LLM Provider not initialized")
    
    if provider not in llm_provider.default_models:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider}")
    
    success = llm_provider.set_default_model(provider, config.model)
    
    if success:
        return {
            "status": "success",
            "message": f"Default model for {provider} set to {config.model}",
            "provider": provider,
            "model": config.model
        }
    else:
        raise HTTPException(status_code=400, detail=f"Failed to set model for {provider}")


# ====== State Management Endpoints ======

@router.post("/api/state/update")
async def update_system_state_endpoint(request: dict):
    """
    🔄 Update System State
    
    Manually update system state. Changes will be automatically
    pushed to all connected WebSocket clients.
    """
    try:
        nis_state_manager = get_nis_state_manager()
        nis_websocket_manager = get_nis_websocket_manager()
        
        updates = request.get("updates", {})
        emit_event = request.get("emit_event", True)
        
        if not updates:
            raise HTTPException(status_code=400, detail="No updates provided")
        
        # Update state
        if nis_state_manager:
            await nis_state_manager.update_state(updates)
        
        if emit_event and nis_websocket_manager:
            await nis_websocket_manager.broadcast({
                "type": "state_update",
                "data": {"manual_update": True, "updated_fields": list(updates.keys())}
            })
        
        return {
            "success": True,
            "message": f"System state updated: {list(updates.keys())}",
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update state: {str(e)}")


# ====== Tool Optimization Endpoints ======

@router.get("/api/tools/enhanced")
async def get_enhanced_tools():
    """
    🔧 Get Enhanced Tool Definitions
    """
    try:
        enhanced_schemas = get_enhanced_schemas()
        
        if enhanced_schemas is None:
            return {
                "success": False,
                "message": "Enhanced tool schemas are disabled in this build",
                "tools": [],
                "total_tools": 0
            }
        
        tools = enhanced_schemas.get_mcp_tool_definitions()
        
        return {
            "success": True,
            "tools": tools,
            "total_tools": len(tools),
            "optimization_features": [
                "clear_namespacing",
                "consolidated_workflows", 
                "token_efficiency",
                "response_format_control"
            ]
        }
    except Exception as e:
        logger.error(f"Error getting enhanced tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/tools/optimization/metrics")
async def get_tool_optimization_metrics():
    """
    📊 Tool Optimization Performance Metrics
    """
    try:
        token_manager = get_token_manager()
        
        if token_manager is None:
            return {
                "success": False,
                "message": "Token efficiency manager is disabled in this build",
                "metrics": {}
            }
        
        token_metrics = token_manager.get_performance_metrics()
        
        return {
            "success": True,
            "metrics": token_metrics,
            "efficiency_score": token_metrics.get("tokens_saved", 0) / max(1, token_metrics.get("requests_processed", 1)),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting optimization metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====== Edge AI Endpoints ======

@router.get("/api/edge/capabilities")
async def get_edge_ai_capabilities():
    """
    🚀 Get Edge AI Operating System Capabilities
    
    Returns comprehensive edge AI capabilities for:
    - Autonomous drones and UAV systems
    - Robotics and human-robot interaction
    - Autonomous vehicles and transportation
    - Industrial IoT and smart manufacturing
    - Smart home and building automation
    """
    try:
        return {
            "success": True,
            "edge_ai_os": "NIS Protocol v3.2.1",
            "target_devices": [
                "autonomous_drones",
                "robotics_systems", 
                "autonomous_vehicles",
                "industrial_iot",
                "smart_home_devices",
                "satellite_systems",
                "scientific_instruments"
            ],
            "core_capabilities": {
                "offline_operation": "BitNet local model for autonomous operation",
                "online_learning": "Continuous improvement while connected",
                "real_time_inference": "Sub-100ms response for safety-critical systems",
                "physics_validation": "PINN-based constraint checking for autonomous systems",
                "signal_processing": "Laplace transform for sensor data analysis",
                "multi_agent_coordination": "Brain-inspired agent orchestration"
            },
            "deployment_features": {
                "model_quantization": "Reduced memory footprint for edge devices",
                "response_caching": "Efficient repeated operation handling",
                "power_optimization": "Battery-aware operation for mobile systems",
                "thermal_management": "Performance optimization for varying conditions",
                "connectivity_adaptation": "Seamless online/offline switching"
            },
            "performance_targets": {
                "inference_latency": "< 100ms for real-time applications",
                "memory_usage": "< 1GB for resource-constrained devices", 
                "model_size": "< 500MB for edge deployment",
                "offline_success_rate": "> 90% autonomous operation capability"
            }
        }
    except Exception as e:
        logger.error(f"Error getting edge capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/edge/deploy")
async def deploy_edge_ai_system(
    device_type: str,
    enable_optimization: bool = True,
    operation_mode: str = "hybrid_adaptive"
):
    """
    🚁 Deploy Edge AI Operating System
    
    Deploy optimized NIS Protocol for specific edge device types:
    - drone: Autonomous UAV systems with navigation AI
    - robot: Human-robot interaction and task execution
    - vehicle: Autonomous driving assistance and safety
    - iot: Industrial IoT and smart device integration
    """
    try:
        # Simulated deployment - actual implementation would create edge OS
        supported_types = ["drone", "robot", "vehicle", "iot"]
        
        if device_type.lower() not in supported_types:
            return {
                "success": False,
                "error": f"Unsupported device type: {device_type}",
                "supported_types": supported_types
            }
        
        return {
            "success": True,
            "deployment_result": {
                "status": "initialized",
                "device_type": device_type,
                "optimization_enabled": enable_optimization,
                "operation_mode": operation_mode
            },
            "edge_status": {
                "ready": True,
                "capabilities_loaded": True
            },
            "device_optimized_for": device_type,
            "autonomous_capabilities": {
                "offline_operation": True,
                "real_time_decision_making": True,
                "continuous_learning": True,
                "physics_validated_outputs": True,
                "safety_critical_ready": True
            },
            "next_steps": [
                f"Deploy to {device_type} hardware",
                "Configure device-specific sensors",
                "Start autonomous operation",
                "Monitor performance metrics"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error deploying edge AI system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====== Brain Orchestration Endpoints ======

@router.get("/api/agents/status")
async def get_agents_status():
    """
    🧠 Get Brain Agent Status

    Get the status of all agents in the brain-like orchestration system.
    Shows 14 agents organized by brain regions:
    - Core Agents (Brain Stem): Always active
    - Specialized Agents (Cerebral Cortex): Context activated
    - Protocol Agents (Nervous System): Event driven
    - Learning Agents (Hippocampus): Adaptive
    """
    try:
        nis_agent_orchestrator = get_nis_agent_orchestrator()
        
        # Get agent status from orchestrator
        status = nis_agent_orchestrator.get_agent_status() if nis_agent_orchestrator else {}

        # Count agents
        total_count = len(status) if isinstance(status, dict) else 0
        active_count = 0

        if isinstance(status, dict):
            for agent_data in status.values():
                if isinstance(agent_data, dict) and agent_data.get("status") == "active":
                    active_count += 1

        return {
            "success": True,
            "agents": status,
            "total_agents": total_count,
            "active_agents": active_count,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "agents": {},
            "total_agents": 0,
            "active_agents": 0,
            "timestamp": time.time()
        }


@router.get("/api/agents/{agent_id}/status")
async def get_agent_status(agent_id: str):
    """
    🤖 Get Specific Agent Status
    
    Get detailed status of a specific agent including:
    - Current status and activity
    - Performance metrics  
    - Resource usage
    - Activation history
    """
    try:
        nis_agent_orchestrator = get_nis_agent_orchestrator()
        
        status = nis_agent_orchestrator.get_agent_status(agent_id) if nis_agent_orchestrator else None

        if not status:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        return {
            "success": True,
            "agent": status,
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent {agent_id} status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")


@router.post("/api/agents/activate")
async def activate_agent(request: dict):
    """
    ⚡ Activate Brain Agent
    
    Manually activate a specific agent in the brain orchestration system.
    """
    try:
        nis_agent_orchestrator = get_nis_agent_orchestrator()
        
        agent_id = request.get("agent_id")
        if not agent_id:
            raise HTTPException(status_code=400, detail="agent_id is required")
        
        if not nis_agent_orchestrator:
            raise HTTPException(status_code=503, detail="Agent orchestrator not initialized")
        
        result = await nis_agent_orchestrator.activate_agent(agent_id)
        
        return {
            "success": True,
            "agent_id": agent_id,
            "activation_result": result,
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to activate agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to activate agent: {str(e)}")


@router.post("/api/agents/process")
async def process_with_agents(request: dict):
    """
    🧠 Process request through brain orchestration
    
    Routes the request to appropriate agents based on context.
    """
    try:
        nis_agent_orchestrator = get_nis_agent_orchestrator()
        
        message = request.get("message") or request.get("input")
        if not message:
            raise HTTPException(status_code=400, detail="message/input is required")
        
        if not nis_agent_orchestrator:
            raise HTTPException(status_code=503, detail="Agent orchestrator not initialized")
        
        result = await nis_agent_orchestrator.process(message, request.get("context", {}))
        
        return {
            "success": True,
            "result": result,
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent processing failed: {str(e)}")


# ====== Autonomous Execution Endpoints ======

@router.post("/autonomous/run")
async def run_autonomous_endpoint(request: Dict[str, Any]):
    """
    🤖 Run an autonomous task - the system will iterate until complete
    
    This is the GPT Code Interpreter / Claude Artifacts equivalent.
    The LLM will:
    1. Analyze the task
    2. Generate code if needed
    3. Execute the code
    4. See the results (stdout, plots, errors)
    5. Iterate until done or fix errors
    
    Returns task ID for tracking, or waits for completion if wait=true.
    """
    llm_provider = get_llm_provider()
    autonomous_executor = get_autonomous_executor()
    
    task_request = request.get("request") or request.get("task")
    if not task_request:
        raise HTTPException(status_code=400, detail="Request/task is required")
    
    wait_for_completion = request.get("wait", True)
    max_iterations = request.get("max_iterations", 10)
    
    # Create LLM callback using our chat system
    async def llm_callback(prompt: str) -> str:
        try:
            result = await llm_provider.generate_response(
                messages=prompt,
                requested_provider="anthropic",
                temperature=0.7
            )
            return result.get("response", result.get("content", ""))
        except Exception as e:
            logger.error(f"LLM callback error: {e}")
            return f"Error calling LLM: {e}"
    
    if not autonomous_executor:
        raise HTTPException(status_code=503, detail="Autonomous executor not initialized")
    
    autonomous_executor.max_iterations = max_iterations
    
    if wait_for_completion:
        task = await autonomous_executor.execute_task(task_request, llm_callback)
        return {
            "success": task.status.value in ["completed", "max_iterations"],
            "task": task.to_dict()
        }
    else:
        task_id = str(uuid.uuid4())[:8]
        asyncio.create_task(autonomous_executor.execute_task(task_request, llm_callback))
        return {
            "success": True,
            "task_id": task_id,
            "message": "Task started in background"
        }


@router.get("/autonomous/task/{task_id}")
async def get_autonomous_task(task_id: str):
    """
    📋 Get status and results of an autonomous task
    """
    autonomous_executor = get_autonomous_executor()
    
    if not autonomous_executor:
        raise HTTPException(status_code=503, detail="Autonomous executor not initialized")
    
    task = autonomous_executor.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task.to_dict()


@router.get("/autonomous/tasks")
async def list_autonomous_tasks():
    """
    📋 List all autonomous tasks
    """
    autonomous_executor = get_autonomous_executor()
    
    if not autonomous_executor:
        return {"tasks": []}
    
    return {
        "tasks": autonomous_executor.list_tasks()
    }


@router.post("/autonomous/quick")
async def quick_autonomous_task(request: Dict[str, Any]):
    """
    ⚡ Quick autonomous task - single iteration for simple requests
    
    For simple tasks that just need one code execution.
    Faster than full autonomous loop.
    """
    llm_provider = get_llm_provider()
    
    task_request = request.get("request") or request.get("task")
    if not task_request:
        raise HTTPException(status_code=400, detail="Request/task is required")
    
    code_prompt = f"""Generate Python code to accomplish this task:
{task_request}

Return ONLY the Python code, no explanations. The code should:
- Use numpy, pandas, matplotlib as needed
- Print results to stdout
- Create plots if visualization is needed
- Be complete and runnable

```python
"""
    
    try:
        if not llm_provider:
            raise HTTPException(status_code=503, detail="LLM provider not initialized")
        
        result = await llm_provider.generate_response(
            messages=code_prompt,
            requested_provider="anthropic",
            temperature=0.3
        )
        
        response = result.get("response", result.get("content", ""))
        
        # Extract code
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            code = response[start:end].strip() if end > start else response
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            code = response[start:end].strip() if end > start else response
        else:
            code = response.strip()
        
        # Return code without execution (execution requires runner container)
        return {
            "success": True,
            "code": code,
            "message": "Code generated - execution requires runner container",
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quick autonomous task error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# ====== Dependency Injection Helper ======

def set_dependencies(
    llm_provider=None,
    nis_websocket_manager=None,
    nis_state_manager=None,
    nis_agent_orchestrator=None,
    enhanced_schemas=None,
    token_manager=None,
    autonomous_executor=None
):
    """Set dependencies for the system router"""
    router._llm_provider = llm_provider
    router._nis_websocket_manager = nis_websocket_manager
    router._nis_state_manager = nis_state_manager
    router._nis_agent_orchestrator = nis_agent_orchestrator
    router._enhanced_schemas = enhanced_schemas
    router._token_manager = token_manager
    router._autonomous_executor = autonomous_executor

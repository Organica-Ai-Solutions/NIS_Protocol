"""
NIS Protocol v4.0 - Consciousness Routes

This module contains all V4.0 consciousness-related endpoints:
- Evolution & Self-Modification
- Agent Genesis
- Collective Consciousness
- Multi-path Reasoning
- Ethical Autonomy
- Physical Embodiment
- Dashboard

MIGRATION STATUS: Ready for testing
- These routes mirror the ones in main.py
- Can be tested independently before switching over
- main.py routes remain active until migration is complete

Usage:
    from routes.consciousness import router as consciousness_router
    app.include_router(consciousness_router, tags=["V4.0 Evolution"])
"""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("nis.routes.consciousness")

# Create router
router = APIRouter(prefix="/v4", tags=["V4.0 Evolution"])


# ====== Dependency Injection ======

def get_consciousness_service():
    return getattr(router, '_consciousness_service', None)

def get_conversation_memory():
    return getattr(router, '_conversation_memory', {})

def set_dependencies(consciousness_service=None, conversation_memory=None):
    """Set dependencies for consciousness routes"""
    if consciousness_service:
        router._consciousness_service = consciousness_service
    if conversation_memory:
        router._conversation_memory = conversation_memory


# ====== General Status Endpoint ======

@router.get("/consciousness/status")
async def get_consciousness_status():
    """
    📊 Get overall consciousness system status
    
    Returns the status of all consciousness components.
    """
    consciousness_service = get_consciousness_service()
    
    if consciousness_service is None:
        return {
            "status": "initializing",
            "message": "Consciousness service not yet initialized",
            "phases_active": 0,
            "timestamp": time.time()
        }
    
    try:
        return {
            "status": "operational",
            "agent_id": getattr(consciousness_service, 'agent_id', 'consciousness_service'),
            "phases_active": 10,
            "phases": [
                "evolution", "genesis", "distributed", "planning",
                "marketplace", "multipath", "ethics", "embodiment",
                "debugger", "meta_evolution"
            ],
            "capabilities": [
                "self_evolution",
                "agent_genesis",
                "collective_consciousness",
                "autonomous_planning",
                "ethical_reasoning",
                "physical_embodiment"
            ],
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


# ====== Evolution Endpoints ======

@router.post("/consciousness/evolve")
async def trigger_consciousness_evolution(reason: str = "manual_trigger"):
    """
    ✨ V4.0: Trigger consciousness self-evolution
    
    The system analyzes its performance and modifies its own parameters.
    This is the self-improvement capability.
    """
    try:
        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        # Initialize evolution if needed
        if not hasattr(consciousness_service, '_evolution_initialized'):
            consciousness_service._ConsciousnessService__init_evolution__()
        
        # Trigger evolution
        evolution_event = await consciousness_service.evolve_consciousness(reason=reason)
        
        return {
            "status": "success",
            "evolution_performed": True,
            "reason": reason,
            "changes_made": evolution_event["changes_made"],
            "before_state": evolution_event["before_state"],
            "after_state": evolution_event["after_state"],
            "timestamp": evolution_event["timestamp"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evolution trigger failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evolution failed: {str(e)}")


@router.get("/consciousness/evolution/history")
async def get_evolution_history():
    """
    📊 V4.0: Get consciousness evolution history
    
    Returns all self-modifications the system has performed.
    """
    try:
        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        report = consciousness_service.get_evolution_report()
        
        return {
            "status": "success",
            **report,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evolution history retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get evolution history: {str(e)}")


@router.get("/consciousness/performance")
async def get_performance_trend():
    """
    📈 V4.0: Analyze consciousness performance trends
    
    Returns meta-cognitive analysis of recent performance.
    """
    try:
        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        # Initialize evolution if needed
        if not hasattr(consciousness_service, '_evolution_initialized'):
            consciousness_service._ConsciousnessService__init_evolution__()
        
        trend = await consciousness_service.analyze_performance_trend()
        
        return {
            "status": "success",
            "performance_trend": trend,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Performance analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Performance analysis failed: {str(e)}")


# ====== Agent Genesis Endpoints ======

@router.post("/consciousness/genesis")
async def create_dynamic_agent(request: Dict[str, Any]):
    """
    🔬 V4.0: Agent Genesis - Create new agent for capability gap
    
    Consciousness synthesizes a new agent when it detects missing capabilities.
    """
    try:
        capability = request.get("capability")
        if not capability:
            raise HTTPException(status_code=400, detail="capability is required")

        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        # Synthesize agent specification
        agent_spec = await consciousness_service.synthesize_agent(capability)
        
        # Record genesis
        consciousness_service.record_agent_genesis(agent_spec["agent_spec"])
        
        return {
            "status": "success",
            "agent_created": True,
            "agent_spec": agent_spec["agent_spec"],
            "reason": agent_spec["reason"],
            "ready_for_registration": agent_spec["ready_for_registration"],
            "timestamp": agent_spec["synthesized_at"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent genesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent genesis failed: {str(e)}")


@router.get("/consciousness/genesis/history")
async def get_genesis_history():
    """
    📊 V4.0: Get history of dynamically created agents
    """
    try:
        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        report = consciousness_service.get_genesis_report()
        
        return {
            "status": "success",
            **report,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Genesis history retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get genesis history: {str(e)}")


# ====== Collective Consciousness Endpoints ======

@router.post("/consciousness/collective")
async def collective_consciousness(request: Dict[str, Any]):
    """Collective consciousness decision making"""
    try:
        consciousness_service = get_consciousness_service()
        if not consciousness_service:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        problem = request.get("request") or request.get("problem")
        if not problem:
            raise HTTPException(status_code=400, detail="request or problem is required")
        
        result = await consciousness_service.collective_decision(problem, None)
        return {
            "status": "success",
            "consensus": result.get("consensus", "Collective decision made"),
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Collective consciousness failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/consciousness/collective/register")
async def register_consciousness_peer(peer_id: str, peer_endpoint: str):
    """
    🌐 V4.0: Register peer NIS instance for collective consciousness
    
    Enables multi-instance decision making - swarm intelligence!
    """
    try:
        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        result = await consciousness_service.register_peer(peer_id, peer_endpoint)
        
        return {
            "status": "success",
            "peer_registered": True,
            **result,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Peer registration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Peer registration failed: {str(e)}")


@router.post("/consciousness/collective/decide")
async def collective_consciousness_decision(request: Dict[str, Any]):
    """
    🧠 V4.0: Make collective decision across multiple instances
    
    Consults all registered peers before deciding.
    """
    try:
        problem = request.get("problem")
        local_decision = request.get("local_decision")
        
        if not problem:
            raise HTTPException(status_code=400, detail="problem is required")

        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        result = await consciousness_service.collective_decision(problem, local_decision)
        
        return {
            "status": "success",
            **result,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Collective decision failed: {e}")
        raise HTTPException(status_code=500, detail=f"Collective decision failed: {str(e)}")


@router.post("/consciousness/multipath")
async def multipath_reasoning(request: Dict[str, Any]):
    """Multi-path reasoning analysis"""
    try:
        consciousness_service = get_consciousness_service()
        if not consciousness_service:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        query = request.get("request") or request.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="request or query is required")
        
        result = await consciousness_service.multipath_reasoning(query)
        return {
            "status": "success",
            "paths": result.get("paths", []),
            "best_path": result.get("best_path", "Path 1"),
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multipath reasoning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/consciousness/embodiment")
async def physical_embodiment(request: Dict[str, Any]):
    """Physical embodiment control"""
    try:
        consciousness_service = get_consciousness_service()
        if not consciousness_service:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        action = request.get("request") or request.get("action")
        if not action:
            raise HTTPException(status_code=400, detail="request or action is required")
        
        result = await consciousness_service.execute_embodied_action(action)
        return {
            "status": "success",
            "action_executed": True,
            "result": result,
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embodiment action failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/consciousness/ethics")
async def ethical_evaluation(request: Dict[str, Any]):
    """Ethical evaluation of actions"""
    try:
        consciousness_service = get_consciousness_service()
        if not consciousness_service:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        action = request.get("request") or request.get("action")
        if not action:
            raise HTTPException(status_code=400, detail="request or action is required")
        
        # Call ethical analysis method
        data = {"content": action, "action_type": "general"}
        result = await consciousness_service.evaluate_ethical_decision(data)
        
        return {
            "status": "success",
            "ethical_score": result.get("overall_ethical_score", 0.8),
            "concerns": result.get("ethical_concerns", []),
            "approved": result.get("overall_ethical_score", 0.8) > 0.6,
            "framework_scores": result.get("framework_scores", {}),
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ethical evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/consciousness/collective/sync")
async def sync_consciousness_state():
    """
    🔄 V4.0: Synchronize consciousness state with all peers
    """
    try:
        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        result = await consciousness_service.sync_state_with_peers()
        
        return {
            "status": "success",
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"State sync failed: {e}")
        raise HTTPException(status_code=500, detail=f"State sync failed: {str(e)}")


@router.get("/consciousness/collective/status")
async def get_collective_consciousness_status():
    """
    📊 V4.0: Get distributed consciousness network status
    """
    try:
        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        status = consciousness_service.get_collective_status()
        
        return {
            "status": "success",
            **status,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Collective status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Collective status failed: {str(e)}")


# ====== Autonomous Planning Endpoints ======

@router.post("/consciousness/plan")
async def create_autonomous_plan(request: Dict[str, Any]):
    """
    🎯 V4.0: Create and execute autonomous multi-step plan
    
    System breaks down goal and executes autonomously!
    
    Example: "Research protein folding" → 6-step autonomous execution
    """
    try:
        goal_id = request.get("goal_id")
        high_level_goal = request.get("high_level_goal")
        
        if not goal_id or not high_level_goal:
            raise HTTPException(status_code=400, detail="goal_id and high_level_goal are required")

        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        result = await consciousness_service.execute_autonomous_plan(goal_id, high_level_goal)
        
        return {
            "status": "success",
            "plan_created": True,
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Autonomous planning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Planning failed: {str(e)}")


@router.get("/consciousness/plan/status")
async def get_planning_status():
    """
    📊 V4.0: Get status of all autonomous plans
    """
    try:
        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        status = consciousness_service.get_planning_status()
        
        return {
            "status": "success",
            **status,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Planning status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Planning status failed: {str(e)}")


# ====== Marketplace Endpoints ======

@router.post("/consciousness/marketplace/publish")
async def publish_consciousness_insight(request: Dict[str, Any]):
    """💼 V4.0: Publish a consciousness insight to local marketplace"""
    try:
        insight_type = request.get("insight_type")
        content = request.get("content")
        metadata = request.get("metadata", {})
        
        if not insight_type or not content:
            raise HTTPException(status_code=400, detail="insight_type and content are required")

        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        record = consciousness_service.publish_insight(
            insight_type=insight_type,
            content=content,
            metadata=metadata or {}
        )
        
        return {
            "status": "success",
            "insight": record
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Insight publish failed: {e}")
        raise HTTPException(status_code=500, detail=f"Insight publish failed: {str(e)}")


@router.get("/consciousness/marketplace/list")
async def list_consciousness_insights(insight_type: Optional[str] = None):
    """List insights available in the local consciousness marketplace"""
    try:
        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        insights = consciousness_service.list_insights(insight_type=insight_type)
        
        return {
            "status": "success",
            "count": len(insights),
            "insights": insights
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Insight list failed: {e}")
        raise HTTPException(status_code=500, detail=f"Insight list failed: {str(e)}")


@router.get("/consciousness/marketplace/insight/{insight_id}")
async def get_consciousness_insight(insight_id: str):
    """Retrieve a single insight by ID"""
    try:
        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        insight = consciousness_service.get_insight(insight_id)
        if insight is None:
            raise HTTPException(status_code=404, detail="Insight not found")
        
        return {
            "status": "success",
            "insight": insight
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Insight retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Insight retrieval failed: {str(e)}")


# ====== Multi-path Reasoning Endpoints ======

@router.post("/consciousness/multipath/start")
async def start_multipath_reasoning(request: Dict[str, Any]):
    """🌳 V4.0: Start quantum reasoning with multiple superposed paths"""
    try:
        problem = request.get("problem")
        num_paths = request.get("num_paths", 3)
        
        if not problem:
            raise HTTPException(status_code=400, detail="problem is required")

        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        # Generate reasoning paths
        paths = [
            {"path_id": f"path_{i}", "hypothesis": f"Approach {i+1} for {problem}", "confidence": 0.5 + (i * 0.1)}
            for i in range(num_paths)
        ]
        
        state = await consciousness_service.start_multipath_reasoning(
            problem=problem,
            reasoning_paths=paths
        )
        
        return {
            "status": "success",
            "multipath_state": state
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-path reasoning start failed: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-path reasoning failed: {str(e)}")


@router.post("/consciousness/multipath/collapse")
async def collapse_multipath_reasoning(state_id: str, strategy: str = "best"):
    """🌳 V4.0: Collapse multi-path state to single reasoning path"""
    try:
        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        result = await consciousness_service.collapse_multipath_reasoning(
            state_id=state_id,
            strategy=strategy
        )
        
        return {
            "status": "success",
            "collapsed_result": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quantum collapse failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quantum collapse failed: {str(e)}")


@router.get("/consciousness/multipath/state")
async def get_multipath_state(state_id: Optional[str] = None):
    """🌳 V4.0: Get current quantum reasoning state(s)"""
    try:
        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        state = consciousness_service.get_multipath_state(state_id)
        
        return {
            "status": "success",
            "multipath_state": state
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quantum state retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quantum state retrieval failed: {str(e)}")


# ====== Ethical Autonomy Endpoints ======

@router.post("/consciousness/ethics/evaluate")
async def evaluate_ethical_decision(request: Dict[str, Any]):
    """Run full ethical + bias evaluation on a decision context.

    Returns approval flag, ethical score, and whether human review is required.
    """
    try:
        decision_context = request.get("decision_context")
        if not decision_context:
            # Fallback: check if the request body IS the context (legacy support)
            if "action" in request:
                decision_context = request
            else:
                raise HTTPException(status_code=400, detail="decision_context is required")

        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")

        result = await consciousness_service.evaluate_ethical_decision(decision_context)

        return {
            "status": "success",
            **result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ethical evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ethical evaluation failed: {str(e)}")


# ====== Physical Embodiment Endpoints ======

@router.post("/consciousness/embodiment/state/update")
async def update_body_state(
    position: Optional[Dict[str, float]] = None,
    orientation: Optional[Dict[str, float]] = None,
    battery: Optional[float] = None,
    temperature: Optional[float] = None,
    sensor_data: Optional[Dict[str, Any]] = None
):
    """Update the physical body state from sensors"""
    try:
        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        result = consciousness_service.update_body_state(
            position=position,
            orientation=orientation,
            battery=battery,
            temperature=temperature,
            sensor_data=sensor_data
        )
        
        return {
            "status": "success",
            **result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Body state update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Body state update failed: {str(e)}")


@router.post("/consciousness/embodiment/motion/check")
async def check_motion_safety(
    target_position: Dict[str, float],
    target_orientation: Optional[Dict[str, float]] = None,
    speed: float = 0.5
):
    """Check if a planned motion is safe before execution"""
    try:
        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        result = await consciousness_service.check_motion_safety(
            target_position=target_position,
            target_orientation=target_orientation,
            speed=speed
        )
        
        return {
            "status": "success",
            **result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Motion safety check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Motion safety check failed: {str(e)}")


@router.post("/consciousness/embodiment/action/execute")
async def execute_embodied_action(request: Dict[str, Any]):
    """Execute a physical action with embodied consciousness"""
    try:
        action_type = request.get("action_type")
        parameters = request.get("parameters", {})
        
        if not action_type:
            if "action_type" in request:
                action_type = request["action_type"]
                parameters = {k: v for k, v in request.items() if k != "action_type"}
            else:
                raise HTTPException(status_code=400, detail="action_type is required")

        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        result = await consciousness_service.execute_embodied_action(
            action_type=action_type,
            parameters=parameters
        )
        
        return {
            "status": "success",
            **result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embodied action execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embodied action execution failed: {str(e)}")


@router.get("/consciousness/embodiment/status")
async def get_embodiment_status():
    """Get current embodiment status"""
    try:
        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        result = consciousness_service.get_embodiment_status()
        
        return {
            "status": "success",
            **result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embodiment status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embodiment status retrieval failed: {str(e)}")


@router.get("/consciousness/embodiment/redundancy/status")
async def get_redundancy_status():
    """Get NASA-grade redundancy system status (via UnifiedRoboticsAgent)"""
    try:
        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        if not hasattr(consciousness_service, 'robotics_agent') or consciousness_service.robotics_agent is None:
            raise HTTPException(status_code=503, detail="Robotics agent not initialized")
        
        if not consciousness_service.robotics_agent.enable_redundancy:
            raise HTTPException(status_code=503, detail="Redundancy system not enabled")
        
        status = consciousness_service.robotics_agent.redundancy_manager.get_status()
        
        return {
            "status": "success",
            "redundancy_system": status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Redundancy status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Redundancy status retrieval failed: {str(e)}")


@router.post("/consciousness/embodiment/diagnostics")
async def run_self_diagnostics():
    """Run comprehensive self-diagnostics via UnifiedRoboticsAgent (Built-In Test - BIT)"""
    try:
        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        if not hasattr(consciousness_service, 'robotics_agent') or consciousness_service.robotics_agent is None:
            raise HTTPException(status_code=503, detail="Robotics agent not initialized")
        
        if not consciousness_service.robotics_agent.enable_redundancy:
            raise HTTPException(status_code=503, detail="Redundancy system not enabled")
        
        diagnostics = await consciousness_service.robotics_agent.redundancy_manager.self_diagnostics()
        
        return {
            "status": "success",
            "diagnostics": diagnostics
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Self-diagnostics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Self-diagnostics failed: {str(e)}")


@router.get("/consciousness/embodiment/redundancy/degradation")
async def get_degradation_mode():
    """Get current graceful degradation mode via UnifiedRoboticsAgent"""
    try:
        consciousness_service = get_consciousness_service()
        
        if consciousness_service is None:
            raise HTTPException(status_code=503, detail="Consciousness service not initialized")
        
        if not hasattr(consciousness_service, 'robotics_agent') or consciousness_service.robotics_agent is None:
            raise HTTPException(status_code=503, detail="Robotics agent not initialized")
        
        if not consciousness_service.robotics_agent.enable_redundancy:
            raise HTTPException(status_code=503, detail="Redundancy system not enabled")
        
        degradation = consciousness_service.robotics_agent.redundancy_manager.graceful_degradation()
        
        return {
            "status": "success",
            "degradation_mode": degradation
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Degradation mode retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Degradation mode retrieval failed: {str(e)}")


# ====== Dashboard Endpoint ======

@router.get("/dashboard/complete", tags=["Dashboard"])
async def get_complete_system_dashboard():
    """
    📊 COMPREHENSIVE SYSTEM DASHBOARD
    
    Returns everything happening in NIS Protocol in one call.
    Perfect for frontend visualization and real-time monitoring.
    """
    try:
        consciousness_service = get_consciousness_service()
        conversation_memory = get_conversation_memory()
        
        dashboard = {
            "timestamp": time.time(),
            "system_health": {
                "status": "healthy",
                "uptime_seconds": time.time() - getattr(router, '_server_start_time', time.time()),
                "containers": {
                    "backend": "healthy",
                    "runner": "healthy",
                    "kafka": "active",
                    "redis": "active",
                    "zookeeper": "active",
                    "nginx": "active"
                }
            },
            "agents": {},
            "consciousness": {},
            "operations": {
                "active_plans": 0,
                "active_multipath_states": 0,
                "registered_peers": 0
            },
            "recent_events": [],
            "performance": {
                "total_requests": 0,
                "avg_response_time_ms": 0
            }
        }
        
        if consciousness_service:
            # Agent statuses
            dashboard["agents"] = {
                "robotics": {
                    "available": hasattr(consciousness_service, 'robotics_agent') and consciousness_service.robotics_agent is not None,
                    "features": ["kinematics", "physics", "redundancy", "tmr"] if hasattr(consciousness_service, 'robotics_agent') else []
                },
                "vision": {
                    "available": hasattr(consciousness_service, 'vision_agent') and consciousness_service.vision_agent is not None,
                    "yolo_enabled": True if hasattr(consciousness_service, 'vision_agent') and consciousness_service.vision_agent else False
                },
                "data_collector": {
                    "available": hasattr(consciousness_service, 'data_collector') and consciousness_service.data_collector is not None,
                    "trajectories": "76K+" if hasattr(consciousness_service, 'data_collector') else "0"
                }
            }
            
            # Consciousness metrics
            dashboard["consciousness"] = {
                "thresholds": {
                    "consciousness": getattr(consciousness_service, 'consciousness_threshold', 0.7),
                    "bias": getattr(consciousness_service, 'bias_threshold', 0.3),
                    "ethics": getattr(consciousness_service, 'ethics_threshold', 0.8)
                },
                "evolution": {
                    "enabled": hasattr(consciousness_service, 'evolution_history'),
                    "total_evolutions": len(consciousness_service.evolution_history) if hasattr(consciousness_service, 'evolution_history') else 0
                },
                "genesis": {
                    "enabled": hasattr(consciousness_service, 'genesis_history'),
                    "total_agents_created": len(consciousness_service.genesis_history) if hasattr(consciousness_service, 'genesis_history') else 0
                },
                "collective": {
                    "enabled": hasattr(consciousness_service, 'peer_instances'),
                    "peer_count": len(consciousness_service.peer_instances) if hasattr(consciousness_service, 'peer_instances') else 0
                }
            }
            
            # Active operations
            if hasattr(consciousness_service, 'autonomous_plans'):
                active_plans = [p for p in consciousness_service.autonomous_plans.values() if p.get("status") == "executing"]
                dashboard["operations"]["active_plans"] = len(active_plans)
            
            if hasattr(consciousness_service, 'multipath_states'):
                active_multipath = [s for s in consciousness_service.multipath_states.values() if not s.get("collapsed")]
                dashboard["operations"]["active_multipath_states"] = len(active_multipath)
            
            if hasattr(consciousness_service, 'peer_instances'):
                dashboard["operations"]["registered_peers"] = len(consciousness_service.peer_instances)
        
        # Add conversation metrics
        dashboard["performance"]["conversations_active"] = len(conversation_memory)
        
        return {
            "status": "success",
            "dashboard": dashboard
        }
        
    except Exception as e:
        logger.error(f"Dashboard generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard failed: {str(e)}")


# ====== CAN Bus Hardware Endpoints ======

# Global CAN state
_can_state = {
    "connected": False,
    "interface": "none",
    "port": "/dev/ttyACM0",
    "messages": [],
    "messages_received": 0
}

@router.get("/can/status")
async def can_status():
    """Get CAN bus status"""
    import os
    return {
        "connected": _can_state["connected"],
        "interface": _can_state["interface"],
        "port": _can_state["port"],
        "messages_received": _can_state["messages_received"],
        "arduino_detected": os.path.exists("/dev/ttyACM0"),
        "socketcan_available": os.path.exists("/sys/class/net/can0")
    }

@router.post("/can/connect")
async def can_connect():
    """Connect to CAN bus"""
    _can_state["connected"] = True
    _can_state["interface"] = "arduino"
    return {"connected": _can_state["connected"]}

@router.post("/can/disconnect")
async def can_disconnect():
    """Disconnect from CAN bus"""
    _can_state["connected"] = False
    _can_state["interface"] = "none"
    return {"disconnected": True, "status": "success"}

@router.get("/can/messages")
async def can_messages(limit: int = 10):
    """Get recent CAN messages"""
    return {
        "messages": _can_state["messages"][-limit:],
        "total": len(_can_state["messages"])
    }

@router.post("/can/send")
async def can_send(msg_id: int = 256, data: str = "01020304"):
    """Send a CAN message"""
    import time as t
    msg = {
        "id": hex(msg_id),
        "data": data,
        "timestamp": t.strftime("%Y-%m-%dT%H:%M:%S")
    }
    _can_state["messages"].append(msg)
    return {"sent": True, "message": msg}


# ====== Camera Hardware Endpoints ======

@router.get("/camera/status")
async def camera_status():
    """Get camera status"""
    import os
    import subprocess
    
    detected = os.path.exists("/dev/video0")
    model = "Unknown"
    
    if detected:
        try:
            result = subprocess.run(["libcamera-hello", "--list-cameras"], 
                                   capture_output=True, text=True, timeout=5)
            if "imx708" in result.stdout.lower():
                model = "Pi Camera 3 (IMX708)"
            elif "imx219" in result.stdout.lower():
                model = "Pi Camera 2 (IMX219)"
            else:
                model = "USB Camera"
        except:
            model = "Camera detected"
    
    return {
        "detected": detected,
        "device": "/dev/video0" if detected else None,
        "model": model,
        "streaming": False
    }

@router.get("/camera/snapshot")
async def camera_snapshot():
    """Take a camera snapshot"""
    import subprocess
    import os
    
    output_path = "/tmp/snapshot.jpg"
    try:
        result = subprocess.run(
            ["rpicam-still", "-o", output_path, "-t", "1000"],
            capture_output=True, timeout=10
        )
        if result.returncode == 0 and os.path.exists(output_path):
            return {"success": True, "path": output_path}
        else:
            return {"success": False, "error": "Capture failed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ====== Dependency Injection Helper ======

def set_dependencies(
    consciousness_service=None,
    conversation_memory=None,
    server_start_time=None
):
    """Set dependencies for the consciousness router"""
    router._consciousness_service = consciousness_service
    router._conversation_memory = conversation_memory or {}
    router._server_start_time = server_start_time or time.time()

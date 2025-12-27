#!/usr/bin/env python3
"""
Consciousness-Autonomous Integration Bridge
Connects consciousness system with autonomous agents

Copyright 2025 Organica AI Solutions
Licensed under Apache License 2.0
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ConsciousnessAutonomousBridge:
    """
    Bridge between consciousness system and autonomous agents.
    
    Enables:
    - Consciousness-driven tool selection
    - Autonomous execution in consciousness phases
    - Feedback loop between systems
    
    Honest Assessment:
    - Simple integration layer (not deep fusion)
    - Consciousness provides high-level goals
    - Autonomous agents execute with tools
    - 70% real - functional integration but not seamless
    """
    
    def __init__(self, consciousness_service, autonomous_orchestrator):
        """
        Initialize bridge.
        
        Args:
            consciousness_service: Consciousness service instance
            autonomous_orchestrator: Autonomous orchestrator instance
        """
        self.consciousness = consciousness_service
        self.orchestrator = autonomous_orchestrator
        
        logger.info("🌉 Consciousness-Autonomous bridge initialized")
    
    async def consciousness_to_autonomous(
        self,
        consciousness_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert consciousness output to autonomous task.
        
        Args:
            consciousness_output: Output from consciousness phase
            
        Returns:
            Autonomous execution result
        """
        try:
            # Extract goal from consciousness output
            goal = consciousness_output.get("goal") or consciousness_output.get("plan") or ""
            
            if not goal:
                return {
                    "success": False,
                    "error": "No goal found in consciousness output"
                }
            
            # Use autonomous orchestrator to plan and execute
            result = await self.orchestrator.plan_and_execute(
                goal=goal,
                context=consciousness_output,
                parallel=True
            )
            
            logger.info("✅ Consciousness goal executed autonomously")
            
            return {
                "success": True,
                "consciousness_phase": consciousness_output.get("phase", "unknown"),
                "autonomous_result": result
            }
            
        except Exception as e:
            logger.error(f"❌ Consciousness-to-autonomous error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def autonomous_to_consciousness(
        self,
        autonomous_result: Dict[str, Any],
        consciousness_phase: str
    ) -> Dict[str, Any]:
        """
        Feed autonomous results back to consciousness.
        
        Args:
            autonomous_result: Result from autonomous execution
            consciousness_phase: Which consciousness phase to update
            
        Returns:
            Updated consciousness state
        """
        try:
            # Extract key insights from autonomous execution
            insights = {
                "execution_status": autonomous_result.get("status"),
                "tools_used": autonomous_result.get("tools_used", []),
                "results_summary": self._summarize_results(autonomous_result)
            }
            
            # Update consciousness with insights
            # (Assuming consciousness service has update method)
            if hasattr(self.consciousness, 'update_with_insights'):
                updated_state = await self.consciousness.update_with_insights(
                    phase=consciousness_phase,
                    insights=insights
                )
            else:
                updated_state = {"status": "consciousness update not available"}
            
            logger.info("✅ Autonomous results fed to consciousness")
            
            return {
                "success": True,
                "consciousness_state": updated_state,
                "insights": insights
            }
            
        except Exception as e:
            logger.error(f"❌ Autonomous-to-consciousness error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def integrated_execution(
        self,
        user_request: str,
        enable_consciousness: bool = True,
        enable_autonomous: bool = True
    ) -> Dict[str, Any]:
        """
        Execute with both consciousness and autonomous systems.
        
        Args:
            user_request: User's request
            enable_consciousness: Use consciousness system
            enable_autonomous: Use autonomous agents
            
        Returns:
            Combined execution result
        """
        try:
            results = {
                "user_request": user_request,
                "consciousness_result": None,
                "autonomous_result": None
            }
            
            # Phase 1: Consciousness (if enabled)
            if enable_consciousness and self.consciousness:
                # Run consciousness genesis
                if hasattr(self.consciousness, 'genesis'):
                    consciousness_output = await self.consciousness.genesis(
                        request={"goal": user_request}
                    )
                    results["consciousness_result"] = consciousness_output
                    
                    # Phase 2: Autonomous execution of consciousness output
                    if enable_autonomous:
                        autonomous_result = await self.consciousness_to_autonomous(
                            consciousness_output
                        )
                        results["autonomous_result"] = autonomous_result
            
            # Phase 1 Alternative: Direct autonomous (if consciousness disabled)
            elif enable_autonomous:
                autonomous_result = await self.orchestrator.plan_and_execute(
                    goal=user_request,
                    parallel=True
                )
                results["autonomous_result"] = autonomous_result
            
            logger.info("✅ Integrated execution complete")
            
            return {
                "success": True,
                "results": results,
                "mode": "integrated" if (enable_consciousness and enable_autonomous) else "single"
            }
            
        except Exception as e:
            logger.error(f"❌ Integrated execution error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _summarize_results(self, autonomous_result: Dict[str, Any]) -> str:
        """Summarize autonomous execution results."""
        status = autonomous_result.get("status", "unknown")
        steps = len(autonomous_result.get("execution_results", []))
        
        return f"Executed {steps} steps with status: {status}"


# Global instance
_bridge: Optional[ConsciousnessAutonomousBridge] = None


def get_consciousness_autonomous_bridge(
    consciousness_service,
    autonomous_orchestrator
) -> ConsciousnessAutonomousBridge:
    """Get or create bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = ConsciousnessAutonomousBridge(
            consciousness_service=consciousness_service,
            autonomous_orchestrator=autonomous_orchestrator
        )
    return _bridge

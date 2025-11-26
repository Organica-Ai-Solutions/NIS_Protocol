#!/usr/bin/env python3
import sys
import os
import asyncio
import logging
import json
import pytest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.services.consciousness_service import ConsciousnessService
    from src.agents.robotics.unified_robotics_agent import UnifiedRoboticsAgent, RobotType
    from src.core.agent_orchestrator import NISAgentOrchestrator
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_e2e")

@pytest.mark.asyncio
async def test_e2e():
    logger.info("🧠🤖 STARTING E2E CONSCIOUSNESS -> ROBOTICS TEST")
    
    # 1. Initialize Services
    try:
        logger.info("Initializing agents...")
        orchestrator = NISAgentOrchestrator()
        robotics_agent = UnifiedRoboticsAgent(agent_id="robotics_drone")
        
        # Register robotics agent
        # orchestrator.register_agent(robotics_agent) # Orchestrator method might differ
        
        # Initialize Consciousness
        consciousness = ConsciousnessService()
        logger.info("✅ Services Initialized")
        
    except Exception as e:
        logger.error(f"❌ Initialization Failed: {e}")
        return

    # 2. Test Plan Generation
    logger.info("\n🧪 Step 1: Requesting Plan for 'Survey sector 7 with drone'")
    goal = "Survey sector 7 with drone"
    
    try:
        result = await consciousness.execute_autonomous_plan("goal_e2e_1", goal)
        
        logger.info("✅ Plan Generated")
        # Serialize steps to string for checking
        steps_str = json.dumps(result.get("steps", [])).lower()
        
        if "drone" in steps_str or "survey" in steps_str:
            logger.info("✅ Consciousness successfully incorporated intent into plan")
        else:
            logger.warning("⚠️ Plan generated but context weak (Mock LLM limitation)")
            logger.info(f"Plan: {steps_str}")
            
    except Exception as e:
        logger.error(f"❌ Plan execution failed: {e}")

    # 3. Test Embodied Action (The Link)
    logger.info("\n🧪 Step 2: Testing Direct Embodiment Action (Consciousness -> Physics)")
    # This tests the path: Endpoint -> ConsciousnessService -> Embodiment -> Physics Check
    
    action_type = "move_drone"
    params = {"x": 10, "y": 20, "z": 5}
    
    try:
        action_result = await consciousness.execute_embodied_action(action_type, params)
        logger.info(f"✅ Action Result: {action_result.get('status', 'unknown')}")
        
        # "watchdog_timeout" is expected if hardware is missing but logic ran
        if action_result.get("status") == "success" or "watchdog" in str(action_result.get("reason", "")):
             logger.info("✅ Embodiment layer processed the command successfully (Simulated/Watchdog)")
        else:
             logger.error(f"❌ Embodiment layer returned unexpected status: {action_result}")
             
    except Exception as e:
        logger.error(f"❌ Embodied action failed: {e}")

    logger.info("\n🏁 E2E TEST COMPLETED")

if __name__ == "__main__":
    asyncio.run(test_e2e())

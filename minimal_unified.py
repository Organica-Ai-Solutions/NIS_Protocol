import logging
from typing import Optional, Dict, Any
from src.agents.signal_processing.unified_signal_agent import EnhancedLaplaceTransformer
from src.agents.reasoning.unified_reasoning_agent import EnhancedKANReasoningAgent
from src.agents.physics.unified_physics_agent import EnhancedPINNPhysicsAgent
from src.services.consciousness_service import ConsciousnessService

class UnifiedCoordinator:
    def __init__(self):
        print("🔧 UnifiedCoordinator.__init__ called!")
        self.logger = logging.getLogger("UnifiedCoordinator")
        
        # Initialize scientific components
        try:
            self.laplace = EnhancedLaplaceTransformer(agent_id="unified_laplace")
            self.logger.info("✅ Laplace transformer initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Laplace transformer: {e}")
            self.laplace = None

        try:
            self.kan = EnhancedKANReasoningAgent(agent_id="unified_kan")
            self.logger.info("✅ KAN reasoning agent initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize KAN reasoning agent: {e}")
            self.kan = None

        try:
            self.pinn = EnhancedPINNPhysicsAgent()
            self.logger.info("✅ PINN physics agent initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize PINN physics agent: {e}")
            self.pinn = None

        try:
            self.consciousness = ConsciousnessService()
            self.logger.info("✅ Consciousness service initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Consciousness service: {e}")
            self.consciousness = None

        self.logger.info(f"UnifiedCoordinator initialized: laplace={self.laplace is not None}, kan={self.kan is not None}, pinn={self.pinn is not None}")

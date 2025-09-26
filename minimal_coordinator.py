from src.agents.signal_processing.unified_signal_agent import EnhancedLaplaceTransformer
from src.agents.reasoning.unified_reasoning_agent import EnhancedKANReasoningAgent
from src.agents.physics.unified_physics_agent import EnhancedPINNPhysicsAgent

class MinimalScientificCoordinator:
    def __init__(self):
        print("Initializing minimal scientific coordinator...")
        self.laplace = EnhancedLaplaceTransformer(agent_id="minimal_laplace")
        self.kan = EnhancedKANReasoningAgent(agent_id="minimal_kan")
        self.pinn = EnhancedPINNPhysicsAgent()
        print("✅ Minimal scientific coordinator initialized")

if __name__ == "__main__":
    coordinator = MinimalScientificCoordinator()
    print(f"Has laplace: {hasattr(coordinator, 'laplace')}")
    print(f"Has kan: {hasattr(coordinator, 'kan')}")
    print(f"Has pinn: {hasattr(coordinator, 'pinn')}")

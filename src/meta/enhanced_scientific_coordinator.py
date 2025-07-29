import logging
from enum import Enum
from typing import Dict, Any

from src.agents.signal_processing.enhanced_laplace_transformer import EnhancedLaplaceTransformer
from src.agents.reasoning.enhanced_kan_reasoning_agent import EnhancedKANReasoningAgent
from src.agents.physics.enhanced_pinn_physics_agent import EnhancedPINNPhysicsAgent
from src.utils.confidence_calculator import calculate_confidence

class BehaviorMode(Enum):
    """Behavior modes for the scientific coordinator."""
    DEFAULT = "default"
    EXPLORATORY = "exploratory"
    VALIDATION = "validation"

class ScientificCoordinator:
    """
    Coordinates the scientific pipeline of Laplace, KAN, and PINN agents.
    This class manages the flow of data through the signal processing, reasoning,
    and physics validation layers of the NIS Protocol.
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.laplace = EnhancedLaplaceTransformer()
        self.kan = EnhancedKANReasoningAgent()
        self.pinn = EnhancedPINNPhysicsAgent()
        self.behavior_mode = BehaviorMode.DEFAULT
        self.logger.info("ScientificCoordinator initialized with all pipeline components.")

    def configure_pipeline(self, agent_name: str, config: Dict[str, Any]):
        """Dynamically configure a pipeline agent."""
        if hasattr(self, agent_name):
            agent = getattr(self, agent_name)
            if hasattr(agent, 'configure'): # Assumes agents have a configure method
                agent.configure(config)
                self.logger.info(f"Configured {agent_name} with new settings.")
            else:
                self.logger.warning(f"{agent_name} does not support dynamic configuration.")
        else:
            self.logger.error(f"Unknown agent: {agent_name}")

    def set_behavior_mode(self, mode: BehaviorMode):
        """Sets the operational behavior mode of the coordinator."""
        self.behavior_mode = mode
        self.logger.info(f"Behavior mode set to: {self.behavior_mode.value}")

    def process_data_pipeline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes input data through the full scientific pipeline.
        
        Args:
            data: The input data, expected to contain a 'signal' key.
        
        Returns:
            A dictionary containing the results from each stage of the pipeline.
        """
        self.logger.info(f"Processing data pipeline in {self.behavior_mode.value} mode.")
        
        # 1. Laplace Transform Layer
        laplace_input = {"signal": data.get("signal", [])}
        laplace_result = self.laplace.process(laplace_input)
        
        # 2. KAN Reasoning Layer
        kan_result = self.kan.process(laplace_result)
        
        # 3. PINN Physics Validation Layer
        pinn_result = self.pinn.process(kan_result)
        
        # Compile final results
        final_output = {
            "input_signal_length": len(data.get("signal", [])),
            "laplace_output": laplace_result,
            "kan_output": kan_result,
            "pinn_validation": pinn_result,
            "overall_confidence": self.get_overall_confidence([
                laplace_result.get("confidence", 0.5),
                kan_result.get("confidence", 0.5),
                pinn_result.get("confidence", 0.5)
            ])
        }
        
        return final_output

    def get_overall_confidence(self, confidences: list) -> float:
        """Calculates a weighted average confidence for the pipeline."""
        if not confidences:
            return 0.0
        return calculate_confidence(confidences)

    def get_status(self) -> Dict[str, Any]:
        """Returns the status of the coordinator and its components."""
        return {
            "coordinator_status": "operational",
            "behavior_mode": self.behavior_mode.value,
            "laplace_agent_status": self.laplace.get_status(),
            "kan_agent_status": self.kan.get_status(),
            "pinn_agent_status": self.pinn.get_status(),
        }

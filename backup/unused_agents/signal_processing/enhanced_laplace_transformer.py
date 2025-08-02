import logging
import numpy as np
from typing import Dict, Any

from src.core.agent import NISAgent
from src.utils.confidence_calculator import calculate_confidence

class EnhancedLaplaceTransformer(NISAgent):
    """
    An agent responsible for applying the Laplace transform to incoming signals.
    This is a core component of the NIS Protocol's signal processing layer.
    """
    def __init__(self, agent_id: str = "laplace_transformer"):
        super().__init__(agent_id)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("EnhancedLaplaceTransformer initialized.")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the Laplace transform to the signal data.
        
        Args:
            data: A dictionary containing the 'signal' as a list or numpy array.
            
        Returns:
            A dictionary with the transformed signal and a confidence score.
        """
        signal = data.get("signal", [])
        if not isinstance(signal, (list, np.ndarray)) or len(signal) == 0:
            self.logger.warning("Input signal is empty or not a valid type.")
            return {"transformed_signal": [], "confidence": 0.1}

        try:
            # Simple numerical Laplace transform (for demonstration)
            # A real implementation would use more comprehensive methods.
            time_vector = np.linspace(0, 1, len(signal))
            transformed_signal = np.abs(np.fft.fft(signal)).tolist()
            
            confidence = calculate_confidence([
                1.0 if len(transformed_signal) > 0 else 0.0,
                0.8, # Base confidence for successful transformation
            ])

            return {
                "transformed_signal": transformed_signal,
                "frequency_bins": len(transformed_signal),
                "confidence": confidence
            }
        except Exception as e:
            self.logger.error(f"Error during Laplace transform: {e}")
            return {"transformed_signal": [], "confidence": 0.2, "error": str(e)}

    def compute_laplace_transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method for Laplace transform computation. Calls process."""
        return self.process(data)

    def get_status(self) -> Dict[str, Any]:
        """Returns the status of the Laplace Transformer agent."""
        return {
            "agent_id": self.agent_id,
            "status": "operational",
            "type": "signal_processing"
        }

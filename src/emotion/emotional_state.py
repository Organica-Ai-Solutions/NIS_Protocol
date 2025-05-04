"""
NIS Protocol Emotional State Module

This module implements the emotional state management for the NIS Protocol.
"""

from enum import Enum
from typing import Dict, Any, Optional
import time


class EmotionalDimension(Enum):
    """Emotional dimensions used in the NIS Protocol."""
    
    SUSPICION = "suspicion"  # Increases scrutiny of unusual patterns
    URGENCY = "urgency"      # Prioritizes time-sensitive processing
    CONFIDENCE = "confidence"  # Influences threshold for decision-making
    INTEREST = "interest"    # Directs attention to specific features
    NOVELTY = "novelty"      # Highlights deviation from expectations


class EmotionalState:
    """
    Manages the emotional state of NIS Protocol agents.
    
    Emotional state influences decision-making, attention, and resource allocation
    in the NIS Protocol.
    """
    
    def __init__(self, initial_state: Optional[Dict[str, float]] = None):
        """
        Initialize emotional state with default or provided values.
        
        Args:
            initial_state: Optional initial emotional state values (0.0-1.0)
        """
        # Default neutral state (0.5 for all dimensions)
        self.state = {
            EmotionalDimension.SUSPICION.value: 0.5,
            EmotionalDimension.URGENCY.value: 0.5,
            EmotionalDimension.CONFIDENCE.value: 0.5,
            EmotionalDimension.INTEREST.value: 0.5,
            EmotionalDimension.NOVELTY.value: 0.5
        }
        
        # Override with any provided values
        if initial_state:
            for dimension, value in initial_state.items():
                if dimension in self.state:
                    self.state[dimension] = max(0.0, min(1.0, value))
        
        # Decay rates determine how quickly emotions return to neutral
        self.decay_rates = {
            EmotionalDimension.SUSPICION.value: 0.05,  # Suspicion decays moderately
            EmotionalDimension.URGENCY.value: 0.1,     # Urgency decays quickly
            EmotionalDimension.CONFIDENCE.value: 0.02, # Confidence decays slowly
            EmotionalDimension.INTEREST.value: 0.03,   # Interest decays slowly
            EmotionalDimension.NOVELTY.value: 0.15     # Novelty decays very quickly
        }
        
        # Last update timestamp for decay calculation
        self.last_update = time.time()
    
    def update(self, dimension: str, value: float) -> None:
        """
        Update a single emotional dimension.
        
        Args:
            dimension: The emotional dimension to update
            value: The new value (0.0-1.0)
        """
        # Apply decay first to get current state
        self._apply_decay()
        
        # Ensure dimension exists
        if dimension not in self.state:
            return
        
        # Ensure value is in range [0.0, 1.0]
        value = max(0.0, min(1.0, value))
        
        # Update with exponential moving average (70% current, 30% new)
        current = self.state[dimension]
        self.state[dimension] = 0.7 * current + 0.3 * value
        
        # Update timestamp
        self.last_update = time.time()
    
    def get_state(self) -> Dict[str, float]:
        """
        Get the current emotional state.
        
        Returns:
            Dictionary of emotional dimensions and values
        """
        # Apply decay before returning
        self._apply_decay()
        return self.state.copy()
    
    def get_dimension(self, dimension: str) -> float:
        """
        Get the current value of a specific emotional dimension.
        
        Args:
            dimension: The emotional dimension to get
            
        Returns:
            The current value (0.0-1.0)
        """
        # Apply decay first
        self._apply_decay()
        return self.state.get(dimension, 0.5)
    
    def _apply_decay(self) -> None:
        """Apply time-based decay to all emotional dimensions."""
        current_time = time.time()
        elapsed = current_time - self.last_update
        
        # Only apply decay if some time has passed
        if elapsed < 0.01:
            return
        
        # Apply decay to each dimension
        for dimension, value in self.state.items():
            decay_rate = self.decay_rates.get(dimension, 0.05)
            decay_amount = decay_rate * elapsed
            
            # Move toward neutral (0.5)
            if value > 0.5:
                self.state[dimension] = max(0.5, value - decay_amount)
            elif value < 0.5:
                self.state[dimension] = min(0.5, value + decay_amount)
        
        # Update timestamp
        self.last_update = current_time 
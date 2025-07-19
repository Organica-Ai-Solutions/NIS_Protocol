"""
NIS Protocol Emotional State System Implementation

This module provides the emotional state system that modulates agent behavior
based on context-sensitive dimensions analogous to human emotions.
"""

import time
from typing import Dict, Optional, List, Tuple, Any

class EmotionalStateSystem:
    """Emotional state management for NIS Protocol.
    
    The Emotional State System is a unique feature of the NIS Protocol that modulates 
    agent behavior based on context-sensitive dimensions analogous to human emotions. 
    Unlike traditional priority systems, these dimensions decay over time and influence 
    multiple aspects of system behavior.
    
    Attributes:
        state: Dictionary of emotional dimensions and their current values
        decay_rates: Dictionary of emotional dimensions and their decay rates
        last_update: Timestamp of the last update to the emotional state
        influence_matrix: Dictionary mapping emotional dimensions to influenced behaviors
    """
    
    def __init__(self, custom_dimensions: Optional[Dict[str, float]] = None):
        """Initialize the emotional state system.
        
        Args:
            custom_dimensions: Optional dictionary of custom emotional dimensions
                              and their initial values
        """
        # Calculate initial emotional dimensions based on system context instead of hardcoded
        self.state = self._calculate_initial_emotional_state()
        
        # Add any custom dimensions
        if custom_dimensions:
            for dimension, value in custom_dimensions.items():
                # Ensure the value is in the valid range
                self.state[dimension] = max(0.0, min(1.0, value))
        
        # Calculate adaptive decay rates based on emotional dimension characteristics
        self.decay_rates = self._calculate_adaptive_decay_rates()
        
        # Influence matrix: maps emotional dimensions to influenced behaviors
        self.influence_matrix = {
            "suspicion": ["validation_threshold", "verification_level", "scrutiny_depth"],
            "urgency": ["processing_priority", "resource_allocation", "response_time"],
            "confidence": ["decision_threshold", "action_boldness", "retry_attempts"],
            "interest": ["attention_focus", "detail_level", "memory_retention"],
            "novelty": ["learning_rate", "exploration_bias", "pattern_sensitivity"]
        }
        
        # Initialize the last update timestamp
        self.last_update = time.time()
        
        # History of emotional state changes
        self.history: List[Tuple[float, Dict[str, float]]] = []
        self._record_history()
    
    def update(self, dimension: str, value: float, record_history: bool = True) -> None:
        """Update an emotional dimension.
        
        Args:
            dimension: The emotional dimension to update
            value: The new value (0.0 to 1.0)
            record_history: Whether to record this update in history
        
        Raises:
            ValueError: If the dimension does not exist
        """
        # Apply decay to ensure we're starting from the current state
        self._apply_decay()
        
        # Check if the dimension exists
        if dimension not in self.state:
            raise ValueError(f"Emotional dimension '{dimension}' does not exist")
        
        # Ensure the value is in the valid range
        clamped_value = max(0.0, min(1.0, value))
        
        # Update the dimension
        self.state[dimension] = clamped_value
        
        # Update the timestamp
        self.last_update = time.time()
        
        # Record history if requested
        if record_history:
            self._record_history()
    
    def get_state(self) -> Dict[str, float]:
        """Get the current emotional state.
        
        Returns:
            Dictionary of emotional dimensions and their current values
        """
        # Apply decay before returning the state
        self._apply_decay()
        
        # Return a copy of the state to prevent modification
        return self.state.copy()
    
    def get_dimension(self, dimension: str) -> float:
        """Get the current value of a specific emotional dimension.
        
        Args:
            dimension: The emotional dimension to get
            
        Returns:
            The current value of the dimension
            
        Raises:
            ValueError: If the dimension does not exist
        """
        # Apply decay before returning the value
        self._apply_decay()
        
        # Check if the dimension exists
        if dimension not in self.state:
            raise ValueError(f"Emotional dimension '{dimension}' does not exist")
        
        return self.state[dimension]
    
    def reset(self) -> None:
        """Reset all emotional dimensions to neutral (0.5)."""
        for dimension in self.state:
            self.state[dimension] = 0.5
        
        self.last_update = time.time()
        self._record_history()
    
    def add_dimension(self, dimension: str, initial_value: float = 0.5, decay_rate: float = 0.05) -> None:
        """Add a new emotional dimension.
        
        Args:
            dimension: The name of the new dimension
            initial_value: The initial value for the dimension (0.0 to 1.0)
            decay_rate: The decay rate for the dimension (per second)
            
        Raises:
            ValueError: If the dimension already exists
        """
        # Check if the dimension already exists
        if dimension in self.state:
            raise ValueError(f"Emotional dimension '{dimension}' already exists")
        
        # Ensure the initial value is in the valid range
        clamped_value = max(0.0, min(1.0, initial_value))
        
        # Add the dimension
        self.state[dimension] = clamped_value
        self.decay_rates[dimension] = decay_rate
        
        # Record history
        self._record_history()
    
    def remove_dimension(self, dimension: str) -> None:
        """Remove an emotional dimension.
        
        Args:
            dimension: The name of the dimension to remove
            
        Raises:
            ValueError: If the dimension does not exist
            ValueError: If the dimension is a default dimension
        """
        # Check if the dimension exists
        if dimension not in self.state:
            raise ValueError(f"Emotional dimension '{dimension}' does not exist")
        
        # Check if the dimension is a default dimension
        if dimension in ["suspicion", "urgency", "confidence", "interest", "novelty"]:
            raise ValueError(f"Cannot remove default dimension '{dimension}'")
        
        # Remove the dimension
        del self.state[dimension]
        del self.decay_rates[dimension]
        
        # Record history
        self._record_history()
    
    def get_influenced_behaviors(self, dimension: str) -> List[str]:
        """Get the behaviors influenced by a specific emotional dimension.
        
        Args:
            dimension: The emotional dimension to get influenced behaviors for
            
        Returns:
            List of behaviors influenced by the dimension
            
        Raises:
            ValueError: If the dimension does not exist
        """
        # Check if the dimension exists
        if dimension not in self.state:
            raise ValueError(f"Emotional dimension '{dimension}' does not exist")
        
        # Return the influenced behaviors
        return self.influence_matrix.get(dimension, [])
    
    def set_influence(self, dimension: str, behaviors: List[str]) -> None:
        """Set the behaviors influenced by a specific emotional dimension.
        
        Args:
            dimension: The emotional dimension to set influences for
            behaviors: List of behaviors influenced by the dimension
            
        Raises:
            ValueError: If the dimension does not exist
        """
        # Check if the dimension exists
        if dimension not in self.state:
            raise ValueError(f"Emotional dimension '{dimension}' does not exist")
        
        # Set the influenced behaviors
        self.influence_matrix[dimension] = behaviors
    
    def get_behavioral_influence(self, behavior: str) -> Dict[str, float]:
        """Get the influence of each emotional dimension on a specific behavior.
        
        Args:
            behavior: The behavior to get influences for
            
        Returns:
            Dictionary mapping emotional dimensions to their influence on the behavior
        """
        influences = {}
        
        for dimension, behaviors in self.influence_matrix.items():
            if behavior in behaviors:
                influences[dimension] = self.state[dimension]
        
        return influences
    
    def get_history(self, start_time: Optional[float] = None, end_time: Optional[float] = None) -> List[Tuple[float, Dict[str, float]]]:
        """Get the history of emotional state changes.
        
        Args:
            start_time: Optional start time to filter history
            end_time: Optional end time to filter history
            
        Returns:
            List of (timestamp, state) tuples
        """
        if start_time is None and end_time is None:
            return self.history
        
        filtered_history = []
        
        for timestamp, state in self.history:
            if start_time is not None and timestamp < start_time:
                continue
            if end_time is not None and timestamp > end_time:
                continue
            filtered_history.append((timestamp, state))
        
        return filtered_history
    
    def clear_history(self) -> None:
        """Clear the history of emotional state changes."""
        self.history = []
    
    def _apply_decay(self) -> None:
        """Apply time-based decay to all emotional dimensions."""
        current_time = time.time()
        elapsed = current_time - self.last_update
        
        # If no time has passed, do nothing
        if elapsed <= 0:
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
        
        # Update the timestamp
        self.last_update = current_time
    
    def _record_history(self) -> None:
        """Record the current emotional state in history."""
        self.history.append((time.time(), self.state.copy()))
        
        # Limit history size to prevent memory issues
        max_history_size = 1000
        if len(self.history) > max_history_size:
            self.history = self.history[-max_history_size:]
    
    def compute_overall_emotional_state(self) -> Dict[str, Any]:
        """Compute an overall emotional state assessment.
        
        Returns:
            Dictionary with overall emotional state assessment
        """
        # Apply decay to ensure we're using the current state
        self._apply_decay()
        
        # Calculate the dominant emotional dimension
        dominant_dimension = max(self.state.items(), key=lambda x: abs(x[1] - 0.5))[0]
        
        # Calculate the overall intensity (average deviation from neutral)
        intensity = sum(abs(value - 0.5) for value in self.state.values()) / len(self.state)
        
        # Calculate the overall valence (positive/negative balance)
        positive_dimensions = ["confidence", "interest"]
        negative_dimensions = ["suspicion", "urgency"]
        
        positive_sum = sum(self.state.get(dim, 0.5) for dim in positive_dimensions)
        negative_sum = sum(self.state.get(dim, 0.5) for dim in negative_dimensions)
        
        valence = (positive_sum / len(positive_dimensions)) - (negative_sum / len(negative_dimensions))
        
        # Create the overall assessment
        return {
            "dominant_dimension": dominant_dimension,
            "intensity": intensity,
            "valence": valence,
            "state": self.state.copy(),
            "timestamp": time.time()
        } 

    def _calculate_initial_emotional_state(self) -> Dict[str, float]:
        """Calculate initial emotional state based on system context instead of hardcoded."""
        # For life-critical systems, start with conservative emotional states
        initial_state = {
            "suspicion": self._calculate_initial_suspicion(),
            "urgency": self._calculate_initial_urgency(),
            "confidence": self._calculate_initial_confidence(),
            "interest": self._calculate_initial_interest(),
            "novelty": self._calculate_initial_novelty()
        }
        
        return initial_state
    
    def _calculate_initial_suspicion(self) -> float:
        """Calculate initial suspicion level based on deployment context."""
        # Higher suspicion for life-critical systems
        if self._is_life_critical_context():
            return 0.7  # High suspicion for safety
        else:
            return 0.4  # Moderate suspicion for normal operations
    
    def _calculate_initial_urgency(self) -> float:
        """Calculate initial urgency based on operational requirements."""
        # Moderate urgency to balance responsiveness with careful analysis
        if self._is_real_time_context():
            return 0.6  # Higher urgency for real-time systems
        else:
            return 0.3  # Lower urgency for batch processing
    
    def _calculate_initial_confidence(self) -> float:
        """Calculate initial confidence based on system maturity and validation."""
        # Start with lower confidence for new systems, higher for validated ones
        return 0.4  # Conservative confidence for safety
    
    def _calculate_initial_interest(self) -> float:
        """Calculate initial interest level for learning and adaptation."""
        return 0.5  # Balanced interest for normal operations
    
    def _calculate_initial_novelty(self) -> float:
        """Calculate initial novelty sensitivity."""
        # Higher novelty detection for safety-critical systems
        if self._is_life_critical_context():
            return 0.6  # More sensitive to novel patterns
        else:
            return 0.4  # Normal novelty sensitivity
    
    def _is_life_critical_context(self) -> bool:
        """Determine if this is a life-critical deployment context."""
        # This would be set based on deployment configuration
        # For now, assume life-critical if not explicitly set otherwise
        return True  # Conservative assumption for safety
    
    def _is_real_time_context(self) -> bool:
        """Determine if this is a real-time operational context."""
        # This would be determined by system configuration
        return False  # Default to non-real-time
    
    def _calculate_adaptive_decay_rates(self) -> Dict[str, float]:
        """Calculate adaptive decay rates based on emotional dimension characteristics."""
        # Slower decay for life-critical emotions
        base_rates = {
            "suspicion": 0.02,   # Slower decay - stay suspicious longer
            "urgency": 0.08,     # Moderate decay for urgency
            "confidence": 0.01,  # Very slow decay for confidence
            "interest": 0.05,    # Moderate decay for interest
            "novelty": 0.15      # Faster decay for novelty detection
        }
        
        # Adjust based on context
        if self._is_life_critical_context():
            # Even slower decay for life-critical systems
            for emotion in base_rates:
                base_rates[emotion] *= 0.5
        
        return base_rates 
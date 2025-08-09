"""
NIS Protocol Base Agent Implementation

This module provides the base agent class that all NIS Protocol agents inherit from.
"""

import time
from enum import Enum
from typing import Dict, Any, Optional, List
import logging

class NISLayer(Enum):
    """Enumeration of cognitive layers in the NIS Protocol."""
    PERCEPTION = "perception"
    INTERPRETATION = "interpretation"
    MEMORY = "memory"
    EMOTION = "emotion"
    REASONING = "reasoning"
    ACTION = "action"
    LEARNING = "learning"
    COORDINATION = "coordination"


class NISAgent:
    """Base class for all agents in the NIS Protocol."""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"nis.agent.{self.agent_id}")
        self.logger.info(f"Initialized agent: {self.agent_id}")

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the agent."""
        return {
            "agent_id": self.agent_id,
            "status": "active",
            "timestamp": time.time()
        }
        
    async def process(self, data: Any) -> Any:
        """
        Main processing method for the agent.
        
        Args:
            data: The data to process
            
        Returns:
            The processed data
            
        Raises:
            NotImplementedError: If the subclass has not implemented this method
        """
        raise NotImplementedError("Subclasses must implement process()")
    
    def _start_processing_timer(self) -> float:
        """Start the processing timer.
        
        Returns:
            The start time
        """
        return time.time()
    
    def _end_processing_timer(self, start_time: float) -> float:
        """End the processing timer and update the last processing time.
        
        Args:
            start_time: The start time from _start_processing_timer
            
        Returns:
            The processing time in seconds
        """
        processing_time = time.time() - start_time
        self.last_processing_time = processing_time
        return processing_time
    
    def _create_response(
        self,
        status: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        emotional_state: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Create a standardized response message.
        
        Args:
            status: The status of the processing ("success", "error", or "pending")
            payload: The primary data for the message
            metadata: Additional information about the processing
            emotional_state: The current emotional state
            
        Returns:
            A standardized message dictionary
        """
        if metadata is None:
            metadata = {}
        
        metadata["processing_time"] = self.last_processing_time
        
        return {
            "agent_id": self.agent_id,
            "timestamp": time.time(),
            "status": status,
            "payload": payload,
            "metadata": metadata,
            "emotional_state": emotional_state or {}
        }
    
    def _update_emotional_state(
        self,
        emotional_state: Dict[str, float],
        dimension: str,
        value: float
    ) -> Dict[str, float]:
        """Update an emotional dimension.
        
        Args:
            emotional_state: The current emotional state
            dimension: The emotional dimension to update
            value: The new value (0.0 to 1.0)
            
        Returns:
            The updated emotional state
        """
        # Create a new dictionary to avoid modifying the input
        updated_state = dict(emotional_state)
        
        # Ensure the value is in the valid range
        clamped_value = max(0.0, min(1.0, value))
        
        # Update the dimension
        updated_state[dimension] = clamped_value
        
        return updated_state
    
    def get_id(self) -> str:
        """Get the agent's ID.
        
        Returns:
            The agent's ID
        """
        return self.agent_id
    
    def get_layer(self) -> NISLayer:
        """Get the agent's cognitive layer.
        
        Returns:
            The agent's layer
        """
        return self.layer
    
    def get_description(self) -> str:
        """Get the agent's description.
        
        Returns:
            The agent's description
        """
        return self.description
    
    def is_active(self) -> bool:
        """Check if the agent is active.
        
        Returns:
            True if the agent is active, False otherwise
        """
        return self.active
    
    def set_active(self, active: bool) -> None:
        """Set the agent's active state.
        
        Args:
            active: Whether the agent should be active
        """
        self.active = active 
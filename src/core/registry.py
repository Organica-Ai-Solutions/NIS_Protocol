"""
NIS Protocol Registry

This module implements the central registry for managing NIS Protocol agents.
"""

from enum import Enum
from typing import Dict, List, Optional, Any


class NISLayer(Enum):
    """Layers in the NIS Protocol cognitive hierarchy."""
    
    PERCEPTION = "perception"
    INTERPRETATION = "interpretation"
    REASONING = "reasoning"
    MEMORY = "memory"
    ACTION = "action"
    LEARNING = "learning"
    COORDINATION = "coordination"


class NISAgent:
    """Base class for all NIS Protocol agents."""
    
    def __init__(
        self,
        agent_id: str,
        layer: NISLayer,
        description: str
    ):
        """
        Initialize a NIS Protocol agent.
        
        Args:
            agent_id: Unique identifier for the agent
            layer: Cognitive layer this agent belongs to
            description: Human-readable description of the agent's role
        """
        self.agent_id = agent_id
        self.layer = layer
        self.description = description
        self.active = True
        
        # Register with the global registry
        NISRegistry().register(self)
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming message.
        
        Args:
            message: The message to process
            
        Returns:
            Response message
        """
        raise NotImplementedError("Subclasses must implement process()")


class NISRegistry:
    """Central registry for all NIS Protocol agents."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(NISRegistry, cls).__new__(cls)
            cls._instance.agents = {}
        return cls._instance
    
    def register(self, agent: NISAgent) -> None:
        """
        Register an agent with the registry.
        
        Args:
            agent: The agent to register
        """
        self.agents[agent.agent_id] = agent
    
    def get_agent(self, agent_id: str) -> Optional[NISAgent]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: The ID of the agent to retrieve
            
        Returns:
            The agent, or None if not found
        """
        return self.agents.get(agent_id)
    
    def get_agents_by_layer(self, layer: NISLayer) -> List[NISAgent]:
        """
        Get all agents in a specific layer.
        
        Args:
            layer: The layer to retrieve agents for
            
        Returns:
            List of agents in the specified layer
        """
        return [
            agent for agent in self.agents.values()
            if agent.layer == layer and agent.active
        ]
    
    def deactivate_agent(self, agent_id: str) -> bool:
        """
        Deactivate an agent.
        
        Args:
            agent_id: ID of the agent to deactivate
            
        Returns:
            True if successful, False otherwise
        """
        agent = self.get_agent(agent_id)
        if agent:
            agent.active = False
            return True
        return False 
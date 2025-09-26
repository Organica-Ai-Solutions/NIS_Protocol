"""
NIS Protocol Agent Registry
==========================

Registry for managing NIS Protocol agents.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union

from .agent import NISAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Registry for managing NIS Protocol agents.
    
    The registry keeps track of all agents in the system and provides
    methods for registering, unregistering, and querying agents.
    """
    
    def __init__(self):
        """Initialize a new agent registry."""
        self.agents = {}
        logger.info("Agent registry initialized")
        
    def register(self, agent: NISAgent) -> bool:
        """
        Register an agent with the registry.
        
        Args:
            agent: The agent to register
            
        Returns:
            bool: True if registration was successful
        """
        if agent.agent_id in self.agents:
            logger.warning(f"Agent {agent.agent_id} already registered")
            return False
        
        self.agents[agent.agent_id] = agent
        logger.info(f"Agent {agent.agent_id} registered")
        return True
    
    def unregister(self, agent_id: str) -> bool:
        """
        Unregister an agent from the registry.
        
        Args:
            agent_id: ID of the agent to unregister
            
        Returns:
            bool: True if unregistration was successful
        """
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found in registry")
            return False
        
        del self.agents[agent_id]
        logger.info(f"Agent {agent_id} unregistered")
        return True
    
    def get_agent(self, agent_id: str) -> Optional[NISAgent]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            NISAgent: The agent with the given ID, or None if not found
        """
        return self.agents.get(agent_id)
    
    def get_all_agents(self) -> List[NISAgent]:
        """
        Get all registered agents.
        
        Returns:
            list: All registered agents
        """
        return list(self.agents.values())
    
    def get_agent_ids(self) -> List[str]:
        """
        Get all registered agent IDs.
        
        Returns:
            list: All registered agent IDs
        """
        return list(self.agents.keys())
    
    def get_agent_count(self) -> int:
        """
        Get the number of registered agents.
        
        Returns:
            int: Number of registered agents
        """
        return len(self.agents)
    
    def get_agents_by_layer(self, layer: str) -> List[NISAgent]:
        """
        Get all agents in a specific layer.
        
        Args:
            layer: Layer to filter by
            
        Returns:
            list: Agents in the specified layer
        """
        return [
            agent for agent in self.agents.values()
            if hasattr(agent, 'layer') and (
                agent.layer.value == layer if hasattr(agent.layer, 'value') else str(agent.layer) == layer
            )
        ]
    
    def get_registry_info(self) -> Dict[str, Any]:
        """
        Get registry information.
        
        Returns:
            dict: Registry information
        """
        return {
            "agent_count": len(self.agents),
            "agent_ids": list(self.agents.keys()),
            "layers": self._get_layer_counts(),
        }
    
    def _get_layer_counts(self) -> Dict[str, int]:
        """
        Get count of agents by layer.
        
        Returns:
            dict: Layer counts
        """
        layer_counts = {}
        for agent in self.agents.values():
            if hasattr(agent, 'layer'):
                layer = agent.layer.value if hasattr(agent.layer, 'value') else str(agent.layer)
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
        return layer_counts

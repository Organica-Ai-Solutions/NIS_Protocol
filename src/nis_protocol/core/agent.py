"""
NIS Protocol Agent Core
======================

Base classes and utilities for NIS Protocol agents.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class NISLayer(Enum):
    """NIS Protocol agent layers."""
    PERCEPTION = auto()
    PHYSICS = auto()
    REASONING = auto()
    CONSCIOUSNESS = auto()
    MEMORY = auto()
    ACTION = auto()
    COORDINATION = auto()
    INTEGRATION = auto()


class NISAgent:
    """
    Base class for all NIS Protocol agents.
    
    All agents in the NIS Protocol ecosystem inherit from this class.
    """
    
    def __init__(self, agent_id: str, layer: NISLayer = NISLayer.INTEGRATION):
        """
        Initialize a new NIS agent.
        
        Args:
            agent_id: Unique identifier for this agent
            layer: The NIS layer this agent belongs to
        """
        self.agent_id = agent_id
        self.layer = layer
        self.name = "Base NIS Agent"
        self.description = "Base agent class for NIS Protocol"
        self.created_at = time.time()
        self.last_active = time.time()
        self.status = "initialized"
        self.metrics = {
            "requests_processed": 0,
            "avg_processing_time": 0,
            "total_processing_time": 0,
        }
        
        logger.info(f"Agent {self.agent_id} initialized")
        
    async def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process input data and return results.
        
        Args:
            input_data: Input data to process
            
        Returns:
            dict: Processed result
        """
        start_time = time.time()
        
        # Default implementation - override in subclasses
        result = {
            "agent_id": self.agent_id,
            "layer": self.layer.value if hasattr(self.layer, "value") else str(self.layer),
            "status": "success",
            "result": f"Processed by {self.name}",
            "timestamp": time.time(),
            "content": "Default agent response",
            "text": "Default agent response"
        }
        
        # Update metrics
        processing_time = time.time() - start_time
        self.metrics["requests_processed"] += 1
        self.metrics["total_processing_time"] += processing_time
        self.metrics["avg_processing_time"] = (
            self.metrics["total_processing_time"] / self.metrics["requests_processed"]
        )
        self.last_active = time.time()
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status information.
        
        Returns:
            dict: Agent status
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "layer": self.layer.value if hasattr(self.layer, "value") else str(self.layer),
            "status": self.status,
            "uptime": time.time() - self.created_at,
            "last_active": self.last_active,
            "metrics": self.metrics,
        }
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get detailed agent information.
        
        Returns:
            dict: Agent information
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "layer": self.layer.value if hasattr(self.layer, "value") else str(self.layer),
            "created_at": self.created_at,
            "status": self.status,
            "metrics": self.metrics,
        }
    
    async def initialize(self) -> bool:
        """
        Initialize agent resources.
        
        Returns:
            bool: True if initialization was successful
        """
        self.status = "active"
        return True
    
    async def shutdown(self) -> bool:
        """
        Release agent resources.
        
        Returns:
            bool: True if shutdown was successful
        """
        self.status = "shutdown"
        return True

"""
NIS Protocol Base Adapter

This module provides the base adapter interface that all protocol adapters must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseProtocolAdapter(ABC):
    """Base interface for all protocol adapters.
    
    All protocol adapters must implement this interface to ensure consistent
    behavior and interoperability.
    
    Attributes:
        protocol_name: The name of the external protocol
        config: Configuration for the adapter
    """
    
    def __init__(self, protocol_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize a new protocol adapter.
        
        Args:
            protocol_name: The name of the external protocol
            config: Optional configuration for the adapter
        """
        self.protocol_name = protocol_name
        self.config = config or {}
    
    @abstractmethod
    def translate_to_nis(self, external_message: Dict[str, Any]) -> Dict[str, Any]:
        """Translate a message from the external protocol to NIS Protocol format.
        
        Args:
            external_message: A message in the external protocol format
            
        Returns:
            The message translated to NIS Protocol format
        """
        pass
    
    @abstractmethod
    def translate_from_nis(self, nis_message: Dict[str, Any]) -> Dict[str, Any]:
        """Translate a message from NIS Protocol format to the external protocol format.
        
        Args:
            nis_message: A message in the NIS Protocol format
            
        Returns:
            The message translated to the external protocol format
        """
        pass
    
    @abstractmethod
    def send_to_external_agent(self, agent_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to an external agent using this protocol.
        
        Args:
            agent_id: The ID of the external agent
            message: The message to send
            
        Returns:
            The response from the external agent
        """
        pass
    
    def validate_config(self) -> bool:
        """Validate the adapter configuration.
        
        Returns:
            True if the configuration is valid, False otherwise
        """
        return True 
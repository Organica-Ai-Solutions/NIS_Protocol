"""
NIS Protocol Coordinator Agent

This module provides the CoordinatorAgent class which is responsible for:
1. Routing messages between internal NIS agents
2. Translating between NIS Protocol and external protocols (MCP, ACP, A2A)
3. Managing multi-agent workflows
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union

from core.agent import NISAgent, NISLayer

class CoordinatorAgent(NISAgent):
    """Coordinator Agent for managing inter-agent communication and protocol translation.
    
    This agent acts as a central hub for message routing and protocol translation,
    allowing NIS Protocol agents to interact seamlessly with external protocols.
    
    Attributes:
        agent_id: Unique identifier for the agent
        description: Human-readable description of the agent's purpose
        protocol_adapters: Dictionary of protocol adapters for translation
    """
    
    def __init__(
        self,
        agent_id: str = "coordinator_agent",
        description: str = "Coordinates agent communication and handles protocol translation"
    ):
        """Initialize a new Coordinator agent.
        
        Args:
            agent_id: Unique identifier for the agent
            description: Human-readable description of the agent's purpose
        """
        super().__init__(agent_id, NISLayer.COORDINATION, description)
        self.protocol_adapters = {}
        self.routing_rules = {}
        
    def register_protocol_adapter(self, protocol_name: str, adapter) -> None:
        """Register a protocol adapter.
        
        Args:
            protocol_name: Name of the protocol (e.g., "mcp", "acp", "a2a")
            adapter: The adapter instance for the protocol
        """
        self.protocol_adapters[protocol_name] = adapter
        
    def load_routing_config(self, config_path: str) -> None:
        """Load routing configuration from a file.
        
        Args:
            config_path: Path to the routing configuration file
        """
        try:
            with open(config_path, 'r') as f:
                self.routing_rules = json.load(f)
        except Exception as e:
            print(f"Error loading routing configuration: {e}")
            
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming message.
        
        Determines the appropriate action based on message type and content:
        1. For NIS internal messages, routes to appropriate internal agents
        2. For external protocol messages, translates and routes accordingly
        
        Args:
            message: The incoming message to process
            
        Returns:
            The processed message with routing information
        """
        start_time = self._start_processing_timer()
        
        try:
            # Determine message protocol
            protocol = message.get("protocol", "nis")
            
            # Handle based on protocol
            if protocol == "nis":
                result = self._handle_nis_message(message)
            else:
                result = self._handle_external_protocol_message(message, protocol)
                
            self._end_processing_timer(start_time)
            return self._create_response("success", result)
            
        except Exception as e:
            self._end_processing_timer(start_time)
            return self._create_response(
                "error",
                {"error": str(e), "original_message": message},
                {"exception_type": type(e).__name__}
            )
    
    def _handle_nis_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a message in the NIS Protocol format.
        
        Args:
            message: The NIS Protocol message
            
        Returns:
            The processed message
        """
        # Check if the message needs to be routed to an external protocol
        target_protocol = message.get("target_protocol")
        if target_protocol and target_protocol != "nis":
            if target_protocol in self.protocol_adapters:
                adapter = self.protocol_adapters[target_protocol]
                return adapter.translate_from_nis(message)
            else:
                raise ValueError(f"No adapter registered for protocol: {target_protocol}")
        
        # Route to internal NIS agents
        from src.core.registry import NISRegistry
        registry = NISRegistry()
        
        target_layer = NISLayer[message.get("target_layer", "COORDINATION").upper()]
        target_agent_id = message.get("target_agent_id")
        
        if target_agent_id:
            agent = registry.get_agent_by_id(target_agent_id)
            if agent and agent.is_active():
                return agent.process(message)
            else:
                raise ValueError(f"Agent not found or inactive: {target_agent_id}")
        else:
            responses = registry.process_message(message, target_layer)
            return {"responses": responses}
    
    def _handle_external_protocol_message(
        self,
        message: Dict[str, Any],
        protocol: str
    ) -> Dict[str, Any]:
        """Handle a message from an external protocol.
        
        Args:
            message: The external protocol message
            protocol: The protocol name
            
        Returns:
            The processed message
        """
        if protocol not in self.protocol_adapters:
            raise ValueError(f"No adapter registered for protocol: {protocol}")
        
        adapter = self.protocol_adapters[protocol]
        
        # Translate to NIS format
        nis_message = adapter.translate_to_nis(message)
        
        # Process with internal agents
        result = self._handle_nis_message(nis_message)
        
        # Translate back to original protocol if needed
        if message.get("respond_in_original_protocol", True):
            return adapter.translate_from_nis(result)
        
        return result
    
    def route_to_external_agent(
        self,
        protocol: str,
        agent_id: str,
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route a message to an external agent using the appropriate protocol.
        
        Args:
            protocol: The protocol to use
            agent_id: The ID of the external agent
            message: The message to send
            
        Returns:
            The response from the external agent
        """
        if protocol not in self.protocol_adapters:
            raise ValueError(f"No adapter registered for protocol: {protocol}")
        
        adapter = self.protocol_adapters[protocol]
        return adapter.send_to_external_agent(agent_id, message) 
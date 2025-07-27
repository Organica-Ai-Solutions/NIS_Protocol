"""
NIS Protocol A2A Adapter

This module provides the adapter for Google's Agent2Agent Protocol (A2A).
"""

import json
import time
import requests
from typing import Dict, Any, List, Optional, Union

from .base_adapter import BaseAdapter


class A2AAdapter(BaseAdapter):
    """Adapter for Google's Agent2Agent Protocol (A2A).
    
    This adapter translates between NIS Protocol and A2A, allowing NIS agents
    to interact with external A2A-compliant agents across different platforms.
    
    Attributes:
        protocol_name: The name of the protocol ('a2a')
        config: Configuration for the adapter, including API endpoints
        session: HTTP session for making API requests
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize a new A2A adapter.
        
        Args:
            config: Configuration for the adapter, including API endpoints and auth
        """
        super().__init__("a2a", config)
        self.session = requests.Session()
        
        # Set default headers for API requests
        if self.config.get("api_key"):
            self.session.headers.update({
                "Authorization": f"Bearer {self.config.get('api_key')}",
                "Content-Type": "application/json"
            })
        
        # Cache of discovered agent capabilities
        self.agent_capabilities = {}
    
    def validate_config(self) -> bool:
        """Validate the adapter configuration.
        
        Returns:
            True if the configuration is valid, False otherwise
        """
        required_fields = ["base_url"]
        return all(field in self.config for field in required_fields)
    
    def translate_to_nis(self, a2a_message: Dict[str, Any]) -> Dict[str, Any]:
        """Translate a message from A2A format to NIS Protocol format.
        
        Args:
            a2a_message: A message in the A2A format (Agent Card)
            
        Returns:
            The message translated to NIS Protocol format
        """
        # A2A uses Agent Cards with header and content
        header = a2a_message.get("agentCardHeader", {})
        content = a2a_message.get("agentCardContent", {})
        
        # Extract the action from the content - handle different formats
        action = "unknown_action"
        data = {}
        
        if "actionRequest" in content:
            action = content["actionRequest"].get("actionName", "unknown_action")
            if "arguments" in content["actionRequest"]:
                data = content["actionRequest"]["arguments"]
        elif "actionResponse" in content:
            action = "response"
            if "returnValue" in content["actionResponse"]:
                data = content["actionResponse"]["returnValue"]
        elif "request" in content:
            # Handle simple request format used in tests
            request = content["request"]
            action = request.get("action", "unknown_action")
            data = request.get("data", {})
        
        # Map to NIS format with action at top level
        nis_message = {
            "protocol": "nis",
            "timestamp": time.time(),
            "action": action,  # Action at top level for compatibility
            "source_protocol": "a2a",
            "original_message": a2a_message,
            "payload": {
                "action": action,
                "data": data
            },
            "metadata": {
                "a2a_session_id": header.get("sessionId", ""),
                "a2a_agent_id": header.get("agentId", ""),
                "a2a_message_id": header.get("messageId", "")
            }
        }
        
        # Include state information if present
        if "stateMap" in content:
            nis_message["state"] = content["stateMap"]
        
        return nis_message
    
    def translate_from_nis(self, nis_message: Dict[str, Any]) -> Dict[str, Any]:
        """Translate a message from NIS Protocol format to A2A format.
        
        Args:
            nis_message: A message in the NIS Protocol format
            
        Returns:
            The message translated to A2A Agent Card format
        """
        # Extract data from NIS message
        payload = nis_message.get("payload", {})
        metadata = nis_message.get("metadata", {})
        
        # Create A2A header
        header = {
            "messageId": metadata.get("a2a_message_id", f"nis-{time.time()}"),
            "sessionId": metadata.get("a2a_session_id", ""),
            "agentId": "nis_protocol",
            "version": "1.0"
        }
        
        # Create A2A content based on whether this is a request or response
        content = {}
        
        # Determine if this is a request or response
        is_response = "status" in nis_message and nis_message["status"] in ["success", "error"]
        
        if is_response:
            # This is a response message
            content["actionResponse"] = {
                "returnValue": payload.get("data", {})
            }
            
            # Include error information if present
            if nis_message.get("status") == "error":
                content["actionResponse"]["error"] = {
                    "message": str(payload.get("error", "Unknown error")),
                    "code": "ERROR"
                }
        else:
            # This is a request message
            content["actionRequest"] = {
                "actionName": payload.get("action", "default_action"),
                "arguments": payload.get("data", {})
            }
        
        # Include state information if present
        if "state" in nis_message:
            content["stateMap"] = nis_message["state"]
        
        # Build complete A2A Agent Card
        a2a_message = {
            "agentCardHeader": header,
            "agentCardContent": content
        }
        
        return a2a_message
    
    def discover_agent_capabilities(self, agent_id: str) -> Dict[str, Any]:
        """Discover the capabilities of an A2A agent.
        
        Args:
            agent_id: The ID of the A2A agent
            
        Returns:
            Dictionary of agent capabilities
        """
        if not self.validate_config():
            raise ValueError("A2A adapter configuration is invalid")
        
        # Check cache first
        if agent_id in self.agent_capabilities:
            return self.agent_capabilities[agent_id]
        
        base_url = self.config["base_url"]
        
        # Create a discover capabilities request
        discover_request = {
            "agentCardHeader": {
                "messageId": f"discover-{time.time()}",
                "sessionId": f"session-{time.time()}",
                "agentId": "nis_protocol",
                "version": "1.0"
            },
            "agentCardContent": {
                "discoverRequest": {}
            }
        }
        
        # Make the API request
        try:
            response = self.session.post(f"{base_url}/agents/{agent_id}/discover", json=discover_request)
            response.raise_for_status()
            capabilities = response.json()
            
            # Cache the capabilities
            self.agent_capabilities[agent_id] = capabilities
            
            return capabilities
        except requests.exceptions.RequestException as e:
            return {
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def send_to_external_agent(self, agent_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to an external A2A agent.
        
        Args:
            agent_id: The ID of the external A2A agent
            message: The message to send
            
        Returns:
            The response from the external agent
        """
        if not self.validate_config():
            raise ValueError("A2A adapter configuration is invalid")
        
        base_url = self.config["base_url"]
        
        # Translate to A2A format if not already
        if message.get("source_protocol", "") != "a2a":
            a2a_message = self.translate_from_nis(message)
        else:
            a2a_message = message
        
        # Update the agent ID in the Agent Card header
        a2a_message["agentCardHeader"]["agentId"] = "nis_protocol"
        
        # Make the API request
        try:
            response = self.session.post(f"{base_url}/agents/{agent_id}/exchange", json=a2a_message)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # Create an error Agent Card
            return {
                "agentCardHeader": {
                    "messageId": f"error-{time.time()}",
                    "sessionId": a2a_message["agentCardHeader"].get("sessionId", ""),
                    "agentId": "a2a_adapter",
                    "version": "1.0"
                },
                "agentCardContent": {
                    "actionResponse": {
                        "error": {
                            "message": str(e),
                            "code": "ERROR"
                        }
                    }
                }
            } 
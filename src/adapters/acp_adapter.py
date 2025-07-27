"""
NIS Protocol ACP Adapter

This module provides the adapter for IBM's Agent Communication Protocol (ACP).
"""

import json
import time
import requests
from typing import Dict, Any, List, Optional, Union

from .base_adapter import BaseAdapter


class ACPAdapter(BaseAdapter):
    """Adapter for IBM's Agent Communication Protocol (ACP).
    
    This adapter translates between NIS Protocol and ACP, allowing NIS agents
    to interact with external ACP-compliant agents and systems.
    
    Attributes:
        protocol_name: The name of the protocol ('acp')
        config: Configuration for the adapter, including API endpoints
        session: HTTP session for making API requests
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize a new ACP adapter.
        
        Args:
            config: Configuration for the adapter, including API endpoints and auth
        """
        super().__init__("acp", config)
        self.session = requests.Session()
        
        # Set default headers for API requests
        if self.config.get("api_key"):
            self.session.headers.update({
                "Authorization": f"Bearer {self.config.get('api_key')}",
                "Content-Type": "application/json"
            })
    
    def validate_config(self) -> bool:
        """Validate the adapter configuration.
        
        Returns:
            True if the configuration is valid, False otherwise
        """
        required_fields = ["base_url"]
        return all(field in self.config for field in required_fields)
    
    def translate_to_nis(self, acp_message: Dict[str, Any]) -> Dict[str, Any]:
        """Translate a message from ACP format to NIS Protocol format.
        
        Args:
            acp_message: A message in the ACP format
            
        Returns:
            The message translated to NIS Protocol format
        """
        # ACP messages typically have headers and a body
        headers = acp_message.get("headers", {})
        body = acp_message.get("body", {})
        
        # Map to NIS format
        nis_message = {
            "protocol": "nis",
            "timestamp": time.time(),
            "source_protocol": "acp",
            "original_message": acp_message,
            "payload": {
                "action": headers.get("action", "unknown_action"),
                "data": body
            },
            "metadata": {
                "acp_message_id": headers.get("message_id", ""),
                "acp_sender_id": headers.get("sender_id", ""),
                "acp_conversation_id": headers.get("conversation_id", "")
            }
        }
        
        # Map emotional state if present in ACP message
        if "emotional_state" in body:
            nis_message["emotional_state"] = body["emotional_state"]
        
        return nis_message
    
    def translate_from_nis(self, nis_message: Dict[str, Any]) -> Dict[str, Any]:
        """Translate a message from NIS Protocol format to ACP format.
        
        Args:
            nis_message: A message in the NIS Protocol format
            
        Returns:
            The message translated to ACP format
        """
        # Extract data from NIS message
        payload = nis_message.get("payload", {})
        metadata = nis_message.get("metadata", {})
        
        # Create ACP headers
        headers = {
            "message_id": metadata.get("acp_message_id", f"nis-{time.time()}"),
            "sender_id": "nis_protocol",
            "receiver_id": metadata.get("acp_sender_id", "acp_agent"),
            "conversation_id": metadata.get("acp_conversation_id", ""),
            "timestamp": int(time.time() * 1000),  # ACP uses milliseconds
            "action": payload.get("action", "response")
        }
        
        # Create ACP body
        body = payload.get("data", {})
        
        # Include emotional state if present
        if "emotional_state" in nis_message and nis_message["emotional_state"]:
            body["emotional_state"] = nis_message["emotional_state"]
        
        # Build complete ACP message
        acp_message = {
            "headers": headers,
            "body": body
        }
        
        return acp_message
    
    def send_to_external_agent(self, agent_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to an external ACP agent.
        
        Args:
            agent_id: The ID of the external ACP agent
            message: The message to send
            
        Returns:
            The response from the external agent
        """
        if not self.validate_config():
            raise ValueError("ACP adapter configuration is invalid")
        
        base_url = self.config["base_url"]
        
        # Translate to ACP format if not already
        if message.get("protocol", "") != "acp":
            acp_message = self.translate_from_nis(message)
        else:
            acp_message = message
        
        # Ensure the receiver is set correctly
        acp_message["headers"]["receiver_id"] = agent_id
        
        # Make the API request
        try:
            response = self.session.post(f"{base_url}/agents/{agent_id}/messages", json=acp_message)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # Create an error message in ACP format
            return {
                "headers": {
                    "message_id": f"error-{time.time()}",
                    "sender_id": "acp_adapter",
                    "receiver_id": "nis_protocol",
                    "timestamp": int(time.time() * 1000),
                    "action": "error"
                },
                "body": {
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            } 
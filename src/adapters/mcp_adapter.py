"""
NIS Protocol MCP Adapter

This module provides the adapter for the Model Context Protocol (MCP) by Anthropic.
"""

import json
import time
import requests
from typing import Dict, Any, List, Optional

from .base_adapter import BaseAdapter


class MCPAdapter(BaseAdapter):
    """Adapter for the Model Context Protocol (MCP).
    
    This adapter translates between NIS Protocol and MCP, allowing NIS agents
    to interact with external tools and data sources using the MCP standard.
    
    Attributes:
        protocol_name: The name of the protocol ('mcp')
        config: Configuration for the adapter, including API endpoints
        session: HTTP session for making API requests
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize a new MCP adapter.
        
        Args:
            config: Configuration for the adapter, including API endpoints and auth
        """
        super().__init__("mcp", config)
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
    
    def translate_to_nis(self, mcp_message: Dict[str, Any]) -> Dict[str, Any]:
        """Translate a message from MCP format to NIS Protocol format.
        
        Args:
            mcp_message: A message in the MCP format
            
        Returns:
            The message translated to NIS Protocol format
        """
        # Extract key elements from MCP message
        function_call = mcp_message.get("function_call", {})
        arguments = {}
        
        if "arguments" in function_call:
            # Parse JSON arguments if present
            try:
                if isinstance(function_call["arguments"], str):
                    arguments = json.loads(function_call["arguments"])
                else:
                    arguments = function_call["arguments"]
            except Exception:
                arguments = {"raw_arguments": function_call.get("arguments", "")}
        elif "parameters" in function_call:
            # Handle parameters field as well
            arguments = function_call.get("parameters", {})
        
        # Map to NIS format with action at top level
        nis_message = {
            "protocol": "nis",
            "timestamp": time.time(),
            "action": function_call.get("name", "unknown_action"),  # Action at top level
            "source_protocol": "mcp",
            "original_message": mcp_message,
            "payload": {
                "action": function_call.get("name", "unknown_action"),
                "data": arguments
            },
            "metadata": {
                "mcp_conversation_id": mcp_message.get("conversation_id", ""),
                "mcp_tool_id": mcp_message.get("tool_id", "")
            }
        }
        
        return nis_message
    
    def translate_from_nis(self, nis_message: Dict[str, Any]) -> Dict[str, Any]:
        """Translate a message from NIS Protocol format to MCP format.
        
        Args:
            nis_message: A message in the NIS Protocol format
            
        Returns:
            The message translated to MCP format
        """
        # Extract the payload from NIS message
        payload = nis_message.get("payload", {})
        
        # Prepare MCP message structure
        mcp_message = {
            "tool_response": {
                "content": None,
                "error": None
            }
        }
        
        # Check if there was an error
        if nis_message.get("status") == "error":
            mcp_message["tool_response"]["error"] = {
                "message": str(payload.get("error", "Unknown error")),
                "type": nis_message.get("metadata", {}).get("exception_type", "Error")
            }
        else:
            # Format successful response
            if isinstance(payload, dict):
                mcp_message["tool_response"]["content"] = payload
            else:
                mcp_message["tool_response"]["content"] = {"result": payload}
        
        # Preserve original MCP message IDs if present
        if "metadata" in nis_message and "mcp_conversation_id" in nis_message["metadata"]:
            mcp_message["conversation_id"] = nis_message["metadata"]["mcp_conversation_id"]
        if "metadata" in nis_message and "mcp_tool_id" in nis_message["metadata"]:
            mcp_message["tool_id"] = nis_message["metadata"]["mcp_tool_id"]
            
        return mcp_message
    
    def send_to_external_agent(self, tool_name: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to an external MCP tool.
        
        Args:
            tool_name: The name of the MCP tool to invoke
            message: The message to send
            
        Returns:
            The response from the external tool
        """
        if not self.validate_config():
            raise ValueError("MCP adapter configuration is invalid")
        
        base_url = self.config["base_url"]
        
        # Prepare the request payload
        payload = {
            "tool_id": tool_name,
            "function_call": {
                "name": message.get("payload", {}).get("action", "default_action"),
                "arguments": json.dumps(message.get("payload", {}).get("data", {}))
            }
        }
        
        # Add conversation ID if present
        if "metadata" in message and "mcp_conversation_id" in message["metadata"]:
            payload["conversation_id"] = message["metadata"]["mcp_conversation_id"]
        
        # Make the API request
        try:
            response = self.session.post(f"{base_url}/tools/run", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "tool_response": {
                    "error": {
                        "message": str(e),
                        "type": "RequestError"
                    }
                }
            } 
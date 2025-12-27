"""
NIS Protocol MCP Adapter

This module provides the adapter for the Model Context Protocol (MCP) by Anthropic.
"""

import json
import time
import logging
import requests
from typing import Dict, Any, List, Optional

from .base_adapter import BaseAdapter
from .protocol_errors import (
    ProtocolError,
    ProtocolConnectionError,
    ProtocolTimeoutError,
    ProtocolAuthError,
    ProtocolValidationError,
    get_error_from_response
)
from .protocol_utils import (
    with_retry,
    CircuitBreaker,
    ProtocolMetrics
)

logger = logging.getLogger(__name__)


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
        super().__init__(config or {})
        self.protocol_name = "mcp"  # Set the protocol name
        self.session = requests.Session()
        
        # Enhanced: Track server capabilities and initialization state
        self.server_capabilities = {}
        self.initialized = False
        self.tools_registry = {}
        self.resources_registry = {}
        self.prompts_registry = {}
        self.request_id_counter = 0
        
        # Week 2: Error handling and monitoring
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.get("failure_threshold", 5),
            recovery_timeout=config.get("recovery_timeout", 60),
            success_threshold=config.get("success_threshold", 2)
        )
        self.metrics = ProtocolMetrics(protocol_name="mcp")
        self.default_timeout = config.get("timeout", 30)
        
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
    
    async def handle_mcp_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an MCP chat request.
        
        Args:
            request: The request containing message and optional parameters
            
        Returns:
            Response from MCP processing
        """
        message = request.get("message", "")
        if not message:
            return {
                "status": "error",
                "error": "Message is required"
            }
        
        # Simple echo response for now - can be enhanced with actual MCP tool calls
        return {
            "status": "success",
            "response": f"MCP Adapter received: {message}",
            "mcp_initialized": True,
            "tools_available": list(self.tools_registry.keys()) if self.tools_registry else []
        }
    
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
    
    # =============================================================================
    # ENHANCED MCP PROTOCOL FEATURES (Week 1 Implementation)
    # =============================================================================
    
    def _next_id(self) -> int:
        """Generate next request ID for JSON-RPC"""
        self.request_id_counter += 1
        return self.request_id_counter
    
    @with_retry(max_attempts=3, backoff_base=2.0)
    async def initialize(self) -> Dict[str, Any]:
        """
        Perform MCP initialization handshake per specification.
        
        Enhanced with retry logic, error handling, and metrics tracking.
        
        This implements the complete MCP lifecycle:
        1. Send initialize request with client capabilities
        2. Receive server capabilities
        3. Send initialized notification
        
        Returns:
            Server information including capabilities
            
        Raises:
            ProtocolConnectionError: Failed to connect to server
            ProtocolTimeoutError: Request timed out
            ProtocolAuthError: Authentication failed
            ProtocolValidationError: Invalid response format
        """
        if self.initialized:
            return {"already_initialized": True, "capabilities": self.server_capabilities}
        
        start_time = time.time()
        
        try:
            # Execute with circuit breaker protection
            result = await self.circuit_breaker.call(self._do_initialize)
            
            # Record success
            response_time = time.time() - start_time
            self.metrics.record_request(True, response_time)
            self.metrics.circuit_state = self.circuit_breaker.state.value
            
            logger.info(f"MCP initialization successful in {response_time:.3f}s")
            
            return result
            
        except requests.exceptions.Timeout as e:
            response_time = time.time() - start_time
            self.metrics.record_request(False, response_time, "timeout")
            self.metrics.circuit_state = self.circuit_breaker.state.value
            raise ProtocolTimeoutError(
                "MCP initialization timed out",
                timeout=self.default_timeout
            ) from e
            
        except requests.exceptions.ConnectionError as e:
            response_time = time.time() - start_time
            self.metrics.record_request(False, response_time, "connection")
            self.metrics.circuit_state = self.circuit_breaker.state.value
            raise ProtocolConnectionError(
                f"Failed to connect to MCP server: {e}"
            ) from e
        
        except requests.exceptions.HTTPError as e:
            response_time = time.time() - start_time
            status_code = e.response.status_code if e.response else 0
            
            # Parse error from response
            try:
                error_data = e.response.json()
                specific_error = get_error_from_response(status_code, error_data)
            except:
                specific_error = ProtocolError(f"HTTP {status_code}: {e}")
            
            self.metrics.record_request(False, response_time, type(specific_error).__name__)
            self.metrics.circuit_state = self.circuit_breaker.state.value
            raise specific_error from e
        
        except Exception as e:
            response_time = time.time() - start_time
            self.metrics.record_request(False, response_time, "unknown")
            self.metrics.circuit_state = self.circuit_breaker.state.value
            raise
    
    async def _do_initialize(self) -> Dict[str, Any]:
        """Actual initialization logic (called by circuit breaker)"""
        # Create initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "clientInfo": {
                    "name": "nis-protocol",
                    "version": "3.2"
                }
            }
        }
        
        # Send request with timeout
        base_url = self.config.get("base_url")
        response = self.session.post(
            base_url,
            json=init_request,
            timeout=self.default_timeout
        )
        response.raise_for_status()
        result = response.json()
        
        # Validate response
        if "error" in result:
            raise ProtocolValidationError(
                f"MCP initialization failed: {result['error']}",
                response=result
            )
        
        if "result" not in result:
            raise ProtocolValidationError(
                "Invalid MCP response: missing 'result' field",
                response=result
            )
        
        # Store server capabilities
        self.server_capabilities = result.get("result", {}).get("capabilities", {})
        self.initialized = True
        
        # Send initialized notification
        await self._send_notification("notifications/initialized")
        
        return result
    
    async def _send_notification(self, method: str, params: Optional[Dict[str, Any]] = None):
        """Send JSON-RPC notification (no response expected)"""
        notification = {
            "jsonrpc": "2.0",
            "method": method
        }
        
        if params:
            notification["params"] = params
        
        base_url = self.config.get("base_url")
        self.session.post(base_url, json=notification)
    
    async def discover_tools(self) -> List[Dict[str, Any]]:
        """
        Discover available tools from MCP server.
        
        Returns:
            List of available tools with schemas
        """
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list"
        }
        
        base_url = self.config.get("base_url")
        response = self.session.post(base_url, json=request)
        response.raise_for_status()
        
        result = response.json()
        if "error" in result:
            return []
        
        tools = result.get("result", {}).get("tools", [])
        
        # Update registry
        for tool in tools:
            self.tools_registry[tool["name"]] = tool
        
        return tools
    
    async def discover_resources(self) -> List[Dict[str, Any]]:
        """
        Discover available resources from MCP server.
        
        Resources provide contextual data (files, DB records, API responses).
        
        Returns:
            List of available resources
        """
        if not self.server_capabilities.get("resources"):
            return []
        
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "resources/list"
        }
        
        base_url = self.config.get("base_url")
        response = self.session.post(base_url, json=request)
        response.raise_for_status()
        
        result = response.json()
        if "error" in result:
            return []
        
        resources = result.get("result", {}).get("resources", [])
        
        # Update registry
        for resource in resources:
            self.resources_registry[resource["uri"]] = resource
        
        return resources
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """
        Read a specific resource from MCP server.
        
        Args:
            uri: Resource URI (e.g., "file:///path/to/file")
            
        Returns:
            Resource contents
        """
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "resources/read",
            "params": {
                "uri": uri
            }
        }
        
        base_url = self.config.get("base_url")
        response = self.session.post(base_url, json=request)
        response.raise_for_status()
        
        result = response.json()
        if "error" in result:
            raise Exception(f"Resource read failed: {result['error']}")
        
        return result.get("result", {})
    
    async def discover_prompts(self) -> List[Dict[str, Any]]:
        """
        Discover available prompt templates from MCP server.
        
        Prompts are reusable templates for structuring interactions.
        
        Returns:
            List of available prompts
        """
        if not self.server_capabilities.get("prompts"):
            return []
        
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "prompts/list"
        }
        
        base_url = self.config.get("base_url")
        response = self.session.post(base_url, json=request)
        response.raise_for_status()
        
        result = response.json()
        if "error" in result:
            return []
        
        prompts = result.get("result", {}).get("prompts", [])
        
        # Update registry
        for prompt in prompts:
            self.prompts_registry[prompt["name"]] = prompt
        
        return prompts
    
    async def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get a specific prompt with arguments filled in.
        
        Args:
            name: Prompt name
            arguments: Optional arguments to fill in the prompt
            
        Returns:
            Prompt with filled arguments
        """
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "prompts/get",
            "params": {
                "name": name,
                "arguments": arguments or {}
            }
        }
        
        base_url = self.config.get("base_url")
        response = self.session.post(base_url, json=request)
        response.raise_for_status()
        
        result = response.json()
        if "error" in result:
            raise Exception(f"Prompt retrieval failed: {result['error']}")
        
        return result.get("result", {})
    
    # =============================================================================
    # HEALTH & MONITORING (Week 2 Implementation)
    # =============================================================================
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get adapter health status and metrics.
        
        Returns health information including:
        - Initialization state
        - Circuit breaker state
        - Performance metrics
        - Error rates
        
        Returns:
            Health status dictionary
        """
        return {
            "protocol": self.protocol_name,
            "initialized": self.initialized,
            "healthy": (
                self.initialized and
                self.circuit_breaker.state.value == "closed" and
                self.metrics.success_rate > 0.9
            ),
            "circuit_breaker": self.circuit_breaker.get_state(),
            "metrics": self.metrics.to_dict(),
            "capabilities": {
                "tools": len(self.tools_registry),
                "resources": len(self.resources_registry),
                "prompts": len(self.prompts_registry)
            }
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics.reset()
        logger.info("MCP adapter metrics reset")
    
    def reset_circuit_breaker(self):
        """Manually reset circuit breaker to closed state"""
        self.circuit_breaker.reset()
        logger.info("MCP adapter circuit breaker reset") 
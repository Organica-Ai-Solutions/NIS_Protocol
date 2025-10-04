"""
NIS Protocol A2A Adapter

This module provides the adapter for Google's Agent2Agent Protocol (A2A).
"""

import json
import time
import uuid
import asyncio
import logging
import requests
from typing import Dict, Any, List, Optional, Union

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
        super().__init__(config or {})
        self.protocol_name = "a2a"  # Set the protocol name
        self.session = requests.Session()
        
        # Set default headers for API requests
        if self.config.get("api_key"):
            self.session.headers.update({
                "Authorization": f"Bearer {self.config.get('api_key')}",
                "Content-Type": "application/json"
            })
        
        # Enhanced: Task lifecycle management
        self.active_tasks = {}
        self.session_id = str(uuid.uuid4())
        
        # Week 2: Error handling and monitoring
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.get("failure_threshold", 5),
            recovery_timeout=config.get("recovery_timeout", 60),
            success_threshold=config.get("success_threshold", 2)
        )
        self.metrics = ProtocolMetrics(protocol_name="a2a")
        self.default_timeout = config.get("timeout", 30)
        
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
    
    # =============================================================================
    # ENHANCED A2A PROTOCOL FEATURES (Week 1 Implementation)
    # =============================================================================
    
    @with_retry(max_attempts=3, backoff_base=2.0)
    async def create_task(
        self,
        description: str,
        agent_id: str,
        parameters: Dict[str, Any],
        callback_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a long-running A2A task.
        
        Enhanced with retry logic, error handling, and metrics tracking.
        
        Tasks enable workflows that can take hours or days to complete,
        with progress tracking and artifact generation.
        
        Args:
            description: Task description
            agent_id: ID of the A2A agent to execute the task
            parameters: Task parameters
            callback_url: Optional callback URL for status updates
            
        Returns:
            Task object with task_id and status
            
        Raises:
            ProtocolConnectionError: Failed to connect to server
            ProtocolTimeoutError: Request timed out
            ProtocolAuthError: Authentication failed
            ProtocolValidationError: Invalid response format
        """
        if not self.validate_config():
            raise ValueError("A2A adapter configuration is invalid")
        
        start_time = time.time()
        task_id = str(uuid.uuid4())
        
        try:
            # Execute with circuit breaker protection
            result = await self.circuit_breaker.call(
                self._do_create_task,
                task_id,
                description,
                agent_id,
                parameters,
                callback_url
            )
            
            # Record success
            response_time = time.time() - start_time
            self.metrics.record_request(True, response_time)
            self.metrics.circuit_state = self.circuit_breaker.state.value
            
            logger.info(f"A2A task created: {task_id} in {response_time:.3f}s")
            
            return result
            
        except requests.exceptions.Timeout as e:
            response_time = time.time() - start_time
            self.metrics.record_request(False, response_time, "timeout")
            self.metrics.circuit_state = self.circuit_breaker.state.value
            raise ProtocolTimeoutError(
                "A2A task creation timed out",
                timeout=self.default_timeout
            ) from e
            
        except requests.exceptions.ConnectionError as e:
            response_time = time.time() - start_time
            self.metrics.record_request(False, response_time, "connection")
            self.metrics.circuit_state = self.circuit_breaker.state.value
            raise ProtocolConnectionError(
                f"Failed to connect to A2A server: {e}"
            ) from e
        
        except requests.exceptions.HTTPError as e:
            response_time = time.time() - start_time
            status_code = e.response.status_code if e.response else 0
            
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
    
    async def _do_create_task(
        self,
        task_id: str,
        description: str,
        agent_id: str,
        parameters: Dict[str, Any],
        callback_url: Optional[str]
    ) -> Dict[str, Any]:
        """Actual task creation logic (called by circuit breaker)"""
        # Create task request per A2A spec
        request = {
            "agentCardHeader": {
                "messageId": str(uuid.uuid4()),
                "sessionId": self.session_id,
                "agentId": "nis_protocol",
                "version": "1.0"
            },
            "agentCardContent": {
                "taskRequest": {
                    "taskId": task_id,
                    "description": description,
                    "parameters": parameters
                }
            }
        }
        
        if callback_url:
            request["agentCardContent"]["taskRequest"]["callbackUrl"] = callback_url
        
        # Send request with timeout
        base_url = self.config["base_url"]
        response = self.session.post(
            f"{base_url}/agents/{agent_id}/tasks",
            json=request,
            timeout=self.default_timeout
        )
        response.raise_for_status()
        
        # Track task
        self.active_tasks[task_id] = {
            "status": "pending",
            "agent_id": agent_id,
            "description": description,
            "created_at": time.time(),
            "progress": 0.0,
            "artifacts": []
        }
        
        return {
            "task_id": task_id,
            "status": "pending",
            "agent_id": agent_id,
            "description": description
        }
    
    async def get_task_status(
        self,
        task_id: str,
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Get current status of a task.
        
        Args:
            task_id: Task ID
            agent_id: Agent ID
            
        Returns:
            Task status including progress and artifacts
        """
        if not self.validate_config():
            raise ValueError("A2A adapter configuration is invalid")
        
        request = {
            "agentCardHeader": {
                "messageId": str(uuid.uuid4()),
                "sessionId": self.session_id,
                "agentId": "nis_protocol",
                "version": "1.0"
            },
            "agentCardContent": {
                "taskStatusRequest": {
                    "taskId": task_id
                }
            }
        }
        
        base_url = self.config["base_url"]
        response = self.session.post(
            f"{base_url}/agents/{agent_id}/tasks/{task_id}/status",
            json=request
        )
        response.raise_for_status()
        
        result = response.json()
        task_status = result.get("taskStatus", {})
        
        # Update local tracking
        if task_id in self.active_tasks:
            self.active_tasks[task_id].update({
                "status": task_status.get("status", "pending"),
                "progress": task_status.get("progress", 0.0),
                "artifacts": task_status.get("artifacts", []),
                "updated_at": time.time()
            })
        
        return {
            "task_id": task_id,
            "status": task_status.get("status", "pending"),
            "progress": task_status.get("progress", 0.0),
            "artifacts": task_status.get("artifacts", []),
            "error": task_status.get("error")
        }
    
    async def cancel_task(
        self,
        task_id: str,
        agent_id: str
    ) -> bool:
        """
        Cancel a running task.
        
        Args:
            task_id: Task ID
            agent_id: Agent ID
            
        Returns:
            True if successfully cancelled
        """
        if not self.validate_config():
            raise ValueError("A2A adapter configuration is invalid")
        
        request = {
            "agentCardHeader": {
                "messageId": str(uuid.uuid4()),
                "sessionId": self.session_id,
                "agentId": "nis_protocol",
                "version": "1.0"
            },
            "agentCardContent": {
                "taskCancelRequest": {
                    "taskId": task_id
                }
            }
        }
        
        try:
            base_url = self.config["base_url"]
            response = self.session.post(
                f"{base_url}/agents/{agent_id}/tasks/{task_id}/cancel",
                json=request
            )
            response.raise_for_status()
            
            # Update local tracking
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["status"] = "cancelled"
                self.active_tasks[task_id]["updated_at"] = time.time()
            
            return True
            
        except requests.exceptions.RequestException:
            return False
    
    async def wait_for_task_completion(
        self,
        task_id: str,
        agent_id: str,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Wait for task to complete with polling.
        
        Args:
            task_id: Task ID
            agent_id: Agent ID
            poll_interval: Seconds between status checks
            timeout: Optional timeout in seconds
            
        Returns:
            Completed task status
            
        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        start_time = time.time()
        
        while True:
            status = await self.get_task_status(task_id, agent_id)
            
            if status["status"] in ["completed", "failed", "cancelled"]:
                return status
            
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Task {task_id} timed out after {timeout}s")
            
            await asyncio.sleep(poll_interval)
    
    def create_message_with_parts(
        self,
        parts: List[Dict[str, Any]],
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Create A2A message with rich content parts for UX negotiation.
        
        Enables sending multi-format content:
        - Text: analysis results
        - Image: visualization charts
        - Video: demonstrations
        - Iframe: interactive dashboards
        - Form: user input requests
        
        Example:
            parts = [
                {"contentType": "text", "content": "Analysis complete"},
                {"contentType": "image", "content": "https://chart.png"},
                {"contentType": "iframe", "content": "https://dashboard.html"}
            ]
            message = adapter.create_message_with_parts(parts, "reporting_agent")
        
        Args:
            parts: List of content parts with contentType and content
            agent_id: Target agent ID
            
        Returns:
            A2A message with parts
        """
        return {
            "agentCardHeader": {
                "messageId": str(uuid.uuid4()),
                "sessionId": self.session_id,
                "agentId": "nis_protocol",
                "version": "1.0"
            },
            "agentCardContent": {
                "message": {
                    "parts": parts
                }
            }
        }
    
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
        - Active tasks count
        
        Returns:
            Health status dictionary
        """
        return {
            "protocol": self.protocol_name,
            "session_id": self.session_id,
            "healthy": (
                self.circuit_breaker.state.value == "closed" and
                self.metrics.success_rate > 0.9
            ),
            "circuit_breaker": self.circuit_breaker.get_state(),
            "metrics": self.metrics.to_dict(),
            "active_tasks": {
                "count": len(self.active_tasks),
                "tasks": {
                    task_id: {
                        "status": task["status"],
                        "progress": task["progress"],
                        "agent_id": task["agent_id"]
                    }
                    for task_id, task in self.active_tasks.items()
                }
            },
            "discovered_agents": len(self.agent_capabilities)
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics.reset()
        logger.info("A2A adapter metrics reset")
    
    def reset_circuit_breaker(self):
        """Manually reset circuit breaker to closed state"""
        self.circuit_breaker.reset()
        logger.info("A2A adapter circuit breaker reset") 
"""
NIS Protocol ACP Adapter

This module provides the adapter for IBM's Agent Communication Protocol (ACP).
Enhanced with production-grade error handling, retry logic, and monitoring.
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
        super().__init__(config or {})
        self.protocol_name = "acp"
        self.session = requests.Session()
        
        # Error handling and monitoring (Week 2)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.get("failure_threshold", 5),
            recovery_timeout=config.get("recovery_timeout", 60),
            success_threshold=config.get("success_threshold", 2)
        )
        self.metrics = ProtocolMetrics(protocol_name="acp")
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
    
    # =============================================================================
    # ACP AGENT CARD (Week 3 - Offline Discovery)
    # =============================================================================
    
    def export_agent_card(self) -> Dict[str, Any]:
        """
        Export NIS Protocol Agent Card for offline discovery.
        
        Per IBM ACP spec, this metadata can be embedded in package.json
        to enable scale-to-zero and offline discovery.
        
        Returns:
            Agent Card dictionary with NIS Protocol capabilities
        """
        return {
            "acp": {
                "version": "1.0",
                "agent": {
                    "id": "nis_protocol_v3.2",
                    "name": "NIS Protocol",
                    "version": "3.2",
                    "description": "Physics-informed AI protocol with Laplace→KAN→PINN→LLM pipeline",
                    "capabilities": [
                        "physics_validation",
                        "symbolic_reasoning",
                        "consciousness_assessment",
                        "multi_llm_orchestration",
                        "laplace_signal_processing",
                        "kan_interpretability",
                        "pinn_physics_constraints"
                    ],
                    "endpoints": {
                        "base": self.config.get("base_url", "http://localhost:5000"),
                        "execute": "/api/acp/execute",
                        "status": "/api/acp/status",
                        "capabilities": "/api/acp/capabilities"
                    },
                    "authentication": {
                        "type": "bearer",
                        "required": bool(self.config.get("api_key"))
                    },
                    "metadata": {
                        "framework": "nis-protocol",
                        "pipeline_stages": ["laplace", "kan", "pinn", "llm"],
                        "supports_async": True,
                        "supports_streaming": True,
                        "language": "python",
                        "runtime": "docker"
                    }
                }
            }
        }
    
    @with_retry(max_attempts=3, backoff_base=2.0)
    async def execute_agent(
        self,
        agent_url: str,
        message: Dict[str, Any],
        async_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Execute ACP agent with proper REST conventions.
        
        ACP uses simple REST per IBM spec:
        POST {agent_url}/execute
        
        Args:
            agent_url: Base URL of the ACP agent
            message: Message to send
            async_mode: True for async execution (default per ACP spec)
            
        Returns:
            Agent response
            
        Raises:
            ProtocolConnectionError: Failed to connect
            ProtocolTimeoutError: Request timed out
        """
        start_time = time.time()
        
        try:
            result = await self.circuit_breaker.call(
                self._do_execute_agent,
                agent_url,
                message,
                async_mode
            )
            
            response_time = time.time() - start_time
            self.metrics.record_request(True, response_time)
            self.metrics.circuit_state = self.circuit_breaker.state.value
            
            logger.info(f"ACP execute successful in {response_time:.3f}s")
            
            return result
            
        except requests.exceptions.Timeout as e:
            response_time = time.time() - start_time
            self.metrics.record_request(False, response_time, "timeout")
            self.metrics.circuit_state = self.circuit_breaker.state.value
            raise ProtocolTimeoutError(
                "ACP execute timed out",
                timeout=self.default_timeout
            ) from e
            
        except requests.exceptions.ConnectionError as e:
            response_time = time.time() - start_time
            self.metrics.record_request(False, response_time, "connection")
            self.metrics.circuit_state = self.circuit_breaker.state.value
            raise ProtocolConnectionError(
                f"Failed to connect to ACP agent: {e}"
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
    
    async def _do_execute_agent(
        self,
        agent_url: str,
        message: Dict[str, Any],
        async_mode: bool
    ) -> Dict[str, Any]:
        """Actual execution logic (called by circuit breaker)"""
        if async_mode:
            # Async execution (default for ACP)
            response = self.session.post(
                f"{agent_url}/execute",
                json={
                    "input": message,
                    "mode": "async",
                    "callback_url": f"{self.config.get('base_url', 'http://localhost:5000')}/callbacks"
                },
                timeout=self.default_timeout
            )
            response.raise_for_status()
            result = response.json()
            
            # Poll for result if task_id provided
            if result.get("task_id"):
                return await self._poll_task_result(agent_url, result["task_id"])
            
            return result
        else:
            # Synchronous execution
            response = self.session.post(
                f"{agent_url}/execute",
                json={
                    "input": message,
                    "mode": "sync"
                },
                timeout=self.default_timeout
            )
            response.raise_for_status()
            return response.json()
    
    async def _poll_task_result(
        self,
        agent_url: str,
        task_id: str,
        poll_interval: float = 2.0,
        timeout: float = 300.0
    ) -> Dict[str, Any]:
        """Poll for async task result"""
        start_time = time.time()
        
        while True:
            if (time.time() - start_time) > timeout:
                raise ProtocolTimeoutError(f"Task {task_id} timed out", timeout=timeout)
            
            response = self.session.get(
                f"{agent_url}/results/{task_id}",
                timeout=self.default_timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") in ["completed", "failed"]:
                return result
            
            await asyncio.sleep(poll_interval)
    
    # =============================================================================
    # HEALTH & MONITORING (Week 2)
    # =============================================================================
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get adapter health status and metrics.
        
        Returns:
            Health status dictionary
        """
        return {
            "protocol": self.protocol_name,
            "healthy": (
                self.circuit_breaker.state.value == "closed" and
                self.metrics.success_rate > 0.9
            ),
            "circuit_breaker": self.circuit_breaker.get_state(),
            "metrics": self.metrics.to_dict(),
            "agent_card": {
                "id": "nis_protocol_v3.2",
                "capabilities_count": 7
            }
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics.reset()
        logger.info("ACP adapter metrics reset")
    
    def reset_circuit_breaker(self):
        """Manually reset circuit breaker to closed state"""
        self.circuit_breaker.reset()
        logger.info("ACP adapter circuit breaker reset") 
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from datetime import datetime
import json
import os
import requests
from enum import Enum

from ..base_neural_agent import NeuralAgent, NeuralLayer, NeuralSignal

class ProtocolType(Enum):
    """Supported communication protocols"""
    A2A = "a2a"  # Agent-to-Agent Protocol
    ACP = "acp"  # Agent Computing Protocol
    MCP = "mcp"  # Managed Compute Protocol (for completeness)

@dataclass
class ProtocolConfig:
    """Configuration for a communication protocol"""
    protocol_type: ProtocolType
    base_url: str
    api_key: str
    version: str = "v1"
    headers: Dict[str, str] = None
    timeout: int = 30

@dataclass
class CommunicationMessage:
    """Represents a message in the communication system"""
    protocol: ProtocolType
    message_type: str
    content: Dict[str, Any]
    sender: str
    receiver: str = None
    timestamp: datetime = datetime.now()
    metadata: Dict[str, Any] = None

class A2AProtocolHandler:
    """Handles Agent-to-Agent Protocol communications"""
    
    def __init__(self, config: ProtocolConfig):
        self.config = config
        self.headers = config.headers or {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "X-Protocol-Version": config.version
        }
    
    def send_message(self, message: CommunicationMessage) -> Dict[str, Any]:
        """Send a message using A2A protocol"""
        endpoint = f"{self.config.base_url}/agents/{message.receiver}/messages"
        
        payload = {
            "message_type": message.message_type,
            "content": message.content,
            "sender": message.sender,
            "metadata": message.metadata or {}
        }
        
        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"A2A message sending failed: {str(e)}")
    
    def get_messages(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get messages for an agent"""
        endpoint = f"{self.config.base_url}/agents/{agent_id}/messages"
        
        try:
            response = requests.get(
                endpoint,
                headers=self.headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()["messages"]
        except Exception as e:
            raise Exception(f"A2A message retrieval failed: {str(e)}")

class ACPProtocolHandler:
    """Handles Agent Computing Protocol communications"""
    
    def __init__(self, config: ProtocolConfig):
        self.config = config
        self.headers = config.headers or {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "X-Protocol-Version": config.version
        }
    
    def execute_computation(self, computation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a computation using ACP"""
        endpoint = f"{self.config.base_url}/compute"
        
        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=computation,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"ACP computation failed: {str(e)}")
    
    def get_computation_status(self, computation_id: str) -> Dict[str, Any]:
        """Get status of a computation"""
        endpoint = f"{self.config.base_url}/compute/{computation_id}"
        
        try:
            response = requests.get(
                endpoint,
                headers=self.headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"ACP status check failed: {str(e)}")
    
    def cancel_computation(self, computation_id: str) -> bool:
        """Cancel a running computation"""
        endpoint = f"{self.config.base_url}/compute/{computation_id}/cancel"
        
        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return True
        except Exception:
            return False

class CommunicationAgent(NeuralAgent):
    """Agent responsible for inter-agent communication using various protocols"""
    
    def __init__(
        self,
        agent_id: str = "communication_agent",
        a2a_config: Optional[ProtocolConfig] = None,
        acp_config: Optional[ProtocolConfig] = None,
        message_queue_size: int = 100
    ):
        """Initialize the communication agent.
        
        Args:
            agent_id: Unique identifier for this agent
            a2a_config: Configuration for A2A protocol
            acp_config: Configuration for ACP protocol
            message_queue_size: Maximum number of messages to queue
        """
        super().__init__(
            agent_id=agent_id,
            layer=NeuralLayer.EXECUTIVE,  # Communication happens at executive level
            description="Handles inter-agent communication using A2A and ACP protocols"
        )
        
        # Initialize protocol handlers
        self.a2a_handler = A2AProtocolHandler(a2a_config) if a2a_config else None
        self.acp_handler = ACPProtocolHandler(acp_config) if acp_config else None
        
        # Message queue and history
        self.message_queue: List[CommunicationMessage] = []
        self.message_history: List[CommunicationMessage] = []
        self.message_queue_size = message_queue_size
        
        # Track active computations
        self.active_computations: Dict[str, Dict[str, Any]] = {}
    
    def process_signal(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Process incoming signal for communication"""
        if not isinstance(signal.content, dict):
            return None
            
        # Extract communication request
        comm_request = signal.content.get("communication", {})
        protocol = comm_request.get("protocol")
        action = comm_request.get("action")
        
        if not protocol or not action:
            return None
            
        try:
            # Handle A2A protocol requests
            if protocol == ProtocolType.A2A.value:
                if not self.a2a_handler:
                    raise Exception("A2A protocol not configured")
                    
                if action == "send_message":
                    message = CommunicationMessage(
                        protocol=ProtocolType.A2A,
                        message_type=comm_request.get("message_type", "default"),
                        content=comm_request.get("content", {}),
                        sender=self.agent_id,
                        receiver=comm_request.get("receiver"),
                        metadata=comm_request.get("metadata")
                    )
                    
                    result = self.a2a_handler.send_message(message)
                    self._add_to_history(message)
                    
                    return self._create_response_signal("message_sent", result)
                    
                elif action == "get_messages":
                    agent_id = comm_request.get("agent_id")
                    messages = self.a2a_handler.get_messages(agent_id)
                    
                    return self._create_response_signal("messages_retrieved", {
                        "agent_id": agent_id,
                        "messages": messages
                    })
            
            # Handle ACP protocol requests
            elif protocol == ProtocolType.ACP.value:
                if not self.acp_handler:
                    raise Exception("ACP protocol not configured")
                    
                if action == "execute_computation":
                    computation = comm_request.get("computation", {})
                    result = self.acp_handler.execute_computation(computation)
                    
                    # Track computation
                    computation_id = result.get("computation_id")
                    if computation_id:
                        self.active_computations[computation_id] = {
                            "status": "running",
                            "start_time": datetime.now(),
                            "computation": computation
                        }
                    
                    return self._create_response_signal("computation_started", result)
                    
                elif action == "check_status":
                    computation_id = comm_request.get("computation_id")
                    status = self.acp_handler.get_computation_status(computation_id)
                    
                    # Update tracked status
                    if computation_id in self.active_computations:
                        self.active_computations[computation_id]["status"] = status.get("status")
                    
                    return self._create_response_signal("computation_status", status)
                    
                elif action == "cancel_computation":
                    computation_id = comm_request.get("computation_id")
                    cancelled = self.acp_handler.cancel_computation(computation_id)
                    
                    if cancelled and computation_id in self.active_computations:
                        self.active_computations[computation_id]["status"] = "cancelled"
                    
                    return self._create_response_signal("computation_cancelled", {
                        "computation_id": computation_id,
                        "cancelled": cancelled
                    })
            
            return self._create_response_signal("error", {
                "error": f"Unknown protocol or action: {protocol}/{action}"
            })
            
        except Exception as e:
            return self._create_response_signal("error", {
                "error": str(e)
            })
    
    def _create_response_signal(
        self,
        response_type: str,
        content: Dict[str, Any]
    ) -> NeuralSignal:
        """Create a response signal"""
        return NeuralSignal(
            source_layer=self.layer,
            target_layer=NeuralLayer.EXECUTIVE,
            content={
                "response_type": response_type,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def _add_to_history(self, message: CommunicationMessage):
        """Add a message to history"""
        self.message_history.append(message)
        if len(self.message_history) > self.message_queue_size:
            self.message_history = self.message_history[-self.message_queue_size:]
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get statistics about communication"""
        return {
            "a2a_available": self.a2a_handler is not None,
            "acp_available": self.acp_handler is not None,
            "message_history_size": len(self.message_history),
            "active_computations": len(self.active_computations),
            "computation_statuses": {
                comp_id: data["status"]
                for comp_id, data in self.active_computations.items()
            }
        }
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.message_queue.clear()
        self.message_history.clear()
        self.active_computations.clear() 
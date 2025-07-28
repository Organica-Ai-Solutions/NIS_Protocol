"""
NIS Protocol Integration Example

This example demonstrates how to integrate NIS Protocol with external protocols:
- MCP (Anthropic's Model Context Protocol)
- ACP (IBM's Agent Communication Protocol)
- A2A (Google's Agent2Agent Protocol)

The example sets up a coordinator agent and demonstrates sending and receiving
messages between NIS agents and external protocol agents.
"""

import json
import os
import sys
import time
from typing import Dict, Any

# Add parent directory to path to import NIS Protocol
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.agent import NISAgent, NISLayer
from src.core.registry import NISRegistry
from src.agents.coordination.coordinator_agent import CoordinatorAgent
from src.adapters.bootstrap import configure_coordinator_agent


class ExampleVisionAgent(NISAgent):
    """Example agent that processes image data."""
    
    def __init__(self, agent_id: str = "vision_agent"):
        super().__init__(agent_id, NISLayer.PERCEPTION, "Processes visual data")
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        start_time = self._start_processing_timer()
        
        # Extract image data from the message
        image_data = message.get("payload", {}).get("data", {}).get("image_data")
        
        result = {
            "detected_objects": ["person", "car", "tree"],
            "confidence_scores": [0.92, 0.87, 0.76]
        }
        
        # If this is from an external protocol, add protocol-specific metadata
        if "source_protocol" in message:
            result["source"] = f"Processed from {message['source_protocol']}"
        
        self._end_processing_timer(start_time)
        return self._create_response("success", result)


class ExampleMemoryAgent(NISAgent):
    """Example agent that manages memory storage and retrieval."""
    
    def __init__(self, agent_id: str = "memory_agent"):
        super().__init__(agent_id, NISLayer.MEMORY, "Manages memory storage and retrieval")
        self.memory_store = {}
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        start_time = self._start_processing_timer()
        
        # Extract action and data from the message
        payload = message.get("payload", {})
        action = payload.get("action", "unknown")
        data = payload.get("data", {})
        
        result = {}
        
        if action == "store":
            key = data.get("key")
            value = data.get("value")
            if key and value:
                self.memory_store[key] = value
                result = {"status": "stored", "key": key}
            else:
                result = {"error": "Missing key or value"}
        
        elif action == "retrieve":
            key = data.get("key")
            if key in self.memory_store:
                result = {"value": self.memory_store[key]}
            else:
                result = {"error": f"Key not found: {key}"}
        
        elif action == "list":
            result = {"keys": list(self.memory_store.keys())}
        
        else:
            result = {"error": f"Unknown action: {action}"}
        
        self._end_processing_timer(start_time)
        return self._create_response("success", result)


def setup_system():
    """Set up the NIS Protocol system with external protocol integration."""
    # Create coordinator agent
    coordinator = CoordinatorAgent()
    
    # Register example agents
    vision_agent = ExampleVisionAgent()
    memory_agent = ExampleMemoryAgent()
    
    # Configure coordinator with protocol adapters
    example_config = {
        "mcp": {
            "base_url": "https://api.example.com/mcp",
            "api_key": "dummy_key",
            "tool_mappings": {
                "vision_tool": {
                    "nis_agent": "vision_agent",
                    "target_layer": "PERCEPTION"
                },
                "memory_tool": {
                    "nis_agent": "memory_agent",
                    "target_layer": "MEMORY"
                }
            }
        },
        "acp": {
            "base_url": "https://api.example.com/acp",
            "api_key": "dummy_key"
        },
        "a2a": {
            "base_url": "https://api.example.com/a2a",
            "api_key": "dummy_key"
        }
    }
    
    configure_coordinator_agent(coordinator, config_dict=example_config)
    
    return coordinator


def simulate_mcp_request():
    """Simulate an MCP request from an external system."""
    return {
        "tool_id": "vision_tool",
        "conversation_id": "conv123",
        "function_call": {
            "name": "analyze_image",
            "arguments": json.dumps({
                "image_data": "base64_encoded_image_data",
                "analysis_type": "object_detection"
            })
        }
    }


def simulate_acp_request():
    """Simulate an ACP request from an external agent."""
    return {
        "headers": {
            "message_id": "msg123",
            "sender_id": "external_agent",
            "receiver_id": "nis_protocol",
            "conversation_id": "conv456",
            "timestamp": int(time.time() * 1000),
            "action": "store_memory"
        },
        "body": {
            "key": "last_observation",
            "value": {
                "location": "warehouse",
                "time": "2023-05-01T14:30:00Z",
                "objects": ["forklift", "pallet", "worker"]
            }
        }
    }


def simulate_a2a_request():
    """Simulate an A2A request from an external agent."""
    return {
        "agentCardHeader": {
            "messageId": "a2a123",
            "sessionId": "session789",
            "agentId": "external_a2a_agent",
            "version": "1.0"
        },
        "agentCardContent": {
            "actionRequest": {
                "actionName": "retrieve",
                "arguments": {
                    "key": "last_observation"
                }
            }
        }
    }


def main():
    """Run the protocol integration example."""
    print("Setting up NIS Protocol system...")
    coordinator = setup_system()
    
    print("\n--- MCP Example ---")
    mcp_request = simulate_mcp_request()
    print(f"Incoming MCP request: {json.dumps(mcp_request, indent=2)}")
    
    # Process MCP request
    mcp_response = coordinator.process({
        "protocol": "mcp",
        "original_message": mcp_request
    })
    print(f"NIS response to MCP request: {json.dumps(mcp_response, indent=2)}")
    
    print("\n--- ACP Example ---")
    acp_request = simulate_acp_request()
    print(f"Incoming ACP request: {json.dumps(acp_request, indent=2)}")
    
    # Process ACP request
    acp_response = coordinator.process({
        "protocol": "acp",
        "original_message": acp_request
    })
    print(f"NIS response to ACP request: {json.dumps(acp_response, indent=2)}")
    
    print("\n--- A2A Example ---")
    a2a_request = simulate_a2a_request()
    print(f"Incoming A2A request: {json.dumps(a2a_request, indent=2)}")
    
    # Process A2A request
    a2a_response = coordinator.process({
        "protocol": "a2a",
        "original_message": a2a_request
    })
    print(f"NIS response to A2A request: {json.dumps(a2a_response, indent=2)}")
    
    print("\n--- Example Complete ---")


if __name__ == "__main__":
    main() 
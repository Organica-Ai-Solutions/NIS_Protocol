#!/usr/bin/env python3
"""
Basic Agent Communication Example

This example demonstrates how agents in the NIS Protocol communicate
and how the emotional state system modulates their behavior.
"""

import os
import sys
import time
from typing import Dict, Any

# Add the parent directory to the path so we can import the NIS Protocol
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.agent import NISAgent, NISLayer
from src.core.registry import NISRegistry
from src.emotion.emotional_state import EmotionalStateSystem
from src.memory.memory_manager import MemoryManager

class VisionAgent(NISAgent):
    """Agent for processing visual inputs."""
    
    def __init__(self, agent_id: str, description: str):
        super().__init__(agent_id, NISLayer.PERCEPTION, description)
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming message.
        
        In this simple example, we simulate detecting objects in an image.
        """
        start_time = self._start_processing_timer()
        
        # Extract the image data from the message
        image_data = message.get("payload", {}).get("image_data")
        
        if not image_data:
            # No image data provided
            processing_time = self._end_processing_timer(start_time)
            
            return self._create_response(
                status="error",
                payload={"error": "No image data provided"},
                metadata={},
                emotional_state=message.get("emotional_state")
            )
        
        # Simulate object detection
        # In a real implementation, this would use computer vision libraries
        detected_objects = []
        
        if "car" in image_data:
            detected_objects.append({
                "type": "vehicle",
                "confidence": 0.95,
                "location": [100, 200, 300, 400]
            })
        
        if "person" in image_data:
            detected_objects.append({
                "type": "person",
                "confidence": 0.85,
                "location": [50, 100, 100, 200]
            })
        
        # Suspicious object detection
        if "suspicious" in image_data:
            detected_objects.append({
                "type": "unknown",
                "confidence": 0.3,
                "location": [400, 300, 450, 350]
            })
            
            # Update emotional state - increase suspicion
            emotional_state = message.get("emotional_state", {})
            emotional_state = self._update_emotional_state(emotional_state, "suspicion", 0.8)
        else:
            emotional_state = message.get("emotional_state")
        
        processing_time = self._end_processing_timer(start_time)
        
        return self._create_response(
            status="success",
            payload={"detected_objects": detected_objects},
            metadata={"object_count": len(detected_objects)},
            emotional_state=emotional_state
        )

class MemoryAgent(NISAgent):
    """Agent for storing and retrieving information."""
    
    def __init__(self, agent_id: str, description: str):
        super().__init__(agent_id, NISLayer.MEMORY, description)
        self.memory_manager = MemoryManager()
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming message."""
        start_time = self._start_processing_timer()
        
        # Get the current emotion state
        emotional_state = message.get("emotional_state", {})
        
        # Process the message based on the operation
        operation = message.get("payload", {}).get("operation")
        
        if operation == "store":
            key = message["payload"].get("key")
            data = message["payload"].get("data")
            
            if not key or not data:
                processing_time = self._end_processing_timer(start_time)
                return self._create_response(
                    status="error",
                    payload={"error": "Missing key or data for store operation"},
                    metadata={},
                    emotional_state=emotional_state
                )
            
            self.memory_manager.store(key, data)
            
            processing_time = self._end_processing_timer(start_time)
            return self._create_response(
                status="success",
                payload={"message": f"Data stored with key {key}"},
                metadata={},
                emotional_state=emotional_state
            )
            
        elif operation == "retrieve":
            key = message["payload"].get("key")
            
            if not key:
                processing_time = self._end_processing_timer(start_time)
                return self._create_response(
                    status="error",
                    payload={"error": "Missing key for retrieve operation"},
                    metadata={},
                    emotional_state=emotional_state
                )
            
            data = self.memory_manager.retrieve(key)
            
            processing_time = self._end_processing_timer(start_time)
            return self._create_response(
                status="success",
                payload={"key": key, "data": data},
                metadata={},
                emotional_state=emotional_state
            )
        
        else:
            processing_time = self._end_processing_timer(start_time)
            return self._create_response(
                status="error",
                payload={"error": f"Unknown operation: {operation}"},
                metadata={},
                emotional_state=emotional_state
            )

class CortexAgent(NISAgent):
    """Agent for high-level decision making."""
    
    def __init__(self, agent_id: str, description: str):
        super().__init__(agent_id, NISLayer.REASONING, description)
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming message."""
        start_time = self._start_processing_timer()
        
        detected_objects = message.get("payload", {}).get("detected_objects", [])
        emotional_state = message.get("emotional_state", {})
        
        # Determine the appropriate action based on detected objects and emotional state
        decision = self._make_decision(detected_objects, emotional_state)
        
        processing_time = self._end_processing_timer(start_time)
        return self._create_response(
            status="success",
            payload={"decision": decision},
            metadata={},
            emotional_state=emotional_state
        )
    
    def _make_decision(self, detected_objects, emotional_state):
        """Make a decision based on detected objects and emotional state."""
        suspicion_level = emotional_state.get("suspicion", 0.5)
        
        # Initialize decision
        decision = {
            "action": "monitor",
            "priority": "low",
            "details": "Continue normal monitoring"
        }
        
        # Check for suspicious objects or high suspicion level
        unknown_objects = [obj for obj in detected_objects if obj.get("type") == "unknown"]
        low_confidence_objects = [obj for obj in detected_objects if obj.get("confidence", 1.0) < 0.5]
        
        if unknown_objects or low_confidence_objects:
            if suspicion_level > 0.7:
                decision = {
                    "action": "alert",
                    "priority": "high",
                    "details": "Alert security personnel about suspicious objects"
                }
            elif suspicion_level > 0.5:
                decision = {
                    "action": "investigate",
                    "priority": "medium",
                    "details": "Investigate suspicious objects further"
                }
            else:
                decision = {
                    "action": "flag",
                    "priority": "low",
                    "details": "Flag objects for later review"
                }
        
        return decision

class ActionAgent(NISAgent):
    """Agent for executing actions in the environment."""
    
    def __init__(self, agent_id: str, description: str):
        super().__init__(agent_id, NISLayer.ACTION, description)
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming message."""
        start_time = self._start_processing_timer()
        
        decision = message.get("payload", {}).get("decision", {})
        emotional_state = message.get("emotional_state", {})
        
        # Execute the decision
        result = self._execute_action(decision)
        
        processing_time = self._end_processing_timer(start_time)
        return self._create_response(
            status="success",
            payload={"result": result},
            metadata={},
            emotional_state=emotional_state
        )
    
    def _execute_action(self, decision):
        """Execute an action based on the decision."""
        action = decision.get("action", "monitor")
        priority = decision.get("priority", "low")
        details = decision.get("details", "")
        
        # In a real implementation, this would interact with the environment
        result = {
            "action_taken": action,
            "priority": priority,
            "timestamp": time.time(),
            "status": "completed",
            "details": f"Executed: {details}"
        }
        
        # Print the action for demonstration purposes
        print(f"Executing action: {action} (Priority: {priority})")
        print(f"Details: {details}")
        
        return result


def main():
    """Run the example."""
    # Create the registry
    registry = NISRegistry()
    
    # Create the emotional state system
    emotional_state_system = EmotionalStateSystem()
    registry.set_emotional_state(emotional_state_system)
    
    # Create agents
    vision_agent = VisionAgent("vision_1", "Processes visual input")
    memory_agent = MemoryAgent("memory_1", "Stores and retrieves information")
    cortex_agent = CortexAgent("cortex_1", "Makes decisions based on inputs")
    action_agent = ActionAgent("action_1", "Executes actions in the environment")
    
    # Create a simple scenario
    print("\n=== Scenario 1: Normal Image ===")
    
    # Process a normal image
    image_message = {
        "payload": {
            "image_data": "normal image with car and person"
        },
        "emotional_state": emotional_state_system.get_state()
    }
    
    # Process through each agent
    vision_result = vision_agent.process(image_message)
    memory_store = memory_agent.process({
        "payload": {
            "operation": "store",
            "key": "scene_1",
            "data": vision_result["payload"]
        },
        "emotional_state": vision_result["emotional_state"]
    })
    cortex_result = cortex_agent.process(vision_result)
    action_result = action_agent.process(cortex_result)
    
    # Display the emotional state
    print("\nEmotional State after normal image:")
    for dimension, value in vision_result["emotional_state"].items():
        print(f"  {dimension}: {value:.2f}")
    
    # Create a suspicious scenario
    print("\n=== Scenario 2: Suspicious Image ===")
    
    # Reset emotional state
    emotional_state_system.reset()
    
    # Process a suspicious image
    suspicious_message = {
        "payload": {
            "image_data": "suspicious person near car"
        },
        "emotional_state": emotional_state_system.get_state()
    }
    
    # Process through each agent
    vision_result = vision_agent.process(suspicious_message)
    memory_store = memory_agent.process({
        "payload": {
            "operation": "store",
            "key": "scene_2",
            "data": vision_result["payload"]
        },
        "emotional_state": vision_result["emotional_state"]
    })
    cortex_result = cortex_agent.process(vision_result)
    action_result = action_agent.process(cortex_result)
    
    # Display the emotional state
    print("\nEmotional State after suspicious image:")
    for dimension, value in vision_result["emotional_state"].items():
        print(f"  {dimension}: {value:.2f}")
    
    # Retrieve memory
    print("\n=== Memory Retrieval ===")
    
    # Retrieve the first scene
    memory_retrieve = memory_agent.process({
        "payload": {
            "operation": "retrieve",
            "key": "scene_1"
        },
        "emotional_state": emotional_state_system.get_state()
    })
    
    if memory_retrieve["payload"]["data"]:
        print("\nRetrieved scene 1:")
        objects = memory_retrieve["payload"]["data"].get("detected_objects", [])
        for obj in objects:
            print(f"  {obj['type']} (confidence: {obj['confidence']:.2f})")
    
    # Retrieve the second scene
    memory_retrieve = memory_agent.process({
        "payload": {
            "operation": "retrieve",
            "key": "scene_2"
        },
        "emotional_state": emotional_state_system.get_state()
    })
    
    if memory_retrieve["payload"]["data"]:
        print("\nRetrieved scene 2:")
        objects = memory_retrieve["payload"]["data"].get("detected_objects", [])
        for obj in objects:
            print(f"  {obj['type']} (confidence: {obj['confidence']:.2f})")

if __name__ == "__main__":
    main() 
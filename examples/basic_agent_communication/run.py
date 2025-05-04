#!/usr/bin/env python3
"""
Basic NIS Protocol Agent Communication Example

This example demonstrates the basics of NIS Protocol agent communication,
emotional weighting, and decision-making.
"""

import sys
import time
import random
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, "../../src")

from core.registry import NISRegistry, NISAgent, NISLayer
from emotion.emotional_state import EmotionalState, EmotionalDimension


# Define some basic agents
class VisionAgentSimple(NISAgent):
    """Simple vision agent that simulates detecting vehicles."""
    
    def __init__(self):
        super().__init__(
            agent_id="vision_agent",
            layer=NISLayer.PERCEPTION,
            description="Basic vision agent for detecting vehicles"
        )
        self.vehicle_types = ["sedan", "suv", "truck", "motorcycle", "bus"]
        
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate detecting a vehicle."""
        vehicle_type = random.choice(self.vehicle_types)
        confidence = random.uniform(0.7, 0.99)
        
        return {
            "vehicle_detected": True,
            "vehicle_type": vehicle_type,
            "confidence": confidence,
            "timestamp": time.time()
        }


class MemoryAgentSimple(NISAgent):
    """Simple memory agent that stores and retrieves data."""
    
    def __init__(self):
        super().__init__(
            agent_id="memory_agent",
            layer=NISLayer.MEMORY,
            description="Basic memory agent for storing vehicle data"
        )
        self.memory = {}
        
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Store or retrieve data."""
        operation = message.get("operation", "")
        
        if operation == "store":
            key = message.get("key", "")
            data = message.get("data", {})
            if key:
                self.memory[key] = data
                return {"success": True, "key": key}
            else:
                return {"success": False, "error": "No key provided"}
                
        elif operation == "retrieve":
            key = message.get("key", "")
            if key in self.memory:
                return {"success": True, "data": self.memory[key]}
            else:
                return {"success": False, "error": "Key not found"}
                
        return {"success": False, "error": "Unknown operation"}


class EmotionAgentSimple(NISAgent):
    """Simple emotion agent that manages emotional state."""
    
    def __init__(self):
        super().__init__(
            agent_id="emotion_agent",
            layer=NISLayer.INTERPRETATION,
            description="Basic emotion agent for weighting decisions"
        )
        self.emotional_state = EmotionalState()
        
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Update or retrieve emotional state."""
        operation = message.get("operation", "")
        
        if operation == "update":
            dimension = message.get("dimension", "")
            value = message.get("value", 0.5)
            
            if dimension:
                self.emotional_state.update(dimension, value)
                return {"success": True, "dimension": dimension, "value": value}
            else:
                return {"success": False, "error": "No dimension provided"}
                
        elif operation == "get_state":
            return {
                "success": True,
                "emotional_state": self.emotional_state.get_state()
            }
            
        return {"success": False, "error": "Unknown operation"}


class ReasoningAgentSimple(NISAgent):
    """Simple reasoning agent that makes decisions."""
    
    def __init__(self, emotion_agent):
        super().__init__(
            agent_id="reasoning_agent",
            layer=NISLayer.REASONING,
            description="Basic reasoning agent for decision-making"
        )
        self.emotion_agent = emotion_agent
        
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Make a decision based on input and emotional state."""
        # Get current emotional state
        emotion_response = self.emotion_agent.process({"operation": "get_state"})
        emotional_state = emotion_response.get("emotional_state", {})
        
        # Extract relevant data from message
        vehicle_type = message.get("vehicle_type", "")
        confidence = message.get("confidence", 0.0)
        
        # Apply emotional weighting to decision
        suspicion = emotional_state.get("suspicion", 0.5)
        urgency = emotional_state.get("urgency", 0.5)
        
        # Calculate inspection threshold based on emotional state
        inspection_threshold = 0.3 - (suspicion * 0.2)  # Lower threshold when suspicious
        
        # Determine if vehicle needs inspection
        random_factor = random.uniform(0.0, 0.1)  # Small random factor
        needs_inspection = (confidence < inspection_threshold) or (random_factor > 0.95)
        
        # Update emotional state based on decision
        if needs_inspection:
            # Increase suspicion when we need to inspect
            self.emotion_agent.process({
                "operation": "update",
                "dimension": "suspicion",
                "value": suspicion + 0.1
            })
        
        return {
            "decision": "inspect" if needs_inspection else "proceed",
            "confidence": confidence,
            "inspection_threshold": inspection_threshold,
            "suspicion_level": suspicion,
            "urgency_level": urgency
        }


class ActionAgentSimple(NISAgent):
    """Simple action agent that executes decisions."""
    
    def __init__(self):
        super().__init__(
            agent_id="action_agent",
            layer=NISLayer.ACTION,
            description="Basic action agent for executing decisions"
        )
        
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the decision."""
        decision = message.get("decision", "")
        
        if decision == "inspect":
            print(f"🔴 GATE: Stopping vehicle for inspection")
            return {"action": "stop_vehicle", "gate": "closed"}
        else:
            print(f"🟢 GATE: Allowing vehicle to proceed")
            return {"action": "allow_vehicle", "gate": "open"}


def main():
    """Run the basic agent communication example."""
    print("=== NIS Protocol Basic Agent Communication Example ===\n")
    
    # Create agents
    vision_agent = VisionAgentSimple()
    memory_agent = MemoryAgentSimple()
    emotion_agent = EmotionAgentSimple()
    reasoning_agent = ReasoningAgentSimple(emotion_agent)
    action_agent = ActionAgentSimple()
    
    # Initialize emotional state with some values
    emotion_agent.process({
        "operation": "update",
        "dimension": "suspicion",
        "value": 0.3  # Start with low suspicion
    })
    
    emotion_agent.process({
        "operation": "update",
        "dimension": "urgency",
        "value": 0.6  # Moderate urgency
    })
    
    # Simulate processing 10 vehicles
    for i in range(10):
        print(f"\n--- Vehicle {i+1} ---")
        
        # 1. Vision agent detects vehicle
        vehicle_data = vision_agent.process({})
        vehicle_type = vehicle_data["vehicle_type"]
        confidence = vehicle_data["confidence"]
        print(f"🚗 VISION: Detected {vehicle_type} (confidence: {confidence:.2f})")
        
        # 2. Memory agent stores vehicle data
        memory_result = memory_agent.process({
            "operation": "store",
            "key": f"vehicle_{i}",
            "data": vehicle_data
        })
        
        # 3. Reasoning agent makes decision
        decision_result = reasoning_agent.process(vehicle_data)
        print(f"🧠 REASONING: Decision '{decision_result['decision']}' " +
              f"(suspicion: {decision_result['suspicion_level']:.2f}, " +
              f"threshold: {decision_result['inspection_threshold']:.2f})")
        
        # 4. Action agent executes decision
        action_result = action_agent.process(decision_result)
        
        # Wait a moment between vehicles
        time.sleep(1)
    
    # Print final emotional state
    final_state = emotion_agent.process({"operation": "get_state"})
    print("\n--- Final Emotional State ---")
    for dimension, value in final_state["emotional_state"].items():
        print(f"{dimension}: {value:.2f}")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main() 
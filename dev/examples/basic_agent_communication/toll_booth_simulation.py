#!/usr/bin/env python3
"""
NIS Protocol Toll Booth Simulation Example

This example demonstrates how to use the NIS Protocol to build a toll booth system
with perception, memory, reasoning, and action components.
"""

import os
import sys
import time
import json
import random
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add the parent directory to sys.path to import NIS Protocol modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.registry import NISRegistry, NISLayer
from src.emotion.emotional_state import EmotionalState
from src.agents.perception import VisionAgent, InputAgent
from src.agents.memory import MemoryAgent, LogAgent

# Local imports
from toll_utils import Vehicle, generate_random_vehicle, set_traffic_conditions


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class CortexAgent:
    """
    Reasoning agent for the toll booth system.
    
    Makes decisions about whether to allow vehicles through based on
    vision data, payment information, and historical context.
    """
    
    def __init__(self, agent_id: str, description: str):
        from src.core.registry import NISAgent, NISLayer
        
        # Create a NIS agent for registration
        self.agent = NISAgent(agent_id, NISLayer.REASONING, description)
        self.agent_id = agent_id
        self.emotional_state = EmotionalState()
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs and make a decision about the vehicle."""
        
        # Extract data from message
        vision_data = message.get("vision_data", {})
        input_data = message.get("input_data", {})
        memory_data = message.get("memory_data", {})
        vehicle = message.get("vehicle")
        
        if not vehicle:
            return {
                "status": "error",
                "error": "No vehicle data provided",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        # Get detection information
        detections = vision_data.get("detections", [])
        vehicle_type = "unknown"
        for detection in detections:
            if detection.get("type") == "vehicle":
                vehicle_type = detection.get("vehicle_type", "unknown")
        
        # Get payment information
        structured_data = input_data.get("structured_data", {})
        payment_status = "unknown"
        if "payment_status" in structured_data:
            payment_status = structured_data["payment_status"]
        
        # Get historical records
        results = memory_data.get("results", [])
        previous_passages = len(results)
        previous_violations = sum(1 for r in results if r.get("data", {}).get("decision") == "deny_passage")
        
        # Calculate decision factors
        suspicion_level = self.emotional_state.get_dimension("suspicion")
        urgency_level = self.emotional_state.get_dimension("urgency")
        
        # Decision logic
        decision = "allow_passage"
        importance = 0.5
        reason = "Valid payment and no issues detected"
        
        # Payment issues
        if payment_status in ["missing_payment", "expired_payment", "insufficient_funds"]:
            decision = "deny_passage"
            importance = 0.7
            reason = f"Payment issue: {payment_status}"
            
            # Update emotional state - increase suspicion
            self.emotional_state.update("suspicion", 0.7)
        
        # Previous violations increase suspicion
        if previous_violations > 0:
            violation_ratio = previous_violations / max(1, previous_passages)
            if violation_ratio > 0.3:  # More than 30% violations
                self.emotional_state.update("suspicion", 0.8)
                
                if decision == "allow_passage":
                    # Only deny if not already denied for payment
                    decision = "flag_for_inspection"
                    importance = 0.6
                    reason = f"History of violations ({previous_violations}/{previous_passages})"
        
        # Unknown vehicle type increases suspicion
        if vehicle_type == "unknown":
            self.emotional_state.update("suspicion", 0.6)
        
        # High urgency (high traffic) might lower inspection threshold
        # to keep traffic flowing
        if urgency_level > 0.7 and decision == "flag_for_inspection":
            # In high urgency situations, we might let more vehicles through
            # to prevent traffic jams, accepting some risk
            if random.random() > suspicion_level:
                decision = "allow_passage"
                reason += " (expedited due to high traffic)"
        
        return {
            "status": "success",
            "decision": decision,
            "reason": reason,
            "importance": importance,
            "suspicion_level": suspicion_level,
            "urgency_level": urgency_level,
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }


class ActionAgent:
    """
    Action agent for the toll booth system.
    
    Executes decisions by controlling the toll gate, signaling
    operators, and managing physical systems.
    """
    
    def __init__(self, agent_id: str, description: str):
        from src.core.registry import NISAgent, NISLayer
        
        # Create a NIS agent for registration
        self.agent = NISAgent(agent_id, NISLayer.ACTION, description)
        self.agent_id = agent_id
        self.emotional_state = EmotionalState()
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action based on the decision."""
        
        # Extract decision
        decision = message.get("decision")
        vehicle = message.get("vehicle")
        
        if not decision or not vehicle:
            return {
                "status": "error",
                "error": "Missing decision or vehicle data",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        # Execute appropriate action based on decision
        action = "none"
        
        if decision == "allow_passage":
            action = "opened_gate"
            logger.info(f"GATE OPENED for vehicle {vehicle.plate_number}")
            
            # Successful passage decreases suspicion slightly
            self.emotional_state.update("suspicion", 0.4)
            
        elif decision == "deny_passage":
            action = "alert_operator"
            logger.info(f"⚠️ ALERT: Vehicle {vehicle.plate_number} denied passage. Operator notified.")
            
            # Denied passage keeps suspicion elevated
            self.emotional_state.update("suspicion", 0.6)
            
        elif decision == "flag_for_inspection":
            action = "divert_to_inspection"
            logger.info(f"⚠️ Vehicle {vehicle.plate_number} diverted to inspection lane")
            
            # Inspection keeps suspicion moderate
            self.emotional_state.update("suspicion", 0.5)
        
        return {
            "status": "success",
            "action": action,
            "decision": decision,
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }


def process_vehicle(vehicle: Vehicle, agents: Dict[str, Any]) -> None:
    """Process a single vehicle through the toll system."""
    
    logger.info(f"Processing vehicle: {vehicle.plate_number}")
    
    # 1. Vision Agent processes the vehicle image
    vision_result = agents["vision"].process({
        "image_data": vehicle.image_data
    })
    
    for detection in vision_result.get("detections", []):
        if detection.get("type") == "vehicle":
            logger.info(f"Vision Agent detected vehicle type: {detection.get('vehicle_type', 'unknown')}")
    
    # 2. Input Agent processes payment information
    input_result = agents["input"].process({
        "text": vehicle.payment_info
    })
    
    payment_status = input_result.get("structured_data", {}).get("payment_status", "unknown")
    logger.info(f"Input Agent processed payment info: {payment_status}")
    
    # 3. Memory Agent retrieves vehicle history
    memory_result = agents["memory"].process({
        "operation": "query",
        "query": {
            "tags": ["vehicle", vehicle.plate_number],
            "max_results": 10
        }
    })
    
    previous_entries = memory_result.get("result_count", 0)
    logger.info(f"Memory Agent found previous entries: {previous_entries}")
    
    # 4. Cortex Agent makes a decision
    cortex_result = agents["cortex"].process({
        "vision_data": vision_result,
        "input_data": input_result,
        "memory_data": memory_result,
        "vehicle": vehicle
    })
    
    decision = cortex_result.get("decision", "unknown")
    suspicion_level = cortex_result.get("suspicion_level", 0.5)
    
    if suspicion_level > 0.6:
        logger.info(f"Cortex Agent decision: {decision} (suspicion level: {suspicion_level:.2f})")
    else:
        logger.info(f"Cortex Agent decision: {decision}")
    
    # 5. Action Agent executes the decision
    action_result = agents["action"].process({
        "decision": decision,
        "vehicle": vehicle
    })
    
    action = action_result.get("action", "none")
    logger.info(f"Action Agent executed: {action}")
    
    # 6. Store the result in memory
    store_result = agents["memory"].process({
        "operation": "store",
        "data": {
            "vehicle": vehicle.plate_number,
            "vehicle_type": vehicle.vehicle_type,
            "payment_info": vehicle.payment_info,
            "decision": decision,
            "action": action,
            "timestamp": time.time()
        },
        "tags": ["vehicle", vehicle.plate_number, decision],
        "importance": cortex_result.get("importance", 0.5)
    })
    
    logger.info(f"Memory updated with {decision} passage\n")
    
    # 7. Log the entire transaction
    agents["log"].process({
        "level": "info",
        "content": f"Vehicle {vehicle.plate_number} processed: {decision}",
        "source_agent": "toll_system",
        "metadata": {
            "vehicle": vehicle.plate_number,
            "decision": decision,
            "action": action
        }
    })


def setup_agents() -> Dict[str, Any]:
    """Set up all agents required for the toll booth system."""
    
    # Create data directories if they don't exist
    os.makedirs("./data/memory", exist_ok=True)
    os.makedirs("./data/logs", exist_ok=True)
    
    # Initialize agents
    vision_agent = VisionAgent(
        agent_id="toll_vision",
        description="Toll booth camera system"
    )
    
    input_agent = InputAgent(
        agent_id="toll_payment",
        description="Payment information processor"
    )
    
    memory_agent = MemoryAgent(
        agent_id="toll_memory",
        description="Vehicle history database",
        storage_path="./data/memory"
    )
    
    log_agent = LogAgent(
        agent_id="toll_log",
        description="System event logger",
        log_path="./data/logs"
    )
    
    cortex_agent = CortexAgent(
        agent_id="toll_decision",
        description="Passage decision maker"
    )
    
    action_agent = ActionAgent(
        agent_id="toll_action",
        description="Gate and alert controller"
    )
    
    agents = {
        "vision": vision_agent,
        "input": input_agent,
        "memory": memory_agent,
        "log": log_agent,
        "cortex": cortex_agent,
        "action": action_agent
    }
    
    logger.info(f"System initialized with {len(agents)} agents")
    return agents


def simulate_toll_booth(num_vehicles: int = 10, traffic_condition: str = "normal") -> None:
    """Run the toll booth simulation with the specified number of vehicles."""
    
    # Set up agents
    agents = setup_agents()
    
    # Set traffic conditions (affects urgency emotional state)
    set_traffic_conditions(traffic_condition, agents)
    
    # Process vehicles
    for _ in range(num_vehicles):
        vehicle = generate_random_vehicle()
        process_vehicle(vehicle, agents)
        
        # Pause between vehicles
        time.sleep(0.5)
    
    # Get log statistics
    log_stats = agents["log"].get_statistics()
    logger.info(f"Simulation complete. Processed {num_vehicles} vehicles.")
    logger.info(f"System log statistics: {log_stats['level_counts']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NIS Protocol Toll Booth Simulation")
    parser.add_argument("--vehicles", type=int, default=5, help="Number of vehicles to simulate")
    parser.add_argument("--traffic", choices=["light", "normal", "heavy"], default="normal", 
                       help="Traffic condition (affects urgency)")
    
    args = parser.parse_args()
    
    simulate_toll_booth(args.vehicles, args.traffic) 
"""
Toll Booth Utilities

Helper functions and classes for the NIS Protocol toll booth simulation example.
"""

import random
import string
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class Vehicle:
    """Representation of a vehicle passing through the toll."""
    plate_number: str
    vehicle_type: str
    payment_info: str
    image_data: Dict[str, Any]  # Simulated image data


def generate_random_vehicle() -> Vehicle:
    """Generate a random vehicle for simulation."""
    
    # Generate a random license plate
    plate = ''.join(random.choices(string.ascii_uppercase, k=3)) + \
            ''.join(random.choices(string.digits, k=3))
    
    # Select a random vehicle type
    vehicle_types = ["sedan", "truck", "van", "motorcycle", "bus", "unknown"]
    vehicle_type = random.choice(vehicle_types)
    
    # Generate payment info
    payment_types = [
        "valid_ezpass",
        "valid_cash",
        "expired_payment",
        "missing_payment",
        "insufficient_funds"
    ]
    
    # Weight the payment types (valid payments are more common)
    weights = [0.7, 0.1, 0.05, 0.1, 0.05]
    payment_info = random.choices(payment_types, weights=weights)[0]
    
    # Create simulated image data
    image_data = {
        "width": 640,
        "height": 480,
        "format": "RGB",
        "timestamp": time.time(),
        "data": f"<simulated_image_data_for_{vehicle_type}>"
    }
    
    return Vehicle(
        plate_number=plate,
        vehicle_type=vehicle_type,
        payment_info=payment_info,
        image_data=image_data
    )


def set_traffic_conditions(condition: str, agents: Dict[str, Any]) -> None:
    """Set traffic conditions which affect agent emotional states."""
    
    # Map condition to urgency level
    urgency_levels = {
        "light": 0.3,    # Low urgency
        "normal": 0.5,   # Normal urgency
        "heavy": 0.8     # High urgency
    }
    
    urgency = urgency_levels.get(condition, 0.5)
    
    # Update urgency for all agents
    for agent_id, agent in agents.items():
        if hasattr(agent, "emotional_state") and hasattr(agent.emotional_state, "update"):
            agent.emotional_state.update("urgency", urgency)
    
    print(f"Traffic condition set to {condition} (urgency: {urgency:.1f})")


def process_payment_info(payment_text: str) -> Dict[str, Any]:
    """
    Process payment information text into structured data.
    
    This simulates what the InputAgent would do with payment info.
    """
    
    # Simple keyword mapping
    status_mapping = {
        "valid_ezpass": {
            "payment_status": "valid_ezpass",
            "payment_type": "electronic",
            "confidence": 0.95
        },
        "valid_cash": {
            "payment_status": "valid_cash",
            "payment_type": "manual",
            "confidence": 0.9
        },
        "expired_payment": {
            "payment_status": "expired_payment",
            "payment_type": "electronic",
            "confidence": 0.85,
            "expiration": "past"
        },
        "missing_payment": {
            "payment_status": "missing_payment",
            "payment_type": "none",
            "confidence": 0.8
        },
        "insufficient_funds": {
            "payment_status": "insufficient_funds",
            "payment_type": "electronic",
            "confidence": 0.9,
            "balance": "low"
        }
    }
    
    # Return structured data if we recognize the payment text
    if payment_text in status_mapping:
        return status_mapping[payment_text]
    
    # Default response for unknown payment info
    return {
        "payment_status": "unknown",
        "confidence": 0.3
    }


def process_vehicle_image(image_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process vehicle image data into detections.
    
    This simulates what the VisionAgent would do with image data.
    """
    
    # Extract vehicle type from simulated image data
    image_str = image_data.get("data", "")
    
    # Parse the vehicle type from the simulated data
    vehicle_type = "unknown"
    if "_for_" in image_str and ">" in image_str:
        start_idx = image_str.find("_for_") + 5
        end_idx = image_str.find(">", start_idx)
        if start_idx > 5 and end_idx > start_idx:
            vehicle_type = image_str[start_idx:end_idx]
    
    # Create detections based on vehicle type
    detections = [
        {
            "type": "vehicle",
            "vehicle_type": vehicle_type,
            "confidence": 0.9 if vehicle_type != "unknown" else 0.5,
            "position": [100, 100, 400, 300]  # x1, y1, x2, y2
        }
    ]
    
    # Add additional detections based on vehicle type
    if vehicle_type == "truck":
        detections.append({
            "type": "size_classification",
            "size": "large",
            "confidence": 0.85
        })
    elif vehicle_type == "motorcycle":
        detections.append({
            "type": "size_classification",
            "size": "small",
            "confidence": 0.9
        })
    
    return detections 
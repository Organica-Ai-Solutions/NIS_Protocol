"""
NIS Protocol — Physics Utilities
Stub providing PhysicsCalculator used by reasoning agents.
"""
import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nis.utils.physics")


class PhysicsCalculator:
    """Basic physics calculations for NIS reasoning agents."""

    def __init__(self):
        self.logger = logging.getLogger("nis.utils.physics_calculator")

    def calculate_trajectory(
        self,
        start: List[float],
        end: List[float],
        velocity: float = 1.0,
        gravity: float = 9.81,
    ) -> Dict[str, Any]:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2] if len(start) > 2 and len(end) > 2 else 0.0
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        duration = distance / max(velocity, 1e-6)
        return {
            "start": start,
            "end": end,
            "distance": round(distance, 4),
            "duration": round(duration, 4),
            "velocity": velocity,
            "feasible": True,
        }

    def validate_motion(
        self,
        trajectory: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        constraints = constraints or {}
        max_dist = constraints.get("max_distance", 2.0)
        dist = trajectory.get("distance", 0.0)
        safe = dist <= max_dist
        return {
            "safe": safe,
            "violations": [] if safe else [f"distance {dist:.2f}m exceeds limit {max_dist}m"],
            "confidence": 0.9 if safe else 0.3,
        }

    def estimate_force(self, mass_kg: float, acceleration: float = 9.81) -> float:
        return mass_kg * acceleration

    def check_stability(self, center_of_mass: List[float], support_polygon: List[List[float]]) -> bool:
        if not support_polygon:
            return False
        cx, cy = center_of_mass[0], center_of_mass[1]
        n = len(support_polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = support_polygon[i]
            xj, yj = support_polygon[j]
            if ((yi > cy) != (yj > cy)) and (cx < (xj - xi) * (cy - yi) / (yj - yi + 1e-9) + xi):
                inside = not inside
            j = i
        return inside

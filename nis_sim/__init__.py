"""
NIS Simulation Environment
Virtual testing for NIS Protocol deployments
"""

__version__ = "0.1.0"

from .core.engine import SimulationEngine
from .core.world import World
from .agents.drone import DroneAgent
from .agents.vehicle import VehicleAgent
from .bridge.nis_connector import NISConnector

__all__ = [
    "SimulationEngine",
    "World", 
    "DroneAgent",
    "VehicleAgent",
    "NISConnector"
]

#!/usr/bin/env python3
"""
NIS Protocol Plugin System
===========================

Plugin architecture for domain-specific adaptations:
- NIS-DRONE: Drone/UAV integration
- NIS-AUTO: Vehicle diagnostics and control
- NIS-CITY: Smart city IoT integration
- NIS-X: Space/scientific data analysis

Example::

    from nis_protocol import NISCore
    from nis_protocol.plugins import DronePlugin
    
    # Initialize core
    nis = NISCore()
    
    # Register drone plugin
    drone_plugin = DronePlugin(
        sensors=['camera', 'lidar', 'gps'],
        actuators=['motors', 'servos']
    )
    nis.register_plugin(drone_plugin)
    
    # Now NIS Protocol can process drone-specific requests
    result = await nis.process_autonomously(
        "Navigate to coordinates 37.7749, -122.4194"
    )
"""

from .base import BasePlugin, PluginManager
from .drone_plugin import DronePlugin
from .auto_plugin import AutoPlugin
from .city_plugin import CityPlugin

__all__ = [
    "BasePlugin",
    "PluginManager",
    "DronePlugin",
    "AutoPlugin",
    "CityPlugin",
]


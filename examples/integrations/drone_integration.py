"""
NIS-DRONE Integration Template
Complete example of integrating NIS Protocol into a drone control system
"""

import asyncio
from nis_protocol import NISCore
from nis_protocol.plugins import DronePlugin

class DroneControlSystem:
    """Example drone control system using NIS Protocol"""
    
    def __init__(self, config: dict = None):
        self.nis = NISCore(config or {})
        self.drone_plugin = DronePlugin(config={
            'sensors': {
                'camera': {'resolution': '4K', 'fps': 60},
                'lidar': {'range': 100, 'accuracy': 0.05},
                'gps': {'frequency': 10, 'accuracy': 2.5}
            },
            'flight_parameters': {
                'max_altitude': 120,  # meters
                'max_speed': 15,      # m/s
                'battery_reserve': 20  # percent
            }
        })
        
    async def initialize(self):
        """Initialize NIS Protocol and register drone plugin"""
        await self.nis.initialize()
        await self.nis.register_plugin(self.drone_plugin)
        print("✅ Drone Control System initialized with NIS Protocol")
        
    async def autonomous_flight(self, mission: str):
        """Execute autonomous flight mission"""
        print(f"🚁 Mission: {mission}")
        
        # NIS Protocol analyzes intent and executes autonomously
        result = await self.nis.process_autonomously(mission)
        
        return result
        
    async def obstacle_avoidance(self, sensor_data: dict):
        """Real-time obstacle detection and avoidance"""
        # Use NIS Protocol's physics validation for collision detection
        query = f"Analyze obstacle at {sensor_data} and suggest avoidance maneuver"
        result = await self.nis.process_autonomously(query)
        return result
        
    async def mission_planning(self, waypoints: list):
        """Plan optimal flight path"""
        query = f"Plan optimal drone path through waypoints: {waypoints}"
        result = await self.nis.process_autonomously(query)
        return result


async def main():
    # Example 1: Basic drone control
    drone = DroneControlSystem()
    await drone.initialize()
    
    # Example 2: Autonomous navigation
    result = await drone.autonomous_flight(
        "Navigate to GPS coordinates 37.7749° N, 122.4194° W at 50m altitude"
    )
    print(f"Navigation result: {result}")
    
    # Example 3: Obstacle avoidance
    sensor_data = {
        'obstacle_distance': 15,
        'obstacle_direction': 'front',
        'drone_speed': 5
    }
    avoidance = await drone.obstacle_avoidance(sensor_data)
    print(f"Avoidance maneuver: {avoidance}")
    
    # Example 4: Mission planning
    waypoints = [
        {'lat': 37.7749, 'lon': -122.4194, 'alt': 50},
        {'lat': 37.7750, 'lon': -122.4195, 'alt': 60},
        {'lat': 37.7751, 'lon': -122.4196, 'alt': 55}
    ]
    plan = await drone.mission_planning(waypoints)
    print(f"Mission plan: {plan}")


if __name__ == "__main__":
    asyncio.run(main())


"""
Drone Delivery Scenario
Test autonomous drone delivery missions
"""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

from ..core.engine import SimulationEngine, SimulationConfig
from ..core.world import AirspaceWorld
from ..agents.drone import DroneAgent
from ..bridge.nis_connector import NISConnector, NISConfig


@dataclass
class DeliveryMission:
    """A delivery mission"""
    mission_id: str
    pickup: tuple  # (x, y, z)
    dropoff: tuple
    priority: int = 1
    payload_weight: float = 0.5  # kg


class DroneDeliveryScenario:
    """
    Drone delivery scenario
    Tests: takeoff, navigation, obstacle avoidance, landing
    """
    
    def __init__(self, 
                 num_drones: int = 1,
                 connect_nis: bool = True,
                 nis_config: Optional[NISConfig] = None):
        self.num_drones = num_drones
        self.connect_nis = connect_nis
        self.nis_config = nis_config
        
        self.engine: Optional[SimulationEngine] = None
        self.world: Optional[AirspaceWorld] = None
        self.drones: Dict[str, DroneAgent] = {}
        self.connector: Optional[NISConnector] = None
        self.missions: List[DeliveryMission] = []
        self.results: Dict = {
            "missions_completed": 0,
            "missions_failed": 0,
            "total_distance": 0,
            "collisions": 0,
            "battery_usage": []
        }
    
    async def setup(self):
        """Setup the scenario"""
        print("🚁 Setting up Drone Delivery Scenario...")
        
        # Create simulation engine
        self.engine = SimulationEngine(SimulationConfig(
            timestep=1/60,  # 60 Hz
            realtime=False,
            gui=False
        ))
        self.engine.initialize()
        
        # Create world
        self.world = AirspaceWorld()
        self.world.initialize(self.engine.physics_client)
        
        # Add some obstacles
        self.world.add_obstacle("tower_1", (100, 100, 25), (10, 10, 50))
        self.world.add_obstacle("tower_2", (200, 150, 30), (10, 10, 60))
        self.world.add_obstacle("building_1", (150, 200, 15), (30, 30, 30))
        
        # Spawn drones
        for i in range(self.num_drones):
            drone_id = f"drone_{i}"
            drone = DroneAgent(
                agent_id=drone_id,
                initial_position=(10 + i * 5, 10, 0.5)
            )
            self.engine.add_agent(drone_id, drone)
            self.drones[drone_id] = drone
        
        # Connect to NIS Protocol
        if self.connect_nis:
            self.connector = NISConnector(self.nis_config)
            await self.connector.connect()
            
            # Register drones
            for drone_id in self.drones:
                await self.connector.register_agent(
                    drone_id,
                    "quadrotor",
                    ["fly", "hover", "land", "carry_payload"]
                )
        
        print(f"✅ Scenario ready: {self.num_drones} drones")
    
    def add_mission(self, pickup: tuple, dropoff: tuple, priority: int = 1):
        """Add a delivery mission"""
        mission = DeliveryMission(
            mission_id=f"mission_{len(self.missions)}",
            pickup=pickup,
            dropoff=dropoff,
            priority=priority
        )
        self.missions.append(mission)
        return mission
    
    async def run(self, duration: float = 60.0):
        """Run the scenario"""
        print(f"🚀 Running scenario for {duration}s...")
        
        # Assign first mission to first drone
        if self.missions and self.drones:
            drone = list(self.drones.values())[0]
            mission = self.missions[0]
            
            # Takeoff
            drone.apply_command({"type": "takeoff", "altitude": 20})
            
            # Wait for takeoff (simplified)
            await asyncio.sleep(0.1)
            
            # Go to pickup
            drone.apply_command({"type": "goto", "position": list(mission.pickup)})
        
        # Run simulation
        step_count = 0
        async for state in self.engine.run(duration):
            step_count += 1
            
            # Send telemetry to NIS every 10 steps
            if self.connector and step_count % 10 == 0:
                for drone_id, drone in self.drones.items():
                    await self.connector.send_telemetry(drone_id, drone.get_state())
            
            # Check for collisions
            if state.events:
                for event in state.events:
                    if event["type"] == "collision":
                        self.results["collisions"] += 1
                        print(f"⚠️ Collision at {event['position']}")
            
            # Progress update
            if step_count % 600 == 0:  # Every 10 seconds at 60Hz
                print(f"  Time: {state.time:.1f}s, Drones active: {len(self.drones)}")
        
        # Collect results
        for drone in self.drones.values():
            self.results["battery_usage"].append(100 - drone.battery_level)
        
        print(f"✅ Scenario complete!")
        return self.results
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.connector:
            await self.connector.disconnect()
        if self.engine:
            self.engine.shutdown()


async def run_demo():
    """Run a demo of the drone delivery scenario"""
    scenario = DroneDeliveryScenario(
        num_drones=3,
        connect_nis=False  # Run standalone for demo
    )
    
    await scenario.setup()
    
    # Add missions
    scenario.add_mission(
        pickup=(50, 50, 20),
        dropoff=(200, 200, 20)
    )
    scenario.add_mission(
        pickup=(100, 50, 20),
        dropoff=(50, 200, 20)
    )
    
    # Run for 30 seconds
    results = await scenario.run(duration=30.0)
    
    print("\n📊 Results:")
    print(f"  Collisions: {results['collisions']}")
    print(f"  Battery usage: {results['battery_usage']}")
    
    await scenario.cleanup()


if __name__ == "__main__":
    asyncio.run(run_demo())

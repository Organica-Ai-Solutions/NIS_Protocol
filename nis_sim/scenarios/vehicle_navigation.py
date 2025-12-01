"""
Vehicle Navigation Scenario
Test autonomous vehicle navigation (NIS-AUTO)
"""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

from ..core.engine import SimulationEngine, SimulationConfig
from ..core.world import CityWorld
from ..agents.vehicle import VehicleAgent
from ..bridge.nis_connector import NISConnector, NISConfig


@dataclass
class Waypoint:
    """Navigation waypoint"""
    position: tuple
    speed_limit: float = 50.0  # km/h
    stop_required: bool = False


class VehicleNavigationScenario:
    """
    Vehicle navigation scenario for NIS-AUTO testing
    Tests: steering, speed control, waypoint navigation
    """
    
    def __init__(self,
                 num_vehicles: int = 1,
                 connect_nis: bool = True,
                 nis_config: Optional[NISConfig] = None):
        self.num_vehicles = num_vehicles
        self.connect_nis = connect_nis
        self.nis_config = nis_config
        
        self.engine: Optional[SimulationEngine] = None
        self.world: Optional[CityWorld] = None
        self.vehicles: Dict[str, VehicleAgent] = {}
        self.connector: Optional[NISConnector] = None
        self.waypoints: List[Waypoint] = []
        self.results: Dict = {
            "distance_traveled": 0,
            "average_speed": 0,
            "max_speed": 0,
            "collisions": 0,
            "waypoints_reached": 0,
            "obd_log": []
        }
    
    async def setup(self):
        """Setup the scenario"""
        print("🚗 Setting up Vehicle Navigation Scenario...")
        
        # Create simulation engine
        self.engine = SimulationEngine(SimulationConfig(
            timestep=1/60,
            realtime=False,
            gui=False
        ))
        self.engine.initialize()
        
        # Create city world
        self.world = CityWorld("test_city")
        self.world.initialize(self.engine.physics_client)
        self.world.generate_city_grid(blocks_x=3, blocks_y=3, block_size=100)
        
        # Spawn vehicles
        for i in range(self.num_vehicles):
            vehicle_id = f"vehicle_{i}"
            vehicle = VehicleAgent(
                agent_id=vehicle_id,
                initial_position=(50 + i * 10, 50, 0.5)
            )
            self.engine.add_agent(vehicle_id, vehicle)
            self.vehicles[vehicle_id] = vehicle
        
        # Connect to NIS Protocol
        if self.connect_nis:
            self.connector = NISConnector(self.nis_config)
            await self.connector.connect()
            
            for vehicle_id in self.vehicles:
                await self.connector.register_agent(
                    vehicle_id,
                    "ground_vehicle",
                    ["drive", "steer", "brake", "obd_read"]
                )
        
        print(f"✅ Scenario ready: {self.num_vehicles} vehicles")
    
    def add_waypoint(self, position: tuple, speed_limit: float = 50.0, stop: bool = False):
        """Add navigation waypoint"""
        wp = Waypoint(position=position, speed_limit=speed_limit, stop_required=stop)
        self.waypoints.append(wp)
        return wp
    
    async def run(self, duration: float = 60.0):
        """Run the scenario"""
        print(f"🚀 Running scenario for {duration}s...")
        
        # Start first vehicle toward first waypoint
        if self.waypoints and self.vehicles:
            vehicle = list(self.vehicles.values())[0]
            wp = self.waypoints[0]
            vehicle.apply_command({"type": "goto", "position": list(wp.position)})
        
        speeds = []
        step_count = 0
        
        async for state in self.engine.run(duration):
            step_count += 1
            
            for vehicle_id, vehicle in self.vehicles.items():
                # Track speed
                speeds.append(vehicle.speed * 3.6)  # km/h
                
                # Log OBD data periodically
                if step_count % 60 == 0:  # Every second
                    self.results["obd_log"].append({
                        "time": state.time,
                        "vehicle": vehicle_id,
                        "obd": vehicle.obd_data.copy()
                    })
                
                # Send telemetry to NIS
                if self.connector and step_count % 10 == 0:
                    await self.connector.send_telemetry(vehicle_id, vehicle.get_state())
            
            # Check collisions
            if state.events:
                for event in state.events:
                    if event["type"] == "collision":
                        self.results["collisions"] += 1
            
            # Progress
            if step_count % 600 == 0:
                v = list(self.vehicles.values())[0]
                print(f"  Time: {state.time:.1f}s, Speed: {v.speed*3.6:.1f} km/h")
        
        # Calculate results
        if speeds:
            self.results["average_speed"] = np.mean(speeds)
            self.results["max_speed"] = np.max(speeds)
        
        print(f"✅ Scenario complete!")
        return self.results
    
    async def cleanup(self):
        """Cleanup"""
        if self.connector:
            await self.connector.disconnect()
        if self.engine:
            self.engine.shutdown()


async def run_demo():
    """Demo vehicle navigation"""
    scenario = VehicleNavigationScenario(
        num_vehicles=1,
        connect_nis=False
    )
    
    await scenario.setup()
    
    # Add waypoints (simple route)
    scenario.add_waypoint((100, 50, 0), speed_limit=30)
    scenario.add_waypoint((100, 150, 0), speed_limit=50)
    scenario.add_waypoint((200, 150, 0), speed_limit=40)
    
    results = await scenario.run(duration=20.0)
    
    print("\n📊 Results:")
    print(f"  Average speed: {results['average_speed']:.1f} km/h")
    print(f"  Max speed: {results['max_speed']:.1f} km/h")
    print(f"  Collisions: {results['collisions']}")
    print(f"  OBD samples: {len(results['obd_log'])}")
    
    await scenario.cleanup()


if __name__ == "__main__":
    asyncio.run(run_demo())

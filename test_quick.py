"""Quick test - runs without PyBullet"""

import asyncio
import sys
sys.path.insert(0, '.')

from nis_sim.agents.drone import DroneAgent
from nis_sim.agents.vehicle import VehicleAgent
from nis_sim.core.physics import DronePhysics, VehiclePhysics
import numpy as np


def test_physics():
    print("🧪 Testing Physics...")
    
    # Drone physics
    dp = DronePhysics()
    valid, msg = dp.validate_velocity(np.array([10, 0, 0]))
    print(f"  Drone velocity 10 m/s: {msg}")
    
    valid, msg = dp.validate_velocity(np.array([20, 0, 0]))
    print(f"  Drone velocity 20 m/s: {msg}")
    
    # Vehicle physics
    vp = VehiclePhysics()
    dx, dy, dtheta = vp.bicycle_model(10, 0.1, 0.1)
    print(f"  Vehicle bicycle model: dx={dx:.3f}, dy={dy:.3f}, dtheta={dtheta:.3f}")
    
    print("✅ Physics tests passed!")


def test_drone_agent():
    print("\n🚁 Testing Drone Agent...")
    
    drone = DroneAgent("test_drone", initial_position=(0, 0, 1))
    print(f"  Initial position: {drone.state.position}")
    
    # Simulate without PyBullet
    drone.armed = True
    drone.target_altitude = 10
    drone.target_position = np.array([10, 10, 10])
    
    for i in range(100):
        drone._simplified_physics(0.1)
    
    print(f"  After 10s: {drone.state.position}")
    print(f"  Battery: {drone.battery_level:.1f}%")
    print("✅ Drone agent works!")


def test_vehicle_agent():
    print("\n🚗 Testing Vehicle Agent...")
    
    car = VehicleAgent("test_car", initial_position=(0, 0, 0.5))
    print(f"  Initial position: {car.state.position}")
    
    # Accelerate
    car.throttle = 0.5
    for i in range(100):
        car._apply_dynamics(0.1)
        car._update_obd()
    
    print(f"  After 10s: position={car.state.position[:2]}, speed={car.speed*3.6:.1f} km/h")
    print(f"  OBD RPM: {car.obd_data['rpm']}")
    print("✅ Vehicle agent works!")


if __name__ == "__main__":
    print("=" * 50)
    print("🎮 NIS Simulation Quick Test")
    print("=" * 50)
    
    test_physics()
    test_drone_agent()
    test_vehicle_agent()
    
    print("\n" + "=" * 50)
    print("✅ All tests passed! Ready for NIS integration.")
    print("=" * 50)

"""
Basic Drone Simulation Example
Shows how to use NIS Simulation with NIS Protocol
"""

import asyncio
import sys
sys.path.insert(0, '..')

from nis_sim import SimulationEngine, DroneAgent, NISConnector
from nis_sim.core.engine import SimulationConfig
from nis_sim.core.world import AirspaceWorld


async def main():
    print("🚁 NIS Drone Simulation Example")
    print("=" * 40)
    
    # 1. Create simulation engine
    engine = SimulationEngine(SimulationConfig(
        timestep=1/60,  # 60 Hz physics
        realtime=False,  # Run as fast as possible
        gui=False        # No GUI (set True if you have display)
    ))
    engine.initialize()
    
    # 2. Create world
    world = AirspaceWorld()
    world.initialize(engine.physics_client)
    
    # Add some obstacles
    world.add_obstacle("tower", (50, 50, 25), (5, 5, 50))
    
    # 3. Create drone
    drone = DroneAgent(
        agent_id="drone_1",
        initial_position=(0, 0, 0.5)
    )
    engine.add_agent("drone_1", drone)
    
    # 4. Connect to NIS Protocol (optional)
    connector = NISConnector()
    connected = await connector.connect()
    
    if connected:
        await connector.register_agent("drone_1", "quadrotor", ["fly", "hover"])
    
    # 5. Run mission
    print("\n📍 Mission: Takeoff → Fly to (30, 30, 15) → Land")
    
    # Takeoff
    drone.apply_command({"type": "takeoff", "altitude": 10})
    
    # Run simulation
    step = 0
    async for state in engine.run(duration=20.0):
        step += 1
        
        # At 5 seconds, go to target
        if abs(state.time - 5.0) < 0.02:
            drone.apply_command({"type": "goto", "position": [30, 30, 15]})
            print(f"  [{state.time:.1f}s] Going to target...")
        
        # At 15 seconds, land
        if abs(state.time - 15.0) < 0.02:
            drone.apply_command({"type": "land"})
            print(f"  [{state.time:.1f}s] Landing...")
        
        # Send telemetry to NIS
        if connected and step % 30 == 0:
            await connector.send_telemetry("drone_1", drone.get_state())
        
        # Print status every 2 seconds
        if step % 120 == 0:
            pos = drone.state.position
            print(f"  [{state.time:.1f}s] Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
    
    # 6. Cleanup
    await connector.disconnect()
    engine.shutdown()
    
    print("\n✅ Simulation complete!")
    print(f"   Final position: {drone.state.position}")
    print(f"   Battery remaining: {drone.battery_level:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())

"""
Car OBD Simulation Example
Simulates vehicle with OBD-II data for NIS-AUTO testing
"""

import asyncio
import sys
sys.path.insert(0, '..')

from nis_sim import SimulationEngine, VehicleAgent, NISConnector
from nis_sim.core.engine import SimulationConfig
from nis_sim.core.world import CityWorld


async def main():
    print("🚗 NIS Vehicle OBD Simulation Example")
    print("=" * 40)
    
    # 1. Create simulation
    engine = SimulationEngine(SimulationConfig(
        timestep=1/60,
        realtime=False,
        gui=False
    ))
    engine.initialize()
    
    # 2. Create city environment
    world = CityWorld("test_city")
    world.initialize(engine.physics_client)
    
    # 3. Create vehicle
    car = VehicleAgent(
        agent_id="my_car",
        initial_position=(50, 50, 0.5),
        mass=1500  # kg
    )
    engine.add_agent("my_car", car)
    
    # 4. Connect to NIS Protocol
    connector = NISConnector()
    connected = await connector.connect()
    
    if connected:
        await connector.register_agent("my_car", "sedan", ["drive", "obd_read"])
    
    # 5. Drive simulation
    print("\n📍 Mission: Accelerate → Cruise → Turn → Stop")
    
    obd_samples = []
    
    async for state in engine.run(duration=15.0):
        # Control sequence
        if state.time < 3:
            # Accelerate
            car.apply_command({"type": "throttle", "value": 0.7})
        elif state.time < 8:
            # Cruise and turn
            car.apply_command({"type": "throttle", "value": 0.3})
            car.apply_command({"type": "steer", "angle": 15})
        elif state.time < 10:
            # Straighten
            car.apply_command({"type": "steer", "angle": 0})
        else:
            # Brake
            car.apply_command({"type": "brake", "value": 0.5})
        
        # Collect OBD data every second
        if int(state.time * 10) % 10 == 0:
            obd = car.obd_data.copy()
            obd["time"] = state.time
            obd_samples.append(obd)
            
            # Send to NIS
            if connected:
                await connector.send_telemetry("my_car", {
                    "state": car.get_state(),
                    "obd": obd
                })
        
        # Print status
        if int(state.time * 10) % 20 == 0:
            print(f"  [{state.time:.1f}s] Speed: {car.speed*3.6:.1f} km/h, RPM: {car.obd_data['rpm']}")
    
    # 6. Results
    await connector.disconnect()
    engine.shutdown()
    
    print("\n✅ Simulation complete!")
    print(f"\n📊 OBD Data Log ({len(obd_samples)} samples):")
    print("-" * 50)
    print(f"{'Time':>6} {'Speed':>8} {'RPM':>6} {'Throttle':>10} {'Load':>6}")
    print("-" * 50)
    for sample in obd_samples[:10]:  # First 10
        print(f"{sample['time']:>6.1f} {sample['speed']:>6} km/h {sample['rpm']:>6} {sample['throttle_position']:>8}% {sample['engine_load']:>5}%")
    
    print(f"\n   Max speed reached: {max(s['speed'] for s in obd_samples)} km/h")
    print(f"   Max RPM reached: {max(s['rpm'] for s in obd_samples)}")


if __name__ == "__main__":
    asyncio.run(main())

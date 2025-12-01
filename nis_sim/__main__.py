"""
NIS Simulation CLI
Run simulations from command line
"""

import asyncio
import argparse
from .scenarios.drone_delivery import DroneDeliveryScenario
from .scenarios.vehicle_navigation import VehicleNavigationScenario


async def main():
    parser = argparse.ArgumentParser(description="NIS Simulation Environment")
    parser.add_argument("--scenario", "-s", 
                        choices=["drone", "vehicle"],
                        default="drone",
                        help="Scenario to run")
    parser.add_argument("--duration", "-d",
                        type=float,
                        default=30.0,
                        help="Simulation duration in seconds")
    parser.add_argument("--agents", "-a",
                        type=int,
                        default=1,
                        help="Number of agents")
    parser.add_argument("--nis-host",
                        default="localhost",
                        help="NIS Protocol host")
    parser.add_argument("--nis-port",
                        type=int,
                        default=8000,
                        help="NIS Protocol port")
    parser.add_argument("--no-nis",
                        action="store_true",
                        help="Run without NIS Protocol connection")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🎮 NIS Simulation Environment")
    print("=" * 50)
    
    if args.scenario == "drone":
        scenario = DroneDeliveryScenario(
            num_drones=args.agents,
            connect_nis=not args.no_nis
        )
        await scenario.setup()
        
        # Add default missions
        scenario.add_mission((50, 50, 20), (200, 200, 20))
        scenario.add_mission((100, 50, 20), (50, 200, 20))
        
        results = await scenario.run(duration=args.duration)
        await scenario.cleanup()
        
    elif args.scenario == "vehicle":
        scenario = VehicleNavigationScenario(
            num_vehicles=args.agents,
            connect_nis=not args.no_nis
        )
        await scenario.setup()
        
        # Add default waypoints
        scenario.add_waypoint((100, 50, 0), speed_limit=30)
        scenario.add_waypoint((100, 150, 0), speed_limit=50)
        scenario.add_waypoint((200, 150, 0), speed_limit=40)
        
        results = await scenario.run(duration=args.duration)
        await scenario.cleanup()
    
    print("\n" + "=" * 50)
    print("📊 Final Results:")
    print("=" * 50)
    for key, value in results.items():
        if not isinstance(value, list) or len(value) < 5:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())

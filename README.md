# NIS Simulation Environment

**Virtual testing for NIS Protocol - Drones, Vehicles, Cities, Space**

## 🎯 Why This Exists

Test NIS-DRONE and NIS-AUTO deployments **before** touching real hardware.

## 🏗️ Architecture Decision

**NO Unreal/Unity needed!** Too heavy for MVP. We use:

| Component | Tool | Why |
|-----------|------|-----|
| **Physics** | PyBullet | Robotics-focused, OpenAI/Google use it |
| **Visualization** | Three.js (later) | Lightweight web-based |
| **Control** | NIS Protocol API | Your existing system |

Later: Add **NVIDIA Isaac Sim** for enterprise customers.

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     NIS Protocol API                         │
│              (localhost:8000 or cloud)                       │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP/WebSocket
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   NIS Simulation Bridge                      │
│         (nis_sim/bridge/nis_connector.py)                   │
│   • Send telemetry    • Receive commands                    │
│   • Physics validation • Consciousness eval                 │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│                   Simulation Engine                          │
│              (nis_sim/core/engine.py)                       │
│   • 60-240 Hz physics loop                                  │
│   • Agent management                                        │
│   • Event detection                                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  DroneAgent   │ │ VehicleAgent  │ │ (Future)      │
│  🚁           │ │ 🚗            │ │ SatelliteAgent│
│  • Quadrotor  │ │ • Bicycle     │ │ • Orbital     │
│    dynamics   │ │   model       │ │   mechanics   │
│  • Battery    │ │ • OBD-II data │ │ • Comms delay │
└───────────────┘ └───────────────┘ └───────────────┘
        │                 │
        ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    PyBullet Physics                          │
│   • Collision detection  • Rigid body dynamics              │
│   • Gravity              • Contact forces                   │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Structure

```
NIS_Simulation/
├── nis_sim/
│   ├── core/
│   │   ├── engine.py      # Main simulation loop
│   │   ├── physics.py     # Physics validation
│   │   └── world.py       # Environment (city, airspace)
│   ├── agents/
│   │   ├── drone.py       # Quadrotor simulation
│   │   └── vehicle.py     # Ground vehicle + OBD
│   ├── bridge/
│   │   └── nis_connector.py  # NIS Protocol API bridge
│   └── scenarios/
│       ├── drone_delivery.py
│       └── vehicle_navigation.py
├── examples/
│   ├── basic_drone_sim.py
│   └── car_obd_sim.py
├── requirements.txt
└── test_quick.py
```

## 🚀 Quick Start

```bash
# Install (PyBullet optional for basic testing)
pip install -r requirements.txt

# Quick test (no PyBullet needed)
python test_quick.py

# Run drone scenario
python -m nis_sim --scenario drone --duration 30

# Run vehicle scenario  
python -m nis_sim --scenario vehicle --duration 20

# Connect to NIS Protocol
python -m nis_sim --scenario drone --nis-host localhost --nis-port 8000
```

## 🔌 NIS Protocol Integration

```python
from nis_sim import SimulationEngine, DroneAgent, NISConnector

# Create simulation
engine = SimulationEngine()
engine.initialize()

# Add drone
drone = DroneAgent("drone_1", initial_position=(0, 0, 1))
engine.add_agent("drone_1", drone)

# Connect to NIS Protocol
connector = NISConnector()
await connector.connect()

# Run with NIS control
async for state in engine.run(duration=60):
    # Send telemetry to NIS
    await connector.send_telemetry("drone_1", drone.get_state())
    
    # Get commands from NIS consciousness
    command = await connector.get_command("drone_1")
    if command:
        drone.apply_command(command)
```

## 🎮 Supported Scenarios

| Scenario | Agents | Tests |
|----------|--------|-------|
| **Drone Delivery** | Quadrotors | Takeoff, navigation, landing, battery |
| **Vehicle Navigation** | Cars | Steering, speed, OBD data, waypoints |
| **City (future)** | Mixed | Traffic, infrastructure, emergencies |
| **Space (future)** | Satellites | Orbital mechanics, comms delay |

## 🔮 Roadmap

- [ ] **Phase 1** (Now): Basic drone + vehicle simulation ✅
- [ ] **Phase 2**: Three.js web visualization
- [ ] **Phase 3**: Multi-agent swarm scenarios
- [ ] **Phase 4**: NVIDIA Isaac Sim integration
- [ ] **Phase 5**: Digital twin from real sensor data

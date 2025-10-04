# NIS Protocol Integration Templates

Complete integration examples for domain-specific deployments of NIS Protocol.

## Overview

These templates demonstrate how to integrate NIS Protocol into real-world applications across different domains:

- **🚁 Drone Integration** - UAV/drone control systems
- **🚗 Auto Integration** - Vehicle diagnostics and fleet management
- **🏙️ City Integration** - Smart city IoT systems

## Quick Start

### 1. NIS-DRONE (Autonomous Drones)

```python
from nis_protocol import NISCore
from nis_protocol.plugins import DronePlugin

# Initialize
drone = DroneControlSystem()
await drone.initialize()

# Autonomous navigation
result = await drone.autonomous_flight(
    "Navigate to GPS coordinates 37.7749° N, 122.4194° W at 50m altitude"
)

# Obstacle avoidance
sensor_data = {'obstacle_distance': 15, 'obstacle_direction': 'front'}
avoidance = await drone.obstacle_avoidance(sensor_data)

# Mission planning
waypoints = [
    {'lat': 37.7749, 'lon': -122.4194, 'alt': 50},
    {'lat': 37.7750, 'lon': -122.4195, 'alt': 60}
]
plan = await drone.mission_planning(waypoints)
```

**Use Cases:**
- Autonomous delivery drones
- Inspection and surveillance
- Agricultural monitoring
- Search and rescue operations

### 2. NIS-AUTO (Vehicle Diagnostics)

```python
from nis_protocol import NISCore
from nis_protocol.plugins import AutoPlugin

# Initialize
vehicle = VehicleDiagnosticSystem({
    'make': 'Toyota',
    'model': 'Camry',
    'year': 2022
})
await vehicle.initialize()

# Diagnose error codes
diagnosis = await vehicle.diagnose_error_code('P0420')

# Predictive maintenance
maintenance = await vehicle.predictive_maintenance(
    mileage=45000,
    last_service={'date': '2024-06-15', 'type': 'oil_change'}
)

# Performance analysis
telemetry = {
    'avg_speed': 65,
    'fuel_efficiency': 28.5,
    'engine_temp': 195
}
performance = await vehicle.performance_analysis(telemetry)

# Fuel optimization
optimization = await vehicle.fuel_optimization(driving_data)
```

**Use Cases:**
- Fleet management systems
- Personal vehicle diagnostics
- Automotive repair shops
- Insurance telematics

### 3. NIS-CITY (Smart City IoT)

```python
from nis_protocol import NISCore
from nis_protocol.plugins import CityPlugin

# Initialize
city = SmartCitySystem({
    'city_name': 'San Francisco',
    'population': 873965
})
await city.initialize()

# Traffic optimization
traffic_data = {
    'congestion_zones': ['Downtown', 'Mission District'],
    'avg_speed_mph': 15
}
traffic = await city.optimize_traffic(traffic_data)

# Energy management
consumption = {
    'current_mw': 450,
    'predicted_peak_mw': 520,
    'renewable_percent': 35
}
energy = await city.energy_management(consumption)

# Waste collection optimization
collection_data = {
    'bins': [
        {'id': 'BIN001', 'fill_level': 85, 'location': [37.7749, -122.4194]}
    ]
}
waste = await city.waste_optimization(collection_data)

# Environmental monitoring
sensors = {
    'air_quality_index': 65,
    'pm25': 35,
    'temperature_f': 72
}
environment = await city.environmental_monitoring(sensors)
```

**Use Cases:**
- Smart city management
- Traffic optimization
- Energy grid management
- Waste collection routing
- Environmental monitoring

## Features

### Common Capabilities

All integrations share these NIS Protocol features:

✅ **Autonomous Decision Making** - AI analyzes intent and executes autonomously
✅ **Physics Validation** - Real-time physics checking (PINN)
✅ **Multi-LLM Support** - OpenAI, Anthropic, Google, DeepSeek, Kimi
✅ **Smart Consensus** - Combine multiple LLMs for better decisions
✅ **Real-time Processing** - Low-latency responses
✅ **Conversation Memory** - Context-aware interactions
✅ **Plugin Architecture** - Domain-specific extensions

### Domain-Specific Features

#### Drone Plugin
- GPS navigation
- Obstacle detection
- Flight planning
- Sensor fusion (camera, LIDAR, GPS)
- Weather-aware decisions
- Multi-drone coordination

#### Auto Plugin
- OBD-II diagnostics
- Error code interpretation
- Predictive maintenance
- Performance monitoring
- Fuel optimization
- Fleet coordination

#### City Plugin
- Traffic flow optimization
- Energy distribution
- Waste collection routing
- Environmental monitoring
- Public safety coordination
- Smart lighting control

## Installation

```bash
# Install NIS Protocol
pip install nis-protocol

# Or from source
cd /path/to/NIS_Protocol
pip install -e .

# Install domain plugins
pip install nis-protocol[drone]   # For drone integration
pip install nis-protocol[auto]    # For vehicle integration
pip install nis-protocol[city]    # For smart city integration
```

## Running Examples

```bash
# Drone integration
python examples/integrations/drone_integration.py

# Auto integration
python examples/integrations/auto_integration.py

# City integration
python examples/integrations/city_integration.py
```

## Configuration

Each integration can be configured via environment variables or config files:

```python
# Option 1: Environment variables
import os
os.environ['NIS_LLM_PROVIDER'] = 'openai'
os.environ['NIS_API_KEY'] = 'your-key'

# Option 2: Config dict
config = {
    'llm_provider': 'openai',
    'api_key': 'your-key',
    'enable_physics': True,
    'enable_consensus': True
}

nis = NISCore(config)
```

## Architecture

```
NIS Protocol Core
       │
  Plugin Manager
       │
   ┌───┴───┐
   ▼       ▼       ▼
Drone   Auto    City
Plugin  Plugin  Plugin
   │       │       │
   ▼       ▼       ▼
Hardware/API Integration
```

## Customization

### Creating Custom Plugins

```python
from nis_protocol.plugins import BasePlugin

class MyCustomPlugin(BasePlugin):
    def __init__(self, config: dict):
        super().__init__("my_plugin", config)
        
    async def initialize(self):
        # Setup logic
        pass
        
    async def process(self, query: str):
        # Processing logic
        pass
```

### Extending Existing Plugins

```python
from nis_protocol.plugins import DronePlugin

class MyDronePlugin(DronePlugin):
    async def custom_feature(self):
        # Add custom functionality
        pass
```

## Production Deployment

### Docker Deployment

```bash
# Build with specific plugin
docker build -t nis-drone -f Dockerfile --build-arg PLUGIN=drone .
docker run -p 8000:8000 nis-drone
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nis-drone
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: nis-drone
        image: nis-drone:latest
        env:
        - name: NIS_PLUGIN
          value: "drone"
```

## Performance Optimization

- **Enable Caching**: Reduce LLM calls
- **Use Smart Consensus**: Only for critical decisions
- **Optimize Context**: Limit conversation history
- **Batch Requests**: Process multiple queries together

## Troubleshooting

### Common Issues

**Plugin not loading:**
```bash
# Check plugin is installed
python -c "from nis_protocol.plugins import DronePlugin"
```

**LLM not responding:**
```bash
# Verify API keys
echo $NIS_API_KEY

# Test connection
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test"}'
```

**Physics validation failing:**
```bash
# Check PyTorch installation
python -c "import torch; print(torch.__version__)"
```

## Support

- **Documentation**: `/system/docs/`
- **Examples**: `/examples/`
- **Issues**: GitHub Issues
- **Community**: Discord/Slack

## License

Same as NIS Protocol main project

## Next Steps

1. **Try the examples**: Run the integration scripts
2. **Customize config**: Adapt to your use case
3. **Deploy to production**: Use Docker/Kubernetes
4. **Monitor performance**: Track metrics and optimize
5. **Scale horizontally**: Add more instances as needed


#!/usr/bin/env python3
"""
NIS Protocol Project Initialization
===================================

Create new NIS Protocol projects from templates.

Templates:
- basic: Simple single-agent project
- edge: Edge device deployment
- drone: Drone/UAV control system
- robot: Robotics application
- city: Smart city infrastructure
- industrial: Industrial automation
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any
import click


# Project Templates
TEMPLATES = {
    "basic": {
        "description": "Basic NIS Protocol project with a single agent",
        "files": {
            "main.py": """#!/usr/bin/env python3
\"\"\"
Basic NIS Protocol Application
=============================

A simple example showing how to create and run a NIS Protocol agent.
\"\"\"

import asyncio
from nis_protocol import NISPlatform, create_development_platform
from nis_protocol.agents import ConsciousnessAgent

async def main():
    # Create development platform
    platform = create_development_platform("basic-project")
    
    # Create a consciousness agent
    agent = ConsciousnessAgent("consciousness_001")
    
    # Add agent to platform
    await platform.add_agent(agent)
    
    # Start the platform
    await platform.start()
    
    print("🧠 NIS Protocol Basic Project Running!")
    print("Platform Status:", platform.get_status())
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\\n👋 Shutting down...")
        await platform.stop()

if __name__ == "__main__":
    asyncio.run(main())
""",
            "requirements.txt": """nis-protocol>=3.2.0
asyncio
""",
            "config.json": """{
  "platform": {
    "name": "basic-project",
    "max_agents": 10,
    "deployment_target": "local"
  },
  "agents": {
    "consciousness_001": {
      "type": "consciousness",
      "config": {
        "reflection_interval": 30
      }
    }
  }
}""",
        }
    },
    
    "edge": {
        "description": "Edge device deployment with optimized configuration",
        "files": {
            "main.py": """#!/usr/bin/env python3
\"\"\"
NIS Protocol Edge Device Application
===================================

Optimized for deployment on edge devices like Raspberry Pi.
\"\"\"

import asyncio
from nis_protocol import NISPlatform, create_edge_platform
from nis_protocol.agents import ConsciousnessAgent, PhysicsAgent, VisionAgent

async def main():
    # Create edge-optimized platform
    platform = create_edge_platform("edge-device", device_type="raspberry_pi")
    
    # Add core agents for edge processing
    consciousness = ConsciousnessAgent("consciousness_edge")
    physics = PhysicsAgent("physics_validator") 
    vision = VisionAgent("vision_processor")
    
    await platform.add_agent(consciousness)
    await platform.add_agent(physics)
    await platform.add_agent(vision)
    
    # Deploy to edge
    await platform.deploy("edge", device_type="raspberry_pi")
    
    # Start the platform
    await platform.start()
    
    print("🔧 NIS Protocol Edge Device Running!")
    print("Device Type: Raspberry Pi")
    print("Agents:", len(platform.agents))
    
    # Edge monitoring loop
    try:
        while True:
            status = platform.get_status()
            print(f"📊 Status: {status['platform']['state']} | Agents: {status['agents']['count']}")
            await asyncio.sleep(10)
    except KeyboardInterrupt:
        print("\\n👋 Shutting down edge device...")
        await platform.stop()

if __name__ == "__main__":
    asyncio.run(main())
""",
            "requirements.txt": """nis-protocol[edge]>=3.2.0
psutil>=5.9.0
""",
            "edge_config.json": """{
  "platform": {
    "name": "edge-device",
    "max_agents": 5,
    "deployment_target": "edge",
    "auto_scaling": false,
    "edge_config": {
      "device_type": "raspberry_pi",
      "cpu_limit": 80,
      "memory_limit": 512
    }
  },
  "optimization": {
    "low_power_mode": true,
    "vision_resolution": "720p",
    "processing_interval": 1.0
  }
}""",
            "deploy.sh": """#!/bin/bash
# Edge deployment script

echo "🚀 Deploying NIS Protocol to Edge Device..."

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export NIS_DEVICE_TYPE=raspberry_pi
export NIS_LOW_POWER=true

# Run the application
python main.py
""",
        }
    },
    
    "drone": {
        "description": "Drone/UAV control system with flight safety",
        "files": {
            "main.py": """#!/usr/bin/env python3
\"\"\"
NIS Protocol Drone Control System
=================================

Autonomous drone control with physics validation and safety systems.
\"\"\"

import asyncio
from nis_protocol import NISPlatform, create_edge_platform
from nis_protocol.agents import ConsciousnessAgent, PhysicsAgent, VisionAgent

class DroneControlAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.altitude = 0.0
        self.velocity = [0.0, 0.0, 0.0]  # x, y, z
        self.position = [0.0, 0.0, 0.0]
        
    async def takeoff(self, target_altitude: float):
        print(f"🚁 Taking off to {target_altitude}m...")
        # Physics-validated takeoff
        
    async def land(self):
        print("🛬 Landing...")
        
    async def navigate_to(self, coordinates: list):
        print(f"🧭 Navigating to {coordinates}...")

async def main():
    # Create drone platform
    platform = create_edge_platform("drone-system", device_type="drone_controller")
    
    # Add AI agents
    consciousness = ConsciousnessAgent("flight_consciousness")
    physics = PhysicsAgent("flight_physics")
    vision = VisionAgent("navigation_vision")
    
    # Create drone control agent
    drone_control = DroneControlAgent("drone_control_001")
    
    await platform.add_agent(consciousness)
    await platform.add_agent(physics)
    await platform.add_agent(vision)
    
    # Deploy and start
    await platform.deploy("edge", device_type="drone")
    await platform.start()
    
    print("🚁 NIS Protocol Drone System Active!")
    
    # Flight mission simulation
    try:
        await drone_control.takeoff(10.0)
        await asyncio.sleep(2)
        
        await drone_control.navigate_to([100, 100, 10])
        await asyncio.sleep(5)
        
        await drone_control.land()
        
        # Keep monitoring
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\\n🛬 Emergency landing...")
        await drone_control.land()
        await platform.stop()

if __name__ == "__main__":
    asyncio.run(main())
""",
            "requirements.txt": """nis-protocol[drone]>=3.2.0
pyserial>=3.5
gps>=3.20
""",
            "flight_config.json": """{
  "drone": {
    "max_altitude": 120,
    "max_speed": 15,
    "safety_radius": 500,
    "return_to_home_battery": 25
  },
  "physics_validation": {
    "wind_resistance": true,
    "weight_limits": true,
    "battery_physics": true
  },
  "navigation": {
    "gps_required": true,
    "vision_backup": true,
    "obstacle_avoidance": true
  }
}""",
        }
    },
    
    "robot": {
        "description": "Robotics application with motor control and sensors",
        "files": {
            "main.py": """#!/usr/bin/env python3
\"\"\"
NIS Protocol Robotics Application
=================================

Intelligent robotics control with sensor fusion and physics validation.
\"\"\"

import asyncio
from nis_protocol import NISPlatform, create_edge_platform
from nis_protocol.agents import ConsciousnessAgent, PhysicsAgent, VisionAgent

class RobotControlAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.position = [0.0, 0.0]  # x, y
        self.orientation = 0.0  # degrees
        
    async def move_forward(self, distance: float):
        print(f"🤖 Moving forward {distance}m...")
        
    async def turn(self, angle: float):
        print(f"🔄 Turning {angle} degrees...")
        
    async def pick_object(self, object_id: str):
        print(f"🦾 Picking up object: {object_id}")
        
    async def scan_environment(self):
        print("👁️ Scanning environment...")
        return {"objects": ["box", "wall", "person"], "obstacles": ["chair"]}

async def main():
    # Create robotics platform
    platform = create_edge_platform("robot-system", device_type="robot_controller")
    
    # Add AI agents
    consciousness = ConsciousnessAgent("robot_consciousness")
    physics = PhysicsAgent("movement_physics")
    vision = VisionAgent("object_vision")
    
    # Create robot control agent
    robot = RobotControlAgent("robot_001")
    
    await platform.add_agent(consciousness)
    await platform.add_agent(physics)
    await platform.add_agent(vision)
    
    # Deploy and start
    await platform.deploy("edge", device_type="robot")
    await platform.start()
    
    print("🤖 NIS Protocol Robot System Online!")
    
    # Robot mission
    try:
        # Scan environment
        environment = await robot.scan_environment()
        print(f"Environment: {environment}")
        
        # Navigate and interact
        await robot.move_forward(2.0)
        await robot.turn(90)
        await robot.pick_object("box")
        
        # Monitoring loop
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\\n🛑 Robot stopped")
        await platform.stop()

if __name__ == "__main__":
    asyncio.run(main())
""",
            "requirements.txt": """nis-protocol[robotics]>=3.2.0
pyserial>=3.5
""",
            "robot_config.json": """{
  "robot": {
    "max_speed": 2.0,
    "max_payload": 5.0,
    "sensor_update_rate": 10
  },
  "safety": {
    "collision_avoidance": true,
    "emergency_stop": true,
    "human_detection": true
  },
  "capabilities": {
    "manipulation": true,
    "navigation": true,
    "object_recognition": true
  }
}""",
        }
    },
    
    "city": {
        "description": "Smart city infrastructure with distributed agents",
        "files": {
            "main.py": """#!/usr/bin/env python3
\"\"\"
NIS Protocol Smart City System
==============================

Distributed AI infrastructure for smart city management.
\"\"\"

import asyncio
from nis_protocol import NISPlatform, create_cloud_platform
from nis_protocol.agents import ConsciousnessAgent, PhysicsAgent, VisionAgent

class TrafficControlAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.traffic_lights = {}
        
    async def optimize_traffic_flow(self, intersection_id: str):
        print(f"🚦 Optimizing traffic at intersection {intersection_id}")
        
    async def detect_incidents(self):
        print("🚨 Monitoring for traffic incidents...")

class EnvironmentalAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        
    async def monitor_air_quality(self):
        print("🌱 Monitoring air quality...")
        
    async def manage_energy_grid(self):
        print("⚡ Optimizing energy distribution...")

async def main():
    # Create smart city platform
    platform = create_cloud_platform("smart-city", provider="aws")
    
    # Add core city agents
    consciousness = ConsciousnessAgent("city_consciousness")
    physics = PhysicsAgent("physics_validator")
    vision = VisionAgent("city_vision")
    
    # Add specialized city agents
    traffic = TrafficControlAgent("traffic_control")
    environment = EnvironmentalAgent("environment_monitor")
    
    await platform.add_agent(consciousness)
    await platform.add_agent(physics)
    await platform.add_agent(vision)
    
    # Deploy to cloud
    await platform.deploy("cloud", provider="aws")
    await platform.start()
    
    print("🏙️ NIS Protocol Smart City System Online!")
    
    # City management loop
    try:
        while True:
            await traffic.optimize_traffic_flow("intersection_001")
            await environment.monitor_air_quality()
            await environment.manage_energy_grid()
            
            await asyncio.sleep(30)  # Update every 30 seconds
            
    except KeyboardInterrupt:
        print("\\n🏙️ Smart city system shutting down...")
        await platform.stop()

if __name__ == "__main__":
    asyncio.run(main())
""",
            "requirements.txt": """nis-protocol[city]>=3.2.0
geopandas>=0.13.0
folium>=0.14.0
""",
            "city_config.json": """{
  "city": {
    "name": "SmartCity",
    "population": 500000,
    "area_km2": 150
  },
  "infrastructure": {
    "traffic_lights": 200,
    "sensors": 1000,
    "cameras": 300
  },
  "services": {
    "traffic_optimization": true,
    "energy_management": true,
    "environmental_monitoring": true,
    "emergency_response": true
  }
}""",
        }
    },
    
    "industrial": {
        "description": "Industrial automation with quality control",
        "files": {
            "main.py": """#!/usr/bin/env python3
\"\"\"
NIS Protocol Industrial Automation
==================================

Factory automation with AI-driven quality control and optimization.
\"\"\"

import asyncio
from nis_protocol import NISPlatform, create_edge_platform
from nis_protocol.agents import ConsciousnessAgent, PhysicsAgent, VisionAgent

class ProductionLineAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.production_rate = 0
        self.quality_score = 0.0
        
    async def start_production(self):
        print("🏭 Starting production line...")
        
    async def quality_inspection(self, item_id: str):
        print(f"🔍 Inspecting item {item_id}...")
        return {"passed": True, "score": 0.95}
        
    async def optimize_efficiency(self):
        print("⚡ Optimizing production efficiency...")

class SafetyAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        
    async def monitor_safety(self):
        print("🦺 Monitoring safety conditions...")
        
    async def emergency_shutdown(self):
        print("🚨 EMERGENCY SHUTDOWN ACTIVATED!")

async def main():
    # Create industrial platform
    platform = create_edge_platform("factory-system", device_type="industrial_pc")
    
    # Add AI agents
    consciousness = ConsciousnessAgent("factory_consciousness")
    physics = PhysicsAgent("process_physics")
    vision = VisionAgent("quality_vision")
    
    # Add industrial agents
    production = ProductionLineAgent("production_001")
    safety = SafetyAgent("safety_monitor")
    
    await platform.add_agent(consciousness)
    await platform.add_agent(physics)
    await platform.add_agent(vision)
    
    # Deploy to industrial environment
    await platform.deploy("edge", device_type="industrial")
    await platform.start()
    
    print("🏭 NIS Protocol Industrial System Active!")
    
    # Production loop
    try:
        await production.start_production()
        
        item_count = 0
        while True:
            # Process items
            item_id = f"item_{item_count:06d}"
            quality = await production.quality_inspection(item_id)
            
            if quality["passed"]:
                print(f"✅ Item {item_id} passed quality control")
            else:
                print(f"❌ Item {item_id} failed quality control")
            
            # Safety monitoring
            await safety.monitor_safety()
            
            # Optimize periodically
            if item_count % 100 == 0:
                await production.optimize_efficiency()
            
            item_count += 1
            await asyncio.sleep(1)  # 1 item per second
            
    except KeyboardInterrupt:
        print("\\n🏭 Shutting down production...")
        await safety.emergency_shutdown()
        await platform.stop()

if __name__ == "__main__":
    asyncio.run(main())
""",
            "requirements.txt": """nis-protocol[industrial]>=3.2.0
modbus-tk>=1.1.2
opcua>=0.98.13
""",
            "factory_config.json": """{
  "factory": {
    "name": "SmartFactory",
    "production_lines": 3,
    "target_efficiency": 85
  },
  "quality_control": {
    "inspection_rate": 100,
    "quality_threshold": 0.95,
    "reject_handling": "automatic"
  },
  "safety": {
    "emergency_stops": 10,
    "safety_cameras": 15,
    "gas_sensors": 8
  }
}""",
        }
    }
}


def init_project(project_name: str, template: str, path: str, verbose: bool = False) -> bool:
    """
    Initialize a new NIS Protocol project from template.
    
    Args:
        project_name: Name of the project
        template: Template to use
        path: Base path for project creation
        verbose: Enable verbose output
        
    Returns:
        bool: True if project was created successfully
    """
    try:
        if template not in TEMPLATES:
            click.echo(f"❌ Unknown template: {template}")
            click.echo(f"Available templates: {', '.join(TEMPLATES.keys())}")
            return False
        
        # Create project directory
        project_path = Path(path) / project_name
        if project_path.exists():
            click.echo(f"❌ Directory {project_path} already exists")
            return False
        
        project_path.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            click.echo(f"📁 Created directory: {project_path}")
        
        # Create files from template
        template_data = TEMPLATES[template]
        for filename, content in template_data["files"].items():
            file_path = project_path / filename
            
            # Create subdirectories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file content
            with open(file_path, 'w') as f:
                f.write(content)
            
            if verbose:
                click.echo(f"📄 Created file: {filename}")
        
        # Create README
        readme_content = f"""# {project_name}

{template_data['description']}

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python main.py
   ```

## Project Structure

- `main.py` - Main application entry point
- `requirements.txt` - Python dependencies
- Configuration files for platform settings

## Template: {template}

This project was created using the '{template}' template from NIS Protocol v3.2.

## Next Steps

- Customize the agents for your specific use case
- Add additional sensors or actuators
- Deploy to your target environment
- Integrate with external systems

## Documentation

- [NIS Protocol Documentation](https://docs.nis-protocol.org)
- [API Reference](https://api.nis-protocol.org)
- [Community Forum](https://community.nis-protocol.org)

## Support

For help and support:
- GitHub Issues: https://github.com/Organica-Ai-Solutions/NIS_Protocol/issues
- Community Forum: https://community.nis-protocol.org
- Email: developers@organicaai.com
"""
        
        readme_path = project_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        if verbose:
            click.echo(f"📄 Created README.md")
        
        # Create .gitignore
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# NIS Protocol
logs/
cache/
models/
*.log
config.local.json
"""
        
        gitignore_path = project_path / ".gitignore"
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        
        if verbose:
            click.echo(f"📄 Created .gitignore")
        
        click.echo(f"✅ Project '{project_name}' created successfully!")
        click.echo(f"📁 Location: {project_path.absolute()}")
        click.echo(f"📝 Template: {template} - {template_data['description']}")
        
        return True
        
    except Exception as e:
        click.echo(f"❌ Error creating project: {e}")
        if verbose:
            import traceback
            click.echo(traceback.format_exc())
        return False


if __name__ == '__main__':
    import click
    
    @click.command()
    @click.argument('project_name')
    @click.option('--template', '-t', default='basic', type=click.Choice(list(TEMPLATES.keys())))
    @click.option('--path', '-p', default='.')
    @click.option('--verbose', '-v', is_flag=True)
    def main(project_name: str, template: str, path: str, verbose: bool):
        """Initialize a new NIS Protocol project."""
        success = init_project(project_name, template, path, verbose)
        if not success:
            exit(1)
    
    main()

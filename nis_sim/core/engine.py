"""
NIS Simulation Engine
Core simulation loop using PyBullet physics
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("⚠️ PyBullet not installed. Run: pip install pybullet")


@dataclass
class SimulationConfig:
    """Simulation configuration"""
    timestep: float = 1/240  # 240 Hz physics
    realtime: bool = False   # Run at real speed or max speed
    gravity: tuple = (0, 0, -9.81)
    gui: bool = False        # PyBullet GUI (for debugging)
    

@dataclass
class SimulationState:
    """Current simulation state"""
    time: float = 0.0
    step: int = 0
    agents: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class SimulationEngine:
    """
    Main simulation engine
    Wraps PyBullet physics with NIS Protocol integration
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.state = SimulationState()
        self.physics_client = None
        self.agents: Dict[str, Any] = {}
        self.world = None
        self._running = False
        
    def initialize(self):
        """Initialize physics engine"""
        if not PYBULLET_AVAILABLE:
            raise RuntimeError("PyBullet required. Install with: pip install pybullet")
        
        # Connect to physics server
        if self.config.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)  # Headless
        
        # Configure physics
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*self.config.gravity)
        p.setTimeStep(self.config.timestep)
        
        # Load ground plane
        p.loadURDF("plane.urdf")
        
        print(f"✅ Simulation engine initialized (GUI: {self.config.gui})")
        return self
    
    def add_agent(self, agent_id: str, agent: Any) -> str:
        """Add an agent to the simulation"""
        self.agents[agent_id] = agent
        agent.spawn(self.physics_client)
        self.state.agents[agent_id] = agent.get_state()
        print(f"✅ Agent '{agent_id}' added to simulation")
        return agent_id
    
    def remove_agent(self, agent_id: str):
        """Remove an agent from simulation"""
        if agent_id in self.agents:
            self.agents[agent_id].despawn()
            del self.agents[agent_id]
            del self.state.agents[agent_id]
    
    def step(self) -> SimulationState:
        """Advance simulation by one timestep"""
        # Step physics
        p.stepSimulation()
        
        # Update time
        self.state.time += self.config.timestep
        self.state.step += 1
        
        # Update all agents
        for agent_id, agent in self.agents.items():
            agent.update(self.config.timestep)
            self.state.agents[agent_id] = agent.get_state()
        
        # Check for collisions/events
        self._check_events()
        
        # Real-time sync if enabled
        if self.config.realtime:
            time.sleep(self.config.timestep)
        
        return self.state
    
    async def run(self, duration: float = 10.0, callback=None):
        """
        Run simulation for specified duration
        Yields state at each step for async iteration
        """
        self._running = True
        steps = int(duration / self.config.timestep)
        
        for i in range(steps):
            if not self._running:
                break
                
            state = self.step()
            
            if callback:
                await callback(state)
            
            yield state
            
            # Allow other async tasks
            if i % 100 == 0:
                await asyncio.sleep(0)
        
        self._running = False
    
    def stop(self):
        """Stop simulation"""
        self._running = False
    
    def reset(self):
        """Reset simulation to initial state"""
        self.state = SimulationState()
        for agent in self.agents.values():
            agent.reset()
    
    def _check_events(self):
        """Check for collisions and other events"""
        # Get contact points
        contacts = p.getContactPoints()
        
        for contact in contacts:
            body_a, body_b = contact[1], contact[2]
            
            # Find which agents are involved
            for agent_id, agent in self.agents.items():
                if hasattr(agent, 'body_id'):
                    if agent.body_id in (body_a, body_b):
                        self.state.events.append({
                            "type": "collision",
                            "time": self.state.time,
                            "agent": agent_id,
                            "position": contact[5]  # Contact position
                        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get simulation metrics"""
        return {
            "time": self.state.time,
            "steps": self.state.step,
            "agents": len(self.agents),
            "collisions": len([e for e in self.state.events if e["type"] == "collision"]),
            "fps": self.state.step / max(self.state.time, 0.001)
        }
    
    def shutdown(self):
        """Clean up physics engine"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
        print("🛑 Simulation engine shutdown")
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

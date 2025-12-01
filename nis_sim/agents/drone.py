"""
Drone Agent
Quadrotor drone simulation
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from .base import BaseAgent, AgentState
from ..core.physics import DronePhysics

try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False


class DroneAgent(BaseAgent):
    """
    Simulated quadrotor drone
    Physics-validated flight dynamics
    """
    
    def __init__(self, 
                 agent_id: str,
                 initial_position: Tuple[float, float, float] = (0, 0, 1),
                 mass: float = 2.0):
        super().__init__(agent_id, initial_position)
        self.state.position = np.array(initial_position, dtype=np.float64)
        self.mass = mass
        self.physics = DronePhysics()
        
        # Drone-specific state
        self.rotor_speeds = np.zeros(4)  # RPM
        self.battery_level = 100.0  # Percentage
        self.armed = False
        
        # Control targets
        self.target_position: Optional[np.ndarray] = None
        self.target_velocity: Optional[np.ndarray] = None
        self.target_altitude: Optional[float] = None
        
        # Sensors
        self.sensors = {
            "gps": {"position": self.state.position.tolist(), "accuracy": 1.0},
            "imu": {"acceleration": [0, 0, 0], "gyro": [0, 0, 0]},
            "barometer": {"altitude": initial_position[2], "pressure": 101325},
            "battery": {"voltage": 12.6, "current": 0, "percentage": 100}
        }
    
    def spawn(self, physics_client: int):
        """Spawn drone in physics simulation"""
        self.physics_client = physics_client
        
        if not PYBULLET_AVAILABLE:
            print(f"⚠️ PyBullet not available, using simplified physics for {self.agent_id}")
            return
        
        # Create drone body (simplified as box for now)
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.2, 0.2, 0.05]
        )
        
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.2, 0.2, 0.05],
            rgbaColor=[0.2, 0.2, 0.8, 1]  # Blue
        )
        
        self.body_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.state.position.tolist()
        )
        
        # Set initial velocity to zero
        p.resetBaseVelocity(self.body_id, [0, 0, 0], [0, 0, 0])
        
        print(f"✅ Drone '{self.agent_id}' spawned at {self.state.position}")
    
    def update(self, dt: float):
        """Update drone state from physics"""
        self._process_commands()
        
        if PYBULLET_AVAILABLE and self.body_id is not None:
            # Get state from physics engine
            pos, orn = p.getBasePositionAndOrientation(self.body_id)
            vel, ang_vel = p.getBaseVelocity(self.body_id)
            
            self.state.position = np.array(pos)
            self.state.orientation = np.array(orn)
            self.state.velocity = np.array(vel)
            self.state.angular_velocity = np.array(ang_vel)
        else:
            # Simplified physics (no PyBullet)
            self._simplified_physics(dt)
        
        # Update sensors
        self._update_sensors()
        
        # Update battery
        self._update_battery(dt)
        
        # Apply control if targets set
        if self.armed:
            self._apply_control(dt)
    
    def _simplified_physics(self, dt: float):
        """Simple physics when PyBullet not available"""
        if self.target_position is not None:
            # Simple P controller to target
            error = self.target_position - self.state.position
            desired_vel = error * 2.0  # P gain
            desired_vel = self.physics.clamp_velocity(desired_vel)
            
            # Update velocity with simple acceleration
            accel = (desired_vel - self.state.velocity) * 5.0
            self.state.velocity += accel * dt
            self.state.velocity = self.physics.clamp_velocity(self.state.velocity)
        
        # Apply gravity if not armed
        if not self.armed:
            self.state.velocity[2] -= 9.81 * dt
        
        # Update position
        self.state.position += self.state.velocity * dt
        
        # Ground collision
        if self.state.position[2] < 0:
            self.state.position[2] = 0
            self.state.velocity[2] = 0
    
    def _apply_control(self, dt: float):
        """Apply flight control"""
        if not PYBULLET_AVAILABLE or self.body_id is None:
            return
        
        # Compute required thrust to hover + control
        gravity_compensation = self.mass * 9.81
        
        # Simple altitude hold
        if self.target_altitude is not None:
            altitude_error = self.target_altitude - self.state.position[2]
            vertical_vel_error = -self.state.velocity[2]
            thrust_adjustment = altitude_error * 10.0 + vertical_vel_error * 5.0
        else:
            thrust_adjustment = 0
        
        total_thrust = gravity_compensation + thrust_adjustment
        
        # Validate thrust
        valid, msg = self.physics.validate_thrust(total_thrust)
        if not valid:
            total_thrust = self.physics.constraints.max_thrust
        
        # Apply force (simplified - just vertical)
        p.applyExternalForce(
            self.body_id,
            -1,  # Base link
            [0, 0, total_thrust],
            self.state.position.tolist(),
            p.WORLD_FRAME
        )
        
        # Position control (horizontal)
        if self.target_position is not None:
            pos_error = self.target_position - self.state.position
            pos_error[2] = 0  # Horizontal only
            
            horizontal_force = pos_error * 5.0 * self.mass
            horizontal_force = np.clip(horizontal_force, -20, 20)
            
            p.applyExternalForce(
                self.body_id,
                -1,
                horizontal_force.tolist(),
                self.state.position.tolist(),
                p.WORLD_FRAME
            )
    
    def _update_sensors(self):
        """Update sensor readings"""
        self.sensors["gps"]["position"] = self.state.position.tolist()
        self.sensors["barometer"]["altitude"] = self.state.position[2]
        self.sensors["imu"]["gyro"] = self.state.angular_velocity.tolist()
        self.sensors["battery"]["percentage"] = self.battery_level
    
    def _update_battery(self, dt: float):
        """Simulate battery drain"""
        if self.armed:
            # Drain based on thrust (simplified)
            drain_rate = 0.01  # % per second at hover
            self.battery_level -= drain_rate * dt
            self.battery_level = max(0, self.battery_level)
    
    def apply_command(self, command: Dict[str, Any]):
        """Apply control command"""
        cmd_type = command.get("type")
        
        if cmd_type == "arm":
            self.armed = True
            print(f"🚁 {self.agent_id}: Armed")
            
        elif cmd_type == "disarm":
            self.armed = False
            print(f"🚁 {self.agent_id}: Disarmed")
            
        elif cmd_type == "takeoff":
            altitude = command.get("altitude", 10.0)
            self.armed = True
            self.target_altitude = altitude
            self.target_position = self.state.position.copy()
            self.target_position[2] = altitude
            print(f"🚁 {self.agent_id}: Taking off to {altitude}m")
            
        elif cmd_type == "land":
            self.target_altitude = 0.0
            self.target_position = self.state.position.copy()
            self.target_position[2] = 0
            print(f"🚁 {self.agent_id}: Landing")
            
        elif cmd_type == "goto":
            position = command.get("position")
            if position:
                self.target_position = np.array(position)
                self.target_altitude = position[2]
                print(f"🚁 {self.agent_id}: Going to {position}")
                
        elif cmd_type == "velocity":
            velocity = command.get("velocity")
            if velocity:
                self.target_velocity = np.array(velocity)
                
        elif cmd_type == "hover":
            self.target_position = self.state.position.copy()
            self.target_velocity = None
            print(f"🚁 {self.agent_id}: Hovering")
    
    def get_state(self) -> Dict:
        """Get drone state"""
        base_state = super().get_state()
        base_state.update({
            "armed": self.armed,
            "battery": self.battery_level,
            "rotor_speeds": self.rotor_speeds.tolist(),
            "target_position": self.target_position.tolist() if self.target_position is not None else None,
            "target_altitude": self.target_altitude
        })
        return base_state

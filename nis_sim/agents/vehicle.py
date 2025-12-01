"""
Vehicle Agent
Ground vehicle (car) simulation
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from .base import BaseAgent, AgentState
from ..core.physics import VehiclePhysics

try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False


class VehicleAgent(BaseAgent):
    """
    Simulated ground vehicle
    For NIS-AUTO testing
    """
    
    def __init__(self,
                 agent_id: str,
                 initial_position: Tuple[float, float, float] = (0, 0, 0.5),
                 mass: float = 1500.0):
        super().__init__(agent_id, initial_position)
        self.mass = mass
        self.physics = VehiclePhysics()
        
        # Vehicle state
        self.speed = 0.0  # m/s
        self.steering_angle = 0.0  # radians
        self.throttle = 0.0  # 0-1
        self.brake = 0.0  # 0-1
        self.heading = 0.0  # radians
        
        # OBD-like data (for NIS-AUTO integration)
        self.obd_data = {
            "speed": 0,  # km/h
            "rpm": 800,
            "throttle_position": 0,
            "engine_load": 0,
            "coolant_temp": 90,
            "fuel_level": 75,
            "dtc_codes": []
        }
        
        # Sensors
        self.sensors = {
            "gps": {"position": self.state.position.tolist(), "speed": 0},
            "speedometer": {"speed_kmh": 0, "speed_mph": 0},
            "steering": {"angle": 0},
            "accelerometer": {"x": 0, "y": 0, "z": 0},
            "lidar": {"distances": []},  # Simplified
            "camera": {"objects": []}  # Simplified
        }
    
    def spawn(self, physics_client: int):
        """Spawn vehicle in physics simulation"""
        self.physics_client = physics_client
        
        if not PYBULLET_AVAILABLE:
            print(f"⚠️ PyBullet not available, using simplified physics for {self.agent_id}")
            return
        
        # Create vehicle body (simplified as box)
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[2.0, 1.0, 0.5]  # Car-sized
        )
        
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[2.0, 1.0, 0.5],
            rgbaColor=[0.8, 0.2, 0.2, 1]  # Red
        )
        
        self.body_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.state.position.tolist()
        )
        
        print(f"✅ Vehicle '{self.agent_id}' spawned at {self.state.position}")
    
    def update(self, dt: float):
        """Update vehicle state"""
        self._process_commands()
        
        # Apply vehicle dynamics
        self._apply_dynamics(dt)
        
        if PYBULLET_AVAILABLE and self.body_id is not None:
            # Sync with physics engine
            p.resetBasePositionAndOrientation(
                self.body_id,
                self.state.position.tolist(),
                p.getQuaternionFromEuler([0, 0, self.heading])
            )
            p.resetBaseVelocity(
                self.body_id,
                self.state.velocity.tolist(),
                [0, 0, 0]
            )
        
        # Update sensors
        self._update_sensors()
        
        # Update OBD data
        self._update_obd()
    
    def _apply_dynamics(self, dt: float):
        """Apply vehicle dynamics (bicycle model)"""
        # Compute acceleration from throttle/brake
        max_accel = 5.0  # m/s²
        max_decel = 8.0  # m/s²
        
        if self.throttle > 0:
            accel = self.throttle * max_accel
        elif self.brake > 0:
            accel = -self.brake * max_decel
        else:
            # Drag/friction
            accel = -0.5 * self.speed
        
        # Update speed
        self.speed += accel * dt
        self.speed = max(0, min(self.speed, self.physics.constraints.max_velocity))
        
        # Apply bicycle model for position update
        dx, dy, dtheta = self.physics.bicycle_model(
            self.speed,
            self.steering_angle,
            dt
        )
        
        # Update heading
        self.heading += dtheta
        
        # Update position in world frame
        cos_h, sin_h = np.cos(self.heading), np.sin(self.heading)
        world_dx = dx * cos_h - dy * sin_h
        world_dy = dx * sin_h + dy * cos_h
        
        self.state.position[0] += world_dx
        self.state.position[1] += world_dy
        
        # Update velocity vector
        self.state.velocity = np.array([
            self.speed * cos_h,
            self.speed * sin_h,
            0
        ])
    
    def _update_sensors(self):
        """Update sensor readings"""
        speed_kmh = self.speed * 3.6
        
        self.sensors["gps"]["position"] = self.state.position.tolist()
        self.sensors["gps"]["speed"] = speed_kmh
        self.sensors["speedometer"]["speed_kmh"] = speed_kmh
        self.sensors["speedometer"]["speed_mph"] = speed_kmh * 0.621371
        self.sensors["steering"]["angle"] = np.degrees(self.steering_angle)
    
    def _update_obd(self):
        """Update OBD-II like data"""
        self.obd_data["speed"] = int(self.speed * 3.6)
        self.obd_data["rpm"] = int(800 + self.throttle * 6000)
        self.obd_data["throttle_position"] = int(self.throttle * 100)
        self.obd_data["engine_load"] = int(self.throttle * 80)
    
    def apply_command(self, command: Dict[str, Any]):
        """Apply control command"""
        cmd_type = command.get("type")
        
        if cmd_type == "throttle":
            self.throttle = np.clip(command.get("value", 0), 0, 1)
            self.brake = 0
            
        elif cmd_type == "brake":
            self.brake = np.clip(command.get("value", 0), 0, 1)
            self.throttle = 0
            
        elif cmd_type == "steer":
            angle = command.get("angle", 0)  # degrees
            self.steering_angle = np.clip(
                np.radians(angle),
                -self.physics.max_steering_angle,
                self.physics.max_steering_angle
            )
            
        elif cmd_type == "set_speed":
            # Cruise control style
            target_speed = command.get("speed", 0) / 3.6  # km/h to m/s
            if target_speed > self.speed:
                self.throttle = 0.5
                self.brake = 0
            else:
                self.throttle = 0
                self.brake = 0.3
                
        elif cmd_type == "stop":
            self.throttle = 0
            self.brake = 1.0
            print(f"🚗 {self.agent_id}: Emergency stop")
            
        elif cmd_type == "goto":
            # Simple waypoint navigation
            target = np.array(command.get("position", [0, 0, 0]))
            direction = target[:2] - self.state.position[:2]
            distance = np.linalg.norm(direction)
            
            if distance > 1.0:
                # Compute steering to target
                target_heading = np.arctan2(direction[1], direction[0])
                heading_error = target_heading - self.heading
                
                # Normalize to [-pi, pi]
                while heading_error > np.pi:
                    heading_error -= 2 * np.pi
                while heading_error < -np.pi:
                    heading_error += 2 * np.pi
                
                self.steering_angle = np.clip(heading_error, -0.5, 0.5)
                self.throttle = 0.3
                self.brake = 0
            else:
                self.throttle = 0
                self.brake = 0.5
    
    def get_state(self) -> Dict:
        """Get vehicle state"""
        base_state = super().get_state()
        base_state.update({
            "speed": self.speed,
            "speed_kmh": self.speed * 3.6,
            "heading": np.degrees(self.heading),
            "steering_angle": np.degrees(self.steering_angle),
            "throttle": self.throttle,
            "brake": self.brake,
            "obd_data": self.obd_data
        })
        return base_state

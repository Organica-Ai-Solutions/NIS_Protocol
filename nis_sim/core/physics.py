"""
Physics Controller
Validates commands against physics constraints (mirrors NIS Protocol physics)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PhysicsConstraints:
    """Physical limits for validation"""
    max_velocity: float = 20.0       # m/s
    max_acceleration: float = 10.0   # m/s²
    max_angular_velocity: float = 5.0  # rad/s
    max_thrust: float = 50.0         # N
    mass: float = 1.0                # kg


class PhysicsController:
    """
    Physics validation and control
    Ensures commands respect physical laws
    """
    
    def __init__(self, constraints: Optional[PhysicsConstraints] = None):
        self.constraints = constraints or PhysicsConstraints()
    
    def validate_velocity(self, velocity: np.ndarray) -> Tuple[bool, str]:
        """Check if velocity is within limits"""
        speed = np.linalg.norm(velocity)
        if speed > self.constraints.max_velocity:
            return False, f"Velocity {speed:.2f} exceeds max {self.constraints.max_velocity}"
        return True, "OK"
    
    def validate_acceleration(self, acceleration: np.ndarray) -> Tuple[bool, str]:
        """Check if acceleration is within limits"""
        accel_mag = np.linalg.norm(acceleration)
        if accel_mag > self.constraints.max_acceleration:
            return False, f"Acceleration {accel_mag:.2f} exceeds max {self.constraints.max_acceleration}"
        return True, "OK"
    
    def validate_thrust(self, thrust: float) -> Tuple[bool, str]:
        """Check if thrust command is valid"""
        if thrust < 0:
            return False, "Thrust cannot be negative"
        if thrust > self.constraints.max_thrust:
            return False, f"Thrust {thrust:.2f} exceeds max {self.constraints.max_thrust}"
        return True, "OK"
    
    def clamp_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Clamp velocity to max limits"""
        speed = np.linalg.norm(velocity)
        if speed > self.constraints.max_velocity:
            return velocity * (self.constraints.max_velocity / speed)
        return velocity
    
    def compute_required_thrust(self, 
                                 target_acceleration: np.ndarray,
                                 current_velocity: np.ndarray,
                                 gravity: np.ndarray = np.array([0, 0, -9.81])) -> float:
        """Compute thrust needed for desired acceleration"""
        # F = ma + mg (to counteract gravity)
        net_force = self.constraints.mass * (target_acceleration - gravity)
        return np.linalg.norm(net_force)
    
    def check_energy_conservation(self,
                                   mass: float,
                                   velocity_before: np.ndarray,
                                   velocity_after: np.ndarray,
                                   height_before: float,
                                   height_after: float,
                                   work_done: float = 0.0,
                                   tolerance: float = 0.01) -> Tuple[bool, float]:
        """
        Verify energy conservation
        Returns (is_valid, energy_error)
        """
        g = 9.81
        
        # Kinetic energy
        KE_before = 0.5 * mass * np.dot(velocity_before, velocity_before)
        KE_after = 0.5 * mass * np.dot(velocity_after, velocity_after)
        
        # Potential energy
        PE_before = mass * g * height_before
        PE_after = mass * g * height_after
        
        # Total energy
        E_before = KE_before + PE_before
        E_after = KE_after + PE_after
        
        # Energy should be conserved (minus work done by external forces)
        energy_error = abs((E_after - E_before) - work_done) / max(E_before, 1.0)
        
        return energy_error < tolerance, energy_error
    
    def check_momentum_conservation(self,
                                     masses: list,
                                     velocities_before: list,
                                     velocities_after: list,
                                     tolerance: float = 0.01) -> Tuple[bool, float]:
        """
        Verify momentum conservation in collision
        Returns (is_valid, momentum_error)
        """
        p_before = sum(m * v for m, v in zip(masses, velocities_before))
        p_after = sum(m * v for m, v in zip(masses, velocities_after))
        
        momentum_error = np.linalg.norm(p_after - p_before) / max(np.linalg.norm(p_before), 1.0)
        
        return momentum_error < tolerance, momentum_error


class DronePhysics(PhysicsController):
    """Drone-specific physics"""
    
    def __init__(self):
        super().__init__(PhysicsConstraints(
            max_velocity=15.0,       # m/s
            max_acceleration=8.0,    # m/s²
            max_angular_velocity=3.0,  # rad/s
            max_thrust=30.0,         # N (for ~2kg drone)
            mass=2.0                 # kg
        ))
        self.rotor_count = 4
        self.arm_length = 0.25  # m
    
    def compute_rotor_speeds(self, 
                              thrust: float, 
                              torque: np.ndarray) -> np.ndarray:
        """
        Compute individual rotor speeds for desired thrust and torque
        Simplified quadrotor model
        """
        # Thrust coefficient (simplified)
        k_t = self.constraints.max_thrust / (4 * 1000**2)  # thrust per (rad/s)²
        
        # Base speed for hover
        base_speed = np.sqrt(thrust / (4 * k_t))
        
        # Differential for torque (simplified)
        speeds = np.array([
            base_speed + torque[0] - torque[1],  # Front-right
            base_speed - torque[0] - torque[1],  # Front-left
            base_speed - torque[0] + torque[1],  # Back-left
            base_speed + torque[0] + torque[1],  # Back-right
        ])
        
        return np.clip(speeds, 0, 1500)  # RPM limits


class VehiclePhysics(PhysicsController):
    """Ground vehicle physics"""
    
    def __init__(self):
        super().__init__(PhysicsConstraints(
            max_velocity=50.0,       # m/s (~180 km/h)
            max_acceleration=5.0,    # m/s²
            max_angular_velocity=1.0,  # rad/s
            max_thrust=5000.0,       # N
            mass=1500.0              # kg
        ))
        self.wheelbase = 2.5  # m
        self.max_steering_angle = np.radians(35)
    
    def bicycle_model(self,
                      velocity: float,
                      steering_angle: float,
                      dt: float) -> Tuple[float, float, float]:
        """
        Simple bicycle model for vehicle dynamics
        Returns (dx, dy, dtheta)
        """
        steering_angle = np.clip(steering_angle, 
                                  -self.max_steering_angle, 
                                  self.max_steering_angle)
        
        if abs(steering_angle) < 0.001:
            # Straight line
            dx = velocity * dt
            dy = 0
            dtheta = 0
        else:
            # Curved path
            radius = self.wheelbase / np.tan(steering_angle)
            dtheta = velocity * dt / radius
            dx = radius * np.sin(dtheta)
            dy = radius * (1 - np.cos(dtheta))
        
        return dx, dy, dtheta

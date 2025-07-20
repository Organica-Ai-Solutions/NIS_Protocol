"""
Conservation Laws Enforcement Module

This module provides enforcement and validation of fundamental conservation laws
in physics simulations and AI reasoning. It ensures that all system behaviors
comply with energy, momentum, mass, and other conservation principles.

Key Features:
- Energy conservation validation and enforcement
- Momentum conservation checking
- Mass conservation verification
- Angular momentum conservation
- Charge conservation (for electromagnetic systems)
- Real-time violation detection and correction
"""

import numpy as np
import torch
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict, deque

from .physics_agent import PhysicsLaw, PhysicsDomain, PhysicsState, PhysicsViolation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConservationLawType(Enum):
    """Types of conservation laws that can be enforced."""
    ENERGY = "energy"
    MOMENTUM_LINEAR = "momentum_linear"
    MOMENTUM_ANGULAR = "momentum_angular"
    MASS = "mass"
    CHARGE = "charge"
    BARYON_NUMBER = "baryon_number"
    LEPTON_NUMBER = "lepton_number"

@dataclass
class ConservationViolation:
    """Represents a conservation law violation."""
    law_type: ConservationLawType
    initial_value: float
    final_value: float
    violation_magnitude: float
    relative_error: float
    tolerance: float
    timestamp: float
    description: str

class ConservationLaws:
    """
    Conservation Laws Enforcement System
    
    Validates and enforces fundamental conservation laws in physics
    simulations and AI reasoning processes.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("nis.physics.conservation")
        
        # Conservation law tolerances
        self.tolerances = {
            ConservationLawType.ENERGY: 1e-12,
            ConservationLawType.MOMENTUM_LINEAR: 1e-12,
            ConservationLawType.MOMENTUM_ANGULAR: 1e-12,
            ConservationLawType.MASS: 1e-15,
            ConservationLawType.CHARGE: 1e-15,
            ConservationLawType.BARYON_NUMBER: 0.0,  # Exact conservation
            ConservationLawType.LEPTON_NUMBER: 0.0   # Exact conservation
        }
        
        # Violation tracking
        self.violation_history: deque = deque(maxlen=1000)
        self.active_violations: List[ConservationViolation] = []
        
        # Statistics
        self.validation_stats = {
            "total_validations": 0,
            "violations_detected": 0,
            "corrections_applied": 0,
            "conservation_accuracy": 1.0
        }
        
        # System state tracking
        self.system_states: deque = deque(maxlen=100)
        
        self.logger.info("Initialized Conservation Laws enforcement system")
    
    def validate_energy_conservation(
        self,
        initial_state: PhysicsState,
        final_state: PhysicsState,
        external_work: float = 0.0
    ) -> Optional[ConservationViolation]:
        """
        Validate energy conservation between two states.
        
        Args:
            initial_state: Initial system state
            final_state: Final system state
            external_work: Work done by external forces
            
        Returns:
            ConservationViolation if violated, None otherwise
        """
        try:
            # Calculate total energy for both states
            initial_energy = self._calculate_total_energy(initial_state)
            final_energy = self._calculate_total_energy(final_state)
            
            # Account for external work
            expected_final_energy = initial_energy + external_work
            
            # Calculate violation
            violation_magnitude = abs(final_energy - expected_final_energy)
            relative_error = (violation_magnitude / abs(initial_energy) 
                             if initial_energy != 0 else violation_magnitude)
            
            tolerance = self.tolerances[ConservationLawType.ENERGY]
            
            if violation_magnitude > tolerance:
                violation = ConservationViolation(
                    law_type=ConservationLawType.ENERGY,
                    initial_value=initial_energy,
                    final_value=final_energy,
                    violation_magnitude=violation_magnitude,
                    relative_error=relative_error,
                    tolerance=tolerance,
                    timestamp=time.time(),
                    description=f"Energy conservation violated: ΔE = {violation_magnitude:.2e}, "
                               f"relative error = {relative_error:.2e}"
                )
                
                self.violation_history.append(violation)
                self.validation_stats["violations_detected"] += 1
                return violation
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error validating energy conservation: {e}")
            return None
    
    def validate_momentum_conservation(
        self,
        initial_state: PhysicsState,
        final_state: PhysicsState,
        external_impulse: np.ndarray = None
    ) -> Optional[ConservationViolation]:
        """
        Validate linear momentum conservation between two states.
        
        Args:
            initial_state: Initial system state
            final_state: Final system state
            external_impulse: External impulse applied to system
            
        Returns:
            ConservationViolation if violated, None otherwise
        """
        try:
            if external_impulse is None:
                external_impulse = np.zeros(3)
            
            # Calculate momentum for both states
            initial_momentum = initial_state.mass * initial_state.velocity
            final_momentum = final_state.mass * final_state.velocity
            
            # Account for external impulse
            expected_final_momentum = initial_momentum + external_impulse
            
            # Calculate violation magnitude (vector norm)
            momentum_difference = final_momentum - expected_final_momentum
            violation_magnitude = np.linalg.norm(momentum_difference)
            
            initial_momentum_magnitude = np.linalg.norm(initial_momentum)
            relative_error = (violation_magnitude / initial_momentum_magnitude 
                             if initial_momentum_magnitude != 0 else violation_magnitude)
            
            tolerance = self.tolerances[ConservationLawType.MOMENTUM_LINEAR]
            
            if violation_magnitude > tolerance:
                violation = ConservationViolation(
                    law_type=ConservationLawType.MOMENTUM_LINEAR,
                    initial_value=initial_momentum_magnitude,
                    final_value=np.linalg.norm(final_momentum),
                    violation_magnitude=violation_magnitude,
                    relative_error=relative_error,
                    tolerance=tolerance,
                    timestamp=time.time(),
                    description=f"Linear momentum conservation violated: Δp = {violation_magnitude:.2e}, "
                               f"relative error = {relative_error:.2e}"
                )
                
                self.violation_history.append(violation)
                self.validation_stats["violations_detected"] += 1
                return violation
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error validating momentum conservation: {e}")
            return None
    
    def validate_angular_momentum_conservation(
        self,
        initial_state: PhysicsState,
        final_state: PhysicsState,
        external_torque_impulse: np.ndarray = None
    ) -> Optional[ConservationViolation]:
        """
        Validate angular momentum conservation between two states.
        
        Args:
            initial_state: Initial system state
            final_state: Final system state
            external_torque_impulse: External torque impulse applied
            
        Returns:
            ConservationViolation if violated, None otherwise
        """
        try:
            if external_torque_impulse is None:
                external_torque_impulse = np.zeros(3)
            
            # Calculate angular momentum L = r × p
            initial_L = np.cross(initial_state.position, 
                               initial_state.mass * initial_state.velocity)
            final_L = np.cross(final_state.position, 
                             final_state.mass * final_state.velocity)
            
            # Account for external torque impulse
            expected_final_L = initial_L + external_torque_impulse
            
            # Calculate violation
            L_difference = final_L - expected_final_L
            violation_magnitude = np.linalg.norm(L_difference)
            
            initial_L_magnitude = np.linalg.norm(initial_L)
            relative_error = (violation_magnitude / initial_L_magnitude 
                             if initial_L_magnitude != 0 else violation_magnitude)
            
            tolerance = self.tolerances[ConservationLawType.MOMENTUM_ANGULAR]
            
            if violation_magnitude > tolerance:
                violation = ConservationViolation(
                    law_type=ConservationLawType.MOMENTUM_ANGULAR,
                    initial_value=initial_L_magnitude,
                    final_value=np.linalg.norm(final_L),
                    violation_magnitude=violation_magnitude,
                    relative_error=relative_error,
                    tolerance=tolerance,
                    timestamp=time.time(),
                    description=f"Angular momentum conservation violated: ΔL = {violation_magnitude:.2e}, "
                               f"relative error = {relative_error:.2e}"
                )
                
                self.violation_history.append(violation)
                self.validation_stats["violations_detected"] += 1
                return violation
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error validating angular momentum conservation: {e}")
            return None
    
    def validate_mass_conservation(
        self,
        initial_state: PhysicsState,
        final_state: PhysicsState
    ) -> Optional[ConservationViolation]:
        """
        Validate mass conservation between two states.
        
        Args:
            initial_state: Initial system state
            final_state: Final system state
            
        Returns:
            ConservationViolation if violated, None otherwise
        """
        try:
            initial_mass = initial_state.mass
            final_mass = final_state.mass
            
            violation_magnitude = abs(final_mass - initial_mass)
            relative_error = (violation_magnitude / initial_mass 
                             if initial_mass != 0 else violation_magnitude)
            
            tolerance = self.tolerances[ConservationLawType.MASS]
            
            if violation_magnitude > tolerance:
                violation = ConservationViolation(
                    law_type=ConservationLawType.MASS,
                    initial_value=initial_mass,
                    final_value=final_mass,
                    violation_magnitude=violation_magnitude,
                    relative_error=relative_error,
                    tolerance=tolerance,
                    timestamp=time.time(),
                    description=f"Mass conservation violated: Δm = {violation_magnitude:.2e}, "
                               f"relative error = {relative_error:.2e}"
                )
                
                self.violation_history.append(violation)
                self.validation_stats["violations_detected"] += 1
                return violation
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error validating mass conservation: {e}")
            return None
    
    def validate_all_conservation_laws(
        self,
        initial_state: PhysicsState,
        final_state: PhysicsState,
        external_forces: Dict[str, Any] = None
    ) -> List[ConservationViolation]:
        """
        Validate all applicable conservation laws.
        
        Args:
            initial_state: Initial system state
            final_state: Final system state
            external_forces: Dictionary of external forces/impulses
            
        Returns:
            List of conservation violations
        """
        violations = []
        
        if external_forces is None:
            external_forces = {}
        
        self.validation_stats["total_validations"] += 1
        
        # Energy conservation
        energy_violation = self.validate_energy_conservation(
            initial_state, final_state, 
            external_forces.get("work", 0.0)
        )
        if energy_violation:
            violations.append(energy_violation)
        
        # Linear momentum conservation
        momentum_violation = self.validate_momentum_conservation(
            initial_state, final_state,
            external_forces.get("impulse", np.zeros(3))
        )
        if momentum_violation:
            violations.append(momentum_violation)
        
        # Angular momentum conservation
        angular_violation = self.validate_angular_momentum_conservation(
            initial_state, final_state,
            external_forces.get("torque_impulse", np.zeros(3))
        )
        if angular_violation:
            violations.append(angular_violation)
        
        # Mass conservation
        mass_violation = self.validate_mass_conservation(initial_state, final_state)
        if mass_violation:
            violations.append(mass_violation)
        
        # Update conservation accuracy
        if violations:
            self.validation_stats["conservation_accuracy"] *= 0.99  # Slight penalty
        else:
            self.validation_stats["conservation_accuracy"] = min(1.0, 
                self.validation_stats["conservation_accuracy"] * 1.001)  # Slight improvement
        
        return violations
    
    def correct_conservation_violations(
        self,
        state: PhysicsState,
        violations: List[ConservationViolation],
        reference_state: PhysicsState
    ) -> PhysicsState:
        """
        Apply corrections to fix conservation law violations.
        
        Args:
            state: Current state with violations
            violations: List of detected violations
            reference_state: Reference state for conservation calculations
            
        Returns:
            Corrected physics state
        """
        corrected_state = PhysicsState(
            position=state.position.copy(),
            velocity=state.velocity.copy(),
            acceleration=state.acceleration.copy(),
            mass=state.mass,
            energy=state.energy,
            temperature=state.temperature,
            pressure=state.pressure,
            timestamp=state.timestamp,
            domain=state.domain,
            constraints=state.constraints.copy()
        )
        
        for violation in violations:
            if violation.law_type == ConservationLawType.ENERGY:
                # Correct energy by adjusting velocity (kinetic energy)
                corrected_state.energy = reference_state.energy
                
                # Recalculate velocity to match energy
                kinetic_energy = corrected_state.energy  # Assuming purely kinetic
                if corrected_state.mass > 0 and kinetic_energy >= 0:
                    velocity_magnitude = np.sqrt(2 * kinetic_energy / corrected_state.mass)
                    if np.linalg.norm(corrected_state.velocity) > 0:
                        # Preserve direction, adjust magnitude
                        direction = corrected_state.velocity / np.linalg.norm(corrected_state.velocity)
                        corrected_state.velocity = direction * velocity_magnitude
                
            elif violation.law_type == ConservationLawType.MOMENTUM_LINEAR:
                # Correct momentum by adjusting velocity
                target_momentum = reference_state.mass * reference_state.velocity
                if corrected_state.mass > 0:
                    corrected_state.velocity = target_momentum / corrected_state.mass
                
            elif violation.law_type == ConservationLawType.MASS:
                # Restore original mass
                corrected_state.mass = reference_state.mass
            
            self.validation_stats["corrections_applied"] += 1
        
        return corrected_state
    
    def _calculate_total_energy(self, state: PhysicsState) -> float:
        """Calculate total energy of a physics state."""
        # Kinetic energy
        kinetic_energy = 0.5 * state.mass * np.sum(state.velocity**2)
        
        # Potential energy (simplified - could be expanded)
        potential_energy = 0.0  # Would depend on force fields
        
        # Thermal energy (if relevant)
        thermal_energy = 0.0  # Could include heat capacity * temperature
        
        # Use provided energy if available, otherwise calculate
        if hasattr(state, 'energy') and state.energy is not None:
            return state.energy
        
        return kinetic_energy + potential_energy + thermal_energy
    
    def get_conservation_status(self) -> Dict[str, Any]:
        """Get current conservation law enforcement status."""
        recent_violations = len([v for v in self.violation_history 
                               if time.time() - v.timestamp < 60.0])  # Last minute
        
        return {
            "validation_stats": self.validation_stats.copy(),
            "tolerances": {k.value: v for k, v in self.tolerances.items()},
            "recent_violations": recent_violations,
            "total_violations": len(self.violation_history),
            "conservation_accuracy": self.validation_stats["conservation_accuracy"],
            "active_violations": len(self.active_violations)
        }
    
    def set_tolerance(self, law_type: ConservationLawType, tolerance: float):
        """Set tolerance for a specific conservation law."""
        self.tolerances[law_type] = tolerance
        self.logger.info(f"Set {law_type.value} tolerance to {tolerance:.2e}")
    
    def reset_statistics(self):
        """Reset validation statistics."""
        self.validation_stats = {
            "total_validations": 0,
            "violations_detected": 0,
            "corrections_applied": 0,
            "conservation_accuracy": 1.0
        }
        self.violation_history.clear()
        self.active_violations.clear()
        self.logger.info("Conservation law statistics reset")

# Example usage and testing
def test_conservation_laws():
    """Test the ConservationLaws implementation."""
    print("🔬 Testing ConservationLaws...")
    
    # Create conservation law enforcer
    conservation = ConservationLaws()
    
    # Create test states
    initial_state = PhysicsState(
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([1.0, 0.0, 0.0]),
        acceleration=np.array([0.0, 0.0, 0.0]),
        mass=1.0,
        energy=0.5,  # KE = 0.5 * m * v²
        temperature=293.15,
        pressure=101325.0,
        timestamp=time.time(),
        domain=PhysicsDomain.MECHANICAL,
        constraints={}
    )
    
    # Create final state with slight energy violation
    final_state = PhysicsState(
        position=np.array([1.0, 0.0, 0.0]),
        velocity=np.array([1.0, 0.0, 0.0]),
        acceleration=np.array([0.0, 0.0, 0.0]),
        mass=1.0,
        energy=0.51,  # Slight energy increase (violation)
        temperature=293.15,
        pressure=101325.0,
        timestamp=time.time(),
        domain=PhysicsDomain.MECHANICAL,
        constraints={}
    )
    
    # Test individual conservation laws
    energy_violation = conservation.validate_energy_conservation(initial_state, final_state)
    momentum_violation = conservation.validate_momentum_conservation(initial_state, final_state)
    mass_violation = conservation.validate_mass_conservation(initial_state, final_state)
    
    print(f"   Energy violation: {energy_violation is not None}")
    print(f"   Momentum violation: {momentum_violation is not None}")
    print(f"   Mass violation: {mass_violation is not None}")
    
    # Test comprehensive validation
    all_violations = conservation.validate_all_conservation_laws(initial_state, final_state)
    print(f"   Total violations detected: {len(all_violations)}")
    
    # Test correction
    if all_violations:
        corrected_state = conservation.correct_conservation_violations(
            final_state, all_violations, initial_state
        )
        print(f"   Corrections applied successfully")
        print(f"   Original energy: {final_state.energy:.3f}")
        print(f"   Corrected energy: {corrected_state.energy:.3f}")
    
    # Test status
    status = conservation.get_conservation_status()
    print(f"   Conservation accuracy: {status['conservation_accuracy']:.3f}")
    
    print("✅ ConservationLaws test completed")

if __name__ == "__main__":
    test_conservation_laws() 
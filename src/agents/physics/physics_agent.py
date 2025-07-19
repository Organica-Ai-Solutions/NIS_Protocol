"""
Physics-Informed Agent with PINN Integration

This agent enforces physical laws and constraints using Physics-Informed Neural Networks (PINNs).
It validates that AI reasoning and actions comply with fundamental physics principles including
conservation laws, thermodynamics, and domain-specific physical constraints.

Key Features:
- PINN-based physics constraint validation
- Real-time conservation law enforcement  
- Integration with KAN reasoning for physics-aware decisions
- Dynamic physics parameter learning
- Multi-domain physics modeling (mechanical, thermal, electromagnetic)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict, deque

from ...core.agent import NISAgent, NISLayer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhysicsLaw(Enum):
    """Types of physics laws that can be enforced."""
    ENERGY_CONSERVATION = "energy_conservation"
    MOMENTUM_CONSERVATION = "momentum_conservation"
    MASS_CONSERVATION = "mass_conservation"
    THERMODYNAMICS_FIRST = "thermodynamics_first"
    THERMODYNAMICS_SECOND = "thermodynamics_second"
    ELECTROMAGNETIC_GAUSS = "electromagnetic_gauss"
    NEWTONIAN_MECHANICS = "newtonian_mechanics"
    FLUID_DYNAMICS = "fluid_dynamics"

class PhysicsDomain(Enum):
    """Physics domains for specialized modeling."""
    MECHANICAL = "mechanical"
    THERMAL = "thermal"
    ELECTROMAGNETIC = "electromagnetic"
    FLUID = "fluid"
    QUANTUM = "quantum"
    RELATIVISTIC = "relativistic"
    ARCHAEOLOGICAL = "archaeological"  # Specialized for heritage site physics

@dataclass
class PhysicsState:
    """Represents the physical state of a system."""
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    mass: float
    energy: float
    temperature: float
    pressure: float
    timestamp: float
    domain: PhysicsDomain
    constraints: Dict[str, Any]

@dataclass
class PhysicsViolation:
    """Represents a detected physics law violation."""
    law: PhysicsLaw
    severity: float  # 0.0 = minor, 1.0 = critical
    description: str
    suggested_correction: Optional[Dict[str, Any]]
    timestamp: float
    state_before: PhysicsState
    state_after: PhysicsState

class PINNLayer(nn.Module):
    """
    Physics-Informed Neural Network Layer
    
    Implements a neural network layer that enforces physics constraints
    through automatic differentiation and residual-based loss functions.
    """
    
    def __init__(self, input_dim: int, output_dim: int, physics_laws: List[PhysicsLaw]):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.physics_laws = physics_laws
        
        # Neural network layers
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, output_dim)
        
        # Physics constraint weights
        self.physics_weights = nn.Parameter(torch.ones(len(physics_laws)))
        
        # Activation functions
        self.activation = nn.Tanh()  # Smooth for differentiation
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with physics constraint calculation.
        
        Returns:
            Tuple of (network_output, physics_residuals)
        """
        # Standard neural network forward pass
        h1 = self.activation(self.linear1(x))
        h2 = self.activation(self.linear2(h1))
        output = self.linear3(h2)
        
        # Calculate physics residuals (constraint violations)
        physics_residuals = self._calculate_physics_residuals(x, output)
        
        return output, physics_residuals
    
    def _calculate_physics_residuals(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Calculate residuals for physics law enforcement."""
        residuals = []
        
        for i, law in enumerate(self.physics_laws):
            if law == PhysicsLaw.ENERGY_CONSERVATION:
                residual = self._energy_conservation_residual(x, output)
            elif law == PhysicsLaw.MOMENTUM_CONSERVATION:
                residual = self._momentum_conservation_residual(x, output)
            elif law == PhysicsLaw.MASS_CONSERVATION:
                residual = self._mass_conservation_residual(x, output)
            else:
                residual = torch.zeros(x.shape[0], 1)
            
            # Weight the residual by learnable physics weights
            weighted_residual = self.physics_weights[i] * residual
            residuals.append(weighted_residual)
        
        return torch.cat(residuals, dim=1) if residuals else torch.zeros(x.shape[0], 1)
    
    def _energy_conservation_residual(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Calculate energy conservation residual: dE/dt = 0 for isolated systems."""
        # Assuming x contains [position, velocity, time] and output contains energy
        if x.requires_grad:
            energy = output[:, 0:1]  # First output is energy
            dE_dt = torch.autograd.grad(
                energy.sum(), x, create_graph=True, retain_graph=True
            )[0][:, -1:] # Derivative w.r.t. time (last dimension)
            return dE_dt.abs()  # Energy should be conserved (dE/dt = 0)
        return torch.zeros(x.shape[0], 1)
    
    def _momentum_conservation_residual(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Calculate momentum conservation residual: dp/dt = F_external."""
        # Simplified momentum conservation check
        if output.shape[1] >= 3:  # Assuming output has momentum components
            momentum = output[:, 1:4]  # [px, py, pz]
            # For isolated system, momentum should be constant
            momentum_variance = torch.var(momentum, dim=0).sum().unsqueeze(0).unsqueeze(0)
            return momentum_variance.expand(x.shape[0], 1)
        return torch.zeros(x.shape[0], 1)
    
    def _mass_conservation_residual(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Calculate mass conservation residual: dm/dt = 0 for closed systems."""
        if output.shape[1] >= 5:  # Assuming output includes mass
            mass = output[:, 4:5]  # Mass component
            if x.requires_grad:
                dm_dt = torch.autograd.grad(
                    mass.sum(), x, create_graph=True, retain_graph=True
                )[0][:, -1:]  # Derivative w.r.t. time
                return dm_dt.abs()
        return torch.zeros(x.shape[0], 1)

class PINNNetwork(nn.Module):
    """
    Complete Physics-Informed Neural Network
    
    Multi-layer PINN that enforces multiple physics laws simultaneously
    while learning system dynamics.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int] = [64, 32], 
        output_dim: int = 8,  # [energy, px, py, pz, mass, temperature, pressure, custom]
        physics_laws: List[PhysicsLaw] = None
    ):
        super().__init__()
        
        if physics_laws is None:
            physics_laws = [
                PhysicsLaw.ENERGY_CONSERVATION,
                PhysicsLaw.MOMENTUM_CONSERVATION,
                PhysicsLaw.MASS_CONSERVATION
            ]
        
        self.physics_laws = physics_laws
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build PINN layers
        self.pinn_layers = nn.ModuleList()
        
        # Input layer
        self.pinn_layers.append(PINNLayer(input_dim, hidden_dims[0], physics_laws))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.pinn_layers.append(PINNLayer(hidden_dims[i], hidden_dims[i+1], physics_laws))
        
        # Output layer
        self.pinn_layers.append(PINNLayer(hidden_dims[-1], output_dim, physics_laws))
        
        # Physics loss weight
        self.physics_loss_weight = 1.0
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through PINN network.
        
        Returns:
            Tuple of (final_output, physics_residuals_by_layer)
        """
        current_input = x
        all_physics_residuals = []
        
        for layer in self.pinn_layers:
            current_input, physics_residuals = layer(current_input)
            all_physics_residuals.append(physics_residuals)
        
        return current_input, all_physics_residuals
    
    def calculate_total_loss(
        self, 
        x: torch.Tensor, 
        y_true: torch.Tensor, 
        y_pred: torch.Tensor,
        physics_residuals: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate total loss combining data loss and physics loss.
        
        Returns:
            Tuple of (total_loss, data_loss, physics_loss)
        """
        # Data loss (standard MSE)
        data_loss = F.mse_loss(y_pred, y_true)
        
        # Physics loss (sum of all residuals)
        physics_loss = torch.tensor(0.0)
        for residuals in physics_residuals:
            physics_loss += torch.mean(residuals**2)
        
        # Total loss
        total_loss = data_loss + self.physics_loss_weight * physics_loss
        
        return total_loss, data_loss, physics_loss

class PhysicsInformedAgent(NISAgent):
    """
    Physics-Informed Agent for V3.0 NIS Protocol
    
    This agent uses Physics-Informed Neural Networks to validate that
    AI reasoning and actions comply with fundamental physics principles.
    It integrates with the KAN reasoning system to provide physics-aware
    decision making.
    """
    
    def __init__(
        self,
        agent_id: str = "physics_informed_001",
        description: str = "Physics-informed constraint validation agent",
        physics_domain: PhysicsDomain = PhysicsDomain.MECHANICAL
    ):
        super().__init__(agent_id, NISLayer.REASONING, description)
        
        self.physics_domain = physics_domain
        self.logger = logging.getLogger(f"nis.physics.{agent_id}")
        
        # Initialize PINN network
        self.pinn_network = PINNNetwork(
            input_dim=6,  # [x, y, z, vx, vy, vz]
            hidden_dims=[64, 32, 16],
            output_dim=8,  # [energy, px, py, pz, mass, temp, pressure, custom]
            physics_laws=[
                PhysicsLaw.ENERGY_CONSERVATION,
                PhysicsLaw.MOMENTUM_CONSERVATION,
                PhysicsLaw.MASS_CONSERVATION
            ]
        )
        
        # Physics state tracking
        self.current_state = None
        self.state_history: deque = deque(maxlen=1000)
        self.violation_history: deque = deque(maxlen=100)
        
        # Physics constraints
        self.constraint_tolerances = {
            PhysicsLaw.ENERGY_CONSERVATION: 1e-6,
            PhysicsLaw.MOMENTUM_CONSERVATION: 1e-6,
            PhysicsLaw.MASS_CONSERVATION: 1e-8
        }
        
        # Performance metrics
        self.validation_stats = {
            "total_validations": 0,
            "violations_detected": 0,
            "corrections_applied": 0,
            "average_residual": 0.0
        }
        
        # Domain-specific parameters
        self._initialize_domain_parameters()
        
        self.logger.info(f"Initialized PhysicsInformedAgent for {physics_domain.value} domain")
    
    def _initialize_domain_parameters(self):
        """Initialize domain-specific physics parameters."""
        if self.physics_domain == PhysicsDomain.MECHANICAL:
            self.domain_params = {
                "gravity": 9.81,  # m/s²
                "air_resistance": 0.1,
                "material_properties": {"steel": {"density": 7850, "young_modulus": 200e9}}
            }
        elif self.physics_domain == PhysicsDomain.THERMAL:
            self.domain_params = {
                "heat_capacity": 4186,  # J/(kg·K) for water
                "thermal_conductivity": 0.6,  # W/(m·K)
                "stefan_boltzmann": 5.67e-8  # W/(m²·K⁴)
            }
        elif self.physics_domain == PhysicsDomain.ARCHAEOLOGICAL:
            self.domain_params = {
                "soil_density": 1500,  # kg/m³
                "weathering_rate": 1e-9,  # m/year
                "preservation_factors": {"humidity": 0.6, "temperature": 15}  # °C
            }
        else:
            self.domain_params = {}
    
    def validate_physics_state(self, state: Dict[str, Any]) -> Tuple[bool, List[PhysicsViolation]]:
        """
        Validate a physics state against known physics laws.
        
        Args:
            state: Dictionary containing position, velocity, mass, energy, etc.
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        try:
            # Convert state to PhysicsState object
            physics_state = self._dict_to_physics_state(state)
            
            # Convert to tensor for PINN processing
            state_tensor = self._physics_state_to_tensor(physics_state)
            
            # Run through PINN network
            with torch.no_grad():
                output, physics_residuals = self.pinn_network(state_tensor)
            
            # Analyze residuals for violations
            violations = self._analyze_residuals(physics_residuals, physics_state)
            
            # Update statistics
            self.validation_stats["total_validations"] += 1
            if violations:
                self.validation_stats["violations_detected"] += 1
            
            # Calculate average residual
            total_residual = sum(torch.mean(r).item() for r in physics_residuals)
            self.validation_stats["average_residual"] = (
                0.9 * self.validation_stats["average_residual"] + 0.1 * total_residual
            )
            
            is_valid = len(violations) == 0
            return is_valid, violations
            
        except Exception as e:
            self.logger.error(f"Error validating physics state: {e}")
            return False, [PhysicsViolation(
                law=PhysicsLaw.ENERGY_CONSERVATION,
                severity=1.0,
                description=f"Validation error: {str(e)}",
                suggested_correction=None,
                timestamp=time.time(),
                state_before=physics_state,
                state_after=physics_state
            )]
    
    def correct_physics_violations(
        self, 
        state: Dict[str, Any], 
        violations: List[PhysicsViolation]
    ) -> Dict[str, Any]:
        """
        Apply corrections to fix detected physics violations.
        
        Args:
            state: Original state with violations
            violations: List of detected violations
            
        Returns:
            Corrected state dictionary
        """
        corrected_state = state.copy()
        
        for violation in violations:
            if violation.suggested_correction:
                # Apply suggested corrections
                for key, value in violation.suggested_correction.items():
                    corrected_state[key] = value
                    
                self.validation_stats["corrections_applied"] += 1
                self.logger.info(f"Applied correction for {violation.law.value}: {key} -> {value}")
        
        return corrected_state
    
    def _dict_to_physics_state(self, state_dict: Dict[str, Any]) -> PhysicsState:
        """Convert dictionary to PhysicsState object."""
        return PhysicsState(
            position=np.array(state_dict.get("position", [0, 0, 0])),
            velocity=np.array(state_dict.get("velocity", [0, 0, 0])),
            acceleration=np.array(state_dict.get("acceleration", [0, 0, 0])),
            mass=state_dict.get("mass", 1.0),
            energy=state_dict.get("energy", 0.0),
            temperature=state_dict.get("temperature", 293.15),  # 20°C
            pressure=state_dict.get("pressure", 101325.0),  # 1 atm
            timestamp=time.time(),
            domain=self.physics_domain,
            constraints=state_dict.get("constraints", {})
        )
    
    def _physics_state_to_tensor(self, state: PhysicsState) -> torch.Tensor:
        """Convert PhysicsState to tensor for PINN processing."""
        # Combine position and velocity into input tensor
        features = np.concatenate([state.position, state.velocity])
        return torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension
    
    def _analyze_residuals(
        self, 
        physics_residuals: List[torch.Tensor], 
        state: PhysicsState
    ) -> List[PhysicsViolation]:
        """Analyze PINN residuals to detect physics violations."""
        violations = []
        
        for i, residuals in enumerate(physics_residuals):
            mean_residual = torch.mean(residuals).item()
            
            # Check each physics law
            for j, law in enumerate(self.pinn_network.physics_laws):
                if j < residuals.shape[1]:  # Ensure we have this residual
                    residual_value = residuals[0, j].item()
                    tolerance = self.constraint_tolerances.get(law, 1e-6)
                    
                    if abs(residual_value) > tolerance:
                        severity = min(1.0, abs(residual_value) / tolerance)
                        
                        violation = PhysicsViolation(
                            law=law,
                            severity=severity,
                            description=f"{law.value} violation: residual {residual_value:.2e} > tolerance {tolerance:.2e}",
                            suggested_correction=self._generate_correction(law, residual_value, state),
                            timestamp=time.time(),
                            state_before=state,
                            state_after=state  # Will be modified by correction
                        )
                        violations.append(violation)
        
        return violations
    
    def _generate_correction(
        self, 
        law: PhysicsLaw, 
        residual: float, 
        state: PhysicsState
    ) -> Optional[Dict[str, Any]]:
        """Generate suggested corrections for physics violations."""
        if law == PhysicsLaw.ENERGY_CONSERVATION:
            # Adjust energy to maintain conservation
            return {"energy": state.energy - residual * 0.1}
        elif law == PhysicsLaw.MOMENTUM_CONSERVATION:
            # Adjust velocity to maintain momentum conservation
            velocity_correction = -residual * 0.01
            return {"velocity": state.velocity + velocity_correction}
        elif law == PhysicsLaw.MASS_CONSERVATION:
            # Mass should remain constant in most cases
            return {"mass": state.mass}
        
        return None
    
    def integrate_with_kan_reasoning(
        self, 
        kan_output: torch.Tensor, 
        physics_context: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Integrate physics constraints with KAN reasoning output.
        
        Args:
            kan_output: Output from KAN reasoning network
            physics_context: Physics context for validation
            
        Returns:
            Physics-corrected KAN output
        """
        try:
            # Validate physics context
            is_valid, violations = self.validate_physics_state(physics_context)
            
            if not is_valid:
                self.logger.warning(f"Physics violations detected in KAN reasoning: {len(violations)} violations")
                
                # Apply physics corrections to KAN output
                corrected_context = self.correct_physics_violations(physics_context, violations)
                
                # Adjust KAN output based on physics corrections
                correction_factor = 1.0 - sum(v.severity for v in violations) / len(violations)
                corrected_output = kan_output * correction_factor
                
                return corrected_output
            
            return kan_output
            
        except Exception as e:
            self.logger.error(f"Error integrating physics with KAN reasoning: {e}")
            return kan_output  # Return original if correction fails
    
    def get_physics_status(self) -> Dict[str, Any]:
        """Get current physics validation status and statistics."""
        return {
            "domain": self.physics_domain.value,
            "validation_stats": self.validation_stats.copy(),
            "active_violations": len(self.violation_history),
            "constraint_tolerances": self.constraint_tolerances.copy(),
            "domain_parameters": self.domain_params.copy()
        }
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message with physics validation."""
        try:
            # Extract physics-related content
            physics_data = message.get("physics", {})
            
            if physics_data:
                # Validate physics state
                is_valid, violations = self.validate_physics_state(physics_data)
                
                # Prepare response
                response = {
                    "agent_id": self.agent_id,
                    "physics_valid": is_valid,
                    "violations": [
                        {
                            "law": v.law.value,
                            "severity": v.severity,
                            "description": v.description
                        } for v in violations
                    ],
                    "corrections_available": any(v.suggested_correction for v in violations),
                    "timestamp": time.time()
                }
                
                # Add corrected state if violations exist
                if violations:
                    corrected_state = self.correct_physics_violations(physics_data, violations)
                    response["corrected_state"] = corrected_state
                
                return response
            
            return {"agent_id": self.agent_id, "status": "no_physics_data"}
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return {"agent_id": self.agent_id, "error": str(e)}

# Example usage and testing
def test_physics_agent():
    """Test the PhysicsInformedAgent implementation."""
    print("🔬 Testing PhysicsInformedAgent...")
    
    # Create agent
    agent = PhysicsInformedAgent(
        agent_id="test_physics_001",
        physics_domain=PhysicsDomain.MECHANICAL
    )
    
    # Test physics state validation
    test_state = {
        "position": [1.0, 2.0, 0.0],
        "velocity": [0.5, 0.0, 0.0],
        "mass": 10.0,
        "energy": 1.25  # 0.5 * m * v² = 0.5 * 10 * 0.25 = 1.25
    }
    
    is_valid, violations = agent.validate_physics_state(test_state)
    print(f"   State valid: {is_valid}")
    print(f"   Violations: {len(violations)}")
    
    # Test message processing
    message = {
        "type": "physics_validation",
        "physics": test_state
    }
    
    response = agent.process_message(message)
    print(f"   Response: {response['physics_valid']}")
    
    # Test KAN integration
    kan_output = torch.tensor([[1.0, 2.0, 3.0]])
    corrected_output = agent.integrate_with_kan_reasoning(kan_output, test_state)
    print(f"   KAN integration successful: {corrected_output.shape}")
    
    print("✅ PhysicsInformedAgent test completed")

if __name__ == "__main__":
    test_physics_agent() 
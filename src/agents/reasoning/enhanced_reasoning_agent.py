"""
Enhanced Reasoning Agent with KAN Integration

This module provides KAN (Kolmogorov-Arnold Network) enhanced reasoning capabilities
for the base NIS Protocol. It replaces traditional MLPs with interpretable spline-based
layers that can be used by any specialized application built on top of NIS.

Key Features:
- Universal KAN-based reasoning (domain-agnostic)
- Interpretable spline-based decision making
- Enhanced function approximation capabilities
- Seamless integration with existing NIS Protocol agents
- Cognitive wave field processing for spatial reasoning
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from src.core.agent import NISAgent, NISLayer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReasoningMode(Enum):
    """Different reasoning modes supported by the enhanced agent."""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    CAUSAL = "causal"

@dataclass
class ReasoningResult:
    """Result of enhanced reasoning process."""
    conclusion: Any
    confidence: float
    reasoning_path: List[str]
    interpretability_data: Dict[str, Any]
    cognitive_load: float
    processing_time: float

class UniversalKANLayer(nn.Module):
    """
    Universal KAN layer implementation for the base NIS Protocol.
    
    This layer uses learnable spline functions instead of traditional
    linear transformations, providing better interpretability and
    function approximation for any domain.
    """
    
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Learnable spline coefficients
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size))
        self.spline_scaler = nn.Parameter(torch.ones(out_features, in_features))
        self.base_activation = nn.Parameter(torch.zeros(out_features, in_features))
        
        # Grid points for spline interpolation
        self.register_buffer('grid', torch.linspace(-2, 2, grid_size))
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize spline weights for stable training."""
        nn.init.xavier_uniform_(self.spline_weight)
        nn.init.ones_(self.spline_scaler)
        nn.init.zeros_(self.base_activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through universal KAN layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        batch_size = x.shape[0]
        
        # Normalize input for stable spline interpolation
        x_normalized = torch.tanh(x)
        
        # Expand dimensions for broadcasting
        x_expanded = x_normalized.unsqueeze(1).unsqueeze(-1)  # (batch, 1, in_features, 1)
        grid_expanded = self.grid.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, grid_size)
        
        # Compute spline basis functions (B-spline like)
        distances = torch.abs(x_expanded - grid_expanded)
        basis_functions = torch.exp(-distances * self.spline_scaler.unsqueeze(0).unsqueeze(-1))
        
        # Apply learnable spline transformation
        spline_output = torch.sum(basis_functions * self.spline_weight.unsqueeze(0), dim=-1)
        
        # Add base activation for better expressiveness
        output = spline_output + self.base_activation.unsqueeze(0) * x_normalized.unsqueeze(1)
        
        # Aggregate across input features
        return torch.sum(output, dim=-1)

class CognitiveWaveProcessor:
    """
    Cognitive wave field processor for spatial and temporal reasoning.
    
    Implements wave-like propagation of reasoning signals across
    conceptual spaces, inspired by neural field theories.
    """
    
    def __init__(self, field_size: int = 64, diffusion_rate: float = 0.1, decay_rate: float = 0.05):
        self.field_size = field_size
        self.diffusion_rate = diffusion_rate
        self.decay_rate = decay_rate
        self.cognitive_field = np.zeros((field_size, field_size))
        self.history = []
        
    def propagate_reasoning(self, reasoning_signals: np.ndarray) -> np.ndarray:
        """
        Propagate reasoning signals through cognitive field.
        
        Args:
            reasoning_signals: 2D array of reasoning activations
            
        Returns:
            Updated cognitive field after wave propagation
        """
        # Ensure input matches field size
        if reasoning_signals.shape != self.cognitive_field.shape:
            reasoning_signals = self._resize_signals(reasoning_signals)
        
        # Apply wave equation: ∂φ/∂t = D∇²φ + S - Rφ
        laplacian = self._compute_laplacian(self.cognitive_field)
        
        self.cognitive_field += (
            self.diffusion_rate * laplacian +      # Diffusion term
            reasoning_signals -                     # Source term
            self.decay_rate * self.cognitive_field  # Decay term
        )
        
        # Store history for temporal reasoning
        self.history.append(self.cognitive_field.copy())
        if len(self.history) > 10:  # Keep last 10 states
            self.history.pop(0)
        
        return self.cognitive_field.copy()
    
    def _compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute discrete Laplacian using 5-point stencil."""
        return (
            np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
            4 * field
        )
    
    def _resize_signals(self, signals: np.ndarray) -> np.ndarray:
        """Resize reasoning signals to match cognitive field."""
        from scipy.ndimage import zoom
        zoom_factors = (self.field_size / signals.shape[0], self.field_size / signals.shape[1])
        return zoom(signals, zoom_factors, order=1)
    
    def get_field_statistics(self) -> Dict[str, float]:
        """Get statistical properties of the cognitive field."""
        return {
            "mean_activation": float(np.mean(self.cognitive_field)),
            "max_activation": float(np.max(self.cognitive_field)),
            "field_variance": float(np.var(self.cognitive_field)),
            "coherence": float(1.0 / (1.0 + np.var(self.cognitive_field))),
            "temporal_stability": self._compute_temporal_stability()
        }
    
    def _compute_temporal_stability(self) -> float:
        """Compute temporal stability of the cognitive field."""
        if len(self.history) < 2:
            return 1.0
        
        # Compare current field with previous states
        stability_scores = []
        current_field = self.cognitive_field
        
        for past_field in self.history[-3:]:  # Last 3 states
            correlation = np.corrcoef(current_field.flatten(), past_field.flatten())[0, 1]
            stability_scores.append(max(0, correlation))
        
        return float(np.mean(stability_scores))

class EnhancedReasoningNetwork(nn.Module):
    """
    Enhanced reasoning network using KAN layers.
    
    Provides interpretable reasoning capabilities that can be used
    by any specialized application built on the NIS Protocol.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [32, 16, 8], output_dim: int = 1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build KAN layers
        self.kan_layers = nn.ModuleList()
        
        # Input layer
        self.kan_layers.append(UniversalKANLayer(input_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.kan_layers.append(UniversalKANLayer(hidden_dims[i], hidden_dims[i + 1]))
        
        # Output layer
        self.kan_layers.append(UniversalKANLayer(hidden_dims[-1], output_dim))
        
        # Activation functions
        self.activation = nn.GELU()  # Smooth activation for better spline interaction
        self.output_activation = nn.Tanh()  # Bounded output
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with interpretability tracking.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (output, interpretability_data)
        """
        interpretability_data = {}
        current_input = x
        
        # Process through KAN layers
        for i, kan_layer in enumerate(self.kan_layers[:-1]):
            layer_output = kan_layer(current_input)
            activated_output = self.activation(layer_output)
            
            # Store for interpretability
            interpretability_data[f'kan_layer_{i}_raw'] = layer_output.detach()
            interpretability_data[f'kan_layer_{i}_activated'] = activated_output.detach()
            
            current_input = activated_output
        
        # Final output layer
        final_output = self.kan_layers[-1](current_input)
        output = self.output_activation(final_output)
        
        interpretability_data['final_raw'] = final_output.detach()
        interpretability_data['final_output'] = output.detach()
        
        return output, interpretability_data

class EnhancedReasoningAgent(NISAgent):
    """
    Enhanced Reasoning Agent with KAN Integration.
    
    This agent provides advanced reasoning capabilities using Kolmogorov-Arnold
    Networks for any application built on the NIS Protocol. It replaces traditional
    MLP-based reasoning with interpretable spline-based processing.
    """
    
    def __init__(self, agent_id: str = "enhanced_reasoning_001", description: str = "KAN-enhanced universal reasoning"):
        super().__init__(agent_id, NISLayer.REASONING, description)
        
        # Initialize with flexible architecture
        self.reasoning_network = None
        self.wave_processor = CognitiveWaveProcessor()
        
        # Reasoning configuration
        self.current_mode = ReasoningMode.ANALYTICAL
        self.interpretability_enabled = True
        self.cognitive_load_threshold = 0.8
        
        logger.info(f"Initialized Enhanced Reasoning Agent: {agent_id}")
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process reasoning requests with KAN-enhanced capabilities.
        
        Args:
            message: Input message containing reasoning task
            
        Returns:
            Processed message with reasoning results
        """
        try:
            operation = message.get("operation", "reason")
            payload = message.get("payload", {})
            
            if operation == "reason":
                return self._perform_reasoning(payload)
            elif operation == "configure_network":
                return self._configure_reasoning_network(payload)
            elif operation == "set_mode":
                return self._set_reasoning_mode(payload)
            elif operation == "get_interpretability":
                return self._get_interpretability_data(payload)
            else:
                return self._create_error_response(f"Unknown operation: {operation}")
                
        except Exception as e:
            logger.error(f"Error in enhanced reasoning: {str(e)}")
            return self._create_error_response(str(e))
    
    def _perform_reasoning(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform KAN-enhanced reasoning on input data.
        
        Args:
            payload: Contains input data and reasoning parameters
            
        Returns:
            Reasoning results with interpretability data
        """
        import time
        start_time = time.time()
        
        # Extract input data
        input_data = payload.get("input_data", [])
        reasoning_mode = payload.get("mode", self.current_mode.value)
        
        if not input_data:
            return self._create_error_response("No input data provided")
        
        # Configure network if not already done
        if self.reasoning_network is None:
            input_dim = len(input_data) if isinstance(input_data[0], (int, float)) else len(input_data[0])
            self._auto_configure_network(input_dim)
        
        # Convert to tensor
        if isinstance(input_data[0], (list, tuple)):
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
        else:
            input_tensor = torch.tensor([input_data], dtype=torch.float32)
        
        # Perform reasoning
        with torch.no_grad():
            reasoning_output, interpretability_data = self.reasoning_network(input_tensor)
        
        # Process through cognitive wave field
        reasoning_signals = self._convert_to_spatial_signals(reasoning_output)
        cognitive_field = self.wave_processor.propagate_reasoning(reasoning_signals)
        field_stats = self.wave_processor.get_field_statistics()
        
        # Generate reasoning result
        processing_time = time.time() - start_time
        result = self._generate_reasoning_result(
            reasoning_output, interpretability_data, field_stats, processing_time
        )
        
        return self._create_response("success", {
            "reasoning_result": result.__dict__,
            "cognitive_field_stats": field_stats,
            "interpretability": {k: v.numpy().tolist() for k, v in interpretability_data.items()},
            "processing_mode": reasoning_mode
        })
    
    def _auto_configure_network(self, input_dim: int):
        """Automatically configure reasoning network based on input dimension."""
        # Adaptive hidden layer sizing
        if input_dim <= 5:
            hidden_dims = [16, 8]
        elif input_dim <= 20:
            hidden_dims = [32, 16, 8]
        else:
            hidden_dims = [64, 32, 16, 8]
        
        self.reasoning_network = EnhancedReasoningNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1
        )
        
        logger.info(f"Auto-configured KAN network: {input_dim} -> {hidden_dims} -> 1")
    
    def _convert_to_spatial_signals(self, reasoning_output: torch.Tensor) -> np.ndarray:
        """Convert reasoning output to spatial signals for wave processing."""
        output_np = reasoning_output.numpy()
        
        # Create spatial representation
        field_size = self.wave_processor.field_size
        spatial_signals = np.zeros((field_size, field_size))
        
        # Map reasoning outputs to spatial locations
        for i, value in enumerate(output_np.flatten()):
            x = int((i * 7) % field_size)  # Pseudo-random but deterministic placement
            y = int((i * 11) % field_size)
            spatial_signals[x, y] = float(value)
        
        return spatial_signals
    
    def _generate_reasoning_result(
        self,
        reasoning_output: torch.Tensor,
        interpretability_data: Dict[str, torch.Tensor],
        field_stats: Dict[str, float],
        processing_time: float
    ) -> ReasoningResult:
        """Generate comprehensive reasoning result."""
        
        # Extract conclusion
        conclusion = float(reasoning_output.mean())
        
        # Calculate confidence based on field coherence and output consistency
        output_variance = float(reasoning_output.var())
        confidence = min(0.95, max(0.1, field_stats['coherence'] * (1.0 - output_variance)))
        
        # Generate reasoning path
        reasoning_path = self._trace_reasoning_path(interpretability_data)
        
        # Calculate cognitive load
        cognitive_load = min(1.0, processing_time * 10 + output_variance)
        
        return ReasoningResult(
            conclusion=conclusion,
            confidence=confidence,
            reasoning_path=reasoning_path,
            interpretability_data={
                "spline_activations": "KAN layers provide smooth, interpretable transformations",
                "wave_propagation": f"Cognitive coherence: {field_stats['coherence']:.3f}",
                "temporal_stability": f"Stability: {field_stats['temporal_stability']:.3f}",
                "field_variance": f"Variance: {field_stats['field_variance']:.3f}"
            },
            cognitive_load=cognitive_load,
            processing_time=processing_time
        )
    
    def _trace_reasoning_path(self, interpretability_data: Dict[str, torch.Tensor]) -> List[str]:
        """Trace the reasoning path through KAN layers."""
        path = ["Input received"]
        
        for key in sorted(interpretability_data.keys()):
            if 'kan_layer' in key and 'activated' in key:
                layer_num = key.split('_')[2]
                activation_mean = float(interpretability_data[key].mean())
                path.append(f"KAN Layer {layer_num}: Spline activation {activation_mean:.3f}")
        
        path.append("Final reasoning conclusion reached")
        return path
    
    def _configure_reasoning_network(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Configure the reasoning network architecture."""
        input_dim = payload.get("input_dim", 10)
        hidden_dims = payload.get("hidden_dims", [32, 16, 8])
        output_dim = payload.get("output_dim", 1)
        
        self.reasoning_network = EnhancedReasoningNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        
        return self._create_response("success", {
            "network_configured": True,
            "architecture": f"{input_dim} -> {hidden_dims} -> {output_dim}",
            "total_parameters": sum(p.numel() for p in self.reasoning_network.parameters())
        })
    
    def _set_reasoning_mode(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Set the reasoning mode."""
        mode = payload.get("mode", "analytical")
        
        try:
            self.current_mode = ReasoningMode(mode)
            return self._create_response("success", {
                "mode_set": mode,
                "available_modes": [m.value for m in ReasoningMode]
            })
        except ValueError:
            return self._create_error_response(f"Invalid reasoning mode: {mode}")
    
    def _get_interpretability_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get current interpretability information."""
        field_stats = self.wave_processor.get_field_statistics()
        
        return self._create_response("success", {
            "interpretability_enabled": self.interpretability_enabled,
            "current_mode": self.current_mode.value,
            "cognitive_field_stats": field_stats,
            "network_info": {
                "type": "KAN (Kolmogorov-Arnold Network)",
                "layers": len(self.reasoning_network.kan_layers) if self.reasoning_network else 0,
                "spline_based": True,
                "interpretable": True
            }
        })

# Example usage and testing
if __name__ == "__main__":
    # Create enhanced reasoning agent
    agent = EnhancedReasoningAgent()
    
    # Test with sample data
    test_message = {
        "operation": "reason",
        "payload": {
            "input_data": [0.5, 0.8, 0.3, 0.9, 0.2],  # Sample 5D input
            "mode": "analytical"
        }
    }
    
    result = agent.process(test_message)
    
    if result["status"] == "success":
        reasoning_result = result["payload"]["reasoning_result"]
        print(f"🧠 Enhanced Reasoning Results:")
        print(f"   Conclusion: {reasoning_result['conclusion']:.3f}")
        print(f"   Confidence: {reasoning_result['confidence']:.3f}")
        print(f"   Cognitive Load: {reasoning_result['cognitive_load']:.3f}")
        print(f"   Processing Time: {reasoning_result['processing_time']:.3f}s")
        print(f"   Reasoning Path: {len(reasoning_result['reasoning_path'])} steps")
        print(f"   KAN-Enhanced: ✅ Spline-based interpretable reasoning")
    else:
        print(f"❌ Error: {result['payload']}") 
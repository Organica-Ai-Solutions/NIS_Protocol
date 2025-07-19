"""
KAN-Enhanced Symbolic Reasoning Agent for NIS Protocol V3

This module implements a Kolmogorov-Arnold Network (KAN) based reasoning agent
enhanced with symbolic function extraction capabilities. It serves as the symbolic
layer in the Laplace → KAN → PINN scientific validation pipeline.

Key Features:
- Spline-based function approximation for interpretable reasoning
- Symbolic function extraction from frequency domain patterns
- Integration with Laplace Transform signal processing
- Pattern→Equation translation algorithms
- Physics-informed symbolic validation preparation
- Modular LLM integration support

Architecture Integration:
[Laplace Transform] → [KAN Symbolic Layer] → [Function Extraction] → [PINN Validation]
"""

import numpy as np
import torch
import torch.nn as nn
import sympy as sp
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
import time

from src.core.agent import NISAgent, NISLayer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SymbolicReasoningType(Enum):
    """Types of symbolic reasoning supported."""
    FUNCTION_EXTRACTION = "function_extraction"
    PATTERN_ANALYSIS = "pattern_analysis"
    EQUATION_DISCOVERY = "equation_discovery"
    SYMBOLIC_REGRESSION = "symbolic_regression"
    PHYSICS_MODELING = "physics_modeling"

class KANLayerType(Enum):
    """Types of KAN layers for different reasoning tasks."""
    SIGNAL_PROCESSING = "signal_processing"
    SYMBOLIC_EXTRACTION = "symbolic_extraction"
    PATTERN_RECOGNITION = "pattern_recognition"
    FUNCTION_APPROXIMATION = "function_approximation"

@dataclass
class SymbolicReasoningResult:
    """Result of symbolic reasoning process."""
    symbolic_function: sp.Expr
    confidence: float
    interpretability_score: float
    validation_metrics: Dict[str, float]
    reasoning_type: SymbolicReasoningType
    computational_cost: float
    alternative_functions: List[sp.Expr] = field(default_factory=list)

@dataclass
class FrequencyPatternFeatures:
    """Features extracted from frequency domain for symbolic reasoning."""
    dominant_frequencies: List[float]
    magnitude_peaks: List[float]
    phase_characteristics: Dict[str, float]
    spectral_centroid: float
    bandwidth: float
    energy: float
    pattern_complexity: float

class EnhancedKANLayer(nn.Module):
    """
    Enhanced KAN layer with symbolic extraction capabilities.

    This extends the basic KAN layer with features for symbolic function
    extraction and pattern recognition in the scientific pipeline.
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 5,
                 layer_type: KANLayerType = KANLayerType.FUNCTION_APPROXIMATION):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.layer_type = layer_type

        # Enhanced spline coefficients with symbolic tracking
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size))
        self.spline_scaler = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Grid points for spline interpolation (learnable)
        self.register_parameter('grid', nn.Parameter(torch.linspace(-1, 1, grid_size)))

        # Symbolic extraction parameters
        self.symbolic_threshold = 0.1
        self.pattern_memory = torch.zeros(out_features, in_features)

        # Layer-specific configurations
        if layer_type == KANLayerType.SYMBOLIC_EXTRACTION:
            self.symbolic_weight = nn.Parameter(torch.randn(out_features, grid_size))

    def forward(self, x: torch.Tensor, return_symbolic_info: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Enhanced forward pass with optional symbolic information extraction.

        Args:
            x: Input tensor of shape (batch_size, in_features)
            return_symbolic_info: Whether to return symbolic extraction information

        Returns:
            Output tensor and optionally symbolic information
        """
        batch_size = x.shape[0]

        # Normalize input to [-1, 1] range for spline approximation
        x_normalized = torch.tanh(x)

        # Expand dimensions for broadcasting
        x_expanded = x_normalized.unsqueeze(1).unsqueeze(-1)  # (batch, 1, in_features, 1)
        grid_expanded = self.grid.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, grid_size)

        # Compute basis functions (B-spline like)
        distances = torch.abs(x_expanded - grid_expanded)  # (batch, 1, in_features, grid_size)

        # Enhanced RBF-like interpolation with learnable scaling
        basis_functions = torch.exp(-distances * self.spline_scaler.unsqueeze(0).unsqueeze(-1))

        # Apply spline weights
        spline_output = torch.sum(basis_functions * self.spline_weight.unsqueeze(0), dim=-1)

        # Sum over input features and add bias
        output = torch.sum(spline_output, dim=-1) + self.bias

        if return_symbolic_info:
            # Extract symbolic information
            symbolic_info = self._extract_symbolic_information(x_normalized, basis_functions, spline_output)
            return output, symbolic_info

        return output

    def _extract_symbolic_information(self, x: torch.Tensor, basis_functions: torch.Tensor,
                                    spline_output: torch.Tensor) -> Dict[str, Any]:
        """Extract symbolic information from layer activations."""
        with torch.no_grad():
            # Analyze basis function activations
            basis_importance = torch.mean(basis_functions, dim=0).squeeze()  # (in_features, grid_size)

            # Find dominant basis functions
            max_activations = torch.max(basis_importance, dim=-1)[0]  # (in_features,)
            active_features = torch.where(max_activations > self.symbolic_threshold)[0]

            # Extract spline coefficients for symbolic approximation
            significant_weights = self.spline_weight[0, active_features, :].detach()

            # Pattern analysis
            pattern_variance = torch.var(spline_output, dim=0)
            pattern_complexity = torch.mean(pattern_variance)

            return {
                'active_features': active_features.tolist(),
                'basis_importance': basis_importance.numpy(),
                'significant_weights': significant_weights.numpy(),
                'pattern_complexity': float(pattern_complexity),
                'grid_points': self.grid.detach().numpy(),
                'output_variance': torch.var(spline_output).item()
            }

class SymbolicFunctionExtractor:
    """Extracts symbolic functions from KAN layer activations."""

    def __init__(self):
        self.logger = logging.getLogger("nis.symbolic.extractor")
        self.extraction_cache: Dict[str, sp.Expr] = {}

    def extract_function_from_kan(self, symbolic_info: Dict[str, Any],
                                 frequency_features: Optional[FrequencyPatternFeatures] = None) -> sp.Expr:
        """
        Extract symbolic function from KAN layer information.

        Args:
            symbolic_info: Symbolic information from KAN layer
            frequency_features: Optional frequency domain features

        Returns:
            Symbolic function as SymPy expression
        """
        # Create symbolic variable
        x = sp.Symbol('x', real=True)

        # Extract key information
        active_features = symbolic_info.get('active_features', [])
        significant_weights = symbolic_info.get('significant_weights', np.array([]))
        grid_points = symbolic_info.get('grid_points', np.linspace(-1, 1, 5))

        if len(active_features) == 0 or len(significant_weights) == 0:
            return x  # Return identity function as fallback

        # Build symbolic function based on dominant patterns
        terms = []

        for i, feature_idx in enumerate(active_features[:3]):  # Limit to top 3 features
            if i < len(significant_weights):
                weights = significant_weights[i]

                # Find dominant grid points
                dominant_indices = np.argsort(np.abs(weights))[-2:]  # Top 2 weights

                for idx in dominant_indices:
                    if idx < len(grid_points) and idx < len(weights):
                        coeff = float(weights[idx])
                        grid_point = float(grid_points[idx])

                        if abs(coeff) > 0.1:  # Only include significant terms
                            # Create basis function approximation
                            if frequency_features and frequency_features.dominant_frequencies:
                                # Include frequency information if available
                                freq = frequency_features.dominant_frequencies[0]
                                term = coeff * sp.sin(freq * (x - grid_point))
                            else:
                                # Polynomial approximation
                                term = coeff * (x - grid_point)**2

                            terms.append(term)

        if not terms:
            return x

        # Combine terms
        if len(terms) == 1:
            return terms[0]
        else:
            return sum(terms)

    def simplify_and_validate(self, expression: sp.Expr, validation_data: Optional[np.ndarray] = None) -> Tuple[sp.Expr, float]:
        """
        Simplify symbolic expression and validate against data.

        Args:
            expression: Symbolic expression to simplify
            validation_data: Optional validation data

        Returns:
            Simplified expression and validation score
        """
        try:
            # Simplify expression
            simplified = sp.simplify(expression)

            # Validate if data provided
            validation_score = 1.0
            if validation_data is not None and len(validation_data) > 0:
                validation_score = self._validate_expression(simplified, validation_data)

            return simplified, validation_score

        except Exception as e:
            self.logger.warning(f"Simplification failed: {e}")
            return expression, 0.5

    def _validate_expression(self, expression: sp.Expr, data: np.ndarray) -> float:
        """Validate symbolic expression against numerical data."""
        try:
            # Convert to numerical function
            x_sym = list(expression.free_symbols)[0] if expression.free_symbols else sp.Symbol('x')
            func = sp.lambdify(x_sym, expression, 'numpy')

            # Generate test points
            x_test = np.linspace(-1, 1, min(len(data), 100))
            y_pred = func(x_test)
            y_actual = data[:len(x_test)]

            # Calculate correlation
            correlation = np.corrcoef(y_pred, y_actual)[0, 1]
            return max(0.0, correlation) if not np.isnan(correlation) else 0.0

        except Exception:
            return 0.0

class KANSymbolicReasoningNetwork(nn.Module):
    """
    Enhanced KAN network for symbolic reasoning and function extraction.

    This network is specifically designed for the scientific reasoning pipeline
    in NIS Protocol V3, supporting symbolic function extraction from frequency
    domain patterns.
    """

    def __init__(self, input_dim: int = 10, hidden_dims: List[int] = [16, 8], output_dim: int = 1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Build network layers
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            layer_type = KANLayerType.SYMBOLIC_EXTRACTION if i == 0 else KANLayerType.FUNCTION_APPROXIMATION
            self.layers.append(EnhancedKANLayer(dims[i], dims[i+1], layer_type=layer_type))

        # Symbolic extraction components
        self.symbolic_extractor = SymbolicFunctionExtractor()

        # Activation functions
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()  # Bounded output for better symbolic extraction

    def forward(self, x: torch.Tensor, extract_symbolic: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass with optional symbolic extraction.

        Args:
            x: Input tensor
            extract_symbolic: Whether to extract symbolic information

        Returns:
            Output tensor and optionally symbolic extraction results
        """
        symbolic_data = {}
        h = x

        for i, layer in enumerate(self.layers):
            if extract_symbolic and i == 0:  # Extract from first layer
                h, layer_symbolic = layer(h, return_symbolic_info=True)
                symbolic_data[f'layer_{i}'] = layer_symbolic
            else:
                h = layer(h)

            # Apply activation (except for last layer)
            if i < len(self.layers) - 1:
                h = self.activation(h)
            else:
                h = self.output_activation(h)

        if extract_symbolic:
            return h, symbolic_data
        return h

    def extract_symbolic_function(self, input_data: torch.Tensor,
                                 frequency_features: Optional[FrequencyPatternFeatures] = None) -> SymbolicReasoningResult:
        """
        Extract symbolic function from network processing.

        Args:
            input_data: Input data for symbolic analysis
            frequency_features: Optional frequency domain features

        Returns:
            Complete symbolic reasoning result
        """
        start_time = time.time()

        with torch.no_grad():
            output, symbolic_data = self(input_data, extract_symbolic=True)

        # Extract primary symbolic function
        if symbolic_data:
            first_layer_info = list(symbolic_data.values())[0]
            symbolic_function = self.symbolic_extractor.extract_function_from_kan(
                first_layer_info, frequency_features
            )

            # Simplify and validate
            simplified_function, validation_score = self.symbolic_extractor.simplify_and_validate(
                symbolic_function, input_data.numpy().flatten()
            )
        else:
            x = sp.Symbol('x')
            simplified_function = x
            validation_score = 0.0

        # Calculate metrics
        computational_cost = time.time() - start_time
        confidence = validation_score * 0.8 + (1.0 - computational_cost / 10.0) * 0.2
        confidence = max(0.0, min(1.0, confidence))

        interpretability_score = self._calculate_interpretability(simplified_function)

        return SymbolicReasoningResult(
            symbolic_function=simplified_function,
            confidence=confidence,
            interpretability_score=interpretability_score,
            validation_metrics={'validation_score': validation_score, 'mse': 1.0 - validation_score},
            reasoning_type=SymbolicReasoningType.FUNCTION_EXTRACTION,
            computational_cost=computational_cost
        )

    def _calculate_interpretability(self, expression: sp.Expr) -> float:
        """Calculate interpretability score for symbolic expression."""
        try:
            # Factors affecting interpretability
            expr_str = str(expression)
            complexity = len(expr_str)
            num_operations = expr_str.count('+') + expr_str.count('-') + expr_str.count('*') + expr_str.count('/')
            num_symbols = len(expression.free_symbols)

            # Score based on complexity (simpler = more interpretable)
            complexity_score = max(0.1, 1.0 - complexity / 100.0)
            operations_score = max(0.1, 1.0 - num_operations / 10.0)
            symbols_score = max(0.1, 1.0 - num_symbols / 5.0)

            return (complexity_score + operations_score + symbols_score) / 3

        except Exception:
            return 0.5

class TerrainFeatureType(Enum):
    """Types of terrain features for archaeological analysis."""
    ELEVATION = "elevation"
    WATER_PROXIMITY = "water_proximity"
    SLOPE = "slope"
    VEGETATION_INDEX = "vegetation_index"
    HISTORICAL_MARKERS = "historical_markers"

@dataclass
class ArchaeologicalPrediction:
    """Result of archaeological site prediction."""
    site_probability: float
    confidence: float
    contributing_factors: Dict[str, float]
    cultural_sensitivity_score: float
    recommendations: List[str]
    interpretability_map: Dict[str, Any]

class KANLayer(nn.Module):
    """
    Simplified KAN layer implementation using spline-based activation functions.

    This replaces traditional MLPs with learnable univariate spline functions
    on edges, providing better interpretability and function approximation.
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        # Initialize spline coefficients
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size))
        self.spline_scaler = nn.Parameter(torch.randn(out_features, in_features))

        # Grid points for spline interpolation
        self.register_buffer('grid', torch.linspace(-1, 1, grid_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through KAN layer using spline interpolation.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        batch_size = x.shape[0]

        # Normalize input to [-1, 1] range
        x_normalized = torch.tanh(x)

        # Expand dimensions for broadcasting
        x_expanded = x_normalized.unsqueeze(1).unsqueeze(-1)  # (batch, 1, in_features, 1)
        grid_expanded = self.grid.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, grid_size)

        # Compute distances to grid points
        distances = torch.abs(x_expanded - grid_expanded)  # (batch, 1, in_features, grid_size)

        # RBF-like interpolation weights
        weights = torch.exp(-distances * self.spline_scaler.unsqueeze(0).unsqueeze(-1))

        # Apply spline weights
        spline_output = torch.sum(weights * self.spline_weight.unsqueeze(0), dim=-1)

        # Sum over input features
        output = torch.sum(spline_output, dim=-1)

        return output

class KANReasoningNetwork(nn.Module):
    """
    KAN-based neural network for archaeological site prediction.

    Uses spline-based layers instead of traditional MLPs for better
    interpretability and function approximation capabilities.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 16, output_dim: int = 1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # KAN layers
        self.kan_layer1 = KANLayer(input_dim, hidden_dim)
        self.kan_layer2 = KANLayer(hidden_dim, hidden_dim // 2)
        self.kan_layer3 = KANLayer(hidden_dim // 2, output_dim)

        # Activation functions
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with interpretability tracking.

        Args:
            x: Input features tensor

        Returns:
            Tuple of (output, interpretability_data)
        """
        interpretability_data = {}

        # Layer 1
        h1 = self.kan_layer1(x)
        h1_activated = self.activation(h1)
        interpretability_data['layer1_output'] = h1_activated.detach()

        # Layer 2
        h2 = self.kan_layer2(h1_activated)
        h2_activated = self.activation(h2)
        interpretability_data['layer2_output'] = h2_activated.detach()

        # Output layer
        output = self.kan_layer3(h2_activated)
        output_activated = self.output_activation(output)
        interpretability_data['final_output'] = output_activated.detach()

        return output_activated, interpretability_data

class WaveFieldProcessor:
    """
    Implements cognitive wave propagation for spatial reasoning.

    Models activation spreading across terrain patches using diffusion equations
    inspired by hippocampal spatial navigation maps.
    """

    def __init__(self, grid_size: int = 32, diffusion_rate: float = 0.1, decay_rate: float = 0.05):
        self.grid_size = grid_size
        self.diffusion_rate = diffusion_rate
        self.decay_rate = decay_rate
        self.phi = np.zeros((grid_size, grid_size))

    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute 5-point stencil Laplacian for diffusion."""
        return (
            np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
            4 * field
        )

    def update_field(self, site_probabilities: np.ndarray) -> np.ndarray:
        """
        Update cognitive activation field using wave propagation.

        Args:
            site_probabilities: 2D array of site prediction probabilities

        Returns:
            Updated activation field
        """
        # Apply diffusion equation: ∂φ/∂t = D∇²φ + S - Rφ
        laplacian_phi = self.laplacian(self.phi)
        self.phi += (
            self.diffusion_rate * laplacian_phi +  # Diffusion term
            site_probabilities -                    # Source term
            self.decay_rate * self.phi             # Decay term
        )

        return self.phi.copy()

class MemoryContextManager:
    """
    Manages persistent memory context using moving averages.

    Implements the Model Context Protocol (MCP) for agent coordination
    and maintains spatial memory of archaeological predictions.
    """

    def __init__(self, grid_size: int = 32, memory_alpha: float = 0.9):
        self.grid_size = grid_size
        self.memory_alpha = memory_alpha
        self.context_memory = np.zeros((grid_size, grid_size))

    def update_context(self, activation_field: np.ndarray) -> np.ndarray:
        """
        Update memory context using exponential moving average.

        Args:
            activation_field: Current cognitive activation field

        Returns:
            Updated context memory
        """
        self.context_memory = (
            self.memory_alpha * self.context_memory +
            (1 - self.memory_alpha) * activation_field
        )

        return self.context_memory.copy()

class KANReasoningAgent(NISAgent):
    """
    Enhanced KAN Reasoning Agent for NIS Protocol V3 Scientific Pipeline.

    This agent serves dual purposes:
    1. Symbolic reasoning and function extraction (V3 scientific pipeline)
    2. Archaeological site prediction (backward compatibility)

    The agent uses enhanced Kolmogorov-Arnold Networks for interpretable reasoning
    and integrates with the Laplace → KAN → PINN validation pipeline.
    """

    def __init__(self, agent_id: str = "kan_reasoning_001",
                 description: str = "Enhanced KAN symbolic reasoning with archaeological capabilities",
                 mode: str = "symbolic"):  # "symbolic" or "archaeological"
        super().__init__(agent_id, NISLayer.REASONING, description)

        self.mode = mode

        if mode == "symbolic":
            # Initialize enhanced symbolic reasoning network
            self.symbolic_network = KANSymbolicReasoningNetwork(
                input_dim=10,  # Configurable based on input type
                hidden_dims=[16, 8],
                output_dim=1
            )
        else:
            # Initialize original archaeological network for backward compatibility
            self.kan_network = KANReasoningNetwork(
                input_dim=5,  # [x, y, elevation, water_proximity, slope]
                hidden_dim=16,
                output_dim=1
            )

        # Initialize cognitive processing components (for archaeological mode)
        self.wave_processor = WaveFieldProcessor()
        self.memory_manager = MemoryContextManager()

        # Cultural sensitivity parameters (for archaeological mode)
        self.cultural_sensitivity_threshold = 0.8
        self.indigenous_rights_protection = True

        # V3 Scientific Pipeline Integration
        self.symbolic_processing_enabled = True
        self.frequency_features_cache: Dict[str, FrequencyPatternFeatures] = {}

        # Performance tracking
        self.processing_stats = {
            "symbolic_extractions": 0,
            "archaeological_predictions": 0,
            "successful_extractions": 0,
            "average_confidence": 0.0,
            "average_processing_time": 0.0
        }

        logger.info(f"Initialized Enhanced KAN Reasoning Agent: {agent_id} (mode: {mode})")

    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process reasoning requests - supports both symbolic and archaeological operations.

        Args:
            message: Input message containing operation type and payload

        Returns:
            Processed message with reasoning results
        """
        try:
            operation = message.get("operation", "symbolic_extraction")
            payload = message.get("payload", {})

            # V3 Scientific Pipeline Operations
            if operation == "symbolic_extraction":
                return self._extract_symbolic_function(payload)
            elif operation == "frequency_analysis":
                return self._analyze_frequency_patterns(payload)
            elif operation == "pattern_to_equation":
                return self._convert_pattern_to_equation(payload)
            elif operation == "validate_symbolic":
                return self._validate_symbolic_function(payload)

            # Backward Compatibility - Archaeological Operations
            elif operation == "predict_sites":
                return self._predict_archaeological_sites(payload)
            elif operation == "analyze_terrain":
                return self._analyze_terrain_features(payload)
            elif operation == "cultural_assessment":
                return self._assess_cultural_sensitivity(payload)

            # Hybrid Operations
            elif operation == "enhanced_reasoning":
                return self._perform_enhanced_reasoning(payload)

            else:
                return self._create_error_response(f"Unknown operation: {operation}")

        except Exception as e:
            logger.error(f"Error in KAN reasoning: {str(e)}")
            return self._create_error_response(str(e))

    def _extract_symbolic_function(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract symbolic function using enhanced KAN network.

        Args:
            payload: Contains input data and optional frequency features

        Returns:
            Symbolic function extraction results
        """
        if self.mode != "symbolic":
            return self._create_error_response("Agent not in symbolic mode")

        try:
            start_time = time.time()

            # Extract input data
            input_data = payload.get("input_data", [])
            frequency_features_data = payload.get("frequency_features", {})

            if not input_data:
                return self._create_error_response("No input data provided")

        # Convert to tensor
            if isinstance(input_data, list):
                input_tensor = torch.tensor(input_data, dtype=torch.float32)
                if input_tensor.dim() == 1:
                    input_tensor = input_tensor.unsqueeze(0)
            else:
                input_tensor = torch.tensor(input_data, dtype=torch.float32)

            # Create frequency features if provided
            frequency_features = None
            if frequency_features_data:
                frequency_features = FrequencyPatternFeatures(
                    dominant_frequencies=frequency_features_data.get("dominant_frequencies", []),
                    magnitude_peaks=frequency_features_data.get("magnitude_peaks", []),
                    phase_characteristics=frequency_features_data.get("phase_characteristics", {}),
                    spectral_centroid=frequency_features_data.get("spectral_centroid", 0.0),
                    bandwidth=frequency_features_data.get("bandwidth", 1.0),
                    energy=frequency_features_data.get("energy", 1.0),
                    pattern_complexity=frequency_features_data.get("pattern_complexity", 0.5)
                )

            # Perform symbolic extraction
            result = self.symbolic_network.extract_symbolic_function(input_tensor, frequency_features)

            # Update statistics
            self.processing_stats["symbolic_extractions"] += 1
            if result.confidence > 0.5:
                self.processing_stats["successful_extractions"] += 1

            processing_time = time.time() - start_time
            self._update_processing_stats(result.confidence, processing_time)

            return self._create_response("success", {
                "symbolic_function": str(result.symbolic_function),
                "confidence": result.confidence,
                "interpretability_score": result.interpretability_score,
                "validation_metrics": result.validation_metrics,
                "reasoning_type": result.reasoning_type.value,
                "computational_cost": result.computational_cost,
                "processing_time": processing_time,
                "alternative_functions": [str(f) for f in result.alternative_functions]
            })

        except Exception as e:
            logger.error(f"Symbolic extraction failed: {e}")
            return self._create_error_response(f"Symbolic extraction failed: {str(e)}")

    def _analyze_frequency_patterns(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze frequency domain patterns for symbolic reasoning preparation.

        Args:
            payload: Contains frequency domain data

        Returns:
            Pattern analysis results
        """
        try:
            # This would typically receive data from the Laplace processor
            frequency_data = payload.get("frequency_data", {})

            if not frequency_data:
                return self._create_error_response("No frequency data provided")

            # Extract frequency features
            dominant_freqs = frequency_data.get("dominant_frequencies", [])
            magnitude = frequency_data.get("magnitude", [])
            phase = frequency_data.get("phase", [])

            # Analyze patterns
            pattern_analysis = {
                "dominant_frequencies": dominant_freqs[:5],  # Top 5 frequencies
                "pattern_strength": self._calculate_pattern_strength(magnitude),
                "spectral_centroid": self._calculate_spectral_centroid(dominant_freqs, magnitude),
                "bandwidth": max(dominant_freqs) - min(dominant_freqs) if dominant_freqs else 0.0,
                "pattern_type": self._classify_frequency_pattern(dominant_freqs, magnitude)
            }

            # Cache features for later use
            cache_key = f"freq_{hash(str(frequency_data))}"
            self.frequency_features_cache[cache_key] = FrequencyPatternFeatures(
                dominant_frequencies=dominant_freqs,
                magnitude_peaks=magnitude,
                phase_characteristics={"variance": np.var(phase) if phase else 0.0},
                spectral_centroid=pattern_analysis["spectral_centroid"],
                bandwidth=pattern_analysis["bandwidth"],
                energy=sum(magnitude) if magnitude else 0.0,
                pattern_complexity=pattern_analysis["pattern_strength"]
            )

            return self._create_response("success", {
                "pattern_analysis": pattern_analysis,
                "cache_key": cache_key,
                "features_extracted": len(dominant_freqs)
            })

        except Exception as e:
            logger.error(f"Frequency analysis failed: {e}")
            return self._create_error_response(f"Frequency analysis failed: {str(e)}")

    def _convert_pattern_to_equation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert frequency patterns to symbolic equations.

        Args:
            payload: Contains pattern data and conversion parameters

        Returns:
            Equation conversion results
        """
        try:
            pattern_data = payload.get("pattern_data", {})
            cache_key = payload.get("cache_key", "")

            # Get cached frequency features if available
            frequency_features = None
            if cache_key in self.frequency_features_cache:
                frequency_features = self.frequency_features_cache[cache_key]

            # Create symbolic equations based on patterns
            equations = []
            t = sp.Symbol('t', real=True)

            if frequency_features and frequency_features.dominant_frequencies:
                for i, freq in enumerate(frequency_features.dominant_frequencies[:3]):
                    omega = 2 * np.pi * freq
                    amplitude = frequency_features.magnitude_peaks[i] if i < len(frequency_features.magnitude_peaks) else 1.0

                    if freq > 0:
                        equation = amplitude * sp.sin(omega * t)
                        equations.append(str(equation))

            if not equations:
                # Default polynomial if no patterns detected
                equations = [str(t**2)]

            return self._create_response("success", {
                "equations": equations,
                "primary_equation": equations[0] if equations else "x",
                "pattern_based": len(equations) > 1,
                "frequency_informed": frequency_features is not None
            })

        except Exception as e:
            logger.error(f"Pattern to equation conversion failed: {e}")
            return self._create_error_response(f"Conversion failed: {str(e)}")

    def _validate_symbolic_function(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate symbolic function against input data.

        Args:
            payload: Contains symbolic function and validation data

        Returns:
            Validation results
        """
        try:
            function_str = payload.get("function", "x")
            validation_data = payload.get("validation_data", [])

            if not validation_data:
                return self._create_error_response("No validation data provided")

            # Parse symbolic function
            try:
                function = sp.sympify(function_str)
            except Exception:
                return self._create_error_response("Invalid symbolic function")

            # Validate against data
            x_sym = list(function.free_symbols)[0] if function.free_symbols else sp.Symbol('x')
            func = sp.lambdify(x_sym, function, 'numpy')

            # Generate test points
            x_test = np.linspace(-1, 1, len(validation_data))
            y_pred = func(x_test)
            y_actual = np.array(validation_data)

            # Calculate validation metrics
            mse = np.mean((y_pred - y_actual)**2)
            correlation = np.corrcoef(y_pred, y_actual)[0, 1] if len(y_pred) > 1 else 0.0
            max_error = np.max(np.abs(y_pred - y_actual))

            validation_score = max(0.0, correlation) if not np.isnan(correlation) else 0.0

            return self._create_response("success", {
                "validation_score": validation_score,
                "mse": float(mse),
                "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
                "max_error": float(max_error),
                "function_valid": validation_score > 0.5,
                "recommendations": self._generate_validation_recommendations(validation_score, mse)
            })

        except Exception as e:
            logger.error(f"Function validation failed: {e}")
            return self._create_error_response(f"Validation failed: {str(e)}")

    def _perform_enhanced_reasoning(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform enhanced reasoning combining symbolic and contextual analysis.

        Args:
            payload: Contains multi-modal reasoning data

        Returns:
            Enhanced reasoning results
        """
        try:
            # This method combines both symbolic and archaeological reasoning
            symbolic_result = None
            archaeological_result = None

            # Try symbolic reasoning if in symbolic mode or data provided
            if self.mode == "symbolic" or payload.get("symbolic_data"):
                symbolic_payload = {"input_data": payload.get("symbolic_data", payload.get("input_data", []))}
                symbolic_response = self._extract_symbolic_function(symbolic_payload)
                if symbolic_response["status"] == "success":
                    symbolic_result = symbolic_response["payload"]

            # Try archaeological reasoning if terrain data provided
            if payload.get("terrain_features"):
                archaeological_payload = {"terrain_features": payload["terrain_features"]}
                archaeological_response = self._predict_archaeological_sites(archaeological_payload)
                if archaeological_response["status"] == "success":
                    archaeological_result = archaeological_response["payload"]

            # Combine results
            combined_confidence = 0.0
            if symbolic_result and archaeological_result:
                combined_confidence = (symbolic_result.get("confidence", 0) +
                                     archaeological_result["prediction"]["confidence"]) / 2
            elif symbolic_result:
                combined_confidence = symbolic_result.get("confidence", 0)
            elif archaeological_result:
                combined_confidence = archaeological_result["prediction"]["confidence"]

            return self._create_response("success", {
                "enhanced_reasoning": {
                    "symbolic_analysis": symbolic_result,
                    "spatial_analysis": archaeological_result,
                    "combined_confidence": combined_confidence,
                    "reasoning_type": "hybrid",
                    "recommendations": self._generate_hybrid_recommendations(symbolic_result, archaeological_result)
                }
            })

        except Exception as e:
            logger.error(f"Enhanced reasoning failed: {e}")
            return self._create_error_response(f"Enhanced reasoning failed: {str(e)}")

    def _calculate_pattern_strength(self, magnitude: List[float]) -> float:
        """Calculate strength of frequency pattern."""
        if not magnitude:
            return 0.0

        magnitude_array = np.array(magnitude)
        signal_to_noise = np.max(magnitude_array) / (np.mean(magnitude_array) + 1e-10)
        return min(1.0, signal_to_noise / 10.0)

    def _calculate_spectral_centroid(self, frequencies: List[float], magnitude: List[float]) -> float:
        """Calculate spectral centroid."""
        if not frequencies or not magnitude:
            return 0.0

        frequencies_array = np.array(frequencies)
        magnitude_array = np.array(magnitude[:len(frequencies)])

        if np.sum(magnitude_array) == 0:
            return 0.0

        return float(np.sum(frequencies_array * magnitude_array) / np.sum(magnitude_array))

    def _classify_frequency_pattern(self, frequencies: List[float], magnitude: List[float]) -> str:
        """Classify the type of frequency pattern."""
        if not frequencies or not magnitude:
            return "unknown"

        # Simple heuristic classification
        if len(frequencies) == 1:
            return "single_frequency"
        elif len(frequencies) <= 3:
            return "multi_tonal"
        elif self._calculate_pattern_strength(magnitude) > 0.7:
            return "complex_structured"
        else:
            return "broadband"

    def _generate_validation_recommendations(self, score: float, mse: float) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        if score > 0.8:
            recommendations.append("Excellent symbolic function match - ready for PINN validation")
        elif score > 0.6:
            recommendations.append("Good symbolic function - consider refinement")
        elif score > 0.4:
            recommendations.append("Moderate match - may need additional pattern analysis")
        else:
            recommendations.append("Poor match - recommend re-analysis with different approach")

        if mse > 1.0:
            recommendations.append("High prediction error - check input data quality")

        return recommendations

    def _generate_hybrid_recommendations(self, symbolic_result: Optional[Dict],
                                       archaeological_result: Optional[Dict]) -> List[str]:
        """Generate recommendations for hybrid reasoning results."""
        recommendations = []

        if symbolic_result and archaeological_result:
            recommendations.append("Multi-modal analysis completed successfully")

            if symbolic_result.get("confidence", 0) > 0.7:
                recommendations.append("Strong symbolic patterns detected")

            if archaeological_result["prediction"]["confidence"] > 0.7:
                recommendations.append("High archaeological potential identified")

        elif symbolic_result:
            recommendations.append("Symbolic analysis only - consider spatial context")
        elif archaeological_result:
            recommendations.append("Spatial analysis only - consider symbolic validation")
        else:
            recommendations.append("Insufficient data for comprehensive analysis")

        return recommendations

    def _update_processing_stats(self, confidence: float, processing_time: float):
        """Update processing statistics."""
        total_operations = (self.processing_stats["symbolic_extractions"] +
                          self.processing_stats["archaeological_predictions"])

        if total_operations > 0:
            self.processing_stats["average_confidence"] = (
                (self.processing_stats["average_confidence"] * (total_operations - 1) + confidence) /
                total_operations
            )
            self.processing_stats["average_processing_time"] = (
                (self.processing_stats["average_processing_time"] * (total_operations - 1) + processing_time) /
                total_operations
            )

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        total_extractions = self.processing_stats["symbolic_extractions"]
        success_rate = 0.0
        if total_extractions > 0:
            success_rate = self.processing_stats["successful_extractions"] / total_extractions

        return {
            **self.processing_stats,
            "success_rate": success_rate,
            "mode": self.mode
        }

    def _predict_archaeological_sites(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict archaeological site locations using KAN network.

        Args:
            payload: Contains terrain features and analysis parameters

        Returns:
            Archaeological site predictions with interpretability data
        """
        # Extract terrain features
        terrain_features = payload.get("terrain_features", [])
        grid_size = payload.get("grid_size", 32)

        if not terrain_features:
            return self._create_error_response("No terrain features provided")

        # Convert to tensor
        try:
            features_tensor = torch.tensor(terrain_features, dtype=torch.float32)
            if features_tensor.dim() == 1:
                features_tensor = features_tensor.unsqueeze(0)

            # Ensure correct input size for archaeological network
            if features_tensor.shape[-1] != 5:
                # Pad or truncate to match expected input size
                if features_tensor.shape[-1] < 5:
                    padding = torch.zeros(features_tensor.shape[0], 5 - features_tensor.shape[-1])
                    features_tensor = torch.cat([features_tensor, padding], dim=-1)
                else:
                    features_tensor = features_tensor[:, :5]
        except Exception as e:
            return self._create_error_response(f"Invalid terrain features format: {str(e)}")

        try:
            start_time = time.time()

            # Use archaeological network if available, otherwise use symbolic network
            if hasattr(self, 'kan_network'):
                with torch.no_grad():
                    predictions, interpretability_data = self.kan_network(features_tensor)
            else:
                # Fallback to symbolic network
                with torch.no_grad():
                    predictions = self.symbolic_network(features_tensor)
                    interpretability_data = {}

            # Process predictions
            prediction_array = predictions.numpy().reshape(grid_size, grid_size)

            # Update wave field
            updated_field = self.wave_processor.update_field(prediction_array)

            # Update memory context
            context_memory = self.memory_manager.update_context(updated_field)

            # Calculate metrics
            max_prob = float(np.max(prediction_array))
            mean_prob = float(np.mean(prediction_array))
            confidence = self._calculate_confidence(prediction_array)
            cultural_sensitivity = self._assess_cultural_factors(prediction_array)

            # Generate insights
            contributing_factors = self._analyze_contributing_factors(terrain_features, prediction_array)
            recommendations = self._generate_recommendations(max_prob, confidence, cultural_sensitivity)

            # Create prediction result
            prediction_result = ArchaeologicalPrediction(
                site_probability=max_prob,
            confidence=confidence,
            contributing_factors=contributing_factors,
                cultural_sensitivity_score=cultural_sensitivity,
            recommendations=recommendations,
                interpretability_map=interpretability_data
            )

            # Update statistics
            self.processing_stats["archaeological_predictions"] += 1
            processing_time = time.time() - start_time
            self._update_processing_stats(confidence, processing_time)

            return self._create_response("success", {
                "prediction": {
                    "site_probability": prediction_result.site_probability,
                    "confidence": prediction_result.confidence,
                    "cultural_sensitivity_score": prediction_result.cultural_sensitivity_score,
                    "contributing_factors": prediction_result.contributing_factors,
                    "recommendations": prediction_result.recommendations
                },
                "grid_predictions": prediction_array.tolist(),
                "wave_field": updated_field.tolist(),
                "processing_time": processing_time,
                "interpretability_data": interpretability_data
            })

        except Exception as e:
            logger.error(f"Archaeological prediction failed: {e}")
            return self._create_error_response(f"Prediction failed: {str(e)}")

    def _calculate_confidence(self, predictions: np.ndarray) -> float:
        """Calculate confidence score for archaeological predictions."""
        # Existing implementation from original code
        high_prob_sites = predictions > 0.7
        spatial_clustering = self._calculate_spatial_clustering(high_prob_sites)
        prediction_variance = np.var(predictions)

        # Combine metrics
        confidence = (spatial_clustering * 0.4 +
                     (1.0 - prediction_variance) * 0.3 +
                     np.max(predictions) * 0.3)

        return max(0.0, min(1.0, confidence))

    def _calculate_spatial_clustering(self, high_prob_sites: np.ndarray) -> float:
        """Calculate spatial clustering score."""
        if np.sum(high_prob_sites) == 0:
            return 0.5

        # Calculate clustering using connected components
        try:
            from scipy import ndimage
            labeled_sites, num_clusters = ndimage.label(high_prob_sites)

            # Prefer fewer, larger clusters over many scattered sites
            total_sites = np.sum(high_prob_sites)
            if total_sites == 0:
                return 0.5

            clustering_ratio = num_clusters / total_sites
            return max(0.1, 1.0 - clustering_ratio)
        except ImportError:
            # Fallback if scipy not available
            return 0.5

    def _assess_cultural_factors(self, predictions: np.ndarray) -> float:
        """Assess cultural sensitivity factors."""
        # Simplified cultural assessment
        # In practice, this would integrate with cultural databases
        return 0.8  # Default high sensitivity

    def _analyze_contributing_factors(self, terrain_features: List[List[float]],
                                    predictions: np.ndarray) -> Dict[str, float]:
        """Analyze which terrain features contribute most to predictions."""
        try:
            features_array = np.array(terrain_features)
            if features_array.ndim == 1:
                features_array = features_array.reshape(1, -1)

            # Calculate correlations between features and predictions
            flat_predictions = predictions.flatten()[:len(features_array)]

            factors = {}
            feature_names = ["x_coord", "y_coord", "elevation", "water_proximity", "slope"]

            for i, name in enumerate(feature_names):
                if i < features_array.shape[1]:
                    feature_values = features_array[:, i][:len(flat_predictions)]
                    correlation = np.corrcoef(feature_values, flat_predictions)[0, 1]
                    factors[name] = float(correlation) if not np.isnan(correlation) else 0.0

            return factors
        except Exception:
            return {"terrain_analysis": 0.5}

    def _generate_recommendations(self, max_prob: float, confidence: float,
                                cultural_sensitivity: float) -> List[str]:
        """Generate actionable recommendations for archaeological investigation."""
        recommendations = []

        if max_prob > 0.8 and confidence > 0.7:
            recommendations.append("High-priority site for detailed ground survey")
        elif max_prob > 0.6:
            recommendations.append("Moderate-priority site for preliminary investigation")
        else:
            recommendations.append("Low-priority area, consider for future surveys")

        if cultural_sensitivity < self.cultural_sensitivity_threshold:
            recommendations.append("CULTURAL ALERT: Consult with local indigenous communities")
            recommendations.append("Review cultural appropriation guidelines before proceeding")

        if confidence < 0.5:
            recommendations.append("Gather additional terrain data to improve prediction confidence")

        recommendations.append("Apply First Contact Protocol for any discoveries")

        return recommendations

# Example usage and testing
if __name__ == "__main__":
    # Create KAN reasoning agent
    agent = KANReasoningAgent()

    # Generate sample terrain data
    grid_size = 32
    num_patches = grid_size * grid_size

    # Sample features: [x, y, elevation, water_proximity, slope]
    terrain_features = np.random.rand(num_patches, 5).tolist()

    # Test archaeological site prediction
    test_message = {
        "operation": "predict_sites",
        "payload": {
            "terrain_features": terrain_features,
            "grid_size": grid_size
        }
    }

    result = agent.process(test_message)

    if result["status"] == "success":
        prediction = result["payload"]["prediction"]
        print(f"🏛️ Archaeological Site Prediction Results:")
        print(f"   Site Probability: {prediction['site_probability']:.3f}")
        print(f"   Confidence: {prediction['confidence']:.3f}")
        print(f"   Cultural Sensitivity: {prediction['cultural_sensitivity_score']:.3f}")
        print(f"   Recommendations: {len(prediction['recommendations'])} items")
        print(f"   Interpretability: KAN spline-based reasoning enabled")
    else:
        print(f"❌ Error: {result['payload']}")

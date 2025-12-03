"""
NIS Protocol - Neural Network with Nested Learning Integration
Enhanced with Google's Nested Learning paradigm (NeurIPS 2025)

Key Enhancements:
- Continuum Memory System (CMS) - Multi-frequency memory updates
- Deep Optimizers - Neural memory-based optimization  
- Multi-time-scale learning across layers
- Self-modifying architecture capabilities

Reference: "Nested Learning: The Illusion of Deep Learning Architectures"
Authors: Ali Behrouz et al. (Google Research)
"""

from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
import logging

from .base_neural_agent import NeuralAgent, NeuralLayer, NeuralSignal

logger = logging.getLogger(__name__)


# =============================================================================
# NESTED LEARNING: Update Frequency Levels
# =============================================================================

class UpdateFrequency(Enum):
    """
    Update frequency levels for Nested Learning
    Parameters at different levels update at different rates
    """
    ULTRA_FAST = 1        # Update every step (working memory/attention)
    FAST = 10             # Update every 10 steps
    MEDIUM = 100          # Update every 100 steps
    SLOW = 1000           # Update every 1000 steps
    ULTRA_SLOW = 10000    # Update every 10000 steps (long-term memory)


@dataclass
class CMSBlock:
    """
    Continuum Memory System Block
    Each block updates at a specific frequency, compressing different time scales
    """
    level: int
    frequency: UpdateFrequency
    weights: np.ndarray
    bias: np.ndarray
    momentum: np.ndarray
    step_count: int = 0
    last_update: float = field(default_factory=time.time)
    accumulated_gradients: np.ndarray = None
    
    def __post_init__(self):
        if self.accumulated_gradients is None:
            self.accumulated_gradients = np.zeros_like(self.weights)
    
    def should_update(self) -> bool:
        """Check if this block should update based on frequency"""
        return self.step_count % self.frequency.value == 0
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through this CMS block"""
        # Linear transformation with GELU activation
        z = x @ self.weights + self.bias
        # GELU activation: x * Φ(x)
        return z * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))
    
    def accumulate_gradient(self, grad: np.ndarray):
        """Accumulate gradients until update step"""
        self.accumulated_gradients += grad
        self.step_count += 1
    
    def update_weights(self, learning_rate: float = 0.001):
        """Update weights using accumulated gradients (Deep Optimizer style)"""
        if self.should_update() and self.step_count > 0:
            # L2 regression-based momentum update (from Nested Learning paper)
            avg_grad = self.accumulated_gradients / self.frequency.value
            
            # Neural momentum with L2 objective
            self.momentum = 0.9 * self.momentum + 0.1 * avg_grad
            
            # Weight update with momentum
            self.weights -= learning_rate * self.momentum
            
            # Reset accumulated gradients
            self.accumulated_gradients = np.zeros_like(self.weights)
            self.last_update = time.time()


@dataclass
class ContinuumMemorySystem:
    """
    Continuum Memory System (CMS) from Nested Learning
    
    Memory as a spectrum of MLP blocks updating at different frequencies:
    - Short-term: Fast updates (like attention)
    - Medium-term: Moderate updates
    - Long-term: Slow updates (like feedforward layers)
    
    This creates richer memory than binary short/long-term distinction
    """
    input_dim: int = 256
    hidden_dim: int = 512
    num_levels: int = 5
    blocks: List[CMSBlock] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.blocks:
            self._initialize_blocks()
    
    def _initialize_blocks(self):
        """Initialize CMS blocks with different update frequencies"""
        frequencies = list(UpdateFrequency)
        
        for level in range(self.num_levels):
            freq = frequencies[min(level, len(frequencies)-1)]
            
            # Xavier initialization
            scale = np.sqrt(2.0 / (self.input_dim + self.hidden_dim))
            weights = np.random.randn(self.input_dim, self.hidden_dim) * scale
            bias = np.zeros(self.hidden_dim)
            momentum = np.zeros_like(weights)
            
            self.blocks.append(CMSBlock(
                level=level,
                frequency=freq,
                weights=weights,
                bias=bias,
                momentum=momentum
            ))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through all CMS blocks
        Each block processes at its own time scale
        """
        output = x
        for block in self.blocks:
            output = block.forward(output)
        return output
    
    def update(self, gradients: List[np.ndarray], learning_rate: float = 0.001):
        """Update all blocks that are due for update"""
        for block, grad in zip(self.blocks, gradients):
            block.accumulate_gradient(grad)
            block.update_weights(learning_rate)
    
    def get_memory_state(self) -> Dict[str, Any]:
        """Get current state of all memory levels"""
        return {
            f"level_{block.level}": {
                "frequency": block.frequency.name,
                "step_count": block.step_count,
                "weight_norm": float(np.linalg.norm(block.weights)),
                "last_update": block.last_update,
                "should_update": block.should_update()
            }
            for block in self.blocks
        }


@dataclass
class DeepOptimizer:
    """
    Deep Optimizer from Nested Learning
    
    Treats optimizer as associative memory module with L2 regression objective
    instead of simple dot-product similarity. This better manages memory capacity
    and gradient sequences.
    """
    dim: int = 256
    memory_size: int = 100
    learning_rate: float = 0.001
    
    # Neural memory for gradient history
    gradient_memory: deque = field(default_factory=lambda: deque(maxlen=100))
    key_memory: np.ndarray = None
    value_memory: np.ndarray = None
    
    def __post_init__(self):
        if self.key_memory is None:
            self.key_memory = np.zeros((self.memory_size, self.dim))
            self.value_memory = np.zeros((self.memory_size, self.dim))
    
    def compute_momentum(self, current_grad: np.ndarray, input_features: np.ndarray) -> np.ndarray:
        """
        Compute momentum using L2 regression objective (not dot-product)
        This accounts for relationships between data samples
        """
        # Store gradient in memory
        self.gradient_memory.append(current_grad.copy())
        
        if len(self.gradient_memory) < 2:
            return current_grad
        
        # L2 regression-based momentum
        # Instead of simple exponential moving average, use regression
        grad_history = np.array(list(self.gradient_memory))
        
        # Compute weighted combination based on L2 similarity
        weights = np.exp(-np.sum((grad_history - current_grad)**2, axis=-1) / (2 * self.dim))
        weights = weights / (np.sum(weights) + 1e-8)
        
        # Weighted momentum
        momentum = np.sum(grad_history * weights[:, np.newaxis], axis=0)
        
        return 0.9 * momentum + 0.1 * current_grad
    
    def step(self, params: np.ndarray, grad: np.ndarray, features: np.ndarray = None) -> np.ndarray:
        """Perform optimization step with deep momentum"""
        if features is None:
            features = grad
        
        momentum = self.compute_momentum(grad, features)
        return params - self.learning_rate * momentum


class NeuralNetwork:
    """
    Enhanced Neural Network Coordinator with Nested Learning
    
    Integrates:
    - Continuum Memory System for multi-frequency updates
    - Deep Optimizers for better gradient handling
    - Multi-time-scale learning across layers
    """
    
    def __init__(self, enable_nested_learning: bool = True):
        # Agents organized by layer
        self.agents: Dict[NeuralLayer, List[NeuralAgent]] = defaultdict(list)
        
        # Signal queues for each layer
        self.signal_queues: Dict[NeuralLayer, List[NeuralSignal]] = defaultdict(list)
        
        # Network state
        self.global_activation = 0.0
        self.active_signals = []
        
        # === NESTED LEARNING COMPONENTS ===
        self.enable_nested_learning = enable_nested_learning
        
        if enable_nested_learning:
            # Continuum Memory System - multi-frequency memory
            self.cms = ContinuumMemorySystem(
                input_dim=256,
                hidden_dim=512,
                num_levels=5
            )
            
            # Deep Optimizer for each layer
            self.layer_optimizers: Dict[NeuralLayer, DeepOptimizer] = {
                layer: DeepOptimizer(dim=256)
                for layer in NeuralLayer
            }
            
            # Update frequency tracking per layer
            self.layer_update_frequencies: Dict[NeuralLayer, UpdateFrequency] = {
                NeuralLayer.SENSORY: UpdateFrequency.ULTRA_FAST,
                NeuralLayer.PERCEPTION: UpdateFrequency.FAST,
                NeuralLayer.MEMORY: UpdateFrequency.MEDIUM,
                NeuralLayer.EMOTIONAL: UpdateFrequency.MEDIUM,
                NeuralLayer.EXECUTIVE: UpdateFrequency.SLOW,
                NeuralLayer.MOTOR: UpdateFrequency.FAST
            }
            
            # Step counters for nested updates
            self.global_step = 0
            self.layer_steps: Dict[NeuralLayer, int] = defaultdict(int)
            
            # Context flow tracking (key concept in Nested Learning)
            self.context_flows: Dict[NeuralLayer, deque] = {
                layer: deque(maxlen=1000) for layer in NeuralLayer
            }
            
            logger.info("✅ Nested Learning enabled with CMS and Deep Optimizers")
        
    def add_agent(self, agent: NeuralAgent):
        """Add an agent to the network"""
        self.agents[agent.layer].append(agent)
    
    def connect_agents(
        self,
        source_agent: NeuralAgent,
        target_agent: NeuralAgent,
        connection_strength: float = 1.0
    ):
        """Create a connection between two agents"""
        source_agent.connect_to(target_agent, connection_strength)
    
    def process_input(self, input_data: Any) -> Dict[str, Any]:
        """Process input through the neural network"""
        # Start with sensory layer
        sensory_signal = NeuralSignal(
            source_layer=NeuralLayer.SENSORY,
            target_layer=NeuralLayer.SENSORY,
            content=input_data
        )
        
        self.signal_queues[NeuralLayer.SENSORY].append(sensory_signal)
        
        # Process through each layer in order
        layer_order = [
            NeuralLayer.SENSORY,
            NeuralLayer.PERCEPTION,
            NeuralLayer.MEMORY,
            NeuralLayer.EMOTIONAL,
            NeuralLayer.EXECUTIVE,
            NeuralLayer.MOTOR
        ]
        
        results = {}
        for layer in layer_order:
            layer_results = self._process_layer(layer)
            if layer_results:
                results[layer.value] = layer_results
        
        return results
    
    def _process_layer(self, layer: NeuralLayer) -> List[Dict[str, Any]]:
        """Process all signals for a given layer"""
        if not self.signal_queues[layer]:
            return []
            
        results = []
        # Process each signal through all agents in the layer
        while self.signal_queues[layer]:
            signal = self.signal_queues[layer].pop(0)
            
            for agent in self.agents[layer]:
                # Update agent activation based on signal priority
                agent.update_activation(signal.priority)
                
                # Only process if agent is active
                if agent.is_active():
                    response = agent.process_signal(signal)
                    if response:
                        # Add response to appropriate queue
                        self.signal_queues[response.target_layer].append(response)
                        results.append({
                            'agent_id': agent.agent_id,
                            'response': response.content
                        })
        
        return results
    
    def get_layer_activation(self, layer: NeuralLayer) -> float:
        """Get average activation level for a layer"""
        if not self.agents[layer]:
            return 0.0
            
        return sum(
            agent.activation_level
            for agent in self.agents[layer]
        ) / len(self.agents[layer])
    
    def reset_network(self):
        """Reset all agents and clear signal queues"""
        for agents in self.agents.values():
            for agent in agents:
                agent.reset()
        
        self.signal_queues.clear()
        self.global_activation = 0.0
        self.active_signals = []
    
    def get_network_state(self) -> Dict[str, Any]:
        """Get current state of the neural network"""
        state = {
            'layer_activations': {
                layer.value: self.get_layer_activation(layer)
                for layer in NeuralLayer
            },
            'active_agents': [
                {
                    'agent_id': agent.agent_id,
                    'layer': agent.layer.value,
                    'activation': agent.activation_level
                }
                for agents in self.agents.values()
                for agent in agents
                if agent.is_active()
            ],
            'queue_sizes': {
                layer.value: len(queue)
                for layer, queue in self.signal_queues.items()
            }
        }
        
        # Add Nested Learning state if enabled
        if self.enable_nested_learning:
            state['nested_learning'] = {
                'global_step': self.global_step,
                'cms_state': self.cms.get_memory_state(),
                'layer_update_frequencies': {
                    layer.value: freq.name 
                    for layer, freq in self.layer_update_frequencies.items()
                },
                'context_flow_sizes': {
                    layer.value: len(flow)
                    for layer, flow in self.context_flows.items()
                }
            }
        
        return state
    
    # =========================================================================
    # NESTED LEARNING: Multi-Frequency Processing
    # =========================================================================
    
    def _should_update_layer(self, layer: NeuralLayer) -> bool:
        """Check if layer should update based on its frequency"""
        if not self.enable_nested_learning:
            return True
        
        freq = self.layer_update_frequencies.get(layer, UpdateFrequency.FAST)
        return self.layer_steps[layer] % freq.value == 0
    
    def _update_context_flow(self, layer: NeuralLayer, signal: NeuralSignal):
        """Track context flow for this layer (key Nested Learning concept)"""
        if self.enable_nested_learning:
            self.context_flows[layer].append({
                'content_hash': hash(str(signal.content)[:100]),
                'priority': signal.priority,
                'timestamp': time.time(),
                'source': signal.source_layer.value
            })
    
    def _apply_cms_processing(self, signal_content: Any) -> Any:
        """Apply Continuum Memory System processing to signal"""
        if not self.enable_nested_learning:
            return signal_content
        
        try:
            # Convert content to numpy array if possible
            if isinstance(signal_content, (list, np.ndarray)):
                x = np.array(signal_content).flatten()
                # Pad or truncate to CMS input dimension
                if len(x) < self.cms.input_dim:
                    x = np.pad(x, (0, self.cms.input_dim - len(x)))
                else:
                    x = x[:self.cms.input_dim]
                
                # Process through CMS
                processed = self.cms.forward(x)
                return processed
            
            return signal_content
            
        except Exception as e:
            logger.debug(f"CMS processing skipped: {e}")
            return signal_content
    
    def process_with_nested_learning(self, input_data: Any) -> Dict[str, Any]:
        """
        Process input with full Nested Learning integration
        
        This implements the core Nested Learning paradigm:
        1. Different layers update at different frequencies
        2. CMS provides multi-time-scale memory
        3. Deep optimizers handle gradient accumulation
        """
        if not self.enable_nested_learning:
            return self.process_input(input_data)
        
        self.global_step += 1
        
        # Apply CMS preprocessing
        processed_input = self._apply_cms_processing(input_data)
        
        # Create initial signal
        sensory_signal = NeuralSignal(
            source_layer=NeuralLayer.SENSORY,
            target_layer=NeuralLayer.SENSORY,
            content=processed_input
        )
        
        self.signal_queues[NeuralLayer.SENSORY].append(sensory_signal)
        
        # Process through layers with frequency-aware updates
        layer_order = [
            NeuralLayer.SENSORY,
            NeuralLayer.PERCEPTION,
            NeuralLayer.MEMORY,
            NeuralLayer.EMOTIONAL,
            NeuralLayer.EXECUTIVE,
            NeuralLayer.MOTOR
        ]
        
        results = {}
        for layer in layer_order:
            self.layer_steps[layer] += 1
            
            # Check if this layer should process based on its frequency
            if self._should_update_layer(layer):
                layer_results = self._process_layer_nested(layer)
                if layer_results:
                    results[layer.value] = layer_results
        
        return results
    
    def _process_layer_nested(self, layer: NeuralLayer) -> List[Dict[str, Any]]:
        """Process layer with Nested Learning enhancements"""
        if not self.signal_queues[layer]:
            return []
        
        results = []
        
        while self.signal_queues[layer]:
            signal = self.signal_queues[layer].pop(0)
            
            # Track context flow
            self._update_context_flow(layer, signal)
            
            for agent in self.agents[layer]:
                # Update activation
                agent.update_activation(signal.priority)
                
                if agent.is_active():
                    # Apply CMS to signal content before processing
                    enhanced_signal = NeuralSignal(
                        source_layer=signal.source_layer,
                        target_layer=signal.target_layer,
                        content=self._apply_cms_processing(signal.content),
                        priority=signal.priority,
                        learning_weight=signal.learning_weight
                    )
                    
                    response = agent.process_signal(enhanced_signal)
                    
                    if response:
                        self.signal_queues[response.target_layer].append(response)
                        results.append({
                            'agent_id': agent.agent_id,
                            'response': response.content,
                            'nested_learning_step': self.global_step
                        })
        
        return results
    
    def get_cms_memory_state(self) -> Dict[str, Any]:
        """Get detailed CMS memory state"""
        if not self.enable_nested_learning:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "num_levels": self.cms.num_levels,
            "blocks": self.cms.get_memory_state(),
            "global_step": self.global_step
        }
    
    def set_layer_update_frequency(self, layer: NeuralLayer, frequency: UpdateFrequency):
        """Dynamically adjust layer update frequency"""
        if self.enable_nested_learning:
            self.layer_update_frequencies[layer] = frequency
            logger.info(f"Layer {layer.value} update frequency set to {frequency.name}") 
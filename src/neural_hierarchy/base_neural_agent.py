"""
Enhanced Neural Hierarchy Base Agent

V3.0 Enhancements:
- Neural plasticity with adaptive connection weights
- Advanced learning mechanisms with memory consolidation
- Cross-layer coordination and synchronization
- Performance optimization and convergence tracking
- Cultural neutrality in neural processing
- Template-based architecture for modular integration
"""

from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
import logging
from collections import deque, defaultdict
import math
import time

class NeuralLayer(Enum):
    SENSORY = "sensory"          # Input processing
    PERCEPTION = "perception"    # Pattern recognition and interpretation
    EXECUTIVE = "executive"      # Decision making and control
    MEMORY = "memory"           # Information storage and retrieval
    EMOTIONAL = "emotional"     # Emotional processing and regulation
    MOTOR = "motor"            # Output generation and action

class LearningType(Enum):
    """Types of learning mechanisms"""
    HEBBIAN = "hebbian"              # Strengthen active connections
    TEMPORAL_DIFFERENCE = "td"       # Reward-based learning
    CONTRASTIVE = "contrastive"      # Compare positive/negative examples
    HOMEOSTATIC = "homeostatic"      # Maintain optimal activity levels
    CROSS_LAYER = "cross_layer"      # Learn across layer boundaries

@dataclass
class NeuralSignal:
    """Enhanced neural signal with learning metadata"""
    source_layer: NeuralLayer
    target_layer: NeuralLayer
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    priority: float = 1.0  # Signal priority/strength
    learning_weight: float = 1.0  # Weight for learning updates
    cultural_context: str = "neutral"  # Cultural context for neutral processing
    convergence_factor: float = 0.0  # Factor for convergence tracking
    
@dataclass
class ConnectionWeight:
    """Enhanced connection weight with learning metadata"""
    weight: float
    last_updated: datetime = field(default_factory=datetime.now)
    learning_rate: float = 0.01
    stability_score: float = 0.5
    cultural_neutrality: float = 0.8
    usage_count: int = 0
    
@dataclass 
class LearningMetrics:
    """Metrics for tracking learning progress"""
    learning_rate: float = 0.01
    convergence_score: float = 0.0
    plasticity_index: float = 0.5
    stability_measure: float = 0.5
    cultural_balance: float = 0.8
    adaptation_speed: float = 0.3
    performance_trend: List[float] = field(default_factory=list)

class NeuralAgent(ABC):
    """Enhanced base class for all neural agents with V3 learning capabilities"""
    
    def __init__(
        self,
        agent_id: str,
        layer: NeuralLayer,
        description: str = "",
        activation_threshold: float = 0.5,
        learning_rate: float = 0.01,
        plasticity_factor: float = 0.1,
        cultural_neutrality_weight: float = 0.8
    ):
        # Core agent properties
        self.agent_id = agent_id
        self.layer = layer
        self.description = description
        self.activation_threshold = activation_threshold
        self.activation_level = 0.0
        
        # Enhanced connection management
        self.connected_agents: Dict[str, ConnectionWeight] = {}
        self.connection_history: deque = deque(maxlen=1000)
        
        # Learning and adaptation
        self.learning_metrics = LearningMetrics(
            learning_rate=learning_rate,
            plasticity_index=plasticity_factor,
            cultural_balance=cultural_neutrality_weight
        )
        
        # Neural plasticity
        self.plasticity_factor = plasticity_factor
        self.weight_decay = 0.95  # Gradual weight decay
        self.homeostatic_target = 0.5  # Target activation level
        
        # Cultural neutrality
        self.cultural_neutrality_weight = cultural_neutrality_weight
        self.cultural_bias_detection = True
        self.cultural_balance_tracker = defaultdict(float)
        
        # Performance tracking
        self.state = {}
        self.performance_history: deque = deque(maxlen=100)
        self.signal_processing_stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "average_latency": 0.0
        }
        
        # Cross-layer coordination
        self.layer_coordination_state = {}
        self.synchronization_signals: deque = deque(maxlen=50)
        
        # Convergence tracking
        self.convergence_history: deque = deque(maxlen=200)
        self.stability_window = 20
        
        # Logger
        self.logger = logging.getLogger(f"neural.{agent_id}")
        
        self.logger.info(f"Enhanced NeuralAgent {agent_id} initialized in {layer.value} layer")
        
    @abstractmethod
    def process_signal(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Process incoming neural signal and optionally generate response"""
        pass
    
    def enhanced_process_signal(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Enhanced signal processing with learning and cultural neutrality"""
        start_time = time.time()
        
        try:
            # Pre-processing: Cultural neutrality assessment
            self._assess_cultural_neutrality(signal)
            
            # Update activation based on signal
            self.update_activation(signal.priority * signal.learning_weight)
            
            # Process signal through child implementation
            response = self.process_signal(signal)
            
            # Post-processing: Learning updates
            if response:
                self._apply_learning_updates(signal, response)
                self._update_convergence_tracking(signal, response)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._update_performance_stats(True, processing_time)
            
            # Track signal for analysis
            self.synchronization_signals.append({
                "timestamp": signal.timestamp,
                "source": signal.source_layer.value,
                "target": signal.target_layer.value,
                "priority": signal.priority,
                "cultural_context": signal.cultural_context,
                "processing_time": processing_time
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
            self._update_performance_stats(False, time.time() - start_time)
            return None
    
    def connect_to(self, other_agent: 'NeuralAgent', initial_weight: float = 1.0, 
                  learning_rate: float = None) -> None:
        """Enhanced connection with adaptive weights and learning"""
        connection_weight = ConnectionWeight(
            weight=initial_weight,
            learning_rate=learning_rate or self.learning_metrics.learning_rate,
            cultural_neutrality=self.cultural_neutrality_weight
        )
        
        self.connected_agents[other_agent.agent_id] = connection_weight
        
        # Track connection creation
        self.connection_history.append({
            "action": "connect",
            "target": other_agent.agent_id,
            "initial_weight": initial_weight,
            "timestamp": datetime.now()
        })
        
        self.logger.info(f"Connected to {other_agent.agent_id} with weight {initial_weight}")
    
    def send_signal(self, target_agent_id: str, content: Any, 
                   learning_weight: float = 1.0, cultural_context: str = "neutral") -> Optional[NeuralSignal]:
        """Enhanced signal sending with learning metadata"""
        if target_agent_id not in self.connected_agents:
            return None
            
        connection = self.connected_agents[target_agent_id]
        
        # Calculate effective signal strength
        effective_strength = (
            connection.weight * 
            self.activation_level * 
            learning_weight *
            connection.cultural_neutrality
        )
        
        # Create enhanced signal
        signal = NeuralSignal(
            source_layer=self.layer,
            target_layer=self.layer,  # Will be updated by target
            content=content,
            priority=effective_strength,
            learning_weight=learning_weight,
            cultural_context=cultural_context,
            convergence_factor=self._calculate_convergence_factor()
        )
        
        # Update connection usage
        connection.usage_count += 1
        connection.last_updated = datetime.now()
        
        return signal
    
    def update_activation(self, signal_strength: float):
        """Enhanced activation update with homeostatic regulation"""
        # Standard activation update
        self.activation_level = min(1.0, self.activation_level + signal_strength)
        
        # Homeostatic regulation - maintain optimal activation
        homeostatic_adjustment = (self.homeostatic_target - self.activation_level) * 0.1
        self.activation_level += homeostatic_adjustment
        
        # Ensure bounds
        self.activation_level = max(0.0, min(1.0, self.activation_level))
        
        # Apply gradual decay
        self.activation_level *= self.weight_decay
        
        # Track for convergence analysis
        self.convergence_history.append(self.activation_level)
    
    def apply_neural_plasticity(self, source_agent_id: str, success_signal: bool, 
                               cultural_balance: float = 0.8) -> None:
        """Apply neural plasticity to update connection weights"""
        if source_agent_id not in self.connected_agents:
            return
            
        connection = self.connected_agents[source_agent_id]
        
        # Hebbian learning: strengthen successful connections
        if success_signal:
            weight_increase = connection.learning_rate * self.plasticity_factor * cultural_balance
            connection.weight = min(2.0, connection.weight + weight_increase)
            connection.stability_score = min(1.0, connection.stability_score + 0.05)
        else:
            # Weaken unsuccessful connections
            weight_decrease = connection.learning_rate * self.plasticity_factor * 0.5
            connection.weight = max(0.1, connection.weight - weight_decrease)
            connection.stability_score = max(0.1, connection.stability_score - 0.03)
        
        # Update cultural neutrality
        connection.cultural_neutrality = (
            connection.cultural_neutrality * 0.9 + cultural_balance * 0.1
        )
        
        # Track plasticity change
        self.connection_history.append({
            "action": "plasticity_update",
            "target": source_agent_id,
            "new_weight": connection.weight,
            "success": success_signal,
            "cultural_balance": cultural_balance,
            "timestamp": datetime.now()
        })
    
    def cross_layer_learning(self, layer_signals: Dict[str, NeuralSignal]) -> Dict[str, float]:
        """Implement cross-layer learning coordination"""
        learning_updates = {}
        
        try:
            # Analyze signals from different layers
            layer_activations = {}
            for layer_name, signal in layer_signals.items():
                if signal:
                    layer_activations[layer_name] = signal.priority
            
            # Calculate cross-layer correlation
            if len(layer_activations) > 1:
                # Find most active layer
                max_layer = max(layer_activations, key=layer_activations.get)
                max_activation = layer_activations[max_layer]
                
                # Update coordination with other layers
                for layer_name, activation in layer_activations.items():
                    if layer_name != max_layer:
                        # Calculate coordination strength
                        coordination_strength = activation / max(max_activation, 0.01)
                        learning_updates[layer_name] = coordination_strength
                        
                        # Update layer coordination state
                        self.layer_coordination_state[layer_name] = {
                            "coordination_strength": coordination_strength,
                            "last_update": datetime.now(),
                            "activation_ratio": activation / max_activation
                        }
            
            return learning_updates
            
        except Exception as e:
            self.logger.error(f"Error in cross-layer learning: {e}")
            return {}
    
    def calculate_convergence_metrics(self) -> Dict[str, float]:
        """Calculate convergence metrics for this agent"""
        try:
            if len(self.convergence_history) < self.stability_window:
                return {"status": "insufficient_data", "convergence_score": 0.3}
            
            recent_activations = list(self.convergence_history)[-self.stability_window:]
            
            # Calculate stability (low variance = high stability)
            stability = 1.0 - np.var(recent_activations)
            stability = max(0.0, min(1.0, stability))
            
            # Calculate trend (consistent direction = convergence)
            trend_score = self._calculate_trend_score(recent_activations)
            
            # Calculate oscillation (low oscillation = better convergence)
            oscillation_score = self._calculate_oscillation_score(recent_activations)
            
            # Overall convergence score
            convergence_score = (stability * 0.4 + trend_score * 0.3 + oscillation_score * 0.3)
            
            # Update learning metrics
            self.learning_metrics.convergence_score = convergence_score
            self.learning_metrics.stability_measure = stability
            
            return {
                "convergence_score": convergence_score,
                "stability": stability,
                "trend_score": trend_score,
                "oscillation_score": oscillation_score,
                "status": "converged" if convergence_score > 0.8 else "converging" if convergence_score > 0.6 else "oscillating"
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating convergence metrics: {e}")
            return {"status": "error", "convergence_score": 0.3}
    
    def optimize_learning_parameters(self) -> Dict[str, float]:
        """Optimize learning parameters based on performance"""
        try:
            convergence_metrics = self.calculate_convergence_metrics()
            
            # Adaptive learning rate
            if convergence_metrics["convergence_score"] > 0.8:
                # Reduce learning rate when converged
                self.learning_metrics.learning_rate *= 0.95
            elif convergence_metrics["convergence_score"] < 0.4:
                # Increase learning rate when not learning
                self.learning_metrics.learning_rate *= 1.05
            
            # Adaptive plasticity
            stability = convergence_metrics.get("stability", 0.5)
            if stability < 0.5:
                # Increase plasticity for more adaptation
                self.plasticity_factor = min(0.3, self.plasticity_factor * 1.02)
            else:
                # Reduce plasticity when stable
                self.plasticity_factor = max(0.05, self.plasticity_factor * 0.98)
            
            # Update metrics
            self.learning_metrics.plasticity_index = self.plasticity_factor
            self.learning_metrics.adaptation_speed = self.learning_metrics.learning_rate
            
            return {
                "learning_rate": self.learning_metrics.learning_rate,
                "plasticity_factor": self.plasticity_factor,
                "convergence_score": convergence_metrics["convergence_score"],
                "optimization_applied": True
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing learning parameters: {e}")
            return {"optimization_applied": False}
    
    def get_cultural_neutrality_assessment(self) -> Dict[str, float]:
        """Assess cultural neutrality of agent processing"""
        try:
            # Calculate balance across cultural contexts
            cultural_counts = dict(self.cultural_balance_tracker)
            total_signals = sum(cultural_counts.values()) if cultural_counts else 1
            
            # Calculate entropy (higher = more balanced)
            entropy = 0.0
            if total_signals > 0:
                for count in cultural_counts.values():
                    if count > 0:
                        p = count / total_signals
                        entropy -= p * math.log2(p)
            
            # Normalize entropy (max is log2(n) where n is number of cultures)
            max_entropy = math.log2(max(len(cultural_counts), 1))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
            # Calculate average connection neutrality
            connection_neutrality = 0.8  # Default
            if self.connected_agents:
                connection_neutrality = np.mean([
                    conn.cultural_neutrality for conn in self.connected_agents.values()
                ])
            
            # Overall cultural neutrality score
            overall_neutrality = (normalized_entropy * 0.4 + connection_neutrality * 0.6)
            
            # Update learning metrics
            self.learning_metrics.cultural_balance = overall_neutrality
            
            return {
                "overall_neutrality": overall_neutrality,
                "context_balance_entropy": normalized_entropy,
                "connection_neutrality": connection_neutrality,
                "cultural_contexts_seen": len(cultural_counts),
                "total_signals_processed": total_signals
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing cultural neutrality: {e}")
            return {"overall_neutrality": 0.8}
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get comprehensive status including learning metrics"""
        convergence_metrics = self.calculate_convergence_metrics()
        cultural_assessment = self.get_cultural_neutrality_assessment()
        
        return {
            "agent_id": self.agent_id,
            "layer": self.layer.value,
            "activation_level": self.activation_level,
            "is_active": self.is_active(),
            "connected_agents": len(self.connected_agents),
            "learning_metrics": {
                "learning_rate": self.learning_metrics.learning_rate,
                "plasticity_index": self.learning_metrics.plasticity_index,
                "convergence_score": convergence_metrics["convergence_score"],
                "stability_measure": convergence_metrics.get("stability", 0.5),
                "cultural_balance": cultural_assessment["overall_neutrality"],
                "adaptation_speed": self.learning_metrics.adaptation_speed
            },
            "performance_stats": self.signal_processing_stats,
            "convergence_status": convergence_metrics.get("status", "unknown"),
            "cultural_neutrality": cultural_assessment,
            "last_optimization": datetime.now().isoformat()
        }
    
    # === HELPER METHODS ===
    def _assess_cultural_neutrality(self, signal: NeuralSignal) -> None:
        """Assess and track cultural neutrality of incoming signals"""
        cultural_context = signal.cultural_context
        self.cultural_balance_tracker[cultural_context] += 1
        
        # Limit tracking history
        total_signals = sum(self.cultural_balance_tracker.values())
        if total_signals > 1000:
            # Scale down all counts to maintain proportion
            for context in self.cultural_balance_tracker:
                self.cultural_balance_tracker[context] *= 0.9
    
    def _apply_learning_updates(self, input_signal: NeuralSignal, output_signal: NeuralSignal) -> None:
        """Apply learning updates based on signal processing"""
        # Determine success based on output signal priority
        success = output_signal.priority > 0.5
        
        # Apply neural plasticity
        if input_signal.source_layer != self.layer:
            source_id = f"{input_signal.source_layer.value}_layer"
            cultural_balance = self.get_cultural_neutrality_assessment()["overall_neutrality"]
            self.apply_neural_plasticity(source_id, success, cultural_balance)
    
    def _update_convergence_tracking(self, input_signal: NeuralSignal, output_signal: NeuralSignal) -> None:
        """Update convergence tracking based on signal processing"""
        # Track convergence factor
        convergence_factor = (input_signal.priority + output_signal.priority) / 2.0
        output_signal.convergence_factor = convergence_factor
    
    def _update_performance_stats(self, success: bool, processing_time: float) -> None:
        """Update performance statistics"""
        stats = self.signal_processing_stats
        stats["processed"] += 1
        
        if success:
            stats["successful"] += 1
        else:
            stats["failed"] += 1
        
        # Update average latency
        stats["average_latency"] = (
            (stats["average_latency"] * (stats["processed"] - 1) + processing_time) /
            stats["processed"]
        )
    
    def _calculate_convergence_factor(self) -> float:
        """Calculate convergence factor for outgoing signals"""
        if len(self.convergence_history) < 2:
            return 0.5
        
        recent_change = abs(self.convergence_history[-1] - self.convergence_history[-2])
        return max(0.0, 1.0 - (recent_change * 10))  # Scale change
    
    def _calculate_trend_score(self, values: List[float]) -> float:
        """Calculate trend score for convergence analysis"""
        if len(values) < 3:
            return 0.5
        
        # Calculate slope of best fit line
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Convert slope to score (smaller absolute slope = better)
        trend_score = max(0.0, 1.0 - abs(slope) * 10)
        return min(1.0, trend_score)
    
    def _calculate_oscillation_score(self, values: List[float]) -> float:
        """Calculate oscillation score for convergence analysis"""
        if len(values) < 4:
            return 0.5
        
        # Count direction changes
        direction_changes = 0
        for i in range(2, len(values)):
            prev_diff = values[i-1] - values[i-2]
            curr_diff = values[i] - values[i-1]
            if (prev_diff > 0) != (curr_diff > 0):  # Direction change
                direction_changes += 1
        
        # Convert to score (fewer changes = better)
        max_changes = len(values) - 2
        oscillation_score = 1.0 - (direction_changes / max_changes) if max_changes > 0 else 1.0
        return max(0.0, min(1.0, oscillation_score))
    
    def is_active(self) -> bool:
        """Check if agent is active based on activation threshold"""
        return self.activation_level >= self.activation_threshold
    
    def update_state(self, updates: Dict[str, Any]):
        """Update agent's internal state"""
        self.state.update(updates)
    
    def reset(self):
        """Reset agent to initial state while preserving learned connections"""
        self.activation_level = 0.0
        self.state = {} 
        
        # Don't reset connection weights (preserve learning)
        # But reset usage statistics
        for connection in self.connected_agents.values():
            connection.usage_count = 0
        
        # Clear short-term tracking
        self.synchronization_signals.clear()
        self.performance_history.clear()
        
        self.logger.info(f"Agent {self.agent_id} reset (preserving learned connections)") 
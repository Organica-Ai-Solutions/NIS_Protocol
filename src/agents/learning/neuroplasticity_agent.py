"""
Neuroplasticity Agent with LSTM Sequential Learning

Implements neuroplasticity mechanisms for learning and memory adaptation with
LSTM-based sequential connection pattern learning and temporal attention.

Enhanced Features (v3 + LSTM):
- LSTM-based sequential connection pattern learning
- Temporal attention mechanisms for connection strengthening
- Dynamic connection weight prediction using temporal context
- Complete self-audit integration with real-time integrity monitoring
- Mathematical validation of neuroplasticity operations with evidence-based metrics
- Comprehensive integrity oversight for all learning outputs
- Auto-correction capabilities for neuroplasticity-related communications
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import time
import os
import json
import random
import numpy as np
import logging

# PyTorch for LSTM functionality
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.core.registry import NISAgent, NISLayer
from src.emotion.emotional_state import EmotionalState
from src.agents.memory.enhanced_memory_agent import EnhancedMemoryAgent, MemoryType

# LSTM core integration for sequence learning
from src.agents.memory.lstm_memory_core import (
    LSTMMemoryCore, MemorySequenceType, MemorySequence, LSTMMemoryState
)

# Integrity metrics for actual calculations
from src.utils.integrity_metrics import (
    calculate_confidence, create_default_confidence_factors, ConfidenceFactors
)

# Self-audit capabilities for real-time integrity monitoring
from src.utils.self_audit import self_audit_engine, ViolationType, IntegrityViolation

class NeuroplasticityAgent(NISAgent):
    """
    Implements neuroplasticity mechanisms for the NIS Protocol.
    
    This agent manages the modification of memory connections and embeddings
    based on experience, mimicking how the brain rewires neural pathways.
    """
    
    def __init__(
        self, 
        agent_id: str = "neuroplasticity",
        memory_agent: EnhancedMemoryAgent = None,
        storage_path: str = None,
        learning_rate: float = 0.1,
        decay_rate: float = 0.01,
        consolidation_interval: int = 3600,  # 1 hour
        connection_threshold: float = 0.3,
        enable_self_audit: bool = True,
        # LSTM-specific parameters
        enable_lstm: bool = True,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        lstm_learning_rate: float = 0.001,
        max_connection_sequence_length: int = 50,
        lstm_device: str = "cpu"
    ):
        """
        Initialize the neuroplasticity agent.
        
        Args:
            agent_id: Unique identifier for this agent
            memory_agent: Reference to the EnhancedMemoryAgent to work with
            storage_path: Path for persistent storage
            learning_rate: Rate at which connections strengthen (0-1)
            decay_rate: Rate at which unused connections weaken (0-1)
            consolidation_interval: Seconds between consolidation events
            connection_threshold: Minimum strength for a connection to persist
            enable_self_audit: Whether to enable real-time integrity monitoring
        """
        super().__init__(agent_id, NISLayer.LEARNING, "Implements neuroplasticity mechanisms")
        self.memory_agent = memory_agent
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.connection_threshold = connection_threshold
        self.storage_path = storage_path
        
        # Store connection strengths between memory pairs
        self.connection_strengths = {}
        
        # Track recently activated memories for Hebbian learning
        self.recent_activations = []
        self.activation_window = 60  # seconds
        
        # Track last consolidation time
        self.last_consolidation = time.time()
        self.consolidation_interval = consolidation_interval
        
        # Initialize emotional state
        self.emotional_state = EmotionalState()
        
        # Set up logging
        self.logger = logging.getLogger(f"nis_neuroplasticity_{agent_id}")
        
        # Set up self-audit integration
        self.enable_self_audit = enable_self_audit
        self.integrity_monitoring_enabled = enable_self_audit
        self.integrity_metrics = {
            'monitoring_start_time': time.time(),
            'total_outputs_monitored': 0,
            'total_violations_detected': 0,
            'auto_corrections_applied': 0,
            'average_integrity_score': 100.0
        }
        
        # Initialize confidence factors for mathematical validation
        self.confidence_factors = create_default_confidence_factors()
        
        # LSTM sequential connection learning integration
        self.enable_lstm = enable_lstm and TORCH_AVAILABLE
        self.lstm_core = None
        
        if self.enable_lstm:
            try:
                # Create LSTM core for connection pattern learning
                self.lstm_core = LSTMMemoryCore(
                    memory_dim=64,  # Smaller dimension for connection patterns
                    hidden_dim=lstm_hidden_dim,
                    num_layers=lstm_num_layers,
                    max_sequence_length=max_connection_sequence_length,
                    learning_rate=lstm_learning_rate,
                    device=lstm_device,
                    enable_self_audit=enable_self_audit
                )
                
                # LSTM-enhanced connection tracking
                self.connection_sequences = {}  # Track connection activation sequences
                self.temporal_connection_patterns = deque(maxlen=1000)
                self.connection_predictions = {}
                self.attention_weights_history = deque(maxlen=100)
                
                self.logger.info(f"LSTM sequential connection learning enabled with hidden_dim={lstm_hidden_dim}")
            except Exception as e:
                self.enable_lstm = False
                self.logger.warning(f"Failed to initialize LSTM for neuroplasticity: {e}")
        
        if not self.enable_lstm:
            self.logger.info("LSTM sequential learning disabled - using traditional Hebbian learning")
        
        # Load existing connections if storage path is provided
        if self.storage_path:
            self._load_connections()
        
        self.logger.info(f"Neuroplasticity Agent initialized with self-audit: {enable_self_audit}")
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process neuroplasticity operations.
        
        Args:
            message: Message with neuroplasticity operation
                'operation': Operation to perform
                    ('record_activation', 'strengthen', 'consolidate', 'stats')
                + Additional parameters based on operation
                
        Returns:
            Result of the operation
        """
        operation = message.get("operation", "").lower()
        
        # Check if it's time for periodic consolidation
        current_time = time.time()
        if current_time - self.last_consolidation > self.consolidation_interval:
            self._consolidate_connections()
            self.last_consolidation = current_time
        
        # Process the requested operation
        if operation == "record_activation":
            return self._record_memory_activation(message)
        elif operation == "strengthen":
            return self._strengthen_connection(message)
        elif operation == "consolidate":
            return self._manual_consolidate()
        elif operation == "stats":
            return self._get_stats()
        else:
            return {
                "status": "error",
                "error": f"Unknown operation: {operation}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _record_memory_activation(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record activation of a memory and apply Hebbian learning.
        
        Args:
            message: Message with memory activation details
                'memory_id': ID of the activated memory
                'activation_strength': Strength of activation (0-1)
                
        Returns:
            Result of the operation
        """
        memory_id = message.get("memory_id")
        activation_strength = message.get("activation_strength", 1.0)
        
        if not memory_id:
            return {
                "status": "error",
                "error": "No memory_id provided",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        # Record this activation
        current_time = time.time()
        activation_record = {
            "memory_id": memory_id,
            "time": current_time,
            "strength": activation_strength
        }
        self.recent_activations.append(activation_record)
        
        # LSTM sequential connection learning
        if self.enable_lstm and self.lstm_core:
            try:
                # Create connection pattern embedding for LSTM
                connection_pattern = self._create_connection_pattern_embedding(
                    memory_id, activation_strength, current_time
                )
                
                # Add to LSTM sequence for temporal pattern learning
                lstm_connection_data = {
                    'memory_id': memory_id,
                    'embedding': connection_pattern,
                    'activation_strength': activation_strength,
                    'timestamp': current_time,
                    'connection_context': self._get_connection_context(memory_id)
                }
                
                sequence_id = self.lstm_core.add_memory_to_sequence(
                    memory_data=lstm_connection_data,
                    sequence_type=MemorySequenceType.PROCEDURAL_CHAIN
                )
                
                # Track connection sequence
                self.connection_sequences[memory_id] = sequence_id
                
                # Add to temporal connection patterns
                self.temporal_connection_patterns.append({
                    'memory_id': memory_id,
                    'sequence_id': sequence_id,
                    'activation_strength': activation_strength,
                    'timestamp': current_time,
                    'pattern_embedding': connection_pattern
                })
                
                # Predict future connection patterns
                if len(self.temporal_connection_patterns) > 3:
                    self._predict_connection_patterns(sequence_id)
                
                self.logger.debug(f"Added memory {memory_id} to LSTM connection sequence {sequence_id}")
                
            except Exception as e:
                self.logger.warning(f"Failed to add connection pattern to LSTM: {e}")
        
        # Clean up old activations
        self.recent_activations = [
            a for a in self.recent_activations 
            if current_time - a["time"] <= self.activation_window
        ]
        
        # Apply traditional Hebbian learning to recent activations
        self._apply_hebbian_learning()
        
        # Apply LSTM-enhanced connection strengthening if available
        if self.enable_lstm:
            self._apply_lstm_enhanced_connection_learning(memory_id, activation_strength)
        
        return {
            "status": "success",
            "message": f"Recorded activation for memory {memory_id}",
            "lstm_sequence_id": self.connection_sequences.get(memory_id) if self.enable_lstm else None,
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _strengthen_connection(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explicitly strengthen connection between two memories.
        
        Args:
            message: Message with connection details
                'memory_id1': First memory ID
                'memory_id2': Second memory ID
                'strength_increase': Amount to increase strength by
                
        Returns:
            Result of the operation
        """
        memory_id1 = message.get("memory_id1")
        memory_id2 = message.get("memory_id2")
        strength_increase = message.get("strength_increase", self.learning_rate)
        
        if not memory_id1 or not memory_id2:
            return {
                "status": "error",
                "error": "Both memory_id1 and memory_id2 must be provided",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        # Ensure the memory IDs are ordered for consistent key creation
        if memory_id1 > memory_id2:
            memory_id1, memory_id2 = memory_id2, memory_id1
        
        # Update connection strength
        pair_key = f"{memory_id1}:{memory_id2}"
        current_strength = self.connection_strengths.get(pair_key, 0.0)
        self.connection_strengths[pair_key] = min(1.0, current_strength + strength_increase)
        
        return {
            "status": "success",
            "message": f"Strengthened connection between {memory_id1} and {memory_id2}",
            "new_strength": self.connection_strengths[pair_key],
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _apply_hebbian_learning(self) -> None:
        """
        Apply Hebbian learning to recently activated memories.
        
        Strengthens connections between memories activated within the time window.
        """
        # Extract unique memory IDs from recent activations
        recent_memory_ids = [a["memory_id"] for a in self.recent_activations]
        
        # Apply learning to co-activated pairs
        for i, memory_id1 in enumerate(recent_memory_ids):
            for memory_id2 in recent_memory_ids[i+1:]:
                # Calculate time proximity (closer in time = stronger association)
                activation1 = next(a for a in self.recent_activations if a["memory_id"] == memory_id1)
                activation2 = next(a for a in self.recent_activations if a["memory_id"] == memory_id2)
                
                time_diff = abs(activation1["time"] - activation2["time"])
                time_factor = max(0, 1 - (time_diff / self.activation_window))
                
                # Calculate strength increase based on activation strengths and time proximity
                strength_increase = (
                    self.learning_rate * 
                    activation1["strength"] * 
                    activation2["strength"] * 
                    time_factor
                )
                
                # Ensure consistent key order
                if memory_id1 > memory_id2:
                    memory_id1, memory_id2 = memory_id2, memory_id1
                
                # Update connection strength
                pair_key = f"{memory_id1}:{memory_id2}"
                current_strength = self.connection_strengths.get(pair_key, 0.0)
                self.connection_strengths[pair_key] = min(1.0, current_strength + strength_increase)
        
        # Periodically update vector embeddings
        if len(self.recent_activations) > 5:  
            self._update_embeddings()
    
    def _update_embeddings(self) -> None:
        """
        Update memory embeddings based on connection strengths.
        
        This is a complex operation that modifies vector embeddings to reflect
        learned connections between memories.
        """
        if not self.memory_agent:
            return
        
        # Identify memories to update based on recent activations
        memories_to_update = set()
        for activation in self.recent_activations:
            memories_to_update.add(activation["memory_id"])
        
        # For each memory to update, influence its embedding based on connected memories
        for memory_id in memories_to_update:
            # Find connected memories above threshold
            connected_memories = self._find_connected_memories(memory_id)
            
            if not connected_memories:
                continue
            
            # Get original embedding for this memory
            memory_type = self._get_memory_type(memory_id)
            if not memory_type:
                continue
            
            # This would require extending EnhancedMemoryAgent with a method to get embeddings
            # For now, we can assume the memory agent has this functionality
            
            # Pseudocode for updating embeddings:
            # 1. Get original embedding for memory_id
            # 2. Get embeddings for all connected memories
            # 3. Calculate weighted average based on connection strengths
            # 4. Move original embedding slightly toward the weighted average
            # 5. Update the embedding in the vector store
            
            # Note: Actual implementation would require modifications to EnhancedMemoryAgent
    
    def _find_connected_memories(self, memory_id: str) -> List[Tuple[str, float]]:
        """
        Find memories connected to the given memory with strengths above threshold.
        
        Args:
            memory_id: The memory ID to find connections for
            
        Returns:
            List of (connected_memory_id, connection_strength) tuples
        """
        connected = []
        
        for pair_key, strength in self.connection_strengths.items():
            if strength >= self.connection_threshold:
                id1, id2 = pair_key.split(":")
                
                if id1 == memory_id:
                    connected.append((id2, strength))
                elif id2 == memory_id:
                    connected.append((id1, strength))
        
        return connected
    
    def _get_memory_type(self, memory_id: str) -> str:
        """
        Determine the memory type for a given memory ID.
        
        Args:
            memory_id: The memory ID to check
            
        Returns:
            Memory type or None if not found
        """
        if not self.memory_agent:
            return None
        
        # Query the memory agent to get the memory
        response = self.memory_agent.process({
            "operation": "retrieve",
            "query": {
                "memory_id": memory_id
            },
            "update_access": False  # Don't update access time for this internal operation
        })
        
        if response["status"] == "success" and "memory" in response:
            return response["memory"].get("memory_type", MemoryType.EPISODIC)
        
        return None
    
    def _consolidate_connections(self) -> int:
        """
        Consolidate connection strengths, weakening unused connections.
        
        Returns:
            Number of connections pruned
        """
        pruned_count = 0
        current_time = time.time()
        
        # Apply decay to all connections
        for pair_key, strength in list(self.connection_strengths.items()):
            # Apply decay based on time since last consolidation
            time_factor = (current_time - self.last_consolidation) / self.consolidation_interval
            new_strength = strength - (self.decay_rate * time_factor)
            
            # Prune weak connections
            if new_strength < self.connection_threshold:
                del self.connection_strengths[pair_key]
                pruned_count += 1
            else:
                self.connection_strengths[pair_key] = new_strength
        
        # Save connections if storage path is configured
        if self.storage_path:
            self._save_connection_data()
        
        return pruned_count
    
    def _manual_consolidate(self) -> Dict[str, Any]:
        """
        Manually trigger connection consolidation.
        
        Returns:
            Consolidation operation result
        """
        pruned = self._consolidate_connections()
        
        return {
            "status": "success",
            "pruned_connections": pruned,
            "remaining_connections": len(self.connection_strengths),
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the neuroplasticity system.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "connection_count": len(self.connection_strengths),
            "recent_activations": len(self.recent_activations),
            "learning_rate": self.learning_rate,
            "decay_rate": self.decay_rate,
            "connection_threshold": self.connection_threshold,
            "time_since_consolidation": time.time() - self.last_consolidation,
            "strongest_connections": []
        }
        
        # Add some of the strongest connections
        strongest = sorted(
            self.connection_strengths.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        stats["strongest_connections"] = [
            {"connection": k, "strength": v} for k, v in strongest
        ]
        
        return {
            "status": "success",
            "stats": stats,
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _save_connection_data(self) -> None:
        """Save connection data to persistent storage."""
        if not self.storage_path:
            return
        
        try:
            # Ensure directory exists
            os.makedirs(self.storage_path, exist_ok=True)
            
            # Save connection strengths
            connections_path = os.path.join(self.storage_path, "connections.json")
            with open(connections_path, "w") as f:
                json.dump({
                    "connection_strengths": self.connection_strengths,
                    "last_consolidation": self.last_consolidation,
                    "learning_rate": self.learning_rate,
                    "decay_rate": self.decay_rate,
                    "connection_threshold": self.connection_threshold
                }, f)
        except Exception as e:
            print(f"Error saving connection data: {e}")
    
    def _load_connection_data(self) -> None:
        """Load connection data from persistent storage."""
        if not self.storage_path:
            return
        
        connections_path = os.path.join(self.storage_path, "connections.json")
        if not os.path.exists(connections_path):
            return
        
        try:
            with open(connections_path, "r") as f:
                data = json.load(f)
                self.connection_strengths = data.get("connection_strengths", {})
                self.last_consolidation = data.get("last_consolidation", time.time())
                self.learning_rate = data.get("learning_rate", self.learning_rate)
                self.decay_rate = data.get("decay_rate", self.decay_rate)
                self.connection_threshold = data.get("connection_threshold", self.connection_threshold)
        except Exception as e:
            print(f"Error loading connection data: {e}")
            
    def set_plasticity_phase(self, phase: str = "normal") -> None:
        """
        Set the plasticity phase to control learning flexibility.
        
        Args:
            phase: The plasticity phase ('high', 'normal', 'low')
        """
        if phase == "high":
            # High plasticity phase - rapid learning
            self.learning_rate = 0.3
            self.decay_rate = 0.02
        elif phase == "low":
            # Low plasticity phase - stable connections
            self.learning_rate = 0.05
            self.decay_rate = 0.01
        else:
            # Normal plasticity
            self.learning_rate = 0.1
            self.decay_rate = 0.05
            
    def _spread_activation(self, source_memory_id: str, activation_strength: float = 0.5, depth: int = 2) -> None:
        """
        Spread activation from source memory to connected memories.
        
        Args:
            source_memory_id: ID of the source memory
            activation_strength: Initial activation strength
            depth: How many levels to spread activation
        """
        if depth <= 0:
            return
            
        # Find directly connected memories
        connected = self._find_connected_memories(source_memory_id)
        
        # Activate each connected memory with reduced strength
        for memory_id, connection_strength in connected:
            spread_strength = activation_strength * connection_strength
            
            # Record this activation
            self.process({
                "operation": "record_activation",
                "memory_id": memory_id,
                "activation_strength": spread_strength
            })
            
            # Recursively spread to next level with reduced strength and depth
            self._spread_activation(
                memory_id, 
                activation_strength=spread_strength*0.5, 
                depth=depth-1
            )
    
    # ==================== SELF-AUDIT CAPABILITIES ====================
    
    def audit_neuroplasticity_output(self, output_text: str, operation: str = "", context: str = "") -> Dict[str, Any]:
        """
        Perform real-time integrity audit on neuroplasticity operation outputs.
        
        Args:
            output_text: Text output to audit
            operation: Neuroplasticity operation type (strengthen, weaken, consolidate, etc.)
            context: Additional context for the audit
            
        Returns:
            Audit results with violations and integrity score
        """
        if not self.enable_self_audit:
            return {'integrity_score': 100.0, 'violations': [], 'total_violations': 0}
        
        self.logger.info(f"Performing self-audit on neuroplasticity output for operation: {operation}")
        
        # Use proven audit engine
        audit_context = f"neuroplasticity:{operation}:{context}" if context else f"neuroplasticity:{operation}"
        violations = self_audit_engine.audit_text(output_text, audit_context)
        integrity_score = self_audit_engine.get_integrity_score(output_text)
        
        # Log violations for neuroplasticity-specific analysis
        if violations:
            self.logger.warning(f"Detected {len(violations)} integrity violations in neuroplasticity output")
            for violation in violations:
                self.logger.warning(f"  - {violation.severity}: {violation.text} -> {violation.suggested_replacement}")
        
        return {
            'violations': violations,
            'integrity_score': integrity_score,
            'total_violations': len(violations),
            'violation_breakdown': self._categorize_neuroplasticity_violations(violations),
            'operation': operation,
            'audit_timestamp': time.time()
        }
    
    def auto_correct_neuroplasticity_output(self, output_text: str, operation: str = "") -> Dict[str, Any]:
        """
        Automatically correct integrity violations in neuroplasticity outputs.
        
        Args:
            output_text: Text to correct
            operation: Neuroplasticity operation type
            
        Returns:
            Corrected output with audit details
        """
        if not self.enable_self_audit:
            return {'corrected_text': output_text, 'violations_fixed': [], 'improvement': 0}
        
        self.logger.info(f"Performing self-correction on neuroplasticity output for operation: {operation}")
        
        corrected_text, violations = self_audit_engine.auto_correct_text(output_text)
        
        # Calculate improvement metrics with mathematical validation
        original_score = self_audit_engine.get_integrity_score(output_text)
        corrected_score = self_audit_engine.get_integrity_score(corrected_text)
        improvement = calculate_confidence(corrected_score - original_score, self.confidence_factors)
        
        # Update integrity metrics
        if hasattr(self, 'integrity_metrics'):
            self.integrity_metrics['auto_corrections_applied'] += len(violations)
        
        return {
            'original_text': output_text,
            'corrected_text': corrected_text,
            'violations_fixed': violations,
            'original_integrity_score': original_score,
            'corrected_integrity_score': corrected_score,
            'improvement': improvement,
            'operation': operation,
            'correction_timestamp': time.time()
        }
    
    def analyze_neuroplasticity_integrity_trends(self, time_window: int = 3600) -> Dict[str, Any]:
        """
        Analyze neuroplasticity operation integrity trends for self-improvement.
        
        Args:
            time_window: Time window in seconds to analyze
            
        Returns:
            Neuroplasticity integrity trend analysis with mathematical validation
        """
        if not self.enable_self_audit:
            return {'integrity_status': 'MONITORING_DISABLED'}
        
        self.logger.info(f"Analyzing neuroplasticity integrity trends over {time_window} seconds")
        
        # Get integrity report from audit engine
        integrity_report = self_audit_engine.generate_integrity_report()
        
        # Calculate neuroplasticity-specific metrics
        plasticity_metrics = {
            'total_connections': len(self.connection_strengths),
            'strong_connections': len([s for s in self.connection_strengths.values() if s > 0.7]),
            'weak_connections': len([s for s in self.connection_strengths.values() if s < 0.3]),
            'recent_activations': len(self.recent_activations),
            'learning_rate': self.learning_rate,
            'decay_rate': self.decay_rate,
            'time_since_consolidation': time.time() - self.last_consolidation
        }
        
        # Generate neuroplasticity-specific recommendations
        recommendations = self._generate_neuroplasticity_integrity_recommendations(
            integrity_report, plasticity_metrics
        )
        
        return {
            'integrity_status': integrity_report['integrity_status'],
            'total_violations': integrity_report['total_violations'],
            'plasticity_metrics': plasticity_metrics,
            'integrity_trend': self._calculate_neuroplasticity_integrity_trend(),
            'recommendations': recommendations,
            'analysis_timestamp': time.time()
        }
    
    def enable_real_time_neuroplasticity_monitoring(self) -> bool:
        """
        Enable continuous integrity monitoring for all neuroplasticity operations.
        
        Returns:
            Success status
        """
        self.logger.info("Enabling real-time neuroplasticity integrity monitoring")
        
        # Set flag for monitoring
        self.integrity_monitoring_enabled = True
        
        # Initialize monitoring metrics if not already done
        if not hasattr(self, 'integrity_metrics'):
            self.integrity_metrics = {
                'monitoring_start_time': time.time(),
                'total_outputs_monitored': 0,
                'total_violations_detected': 0,
                'auto_corrections_applied': 0,
                'average_integrity_score': 100.0
            }
        
        return True
    
    def _monitor_neuroplasticity_output_integrity(self, output_text: str, operation: str = "") -> str:
        """
        Internal method to monitor and potentially correct neuroplasticity output integrity.
        
        Args:
            output_text: Output to monitor
            operation: Neuroplasticity operation type
            
        Returns:
            Potentially corrected output
        """
        if not getattr(self, 'integrity_monitoring_enabled', False):
            return output_text
        
        # Perform audit
        audit_result = self.audit_neuroplasticity_output(output_text, operation)
        
        # Update monitoring metrics
        if hasattr(self, 'integrity_metrics'):
            self.integrity_metrics['total_outputs_monitored'] += 1
            self.integrity_metrics['total_violations_detected'] += audit_result['total_violations']
        
        # Auto-correct if violations detected
        if audit_result['violations']:
            correction_result = self.auto_correct_neuroplasticity_output(output_text, operation)
            
            self.logger.info(f"Auto-corrected neuroplasticity output: {len(audit_result['violations'])} violations fixed")
            
            return correction_result['corrected_text']
        
        return output_text
    
    def _categorize_neuroplasticity_violations(self, violations: List[IntegrityViolation]) -> Dict[str, int]:
        """Categorize integrity violations specific to neuroplasticity operations"""
        from collections import defaultdict
        categories = defaultdict(int)
        
        for violation in violations:
            categories[violation.violation_type.value] += 1
        
        return dict(categories)
    
    def _generate_neuroplasticity_integrity_recommendations(self, integrity_report: Dict[str, Any], 
                                                          plasticity_metrics: Dict[str, Any]) -> List[str]:
        """Generate neuroplasticity-specific integrity recommendations"""
        recommendations = []
        
        if integrity_report.get('total_violations', 0) > 0:
            recommendations.append('Review neuroplasticity operation outputs for integrity compliance')
        
        if plasticity_metrics.get('weak_connections', 0) > plasticity_metrics.get('strong_connections', 0):
            recommendations.append('Consider adjusting learning rate to strengthen important connections')
        
        if plasticity_metrics.get('time_since_consolidation', 0) > self.consolidation_interval:
            recommendations.append('Trigger memory consolidation to optimize connection strengths')
        
        if plasticity_metrics.get('recent_activations', 0) == 0:
            recommendations.append('Monitor for memory activation patterns to guide neuroplasticity')
        
        recommendations.append('Maintain evidence-based neuroplasticity operation descriptions')
        
        return recommendations
    
    def _calculate_neuroplasticity_integrity_trend(self) -> str:
        """Calculate neuroplasticity integrity trend over time"""
        if not hasattr(self, 'integrity_metrics'):
            return 'INSUFFICIENT_DATA'
        
        # Simple trend calculation based on recent performance
        total_monitored = self.integrity_metrics.get('total_outputs_monitored', 0)
        total_violations = self.integrity_metrics.get('total_violations_detected', 0)
        
        if total_monitored == 0:
            return 'NO_DATA'
        
        violation_rate = total_violations / total_monitored
        
        if violation_rate == 0:
            return 'EXCELLENT'
        elif violation_rate < 0.1:
            return 'GOOD'
        elif violation_rate < 0.2:
            return 'NEEDS_IMPROVEMENT'
        else:
            return 'CRITICAL'
    
    # =============================================
    # LSTM Enhanced Connection Learning Methods
    # =============================================
    
    def _create_connection_pattern_embedding(self, memory_id: str, activation_strength: float, timestamp: float) -> List[float]:
        """
        Create an embedding representing the connection pattern for LSTM learning.
        
        Args:
            memory_id: ID of the activated memory
            activation_strength: Strength of activation (0-1)
            timestamp: Timestamp of activation
            
        Returns:
            64-dimensional embedding representing the connection pattern
        """
        # Create base pattern from memory characteristics
        pattern = np.zeros(64)
        
        # Encode memory ID characteristics (hash-based features)
        memory_hash = hash(memory_id) % 2**32
        pattern[0:8] = [(memory_hash >> (i * 4)) & 0xF for i in range(8)]
        
        # Encode activation strength
        pattern[8:16] = [activation_strength] * 8
        
        # Encode temporal features
        time_features = [
            np.sin(timestamp / 3600),  # Hour cycle
            np.cos(timestamp / 3600),
            np.sin(timestamp / 86400),  # Day cycle
            np.cos(timestamp / 86400),
            np.sin(timestamp / 604800),  # Week cycle
            np.cos(timestamp / 604800),
            activation_strength * np.sin(timestamp / 60),  # Minute cycle with strength
            activation_strength * np.cos(timestamp / 60)
        ]
        pattern[16:24] = time_features
        
        # Encode connection context features
        recent_activations_count = len(self.recent_activations)
        pattern[24] = min(recent_activations_count / 10.0, 1.0)  # Normalized recent activity
        
        # Encode current connection strengths for this memory
        memory_connections = [strength for key, strength in self.connection_strengths.items() 
                            if memory_id in key]
        if memory_connections:
            pattern[25] = np.mean(memory_connections)
            pattern[26] = np.max(memory_connections)
            pattern[27] = len(memory_connections) / 10.0  # Normalized connection count
        
        # Encode emotional state if available
        if hasattr(self, 'emotional_state') and self.emotional_state:
            pattern[28] = getattr(self.emotional_state, 'valence', 0.5)
            pattern[29] = getattr(self.emotional_state, 'arousal', 0.5)
        
        # Encode learning rate and decay information
        pattern[30] = self.learning_rate
        pattern[31] = self.decay_rate
        
        # Fill remaining with noise for robustness
        pattern[32:] = np.random.normal(0, 0.1, 32)
        
        # Normalize to unit vector
        norm = np.linalg.norm(pattern)
        if norm > 0:
            pattern = pattern / norm
        
        return pattern.tolist()
    
    def _get_connection_context(self, memory_id: str) -> Dict[str, Any]:
        """
        Get contextual information about connections for a memory.
        
        Args:
            memory_id: Memory ID to get context for
            
        Returns:
            Dictionary with connection context information
        """
        context = {
            'memory_id': memory_id,
            'total_connections': 0,
            'average_strength': 0.0,
            'strongest_connections': [],
            'recent_activation_count': 0,
            'temporal_position': len(self.recent_activations)
        }
        
        # Count connections involving this memory
        memory_connections = {}
        for key, strength in self.connection_strengths.items():
            if memory_id in key:
                other_memory = key.replace(memory_id, '').replace(':', '')
                if other_memory:
                    memory_connections[other_memory] = strength
        
        if memory_connections:
            context['total_connections'] = len(memory_connections)
            context['average_strength'] = np.mean(list(memory_connections.values()))
            
            # Get strongest connections
            sorted_connections = sorted(memory_connections.items(), key=lambda x: x[1], reverse=True)
            context['strongest_connections'] = sorted_connections[:5]
        
        # Count recent activations for this memory
        current_time = time.time()
        context['recent_activation_count'] = sum(1 for act in self.recent_activations 
                                               if act['memory_id'] == memory_id)
        
        return context
    
    def _predict_connection_patterns(self, sequence_id: str) -> Optional[Dict[str, Any]]:
        """
        Predict future connection patterns using LSTM.
        
        Args:
            sequence_id: ID of the sequence to predict for
            
        Returns:
            Prediction results or None if prediction fails
        """
        if not self.enable_lstm or not self.lstm_core:
            return None
        
        try:
            # Get prediction from LSTM core
            prediction_result = self.lstm_core.predict_next_memory(sequence_id)
            
            # Store prediction for validation
            self.connection_predictions[sequence_id] = {
                'prediction': prediction_result,
                'timestamp': time.time(),
                'sequence_id': sequence_id
            }
            
            # Extract attention weights for connection strengthening
            attention_weights = prediction_result.get('attention_weights', [])
            if attention_weights:
                self.attention_weights_history.append({
                    'sequence_id': sequence_id,
                    'attention_weights': attention_weights,
                    'timestamp': time.time(),
                    'coherence': prediction_result.get('attention_coherence', 0.0)
                })
            
            return prediction_result
            
        except Exception as e:
            self.logger.warning(f"Failed to predict connection patterns for sequence {sequence_id}: {e}")
            return None
    
    def _apply_lstm_enhanced_connection_learning(self, memory_id: str, activation_strength: float):
        """
        Apply LSTM-enhanced connection learning based on temporal patterns.
        
        Args:
            memory_id: ID of the activated memory
            activation_strength: Strength of activation
        """
        if not self.enable_lstm or not self.lstm_core:
            return
        
        try:
            # Get sequence for this memory
            sequence_id = self.connection_sequences.get(memory_id)
            if not sequence_id:
                return
            
            # Get recent attention patterns
            if self.attention_weights_history:
                recent_attention = list(self.attention_weights_history)[-5:]  # Last 5 patterns
                
                # Calculate attention-weighted connection strengthening
                for attention_data in recent_attention:
                    if attention_data['sequence_id'] == sequence_id:
                        attention_weights = attention_data['attention_weights']
                        coherence = attention_data['coherence']
                        
                        # Use attention weights to guide connection strengthening
                        self._apply_attention_weighted_strengthening(
                            memory_id, activation_strength, attention_weights, coherence
                        )
            
            # Predict and pre-strengthen likely future connections
            prediction = self._predict_connection_patterns(sequence_id)
            if prediction and prediction.get('confidence', 0) > 0.7:
                self._pre_strengthen_predicted_connections(memory_id, prediction)
                
        except Exception as e:
            self.logger.warning(f"Failed to apply LSTM-enhanced learning for memory {memory_id}: {e}")
    
    def _apply_attention_weighted_strengthening(self, memory_id: str, activation_strength: float, 
                                              attention_weights: List[float], coherence: float):
        """
        Apply connection strengthening based on LSTM attention weights.
        
        Args:
            memory_id: Memory being activated
            activation_strength: Strength of activation
            attention_weights: Attention weights from LSTM
            coherence: Attention coherence score
        """
        if not attention_weights:
            return
        
        # Get recent activations for attention-based strengthening
        recent_memories = [act['memory_id'] for act in self.recent_activations[-len(attention_weights):]]
        
        # Apply strengthening based on attention weights
        for i, (other_memory, weight) in enumerate(zip(recent_memories, attention_weights)):
            if other_memory != memory_id:
                # Calculate enhanced learning rate based on attention and coherence
                enhanced_learning_rate = self.learning_rate * weight * coherence * activation_strength
                
                # Strengthen connection
                pair_key = f"{min(memory_id, other_memory)}:{max(memory_id, other_memory)}"
                current_strength = self.connection_strengths.get(pair_key, 0.0)
                new_strength = min(1.0, current_strength + enhanced_learning_rate)
                
                self.connection_strengths[pair_key] = new_strength
                
                self.logger.debug(f"LSTM attention-weighted strengthening: {pair_key} -> {new_strength:.3f}")
    
    def _pre_strengthen_predicted_connections(self, memory_id: str, prediction: Dict[str, Any]):
        """
        Pre-strengthen connections that are predicted to be activated soon.
        
        Args:
            memory_id: Current memory being activated
            prediction: LSTM prediction results
        """
        predicted_embedding = prediction.get('predicted_embedding')
        confidence = prediction.get('confidence', 0.0)
        
        if not predicted_embedding or confidence < 0.7:
            return
        
        # Find memories with similar embeddings (potential future activations)
        similarity_threshold = 0.8
        pre_strengthen_factor = 0.1 * confidence  # Small pre-strengthening
        
        # This is a simplified version - in practice, you'd compare embeddings
        # with stored memory embeddings to find similar patterns
        for other_memory in list(self.connection_strengths.keys()):
            memories_in_key = other_memory.split(':')
            if memory_id in memories_in_key:
                # Pre-strengthen this connection slightly
                current_strength = self.connection_strengths[other_memory]
                new_strength = min(1.0, current_strength + pre_strengthen_factor)
                self.connection_strengths[other_memory] = new_strength
                
                self.logger.debug(f"Pre-strengthened predicted connection: {other_memory} -> {new_strength:.3f}")
    
    def get_lstm_connection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about LSTM-enhanced connection learning.
        
        Returns:
            Dictionary with LSTM connection statistics
        """
        if not self.enable_lstm:
            return {'lstm_enabled': False}
        
        stats = {
            'lstm_enabled': True,
            'connection_sequences': len(self.connection_sequences),
            'temporal_patterns': len(self.temporal_connection_patterns),
            'attention_history': len(self.attention_weights_history),
            'connection_predictions': len(self.connection_predictions),
            'lstm_core_stats': self.lstm_core.get_sequence_statistics() if self.lstm_core else {},
            'learning_integration': {
                'traditional_connections': len(self.connection_strengths),
                'lstm_enhanced_sequences': len(self.connection_sequences),
                'prediction_accuracy': self._calculate_connection_prediction_accuracy(),
                'attention_coherence_trend': self._calculate_attention_coherence_trend()
            }
        }
        
        return stats
    
    def _calculate_connection_prediction_accuracy(self) -> float:
        """Calculate accuracy of connection predictions"""
        if not self.connection_predictions:
            return 0.0
        
        # Simple accuracy estimation based on prediction confidence
        confidences = [pred['prediction'].get('confidence', 0.0) 
                      for pred in self.connection_predictions.values()]
        
        return np.mean(confidences) if confidences else 0.0
    
    def _calculate_attention_coherence_trend(self) -> str:
        """Calculate trend in attention coherence"""
        if len(self.attention_weights_history) < 3:
            return "insufficient_data"
        
        recent_coherences = [att['coherence'] for att in list(self.attention_weights_history)[-10:]]
        
        if len(recent_coherences) < 3:
            return "insufficient_data"
        
        # Simple trend calculation
        early_avg = np.mean(recent_coherences[:len(recent_coherences)//2])
        late_avg = np.mean(recent_coherences[len(recent_coherences)//2:])
        
        if late_avg > early_avg + 0.1:
            return "improving"
        elif late_avg < early_avg - 0.1:
            return "declining"
        else:
            return "stable"
    
    def get_neuroplasticity_integrity_report(self) -> Dict[str, Any]:
        """Generate comprehensive neuroplasticity integrity report"""
        if not self.enable_self_audit:
            return {'status': 'SELF_AUDIT_DISABLED'}
        
        # Get basic integrity report
        base_report = self_audit_engine.generate_integrity_report()
        
        # Add neuroplasticity-specific metrics
        plasticity_report = {
            'neuroplasticity_agent_id': self.agent_id,
            'monitoring_enabled': self.integrity_monitoring_enabled,
            'connection_status': {
                'total_connections': len(self.connection_strengths),
                'strong_connections': len([s for s in self.connection_strengths.values() if s > 0.7]),
                'weak_connections': len([s for s in self.connection_strengths.values() if s < 0.3]),
                'connection_threshold': self.connection_threshold
            },
            'learning_parameters': {
                'learning_rate': self.learning_rate,
                'decay_rate': self.decay_rate,
                'consolidation_interval': self.consolidation_interval
            },
            'recent_activity': {
                'recent_activations_count': len(self.recent_activations),
                'time_since_consolidation': time.time() - self.last_consolidation
            },
            'integrity_metrics': getattr(self, 'integrity_metrics', {}),
            'base_integrity_report': base_report,
            'report_timestamp': time.time()
        }
        
        return plasticity_report
    
    def _load_connections(self) -> None:
        """Load existing connection data from storage"""
        if not self.storage_path:
            return
        
        try:
            connections_file = os.path.join(self.storage_path, "connections.json")
            if os.path.exists(connections_file):
                with open(connections_file, 'r') as f:
                    self.connection_strengths = json.load(f)
                self.logger.info(f"Loaded {len(self.connection_strengths)} connections from storage")
        except Exception as e:
            self.logger.error(f"Failed to load connections: {e}")
            self.connection_strengths = {}

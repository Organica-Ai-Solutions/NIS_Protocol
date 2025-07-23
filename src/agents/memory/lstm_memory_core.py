"""
LSTM-Enhanced Memory Core for NIS Protocol

This module provides LSTM-based temporal memory modeling to enhance the existing
memory architecture with sequential learning, temporal attention, and dynamic
context management.

Enhanced Features:
- LSTM-based temporal sequence modeling for memory patterns
- Attention mechanisms for selective memory retrieval
- Dynamic context management with working memory integration
- Temporal consolidation with learned forgetting curves
- Integration with existing vector store infrastructure
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import json

# NIS Protocol imports
from src.utils.integrity_metrics import (
    calculate_confidence, create_default_confidence_factors, ConfidenceFactors
)
from src.utils.self_audit import self_audit_engine, ViolationType, IntegrityViolation


class MemorySequenceType(Enum):
    """Types of memory sequences for different temporal patterns"""
    EPISODIC_SEQUENCE = "episodic_sequence"      # Event sequences
    SEMANTIC_PATTERN = "semantic_pattern"        # Knowledge patterns
    PROCEDURAL_CHAIN = "procedural_chain"        # Action sequences
    CONTEXTUAL_FLOW = "contextual_flow"          # Context transitions
    ATTENTION_PATTERN = "attention_pattern"      # Attention sequences


@dataclass
class MemorySequence:
    """Represents a sequence of memories for LSTM processing"""
    sequence_id: str
    sequence_type: MemorySequenceType
    memories: List[Dict[str, Any]]
    temporal_order: List[float]  # Timestamps
    attention_weights: List[float]
    consolidation_level: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    importance: float = 0.5


@dataclass
class LSTMMemoryState:
    """Internal state of LSTM memory system"""
    hidden_state: torch.Tensor
    cell_state: torch.Tensor
    context_vector: torch.Tensor
    attention_history: List[torch.Tensor]
    sequence_position: int
    processing_mode: str


class TemporalAttentionMechanism(nn.Module):
    """Attention mechanism for temporal memory selection"""
    
    def __init__(self, hidden_dim: int, memory_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        
        # Attention layers
        self.attention_linear = nn.Linear(hidden_dim + memory_dim, 1)
        self.context_linear = nn.Linear(memory_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, hidden_state: torch.Tensor, memory_vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate attention weights and context vector
        
        Args:
            hidden_state: Current LSTM hidden state [batch_size, hidden_dim]
            memory_vectors: Memory embeddings [batch_size, seq_len, memory_dim]
            
        Returns:
            context_vector: Attended memory context [batch_size, hidden_dim]
            attention_weights: Attention weights [batch_size, seq_len]
        """
        batch_size, seq_len, memory_dim = memory_vectors.shape
        
        # Expand hidden state to match sequence length
        hidden_expanded = hidden_state.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate hidden state with each memory vector
        combined = torch.cat([hidden_expanded, memory_vectors], dim=-1)
        
        # Calculate attention scores
        attention_scores = self.attention_linear(combined).squeeze(-1)
        attention_weights = self.softmax(attention_scores)
        
        # Calculate context vector
        weighted_memories = torch.sum(memory_vectors * attention_weights.unsqueeze(-1), dim=1)
        context_vector = self.context_linear(weighted_memories)
        
        return context_vector, attention_weights


class LSTMMemoryNetwork(nn.Module):
    """LSTM network for temporal memory modeling"""
    
    def __init__(self, 
                 input_dim: int = 768,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = TemporalAttentionMechanism(lstm_output_dim, input_dim)
        
        # Output layers
        self.memory_predictor = nn.Linear(lstm_output_dim, input_dim)
        self.importance_predictor = nn.Linear(lstm_output_dim, 1)
        self.consolidation_predictor = nn.Linear(lstm_output_dim, 1)
        
        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, 
                memory_sequences: torch.Tensor,
                sequence_lengths: torch.Tensor,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through LSTM memory network
        
        Args:
            memory_sequences: Batch of memory sequences [batch_size, max_seq_len, input_dim]
            sequence_lengths: Actual sequence lengths [batch_size]
            initial_state: Optional initial LSTM state
            
        Returns:
            Dictionary containing predictions and attention weights
        """
        batch_size, max_seq_len, _ = memory_sequences.shape
        
        # Pack sequences for efficient LSTM processing
        packed_input = nn.utils.rnn.pack_padded_sequence(
            memory_sequences, sequence_lengths, batch_first=True, enforce_sorted=False
        )
        
        # LSTM forward pass
        packed_output, (hidden_state, cell_state) = self.lstm(packed_input, initial_state)
        
        # Unpack sequences
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Apply attention mechanism
        context_vector, attention_weights = self.attention(
            hidden_state[-1] if not self.bidirectional else torch.cat([hidden_state[-2], hidden_state[-1]], dim=-1),
            memory_sequences
        )
        
        # Predictions
        next_memory = self.memory_predictor(context_vector)
        importance_scores = self.sigmoid(self.importance_predictor(lstm_output))
        consolidation_scores = self.sigmoid(self.consolidation_predictor(lstm_output))
        
        return {
            'lstm_output': lstm_output,
            'hidden_state': hidden_state,
            'cell_state': cell_state,
            'context_vector': context_vector,
            'attention_weights': attention_weights,
            'next_memory_prediction': next_memory,
            'importance_scores': importance_scores,
            'consolidation_scores': consolidation_scores
        }


class LSTMMemoryCore:
    """
    Core LSTM-enhanced memory system for temporal sequence modeling
    """
    
    def __init__(self,
                 memory_dim: int = 768,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 max_sequence_length: int = 100,
                 learning_rate: float = 0.001,
                 device: str = "cpu",
                 enable_self_audit: bool = True):
        """
        Initialize LSTM memory core
        
        Args:
            memory_dim: Dimension of memory embeddings
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            max_sequence_length: Maximum sequence length for processing
            learning_rate: Learning rate for training
            device: Device for computation ('cpu' or 'cuda')
            enable_self_audit: Whether to enable integrity monitoring
        """
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length
        self.device = torch.device(device)
        
        # Initialize LSTM network
        self.lstm_network = LSTMMemoryNetwork(
            input_dim=memory_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.lstm_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Memory sequences storage
        self.memory_sequences: Dict[str, MemorySequence] = {}
        self.active_sequences: List[str] = []
        self.sequence_buffer: deque = deque(maxlen=1000)
        
        # Current LSTM state
        self.current_state: Optional[LSTMMemoryState] = None
        
        # Performance tracking
        self.training_history: List[Dict[str, float]] = []
        self.prediction_accuracy: deque = deque(maxlen=100)
        self.attention_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # Self-audit integration
        self.enable_self_audit = enable_self_audit
        self.integrity_metrics = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'attention_coherence': 0.0,
            'temporal_consistency': 0.0,
            'learning_stability': 0.0
        }
        
        # Initialize confidence factors
        self.confidence_factors = create_default_confidence_factors()
        
        # Logging
        self.logger = logging.getLogger("lstm_memory_core")
        self.logger.info(f"LSTM Memory Core initialized with dim={memory_dim}, hidden={hidden_dim}")
    
    def add_memory_to_sequence(self, 
                              memory_data: Dict[str, Any], 
                              sequence_type: MemorySequenceType = MemorySequenceType.EPISODIC_SEQUENCE) -> str:
        """
        Add a memory to a temporal sequence
        
        Args:
            memory_data: Memory data including embedding and metadata
            sequence_type: Type of sequence to add to
            
        Returns:
            Sequence ID that the memory was added to
        """
        # Extract memory embedding
        if 'embedding' not in memory_data:
            raise ValueError("Memory data must include embedding vector")
        
        embedding = memory_data['embedding']
        if isinstance(embedding, list):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        
        # Find or create appropriate sequence
        sequence_id = self._find_or_create_sequence(memory_data, sequence_type)
        
        # Add memory to sequence
        sequence = self.memory_sequences[sequence_id]
        sequence.memories.append(memory_data)
        sequence.temporal_order.append(time.time())
        sequence.attention_weights.append(memory_data.get('importance', 0.5))
        sequence.last_accessed = time.time()
        
        # Update sequence buffer for training
        self.sequence_buffer.append({
            'sequence_id': sequence_id,
            'memory_data': memory_data,
            'timestamp': time.time()
        })
        
        # Trigger learning if sequence is long enough
        if len(sequence.memories) >= 3:
            self._update_sequence_learning(sequence_id)
        
        self.logger.debug(f"Added memory to sequence {sequence_id}")
        return sequence_id
    
    def predict_next_memory(self, 
                           sequence_id: str, 
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Predict the next memory in a sequence using LSTM
        
        Args:
            sequence_id: ID of the sequence to predict for
            context: Optional context for prediction
            
        Returns:
            Prediction results with confidence scores
        """
        if sequence_id not in self.memory_sequences:
            raise ValueError(f"Sequence {sequence_id} not found")
        
        sequence = self.memory_sequences[sequence_id]
        
        # Prepare sequence for prediction
        memory_tensors = []
        for memory in sequence.memories[-self.max_sequence_length:]:
            embedding = memory.get('embedding', [])
            if isinstance(embedding, list):
                embedding = torch.tensor(embedding, dtype=torch.float32)
            memory_tensors.append(embedding)
        
        if not memory_tensors:
            return {'error': 'No memories in sequence for prediction'}
        
        # Stack tensors and add batch dimension
        sequence_tensor = torch.stack(memory_tensors).unsqueeze(0).to(self.device)
        sequence_lengths = torch.tensor([len(memory_tensors)], dtype=torch.long)
        
        # Forward pass
        self.lstm_network.eval()
        with torch.no_grad():
            results = self.lstm_network(sequence_tensor, sequence_lengths)
        
        # Extract predictions
        next_memory_embedding = results['next_memory_prediction'].cpu().numpy()[0]
        attention_weights = results['attention_weights'].cpu().numpy()[0]
        
        # Calculate confidence based on attention coherence
        attention_entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-8))
        attention_coherence = 1.0 / (1.0 + attention_entropy)
        
        # Update state
        self.current_state = LSTMMemoryState(
            hidden_state=results['hidden_state'],
            cell_state=results['cell_state'],
            context_vector=results['context_vector'],
            attention_history=[results['attention_weights']],
            sequence_position=len(sequence.memories),
            processing_mode='prediction'
        )
        
        # Calculate overall confidence
        confidence = calculate_confidence(
            factors=self.confidence_factors,
            **{
                'attention_coherence': attention_coherence,
                'sequence_length': len(sequence.memories),
                'temporal_consistency': self._calculate_temporal_consistency(sequence),
                'prediction_stability': self._calculate_prediction_stability()
            }
        )
        
        # Apply self-audit monitoring
        prediction_result = {
            'predicted_embedding': next_memory_embedding,
            'attention_weights': attention_weights.tolist(),
            'attention_coherence': attention_coherence,
            'confidence': confidence,
            'sequence_length': len(sequence.memories),
            'temporal_position': sequence.sequence_position if hasattr(sequence, 'sequence_position') else 0,
            'processing_metadata': {
                'lstm_hidden_dim': self.hidden_dim,
                'sequence_type': sequence.sequence_type.value,
                'prediction_timestamp': time.time()
            }
        }
        
        if self.enable_self_audit:
            prediction_result = self._monitor_prediction_integrity(prediction_result, sequence_id)
        
        # Update metrics
        self.integrity_metrics['total_predictions'] += 1
        self.integrity_metrics['attention_coherence'] = attention_coherence
        
        return prediction_result
    
    def _find_or_create_sequence(self, 
                                memory_data: Dict[str, Any], 
                                sequence_type: MemorySequenceType) -> str:
        """Find existing sequence or create new one for memory"""
        # Simple sequence creation for now - can be enhanced with similarity matching
        sequence_id = f"{sequence_type.value}_{int(time.time() * 1000)}"
        
        if sequence_id not in self.memory_sequences:
            self.memory_sequences[sequence_id] = MemorySequence(
                sequence_id=sequence_id,
                sequence_type=sequence_type,
                memories=[],
                temporal_order=[],
                attention_weights=[],
                consolidation_level=0.0,
                last_accessed=time.time(),
                access_count=0,
                importance=memory_data.get('importance', 0.5)
            )
        
        return sequence_id
    
    def _update_sequence_learning(self, sequence_id: str):
        """Update LSTM learning based on sequence"""
        sequence = self.memory_sequences[sequence_id]
        
        if len(sequence.memories) < 3:
            return
        
        # Prepare training data
        memory_tensors = []
        for memory in sequence.memories:
            embedding = memory.get('embedding', [])
            if isinstance(embedding, list):
                embedding = torch.tensor(embedding, dtype=torch.float32)
            memory_tensors.append(embedding)
        
        if len(memory_tensors) < 3:
            return
        
        # Create input-target pairs
        inputs = torch.stack(memory_tensors[:-1]).unsqueeze(0).to(self.device)
        targets = torch.stack(memory_tensors[1:]).unsqueeze(0).to(self.device)
        sequence_lengths = torch.tensor([len(memory_tensors) - 1], dtype=torch.long)
        
        # Training step
        self.lstm_network.train()
        self.optimizer.zero_grad()
        
        results = self.lstm_network(inputs, sequence_lengths)
        predicted_memories = results['next_memory_prediction'].unsqueeze(1).expand(-1, targets.shape[1], -1)
        
        loss = self.criterion(predicted_memories, targets)
        loss.backward()
        self.optimizer.step()
        
        # Track training
        self.training_history.append({
            'sequence_id': sequence_id,
            'loss': loss.item(),
            'sequence_length': len(memory_tensors),
            'timestamp': time.time()
        })
        
        self.logger.debug(f"Updated learning for sequence {sequence_id}, loss: {loss.item():.4f}")
    
    def _calculate_temporal_consistency(self, sequence: MemorySequence) -> float:
        """Calculate temporal consistency score for sequence"""
        if len(sequence.temporal_order) < 2:
            return 1.0
        
        # Check if timestamps are monotonically increasing
        temporal_diffs = np.diff(sequence.temporal_order)
        consistency = np.mean(temporal_diffs > 0)
        
        return float(consistency)
    
    def _calculate_prediction_stability(self) -> float:
        """Calculate prediction stability based on recent predictions"""
        if len(self.prediction_accuracy) < 2:
            return 0.5
        
        recent_accuracies = list(self.prediction_accuracy)[-10:]
        stability = 1.0 - np.std(recent_accuracies)
        
        return max(0.0, min(1.0, stability))
    
    def _monitor_prediction_integrity(self, 
                                    prediction_result: Dict[str, Any], 
                                    sequence_id: str) -> Dict[str, Any]:
        """Monitor prediction integrity using self-audit system"""
        try:
            # Check for common integrity issues
            violations = []
            
            # Check prediction confidence bounds
            confidence = prediction_result.get('confidence', 0.0)
            if confidence < 0.0 or confidence > 1.0:
                violations.append(IntegrityViolation(
                    violation_type=ViolationType.INVALID_METRIC,
                    description=f"Confidence score {confidence} outside valid range [0,1]",
                    severity="HIGH",
                    context={"sequence_id": sequence_id, "confidence": confidence}
                ))
            
            # Check attention weights coherence
            attention_coherence = prediction_result.get('attention_coherence', 0.0)
            if attention_coherence < 0.1:  # Very low coherence indicates potential issues
                violations.append(IntegrityViolation(
                    violation_type=ViolationType.PERFORMANCE_CLAIM,
                    description=f"Very low attention coherence {attention_coherence}",
                    severity="MEDIUM",
                    context={"sequence_id": sequence_id, "coherence": attention_coherence}
                ))
            
            # Apply corrections if violations found
            if violations:
                self.integrity_metrics['total_violations_detected'] += len(violations)
                
                for violation in violations:
                    if violation.violation_type == ViolationType.INVALID_METRIC:
                        # Clamp confidence to valid range
                        prediction_result['confidence'] = max(0.0, min(1.0, confidence))
                        self.integrity_metrics['auto_corrections_applied'] += 1
            
            # Update integrity metrics
            self.integrity_metrics['total_outputs_monitored'] += 1
            
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"Error in prediction integrity monitoring: {e}")
            return prediction_result
    
    def get_sequence_statistics(self) -> Dict[str, Any]:
        """Get statistics about memory sequences and LSTM performance"""
        total_sequences = len(self.memory_sequences)
        total_memories = sum(len(seq.memories) for seq in self.memory_sequences.values())
        
        avg_sequence_length = total_memories / max(total_sequences, 1)
        
        # Calculate average attention coherence
        avg_attention_coherence = np.mean([
            pattern[-1] if pattern else 0.0 
            for pattern in self.attention_patterns.values()
        ]) if self.attention_patterns else 0.0
        
        # Calculate learning progress
        recent_losses = [h['loss'] for h in self.training_history[-10:]] if self.training_history else []
        avg_recent_loss = np.mean(recent_losses) if recent_losses else 0.0
        
        return {
            'total_sequences': total_sequences,
            'total_memories': total_memories,
            'average_sequence_length': avg_sequence_length,
            'average_attention_coherence': avg_attention_coherence,
            'average_recent_loss': avg_recent_loss,
            'prediction_accuracy': np.mean(list(self.prediction_accuracy)) if self.prediction_accuracy else 0.0,
            'lstm_parameters': sum(p.numel() for p in self.lstm_network.parameters()),
            'integrity_metrics': self.integrity_metrics,
            'active_sequences': len(self.active_sequences)
        }
    
    def consolidate_sequences(self, consolidation_threshold: float = 0.8) -> Dict[str, Any]:
        """Consolidate sequences based on importance and access patterns"""
        consolidated_count = 0
        pruned_count = 0
        
        current_time = time.time()
        
        for sequence_id, sequence in list(self.memory_sequences.items()):
            # Calculate consolidation score
            time_factor = min(1.0, (current_time - sequence.last_accessed) / 3600)  # 1 hour normalization
            access_factor = min(1.0, sequence.access_count / 10)  # 10 access normalization
            importance_factor = sequence.importance
            
            consolidation_score = (time_factor + access_factor + importance_factor) / 3
            
            if consolidation_score >= consolidation_threshold:
                # Mark for consolidation
                sequence.consolidation_level = consolidation_score
                consolidated_count += 1
            elif consolidation_score < 0.2 and len(sequence.memories) < 2:
                # Prune very low-value sequences
                del self.memory_sequences[sequence_id]
                pruned_count += 1
        
        self.logger.info(f"Consolidated {consolidated_count} sequences, pruned {pruned_count}")
        
        return {
            'consolidated_sequences': consolidated_count,
            'pruned_sequences': pruned_count,
            'total_remaining': len(self.memory_sequences),
            'consolidation_timestamp': current_time
        } 
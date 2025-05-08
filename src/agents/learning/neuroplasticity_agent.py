"""
Neuroplasticity Agent

Implements neuroplasticity mechanisms for learning and memory adaptation.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import time
import os
import json
import random
import numpy as np

from src.core.registry import NISAgent, NISLayer
from src.emotion.emotional_state import EmotionalState
from src.agents.memory.enhanced_memory_agent import EnhancedMemoryAgent, MemoryType

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
        connection_threshold: float = 0.3
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
        
        # Load existing connection data if available
        if storage_path:
            self._load_connection_data()
    
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
        self.recent_activations.append({
            "memory_id": memory_id,
            "time": current_time,
            "strength": activation_strength
        })
        
        # Clean up old activations
        self.recent_activations = [
            a for a in self.recent_activations 
            if current_time - a["time"] <= self.activation_window
        ]
        
        # Apply Hebbian learning to recent activations
        self._apply_hebbian_learning()
        
        return {
            "status": "success",
            "message": f"Recorded activation for memory {memory_id}",
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

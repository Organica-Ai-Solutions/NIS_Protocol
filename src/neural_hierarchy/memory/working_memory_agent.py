from typing import Optional, Dict, Any, List, Tuple
from ..base_neural_agent import NeuralAgent, NeuralLayer, NeuralSignal
from collections import deque
import numpy as np
from datetime import datetime, timedelta

class MemoryItem:
    def __init__(
        self,
        content: Any,
        importance: float = 1.0,
        lifetime: int = 300  # 5 minutes in seconds
    ):
        self.content = content
        self.importance = importance
        self.creation_time = datetime.now()
        self.access_count = 0
        self.last_access = self.creation_time
        self.lifetime = lifetime
        self.activation = 1.0
    
    def access(self):
        """Record memory access"""
        self.access_count += 1
        self.last_access = datetime.now()
        self.activation = 1.0  # Reset activation
    
    def update_activation(self):
        """Update activation based on time decay"""
        time_since_access = (datetime.now() - self.last_access).total_seconds()
        self.activation = np.exp(-time_since_access / self.lifetime)
        return self.activation
    
    def is_active(self, threshold: float = 0.1) -> bool:
        """Check if memory is still active"""
        return self.update_activation() >= threshold

class WorkingMemoryAgent(NeuralAgent):
    """Agent for managing working memory"""
    
    def __init__(
        self,
        agent_id: str = "working_memory",
        capacity: int = 7,  # Miller's Law: 7 ± 2 items
        activation_threshold: float = 0.1
    ):
        super().__init__(
            agent_id=agent_id,
            layer=NeuralLayer.MEMORY,
            description="Manages working memory"
        )
        
        self.capacity = capacity
        self.activation_threshold = activation_threshold
        
        # Memory structures
        self.active_memories: deque = deque(maxlen=capacity)
        self.recent_memories: List[MemoryItem] = []
        self.associations: Dict[str, List[str]] = {}
    
    def process_signal(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Process incoming signal and manage working memory"""
        # Clean up inactive memories
        self._cleanup_memories()
        
        # Process new information
        if isinstance(signal.content, dict):
            # Create memory item
            memory = MemoryItem(
                content=signal.content,
                importance=signal.priority
            )
            
            # Add to active memories
            self.active_memories.append(memory)
            
            # Create associations
            if 'recognized_patterns' in signal.content:
                self._create_associations(memory, signal.content['recognized_patterns'])
            
            # Generate signal for emotional layer
            return NeuralSignal(
                source_layer=self.layer,
                target_layer=NeuralLayer.EMOTIONAL,
                content={
                    'active_memories': self._get_active_memory_contents(),
                    'associations': self._get_relevant_associations(memory)
                },
                priority=signal.priority
            )
        
        return None
    
    def _cleanup_memories(self):
        """Remove inactive memories"""
        # Update active memories
        active = deque(
            mem for mem in self.active_memories
            if mem.is_active(self.activation_threshold)
        )
        
        # Move inactive to recent memories
        inactive = [
            mem for mem in self.active_memories
            if not mem.is_active(self.activation_threshold)
        ]
        self.recent_memories.extend(inactive)
        
        # Limit recent memories
        if len(self.recent_memories) > 100:
            self.recent_memories = self.recent_memories[-100:]
        
        self.active_memories = active
    
    def _create_associations(self, memory: MemoryItem, patterns: List[Dict]):
        """Create associations between patterns"""
        for pattern in patterns:
            pattern_id = pattern['pattern_type']
            if pattern_id not in self.associations:
                self.associations[pattern_id] = []
            self.associations[pattern_id].append(memory.content.get('original_text', ''))
    
    def _get_active_memory_contents(self) -> List[Dict]:
        """Get contents of active memories"""
        return [
            {
                'content': mem.content,
                'importance': mem.importance,
                'activation': mem.activation,
                'age': (datetime.now() - mem.creation_time).total_seconds()
            }
            for mem in self.active_memories
        ]
    
    def _get_relevant_associations(self, memory: MemoryItem) -> Dict[str, List[str]]:
        """Get associations relevant to current memory"""
        relevant = {}
        if isinstance(memory.content, dict) and 'recognized_patterns' in memory.content:
            for pattern in memory.content['recognized_patterns']:
                pattern_id = pattern['pattern_type']
                if pattern_id in self.associations:
                    relevant[pattern_id] = self.associations[pattern_id]
        return relevant
    
    def get_memory_state(self) -> Dict[str, Any]:
        """Get current state of working memory"""
        return {
            'active_count': len(self.active_memories),
            'recent_count': len(self.recent_memories),
            'association_count': len(self.associations),
            'active_memories': self._get_active_memory_contents(),
            'activation_levels': [
                mem.activation for mem in self.active_memories
            ]
        }
    
    def reset(self):
        """Reset memory state"""
        super().reset()
        self.active_memories.clear()
        self.recent_memories = []
        # Keep associations for learning 
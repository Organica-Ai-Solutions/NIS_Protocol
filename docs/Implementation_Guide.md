# NIS Protocol Implementation Guide

This guide provides detailed instructions for implementing the Neuro-Inspired System Protocol in your applications.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Agent Implementation](#agent-implementation)
3. [Emotional State System](#emotional-state-system)
4. [Memory Management](#memory-management)
5. [Communication Flow](#communication-flow)
6. [Testing and Validation](#testing-and-validation)

## System Architecture

The NIS Protocol is organized into cognitive layers, each with specialized agents:

### Layer Structure

```
NIS Protocol
│
├── Perception Layer
│   ├── Vision Agent
│   └── Input Agent
│
├── Interpretation Layer
│   ├── Parser Agent
│   └── Intent Agent
│
├── Memory Layer
│   ├── Memory Agent
│   └── Log Agent
│
├── Emotional Layer
│   └── Emotion Agent
│
├── Reasoning Layer
│   ├── Cortex Agent
│   └── Planning Agent
│
└── Action Layer
    ├── Builder Agent
    └── Deployer Agent
```

## Agent Implementation

Each agent in the NIS Protocol should follow this implementation pattern:

```python
from nis_protocol.core import NISAgent, NISLayer

class CustomAgent(NISAgent):
    """Custom agent implementation."""
    
    def __init__(self, agent_id: str, description: str):
        super().__init__(
            agent_id=agent_id,
            layer=NISLayer.REASONING,  # Set the appropriate layer
            description=description
        )
        self.custom_state = {}
        
    def process(self, message: dict) -> dict:
        """Process incoming messages.
        
        Args:
            message: The incoming message
            
        Returns:
            Processed message
        """
        # Extract information from the message
        input_data = message.get("payload", {})
        
        # Process the data
        result = self._custom_processing(input_data)
        
        # Update the emotional state if needed
        self._update_emotional_state(result)
        
        # Return the processed message
        return {
            "agent_id": self.agent_id,
            "timestamp": time.time(),
            "status": "success",
            "payload": result,
            "metadata": {
                "processing_time": self.processing_time,
            },
            "emotional_state": message.get("emotional_state", {})
        }
        
    def _custom_processing(self, data: dict) -> dict:
        """Custom processing logic.
        
        Override this method with your specific implementation.
        """
        raise NotImplementedError("Subclasses must implement _custom_processing()")
        
    def _update_emotional_state(self, result: dict) -> None:
        """Update the emotional state based on processing results."""
        # Example: If confidence is low, increase suspicion
        if result.get("confidence", 1.0) < 0.5:
            emotional_state = self.registry.get_emotional_state()
            emotional_state.update("suspicion", 0.7)
```

## Emotional State System

The Emotional State System modulates agent behavior based on context-sensitive dimensions.

### Implementation Example

```python
class EmotionalStateSystem:
    """Emotional state management for NIS Protocol."""
    
    def __init__(self):
        self.state = {
            "suspicion": 0.5,   # Default neutral state
            "urgency": 0.5,
            "confidence": 0.5,
            "interest": 0.5,
            "novelty": 0.5
        }
        self.decay_rates = {
            "suspicion": 0.05,
            "urgency": 0.1,
            "confidence": 0.03,
            "interest": 0.07,
            "novelty": 0.2
        }
        self.last_update = time.time()
        
    def update(self, dimension: str, value: float) -> None:
        """Update an emotional dimension.
        
        Args:
            dimension: The emotional dimension to update
            value: The new value (0.0 to 1.0)
        """
        if dimension in self.state:
            self.state[dimension] = max(0.0, min(1.0, value))
            self.last_update = time.time()
        
    def get_state(self) -> dict:
        """Get the current emotional state."""
        self._apply_decay()
        return self.state.copy()
        
    def _apply_decay(self) -> None:
        """Apply time-based decay to emotional dimensions."""
        current_time = time.time()
        elapsed = current_time - self.last_update
        
        for dimension, value in self.state.items():
            decay_rate = self.decay_rates.get(dimension, 0.05)
            decay_amount = decay_rate * elapsed
            
            # Move toward neutral (0.5)
            if value > 0.5:
                self.state[dimension] = max(0.5, value - decay_amount)
            elif value < 0.5:
                self.state[dimension] = min(0.5, value + decay_amount)
```

## Memory Management

The Memory system stores information for future reference and retrieval.

### Implementation Example

```python
class MemoryManager:
    """Memory management for NIS Protocol."""
    
    def __init__(self, storage_backend=None):
        self.storage = storage_backend or InMemoryStorage()
        
    def store(self, key: str, data: dict, ttl: int = None) -> None:
        """Store data in memory.
        
        Args:
            key: The unique key for the data
            data: The data to store
            ttl: Time-to-live in seconds (None for permanent)
        """
        self.storage.set(key, data, ttl)
        
    def retrieve(self, key: str) -> dict:
        """Retrieve data from memory.
        
        Args:
            key: The unique key for the data
            
        Returns:
            The stored data or None if not found
        """
        return self.storage.get(key)
        
    def search(self, pattern: dict) -> list:
        """Search memory for matching patterns.
        
        Args:
            pattern: Dictionary with search patterns
            
        Returns:
            List of matching memory items
        """
        return self.storage.search(pattern)
        
    def forget(self, key: str) -> bool:
        """Remove data from memory.
        
        Args:
            key: The unique key for the data
            
        Returns:
            True if successful, False otherwise
        """
        return self.storage.delete(key)
```

## Communication Flow

The NIS Protocol defines a standardized message format for inter-agent communication.

### Message Format

```json
{
  "agent_id": "vision_1",
  "timestamp": 1621435234.567,
  "status": "success",
  "payload": {
    "object_detected": "vehicle",
    "confidence": 0.92,
    "location": [120, 340, 220, 380]
  },
  "metadata": {
    "processing_time": 0.032,
    "input_source": "camera_1"
  },
  "emotional_state": {
    "suspicion": 0.3,
    "urgency": 0.7,
    "confidence": 0.8,
    "interest": 0.5,
    "novelty": 0.2
  }
}
```

## Testing and Validation

Implement comprehensive testing to ensure your NIS Protocol implementation functions correctly.

### Unit Tests

Test each agent individually to ensure it processes messages correctly:

```python
def test_vision_agent():
    agent = VisionAgent("test_vision", "Test agent")
    
    input_message = {
        "payload": {"image_data": "base64_encoded_image"}
    }
    
    result = agent.process(input_message)
    
    assert result["status"] == "success"
    assert "object_detected" in result["payload"]
    assert "confidence" in result["payload"]
```

### Integration Tests

Test the entire system to ensure agents communicate correctly:

```python
def test_end_to_end_processing():
    registry = NISRegistry()
    
    vision = VisionAgent("vision", "Test vision")
    memory = MemoryAgent("memory", "Test memory")
    cortex = CortexAgent("cortex", "Test cortex")
    action = ActionAgent("action", "Test action")
    
    input_data = {"image_data": "base64_encoded_image"}
    
    perception_result = vision.process({"payload": input_data})
    memory_result = memory.process(perception_result)
    reasoning_result = cortex.process(memory_result)
    action_result = action.process(reasoning_result)
    
    assert action_result["status"] == "success"
    assert "action_taken" in action_result["payload"]
```

For more information, refer to the [API Reference](API_Reference.md) and [NIS Protocol Whitepaper](NIS_Protocol_Whitepaper.md). 
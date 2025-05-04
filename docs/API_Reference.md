# NIS Protocol API Reference

This document provides a comprehensive reference for the NIS Protocol API.

## Core Components

### NISRegistry

The central registry for all NIS Protocol agents.

```python
class NISRegistry:
    def __init__(self)
    def register(self, agent: NISAgent) -> None
    def get_agents_by_layer(self, layer: NISLayer) -> List[NISAgent]
    def get_agent_by_id(self, agent_id: str) -> Optional[NISAgent]
    def get_emotional_state(self) -> EmotionalStateSystem
    def set_emotional_state(self, emotional_state: EmotionalStateSystem) -> None
```

### NISLayer

Enumeration of cognitive layers in the NIS Protocol.

```python
class NISLayer(Enum):
    PERCEPTION = "perception"
    INTERPRETATION = "interpretation"
    MEMORY = "memory"
    EMOTION = "emotion"
    REASONING = "reasoning"
    ACTION = "action"
    LEARNING = "learning"
    COORDINATION = "coordination"
```

### NISAgent

Base class for all NIS Protocol agents.

```python
class NISAgent:
    def __init__(self, agent_id: str, layer: NISLayer, description: str)
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]
    def get_id(self) -> str
    def get_layer(self) -> NISLayer
    def get_description(self) -> str
    def is_active(self) -> bool
    def set_active(self, active: bool) -> None
```

## Perception Layer

### VisionAgent

Agent for processing visual inputs.

```python
class VisionAgent(NISAgent):
    def __init__(self, agent_id: str, description: str)
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]
    def detect_objects(self, image_data: bytes) -> List[Dict[str, Any]]
    def classify_image(self, image_data: bytes) -> Dict[str, float]
```

### InputAgent

Agent for processing non-visual inputs.

```python
class InputAgent(NISAgent):
    def __init__(self, agent_id: str, description: str)
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]
    def parse_text(self, text: str) -> Dict[str, Any]
    def process_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]
```

## Interpretation Layer

### ParserAgent

Agent for structuring and formatting raw data.

```python
class ParserAgent(NISAgent):
    def __init__(self, agent_id: str, description: str)
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]
    def parse_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]
```

### IntentAgent

Agent for determining the purpose or goal behind inputs.

```python
class IntentAgent(NISAgent):
    def __init__(self, agent_id: str, description: str)
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]
    def detect_intent(self, payload: Dict[str, Any]) -> Dict[str, Any]
```

## Memory Layer

### MemoryAgent

Agent for storing and retrieving information.

```python
class MemoryAgent(NISAgent):
    def __init__(self, agent_id: str, description: str, storage_backend=None)
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]
    def store(self, key: str, data: Dict[str, Any], ttl: int = None) -> None
    def retrieve(self, key: str) -> Optional[Dict[str, Any]]
    def search(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]
    def forget(self, key: str) -> bool
```

### LogAgent

Agent for recording events and system states.

```python
class LogAgent(NISAgent):
    def __init__(self, agent_id: str, description: str, log_file: str = None)
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]
    def log_event(self, event_type: str, event_data: Dict[str, Any]) -> None
    def get_logs(self, start_time: float = None, end_time: float = None) -> List[Dict[str, Any]]
```

## Emotional Layer

### EmotionAgent

Agent for managing emotional state.

```python
class EmotionAgent(NISAgent):
    def __init__(self, agent_id: str, description: str)
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]
    def update_emotion(self, dimension: str, value: float) -> None
    def get_emotional_state(self) -> Dict[str, float]
```

### EmotionalStateSystem

System for managing emotional dimensions.

```python
class EmotionalStateSystem:
    def __init__(self)
    def update(self, dimension: str, value: float) -> None
    def get_state(self) -> Dict[str, float]
    def reset(self) -> None
```

## Reasoning Layer

### CortexAgent

Agent for high-level decision making.

```python
class CortexAgent(NISAgent):
    def __init__(self, agent_id: str, description: str)
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]
    def decide(self, perception: Dict[str, Any], memory: Dict[str, Any], emotional_state: Dict[str, float]) -> Dict[str, Any]
```

### PlanningAgent

Agent for developing sequences of actions.

```python
class PlanningAgent(NISAgent):
    def __init__(self, agent_id: str, description: str)
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]
    def create_plan(self, goal: Dict[str, Any], current_state: Dict[str, Any]) -> List[Dict[str, Any]]
```

## Action Layer

### BuilderAgent

Agent for constructing responses or commands.

```python
class BuilderAgent(NISAgent):
    def __init__(self, agent_id: str, description: str)
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]
    def build_response(self, decision: Dict[str, Any]) -> Dict[str, Any]
```

### DeployerAgent

Agent for executing actions in the environment.

```python
class DeployerAgent(NISAgent):
    def __init__(self, agent_id: str, description: str)
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]
    def execute(self, action: Dict[str, Any]) -> Dict[str, Any]
```

## Learning Layer

### LearningAgent

Agent for updating models and parameters.

```python
class LearningAgent(NISAgent):
    def __init__(self, agent_id: str, description: str)
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]
    def learn(self, input_data: Dict[str, Any], expected_output: Dict[str, Any]) -> Dict[str, Any]
```

### OptimizerAgent

Agent for tuning system performance.

```python
class OptimizerAgent(NISAgent):
    def __init__(self, agent_id: str, description: str)
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]
    def optimize(self, metrics: Dict[str, Any]) -> Dict[str, Any]
```

## Coordination Layer

### CoordinatorAgent

Agent for managing message routing and prioritization.

```python
class CoordinatorAgent(NISAgent):
    def __init__(self, agent_id: str, description: str)
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]
    def route_message(self, message: Dict[str, Any], target_layer: NISLayer) -> Dict[str, Any]
    def prioritize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]
```

## Message Format

Messages in the NIS Protocol follow this standard structure:

```python
message = {
    "agent_id": str,               # ID of the agent that processed the message
    "timestamp": float,            # Unix timestamp
    "status": str,                 # "success", "error", or "pending"
    "payload": Dict[str, Any],     # Primary data
    "metadata": Dict[str, Any],    # Additional information
    "emotional_state": Dict[str, float]  # Current emotional dimensions
}
```

## Storage Backends

### InMemoryStorage

Simple in-memory storage for development and testing.

```python
class InMemoryStorage:
    def __init__(self)
    def set(self, key: str, value: Any, ttl: int = None) -> None
    def get(self, key: str) -> Optional[Any]
    def delete(self, key: str) -> bool
    def search(self, pattern: Dict[str, Any]) -> List[Any]
```

### RedisStorage

Redis-based storage for production use.

```python
class RedisStorage:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0)
    def set(self, key: str, value: Any, ttl: int = None) -> None
    def get(self, key: str) -> Optional[Any]
    def delete(self, key: str) -> bool
    def search(self, pattern: Dict[str, Any]) -> List[Any]
```

## Utilities

### MessageBuilder

Utility for creating standardized messages.

```python
class MessageBuilder:
    @staticmethod
    def create_message(agent_id: str, payload: Dict[str, Any], emotional_state: Dict[str, float] = None) -> Dict[str, Any]
    
    @staticmethod
    def create_error_message(agent_id: str, error_message: str) -> Dict[str, Any]
```

### EmotionalDecayHandler

Utility for managing emotional state decay.

```python
class EmotionalDecayHandler:
    def __init__(self, decay_rates: Dict[str, float] = None)
    def apply_decay(self, emotional_state: Dict[str, float], elapsed_time: float) -> Dict[str, float]
```

### Logger

Utility for logging NIS Protocol events.

```python
class Logger:
    def __init__(self, log_file: str = None, log_level: int = logging.INFO)
    def log(self, level: int, message: str, data: Dict[str, Any] = None) -> None
    def debug(self, message: str, data: Dict[str, Any] = None) -> None
    def info(self, message: str, data: Dict[str, Any] = None) -> None
    def warning(self, message: str, data: Dict[str, Any] = None) -> None
    def error(self, message: str, data: Dict[str, Any] = None) -> None
    def critical(self, message: str, data: Dict[str, Any] = None) -> None
```

For more detailed implementation information, refer to the [Implementation Guide](Implementation_Guide.md) and [NIS Protocol Whitepaper](NIS_Protocol_Whitepaper.md). 
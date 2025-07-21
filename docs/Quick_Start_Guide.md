# NIS Protocol Quick Start Guide

This guide will help you get started with implementing the Neuro-Inspired System Protocol (NIS) in your applications.

## Installation

```bash
# Clone the repository
git clone https://github.com/Organica-Ai-Solutions/NIS_Protocol.git
cd NIS-Protocol

# Install the required dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Basic Usage

Here's a simple example of how to create and connect NIS Protocol agents:

```python
from nis_protocol.core import NISRegistry
from nis_protocol.agents import VisionAgent, MemoryAgent, CortexAgent, ActionAgent
from nis_protocol.emotion import EmotionalStateSystem

# Create the registry
registry = NISRegistry()

# Initialize the emotional state system
emotional_system = EmotionalStateSystem()

# Create agents
vision_agent = VisionAgent("vision_1", "Processes visual input")
memory_agent = MemoryAgent("memory_1", "Stores and retrieves information")
cortex_agent = CortexAgent("cortex_1", "Makes decisions based on inputs")
action_agent = ActionAgent("action_1", "Executes actions in the environment")

# Process an input
input_data = {"type": "image", "data": "base64_encoded_image_data"}

# Processing pipeline
perception_result = vision_agent.process(input_data)
memory_context = memory_agent.retrieve(perception_result)
decision = cortex_agent.decide(perception_result, memory_context, emotional_system.get_state())
result = action_agent.execute(decision)

# Update emotional state
emotional_system.update("suspicion", 0.7)  # Increase suspicion

# Store the results for future reference
memory_agent.store(perception_result, decision, result)
```

## Key Components

When implementing the NIS Protocol, focus on these key components:

1. **Agent Registry**: Central component that tracks all agents
2. **Emotional State System**: Modulates agent behavior and decision-making
3. **Memory Management**: Provides context for decisions
4. **Message Flow**: Standardized communication between agents

## Next Steps

- See the [examples](../examples/) directory for complete implementations
- Read the [Implementation Guide](Implementation_Guide.md) for more details
- Check the [API Reference](API_Reference.md) for detailed documentation

## Common Use Cases

- **Automated monitoring ([health tracking](src/infrastructure/integration_coordinator.py)) Systems**: Create systems that adapt their suspicion levels based on inputs
- **Smart Assistants**: Build assistants that remember user preferences and adjust behavior
- **Autonomous Agents**: Develop agents that make decisions with emotional modulation

For more detailed information, refer to the [NIS Protocol Whitepaper](NIS_Protocol_Whitepaper.md). 
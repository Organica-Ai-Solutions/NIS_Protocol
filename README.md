# Neuro-Inspired System Protocol (NIS Protocol)

![NIS Protocol Logo](assets/images/nis-protocol-logo.png)

The Neuro-Inspired System Protocol (NIS Protocol) is a biologically inspired framework for designing intelligent multi-agent systems. Drawing on the structure and function of the human brain, this protocol integrates perception, memory, emotional weighting, reasoning, and action into a cohesive architecture, enabling more adaptive and naturally intelligent systems.

## 📄 Whitepaper

For a comprehensive overview of the NIS Protocol:

- [Read Online Version](docs/NIS_Protocol_Whitepaper.html)
- [Download PDF Version](docs/finalwhitepaper.pdf)

## 🧠 Key Features

- **Layered Cognitive Architecture**: Inspired by the brain's hierarchical processing
- **Emotional State System**: Dynamic modulation of agent behavior through emotional dimensions
- **Memory Integration**: Short and long-term memory for context-aware decision making
- **Standardized Messaging**: Consistent message format for inter-agent communication
- **Biologically Inspired Learning**: Continuous adaptation based on experience
- **Protocol Integration**: Seamless communication with MCP, ACP, and A2A protocols

![NIS Protocol Architecture](assets/images/diagram.png)

## 📖 Documentation

Comprehensive documentation is available to help you understand and implement the NIS Protocol:

- [Getting Started Guide](getting_started/index.html)
- [Architecture Overview](architecture/index.html)
- [Emotional State System](emotional_state/index.html)
- [Memory System](memory_system/index.html)
- [Protocol Integration](src/adapters/README.md)
- [Implementation Examples](examples/index.html)
- [Frequently Asked Questions](docs/faq.html)

## 🔧 Installation

To install the NIS Protocol framework:

```bash
pip install nis-protocol
```

## 🚀 Quick Start

```python
from nis_protocol import NISRegistry, NISAgent, NISLayer

# Create a registry
registry = NISRegistry()

# Define a simple agent
class MyPerceptionAgent(NISAgent):
    def __init__(self):
        super().__init__(
            agent_id="my_perception_agent",
            layer=NISLayer.PERCEPTION,
            description="A simple perception agent"
        )
    
    def process(self, message):
        # Process incoming data
        # ...
        return processed_data

# Register the agent
agent = MyPerceptionAgent()

# Start processing
registry.start()
```

## 🔄 External Protocol Integration

NIS Protocol can integrate with other major AI agent communication protocols:

```python
from nis_protocol import CoordinatorAgent
from nis_protocol.adapters import configure_coordinator_agent

# Create coordinator
coordinator = CoordinatorAgent()

# Configure with adapters for external protocols
configure_coordinator_agent(coordinator, config_path="config/protocol_routing.json")

# Process messages from external protocols
mcp_response = coordinator.process({
    "protocol": "mcp",
    "original_message": mcp_message  # Anthropic's Model Context Protocol
})

acp_response = coordinator.process({
    "protocol": "acp",
    "original_message": acp_message  # IBM's Agent Communication Protocol
})

a2a_response = coordinator.process({
    "protocol": "a2a",
    "original_message": a2a_message  # Google's Agent2Agent Protocol
})
```

## 📊 Application Examples

The NIS Protocol can be applied to various domains:

![Usage Examples](assets/images/usesExamples.png)

- **Autonomous Systems**: Robotics, drones, self-driving vehicles
- **Smart Infrastructure**: Traffic management, energy distribution
- **Security Applications**: Surveillance, fraud detection
- **Healthcare**: Patient monitoring with adaptive priorities
- **Customer Interaction**: Context-aware support systems

## 🌟 Emotional State System

The emotional state system modulates agent behavior based on context-sensitive dimensions:

![Emotional State Heatmap](assets/images/heatmap.png)

Key emotional dimensions include:
- **Suspicion**: Increases scrutiny for unusual patterns
- **Urgency**: Prioritizes time-sensitive processing
- **Confidence**: Adjusts decision-making thresholds
- **Interest**: Directs focus to specific features
- **Novelty**: Highlights deviations from expectations

## 🤝 Contributing

Contributions to the NIS Protocol are welcome! Please see our [contribution guidelines](CONTRIBUTING.md) for details on how to get involved.

## 📝 License

The NIS Protocol is released under the [MIT License](LICENSE).

## 🔗 Contact

- **GitHub**: [github.com/Organica-AI-Solutions/NIS-Protocol](https://github.com/Organica-AI-Solutions/NIS-Protocol)
- **Email**: hello@organicaai.com
- **Website**: [organicaai.com](https://organicaai.com) 
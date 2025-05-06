# NIS Protocol Adapters for External Protocol Integration

This module provides adapters for integrating NIS Protocol with external AI agent communication protocols.

## Overview

The NIS Protocol can act as a meta-protocol, orchestrating and coordinating agents from different protocol ecosystems. The adapters in this module allow NIS Protocol to seamlessly communicate with:

- **MCP (Model Context Protocol)** - Anthropic's protocol for connecting AI systems to data sources
- **ACP (Agent Communication Protocol)** - IBM's standardized protocol for agent communication
- **A2A (Agent2Agent Protocol)** - Google's protocol for agent interoperability across platforms

## Architecture

The adapter system consists of:

1. **BaseProtocolAdapter** - Abstract base class that all adapters implement
2. **CoordinatorAgent** - Central hub that manages message routing and translation
3. **Protocol-specific adapters** - Translate between NIS Protocol and external formats

```
┌─────────────────────────────────────┐
│          NIS Protocol Core          │
│                                     │
│  ┌─────────┐ ┌─────────┐ ┌────────┐ │
│  │ Vision  │ │ Memory  │ │  Other │ │
│  │ Agent   │ │ Agent   │ │ Agents │ │
│  └─────────┘ └─────────┘ └────────┘ │
│         │         │          │      │
│         └────┬────┴──────────┘      │
│              │                      │
│     ┌────────▼─────────┐            │
│     │  Coordinator     │            │
│     │     Agent        │            │
│     └────────┬─────────┘            │
└──────────────┼──────────────────────┘
               │
    ┌───────────────────────┐
    │   Protocol Adapters   │
    │                       │
┌───▼───┐   ┌───────┐   ┌───▼───┐
│  MCP  │   │  ACP  │   │  A2A  │
│Adapter│   │Adapter│   │Adapter│
└───┬───┘   └───┬───┘   └───┬───┘
    │           │           │
    ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│   MCP   │ │   ACP   │ │   A2A   │
│  Tools  │ │  Agents │ │  Agents │
└─────────┘ └─────────┘ └─────────┘
```

## Usage

### Basic Setup

To configure a NIS Protocol system with external protocol adapters:

```python
from src.agents.coordination.coordinator_agent import CoordinatorAgent
from src.adapters.bootstrap import configure_coordinator_agent

# Create coordinator agent
coordinator = CoordinatorAgent()

# Configure with adapters
configure_coordinator_agent(coordinator, config_path="path/to/config.json")
```

### Configuration

The adapter configuration is specified in a JSON file:

```json
{
  "mcp": {
    "base_url": "https://api.example.com/mcp",
    "api_key": "YOUR_MCP_API_KEY",
    "tool_mappings": {
      "vision_tool": {
        "nis_agent": "vision_agent",
        "target_layer": "PERCEPTION"
      }
    }
  },
  "acp": {
    "base_url": "https://api.example.com/acp",
    "api_key": "YOUR_ACP_API_KEY"
  },
  "a2a": {
    "base_url": "https://api.example.com/a2a",
    "api_key": "YOUR_A2A_API_KEY"
  }
}
```

### Processing External Protocol Messages

To process a message from an external protocol:

```python
# Process MCP message
mcp_response = coordinator.process({
    "protocol": "mcp",
    "original_message": mcp_message
})

# Process ACP message
acp_response = coordinator.process({
    "protocol": "acp",
    "original_message": acp_message
})

# Process A2A message
a2a_response = coordinator.process({
    "protocol": "a2a",
    "original_message": a2a_message
})
```

### Sending Messages to External Agents

To send a message to an external agent:

```python
# Send to MCP tool
mcp_response = coordinator.route_to_external_agent(
    "mcp",
    "vision_tool",
    nis_message
)

# Send to ACP agent
acp_response = coordinator.route_to_external_agent(
    "acp", 
    "factory_control_agent",
    nis_message
)

# Send to A2A agent
a2a_response = coordinator.route_to_external_agent(
    "a2a",
    "natural_language_agent",
    nis_message
)
```

## Adding New Protocol Adapters

To add support for a new protocol:

1. Create a new adapter class inheriting from `BaseProtocolAdapter`
2. Implement the required methods:
   - `translate_to_nis` - Convert from external format to NIS Protocol
   - `translate_from_nis` - Convert from NIS Protocol to external format
   - `send_to_external_agent` - Send messages to external agents

## Example

See the `examples/protocol_integration` directory for a full example demonstrating integration with all supported protocols. 
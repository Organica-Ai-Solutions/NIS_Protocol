# NIS Protocol Integration Examples

This directory contains examples demonstrating how to integrate NIS Protocol with external AI agent communication protocols.

## Supported Protocols

### MCP (Anthropic's Model Context Protocol)
The Model Context Protocol is an open standard for connecting AI assistants with data sources and tools, allowing for real-time data access and tool integration.

### ACP (IBM's Agent Communication Protocol)
The Agent Communication Protocol is designed for standardized communication between AI agents, ensuring interoperability across different frameworks and implementations.

### A2A (Google's Agent2Agent Protocol)
The Agent2Agent Protocol enables communication and interoperability between opaque agentic applications, allowing AI agents to coordinate actions across platforms.

## Examples

### Basic Integration Example

The file `protocol_integration_example.py` demonstrates:

1. Setting up a NIS Protocol system with protocol adapters
2. Processing messages from each external protocol
3. Sending messages to external agents

To run the example:

```bash
python protocol_integration_example.py
```

## Configuration

The example uses a simulated configuration with dummy endpoints. In a real implementation, you would replace these with actual API endpoints and credentials:

```json
{
  "mcp": {
    "base_url": "https://api.example.com/mcp",
    "api_key": "YOUR_MCP_API_KEY"
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

## Handling External Protocol Messages

The example demonstrates three key patterns:

1. **Receiving external protocol messages** - Converting incoming messages to the NIS format
2. **Processing with NIS agents** - Using internal NIS agents to handle the tasks
3. **Responding in external formats** - Translating responses back to the original protocol format

## Adding Your Own Agents

The example includes two simple agents (VisionAgent and MemoryAgent). You can extend the example by:

1. Creating your own agent classes inheriting from NISAgent
2. Registering them with the NIS Registry
3. Updating the routing configuration to map external protocol entities to your agents 
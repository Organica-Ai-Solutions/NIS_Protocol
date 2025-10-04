# NIS Protocol Python Client SDK

Simple, easy-to-use Python client for interacting with NIS Protocol backend.

## Installation

```bash
pip install requests  # Sync client
pip install aiohttp   # Async client (optional)
```

## Quick Start

```python
from nis_client import NISClient

# Initialize client
client = NISClient("http://localhost:8000")

# Check health
if client.is_healthy():
    print("✅ Backend is healthy!")

# Simple chat
response = client.chat("What is quantum computing?")
print(response.response)
print(f"Provider: {response.provider}")
print(f"Tokens: {response.tokens_used}")
```

## Features

### Chat Methods

```python
# Basic chat
response = client.chat(
    message="Hello!",
    user_id="user123",
    conversation_id="conv456",
    provider="openai",  # openai, anthropic, google, deepseek, kimi, smart
    agent_type="reasoning"  # reasoning, creative, analytical
)

# Smart Consensus (multiple LLMs)
response = client.smart_consensus("Explain machine learning")
```

### Agent Methods

```python
# Get all agents
agents = client.get_agents()
for agent in agents:
    print(f"{agent.name}: {agent.status}")
```

### Physics Methods

```python
# Validate physics scenario
result = client.validate_physics(
    scenario="Ball thrown at 45 degrees with initial velocity 20 m/s",
    domain="mechanics",
    mode="true_pinn"
)

# Get capabilities
capabilities = client.get_physics_capabilities()
```

### Research Methods

```python
# Deep research
result = client.deep_research(
    query="Quantum computing applications",
    depth="comprehensive",
    sources=10
)
```

## Async Client

```python
from nis_client import AsyncNISClient

async def main():
    async with AsyncNISClient("http://localhost:8000") as client:
        # Check health
        health = await client.health()
        print(health)
        
        # Chat
        response = await client.chat("Hello!")
        print(response.response)

import asyncio
asyncio.run(main())
```

## Response Objects

### ChatResponse

```python
@dataclass
class ChatResponse:
    response: str           # AI response text
    provider: str          # LLM provider used
    model: str             # Model name
    confidence: float      # Response confidence
    tokens_used: int       # Tokens consumed
    real_ai: bool          # Real AI vs mock
    reasoning_trace: List[str]  # Processing steps
```

### AgentStatus

```python
@dataclass
class AgentStatus:
    name: str              # Agent name
    type: str              # Agent type
    status: str            # Current status
    capabilities: List[str]  # Agent capabilities
```

## Error Handling

```python
try:
    response = client.chat("Hello!")
except requests.HTTPError as e:
    print(f"HTTP error: {e}")
except Exception as e:
    print(f"Error: {e}")
```

## Advanced Usage

### Custom Headers

```python
client = NISClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)
```

### Direct API Access

```python
# Make custom requests
result = client._request("GET", "/custom/endpoint")
```

## Examples

See `examples/` directory for complete examples:
- `01_basic_usage.py` - Basic chat and features
- `02_autonomous_mode.py` - Autonomous agent usage
- `03_fastapi_integration.py` - FastAPI integration

## Requirements

- Python 3.7+
- requests
- aiohttp (optional, for async client)

## License

Same as NIS Protocol main project


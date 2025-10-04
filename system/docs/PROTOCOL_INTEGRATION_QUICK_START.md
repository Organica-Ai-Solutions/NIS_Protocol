# Third-Party Protocol Integration - Quick Start Guide
**NIS Protocol v3.2** | Updated: October 1, 2025

This guide provides practical examples for integrating NIS Protocol with MCP, A2A, and ACP.

---

## MCP (Model Context Protocol) Integration

### Example 1: Connect to Local Filesystem Server

```python
from src.adapters.enhanced_mcp_adapter import EnhancedMCPAdapter

# Connect to local MCP filesystem server via stdio
adapter = EnhancedMCPAdapter({
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/you/data"]
})

# Initialize connection
await adapter.initialize()
# Output: Connected to MCP server: filesystem v1.0.0
# Output: Discovered 8 MCP tools

# List available tools
tools = await adapter.discover_tools()
for tool in tools:
    print(f"- {tool.name}: {tool.description}")

# Execute a tool
result = await adapter.call_tool(
    "read_file",
    {"path": "/Users/you/data/example.txt"}
)
print(result)
```

### Example 2: Use MCP Resources

```python
# Discover available resources
resources = await adapter.discover_resources()
for resource in resources:
    print(f"- {resource.uri}: {resource.name}")

# Read a resource
content = await adapter.read_resource("file:///Users/you/data/config.json")
print(content)
```

### Example 3: Integrate MCP into NIS Pipeline

```python
from src.agents.unified_coordinator import UnifiedCoordinator

coordinator = UnifiedCoordinator()

# Add MCP adapter to coordinator
coordinator.add_external_protocol_adapter("mcp", adapter)

# Process data through NIS pipeline with MCP tools
result = await coordinator.process_with_external_tools(
    data={"query": "Read and analyze file"},
    protocol="mcp",
    tool_name="read_file",
    tool_params={"path": "/data/analysis.csv"}
)

# Result includes:
# - Laplace transform
# - KAN reasoning
# - PINN validation
# - MCP tool execution
# - LLM interpretation
```

---

## A2A (Agent2Agent Protocol) Integration

### Example 1: Create Long-Running Task

```python
from src.adapters.enhanced_a2a_adapter import EnhancedA2AAdapter, TaskStatus

# Connect to A2A agent
adapter = EnhancedA2AAdapter({
    "base_url": "https://api.example.com/a2a",
    "api_key": "your-api-key"
})

# Create a long-running task
task = await adapter.create_task(
    description="Process large dataset with AI analysis",
    agent_id="external_analytics_agent",
    parameters={
        "dataset_url": "https://data.example.com/large.csv",
        "analysis_type": "comprehensive"
    }
)

print(f"Task created: {task.task_id}, status: {task.status.value}")

# Poll for task completion
while task.status != TaskStatus.COMPLETED:
    await asyncio.sleep(5)
    task = await adapter.get_task_status(task.task_id, "external_analytics_agent")
    print(f"Progress: {task.progress * 100:.1f}%")

# Get results
print(f"Artifacts: {task.artifacts}")
```

### Example 2: Rich Content with UX Negotiation

```python
from src.adapters.enhanced_a2a_adapter import A2APart

# Create message with multiple content types
message = adapter.create_message_with_parts(
    parts=[
        # Text analysis
        A2APart(
            content_type="text",
            content="Analysis of Q3 2024 sales data reveals..."
        ),
        # Chart image
        A2APart(
            content_type="image",
            content="https://charts.example.com/q3-sales.png",
            metadata={"alt": "Q3 Sales Chart"}
        ),
        # Interactive dashboard
        A2APart(
            content_type="iframe",
            content="https://dashboard.example.com/q3",
            metadata={"width": "100%", "height": "600px"}
        )
    ],
    agent_id="reporting_agent"
)

# Send rich message
response = await adapter.send_to_external_agent("reporting_agent", message)
```

### Example 3: Multi-Agent Collaboration

```python
# Coordinate multiple A2A agents
agents = ["data_analyst", "visualization_expert", "report_writer"]

tasks = []
for agent in agents:
    task = await adapter.create_task(
        description=f"Contribute to quarterly report",
        agent_id=agent,
        parameters={"quarter": "Q3", "year": 2024}
    )
    tasks.append(task)

# Wait for all agents to complete
results = await asyncio.gather(*[
    adapter.get_task_status(task.task_id, agent)
    for task, agent in zip(tasks, agents)
])

# Combine artifacts from all agents
combined_report = {
    "data_analysis": results[0].artifacts,
    "visualizations": results[1].artifacts,
    "narrative": results[2].artifacts
}
```

---

## ACP (Agent Communication Protocol) Integration

### Example 1: REST-Based Agent Communication

```python
from src.adapters.enhanced_acp_adapter import EnhancedACPAdapter

# Connect to ACP agent
adapter = EnhancedACPAdapter({
    "base_url": "http://localhost:3000"
})

# Execute agent task (async)
result = await adapter.execute_acp_agent(
    agent_url="http://localhost:3000",
    message={
        "task": "code_review",
        "code": open("src/test.py").read(),
        "standards": ["pep8", "security"]
    },
    async_mode=True
)

print(f"Task ID: {result['task_id']}")
print(f"Status: {result['status']}")

# Execute agent task (sync)
result = await adapter.execute_acp_agent(
    agent_url="http://localhost:3000",
    message={
        "task": "quick_validation",
        "data": {"value": 42}
    },
    async_mode=False
)

print(f"Result: {result['result']}")
```

### Example 2: Offline Agent Discovery

```python
# Export NIS Protocol Agent Card for discovery
agent_card = adapter.export_agent_card()

# Save to package.json or pyproject.toml for offline discovery
with open("package.json", "r+") as f:
    package = json.load(f)
    package.update(agent_card)
    f.seek(0)
    json.dump(package, f, indent=2)

print("Agent Card embedded in package.json")
print(json.dumps(agent_card, indent=2))

# Other ACP agents can now discover NIS Protocol's capabilities offline
# by reading the package.json file
```

### Example 3: NIS as ACP Server

```python
from src.services.acp_server import ACPServer

# Start NIS Protocol as ACP-compliant server
server = ACPServer(port=3000)

@server.agent()
async def physics_validator(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    ACP agent endpoint for physics validation.
    Any ACP client can call this via REST.
    """
    from src.agents.unified_coordinator import UnifiedCoordinator
    
    coordinator = UnifiedCoordinator()
    
    # Process through PINN validation
    result = await coordinator.physics_agent.validate(input_data)
    
    return {
        "physics_compliant": result["compliant"],
        "confidence": result["confidence"],
        "violations": result["violations"]
    }

# Server is now accessible at:
# POST http://localhost:3000/execute
# Any ACP-compliant agent can communicate with NIS Protocol
```

---

## Complete Integration Example: Multi-Protocol Workflow

```python
from src.adapters.enhanced_mcp_adapter import EnhancedMCPAdapter
from src.adapters.enhanced_a2a_adapter import EnhancedA2AAdapter
from src.adapters.enhanced_acp_adapter import EnhancedACPAdapter
from src.agents.unified_coordinator import UnifiedCoordinator

async def multi_protocol_analysis():
    """
    Demonstrate NIS Protocol coordinating multiple external protocols.
    
    Workflow:
    1. MCP: Read data from filesystem
    2. NIS: Process through Laplace→KAN→PINN pipeline
    3. A2A: Send to external analytics agent for specialized analysis
    4. ACP: Validate results with external quality assurance agent
    5. NIS: Generate final LLM-interpreted report
    """
    
    # 1. Setup protocol adapters
    mcp_adapter = EnhancedMCPAdapter({
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"]
    })
    await mcp_adapter.initialize()
    
    a2a_adapter = EnhancedA2AAdapter({
        "base_url": "https://analytics.example.com/a2a",
        "api_key": "a2a-key"
    })
    
    acp_adapter = EnhancedACPAdapter({
        "base_url": "http://qa-agent.local:3000"
    })
    
    coordinator = UnifiedCoordinator()
    
    # 2. Read data via MCP
    print("Step 1: Reading data via MCP...")
    raw_data = await mcp_adapter.call_tool(
        "read_file",
        {"path": "/data/sensor_readings.csv"}
    )
    
    # 3. Process through NIS pipeline
    print("Step 2: Processing through NIS pipeline...")
    nis_result = await coordinator.process_data_pipeline({
        "signal_data": raw_data["content"]
    })
    
    # 4. Send to A2A analytics agent
    print("Step 3: Sending to A2A analytics agent...")
    a2a_task = await a2a_adapter.create_task(
        description="Advanced statistical analysis",
        agent_id="analytics_agent",
        parameters={
            "processed_data": nis_result["kan_output"],
            "analysis_type": "anomaly_detection"
        }
    )
    
    # Wait for completion
    while a2a_task.status != TaskStatus.COMPLETED:
        await asyncio.sleep(2)
        a2a_task = await a2a_adapter.get_task_status(
            a2a_task.task_id,
            "analytics_agent"
        )
    
    # 5. Validate with ACP quality assurance
    print("Step 4: Validating with ACP QA agent...")
    qa_result = await acp_adapter.execute_acp_agent(
        agent_url="http://qa-agent.local:3000",
        message={
            "task": "validate_analysis",
            "nis_validation": nis_result["pinn_validation"],
            "external_analysis": a2a_task.artifacts[0]
        },
        async_mode=False
    )
    
    # 6. Generate final report
    print("Step 5: Generating final report...")
    final_report = {
        "data_source": "MCP filesystem server",
        "nis_pipeline": {
            "laplace_transform": nis_result["laplace_output"],
            "kan_reasoning": nis_result["kan_output"],
            "pinn_validation": nis_result["pinn_validation"],
            "physics_compliant": nis_result["pinn_validation"]["compliant"]
        },
        "external_analysis": {
            "a2a_analytics": a2a_task.artifacts,
            "acp_qa_validation": qa_result["result"]
        },
        "llm_interpretation": await coordinator.llm_manager.generate_response(
            messages=[{
                "role": "user",
                "content": f"Interpret these analysis results: {json.dumps(nis_result)}"
            }]
        )
    }
    
    print("Analysis complete!")
    print(json.dumps(final_report, indent=2))
    
    return final_report

# Run the workflow
if __name__ == "__main__":
    import asyncio
    asyncio.run(multi_protocol_analysis())
```

---

## Configuration Best Practices

### Environment Variables

```bash
# .env file
# MCP Configuration
MCP_FILESYSTEM_PATH=/Users/you/data
MCP_TRANSPORT=stdio

# A2A Configuration  
A2A_BASE_URL=https://api.example.com/a2a
A2A_API_KEY=your-a2a-api-key
A2A_ENABLE_TASKS=true

# ACP Configuration
ACP_SERVER_PORT=3000
ACP_BASE_URL=http://localhost:3000
ACP_ENABLE_DISCOVERY=true

# Protocol Bridge Settings
PROTOCOL_RETRY_ATTEMPTS=3
PROTOCOL_TIMEOUT=30
PROTOCOL_CIRCUIT_BREAKER=true
```

### Protocol Routing Configuration

```json
{
  "protocols": {
    "mcp": {
      "enabled": true,
      "transport": "stdio",
      "servers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "${MCP_FILESYSTEM_PATH}"]
        },
        "sentry": {
          "transport": "http",
          "base_url": "https://mcp.sentry.io",
          "api_key": "${SENTRY_MCP_KEY}"
        }
      }
    },
    "a2a": {
      "enabled": true,
      "base_url": "${A2A_BASE_URL}",
      "api_key": "${A2A_API_KEY}",
      "features": {
        "task_management": true,
        "ux_negotiation": true,
        "long_running_tasks": true
      }
    },
    "acp": {
      "enabled": true,
      "mode": "both",
      "server": {
        "port": 3000,
        "enable_discovery": true
      },
      "client": {
        "agents": [
          {
            "name": "qa_agent",
            "url": "http://qa-agent.local:3000"
          }
        ]
      }
    }
  }
}
```

---

## Testing Protocol Integration

### Unit Tests

```python
import pytest
from src.adapters.enhanced_mcp_adapter import EnhancedMCPAdapter

@pytest.mark.asyncio
async def test_mcp_initialization():
    """Test MCP initialization handshake"""
    adapter = EnhancedMCPAdapter({
        "transport": "http",
        "base_url": "http://mock-mcp-server:8080"
    })
    
    server_info = await adapter.initialize()
    
    assert server_info.name is not None
    assert server_info.protocol_version == "2025-06-18"
    assert adapter.initialized is True

@pytest.mark.asyncio
async def test_mcp_tool_discovery():
    """Test tool discovery"""
    adapter = EnhancedMCPAdapter({...})
    await adapter.initialize()
    
    tools = await adapter.discover_tools()
    
    assert len(tools) > 0
    assert all(hasattr(tool, 'name') for tool in tools)
    assert all(hasattr(tool, 'input_schema') for tool in tools)
```

### Integration Tests

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_protocol_workflow():
    """Test complete workflow across MCP, A2A, and ACP"""
    # Setup
    mcp = EnhancedMCPAdapter({...})
    a2a = EnhancedA2AAdapter({...})
    acp = EnhancedACPAdapter({...})
    
    # Execute workflow
    result = await multi_protocol_analysis()
    
    # Verify
    assert result["data_source"] == "MCP filesystem server"
    assert result["nis_pipeline"]["physics_compliant"] is True
    assert len(result["external_analysis"]["a2a_analytics"]) > 0
```

---

## Troubleshooting

### MCP Connection Issues

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check server process
adapter = EnhancedMCPAdapter({
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"]
})

try:
    await adapter.initialize()
except Exception as e:
    # Check stderr for server errors
    if adapter.process:
        stderr = adapter.process.stderr.read()
        print(f"Server error: {stderr}")
```

### A2A Task Timeout

```python
# Set explicit timeout
task = await adapter.create_task(...)

# Poll with timeout
import asyncio
try:
    async with asyncio.timeout(300):  # 5 minute timeout
        while task.status != TaskStatus.COMPLETED:
            await asyncio.sleep(5)
            task = await adapter.get_task_status(task.task_id, agent_id)
except asyncio.TimeoutError:
    # Cancel task
    await adapter.cancel_task(task.task_id, agent_id)
```

### ACP Agent Unreachable

```python
# Health check before execution
try:
    response = requests.get(f"{acp_url}/health", timeout=5)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"ACP agent unreachable: {e}")
    # Use fallback or skip
```

---

## Next Steps

1. **Review Assessment**: Read `/system/docs/PROTOCOL_INTEGRATION_ASSESSMENT.md`
2. **Run Examples**: Try the code examples above
3. **Configure Protocols**: Update your `.env` and `protocol_routing.json`
4. **Write Tests**: Create protocol-specific test suites
5. **Monitor Integration**: Set up protocol health monitoring

For detailed implementation guidance, see:
- Enhanced MCP Adapter: `/src/adapters/enhanced_mcp_adapter.py`
- Protocol Assessment: `/system/docs/PROTOCOL_INTEGRATION_ASSESSMENT.md`
- Integration Examples: `/dev/examples/protocol_integration/`


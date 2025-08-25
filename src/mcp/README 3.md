# MCP + Deep Agents + mcp-ui Integration

## 🎯 Overview

This module provides a complete integration between **LangChain-style Deep Agents**, **Model Context Protocol (MCP)**, and **mcp-ui** for the NIS Protocol. It creates an interactive, intelligent agent system that can execute complex workflows and present rich, dynamic UI interfaces.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐
│   Deep Agents   │────│   MCP Server     │────│   mcp-ui       │
│                 │    │                  │    │                │
│ • Planner       │    │ • Tool Handlers  │    │ • Data Grids   │
│ • Skills        │    │ • UI Generator   │    │ • Progress     │
│ • Sub-agents    │    │ • Intent Router  │    │ • Timelines    │
│ • Memory        │    │ • Security       │    │ • Diff Views   │
└─────────────────┘    └──────────────────┘    └────────────────┘
```

## 📁 Module Structure

```
src/mcp/
├── __init__.py              # Module exports
├── server.py                # Main MCP server implementation
├── schemas/                 # JSON schemas for all tools
│   ├── __init__.py
│   ├── tool_schemas.py      # Schema registry
│   ├── dataset_schemas.py   # Dataset tool schemas
│   ├── pipeline_schemas.py  # Pipeline tool schemas
│   ├── research_schemas.py  # Research tool schemas
│   ├── audit_schemas.py     # Audit tool schemas
│   └── code_schemas.py      # Code tool schemas
├── ui_resources.py          # mcp-ui resource generators
├── intent_validator.py      # Security & intent validation
├── integration.py           # NIS Protocol integration
├── demo.py                  # Demonstration module
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Basic Setup

```python
from src.mcp.integration import setup_mcp_integration

# Initialize with NIS Protocol
integration = await setup_mcp_integration({
    "mcp": {"host": "localhost", "port": 8000},
    "agent": {"provider": "anthropic"},
    "memory": {"backend": "sqlite"}
})
```

### 2. Handle Tool Requests

```python
# Execute a dataset search with UI generation
request = {
    "type": "tool",
    "tool_name": "dataset.search",
    "parameters": {
        "query": "weather data",
        "filters": {"format": "csv"},
        "limit": 20
    }
}

response = await integration.handle_mcp_request(request)
# Returns data + interactive UI resource
```

### 3. Create Execution Plans

```python
# Generate a complex multi-step plan
plan = await integration.create_execution_plan(
    "Analyze climate change impact on agriculture",
    context={"region": "North America", "timeframe": "3 months"}
)

# Execute the plan
results = await integration.execute_plan(plan["plan_id"])
```

## 🛠️ Available Tools

### Dataset Tools
- `dataset.search` - Search datasets with filters → **Data Grid UI**
- `dataset.preview` - Preview dataset structure → **Tabbed Viewer UI**
- `dataset.analyze` - Analyze data quality → **Analysis Dashboard UI**
- `dataset.list` - List available datasets → **Data Grid UI**

### Pipeline Tools
- `pipeline.run` - Execute data pipelines → **Progress Monitor UI**
- `pipeline.status` - Check execution status → **Status Dashboard UI**
- `pipeline.configure` - Configure parameters → **Form UI**
- `pipeline.cancel` - Cancel running pipeline → **Confirmation UI**
- `pipeline.artifacts` - View execution artifacts → **Tabbed Viewer UI**

### Research Tools
- `research.plan` - Generate research plans → **Tree View UI**
- `research.search` - Search literature → **Results Grid UI**
- `research.synthesize` - Synthesize findings → **Report UI**
- `research.analyze` - Analyze topics → **Analysis UI**

### Audit Tools
- `audit.view` - View audit trails → **Timeline UI**
- `audit.analyze` - Analyze performance → **Dashboard UI**
- `audit.compliance` - Check compliance → **Scorecard UI**
- `audit.risk` - Assess risks → **Risk Matrix UI**
- `audit.report` - Generate reports → **Report Viewer UI**

### Code Tools
- `code.edit` - Edit code files → **Diff Viewer UI**
- `code.review` - Review code quality → **Review Panel UI**
- `code.analyze` - Analyze structure → **Metrics Dashboard UI**
- `code.generate` - Generate code → **Code Editor UI**
- `code.refactor` - Refactor code → **Diff Viewer UI**

## 🎨 UI Components

The system generates rich, interactive UI components using mcp-ui:

### Data Grid
```python
ui_resource = ui_generator.create_data_grid(
    items=[{"id": "ds_001", "name": "Weather Data", "size": "2.5MB"}],
    title="Datasets",
    searchable=True,
    pagination=True,
    actions=[{"name": "preview", "label": "Preview"}]
)
```

### Progress Monitor
```python
ui_resource = ui_generator.create_progress_monitor(
    run_id="pipeline_123",
    status="running", 
    progress=45,
    logs=["Starting...", "Processing data..."],
    cancelable=True
)
```

### Tabbed Viewer
```python
ui_resource = ui_generator.create_tabbed_viewer({
    "Schema": {"columns": ["timestamp", "temperature"]},
    "Sample": [{"timestamp": "2025-01-19", "temp": 22.5}],
    "Statistics": {"rows": 10000, "null_values": 12}
})
```

## 🔒 Security Features

### Intent Validation
- **Schema Validation**: All intents validated against JSON schemas
- **Security Rules**: Whitelist of allowed tools, parameters, and URLs
- **Parameter Limits**: String length, array size, object depth restrictions
- **Pattern Filtering**: Blocks script injection, dangerous URLs

### Sandboxed Execution
- **UI Isolation**: All UI components run in sandboxed iframes
- **Message Validation**: All UI → server communication validated
- **Tool Permissions**: Fine-grained access control per tool
- **Error Containment**: Failures isolated to prevent system compromise

## 🧠 Deep Agent Skills

### DatasetSkill
- Search and filter datasets
- Generate previews with schema analysis
- Perform quality assessments
- Handle multiple data formats

### PipelineSkill  
- Execute complex data processing workflows
- Monitor execution with real-time progress
- Handle resource allocation and optimization
- Manage artifacts and outputs

### ResearchSkill
- Generate structured research plans
- Search and synthesize literature
- Perform topic analysis and gap identification
- Create comprehensive research workflows

### AuditSkill
- Generate detailed audit trails
- Analyze system performance and behavior
- Check compliance against frameworks
- Assess risks and generate reports

### CodeSkill
- Edit and modify code files
- Perform comprehensive code reviews
- Analyze code structure and metrics
- Generate code from specifications
- Refactor for improved maintainability

## 🔄 Intent Handling

### UI → Server Communication
```javascript
// From mcp-ui component
window.parent.postMessage({
    type: 'tool',
    payload: {
        toolName: 'dataset.preview',
        params: { dataset_id: 'weather_001' }
    }
}, '*');
```

### Server Response
```python
# Server processes intent and returns
{
    "success": true,
    "data": {...},
    "ui_resource": {
        "type": "resource",
        "resource": {
            "uri": "ui://tabs/preview_123",
            "mimeType": "text/html",
            "text": "<html>...</html>"
        }
    }
}
```

## 📊 Demo & Testing

### Run the Complete Demo
```python
from src.mcp.demo import run_demo

# Demonstrates all capabilities
results = await run_demo()
```

### Individual Component Testing
```python
from src.mcp.integration import MCPIntegration

integration = MCPIntegration()
await integration.initialize()

# Test specific tools
response = await integration.handle_mcp_request({
    "tool_name": "research.plan",
    "parameters": {"goal": "AI safety research"}
})
```

## 🔧 Configuration

### Full Configuration Example
```python
config = {
    "mcp": {
        "host": "localhost",
        "port": 8000,
        "enable_ui": True,
        "security": {
            "validate_intents": True,
            "sandbox_ui": True,
            "max_request_size": "10MB"
        }
    },
    "agent": {
        "provider": "anthropic",
        "model": "claude-3-sonnet-20241022",
        "max_tokens": 4000,
        "temperature": 0.1
    },
    "memory": {
        "backend": "sqlite",
        "connection_string": "data/mcp_memory.db",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "max_memories": 10000
    },
    "deep_agents": {
        "max_plan_steps": 20,
        "execution_timeout": 300,
        "retry_attempts": 3
    }
}
```

## 🌐 Web Integration

### FastAPI Example
```python
from fastapi import FastAPI
from src.mcp.integration import setup_mcp_integration

app = FastAPI()
integration = await setup_mcp_integration(config)

@app.post("/mcp/tools")
async def handle_tool_request(request: dict):
    return await integration.handle_mcp_request(request)

@app.get("/mcp/info")
async def get_server_info():
    return integration.get_server_info()
```

### WebSocket Support
```python
@app.websocket("/mcp/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    async for message in websocket.iter_json():
        response = await integration.handle_mcp_request(message)
        await websocket.send_json(response)
```

## 📈 Performance Considerations

### Optimization Features
- **Async Execution**: All operations use async/await patterns
- **Memory Management**: Efficient cleanup of completed plans
- **Resource Pooling**: Reuse of agent and memory connections
- **Caching**: Schema and UI template caching
- **Streaming**: Support for real-time progress updates

### Scaling Guidelines
- **Horizontal**: Multiple MCP server instances behind load balancer
- **Vertical**: Adjust memory limits and worker processes
- **Storage**: Use distributed memory backends for multi-instance
- **Monitoring**: Built-in metrics and health checks

## 🚨 Error Handling

### Graceful Degradation
- **Tool Failures**: Return error + fallback UI
- **UI Generation**: Fallback to generic data viewer
- **Agent Errors**: Retry with exponential backoff
- **Memory Issues**: Cleanup and warning notifications

### Monitoring & Debugging
- **Structured Logging**: All operations logged with context
- **Error Aggregation**: Centralized error collection
- **Performance Metrics**: Request latency, success rates
- **Health Checks**: Component status monitoring

## 🎯 Next Steps

### Immediate Enhancements
1. **WebSocket Support**: Real-time bidirectional communication
2. **Authentication**: User-based access control
3. **Persistence**: Save and restore agent plans
4. **Metrics**: Detailed performance analytics

### Advanced Features
1. **Multi-Agent Orchestration**: Coordinate multiple agent instances
2. **Custom UI Components**: Plugin system for new UI types
3. **External Integrations**: Connect to external APIs and services
4. **AI-Generated UI**: Dynamic UI generation based on data structure

## 📚 References

- [Model Context Protocol (MCP)](https://github.com/anthropics/mcp)
- [mcp-ui SDK](https://github.com/idosal/mcp-ui)
- [LangChain Deep Agents](https://python.langchain.com/docs/modules/agents/)
- [NIS Protocol Documentation](../README.md)

---

**Built with integrity. Engineered for impact. Designed for the future of AI interaction.**

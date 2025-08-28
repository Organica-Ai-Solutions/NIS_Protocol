# NIS Protocol - Agent Orchestration System Documentation

**Version**: 3.2.1  
**Updated**: 2025-01-19  
**Status**: Production Ready  

## Overview

The NIS Protocol Agent Orchestration System is a brain-inspired AI architecture that intelligently manages and coordinates multiple specialized agents. This system mimics human brain structure to provide efficient resource allocation, context-aware agent activation, and real-time performance monitoring.

## Architecture Overview

### Brain-like Structure

The orchestration system is organized into four main regions, each serving a specific function similar to human brain anatomy:

```
🧠 NIS Brain Architecture
├── Core Agents (Brain Stem) - Always Active
│   ├── Signal Processing Agent (Laplace Transform)
│   ├── Reasoning Agent (KAN Networks)
│   ├── Physics Validation Agent (PINN)
│   ├── Consciousness Agent (Self-awareness)
│   ├── Memory Agent (Storage & Retrieval)
│   └── Meta Coordination Agent (Orchestration)
├── Specialized Agents (Cerebral Cortex) - Context Activated
│   ├── Vision Analysis Agent
│   ├── Document Analysis Agent
│   ├── Web Search Agent
│   └── NVIDIA Physics Simulation Agent
├── Protocol Agents (Nervous System) - Event Driven
│   ├── Agent-to-Agent Protocol (A2A)
│   └── Model Context Protocol (MCP)
└── Learning Agents (Hippocampus) - Adaptive
    ├── Continuous Learning Agent
    └── BitNet Training Agent
```

## Agent Types and Activation Strategies

### 1. Core Agents (Always Active)

**Brain Stem Functions** - Essential operations that must always be running:

- **Signal Processing Agent**: Processes all incoming signals using Laplace transforms
- **Reasoning Agent**: Core reasoning using KAN (Kolmogorov-Arnold) networks
- **Physics Validation Agent**: Validates outputs against physics laws using PINN
- **Consciousness Agent**: Self-awareness and meta-cognitive processing
- **Memory Agent**: Memory storage and retrieval operations
- **Meta Coordination Agent**: High-level orchestration and oversight

**Activation**: `ActivationTrigger.ALWAYS`  
**Status**: Auto-activated on system startup  
**Dependencies**: Some have dependencies (e.g., Reasoning depends on Signal Processing)

### 2. Specialized Agents (Context Activated)

**Cerebral Cortex Functions** - Advanced capabilities activated based on context:

- **Vision Analysis Agent**: Computer vision and image analysis
- **Document Analysis Agent**: Document processing and text analysis
- **Web Search Agent**: Web search and research capabilities
- **NVIDIA Physics Simulation Agent**: Advanced physics simulation using NVIDIA NeMo

**Activation**: `ActivationTrigger.CONTEXT`  
**Triggers**: Activated when specific keywords or contexts are detected  
**Concurrency**: Support multiple concurrent instances (configurable per agent)

### 3. Protocol Agents (Event Driven)

**Nervous System Functions** - External communication and protocol handling:

- **Agent-to-Agent Protocol (A2A)**: Inter-agent communication
- **Model Context Protocol (MCP)**: Model context management

**Activation**: `ActivationTrigger.EVENT_DRIVEN`  
**Triggers**: Activated by specific system events or external requests

### 4. Learning Agents (Adaptive)

**Hippocampus Functions** - Learning and adaptation:

- **Continuous Learning Agent**: Ongoing system learning and improvement
- **BitNet Training Agent**: Neural network training and optimization

**Activation**: `ActivationTrigger.SCHEDULED` or `ActivationTrigger.ON_DEMAND`  
**Purpose**: System adaptation and improvement over time

## Agent Definition Structure

Each agent is defined using the `AgentDefinition` dataclass:

```python
@dataclass
class AgentDefinition:
    agent_id: str                    # Unique identifier
    name: str                        # Human-readable name
    agent_type: AgentType            # core, specialized, protocol, learning
    activation_trigger: ActivationTrigger  # when to activate
    status: AgentStatus = AgentStatus.INACTIVE
    context_keywords: List[str] = None      # keywords that trigger activation
    dependencies: List[str] = None          # required agents
    max_concurrent: int = 1                 # max concurrent instances
    timeout_seconds: float = 30.0           # agent timeout
    priority: int = 5                       # 1-10, higher = more important
    description: str = ""                   # agent description
```

## Context Analysis and Smart Activation

### Context Analyzer

The `ContextAnalyzer` class processes input to determine which agents should be activated:

```python
async def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
    # Analyzes input text, files, context
    # Returns keywords, complexity, requirements
```

**Detection Examples**:
- "image", "photo", "picture" → Activates Vision Agent
- "document", "pdf", "file" → Activates Document Agent  
- "search", "research", "find" → Activates Web Search Agent

### Dependency Resolution

The `DependencyResolver` ensures agents are activated in the correct order:

```python
def resolve_activation_order(self, agents: Dict[str, AgentDefinition], 
                           target_agents: List[str]) -> List[str]:
    # Returns optimal activation order based on dependencies
```

## Performance Monitoring

### Agent Metrics

Each agent tracks detailed performance metrics:

```python
@dataclass
class AgentMetrics:
    activation_count: int = 0               # times activated
    total_processing_time: float = 0.0      # cumulative processing time
    average_response_time: float = 0.0      # average response time
    success_rate: float = 1.0               # success percentage
    error_count: int = 0                    # total errors
    last_active: Optional[float] = None     # last activation timestamp
    resource_usage: Dict[str, float]        # CPU, memory, tokens
```

### Orchestrator Metrics

System-wide performance tracking:

```python
orchestrator_metrics = {
    "total_activations": 0,          # total agent activations
    "concurrent_agents": 0,          # currently active agents
    "queue_size": 0,                 # pending requests
    "average_activation_time": 0.0,  # average time to activate
    "error_rate": 0.0,              # system error rate
    "agent_utilization": 0.0         # % of agents active
}
```

## Real-time State Management

### WebSocket Integration

The orchestrator integrates with the NIS state management system for real-time updates:

```python
# State updates are automatically propagated via WebSocket
await emit_state_event(
    StateEventType.AGENT_STATUS_CHANGE,
    {
        "agent_id": agent_id,
        "instance_id": instance_id,
        "status": "activated",
        "context": context,
        "activation_time": activation_time,
        "agent_type": agent.agent_type.value
    }
)
```

### Frontend Visualization

The enhanced agent chat interface (`enhanced_agent_chat.html`) provides:

- **Live Brain Visualization**: Interactive SVG showing agent regions
- **Real-time Agent Status**: Current status of all agents
- **Neural Connection Animation**: Visual representation of agent communication
- **Interactive Controls**: Click-to-activate agents and brain regions

## API Reference

### Agent Status Endpoints

**Get All Agent Statuses**
```http
GET /api/agents/status

Response:
{
  "success": true,
  "agents": {
    "signal_processing": {
      "agent_id": "signal_processing",
      "status": "active",
      "type": "core",
      "active_instances": 1,
      "metrics": { ... },
      "definition": { ... }
    }
  },
  "timestamp": 1234567890.123
}
```

**Get Specific Agent Status**
```http
GET /api/agents/{agent_id}/status

Response:
{
  "success": true,
  "agent": {
    "agent_id": "vision",
    "status": "inactive",
    "type": "specialized",
    "active_instances": 0,
    "metrics": { ... },
    "definition": { ... }
  },
  "timestamp": 1234567890.123
}
```

### Agent Control Endpoints

**Activate Agent**
```http
POST /api/agents/activate
Content-Type: application/json

{
  "agent_id": "vision",
  "context": "user_request",
  "force": false
}

Response:
{
  "success": true,
  "message": "Agent vision activated",
  "timestamp": 1234567890.123
}
```

**Process Request Through Pipeline**
```http
POST /api/agents/process
Content-Type: application/json

{
  "input": {
    "text": "Analyze this image",
    "context": "visual_analysis",
    "files": ["image.jpg"]
  }
}

Response:
{
  "success": true,
  "result": {
    "request_id": "req_1234567890_1",
    "result": { ... },
    "activated_agents": ["signal_processing", "reasoning", "vision"],
    "context_analysis": { ... },
    "processing_time": 0.5
  },
  "timestamp": 1234567890.123
}
```

## Configuration and Customization

### Adding New Agents

To add a new agent to the orchestration system:

1. **Define the Agent**:
```python
self.register_agent(AgentDefinition(
    agent_id="my_custom_agent",
    name="My Custom Agent",
    agent_type=AgentType.SPECIALIZED,
    activation_trigger=ActivationTrigger.CONTEXT,
    context_keywords=["custom", "special", "my_keyword"],
    dependencies=["signal_processing"],
    max_concurrent=2,
    priority=7,
    description="Custom agent for specialized tasks"
))
```

2. **Implement Agent Logic**: Create the actual agent implementation
3. **Register with Orchestrator**: Add to the `_initialize_brain_structure()` method

### Customizing Activation Logic

Modify the `_determine_required_agents()` method to customize how agents are selected:

```python
async def _determine_required_agents(self, context_analysis: Dict[str, Any]) -> List[str]:
    # Custom logic for agent selection
    # Based on context, user preferences, system load, etc.
```

## Monitoring and Debugging

### Health Checks

The orchestrator continuously monitors agent health:

```python
async def _check_agent_health(self, instance_id: str) -> bool:
    # Check agent timeout, resource usage, responsiveness
    # Return False to deactivate unhealthy agents
```

### Performance Optimization

- **Concurrent Limits**: Configure `max_concurrent` per agent type
- **Timeout Management**: Set appropriate `timeout_seconds` values
- **Priority Weighting**: Use `priority` for resource allocation decisions
- **Context Optimization**: Fine-tune `context_keywords` for accurate activation

## Integration Examples

### Basic Usage

```python
# Initialize the orchestrator
orchestrator = NISAgentOrchestrator()
await orchestrator.start_orchestrator()

# Process a user request
result = await orchestrator.process_request({
    "text": "Analyze this medical image for anomalies",
    "files": ["medical_scan.jpg"],
    "user_context": "medical_analysis"
})

# Check agent status
status = orchestrator.get_agent_status("vision")
```

### Custom Context Analysis

```python
class CustomContextAnalyzer(ContextAnalyzer):
    async def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Custom analysis logic
        # Domain-specific context detection
        # Integration with external services
        pass
```

## Best Practices

1. **Resource Management**: Monitor agent utilization and adjust concurrency limits
2. **Context Tuning**: Regularly update context keywords based on usage patterns  
3. **Dependency Management**: Keep dependency chains short and well-documented
4. **Error Handling**: Implement robust error handling and recovery mechanisms
5. **Performance Monitoring**: Track metrics and optimize based on real usage data
6. **Graceful Degradation**: Design fallback strategies for agent failures

## Troubleshooting

### Common Issues

**Agent Not Activating**:
- Check context keywords match input
- Verify dependencies are satisfied
- Check agent status and availability

**Performance Issues**:
- Monitor concurrent agent limits
- Check for dependency bottlenecks
- Verify resource usage metrics

**State Synchronization Problems**:
- Check WebSocket connectivity
- Verify state manager configuration
- Monitor event emission logs

### Debug Commands

```bash
# Check agent status
curl http://localhost:8000/api/agents/status

# Activate agent manually
curl -X POST http://localhost:8000/api/agents/activate \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "vision", "context": "debug"}'

# Monitor WebSocket events
# Open browser dev tools on http://localhost:8000/enhanced
```

## Future Enhancements

- **Machine Learning Agent Selection**: ML-based context analysis and agent selection
- **Dynamic Agent Creation**: Runtime agent instantiation based on needs
- **Cross-System Orchestration**: Multi-instance agent coordination
- **Advanced Metrics**: Predictive performance analytics
- **Resource Optimization**: Dynamic resource allocation based on system load

---

For more information, see:
- [State Management Documentation](STATE_MANAGEMENT.md)
- [API Documentation](API_REFERENCE.md)
- [Frontend Integration Guide](FRONTEND_INTEGRATION.md)

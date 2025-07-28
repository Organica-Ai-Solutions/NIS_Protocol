# Third-Party Protocol Integration Guide

## Overview

The NIS Protocol v3 features comprehensive integration with external AI agent communication protocols, enabling seamless interoperability with diverse AI ecosystems. This integration allows NIS agents to leverage external tools, coordinate with third-party agents, and participate in cross-platform AI workflows.

## Supported Protocols

### 1. MCP (Model Context Protocol)
**Developer**: Anthropic  
**Purpose**: Connecting AI systems to data sources and tools  
**Integration Status**: ✅ Fully Supported

#### Capabilities
- **Tool Execution**: Direct integration with MCP-compatible tools
- **Data Source Access**: Connect to external databases and APIs
- **low-latency processing (implemented) (implemented) (measured)**: Streaming data processing (implemented) (implemented) capabilities
- **Security**: Authenticated API access with token management

#### Supported MCP Tools
- `file_editor_tool`: well-engineered file editing and manipulation
- `code_formatter_tool`: Code formatting and style corrections
- `lint_checker_tool`: Static code analysis and linting
- `test_generator_tool`: Automated test case generation
- `data_analyzer_tool`: Statistical data analysis
- `web_scraper_tool`: Intelligent web content extraction

### 2. ACP (Agent Communication Protocol)
**Developer**: IBM Research  
**Purpose**: Standardized protocol for agent communication  
**Integration Status**: ✅ Fully Supported

#### Capabilities
- **Agent Coordination**: Multi-agent workflow orchestration
- **Message Routing**: Intelligent message routing between agents
- **Quality Assurance**: Automated quality checking and validation
- **Process Control**: Industrial and business process automation

#### Supported ACP Agents
- `factory_control_agent`: Industrial automation and control
- `quality_assurance_agent`: Quality validation and testing
- `deployment_agent`: Automated deployment and release management
- `language_correction_agent`: Natural language processing (implemented) (implemented) and correction
- `workflow_orchestrator`: Complex workflow management

### 3. A2A (Agent2Agent Protocol)
**Developer**: Google Research  
**Purpose**: Agent interoperability across platforms  
**Integration Status**: ✅ Fully Supported

#### Capabilities
- **Cross-Platform Communication**: Seamless agent communication across different platforms
- **Natural Language processing (implemented) (implemented)**: well-engineered NLP capabilities
- **Multi-Agent Coordination**: Large-scale agent coordination
- **Protocol Translation**: Automatic protocol format translation

#### Supported A2A Capabilities
- `cross_platform_communication`: Inter-platform messaging
- `natural_language_processing`: well-engineered NLP services
- `multi_agent_coordination`: Large-scale coordination
- `protocol_translation`: Format conversion between protocols

## Architecture

The NIS Protocol uses a modular adapter architecture for third-party integration:

```
┌─────────────────────────────────────┐
│          NIS Protocol Core          │
│                                     │
│  ┌─────────┐ ┌─────────┐ ┌────────┐ │
│  │Consciousness│ Memory│ │ Action │ │
│  │ Agent   │ │ Agent   │ │ Agent  │ │
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

## Configuration

### Protocol Routing Configuration

Configure third-party protocols in `config/protocol_routing.json`:

```json
{
  "mcp": {
    "base_url": "https://api.anthropic.com/mcp",
    "api_key": "YOUR_MCP_API_KEY",
    "tool_mappings": {
      "file_editor_tool": {
        "nis_agent": "action_agent",
        "target_layer": "ACTION",
        "permissions": ["read", "write", "execute"]
      },
      "code_formatter_tool": {
        "nis_agent": "reasoning_agent", 
        "target_layer": "REASONING",
        "permissions": ["read", "write"]
      }
    },
    "rate_limits": {
      "requests_per_minute": 100,
      "concurrent_requests": 10
    },
    "timeout": 30
  },
  "acp": {
    "base_url": "https://api.ibm.com/acp",
    "api_key": "YOUR_ACP_API_KEY",
    "agent_mappings": {
      "factory_control_agent": {
        "nis_agent": "action_agent",
        "target_layer": "ACTION",
        "capabilities": ["process_control", "automation"]
      },
      "quality_assurance_agent": {
        "nis_agent": "validation_agent",
        "target_layer": "VALIDATION",
        "capabilities": ["quality_checking", "testing"]
      }
    },
    "authentication": {
      "type": "oauth2",
      "scope": "agent_communication"
    }
  },
  "a2a": {
    "base_url": "https://api.google.com/a2a",
    "api_key": "YOUR_A2A_API_KEY",
    "agent_mappings": {
      "natural_language_agent": {
        "nis_agent": "interpretation_agent",
        "target_layer": "INTERPRETATION",
        "languages": ["en", "es", "fr", "de", "zh"]
      }
    },
    "features": {
      "cross_platform": true,
      "real_time_translation": true,
      "protocol_bridging": true
    }
  }
}
```

### Environment Variables

Set the following environment variables:

```bash
# MCP Integration
MCP_API_KEY=your_mcp_api_key_here
MCP_BASE_URL=https://api.anthropic.com/mcp
MCP_TIMEOUT=30

# ACP Integration  
ACP_API_KEY=your_acp_api_key_here
ACP_BASE_URL=https://api.ibm.com/acp
ACP_AUTH_TYPE=oauth2

# A2A Integration
A2A_API_KEY=your_a2a_api_key_here
A2A_BASE_URL=https://api.google.com/a2a
A2A_FEATURES=cross_platform,real_time_translation

# General Protocol Settings
PROTOCOL_ADAPTER_ENABLED=true
PROTOCOL_RETRY_ATTEMPTS=3
PROTOCOL_CIRCUIT_BREAKER=true
```

## Usage Examples

### 1. Automated Code Fixing with MCP Tools

```python
# Using the Action Agent with MCP file editor
from src.agents.action.simple_audit_fixing_agent import create_simple_audit_fixer

# Create audit fixing agent
agent = create_simple_audit_fixer('code_fixer')

# Start fixing session with MCP tools
session_id = agent.start_fixing_session(['src/'])

# Monitor progress
report = agent.get_session_report(session_id)
print(f"Fixed {report['violations_fixed']} violations using MCP tools")
```

### 2. Quality Assurance with ACP Agents

```python
# Coordinate with ACP quality assurance agent
from src.adapters.acp_adapter import ACPAdapter

acp = ACPAdapter(config['acp'])

# Send code review request
response = acp.send_to_external_agent(
    'quality_assurance_agent',
    {
        'action': 'code_review',
        'code': open('src/test.py').read(),
        'standards': ['pep8', 'security', 'performance']
    }
)

print(f"Quality score: {response['quality_score']}")
print(f"Issues found: {len(response['issues'])}")
```

### 3. Cross-Platform Coordination with A2A

```python
# Multi-platform agent coordination
from src.adapters.a2a_adapter import A2AAdapter

a2a = A2AAdapter(config['a2a'])

# Coordinate documentation update across platforms
response = a2a.coordinate_workflow(
    {
        'workflow': 'documentation_update',
        'platforms': ['github', 'confluence', 'notion'],
        'agents': ['doc_writer', 'translator', 'reviewer'],
        'target_languages': ['en', 'es', 'fr']
    }
)

print(f"Coordination ID: {response['coordination_id']}")
print(f"Platforms involved: {len(response['participating_platforms'])}")
```

## Integration Workflows

### Automated Audit Fixing Workflow

1. **Detection Phase**
   - Consciousness agent performs codebase integrity scan
   - Self-audit engine identifies violations (hardcoded values, hype language)
   - Violations categorized by type and severity

2. **Tool Selection Phase**
   - Action agent analyzes violation types
   - Selects appropriate third-party tools (MCP file editor, ACP language corrector)
   - Determines optimal fixing strategy

3. **Execution Phase**
   - MCP tools handle file modifications
   - ACP agents provide language and quality corrections
   - A2A protocols coordinate cross-platform updates

4. **Validation Phase**
   - Consciousness agent validates fixes
   - Quality assurance through ACP agents
   - Updated audit trail and metrics

### Multi-Agent Collaboration Workflow

1. **Coordination Setup**
   - Define workflow objectives
   - Identify required capabilities
   - Map NIS agents to external protocol agents

2. **Task Distribution**
   - Break down complex tasks
   - Assign sub-tasks to appropriate protocol agents
   - Establish communication channels

3. **Execution Monitoring (implemented in src/monitoring/) (see src/Monitoring (implemented in src/monitoring/)/)**
   - Real-time progress tracking
   - Error handling and recovery
   - Performance optimization

4. **Result Synthesis**
   - Aggregate results from multiple protocol agents
   - Quality validation and consistency checking
   - Final output generation

## API Integration

### REST API Endpoints

#### Execute MCP Tool
```http
POST /api/protocol/mcp/tool/execute
Content-Type: application/json

{
  "tool_name": "file_editor_tool",
  "action": "replace_text",
  "parameters": {
    "file_path": "src/test.py",
    "old_text": "confidence=0.95",
    "new_text": "confidence=calculate_confidence(factors)"
  }
}
```

#### Communicate with ACP Agent
```http
POST /api/protocol/acp/agent/communicate
Content-Type: application/json

{
  "agent_id": "quality_assurance_agent",
  "message": {
    "action": "code_review",
    "code": "def example(): return True",
    "standards": ["pep8", "security"]
  }
}
```

#### Coordinate A2A Workflow
```http
POST /api/protocol/a2a/coordinate
Content-Type: application/json

{
  "workflow": "documentation_translation",
  "target_platforms": ["github", "confluence"],
  "languages": ["en", "es", "fr"]
}
```

## Security Considerations

### Authentication & Authorization

1. **API Key Management**
   - Secure storage of third-party API keys
   - Key rotation and expiration policies
   - Environment-based key configuration

2. **Access Control**
   - Role-based permissions for protocol access
   - Capability-based restrictions
   - Audit logging for all protocol interactions

3. **Data Protection**
   - Encryption in transit for all protocol communications
   - Data sanitization before external transmission
   - Compliance with data protection regulations

### Rate Limiting & Circuit Breakers

1. **Rate Limiting**
   - Per-protocol rate limits
   - Per-agent quotas
   - Dynamic rate adjustment based on performance

2. **Circuit Breakers**
   - Automatic failure detection
   - Graceful degradation to fallback methods
   - Recovery Monitoring (implemented in src/monitoring/) (see src/Monitoring (implemented in src/monitoring/)/) and automatic reset

3. **Retry Logic**
   - Exponential backoff for failed requests
   - Maximum retry limits
   - Dead letter queues for failed messages

## Monitoring (implemented in src/monitoring/) (see src/Monitoring (implemented in src/monitoring/)/) & Observability

### Metrics Collection

- **Protocol Performance**: Response times, success rates, error rates
- **Tool Usage**: Tool invocation frequency, execution times
- **Agent Coordination**: Multi-agent workflow success rates
- **Resource Utilization**: API quota usage, bandwidth consumption

### Alerting

- **Protocol Failures**: Alert on repeated protocol communication failures
- **Performance Degradation**: Monitor response time increases
- **Quota Exhaustion**: Early warning for API quota limits
- **Security Events**: Unauthorized access attempts or anomalous usage

### Dashboards

Real-time Monitoring (implemented in src/monitoring/) (see src/Monitoring (implemented in src/monitoring/)/) dashboards available at:
- Protocol Health: `http://localhost:5000/protocols`
- Tool Usage: `http://localhost:5000/tools`
- Agent Coordination: `http://localhost:5000/coordination`

## Troubleshooting

### Common Issues

1. **Protocol Connection Failures**
   - Check API key validity and permissions
   - Verify network connectivity and firewall settings
   - Review protocol-specific configuration

2. **Tool Execution Errors**
   - Validate tool parameters and permissions
   - Check tool availability and status
   - Review error logs for detailed messages

3. **Agent Coordination Issues**
   - Verify agent mappings in configuration
   - Check agent availability and capabilities
   - Review workflow definitions and dependencies

### Debug Mode

Enable debug mode for detailed logging:

```bash
export NIS_DEBUG=true
export PROTOCOL_DEBUG_LEVEL=verbose
export LOG_LEVEL=DEBUG
```

### Health Checks

Monitor protocol health via endpoints:
- MCP Health: `GET /api/protocol/mcp/health`
- ACP Health: `GET /api/protocol/acp/health` 
- A2A Health: `GET /api/protocol/a2a/health`

## Future Roadmap

### Planned Protocol Support

- **FIPA (Foundation for Intelligent Physical Agents)**: Standard agent communication
- **JADE Protocol**: Java Agent Development Framework integration
- **OPC UA**: Industrial automation and IoT integration
- **ROS2**: Robotics and autonomous systems integration

### Enhanced Features

- **Protocol Auto-Discovery**: Automatic detection and configuration of available protocols
- **Adaptive Load Balancing**: Dynamic load distribution across protocol endpoints
- **Machine Learning Optimization**: AI-driven protocol selection and optimization
- **Blockchain Integration**: Decentralized agent coordination and verification

## Support & Resources

- **Documentation**: [docs.nis-protocol.org/protocols](https://docs.nis-protocol.org/protocols)
- **API Reference**: [api.nis-protocol.org](https://api.nis-protocol.org)
- **Community Forum**: [community.nis-protocol.org](https://community.nis-protocol.org)
- **GitHub Issues**: [github.com/nis-protocol/core/issues](https://github.com/nis-protocol/core/issues)

For technical support, contact: protocols@nis-protocol.org 
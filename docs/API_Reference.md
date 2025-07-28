# NIS Protocol v3 API Reference

## Overview

The NIS Protocol v3 provides a comprehensive REST API for AI agent coordination, consciousness modeling, and third-party protocol integration. All endpoints support real LLM integration with multiple providers.

**Base URL**: `http://localhost:8000` (development) or `http://localhost/api` (through Nginx)

## Authentication

Currently using development mode. Production deployment will require API keys.

## Third-Party Protocol Integration

NIS Protocol v3 supports integration with external AI systems through standardized protocols:

- **MCP (Model Context Protocol)**: Anthropic's protocol for connecting AI systems to data sources
- **ACP (Agent Communication Protocol)**: IBM's standardized protocol for agent communication  
- **A2A (Agent2Agent Protocol)**: Google's protocol for agent interoperability across platforms

## Core Endpoints

### 1. System Status

#### `GET /`
**Description**: Root endpoint with system information and archaeological discovery pattern status

**Response**:
```json
{
  "system": "NIS Protocol v3.1",
  "version": "3.1.0-archaeological",
  "pattern": "OpenAIZChallenge Archaeological Discovery Platform",
  "status": "operational",
  "real_llm_integrated": true,
  "provider": "anthropic",
  "model": "claude-3-sonnet-20240229",
  "features": [
    "Real LLM Integration (OpenAI, Anthropic)",
    "Archaeological Discovery Patterns",
    "Multi-Agent Coordination",
    "Physics-Informed Reasoning", 
    "Consciousness Modeling",
    "Cultural Heritage Analysis",
    "Third-Party Protocol Integration",
    "Automated Audit Fixing"
  ],
  "archaeological_success": "Proven patterns from successful heritage platform",
  "timestamp": 1753654416.629
}
```

#### `GET /health`
**Description**: Health check endpoint with provider and system status

**Response**:
```json
{
  "status": "healthy",
  "timestamp": 1753654416.629,
  "provider": "anthropic",
  "model": "claude-3-sonnet-20240229",
  "real_ai": true,
  "conversations_active": 3,
  "agents_registered": 9,
  "tools_available": 4,
  "pattern": "archaeological_discovery"
}
```

### 2. Chat & Conversation

#### `POST /chat`
**Description**: Enhanced chat with real LLM integration using archaeological discovery patterns

**Request Body**:
```json
{
  "message": "How does the consciousness modeling work?",
  "user_id": "user_123",
  "conversation_id": "conv_456",
  "context": {
    "domain": "consciousness",
    "priority": "high"
  }
}
```

**Response**:
```json
{
  "response": "The consciousness modeling in NIS Protocol uses enhanced meta-cognitive processing (implemented) (implemented)...",
  "user_id": "user_123",
  "conversation_id": "conv_user_123_1753654416_abc123",
  "timestamp": 1753654416.629,
  "confidence": 0.92,
  "provider": "anthropic",
  "real_ai": true,
  "model": "claude-3-sonnet-20240229",
  "tokens_used": 245,
  "reasoning_trace": [
    "archaeological_pattern",
    "context_analysis", 
    "llm_generation",
    "response_synthesis"
  ]
}
```

### 3. Agent Management

#### `POST /agent/create`
**Description**: Create specialized AI agents with real LLM backing

**Request Body**:
```json
{
  "agent_type": "consciousness",
  "capabilities": ["self_reflection", "meta_cognition", "integrity_monitoring"],
  "memory_size": 1000,
  "tools": ["audit_scanner", "violation_detector"],
  "config": {
    "enable_self_audit": true,
    "reflection_interval": 60.0,
    "consciousness_level": "enhanced"
  }
}
```

**Response**:
```json
{
  "agent_id": "agent_consciousness_1753654416_e734c5c4",
  "status": "created",
  "agent_type": "consciousness",
  "capabilities": ["self_reflection", "meta_cognition", "integrity_monitoring"],
  "real_ai_backed": true,
  "provider": "anthropic",
  "model": "claude-3-sonnet-20240229",
  "pattern": "archaeological_discovery",
  "created_at": 1753654416.629
}
```

#### `POST /agent/behavior/{agent_id}`
**Description**: Set behavior mode for an agent (lazy, normal, hyperactive)

**Request Body**:
```json
{
  "mode": "hyperactive"
}
```

**Response**:
```json
{
  "agent_id": "agent_123",
  "behavior_mode": "hyperactive",
  "status": "updated"
}
```

#### `GET /agents`
**Description**: List all registered agents with provider distribution

**Response**:
```json
{
  "agents": {
    "agent_consciousness_1753654416_e734c5c4": {
      "agent_id": "agent_consciousness_1753654416_e734c5c4",
      "agent_type": "consciousness",
      "capabilities": ["self_reflection", "meta_cognition"],
      "status": "active",
      "real_ai_backed": true,
      "provider": "anthropic"
    }
  },
  "total_count": 9,
  "active_agents": 9,
  "real_ai_backed": 9,
  "pattern": "archaeological_discovery",
  "provider_distribution": {
    "anthropic": 3,
    "deepseek": 4,
    "google": 2
  }
}
```

### 4. Action Agent & Audit Fixing

#### `POST /agent/action/audit-fix`
**Description**: Start automated audit fixing session using action agents

**Request Body**:
```json
{
  "target_directories": ["src/", "examples/"],
  "fix_strategies": ["hardcoded_value_replacement", "hype_language_correction"],
  "use_third_party_tools": true,
  "protocols": ["mcp", "acp"]
}
```

**Response**:
```json
{
  "session_id": "audit_fix_1753654416_abc123",
  "status": "started",
  "violations_detected": 12,
  "fixes_scheduled": 12,
  "third_party_tools": ["file_editor_tool", "language_correction_agent"],
  "protocols_used": ["mcp", "acp"],
  "estimated_duration": "30s"
}
```

#### `GET /agent/action/audit-fix/{session_id}`
**Description**: Get audit fixing session status and results

**Response**:
```json
{
  "session_id": "audit_fix_1753654416_abc123",
  "status": "completed",
  "duration": 28.5,
  "violations_detected": 12,
  "violations_fixed": 11,
  "violations_failed": 1,
  "success_rate": 0.92,
  "files_modified": ["src/agents/physics/physics_agent.py", "simple_real_chat_test.py"],
  "fixes_by_type": {
    "hardcoded_value": 8,
    "hype_language": 3,
    "documentation_update": 1
  },
  "tools_used": ["mcp:file_editor_tool", "acp:language_correction_agent"],
  "audit_trail": [
    {
      "file": "simple_real_chat_test.py",
      "violation": "confidence=0.95",
      "fix": "confidence=calculate_confidence(factors)",
      "tool": "mcp:file_editor_tool",
      "success": true
    }
  ]
}
```

### 5. Consciousness Agent Introspection

#### `POST /agent/{agent_id}/introspect`
**Description**: Trigger consciousness agent introspection and integrity assessment

**Request Body**:
```json
{
  "reflection_type": "integrity_assessment",
  "target_agent_id": null,
    "context": {
    "scan_codebase": true,
    "target_directories": ["src/"]
  }
}
```

**Response**:
```json
{
  "introspection_id": "intro_1753654416_xyz789",
  "reflection_type": "integrity_assessment",
  "agent_id": "agent_consciousness_1753654416_e734c5c4",
  "findings": {
    "integrity_score": 95.2,
    "violations_detected": 3,
    "codebase_health": "excellent",
    "recommendations": [
      "Fix hardcoded confidence values in simple_real_chat_test.py",
      "Update hype language in documentation"
    ]
  },
  "confidence": 0.94,
  "processing_time": 1.2,
  "auto_corrections_applied": 0,
  "violations_found": [
    {
      "type": "hardcoded_value",
      "file": "simple_real_chat_test.py",
      "severity": "HIGH"
    }
  ]
}
```

#### `POST /agent/{agent_id}/codebase-scan`
**Description**: Trigger proactive codebase integrity scan by consciousness agent

**Request Body**:
```json
{
  "target_directories": ["src/", "examples/"],
  "scan_depth": "deep",
  "auto_fix": false
}
```

**Response**:
```json
{
  "scan_id": "scan_1753654416_def456",
  "agent_id": "agent_consciousness_1753654416_e734c5c4",
  "total_files_scanned": 45,
  "total_violations": 8,
  "violations_by_type": {
    "hardcoded_value": 5,
    "hype_language": 2,
    "unsubstantiated_claim": 1
  },
  "critical_issues": 5,
  "integrity_score": 88.3,
  "scan_timestamp": 1753654416.629,
  "recommendations": [
    "Run automated fixing session to address hardcoded values",
    "Review documentation for unsubstantiated claims"
  ]
}
```

### 6. Third-Party Protocol Integration

#### `POST /protocol/mcp/tool/execute`
**Description**: Execute MCP (Model Context Protocol) tool

**Request Body**:
```json
{
  "tool_name": "file_editor_tool",
  "action": "replace_text",
  "parameters": {
    "file_path": "src/test.py",
    "old_text": "confidence=0.95",
    "new_text": "confidence=calculate_confidence(factors)"
  },
  "context": {
    "violation_type": "hardcoded_value",
    "session_id": "audit_fix_1753654416_abc123"
  }
}
```

**Response**:
```json
{
  "tool_response": {
    "status": "success",
    "content": {
      "file_modified": "src/test.py",
      "changes_applied": 1,
      "backup_created": true
    },
    "error": null
  },
  "execution_time": 0.8,
  "protocol": "mcp",
  "tool_name": "file_editor_tool"
}
```

#### `POST /protocol/acp/agent/communicate`
**Description**: Communicate with ACP (Agent Communication Protocol) agent

**Request Body**:
```json
{
  "agent_id": "language_correction_agent",
  "message": {
    "action": "detect_hype_language",
    "content": "This is a comprehensive well-engineered system with high-quality accuracy",
    "context": "code_documentation"
  }
}
```

**Response**:
```json
{
  "agent_response": {
    "violations_detected": 3,
    "suggestions": [
      {"original": "comprehensive", "replacement": "comprehensive"},
      {"original": "well-engineered", "replacement": "well-engineered"},
      {"original": "high-quality", "replacement": "high"}
    ],
    "confidence": 0.91
  },
  "protocol": "acp",
  "agent_id": "language_correction_agent"
}
```

#### `POST /protocol/a2a/coordinate`
**Description**: Coordinate with A2A (Agent2Agent Protocol) agents across platforms

**Request Body**:
```json
{
  "target_platform": "external_system",
  "coordination_type": "multi_agent_workflow",
  "workflow": {
    "task": "automated_code_review",
    "agents_required": ["code_analyzer", "quality_checker", "documentation_updater"]
  }
}
```

**Response**:
```json
{
  "coordination_id": "coord_1753654416_ghi789",
  "status": "initiated",
  "participating_agents": 3,
  "external_platform": "external_system",
  "workflow_stages": [
    {"stage": "code_analysis", "agent": "code_analyzer", "status": "queued"},
    {"stage": "quality_check", "agent": "quality_checker", "status": "pending"},
    {"stage": "documentation", "agent": "documentation_updater", "status": "pending"}
  ],
  "protocol": "a2a"
}
```

### 7. Memory & Knowledge Management

#### `POST /memory/store`
**Description**: Store information in agent memory systems

**Request Body**:
```json
{
  "key": "audit_findings_2025_01",
  "data": {
    "violations": 8,
    "fixes_applied": 7,
    "success_rate": 0.875
  },
  "agent_id": "agent_consciousness_1753654416_e734c5c4",
  "memory_type": "episodic",
  "retention_days": 30
}
```

#### `GET /memory/query`
**Description**: Query agent memory systems

**Query Parameters**:
- `key`: Memory key to retrieve
- `agent_id`: Target agent ID
- `memory_type`: Type of memory (episodic, semantic, procedural)

### 8. Tool Management

#### `POST /tool/register`
**Description**: Register new tools for agent use

**Request Body**:
```json
{
  "tool_name": "well-engineered_code_analyzer",
  "description": "well-engineered static code analysis tool",
  "capabilities": ["syntax_analysis", "complexity_metrics", "vulnerability_detection"],
  "protocol": "mcp",
  "endpoint": "https://api.example.com/code-analyzer"
}
```

#### `GET /tools`
**Description**: List available tools and their capabilities

**Response**:
```json
{
  "tools": {
    "calculator": {"description": "Mathematical calculator", "status": "active"},
    "web_search": {"description": "Web search capabilities", "status": "active"},
    "artifact_analysis": {"description": "Archaeological artifact analysis", "status": "active"},
    "file_editor_tool": {"description": "MCP file editing tool", "protocol": "mcp", "status": "active"},
    "language_correction_agent": {"description": "ACP language correction", "protocol": "acp", "status": "active"}
  },
  "total_tools": 5,
  "active_tools": 5,
  "protocol_distribution": {
    "native": 3,
    "mcp": 1,
    "acp": 1
  }
}
```

### 9. Monitoring (implemented in src/monitoring/) (see src/Monitoring (implemented in src/monitoring/)/) & Dashboard

#### `GET /dashboard/metrics`
**Description**: Get real-time system metrics and performance data

**Response**:
```json
{
  "metrics": {
    "consciousness_response_time": {
      "current_value": 0.85,
      "target_value": 1.0,
      "unit": "seconds",
      "alert_level": "normal"
    },
    "memory_usage": {
      "current_value": 67.2,
      "target_value": 80.0,
      "unit": "percent",
      "alert_level": "normal"
    },
    "decision_quality": {
      "current_value": 0.92,
      "target_value": 0.85,
      "unit": "score",
      "alert_level": "excellent"
    }
  },
  "system_health": "excellent",
  "uptime": 3847.2
}
```

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (agent/resource doesn't exist)
- `500`: Internal Server Error

Error responses include detailed information:

```json
{
  "detail": "Agent creation failed: Invalid agent type 'invalid_type'",
  "error_type": "ValidationError",
  "timestamp": 1753654416.629
}
```

## Rate Limiting

Production deployments include rate limiting:
- Standard endpoints: 60 requests/minute
- Chat endpoints: 30 requests/minute  
- Tool execution: 10 requests/minute

## WebSocket Support

Real-time features available via WebSocket:
- Live metrics: `ws://localhost:5000/socket.io/`
- Agent communication: `ws://localhost:8000/ws/agents`
- Audit Monitoring (implemented in src/monitoring/) (see src/Monitoring (implemented in src/monitoring/)/): `ws://localhost:8000/ws/audit`

## Integration Examples

### Automated Audit Fixing Workflow

```bash
# 1. Start audit fixing session
curl -X POST "http://localhost/api/agent/action/audit-fix" \
  -H "Content-Type: application/json" \
  -d '{"target_directories": ["src/"], "use_third_party_tools": true}'

# 2. Monitor session progress
curl "http://localhost/api/agent/action/audit-fix/audit_fix_1753654416_abc123"

# 3. Trigger consciousness agent scan
curl -X POST "http://localhost/api/agent/agent_consciousness_1753654416_e734c5c4/codebase-scan" \
  -H "Content-Type: application/json" \
  -d '{"target_directories": ["src/"], "auto_fix": false}'
```

### Third-Party Tool Integration

```bash
# Execute MCP file editing tool
curl -X POST "http://localhost/api/protocol/mcp/tool/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "file_editor_tool",
    "action": "replace_text",
    "parameters": {
      "file_path": "src/test.py",
      "old_text": "confidence=0.95",
      "new_text": "confidence=calculate_confidence(factors)"
    }
  }'
```

## Production Deployment

For production deployment:

1. **Environment Variables**: Configure API keys for all LLM providers
2. **Database**: Set up PostgreSQL for persistent storage
3. **Monitoring (implemented in src/monitoring/) (see src/Monitoring (implemented in src/monitoring/)/)**: Configure Grafana dashboards
4. **Security**: Enable JWT authentication and HTTPS
5. **Scaling**: Use Docker Compose or Kubernetes

## SDK Support

Official SDKs available for:
- Python: `pip install nis-protocol-sdk`
- JavaScript: `npm install nis-protocol-js`
- Go: Import `github.com/nis-protocol/go-sdk` 
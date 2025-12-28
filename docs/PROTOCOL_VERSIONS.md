# NIS Protocol - External Protocol Integration Status

**Last Updated**: December 25, 2025

## Protocol Versions & Status

### 1. MCP (Model Context Protocol)
- **Version**: 2025-11-25 (Latest)
- **Status**: ✅ Active - Donated to Agentic AI Foundation (Linux Foundation)
- **Governance**: Linux Foundation / Agentic AI Foundation
- **Maintainers**: Anthropic, Block, OpenAI, Google, Microsoft, AWS, Cloudflare, Bloomberg
- **Specification**: https://modelcontextprotocol.io/specification/2025-11-25
- **NIS Implementation Status**: ✅ Implemented with built-in tools

**Key Features**:
- Tools with JSON Schema validation
- Resources (data exposure)
- Prompts (templated interactions)
- Transports: stdio, HTTP with SSE
- User approval/consent mechanisms

**NIS Implementation**:
- Built-in MCP tools (9 tools): code_execute, web_search, physics_solve, robotics_kinematics, llm_chat, memory_store, memory_retrieve, consciousness_genesis, vision_analyze
- Endpoints: `/protocol/mcp/tools`, `/protocol/mcp/execute`
- Mode: Local execution (no external MCP server required)

---

### 2. A2A (Agent-to-Agent Protocol)
- **Version**: DRAFT v1.0 (Latest)
- **Status**: ✅ Active - Google-led with 50+ technology partners
- **Governance**: Linux Foundation (merged with ACP)
- **Partners**: Atlassian, Box, Cohere, Intuit, Langchain, MongoDB, PayPal, Salesforce, SAP, ServiceNow, UKG, Workday, Accenture, Deloitte, PwC
- **Specification**: https://a2a-protocol.org/latest/specification/
- **NIS Implementation Status**: ✅ Implemented with local task execution

**Key Features**:
- Core operations: SendMessage, SendStreamingMessage, GetTask, ListTasks, CancelTask
- Protocol bindings: JSON-RPC, gRPC, HTTP+JSON/REST
- Agent Card for discovery
- Task tracking with artifacts
- Multi-turn interactions
- Streaming support

**NIS Implementation**:
- Local task execution through consciousness pipeline
- Endpoints: `/protocol/a2a/create-task`, `/protocol/a2a/task/{id}`, `/protocol/a2a/tasks`
- Mode: Local execution using NIS Protocol agents
- Task storage: In-memory with full lifecycle tracking

---

### 3. ACP (Agent Communication Protocol)
- **Version**: N/A
- **Status**: ⚠️ DEPRECATED - Merged into A2A
- **Governance**: Previously IBM Research / Linux Foundation
- **Migration**: All ACP functionality now part of A2A
- **Specification**: https://agentcommunicationprotocol.dev (redirects to A2A)
- **NIS Implementation Status**: ⚠️ Redirected to A2A implementation

**Important Notice**:
> ACP has merged with A2A under the Linux Foundation umbrella. The ACP team wound down active development and contributed its technology and expertise to A2A. All new implementations should use A2A.

**NIS Implementation**:
- ACP endpoints redirect to A2A implementation
- `/protocol/acp/run` executes through A2A task system
- Full backward compatibility maintained

---

## Implementation Summary

| Protocol | Version | Status | NIS Endpoints | Execution Mode |
|----------|---------|--------|---------------|----------------|
| **MCP** | 2025-11-25 | ✅ Active | `/protocol/mcp/*` | Built-in tools |
| **A2A** | DRAFT v1.0 | ✅ Active | `/protocol/a2a/*` | Local tasks |
| **ACP** | Deprecated | ⚠️ Merged | `/protocol/acp/*` | Redirects to A2A |

---

## Protocol Health Status

Check protocol health: `GET /protocol/health`

```json
{
  "status": "success",
  "protocols": {
    "mcp": {
      "protocol": "mcp",
      "mode": "builtin",
      "healthy": true,
      "initialized": true,
      "tools_available": 9,
      "version": "2025-11-25"
    },
    "a2a": {
      "protocol": "a2a",
      "mode": "local",
      "healthy": true,
      "initialized": true,
      "version": "DRAFT v1.0"
    },
    "acp": {
      "protocol": "acp",
      "mode": "deprecated",
      "healthy": true,
      "redirects_to": "a2a",
      "note": "ACP merged into A2A - use A2A endpoints"
    }
  },
  "overall_healthy": true
}
```

---

## Migration Guide

### For External MCP Server Users
If you were using an external MCP server:
1. NIS Protocol now provides built-in MCP tools
2. No external server required
3. All tools execute locally through NIS agents
4. Use `/protocol/mcp/execute` for tool execution

### For ACP Users
If you were using ACP:
1. ACP has been deprecated and merged into A2A
2. Update endpoints from `/protocol/acp/*` to `/protocol/a2a/*`
3. Use A2A task model instead of ACP run model
4. Agent Card discovery now follows A2A spec

### For A2A Users
If you're using A2A:
1. Ensure you're using DRAFT v1.0 specification
2. Use task-based model for async operations
3. Implement Agent Card for discovery
4. Support streaming for real-time updates

---

## References

- **MCP Specification**: https://modelcontextprotocol.io/specification/2025-11-25
- **A2A Specification**: https://a2a-protocol.org/latest/specification/
- **ACP Migration Guide**: https://github.com/i-am-bee/beeai-platform/blob/main/docs/community-and-support/acp-a2a-migration-guide.mdx
- **Agentic AI Foundation**: https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation

---

## Next Steps for NIS Protocol

1. ✅ MCP: Fully compliant with 2025-11-25 spec
2. 🔄 A2A: Update to full DRAFT v1.0 compliance (add Agent Card, streaming)
3. ✅ ACP: Deprecated, redirected to A2A
4. 📋 Add protocol version negotiation headers
5. 📋 Implement capability discovery per spec
6. 📋 Add comprehensive protocol tests

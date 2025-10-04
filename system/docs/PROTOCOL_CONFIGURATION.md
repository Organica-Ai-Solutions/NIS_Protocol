# Third-Party Protocol Configuration Guide

## Overview
This document describes how to configure the NIS Protocol's integration with third-party AI agent communication protocols: MCP, A2A, and ACP.

---

## Environment Variables

### MCP (Model Context Protocol) - Anthropic

```bash
# MCP Server Configuration
MCP_SERVER_URL=http://localhost:3000
MCP_TIMEOUT=30
```

**Description:**
- `MCP_SERVER_URL`: URL of your MCP server endpoint
- `MCP_TIMEOUT`: Request timeout in seconds (default: 30)

**Default Behavior:**
If not configured, MCP adapter will attempt to connect to `http://localhost:3000`

---

### A2A (Agent2Agent Protocol) - Google

```bash
# A2A Service Configuration
A2A_BASE_URL=https://api.google.com/a2a/v1
A2A_API_KEY=your-google-a2a-api-key
A2A_TIMEOUT=30
```

**Description:**
- `A2A_BASE_URL`: Base URL of the A2A service endpoint
- `A2A_API_KEY`: Your A2A API key for authentication
- `A2A_TIMEOUT`: Request timeout in seconds (default: 30)

**Default Behavior:**
If not configured, A2A adapter initializes but will fail authentication without a valid API key.

---

### ACP (Agent Communication Protocol) - IBM

```bash
# ACP Agent Configuration
ACP_BASE_URL=http://localhost:8080
ACP_API_KEY=your-ibm-acp-api-key
ACP_TIMEOUT=30
```

**Description:**
- `ACP_BASE_URL`: Base URL of your ACP agent endpoint
- `ACP_API_KEY`: Your ACP API key for authentication (optional for local agents)
- `ACP_TIMEOUT`: Request timeout in seconds (default: 30)

**Default Behavior:**
If not configured, ACP adapter will attempt to connect to `http://localhost:8080`

---

## Docker Configuration

If running NIS Protocol in Docker, add these to your `.env` file or `docker-compose.yml`:

```yaml
services:
  nis-protocol:
    environment:
      # MCP Configuration
      - MCP_SERVER_URL=http://mcp-server:3000
      - MCP_TIMEOUT=30
      
      # A2A Configuration
      - A2A_BASE_URL=https://api.google.com/a2a/v1
      - A2A_API_KEY=${A2A_API_KEY}
      - A2A_TIMEOUT=30
      
      # ACP Configuration
      - ACP_BASE_URL=http://acp-agent:8080
      - ACP_API_KEY=${ACP_API_KEY}
      - ACP_TIMEOUT=30
```

---

## Production Configuration

### Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** for all sensitive configuration
3. **Rotate API keys** regularly
4. **Use HTTPS** for all external protocol connections
5. **Implement rate limiting** for protocol endpoints

### Recommended Settings

```bash
# Production MCP
MCP_SERVER_URL=https://mcp.your-domain.com
MCP_TIMEOUT=60  # Longer timeout for production

# Production A2A
A2A_BASE_URL=https://api.google.com/a2a/v1
A2A_TIMEOUT=90  # Longer for async tasks

# Production ACP
ACP_BASE_URL=https://acp.your-domain.com
ACP_TIMEOUT=60
```

---

## Testing Configuration

### Local Development

```bash
# Local MCP Server
MCP_SERVER_URL=http://localhost:3000
MCP_TIMEOUT=10

# Mock A2A (for testing)
A2A_BASE_URL=http://localhost:4000
A2A_API_KEY=test-key
A2A_TIMEOUT=10

# Local ACP Agent
ACP_BASE_URL=http://localhost:8080
ACP_API_KEY=test-key
ACP_TIMEOUT=10
```

### Integration Testing

For integration tests, you can override configuration programmatically:

```python
from src.adapters.mcp_adapter import MCPAdapter

# Custom configuration for testing
test_config = {
    "server_url": "http://test-mcp:3000",
    "timeout": 5,
    "failure_threshold": 3
}

adapter = MCPAdapter(test_config)
```

---

## API Endpoints

Once configured, the following endpoints become available:

### MCP Endpoints

- `POST /protocol/mcp/initialize` - Initialize MCP connection
- `GET /protocol/mcp/tools` - Discover available tools
- `POST /protocol/mcp/call-tool` - Execute an MCP tool

### A2A Endpoints

- `POST /protocol/a2a/create-task` - Create a task on external agent
- `GET /protocol/a2a/task/{task_id}` - Get task status
- `DELETE /protocol/a2a/task/{task_id}` - Cancel a task

### ACP Endpoints

- `GET /protocol/acp/agent-card` - Get NIS Protocol Agent Card
- `POST /protocol/acp/execute` - Execute external ACP agent

### General Endpoints

- `GET /protocol/health` - Health status of all protocol adapters
- `POST /protocol/translate` - Translate messages between protocols

---

## Health Monitoring

Check protocol adapter health:

```bash
curl http://localhost:5000/protocol/health
```

Response:
```json
{
  "status": "success",
  "protocols": {
    "mcp": {
      "protocol": "mcp",
      "healthy": true,
      "circuit_breaker": {
        "state": "closed",
        "failures": 0
      },
      "metrics": {
        "success_rate": 0.95,
        "total_requests": 100,
        "avg_response_time": 0.523
      }
    },
    "a2a": { ... },
    "acp": { ... }
  },
  "overall_healthy": true
}
```

---

## Troubleshooting

### Common Issues

**1. "MCP adapter not initialized"**
- Check `MCP_SERVER_URL` is correct
- Ensure MCP server is running
- Verify network connectivity

**2. "A2A authentication failed"**
- Verify `A2A_API_KEY` is valid
- Check API key hasn't expired
- Ensure correct API endpoint URL

**3. "Circuit breaker open"**
- Too many failed requests
- Check service availability
- Reset with: `adapter.reset_circuit_breaker()`

**4. "Connection timeout"**
- Increase timeout value
- Check network latency
- Verify service is responsive

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show all protocol adapter activity including:
- Request/response details
- Retry attempts
- Circuit breaker state changes
- Performance metrics

---

## Advanced Configuration

### Circuit Breaker Settings

Customize circuit breaker behavior:

```python
mcp_config = {
    "server_url": "http://localhost:3000",
    "failure_threshold": 5,  # Opens after 5 failures
    "recovery_timeout": 60,  # Try recovery after 60s
    "success_threshold": 2   # Close after 2 successes
}
```

### Custom Timeout Per Operation

Different operations can have different timeouts:

```python
# Short timeout for health checks
await adapter.call_tool("health_check", {}, timeout=5)

# Long timeout for heavy computation
await adapter.call_tool("analyze_dataset", params, timeout=300)
```

---

## Migration Guide

### From Mock to Real Protocols

1. **Remove any mock implementations**
2. **Configure environment variables** for target protocols
3. **Test connectivity** using health endpoint
4. **Deploy real protocol servers** (MCP, ACP) if needed
5. **Verify end-to-end** with integration tests

### From Development to Production

1. **Switch URLs** from localhost to production endpoints
2. **Update API keys** to production credentials
3. **Increase timeouts** for production workloads
4. **Enable monitoring** and alerting
5. **Test failover** and circuit breaker behavior

---

## Support

For issues or questions:
- Check logs for detailed error messages
- Review health endpoint for adapter status
- Consult protocol-specific documentation:
  - [MCP Specification](https://modelcontextprotocol.io)
  - [A2A Documentation](https://agent2agent.dev)
  - [ACP Specification](https://github.com/ibm/agent-communication-protocol)

---

*Last Updated: 2025-10-01*
*NIS Protocol v3.2*


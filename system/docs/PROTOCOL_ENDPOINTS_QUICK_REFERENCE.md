# Protocol Endpoints Quick Reference

## Third-Party Protocol Integration - Production API

All endpoints are now live in `main.py` under the `"Third-Party Protocols"` tag.

---

## 🔌 MCP Protocol (Anthropic)

### Initialize Connection
```http
POST /protocol/mcp/initialize
```
Returns server capabilities and initializes connection.

### Discover Tools
```http
GET /protocol/mcp/tools
```
Lists all available MCP tools from the connected server.

### Execute Tool
```http
POST /protocol/mcp/call-tool
Content-Type: application/json

{
  "tool_name": "calculator_arithmetic",
  "arguments": {
    "expression": "2 + 2"
  }
}
```

---

## 🤝 A2A Protocol (Google)

### Create Task
```http
POST /protocol/a2a/create-task
Content-Type: application/json

{
  "description": "Analyze customer feedback",
  "agent_id": "analyzer-agent-123",
  "parameters": {
    "dataset": "customer_feedback_q4"
  },
  "callback_url": "https://your-server.com/callbacks"
}
```

### Get Task Status
```http
GET /protocol/a2a/task/{task_id}
```

### Cancel Task
```http
DELETE /protocol/a2a/task/{task_id}
```

---

## 🔧 ACP Protocol (IBM)

### Get Agent Card
```http
GET /protocol/acp/agent-card
```
Returns NIS Protocol Agent Card for offline discovery.

### Execute Agent
```http
POST /protocol/acp/execute
Content-Type: application/json

{
  "agent_url": "http://external-agent:8080",
  "message": {
    "action": "validate_physics",
    "data": {...}
  },
  "async_mode": true
}
```

---

## 🏥 Health & Monitoring

### Protocol Health
```http
GET /protocol/health
```
Returns health status of all protocol adapters including:
- Circuit breaker state
- Success rates
- Average response times
- Error counts

---

## 🔄 Message Translation

### Translate ACP Messages
```http
POST /protocol/translate
Content-Type: application/json

{
  "message": {
    "protocol": "nis",
    "payload": {...}
  },
  "target_protocol": "acp"
}
```

---

## Quick Test

Start the server and test:

```bash
# 1. Start NIS Protocol
./start.sh

# 2. Check protocol health
curl http://localhost:5000/protocol/health

# 3. Get ACP Agent Card
curl http://localhost:5000/protocol/acp/agent-card
```

---

## Configuration

Add to your `.env` file (see `configs/protocol.env.example`):

```bash
MCP_SERVER_URL=http://localhost:3000
A2A_BASE_URL=https://api.google.com/a2a/v1
A2A_API_KEY=your-key
ACP_BASE_URL=http://localhost:8080
```

---

## Error Handling

All endpoints return standard HTTP status codes:
- `200` - Success
- `400` - Invalid request
- `503` - Service unavailable (circuit breaker open)
- `504` - Timeout
- `500` - Server error

Circuit breaker automatically opens after 5 consecutive failures and recovers after 60 seconds.

---

**Full Documentation:** See `system/docs/THIRD_PARTY_PROTOCOL_INTEGRATION.md`


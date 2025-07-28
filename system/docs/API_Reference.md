# NIS Protocol v3.1 API Reference

## 1. Overview

The NIS Protocol v3.1 provides a RESTful API for interacting with its **Generative Simulation Engine** and multi-agent system. The API is designed to be simple, robust, and easy to integrate.

**Base URL:** `http://localhost:8000`

## 2. Generative Simulation

### `POST /simulation/run`
**Description:** The flagship endpoint of the NIS Protocol v3.1. It runs the full **design → simulate → analyze** loop for a given natural language concept, leveraging the verifiable AI pipeline and multi-agent system.

**`curl` Example:**
```bash
curl -X POST http://localhost:8000/simulation/run \
  -H "Content-Type: application/json" \
  -d '{"concept": "a drone wing that mimics a falcon s"}'
```

**Request Body:**
```json
{
  "concept": "string"
}
```

**Example Response (`200 OK`):**
```json
{
  "status": "completed",
  "message": "Modulus simulation and inference completed successfully.",
  "key_metrics": {
    "average_pressure": -0.001,
    "max_velocity": 1.0
  },
  "results_file": "data/simulation_results/simulation_1678886400.csv"
}
```

## 3. Core Endpoints

### System Status

#### `GET /`
**Description:** Returns the current status and identification of the NIS Protocol system.

**`curl` Example:**
```bash
curl -X GET http://localhost:8000/
```

**Example Response (`200 OK`):**
```json
{
  "system": "NIS Protocol v3.1",
  "version": "3.1.0-archaeological",
  "pattern": "nis_v3_agnostic",
  "status": "operational",
  "real_llm_integrated": { /* ... */ },
  "provider": { /* ... */ },
  "model": { /* ... */ },
  "features": [
    "Generative Simulation Engine",
    "Verifiable AI Pipeline (Laplace → KAN → PINN)",
    "Offline-First Capabilities with BitNet",
    "Multi-Agent Coordination"
  ],
  "archaeological_success": "Proven patterns from successful heritage platform",
  "timestamp": 1753693993.896
}
```

#### `GET /health`
**Description:** Provides a detailed health check of the system, including the status of LLM providers and the number of registered agents.

**`curl` Example:**
```bash
curl -X GET http://localhost:8000/health
```

**Example Response (`200 OK`):**
```json
{
  "status": "healthy",
  "timestamp": 1753693994.123,
  "provider": { /* ... */ },
  "model": { /* ... */ },
  "real_ai": { /* ... */ },
  "conversations_active": 1,
  "agents_registered": 4,
  "tools_available": 4,
  "pattern": "nis_v3_agnostic"
}
```

### Chat and Processing

#### `POST /chat`
**Description:** The primary endpoint for interacting with the protocol. It takes a user message, runs it through the full **Laplace → KAN → PINN → LLM** pipeline, and returns a validated, natural language response. This endpoint will automatically fall back to the local BitNet model if online providers are unavailable.

**`curl` Example:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain the significance of the PINN validation layer."}'
```

**Request Body:**
```json
{
  "message": "string",
  "user_id": "string (optional)",
  "conversation_id": "string (optional)",
  "context": "object (optional)",
  "agent_type": "string (optional, default: 'default')"
}
```

**Example Response (`200 OK`):**
```json
{
  "response": "The PINN validation layer is crucial...",
  "user_id": "anonymous",
  "conversation_id": "conv_anonymous_1753694055_e8a1b2c3",
  "timestamp": 1753694060.123,
  "confidence": 0.95,
  "provider": "deepseek",
  "real_ai": true,
  "model": "deepseek-chat",
  "tokens_used": 350,
  "reasoning_trace": [
    "archaeological_pattern",
    "context_analysis",
    "llm_generation",
    "response_synthesis"
  ]
}
```

#### `POST /chat/async`
**Description:** Provides a streaming response for real-time applications. The functionality is identical to `/chat`, but the response is delivered as a stream of server-sent events. This endpoint also supports BitNet fallback.

**`curl` Example:**
```bash
curl -X POST http://localhost:8000/chat/async \
  -H "Content-Type: application/json" \
  -d '{"message": "What is a Kolmogorov-Arnold Network?"}'
```

**Example Response (stream):**
```
data: A Kolmogorov-Arnold

data: Network (KAN)

data: is a...

...

data: [DONE]
```

### Agent Management

#### `GET /agents`
**Description:** Lists all agents currently registered and active within the NIS Protocol.

**`curl` Example:**
```bash
curl -X GET http://localhost:8000/agents
```

**Example Response (`200 OK`):**
```json
{
      "agents": {
    "laplace_transformer_01": { /* status */ },
    "kan_reasoning_01": { /* status */ },
    "pinn_physics_01": { /* status */ },
    "consciousness_01": { /* status */ }
  },
  "total_count": 4,
  "active_agents": 4,
  "real_ai_backed": 0,
  "pattern": "nis_v3_agnostic",
  "provider_distribution": {
    "unknown": 4
  }
}
```

### System Monitoring

#### `GET /consciousness/status`
**Description:** Returns the current status of the `EnhancedConsciousAgent`, including its operational level and awareness metrics.

**`curl` Example:**
```bash
curl -X GET http://localhost:8000/consciousness/status
```

**Example Response (`200 OK`):**
```json
{
  "consciousness_level": "enhanced",
  "introspection_active": true,
  "awareness_metrics": {
    "self_awareness": 0.85,
    "environmental_awareness": 0.92
  }
}
```

#### `GET /infrastructure/status`
**Description:** Provides a status check of the system's core infrastructure services.

**`curl` Example:**
```bash
curl -X GET http://localhost:8000/infrastructure/status
```

**Example Response (`200 OK`):**
```json
{
  "status": "healthy",
  "active_services": ["llm", "memory", "agents"],
  "resource_usage": {
    "cpu": 45.2,
    "memory": "2.1GB"
  }
}
```

#### `GET /metrics`
**Description:** Returns key performance metrics for the running application, such as uptime.

**`curl` Example:**
```bash
curl -X GET http://localhost:8000/metrics
```

**Example Response (`200 OK`):**
```json
{
  "uptime": 1234.56,
  "total_requests": 100,
  "average_response_time": 0.15
}
```

## 4. Error Handling

The API returns standard HTTP status codes to indicate the success or failure of a request.

- `200 OK`: The request was successful.
- `404 Not Found`: The requested resource could not be found.
- `422 Unprocessable Entity`: The request body is invalid.
- `500 Internal Server Error`: An unexpected error occurred on the server.

Error responses will contain a `detail` field with a description of the error. 
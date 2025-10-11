# 🚀 Multi-Agent System Quick Reference Guide
**Based on NIS Protocol v3.2.3 Architecture**

---

## 📋 Core Patterns Cheat Sheet

### 1. Agent Hierarchy (Brain-Inspired)

```
🧠 Always Active (Brain Stem)
   ├── Signal Processing
   ├── Reasoning
   ├── Physics Validation
   ├── Memory Management
   └── Coordination

💡 Context-Activated (Cerebral Cortex)
   ├── Vision Analysis
   ├── Document Processing
   ├── Web Research
   └── Domain Specialists

🔌 Event-Driven (Nervous System)
   ├── MCP Protocol
   ├── A2A Protocol
   └── Message Routing

📚 Adaptive (Hippocampus)
   ├── Continuous Learning
   └── Model Training
```

**Key Insight**: Only activate what you need, when you need it.

---

## ⚡ Performance Optimization

### Intelligent Query Routing

| Path | Latency | Use When |
|------|---------|----------|
| FAST | 2-3s | Simple questions, greetings |
| STANDARD | 5-10s | Normal queries, technical |
| FULL | 10-15s | Complex analysis, physics |

**Performance Boost**: 83% faster on simple queries!

**Implementation:**
```python
def route_query(query: str) -> str:
    if is_simple_chat(query):
        return "FAST"  # Skip heavy processing
    elif is_complex_analysis(query):
        return "FULL"   # All processing
    else:
        return "STANDARD"  # Balanced
```

---

## 🏗️ Infrastructure Stack

```yaml
Services:
  ✅ Backend API (FastAPI)
  ✅ Message Queue (Kafka)
  ✅ Cache Layer (Redis)
  ✅ Database (PostgreSQL)
  ✅ Reverse Proxy (Nginx)
  ✅ Monitoring (Grafana/Prometheus)
```

**Why This Stack:**
- **Kafka**: Decoupled agent communication
- **Redis**: Fast shared memory
- **PostgreSQL**: Persistent storage
- **Nginx**: Load balancing + SSL
- **Grafana**: Real-time monitoring

---

## 🎯 Agent Registration Pattern

```python
class Orchestrator:
    def register_agent(self, 
                      agent_id: str,
                      capabilities: List[str],
                      instance: Any):
        """
        Register agent with capabilities
        
        Args:
            agent_id: Unique identifier
            capabilities: What the agent can do
            instance: Agent class instance
        """
        self.agents[agent_id] = {
            'instance': instance,
            'capabilities': capabilities,
            'status': 'ready',
            'load': 0.0
        }
        
        # Index by capability
        for cap in capabilities:
            self.capability_index[cap].add(agent_id)
```

**Capability Examples:**
- `memory_storage`, `memory_retrieval`
- `physics_validation`, `pinn_solving`
- `image_analysis`, `object_detection`
- `web_research`, `fact_checking`

---

## 🔄 Communication Patterns

### 1. Point-to-Point (Kafka Topics)

```python
# Producer (Agent A)
await kafka.send(
    topic="physics_validation",
    message={
        "task_id": "123",
        "data": scenario,
        "from_agent": "coordinator"
    }
)

# Consumer (Agent B)
async for message in kafka.consume("physics_validation"):
    result = await process_validation(message)
    await kafka.send("physics_results", result)
```

### 2. Broadcast (Redis PubSub)

```python
# Publish to all listeners
await redis.publish("system_event", {
    "type": "agent_registered",
    "agent_id": "new_agent_001"
})

# Subscribe
async for message in redis.subscribe("system_event"):
    await handle_event(message)
```

### 3. Shared Memory (Redis Store)

```python
# Store shared context
await redis.setex(
    key=f"context:{conversation_id}",
    value=json.dumps(context),
    ttl=3600  # 1 hour
)

# Retrieve
context = json.loads(await redis.get(f"context:{conversation_id}"))
```

---

## 📊 Metrics to Track

```python
metrics = {
    # Performance
    "avg_response_time": 0.0,
    "p95_latency": 0.0,
    "p99_latency": 0.0,
    
    # Reliability
    "success_rate": 0.0,
    "error_rate": 0.0,
    "uptime": 0.0,
    
    # Resource Usage
    "cpu_utilization": 0.0,
    "memory_usage": 0.0,
    "api_calls": 0,
    
    # Agent Specific
    "agent_utilization": {},
    "capability_usage": {},
    "load_balance_score": 0.0
}
```

**Dashboard Alerts:**
- Response time > 15s
- Error rate > 5%
- Agent offline > 5 minutes
- Memory usage > 80%

---

## 🛡️ Failure Handling

### Graceful Degradation Pattern

```python
async def execute_task(task):
    try:
        # Primary: Full-featured execution
        return await primary_agent.execute(task)
    
    except AgentOverloaded:
        # Fallback 1: Secondary agent
        return await backup_agent.execute(task)
    
    except AgentUnavailable:
        # Fallback 2: Degraded mode
        return await degraded_mode_execute(task)
    
    except Exception as e:
        # Fallback 3: Best-effort response
        log_error(e)
        return generate_fallback_response(task)
```

**Fallback Levels:**
1. **Primary**: Full features, best quality
2. **Secondary**: Reduced features, good quality
3. **Degraded**: Minimal features, acceptable quality
4. **Fallback**: Error recovery, graceful failure

---

## 🚀 Quick Start Template

### Minimal Orchestrator (30 lines)

```python
from collections import defaultdict

class SimpleOrchestrator:
    def __init__(self):
        self.agents = {}
        self.capabilities = defaultdict(set)
    
    def register(self, agent_id, capabilities, instance):
        """Register an agent"""
        self.agents[agent_id] = instance
        for cap in capabilities:
            self.capabilities[cap].add(agent_id)
    
    async def route(self, task, required_caps):
        """Route task to capable agent"""
        # Find agents with all required capabilities
        capable = set.intersection(*[
            self.capabilities[cap] 
            for cap in required_caps
        ])
        
        if not capable:
            raise ValueError(f"No agent for: {required_caps}")
        
        # Simple selection: first available
        agent_id = next(iter(capable))
        return await self.agents[agent_id].execute(task)

# Usage
orchestrator = SimpleOrchestrator()
orchestrator.register("physics", ["validation"], PhysicsAgent())
result = await orchestrator.route(task, ["validation"])
```

---

## 🎓 Implementation Checklist

### Phase 1: Foundation (Week 1-2)
- [ ] Define agent hierarchy (3-5 agents)
- [ ] Implement basic orchestrator
- [ ] Set up agent registration
- [ ] Create simple routing logic
- [ ] Add basic logging

### Phase 2: Intelligence (Week 3-4)
- [ ] Capability-based routing
- [ ] Query classification
- [ ] Performance metrics
- [ ] Health monitoring
- [ ] Error handling

### Phase 3: Infrastructure (Week 5-6)
- [ ] Redis integration
- [ ] Message queue (Kafka/RabbitMQ)
- [ ] Docker Compose
- [ ] Load balancing
- [ ] Database setup

### Phase 4: Production (Week 7-8)
- [ ] Full monitoring stack
- [ ] Auto-scaling
- [ ] Security hardening
- [ ] Documentation
- [ ] Load testing

---

## 💡 Common Patterns

### Pattern 1: Capability Query

```python
def find_agents_for_task(task_type: str) -> List[str]:
    """Find all agents capable of handling task"""
    return [
        agent_id 
        for agent_id, info in self.agents.items()
        if task_type in info['capabilities']
    ]
```

### Pattern 2: Load Balancing

```python
def select_best_agent(capable_agents: List[str]) -> str:
    """Select agent with lowest load"""
    return min(
        capable_agents,
        key=lambda id: self.agents[id]['load']
    )
```

### Pattern 3: Circuit Breaker

```python
class CircuitBreaker:
    def __init__(self, threshold=5, timeout=60):
        self.failures = 0
        self.threshold = threshold
        self.timeout = timeout
        self.last_failure = None
        self.state = "CLOSED"
    
    async def call(self, func):
        if self.state == "OPEN":
            if time.time() - self.last_failure > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitOpen("Service unavailable")
        
        try:
            result = await func()
            self.failures = 0
            self.state = "CLOSED"
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure = time.time()
            if self.failures >= self.threshold:
                self.state = "OPEN"
            raise
```

### Pattern 4: Retry with Backoff

```python
async def retry_with_backoff(func, max_retries=3):
    """Exponential backoff retry"""
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            await asyncio.sleep(wait_time)
```

---

## 🔧 Configuration Examples

### Environment Variables

```bash
# Agent Configuration
AGENT_MAX_CONCURRENT=10
AGENT_TIMEOUT=120
AGENT_RETRY_ATTEMPTS=3

# Infrastructure
KAFKA_SERVERS=kafka:9092
REDIS_HOST=redis
REDIS_PORT=6379
DATABASE_URL=postgresql://user:pass@db:5432/nis

# Performance
ENABLE_QUERY_ROUTER=true
ENABLE_CACHING=true
CACHE_TTL=3600

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_LEVEL=INFO
```

### Docker Compose Minimal

```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - KAFKA_SERVERS=kafka:9092
    depends_on:
      - redis
      - kafka

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://kafka:9092
```

---

## 📚 Key Resources

### NIS Protocol Files
- `src/agents/master_agent_orchestrator.py` - Main orchestrator
- `src/core/query_router.py` - Intelligent routing
- `src/agents/agent_router.py` - Capability routing
- `system/docs/AGENT_ORCHESTRATION.md` - Full docs

### External References
- **LangGraph**: Multi-agent workflows
- **Apache Kafka**: Event streaming
- **FastAPI**: High-performance APIs
- **Docker Compose**: Infrastructure as code

---

## 🎯 Quick Wins

### 1. Add Intelligent Routing (30 min)
```python
def classify_query(query: str) -> str:
    if len(query.split()) < 5:
        return "FAST"
    elif "analyze" in query.lower():
        return "FULL"
    return "STANDARD"
```

### 2. Add Basic Metrics (15 min)
```python
from time import time

class Metrics:
    def __init__(self):
        self.count = 0
        self.total_time = 0
    
    def track(self, duration):
        self.count += 1
        self.total_time += duration
    
    @property
    def avg_time(self):
        return self.total_time / self.count if self.count > 0 else 0
```

### 3. Add Health Check (10 min)
```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agents": len(orchestrator.agents),
        "uptime": time.time() - start_time
    }
```

---

## 🌟 Success Metrics

Track these to measure system health:

| Metric | Target | Critical |
|--------|--------|----------|
| **Avg Response Time** | < 5s | > 15s |
| **Success Rate** | > 95% | < 85% |
| **Agent Uptime** | > 99% | < 95% |
| **Error Rate** | < 1% | > 5% |
| **CPU Usage** | < 70% | > 90% |

---

## 💭 Remember

1. **Start Simple**: Don't build everything at once
2. **Measure First**: Add metrics before optimizing
3. **Fail Gracefully**: Always have fallbacks
4. **Document Everything**: Future you will thank you
5. **Test Continuously**: Break things early, not in production

---

**"Build impressive systems, describe them accurately, deploy them reliably."**

🚀 Happy Building!


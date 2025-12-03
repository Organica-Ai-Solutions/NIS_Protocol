# NIS Protocol Performance Tuning Guide

**Version:** 1.0 | **Last Updated:** December 2025

---

## Table of Contents

1. [Performance Baseline](#performance-baseline)
2. [API Optimization](#api-optimization)
3. [Infrastructure Tuning](#infrastructure-tuning)
4. [LLM Optimization](#llm-optimization)
5. [Robotics Performance](#robotics-performance)
6. [Memory Management](#memory-management)
7. [Monitoring & Profiling](#monitoring--profiling)

---

## Performance Baseline

### Current Performance Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Average Response Time | 17ms | <50ms |
| P95 Latency | 35ms | <100ms |
| P99 Latency | 66ms | <200ms |
| Throughput | 60 req/s | >50 req/s |
| Error Rate | <1% | <1% |

### Benchmark Results

```
Endpoint                          Avg (ms)   P95 (ms)
─────────────────────────────────────────────────────
/health                           16.56      26.82
/system/status                    10.62      23.18
/robotics/capabilities            12.48      29.02
/robotics/forward_kinematics      20.14      42.36
/robotics/inverse_kinematics      35.52      74.99
/physics/validate                  9.10      24.85
/observability/status             13.55      27.33
```

---

## API Optimization

### 1. Enable Response Compression

```python
# In main.py
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

### 2. Optimize JSON Serialization

```python
# Use orjson for faster JSON
import orjson
from fastapi.responses import ORJSONResponse

app = FastAPI(default_response_class=ORJSONResponse)
```

### 3. Connection Pooling

```python
# HTTP client pooling
import httpx

# Create a shared client with connection pooling
http_client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_keepalive_connections=20,
        max_connections=100,
        keepalive_expiry=30
    ),
    timeout=httpx.Timeout(30.0)
)
```

### 4. Async Everywhere

```python
# Bad: Blocking call in async context
def get_data():
    return requests.get(url)  # Blocks event loop

# Good: Async call
async def get_data():
    async with httpx.AsyncClient() as client:
        return await client.get(url)
```

### 5. Request Batching

```python
# Batch multiple operations
@router.post("/batch")
async def batch_operations(requests: List[Request]):
    results = await asyncio.gather(
        *[process_request(r) for r in requests]
    )
    return results
```

---

## Infrastructure Tuning

### Redis Optimization

```bash
# redis.conf optimizations
maxmemory 2gb
maxmemory-policy allkeys-lru
tcp-keepalive 300
timeout 0

# Connection pooling
maxclients 10000
```

```python
# Python Redis connection pool
import redis

pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    max_connections=50,
    socket_timeout=5,
    socket_connect_timeout=5
)
redis_client = redis.Redis(connection_pool=pool)
```

### Kafka Optimization

```properties
# producer.properties
batch.size=65536
linger.ms=5
compression.type=lz4
acks=1
buffer.memory=67108864

# consumer.properties
fetch.min.bytes=1024
fetch.max.wait.ms=500
max.poll.records=500
```

```python
# Python Kafka producer
from aiokafka import AIOKafkaProducer

producer = AIOKafkaProducer(
    bootstrap_servers='localhost:9092',
    compression_type='lz4',
    linger_ms=5,
    batch_size=65536
)
```

### Docker Resource Limits

```yaml
# docker-compose.yml
services:
  nis-protocol:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 4G
        reservations:
          cpus: '2'
          memory: 2G
```

### Uvicorn Workers

```bash
# Production: Multiple workers
uvicorn main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --loop uvloop \
  --http httptools
```

---

## LLM Optimization

### 1. Smart Caching

```python
# Cache LLM responses
from functools import lru_cache
import hashlib

def cache_key(prompt: str, model: str) -> str:
    return hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()

# Redis-based caching
async def get_cached_response(prompt: str, model: str):
    key = f"llm:cache:{cache_key(prompt, model)}"
    cached = await redis.get(key)
    if cached:
        return json.loads(cached)
    
    response = await call_llm(prompt, model)
    await redis.setex(key, 3600, json.dumps(response))  # 1 hour TTL
    return response
```

### 2. Request Deduplication

```python
# Deduplicate concurrent identical requests
import asyncio

_pending_requests = {}

async def deduplicated_llm_call(prompt: str, model: str):
    key = cache_key(prompt, model)
    
    if key in _pending_requests:
        return await _pending_requests[key]
    
    future = asyncio.create_task(call_llm(prompt, model))
    _pending_requests[key] = future
    
    try:
        return await future
    finally:
        del _pending_requests[key]
```

### 3. Token Optimization

```python
# Minimize tokens
def optimize_prompt(prompt: str) -> str:
    # Remove extra whitespace
    prompt = ' '.join(prompt.split())
    
    # Use abbreviations for common patterns
    replacements = {
        "Please provide": "Give",
        "I would like you to": "",
        "Can you please": "",
    }
    for old, new in replacements.items():
        prompt = prompt.replace(old, new)
    
    return prompt.strip()
```

### 4. Streaming Responses

```python
# Stream long responses
@router.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    async def generate():
        async for chunk in llm.stream(request.message):
            yield f"data: {json.dumps({'content': chunk})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### 5. Provider Routing

```python
# Route to fastest provider based on task
def select_provider(task_type: str) -> str:
    if task_type == "simple_chat":
        return "gpt-3.5-turbo"  # Fast, cheap
    elif task_type == "code_generation":
        return "gpt-4"  # Better quality
    elif task_type == "physics":
        return "claude-3-opus"  # Best reasoning
    else:
        return "gpt-4o"  # Default
```

---

## Robotics Performance

### 1. Kinematics Caching

```python
# Cache FK/IK solutions
from functools import lru_cache
import numpy as np

@lru_cache(maxsize=1000)
def cached_forward_kinematics(joint_angles_tuple: tuple) -> dict:
    joint_angles = np.array(joint_angles_tuple)
    return compute_forward_kinematics(joint_angles)

# Usage
result = cached_forward_kinematics(tuple(joint_angles))
```

### 2. Trajectory Precomputation

```python
# Precompute common trajectories
COMMON_TRAJECTORIES = {
    "home_to_pick": precompute_trajectory([HOME, PICK_POSITION]),
    "pick_to_place": precompute_trajectory([PICK_POSITION, PLACE_POSITION]),
}

async def execute_trajectory(name: str):
    if name in COMMON_TRAJECTORIES:
        return COMMON_TRAJECTORIES[name]
    return await compute_trajectory_online(name)
```

### 3. Physics Validation Batching

```python
# Batch physics validations
async def validate_trajectory_batch(trajectories: List[Trajectory]):
    # Validate all points in parallel
    validations = await asyncio.gather(
        *[validate_point(p) for t in trajectories for p in t.points]
    )
    return all(validations)
```

### 4. GPU Acceleration

```python
# Use GPU for matrix operations
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gpu_forward_kinematics(joint_angles: torch.Tensor) -> torch.Tensor:
    joint_angles = joint_angles.to(device)
    # Matrix operations on GPU
    return compute_fk_gpu(joint_angles)
```

---

## Memory Management

### 1. Object Pooling

```python
# Pool frequently created objects
from queue import Queue

class ObjectPool:
    def __init__(self, factory, size=100):
        self.pool = Queue(maxsize=size)
        self.factory = factory
        for _ in range(size):
            self.pool.put(factory())
    
    def acquire(self):
        return self.pool.get()
    
    def release(self, obj):
        self.pool.put(obj)

# Usage
trajectory_pool = ObjectPool(lambda: Trajectory(), size=50)
```

### 2. Memory-Efficient Data Structures

```python
# Use __slots__ for frequently instantiated classes
class RobotState:
    __slots__ = ['position', 'velocity', 'timestamp']
    
    def __init__(self, position, velocity, timestamp):
        self.position = position
        self.velocity = velocity
        self.timestamp = timestamp
```

### 3. Garbage Collection Tuning

```python
import gc

# Disable GC during critical operations
gc.disable()
try:
    result = critical_operation()
finally:
    gc.enable()
    gc.collect()
```

### 4. Memory Profiling

```python
# Profile memory usage
import tracemalloc

tracemalloc.start()

# Your code here
result = process_request()

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.2f} MB")
print(f"Peak: {peak / 1024 / 1024:.2f} MB")

tracemalloc.stop()
```

---

## Monitoring & Profiling

### 1. Request Tracing

```python
# Add timing to requests
import time
from fastapi import Request

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    
    response.headers["X-Response-Time"] = f"{duration*1000:.2f}ms"
    
    # Log slow requests
    if duration > 0.5:
        logger.warning(f"Slow request: {request.url.path} took {duration:.2f}s")
    
    return response
```

### 2. Profiling Endpoints

```python
import cProfile
import pstats
import io

@router.get("/debug/profile/{endpoint}")
async def profile_endpoint(endpoint: str):
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Call the endpoint
    result = await call_endpoint(endpoint)
    
    profiler.disable()
    
    # Get stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    return {
        "result": result,
        "profile": stream.getvalue()
    }
```

### 3. Prometheus Metrics

```python
from prometheus_client import Histogram, Counter

REQUEST_LATENCY = Histogram(
    'request_latency_seconds',
    'Request latency',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
)

REQUEST_COUNT = Counter(
    'request_count_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)
```

### 4. Load Testing

```bash
# Run load test
python tests/load/run_load_test.py --users 10 --duration 60

# With Locust
locust -f tests/load/locustfile.py --host=http://localhost:8000 --headless -u 100 -r 10 -t 60s
```

---

## Quick Wins Checklist

- [ ] Enable GZip compression
- [ ] Use orjson for JSON serialization
- [ ] Configure connection pooling
- [ ] Enable Redis caching for LLM responses
- [ ] Set appropriate Uvicorn workers (CPU cores * 2)
- [ ] Configure Docker resource limits
- [ ] Enable Kafka compression (lz4)
- [ ] Cache FK/IK solutions
- [ ] Use streaming for long responses
- [ ] Add request timing middleware

---

## Performance Targets by Endpoint Type

| Type | Target P95 | Optimization Focus |
|------|------------|-------------------|
| Health checks | <10ms | Minimal processing |
| Simple queries | <50ms | Caching |
| Kinematics | <100ms | GPU acceleration |
| Physics validation | <200ms | Batching |
| LLM requests | <5s | Streaming, caching |
| Trajectory planning | <500ms | Precomputation |

---

**Document Owner:** Platform Team  
**Review Cycle:** Quarterly  
**Last Review:** December 2025

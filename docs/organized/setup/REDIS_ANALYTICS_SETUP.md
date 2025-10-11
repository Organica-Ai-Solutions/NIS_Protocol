# 🎯 Redis + Analytics Setup Guide

## 🚀 Quick Setup

### **1. Install Redis**
```bash
# macOS with Homebrew
brew install redis

# Ubuntu/Debian
sudo apt-get install redis-server

# Start Redis
redis-server
```

### **2. Install Python Redis Client**
```bash
cd /Users/diegofuego/Desktop/NIS_Protocol
pip install redis
```

### **3. Start Your Optimized NIS Protocol**
```bash
# Start with Redis analytics enabled
python main.py
```

## 📊 **Analytics Dashboard Endpoints**

### **🎯 NEW: Unified Analytics Endpoint** (AWS CloudWatch Style)
```bash
# Summary view (default)
curl "http://localhost:8000/analytics"

# Detailed comprehensive view
curl "http://localhost:8000/analytics?view=detailed&hours_back=24"

# Token usage analysis
curl "http://localhost:8000/analytics?view=tokens&hours_back=48"

# Cost breakdown and savings
curl "http://localhost:8000/analytics?view=costs&include_charts=true"

# Performance metrics
curl "http://localhost:8000/analytics?view=performance"

# Real-time monitoring
curl "http://localhost:8000/analytics?view=realtime"

# Provider comparison
curl "http://localhost:8000/analytics?view=providers"

# Historical trends and forecasting
curl "http://localhost:8000/analytics?view=trends&hours_back=168"

# Filter by specific provider
curl "http://localhost:8000/analytics?view=costs&provider=openai"
```

### **Available Views:**
- `summary` - High-level overview (default)
- `detailed` - Comprehensive metrics
- `tokens` - Token usage analysis  
- `costs` - Financial breakdown
- `performance` - Speed & efficiency
- `realtime` - Live monitoring
- `providers` - Provider comparison
- `trends` - Historical patterns

### **Legacy Individual Endpoints** (Still Available)

### **AWS-Style Main Dashboard**
```bash
curl "http://localhost:8000/analytics/dashboard"
```
**Returns**: Complete analytics overview with token usage, costs, performance

### **Token Usage Analytics**
```bash
curl "http://localhost:8000/analytics/tokens?hours_back=24"
```
**Returns**: Input/output token breakdown, efficiency metrics

### **Cost Analytics**
```bash
curl "http://localhost:8000/analytics/costs?hours_back=24"
```
**Returns**: Cost breakdown by provider, savings analysis, budget tracking

### **Performance Metrics**
```bash
curl "http://localhost:8000/analytics/performance"
```
**Returns**: Latency, cache hit rates, error rates, throughput

### **Real-time Monitoring**
```bash
curl "http://localhost:8000/analytics/realtime"
```
**Returns**: Live metrics, current hour stats, system status

## 🎮 **Test the Analytics**

### **Make Some LLM Requests**
```bash
# Single provider request
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Test analytics", "provider": "google"}'

# Consensus request
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Test consensus analytics", "consensus_mode": "dual"}'

# Check analytics
curl "http://localhost:8000/analytics/dashboard"
```

## 📈 **What You'll See**

### **Token Analytics** (Like AWS CloudWatch)
- **Input tokens**: Tokens in your prompts
- **Output tokens**: Tokens in AI responses  
- **Token efficiency**: Input/output ratios
- **Provider breakdown**: Which models use most tokens

### **Cost Analytics** (Like AWS Cost Explorer)
- **Real-time costs**: Current spending by provider
- **Savings tracking**: Money saved through caching
- **Monthly projections**: Estimated monthly bills
- **Cost per request**: Efficiency metrics

### **Performance Analytics** (Like AWS Performance Insights)
- **Response times**: Latency by provider
- **Cache hit rates**: Caching efficiency
- **Error rates**: System reliability
- **Throughput**: Requests per hour

### **Real-time Monitoring** (Like AWS CloudWatch Live)
- **Current hour metrics**: Live usage stats
- **Recent requests**: Last 10 requests with details
- **System status**: Redis, cache, rate limiter status
- **Live alerts**: System health warnings

## 🔧 **Redis Configuration**

### **Default Settings** (Auto-configured)
- **Cache DB**: Redis DB 0 for smart caching
- **Analytics DB**: Redis DB 1 for analytics storage
- **Host**: localhost:6379
- **TTL**: Provider-specific (1-3 hours)

### **Custom Redis Setup**
```python
# In your environment or code
from src.analytics.llm_analytics import init_llm_analytics
from src.llm.smart_cache import init_smart_cache

# Initialize with custom Redis settings
analytics = init_llm_analytics(
    redis_host="your-redis-host",
    redis_port=6379,
    redis_db=1
)

cache = init_smart_cache(
    redis_host="your-redis-host", 
    redis_port=6379,
    redis_db=0
)
```

## 💡 **Key Benefits vs SQLite**

| **Feature** | **SQLite** | **Redis** |
|-------------|------------|-----------|
| **Speed** | Slow disk I/O | Ultra-fast memory |
| **Concurrency** | Locks on writes | High concurrency |
| **Real-time** | No | Yes - instant updates |
| **Scalability** | Single file | Distributed |
| **Analytics** | Complex queries | Built-in operations |
| **TTL** | Manual cleanup | Automatic expiration |

## 🎯 **Why This is Better**

### **1. Real-time Analytics**
- **SQLite**: Batch processing, delays
- **Redis**: Instant metrics, live dashboards

### **2. Performance** 
- **SQLite**: Disk-bound, slower queries
- **Redis**: Memory-fast, sub-millisecond typical response

### **3. Scalability**
- **SQLite**: Single file bottleneck  
- **Redis**: Distributed, clustered

### **4. Built-in Features**
- **SQLite**: Custom code for TTL, expiration
- **Redis**: Native TTL, atomic operations

## 🚨 **Migration Impact**

### **Before** (SQLite)
```
🐌 Cache lookup: 5-50ms
📊 Analytics query: 100-500ms  
💾 Storage: Single file bottleneck
🔧 Maintenance: Manual cleanup scripts
```

### **After** (Redis)
```
⚡ Cache lookup: 0.5-2ms
📊 Analytics query: 1-10ms
💾 Storage: Memory-optimized
🔧 Maintenance: Automatic TTL
```

## 🎮 **Try It Now!**

1. **Start Redis**: `redis-server`
2. **Start NIS Protocol**: `python main.py`
3. **Make requests**: Use the chat endpoint
4. **View analytics**: Visit `/analytics/dashboard`
5. **Watch in real-time**: Visit `/analytics/realtime`

Your AWS-style analytics dashboard is ready! 🎯

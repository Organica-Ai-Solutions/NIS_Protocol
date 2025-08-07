# 🛠️ NIS Protocol v3.1 - Troubleshooting Guide

## 📋 **Quick Diagnosis**

Having issues? Use this **rapid diagnosis checklist** to identify and resolve common problems:

### **⚡ 30-Second Health Check**
```bash
# 1. System status
curl http://localhost/health || echo "❌ API not responding"

# 2. Docker containers
docker ps | grep nis || echo "❌ Containers not running"

# 3. Database connectivity  
curl http://localhost/metrics | grep "database" || echo "❌ Database issues"

# 4. LLM providers
python -c "import os; print('✅ API keys configured' if os.getenv('OPENAI_API_KEY') else '❌ No API keys')"
```

### **🚨 Common Issues Quick Fixes**
| **Symptom** | **Quick Fix** | **Command** |
|:---|:---|:---|
| 🔴 API not responding | Restart system | `./start.sh` |
| 🟡 Slow responses | Clear cache | `docker exec nis-redis redis-cli FLUSHALL` |
| 🔵 Agent errors | Check logs | `docker logs nis-backend` |
| ⚪ Import errors | Rebuild image | `docker-compose build --no-cache` |

## 🚨 **Critical System Issues**

### **🔴 System Won't Start**

**Problem**: `./start.sh` fails or containers won't start

**Diagnosis**:
```bash
# Check Docker is running
docker --version || echo "Install Docker first"

# Check ports are available
netstat -tlnp | grep :80 || echo "Port 80 available"
netstat -tlnp | grep :8000 || echo "Port 8000 available"

# Check disk space
df -h | grep -E "/$|/var"
```

**Solutions**:
```bash
# Solution 1: Force clean restart
docker system prune -a -f
docker-compose -p nis-protocol-v3 down -v
./start.sh

# Solution 2: Port conflicts
# Edit docker-compose.yml to use different ports
sed -i 's/80:80/8080:80/' docker-compose.yml
sed -i 's/8000:8000/8001:8000/' docker-compose.yml

# Solution 3: Permission issues (Linux/Mac)
sudo chown -R $USER:$USER .
chmod +x start.sh

# Solution 4: Memory issues
# Increase Docker memory limit to 8GB+
# Or reduce agent instances:
export NIS_MAX_WORKERS=2
./start.sh
```

### **🟡 API Endpoints Returning Errors**

**Problem**: Getting 500, 502, or timeout errors

**Diagnosis**:
```bash
# Check backend health directly
curl http://localhost:8000/health

# Check Nginx status
docker logs nis-nginx

# Check backend logs for errors
docker logs nis-backend | tail -50
```

**Solutions**:
```bash
# Solution 1: Backend restart
docker restart nis-backend
sleep 10
curl http://localhost/health

# Solution 2: Nginx configuration
docker exec nis-nginx nginx -t
docker restart nis-nginx

# Solution 3: Database connectivity
docker exec nis-redis redis-cli ping
docker exec nis-postgres psql -U nisuser -d nisdb -c "SELECT 1;"

# Solution 4: Clear corrupted cache
docker exec nis-redis redis-cli FLUSHALL
docker restart nis-backend
```

### **🔵 Agent Processing Failures**

**Problem**: Agents returning errors or not responding

**Diagnosis**:
```bash
# Test specific agents
python test_fixed_parameters.py

# Check agent status
curl http://localhost/agents/status

# Monitor agent performance
curl http://localhost/metrics | jq '.agents'
```

**Solutions**:
```bash
# Solution 1: Agent restart
curl -X POST http://localhost/agents/restart \
  -d '{"agent_type": "reasoning"}'

# Solution 2: Memory cleanup
curl -X POST http://localhost/agents/cleanup \
  -d '{"cleanup_type": "memory"}'

# Solution 3: Reset agent state
curl -X POST http://localhost/agents/reset \
  -d '{"agent_id": "unified_reasoning_agent"}'

# Solution 4: Scale agent instances
curl -X POST http://localhost/agents/scale \
  -d '{"agent_type": "physics", "instances": 3}'
```

## 🔧 **Installation & Setup Issues**

### **🐳 Docker Issues**

**Problem**: Docker-related errors

**Common Error Messages & Solutions**:
```bash
# Error: "Cannot connect to Docker daemon"
# Solution: Start Docker service
sudo systemctl start docker  # Linux
# Or start Docker Desktop on Windows/Mac

# Error: "Port already in use"
# Solution: Kill existing processes
sudo lsof -ti:80 | xargs kill -9
sudo lsof -ti:8000 | xargs kill -9

# Error: "No space left on device"
# Solution: Clean Docker resources
docker system prune -a -f --volumes

# Error: "Permission denied"
# Solution: Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker
```

### **📦 Python Dependencies**

**Problem**: Import errors or missing packages

**Solutions**:
```bash
# Solution 1: Fresh virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Solution 2: Update packages
pip install --upgrade pip
pip install -r requirements.txt --upgrade

# Solution 3: Clear pip cache
pip cache purge
pip install -r requirements.txt --no-cache-dir

# Solution 4: Platform-specific installations
# For M1 Macs:
pip install --platform macosx_11_0_arm64 --only-binary=:all: package_name

# For Windows:
pip install package_name --user
```

### **🔑 API Key Configuration**

**Problem**: LLM provider authentication failures

**Diagnosis**:
```bash
# Check environment variables
echo "OpenAI: $OPENAI_API_KEY"
echo "Anthropic: $ANTHROPIC_API_KEY"
echo "DeepSeek: $DEEPSEEK_API_KEY"

# Test API connectivity
python -c "
import openai
import os
openai.api_key = os.getenv('OPENAI_API_KEY')
try:
    openai.Model.list()
    print('✅ OpenAI working')
except Exception as e:
    print(f'❌ OpenAI error: {e}')
"
```

**Solutions**:
```bash
# Solution 1: Proper environment setup
cp dev/environment-template.txt .env
nano .env  # Edit with your API keys

# Reload environment
source .env  # Linux/Mac
# Or restart terminal on Windows

# Solution 2: Container environment
# Ensure Docker containers can access environment
docker-compose -p nis-protocol-v3 down
./start.sh

# Solution 3: API key format verification
# OpenAI: sk-...
# Anthropic: sk-ant-...
# DeepSeek: sk-...

# Solution 4: Alternative providers
# If one fails, configure others:
export FALLBACK_LLM_PROVIDER=anthropic
```

## ⚡ **Performance Issues**

### **🐌 Slow Response Times**

**Problem**: Requests taking > 10 seconds

**Diagnosis**:
```bash
# Check system resources
docker stats --no-stream

# Check request times
time curl http://localhost/simulation/run \
  -d '{"concept": "test"}'

# Monitor database performance
docker exec nis-postgres pg_stat_activity

# Check cache hit rates
docker exec nis-redis redis-cli info stats
```

**Solutions**:
```bash
# Solution 1: Increase resources
# Edit docker-compose.yml:
# mem_limit: 4g  # Increase from 2g
# cpus: 2.0      # Increase from 1.0

# Solution 2: Optimize cache
docker exec nis-redis redis-cli config set maxmemory 1gb
docker exec nis-redis redis-cli config set maxmemory-policy allkeys-lru

# Solution 3: Database optimization
docker exec nis-postgres psql -U nisuser -d nisdb -c "
  UPDATE pg_settings SET setting = '256MB' WHERE name = 'shared_buffers';
  SELECT pg_reload_conf();
"

# Solution 4: Scale horizontally
curl -X POST http://localhost/agents/scale \
  -d '{"scale_all": true, "target_instances": 3}'
```

### **💾 Memory Issues**

**Problem**: Out of memory errors or high memory usage

**Diagnosis**:
```bash
# Check memory usage
free -h
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Check for memory leaks
docker exec nis-backend python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

**Solutions**:
```bash
# Solution 1: Increase Docker memory
# Docker Desktop: Settings > Resources > Memory > 8GB+

# Solution 2: Memory cleanup
curl -X POST http://localhost/agents/cleanup \
  -d '{"cleanup_type": "memory", "force": true}'

# Solution 3: Garbage collection
docker exec nis-backend python -c "
import gc
gc.collect()
print('Garbage collection completed')
"

# Solution 4: Restart heavy agents
curl -X POST http://localhost/agents/restart \
  -d '{"agent_type": "reasoning", "force_restart": true}'
```

## 🤖 **Agent-Specific Issues**

### **🧮 Mathematical Pipeline Errors**

**Problem**: Laplace→KAN→PINN processing failures

**Common Errors**:
```bash
# Error: "KAN network convergence failure"
# Solution: Adjust complexity
curl -X POST http://localhost/agents/reasoning/configure \
  -d '{"kan_complexity": "simple", "iterations": 1000}'

# Error: "Physics validation failed"  
# Solution: Relax constraints
curl -X POST http://localhost/agents/physics/configure \
  -d '{"tolerance": 1e-3, "strict_mode": false}'

# Error: "Laplace transform singular"
# Solution: Preprocess signal
curl -X POST http://localhost/agents/signal/configure \
  -d '{"preprocessing": "filter", "noise_reduction": true}'
```

### **🧠 LLM Provider Issues**

**Problem**: LLM timeouts or rate limiting

**Diagnosis**:
```bash
# Check provider status
curl http://localhost/llm/status

# Monitor request rates
curl http://localhost/metrics | jq '.llm_providers'
```

**Solutions**:
```bash
# Solution 1: Configure rate limits
curl -X POST http://localhost/llm/configure \
  -d '{
    "rate_limits": {
      "openai": 50,
      "anthropic": 30,
      "deepseek": 100
    }
  }'

# Solution 2: Enable fallback providers
curl -X POST http://localhost/llm/configure \
  -d '{
    "fallback_chain": ["openai", "anthropic", "deepseek"],
    "auto_fallback": true
  }'

# Solution 3: Adjust timeouts
curl -X POST http://localhost/llm/configure \
  -d '{
    "timeouts": {
      "request_timeout": 30,
      "connection_timeout": 10
    }
  }'
```

### **😊 Specialized Agent Problems**

**Problem**: Ethics, emotion, or curiosity agents malfunctioning

**Solutions by Agent Type**:
```bash
# Ethics Agent Issues
curl -X POST http://localhost/agents/alignment/reset
curl -X POST http://localhost/agents/alignment/configure \
  -d '{"frameworks": ["utilitarian"], "strictness": "moderate"}'

# Emotion Agent Issues  
curl -X POST http://localhost/agents/emotion/calibrate \
  -d '{"baseline_emotions": "neutral", "sensitivity": 0.7}'

# Curiosity Agent Issues
curl -X POST http://localhost/agents/curiosity/reset \
  -d '{"reset_learning": true, "clear_patterns": false}'

# Memory Agent Issues
curl -X POST http://localhost/agents/memory/cleanup \
  -d '{"cleanup_old": true, "max_age_days": 30}'

# Vision Agent Issues
curl -X POST http://localhost/agents/vision/reset \
  -d '{"clear_cache": true, "reload_models": true}'
```

## 🌐 **Network & Connectivity**

### **🔌 Port and Network Issues**

**Problem**: Cannot access services or internal communication failures

**Diagnosis**:
```bash
# Check port accessibility
nmap -p 80,8000,6379,5432,9092 localhost

# Check internal Docker network
docker network ls
docker network inspect nis-protocol-v3_default

# Test internal connectivity
docker exec nis-backend curl http://nis-redis:6379/ping
docker exec nis-backend nc -zv nis-postgres 5432
```

**Solutions**:
```bash
# Solution 1: Recreate Docker network
docker-compose -p nis-protocol-v3 down
docker network prune -f
docker-compose -p nis-protocol-v3 up -d

# Solution 2: Firewall configuration
# Linux:
sudo ufw allow 80
sudo ufw allow 8000

# Windows: Add firewall rules for Docker

# Solution 3: DNS resolution
# Add to /etc/hosts (Linux/Mac) or C:\Windows\System32\drivers\etc\hosts (Windows):
127.0.0.1 nis-backend
127.0.0.1 nis-redis
127.0.0.1 nis-postgres
```

### **☁️ Cloud Deployment Issues**

**Problem**: AWS or cloud deployment failures

**AWS-Specific Solutions**:
```bash
# Check AWS credentials
aws sts get-caller-identity

# EKS cluster issues
kubectl get nodes
kubectl get pods -n nis-protocol

# RDS connectivity
aws rds describe-db-instances --query 'DBInstances[0].Endpoint'

# ElastiCache status
aws elasticache describe-cache-clusters

# MSK (Kafka) status
aws kafka list-clusters
```

## 📊 **Data & Database Issues**

### **🗄️ PostgreSQL Problems**

**Problem**: Database connectivity or data corruption

**Diagnosis**:
```bash
# Check PostgreSQL status
docker exec nis-postgres pg_isready

# Check database size
docker exec nis-postgres psql -U nisuser -d nisdb -c "
  SELECT pg_size_pretty(pg_database_size('nisdb'));
"

# Check for locks
docker exec nis-postgres psql -U nisuser -d nisdb -c "
  SELECT * FROM pg_stat_activity WHERE state = 'active';
"
```

**Solutions**:
```bash
# Solution 1: Restart PostgreSQL
docker restart nis-postgres
sleep 5
docker exec nis-postgres pg_isready

# Solution 2: Clear locks
docker exec nis-postgres psql -U nisuser -d nisdb -c "
  SELECT pg_terminate_backend(pid) FROM pg_stat_activity 
  WHERE state = 'idle in transaction' AND pid <> pg_backend_pid();
"

# Solution 3: Database vacuum
docker exec nis-postgres psql -U nisuser -d nisdb -c "VACUUM ANALYZE;"

# Solution 4: Reset database (CAUTION: destroys data)
docker-compose -p nis-protocol-v3 down -v
docker volume rm nis-protocol-v3_postgres_data
./start.sh
```

### **💾 Redis Cache Issues**

**Problem**: Cache corruption or connectivity issues

**Solutions**:
```bash
# Solution 1: Clear cache
docker exec nis-redis redis-cli FLUSHALL

# Solution 2: Check Redis health
docker exec nis-redis redis-cli info server

# Solution 3: Restart Redis
docker restart nis-redis
sleep 3
docker exec nis-redis redis-cli ping

# Solution 4: Memory optimization
docker exec nis-redis redis-cli config set maxmemory-policy allkeys-lru
docker exec nis-redis redis-cli config set maxmemory 512mb
```

## 🔍 **Debugging Tools**

### **📝 Log Analysis**

**Comprehensive Logging**:
```bash
# View all container logs
docker-compose -p nis-protocol-v3 logs -f

# Backend-specific logs
docker logs nis-backend -f --tail 100

# Error-only logs
docker logs nis-backend 2>&1 | grep -i error

# Real-time monitoring
watch -n 1 'docker stats --no-stream'
```

**Log File Locations**:
```bash
# Application logs
tail -f logs/nis_application.log

# Error logs
tail -f logs/nis_errors.log

# Agent-specific logs
tail -f logs/agents/unified_reasoning_agent.log
tail -f logs/agents/unified_physics_agent.log

# Performance logs
tail -f logs/performance_metrics.log
```

### **🧪 Testing & Validation**

**System Health Checks**:
```bash
# Comprehensive system test
python dev/testing/comprehensive_system_test.py

# Endpoint validation
python test_fixed_parameters.py

# Performance benchmark
python benchmarks/nis_comprehensive_benchmark_suite.py

# Agent functionality test
python dev/testing/test_all_agents.py
```

**Custom Debugging**:
```python
# Debug individual agents
from src.agents.reasoning.unified_reasoning_agent import UnifiedReasoningAgent
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Test agent directly
agent = UnifiedReasoningAgent(debug_mode=True)
result = await agent.process({"test": "data"})
print(f"Debug result: {result}")
```

## 🆘 **Emergency Procedures**

### **🚨 Complete System Reset**

**When everything else fails**:
```bash
# CAUTION: This destroys all data and containers
echo "⚠️  WARNING: This will destroy all data!"
read -p "Continue? (yes/no): " confirm

if [ "$confirm" = "yes" ]; then
    # Stop everything
    docker-compose -p nis-protocol-v3 down -v
    
    # Remove all images
    docker rmi $(docker images -q nis-protocol-v3*)
    
    # Clean system
    docker system prune -a -f --volumes
    
    # Fresh start
    ./start.sh
    
    echo "✅ Complete system reset completed"
fi
```

### **📋 Emergency Backup**

**Before making drastic changes**:
```bash
# Backup critical configurations
mkdir -p emergency_backup
cp .env emergency_backup/
cp docker-compose.yml emergency_backup/
cp -r system/config emergency_backup/

# Backup database
docker exec nis-postgres pg_dump -U nisuser nisdb > emergency_backup/database.sql

# Backup Redis data
docker exec nis-redis redis-cli save
docker cp nis-redis:/data/dump.rdb emergency_backup/

echo "✅ Emergency backup completed in emergency_backup/"
```

### **🔄 Emergency Recovery**

**Restore from backup**:
```bash
# Restore environment
cp emergency_backup/.env .
cp emergency_backup/docker-compose.yml .

# Start system
./start.sh

# Wait for startup
sleep 30

# Restore database
docker exec -i nis-postgres psql -U nisuser nisdb < emergency_backup/database.sql

# Restore Redis
docker cp emergency_backup/dump.rdb nis-redis:/data/
docker restart nis-redis

echo "✅ Emergency recovery completed"
```

## 📞 **Getting Help**

### **🔍 Self-Help Resources**

1. **Documentation**: Check `/system/docs/` for detailed guides
2. **Examples**: Review `/dev/examples/` for working implementations  
3. **Tests**: Run test suites to identify specific issues
4. **Logs**: Always check logs first for error details

### **🤝 Community Support**

1. **GitHub Issues**: Report bugs with detailed reproduction steps
2. **GitHub Discussions**: Ask questions and share solutions
3. **Documentation PRs**: Improve this troubleshooting guide
4. **Community Examples**: Share your solutions

### **📋 Bug Report Template**

When reporting issues, include:
```markdown
## Bug Description
Brief description of the issue

## Environment
- OS: [Windows/Mac/Linux]
- Docker version: `docker --version`
- Python version: `python --version`
- NIS Protocol version: v3.1

## Reproduction Steps
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Error Messages
```
[paste error messages here]
```

## System Information
```bash
# Output of:
docker ps
curl http://localhost/health
docker logs nis-backend --tail 50
```

## Additional Context
Any other relevant information
```

## 🎯 **Prevention Tips**

### **✅ Regular Maintenance**
```bash
# Weekly health check
python dev/testing/weekly_health_check.py

# Monthly cleanup
docker system prune -f
docker exec nis-redis redis-cli flushdb
docker exec nis-postgres psql -U nisuser -d nisdb -c "VACUUM;"

# Update dependencies
pip install -r requirements.txt --upgrade
docker-compose build --no-cache
```

### **📊 Monitoring Setup**
```bash
# Enable comprehensive monitoring
curl -X POST http://localhost/monitoring/enable \
  -d '{"alerts": true, "metrics": true, "logging": "verbose"}'

# Set up alerting
curl -X POST http://localhost/monitoring/alerts \
  -d '{
    "error_threshold": 5,
    "response_time_threshold": 10,
    "memory_threshold": 80
  }'
```

---

**Remember**: Most issues can be resolved by restarting the system with `./start.sh`. When in doubt, check the logs first! 🚀

**Need more help?** Check our [comprehensive documentation](../README.md) or reach out to the community!

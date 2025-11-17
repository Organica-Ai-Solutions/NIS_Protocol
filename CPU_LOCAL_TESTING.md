# 🖥️ CPU-Only Local Testing Guide
## For Mac & Non-GPU Development

**Purpose:** Test NIS Protocol locally without GPU requirements  
**Platform:** Mac (Intel/Apple Silicon), Linux  
**Build Time:** ~5-10 minutes (vs 15-20 for GPU)  
**Image Size:** ~2-3 GB (vs 5-8 GB for GPU)

---

## 🚀 Quick Start

### **Option 1: Docker Compose (Recommended)**

```bash
# 1. Navigate to project directory
cd /Users/diegofuego/Desktop/NIS_Protocol

# 2. Create .env file with API keys
cat > .env << EOF
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
GOOGLE_API_KEY=your-google-key-here
EOF

# 3. Build and start all services
docker-compose -f docker-compose.cpu.yml up --build

# 4. Wait for services to be ready (~30-40 seconds)
# Watch for: "✅ ScientificCoordinator initialized"

# 5. Test health endpoint
curl http://localhost:8000/health
```

### **Option 2: Docker Only (Backend Only)**

```bash
# 1. Build CPU image
docker build -f Dockerfile.cpu -t nis-protocol:cpu .

# 2. Run with environment variables
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=sk-your-key-here \
  -e ANTHROPIC_API_KEY=sk-ant-your-key-here \
  nis-protocol:cpu

# 3. Test in another terminal
curl http://localhost:8000/health
```

---

## 🧪 Testing the Scientific Pipeline

### **Test 1: Health Check**
```bash
curl http://localhost:8000/health

# Expected Response:
{
  "status": "healthy",
  "timestamp": 1700000000,
  "provider": ["openai", "anthropic", "google"],
  "agents_registered": 14,
  "laplace": true,
  "kan": true,
  "pinn": true
}
```

### **Test 2: Simple Chat**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is 2+2?",
    "conversation_id": "test-001"
  }'
```

### **Test 3: Signal Processing (Scientific Pipeline)**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze the frequency components of this signal: [1.0, 2.5, -0.5, 3.2, 1.8, -1.2, 0.8, 2.1]",
    "conversation_id": "test-signal"
  }'

# Expected: Response with Laplace transform results
```

### **Test 4: Access Web UI**
```bash
# Open in browser
open http://localhost:8000/chat
# or
open http://localhost/chat
```

---

## 📊 Performance Expectations (Mac)

| Task | CPU (Mac) | GPU (AWS g4dn) | Notes |
|------|-----------|----------------|-------|
| Container Start | 20-30s | 30-40s | Faster on Mac (no GPU init) |
| Simple Chat | 200-500ms | 150-300ms | Similar (uses LLM APIs) |
| Signal Processing | 50-150ms | 30-80ms | Slightly slower, still fast |
| Heavy ML Task | 2-5s | 0.5-2s | GPU better for large models |

---

## 🔧 Troubleshooting

### **Issue 1: Port Already in Use**
```bash
# Find and kill process using port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
docker run -p 8080:8000 nis-protocol:cpu
```

### **Issue 2: Docker Build Fails**
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build -f Dockerfile.cpu --no-cache -t nis-protocol:cpu .
```

### **Issue 3: Services Won't Start**
```bash
# Check Docker Desktop is running
docker ps

# Check logs
docker-compose -f docker-compose.cpu.yml logs backend

# Restart services
docker-compose -f docker-compose.cpu.yml restart
```

### **Issue 4: Redis/Kafka Connection Error**
```bash
# Ensure all services started
docker-compose -f docker-compose.cpu.yml ps

# Restart dependencies
docker-compose -f docker-compose.cpu.yml restart redis kafka
```

### **Issue 5: API Keys Not Working**
```bash
# Verify .env file exists
cat .env

# Check environment variables in container
docker exec nis-backend-cpu env | grep API_KEY

# Restart with explicit environment
docker-compose -f docker-compose.cpu.yml down
docker-compose -f docker-compose.cpu.yml up
```

---

## 🎯 Development Workflow

### **Hot Reload Setup** (Code Changes Auto-Reload)
The CPU docker-compose already mounts your code:
```yaml
volumes:
  - ./:/home/nisuser/app  # Your code is live-mounted
```

**Workflow:**
1. Edit code in your IDE
2. Save file
3. Container automatically reloads (uvicorn --reload)
4. Test immediately

### **View Logs**
```bash
# All services
docker-compose -f docker-compose.cpu.yml logs -f

# Backend only
docker-compose -f docker-compose.cpu.yml logs -f backend

# Last 100 lines
docker-compose -f docker-compose.cpu.yml logs --tail=100 backend
```

### **Execute Commands in Container**
```bash
# Open shell in backend container
docker exec -it nis-backend-cpu bash

# Run Python script
docker exec nis-backend-cpu python -c "import torch; print(torch.__version__)"

# Check agent status
docker exec nis-backend-cpu python -c "
from src.meta.unified_coordinator import create_scientific_coordinator
coord = create_scientific_coordinator()
print(f'Laplace: {coord.laplace is not None}')
print(f'KAN: {coord.kan is not None}')
print(f'PINN: {coord.pinn is not None}')
"
```

---

## 📈 Monitoring

### **Container Stats**
```bash
# Real-time resource usage
docker stats nis-backend-cpu

# Expected on Mac:
# CPU: 20-50% (during requests)
# Memory: 1-2 GB (with all agents loaded)
```

### **Health Monitoring**
```bash
# Continuous health check
watch -n 2 'curl -s http://localhost:8000/health | jq'

# Check agent status
curl http://localhost:8000/health | jq '.agents_registered'
```

---

## 🔄 Cleanup

### **Stop Services**
```bash
# Stop all services (keep data)
docker-compose -f docker-compose.cpu.yml stop

# Stop and remove containers
docker-compose -f docker-compose.cpu.yml down

# Stop and remove everything (including volumes)
docker-compose -f docker-compose.cpu.yml down -v
```

### **Clean Docker**
```bash
# Remove CPU images
docker rmi nis-protocol:cpu

# Clean all unused images/containers
docker system prune -a
```

---

## 🆚 CPU vs GPU Comparison

| Feature | CPU (Mac) | GPU (AWS) |
|---------|-----------|-----------|
| **Setup** | ✅ Instant | ⚠️ Requires GPU drivers |
| **Cost** | ✅ Free (local) | 💰 $0.526/hour (g4dn.xlarge) |
| **Build Time** | ✅ 5-10 min | ⚠️ 15-20 min |
| **Image Size** | ✅ 2-3 GB | ⚠️ 5-8 GB |
| **Development** | ✅ Hot reload | ⚠️ Rebuild/redeploy |
| **LLM Inference** | ✅ APIs (same) | ✅ APIs (same) |
| **Heavy ML** | ⚠️ Slower | ✅ Fast |
| **Production** | ❌ Not scalable | ✅ Scalable |

**Recommendation:** Use CPU for development, GPU for production heavy workloads.

---

## ✅ Pre-Deployment Checklist

Before deploying to AWS, verify locally:

- [ ] Build completes without errors
- [ ] All services start (`docker-compose ps`)
- [ ] Health endpoint returns 200 OK
- [ ] All 14 agents register successfully
- [ ] Laplace/KAN/PINN agents initialize
- [ ] Chat endpoint responds correctly
- [ ] Signal processing works
- [ ] No memory leaks (run for 10+ minutes)
- [ ] Logs show no errors

---

## 🚀 Next Steps

Once local testing passes:

1. **Push to ECR** (AWS Container Registry)
   ```bash
   # See DOCKERFILE_VERIFICATION.md for ECR instructions
   ```

2. **Deploy to ECS** (AWS Elastic Container Service)
   - Use CPU task definition (t3.medium/t3.large)
   - Or GPU task definition (g4dn.xlarge) for heavy workloads

3. **Monitor Production**
   - CloudWatch logs
   - Health checks
   - Cost tracking

---

## 📞 Support

**Issues?**
- Check logs: `docker-compose -f docker-compose.cpu.yml logs -f`
- Review health: `curl http://localhost:8000/health`
- Verify API keys in `.env` file
- Ensure Docker Desktop is running

**Documentation:**
- `DOCKERFILE_VERIFICATION.md` - Deployment guide
- `SYSTEM_STUDY_GUIDE.md` - System architecture
- `README.md` - Project overview

---

**Generated:** November 17, 2025  
**For:** Local Mac Testing  
**Status:** 🟢 READY FOR TESTING

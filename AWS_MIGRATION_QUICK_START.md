# 🚀 NIS Protocol v3.2 → AWS Migration Quick Start

## 🎯 **Your Container Split Strategy - Simple Approach**

### **Step 1: Fix Current Issues** *(Before Migration)*

#### **✅ FIXED: DocumentAnalysisAgent**
```bash
# Fixed the missing _extract_tables method
# Status: ✅ Ready for startup testing
```

#### **🔧 Test Current System**
```bash
# Rebuild and test
./stop.sh
docker-compose build --no-cache  
./start.sh

# Verify endpoints
curl -I "http://localhost/health"
curl -I "http://localhost/console"
curl -X POST "http://localhost/image/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test quantum computer", "style": "scientific"}'
```

---

## 🏗️ **Container Split Strategy** *(Your Preferred Approach)*

### **📦 Container 1: Core Agents**
```yaml
Service Name: nis-core-agents
Contents:
├── src/agents/reasoning/
├── src/agents/learning/ 
├── src/agents/memory/
├── src/agents/physics/
├── src/agents/multimodal/  # NEW v3.2
└── src/core/agent.py

Docker Command:
docker build -t nis-core-agents ./core-agents/
```

### **🎪 Container 2: Meta Coordinators**
```yaml
Service Name: nis-meta-coordinators
Contents:
├── src/meta/unified_coordinator.py
├── src/agents/coordination/
├── src/services/consciousness_service.py
└── Agent orchestration

Environment:
- CORE_AGENTS_URL: "http://nis-core-agents:8001"
- REDIS_URL: "your-elasticache-endpoint"
- KAFKA_BROKERS: "your-msk-endpoint"
```

### **🌐 Container 3: Web API**
```yaml
Service Name: nis-web-api
Contents:
├── main.py (FastAPI)
├── static/ (v3.2 console)
├── All API endpoints
└── Image generation routes

Public Access:
- Load Balancer: AWS ALB
- CDN: CloudFront for static files
- Auto-scaling: Based on traffic
```

### **🤖 Container 4: LLM Providers**
```yaml
Service Name: nis-llm-providers
Contents:
├── src/llm/providers/
├── src/llm/llm_manager.py
└── Provider cost optimization

Benefits:
- Isolate external API costs
- Monitor usage by provider
- Implement intelligent routing
```

---

## 🚀 **AWS Services You'll Use** *(Simple & Managed)*

### **🗄️ Infrastructure Services**
```yaml
Instead of Docker Compose:
├── Redis → AWS ElastiCache (managed Redis)
├── Kafka → AWS MSK (managed Kafka)
├── PostgreSQL → AWS RDS (managed database)
└── File Storage → AWS S3 (for models/images)

Cost: ~$1,500-2,000/month for development
```

### **🖥️ Container Services**
```yaml
Instead of Docker Containers:
├── ECS Fargate → Run containers without servers
├── Application Load Balancer → Route traffic
├── Auto Scaling → Scale up/down automatically
└── CloudWatch → Monitoring and logs

Cost: ~$500-1,000/month for development
```

---

## 💡 **Your First AWS Commands**

### **1. Create Basic Infrastructure**
```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name nis-protocol

# Create VPC (or use default)
aws ec2 describe-vpcs --filters "Name=is-default,Values=true"

# Create security group
aws ec2 create-security-group \
  --group-name nis-protocol-sg \
  --description "NIS Protocol security group"
```

### **2. Set Up Managed Services**
```bash
# Create ElastiCache (Redis)
aws elasticache create-cache-cluster \
  --cache-cluster-id nis-redis \
  --engine redis \
  --cache-node-type cache.t3.micro \
  --num-cache-nodes 1

# Create RDS (PostgreSQL)  
aws rds create-db-instance \
  --db-instance-identifier nis-postgres \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --master-username nisuser \
  --master-user-password your-password \
  --allocated-storage 20
```

### **3. Deploy First Container**
```bash
# Build and push to ECR
aws ecr create-repository --repository-name nis-core-agents

# Get login token
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  your-account.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -t nis-core-agents .
docker tag nis-core-agents:latest \
  your-account.dkr.ecr.us-east-1.amazonaws.com/nis-core-agents:latest
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/nis-core-agents:latest
```

---

## 🎯 **Migration Timeline** *(Your Approach)*

### **Week 1: Infrastructure**
- [ ] Set up AWS account and VPC
- [ ] Deploy ElastiCache, RDS, MSK
- [ ] Test connectivity from local

### **Week 2: Core Agents**
- [ ] Create core agents container
- [ ] Deploy to ECS Fargate
- [ ] Connect to managed services

### **Week 3: Meta Coordinators**
- [ ] Deploy coordination container
- [ ] Test inter-service communication
- [ ] Validate agent orchestration

### **Week 4: Web API + LLM**
- [ ] Deploy web API with load balancer
- [ ] Deploy LLM provider container
- [ ] Test end-to-end functionality

---

## 💰 **Simple Cost Estimate**

### **Development Environment**
```yaml
Monthly Costs:
├── ECS Fargate (4 services): $400-600
├── ElastiCache (small Redis): $100-150
├── RDS (small PostgreSQL): $150-250
├── MSK (basic Kafka): $200-300
├── Load Balancer: $20-30
├── S3 + CloudFront: $50-100
└── Total: $920-1,430/month

Compare to: Current Docker Compose: $0 (but no scaling/reliability)
```

### **Production Environment**
```yaml
Monthly Costs:
├── ECS Fargate (auto-scaling): $2,000-4,000
├── ElastiCache (cluster): $500-800
├── RDS Aurora (multi-AZ): $800-1,200
├── MSK (production): $600-1,000
├── Load Balancer + WAF: $100-200
├── S3 + CloudFront: $200-400
└── Total: $4,200-7,600/month

Benefits: 99.9% uptime, auto-scaling, global reach
```

---

## 🔧 **Environment Variables Changes**

### **Current (Docker Compose)**
```python
REDIS_URL = "redis:6379"
KAFKA_BROKERS = "kafka:9092"
DATABASE_URL = "postgres://user:pass@postgres:5432/nisdb"
```

### **AWS (Managed Services)**
```python
REDIS_URL = "nis-redis.abc123.cache.amazonaws.com:6379"
KAFKA_BROKERS = "nis-kafka.abc123.kafka.us-east-1.amazonaws.com:9092"
DATABASE_URL = "nis-postgres.abc123.us-east-1.rds.amazonaws.com:5432/nisdb"
```

---

## ✅ **Next Steps for You**

### **This Week**
1. **Test Fixed System**: Verify DocumentAnalysisAgent fix works
2. **AWS Account**: Set up AWS account (if not done)
3. **Learn Basics**: Familiarize with ECS, ElastiCache, RDS concepts
4. **Plan Budget**: Approve $1,500-2,000/month for development environment

### **Next Week** 
1. **Create Infrastructure**: Set up VPC, managed services
2. **First Container**: Deploy core agents to ECS
3. **Test Connection**: Verify agents can connect to managed Redis/Kafka

### **Success Metrics**
- ✅ Current system fully functional (all endpoints working)
- ✅ AWS infrastructure deployed and accessible
- ✅ First container running on ECS Fargate
- ✅ Cost within $2,000/month development budget

---

**🏺 Simple AWS migration focused on your preferred container split approach with managed services for reliability and no server management complexity.**
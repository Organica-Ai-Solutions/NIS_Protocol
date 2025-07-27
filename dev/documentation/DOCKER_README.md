# 🐳 NIS Protocol v3 - Complete Docker Infrastructure

<div align="center">

![NIS Protocol v3 Logo](assets/images_organized/nis-protocol-logov1.png)

**Production-Ready Containerized AGI Foundation**

[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Kafka](https://img.shields.io/badge/Apache%20Kafka-231F20?style=for-the-badge&logo=apache-kafka&logoColor=white)](https://kafka.apache.org)
[![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)

</div>

## 🚀 One-Command Deployment

Get the complete NIS Protocol v3 infrastructure running in under 5 minutes:

```bash
# 1. Clone the repository
git clone https://github.com/Organica-Ai-Solutions/NIS_Protocol.git
cd NIS_Protocol

# 2. 🔑 Configure LLM API keys (REQUIRED)
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
EOF

# Edit with your actual API keys
nano .env

# 3. Start the complete system
./start.sh
```

> **⚠️ Important**: You need at least one LLM provider API key. Get them from:
> - OpenAI: https://platform.openai.com/api-keys
> - Anthropic: https://console.anthropic.com/

That's it! Your AGI foundation is now running with:
- ✅ Neural Intelligence Processing API
- ✅ Real-time Monitoring Dashboard  
- ✅ Kafka Message Streaming
- ✅ Redis Memory Management
- ✅ PostgreSQL Database
- ✅ Nginx Reverse Proxy

## 📊 Infrastructure Overview

### Core Services Architecture

```mermaid
graph TB
    subgraph "External Access"
        U[Users] --> N[Nginx :80]
        D[Developers] --> G[Grafana :3000]
    end
    
    subgraph "Application Layer"
        N --> A[NIS App :8000]
        A --> M[Dashboard :5000]
    end
    
    subgraph "Data Layer"
        A --> K[Kafka :9092]
        A --> R[Redis :6379] 
        A --> P[PostgreSQL :5432]
    end
    
    subgraph "Infrastructure"
        K --> Z[Zookeeper :2181]
        G --> PR[Prometheus :9090]
    end
    
    subgraph "Optional Monitoring"
        KU[Kafka UI :8080]
        RC[Redis Commander :8081]
    end
```

### Service Breakdown

| Service | Container | Ports | Purpose |
|---------|-----------|-------|---------|
| **NIS Main App** | `nis-main-app` | 8000, 5000 | AGI processing & monitoring |
| **Nginx** | `nis-nginx` | 80, 443 | Reverse proxy & load balancing |
| **PostgreSQL** | `nis-postgres` | 5432 | Persistent data storage |
| **Kafka** | `nis-kafka` | 9092, 29092 | Message streaming |
| **Zookeeper** | `nis-zookeeper` | 2181 | Kafka coordination |
| **Redis** | `nis-redis` | 6379 | Memory cache & sessions |
| **Grafana** | `nis-grafana` | 3000 | Monitoring dashboards |
| **Prometheus** | `nis-prometheus` | 9090 | Metrics collection |

## 🏃‍♂️ Quick Start Guide

### Prerequisites

- **Docker** 20.10+ and **Docker Compose** 2.0+
- **8GB+ RAM** (recommended for full stack)
- **10GB+ free disk space**
- **Git** for cloning the repository

### Installation Options

#### Option 1: Core System Only (Minimal)
```bash
./start.sh
```
Starts: API, Database, Kafka, Redis, Nginx

#### Option 2: Full System with Monitoring
```bash
./start.sh --with-monitoring
```
Adds: Grafana, Prometheus, Kafka UI, Redis Commander

#### Option 3: Development Mode
```bash
# Start with verbose logging and development tools
export LOG_LEVEL=DEBUG
./start.sh --with-monitoring
```

### Service URLs

After startup, access your services at:

| Service | URL | Description |
|---------|-----|-------------|
| **🎯 Main API** | http://localhost/ | Neural Intelligence API |
| **📊 Dashboard** | http://localhost/dashboard/ | Real-time monitoring |
| **📖 API Docs** | http://localhost/docs | Interactive API documentation |
| **🔍 Health Check** | http://localhost/health | System health status |
| **📈 Grafana** | http://localhost:3000 | Advanced monitoring (admin/nis_admin_2025) |
| **🔥 Kafka UI** | http://localhost:8080 | Message queue management |
| **💾 Redis Commander** | http://localhost:8081 | Cache management |
| **📊 Prometheus** | http://localhost:9090 | Metrics collection |

## 🛠️ Management Scripts

### Start System
```bash
./start.sh                    # Core system
./start.sh --with-monitoring  # Full monitoring stack
./start.sh --help            # Show all options
```

### Stop System
```bash
./stop.sh                     # Graceful shutdown
./stop.sh --remove-containers # Remove containers
./stop.sh --remove-volumes    # Remove all data (DESTRUCTIVE)
./stop.sh --force            # Force kill all containers
./stop.sh --save-logs         # Save logs before shutdown
```

### Reset System
```bash
./reset.sh                   # Interactive reset
./reset.sh --force           # Skip confirmation
./reset.sh --deep            # Deep cleanup (all Docker resources)
./reset.sh --force --start   # Reset and immediately restart
```

### System Status
```bash
# Check container status
docker-compose -p nis-protocol-v3 ps

# View logs
docker-compose -p nis-protocol-v3 logs -f nis-app

# Check system health
curl http://localhost/health | jq
```

## 🧪 Testing the Installation

### 1. Health Check
```bash
curl -s http://localhost/health | jq
```
**Expected response:**
```json
{
  "status": "healthy",
  "uptime": 123.45,
  "components": {
    "cognitive_system": "healthy",
    "infrastructure": "healthy",
    "consciousness": "healthy",
    "dashboard": "healthy"
  }
}
```

### 2. Process Intelligence Request
```bash
curl -X POST http://localhost/process \
  -H "Content-Type: application/json" \
  -d '{"text": "Analyze the concept of consciousness in AI systems", "generate_speech": false}' | jq
```

### 3. Monitor Consciousness
```bash
curl -s http://localhost/consciousness/status | jq
```

### 4. Check Infrastructure
```bash
curl -s http://localhost/infrastructure/status | jq
```

## 📋 Configuration Management

### Environment Variables

Key environment variables (auto-generated in `.env`):

```bash
# Database
DATABASE_URL=postgresql://nis_user:nis_password_2025@postgres:5432/nis_protocol_v3

# Kafka
KAFKA_BOOTSTRAP_SERVERS=kafka:9092

# Redis  
REDIS_HOST=redis
REDIS_PORT=6379

# Application
NIS_ENV=production
LOG_LEVEL=INFO
```

### Custom Configuration

1. **Modify before first start:**
   ```bash
   # Edit configuration files
   vim config/enhanced_infrastructure_config.json
   vim docker-compose.yml
   ./start.sh
   ```

2. **Modify running system:**
   ```bash
   # Edit and restart specific service
   vim config/enhanced_infrastructure_config.json
   docker-compose -p nis-protocol-v3 restart nis-app
   ```

### Persistent Data

Data is automatically persisted in Docker volumes:
- **postgres_data**: Database content
- **kafka_data**: Message queues  
- **redis_data**: Cache and sessions
- **grafana_data**: Monitoring dashboards
- **prometheus_data**: Metrics history

## 🔧 Development & Debugging

### Live Development
```bash
# Start with live code reloading
export NIS_ENV=development
./start.sh --with-monitoring

# Watch logs in real-time
docker-compose -p nis-protocol-v3 logs -f nis-app
```

### Debug Specific Services
```bash
# Debug Kafka connectivity
docker-compose -p nis-protocol-v3 exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Debug Redis
docker-compose -p nis-protocol-v3 exec redis redis-cli ping

# Debug PostgreSQL
docker-compose -p nis-protocol-v3 exec postgres psql -U nis_user -d nis_protocol_v3 -c "\l"

# Debug application logs
docker-compose -p nis-protocol-v3 exec nis-app tail -f /app/logs/app.log
```

### Performance Monitoring
```bash
# Real-time resource usage
docker stats

# Specific container metrics
docker stats nis-main-app nis-kafka nis-redis nis-postgres

# System metrics endpoint
curl -s http://localhost/metrics | jq
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Port Conflicts
```bash
# Check what's using ports
netstat -tulpn | grep :8000
netstat -tulpn | grep :5432

# Stop conflicting services
sudo systemctl stop postgresql  # If local PostgreSQL is running
sudo systemctl stop redis      # If local Redis is running
```

#### 2. Insufficient Memory
```bash
# Check available memory
free -h

# Stop unnecessary services
./stop.sh --remove-containers

# Start with minimal configuration
./start.sh  # Without --with-monitoring
```

#### 3. Docker Daemon Issues
```bash
# Restart Docker daemon
sudo systemctl restart docker

# Clean up Docker resources
docker system prune -a -f
./reset.sh --deep  # Nuclear option
```

#### 4. Network Connectivity
```bash
# Check Docker network
docker network ls
docker network inspect nis-protocol-v3_nis-network

# Test internal connectivity
docker-compose -p nis-protocol-v3 exec nis-app ping kafka
docker-compose -p nis-protocol-v3 exec nis-app ping redis
```

### Log Analysis
```bash
# Application logs
docker-compose -p nis-protocol-v3 logs nis-app | grep ERROR

# Infrastructure logs
docker-compose -p nis-protocol-v3 logs kafka | tail -50
docker-compose -p nis-protocol-v3 logs redis | tail -50

# Nginx access logs
docker-compose -p nis-protocol-v3 logs nginx | grep "POST\|GET"
```

## 📊 Production Deployment

### Resource Requirements

**Minimum (Core System):**
- CPU: 4 cores
- RAM: 8GB
- Disk: 20GB SSD
- Network: 1Gbps

**Recommended (Full Stack):**
- CPU: 8 cores
- RAM: 16GB  
- Disk: 50GB SSD
- Network: 1Gbps

### Security Considerations

1. **Change default passwords:**
   ```bash
   # Edit docker-compose.yml before deployment
   vim docker-compose.yml
   # Change POSTGRES_PASSWORD and GRAFANA_ADMIN_PASSWORD
   ```

2. **Enable HTTPS:**
   ```bash
   # Add SSL certificates to nginx.conf
   # Uncomment HTTPS server block in nginx.conf
   ```

3. **Network security:**
   - Use Docker secrets for sensitive data
   - Implement proper firewall rules
   - Consider VPN access for monitoring interfaces

### Scaling

#### Horizontal Scaling
```bash
# Scale application instances
docker-compose -p nis-protocol-v3 up -d --scale nis-app=3

# Scale Kafka consumers
docker-compose -p nis-protocol-v3 up -d --scale kafka-consumer=2
```

#### Vertical Scaling
```bash
# Modify resource limits in docker-compose.yml
services:
  nis-app:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
```

## 🔄 Backup & Recovery

### Automated Backups
```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker-compose -p nis-protocol-v3 exec -T postgres pg_dump -U nis_user nis_protocol_v3 > backup_${DATE}.sql
docker-compose -p nis-protocol-v3 exec -T redis redis-cli --rdb - > redis_backup_${DATE}.rdb
EOF

chmod +x backup.sh
./backup.sh
```

### Recovery
```bash
# Restore PostgreSQL
docker-compose -p nis-protocol-v3 exec -T postgres psql -U nis_user nis_protocol_v3 < backup_20250119_120000.sql

# Restore Redis (requires restart)
./stop.sh
cp redis_backup_20250119_120000.rdb data/redis/dump.rdb
./start.sh
```

## 🤝 Contributing

### Development Setup
```bash
# Fork the repository
git clone https://github.com/YOUR_USERNAME/NIS_Protocol.git
cd NIS_Protocol

# Create development branch
git checkout -b feature/docker-enhancement

# Start development environment
export NIS_ENV=development
./start.sh --with-monitoring

# Make changes and test
# ... your changes ...

# Test the changes
./reset.sh --force --start  # Clean test
curl http://localhost/health
```

### Adding New Services
1. Add service to `docker-compose.yml`
2. Update `nginx.conf` for routing
3. Modify `start.sh` for health checks
4. Update this documentation

## 📞 Support & Resources

- **📚 Main Documentation**: [README.md](README.md)
- **🔬 Technical Whitepaper**: [docs/NIS_Protocol_V3_Technical_Whitepaper.md](docs/NIS_Protocol_V3_Technical_Whitepaper.md)
- **🐛 Issue Tracking**: [GitHub Issues](https://github.com/Organica-Ai-Solutions/NIS_Protocol/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/Organica-Ai-Solutions/NIS_Protocol/discussions)

## 📈 Monitoring & Metrics

The system includes comprehensive monitoring:

- **Real-time Dashboard**: Live system metrics at `/dashboard/`
- **Prometheus Metrics**: Detailed metrics at `:9090`
- **Grafana Dashboards**: Advanced visualization at `:3000`
- **Health Endpoints**: System health at `/health`
- **Infrastructure Status**: Component status at `/infrastructure/status`

## 🌟 What's Next?

After your system is running:

1. **Explore the API**: Visit http://localhost/docs
2. **Monitor Performance**: Check http://localhost/dashboard/
3. **Test Consciousness**: Query http://localhost/consciousness/status
4. **Scale as Needed**: Use `docker-compose scale` commands
5. **Integrate with Your Apps**: Use the REST API endpoints

---

<div align="center">

**🧠 The Future of Neural Intelligence is Containerized 🐳**

*One command to deploy, infinite possibilities to explore*

⭐ **Star this repository if the Docker infrastructure helps your AI research!** ⭐

</div> 
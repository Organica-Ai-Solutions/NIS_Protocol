# NIS Protocol Production Runbook

**Version:** 1.0 | **Last Updated:** December 2025

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Deployment Procedures](#deployment-procedures)
3. [Monitoring & Alerting](#monitoring--alerting)
4. [Incident Response](#incident-response)
5. [Common Operations](#common-operations)
6. [Troubleshooting Playbooks](#troubleshooting-playbooks)
7. [Disaster Recovery](#disaster-recovery)

---

## System Overview

### Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                    NIS Protocol v4.0.1                       │
├─────────────────────────────────────────────────────────────┤
│  API Layer (FastAPI)                                         │
│  ├── 240+ REST Endpoints                                     │
│  ├── WebSocket for real-time control                         │
│  └── Rate limiting (30-1000 rpm)                             │
├─────────────────────────────────────────────────────────────┤
│  Agent Layer (47 Agents)                                     │
│  ├── Robotics Agent (FK/IK/Trajectory)                       │
│  ├── Physics Agent (PINN validation)                         │
│  ├── Isaac Integration (ROS 2 Bridge)                        │
│  └── Consciousness Pipeline                                  │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                        │
│  ├── Kafka (Event streaming)                                 │
│  ├── Redis (Caching, rate limiting)                          │
│  └── Zookeeper (Coordination)                                │
└─────────────────────────────────────────────────────────────┘
```

### Service Ports

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| NIS API | 8000 | HTTP | Main API |
| Kafka | 9092 | TCP | Event streaming |
| Redis | 6379 | TCP | Cache/Rate limit |
| Zookeeper | 2181 | TCP | Coordination |
| Prometheus | 9090 | HTTP | Metrics |
| Grafana | 3000 | HTTP | Dashboards |

### Health Endpoints

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `/health` | Basic health | `{"status": "healthy"}` |
| `/health/ready` | Readiness probe | 200 OK |
| `/health/live` | Liveness probe | 200 OK |
| `/resilience/health/deep` | Full dependency check | All checks pass |

---

## Deployment Procedures

### Pre-Deployment Checklist

```bash
# 1. Verify infrastructure is running
docker-compose ps

# 2. Check Kafka connectivity
docker exec nis-kafka kafka-topics.sh --list --bootstrap-server localhost:9092

# 3. Check Redis connectivity
docker exec nis-redis redis-cli ping

# 4. Run tests
python -m pytest tests/ -v

# 5. Build new image
docker build -t nis-protocol:$(git rev-parse --short HEAD) .
```

### Deployment Steps

```bash
# 1. Tag release
git tag -a v4.0.1 -m "Release v4.0.1"
git push origin v4.0.1

# 2. Build production image
docker build -t nis-protocol:v4.0.1 .
docker tag nis-protocol:v4.0.1 nis-protocol:latest

# 3. Stop old container (graceful)
docker stop nis-protocol --time 30

# 4. Start new container
docker run -d \
  --name nis-protocol \
  --network nis-network \
  -p 8000:8000 \
  -e KAFKA_BOOTSTRAP_SERVERS=nis-kafka:9092 \
  -e REDIS_HOST=nis-redis \
  -e NIS_API_KEY=${NIS_API_KEY} \
  nis-protocol:v4.0.1

# 5. Verify deployment
curl http://localhost:8000/health
curl http://localhost:8000/health/ready
```

### Rollback Procedure

```bash
# 1. Stop current container
docker stop nis-protocol

# 2. Start previous version
docker run -d \
  --name nis-protocol \
  --network nis-network \
  -p 8000:8000 \
  nis-protocol:v4.0.0  # Previous version

# 3. Verify rollback
curl http://localhost:8000/health
```

---

## Monitoring & Alerting

### Key Metrics to Watch

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| `nis_http_request_duration_seconds` P95 | >500ms | >2s | Scale up or optimize |
| `nis_http_requests_total` error rate | >5% | >10% | Check logs, investigate |
| `nis_health_status` | - | 0 | Immediate investigation |
| `nis_active_connections` | >80% capacity | >95% | Scale up |
| `nis_rate_limit_hits_total` | >100/min | >500/min | Review rate limits |

### Grafana Dashboards

Access: http://localhost:3000 (admin/nisprotocol)

| Dashboard | Purpose |
|-----------|---------|
| NIS Protocol Main | System overview, request rates, latency |
| Robotics & Physics | Kinematics, physics validations |
| LLM & AI | Token usage, provider latency |
| Infrastructure | Kafka, Redis, connections |
| Security | Auth attempts, rate limits |

### Alert Response

| Alert | Severity | Response |
|-------|----------|----------|
| `NISProtocolDown` | Critical | Check container, restart if needed |
| `HighLatency` | Warning | Check load, consider scaling |
| `HighErrorRate` | Warning | Check logs for errors |
| `KafkaConnectionLost` | Critical | Check Kafka, restart connection |
| `RedisConnectionLost` | Critical | Check Redis, restart connection |

---

## Incident Response

### Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| P1 | Service down | 15 min | API unreachable, data loss |
| P2 | Major degradation | 1 hour | High latency, partial outage |
| P3 | Minor issue | 4 hours | Non-critical feature broken |
| P4 | Low priority | 24 hours | Cosmetic issues |

### Incident Workflow

```
1. DETECT → Alert fires or user reports
2. TRIAGE → Assess severity, assign owner
3. INVESTIGATE → Check logs, metrics, traces
4. MITIGATE → Apply temporary fix
5. RESOLVE → Deploy permanent fix
6. POSTMORTEM → Document and prevent recurrence
```

### Log Locations

```bash
# Container logs
docker logs nis-protocol --tail 100 -f

# Structured logs (JSON)
docker logs nis-protocol 2>&1 | jq '.level, .message'

# Filter errors
docker logs nis-protocol 2>&1 | grep -i error
```

---

## Common Operations

### Restart Services

```bash
# Restart NIS Protocol
docker restart nis-protocol

# Restart all infrastructure
docker-compose restart

# Restart specific service
docker restart nis-kafka
docker restart nis-redis
```

### Scale Operations

```bash
# Scale with Docker Compose
docker-compose up -d --scale nis-protocol=3

# Check running instances
docker ps | grep nis-protocol
```

### Clear Cache

```bash
# Clear Redis cache
docker exec nis-redis redis-cli FLUSHDB

# Clear specific keys
docker exec nis-redis redis-cli KEYS "nis:cache:*" | xargs redis-cli DEL
```

### Rotate API Keys

```bash
# Generate new master key
export NEW_API_KEY=$(openssl rand -hex 32)

# Update environment
docker stop nis-protocol
docker run -d \
  --name nis-protocol \
  -e NIS_API_KEY=${NEW_API_KEY} \
  nis-protocol:latest

# Rotate via API
curl -X POST http://localhost:8000/security/keys/rotate \
  -H "X-API-Key: ${OLD_API_KEY}"
```

### View Audit Logs

```bash
# Via API
curl http://localhost:8000/security/audit-log?limit=100

# Filter by action
curl "http://localhost:8000/security/audit-log?action=key_created"
```

---

## Troubleshooting Playbooks

### API Not Responding

```bash
# 1. Check container status
docker ps | grep nis-protocol

# 2. Check container logs
docker logs nis-protocol --tail 50

# 3. Check health endpoint
curl -v http://localhost:8000/health

# 4. Check port binding
netstat -tlnp | grep 8000

# 5. Restart if needed
docker restart nis-protocol
```

### High Latency

```bash
# 1. Check current latency
curl -w "@curl-format.txt" http://localhost:8000/health

# 2. Check Prometheus metrics
curl http://localhost:8000/observability/metrics/prometheus | grep duration

# 3. Check Redis connection
docker exec nis-redis redis-cli ping

# 4. Check Kafka lag
docker exec nis-kafka kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --describe --all-groups

# 5. Check system resources
docker stats nis-protocol
```

### Kafka Connection Issues

```bash
# 1. Check Kafka is running
docker ps | grep kafka

# 2. Check Kafka logs
docker logs nis-kafka --tail 50

# 3. Test connectivity
docker exec nis-kafka kafka-topics.sh \
  --list --bootstrap-server localhost:9092

# 4. Check Zookeeper
docker exec nis-zookeeper zkCli.sh -server localhost:2181 stat

# 5. Restart Kafka
docker restart nis-kafka
```

### Redis Connection Issues

```bash
# 1. Check Redis is running
docker ps | grep redis

# 2. Check Redis logs
docker logs nis-redis --tail 50

# 3. Test connectivity
docker exec nis-redis redis-cli ping

# 4. Check memory usage
docker exec nis-redis redis-cli info memory

# 5. Restart Redis
docker restart nis-redis
```

### Rate Limiting Issues

```bash
# 1. Check current rate limit status
curl http://localhost:8000/security/status

# 2. View rate limit metrics
curl http://localhost:8000/observability/metrics/prometheus | grep rate_limit

# 3. Wait for reset (60 seconds)
sleep 60

# 4. Or clear rate limit cache
docker exec nis-redis redis-cli KEYS "nis:ratelimit:*" | xargs redis-cli DEL
```

---

## Disaster Recovery

### Backup Procedures

```bash
# Backup Redis data
docker exec nis-redis redis-cli BGSAVE
docker cp nis-redis:/data/dump.rdb ./backups/redis-$(date +%Y%m%d).rdb

# Backup Kafka topics
docker exec nis-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic nis-events \
  --from-beginning > ./backups/kafka-events-$(date +%Y%m%d).json

# Backup configuration
cp .env ./backups/.env-$(date +%Y%m%d)
cp docker-compose.yml ./backups/docker-compose-$(date +%Y%m%d).yml
```

### Restore Procedures

```bash
# Restore Redis
docker cp ./backups/redis-20251203.rdb nis-redis:/data/dump.rdb
docker restart nis-redis

# Restore Kafka topics
docker exec -i nis-kafka kafka-console-producer.sh \
  --bootstrap-server localhost:9092 \
  --topic nis-events < ./backups/kafka-events-20251203.json
```

### Full System Recovery

```bash
# 1. Start infrastructure
docker-compose up -d kafka redis zookeeper

# 2. Wait for infrastructure
sleep 30

# 3. Restore data
./scripts/restore-backup.sh 20251203

# 4. Start NIS Protocol
docker-compose up -d nis-protocol

# 5. Verify recovery
curl http://localhost:8000/health
curl http://localhost:8000/resilience/health/deep
```

---

## Contacts

| Role | Contact | Escalation |
|------|---------|------------|
| On-Call Engineer | oncall@example.com | PagerDuty |
| Platform Lead | platform@example.com | Slack #platform |
| Security Team | security@example.com | Slack #security |

---

**Document Owner:** Platform Team  
**Review Cycle:** Monthly  
**Last Review:** December 2025

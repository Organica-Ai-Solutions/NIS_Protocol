# NIS Protocol - Developer Dashboard

## 🎯 Overview

Comprehensive Grafana dashboard for the NIS Protocol development team. This dashboard provides real-time insights into system performance, LLM operations, consciousness metrics, and development-specific analytics.

**Dashboard Location**: `monitoring/grafana-dev-dashboard.json`

## 📊 Dashboard Sections

### 1. 🚀 System Overview
- **Total Requests/sec**: Real-time request rate with color-coded thresholds
- **Error Rate**: System-wide error rate monitoring
- **P95 Latency**: 95th percentile response time
- **Active Conversations**: Current active chat sessions
- **Active Agents**: Number of registered AI agents
- **Cache Hit Rate**: Overall caching effectiveness

### 2. 🤖 LLM Performance
- **LLM Requests by Provider**: Distribution across OpenAI, Anthropic, Google, etc.
- **LLM Success Rate**: Provider reliability over time
- **LLM Latency by Provider**: Response time comparison
- **LLM Token Usage**: Token consumption tracking
- **Provider Fallbacks**: Automatic failover events

### 3. 📊 Request Analytics
- **Requests by Endpoint**: Traffic distribution across API endpoints
- **Request Latency Percentiles**: p50, p90, p95, p99 latency tracking
- **Errors by Type**: Categorized error monitoring
- **Errors by Endpoint**: Endpoint-specific error rates

### 4. 🧠 Consciousness & Ethics
- **Consciousness Level**: Real-time consciousness metric (0-1)
- **Ethics Score**: Ethical decision-making score (0-1)
- **Trend Analysis**: Historical consciousness and ethics tracking

### 5. 💾 Cache Performance
- **Cache Hits vs Misses**: Cache operation breakdown
- **Cache Hit Rate by Type**: Performance by cache category

### 6. 🔧 Development Metrics
- **Request Success Rate**: Overall system reliability
- **Average Response Time**: Mean latency across all requests
- **Total Requests (24h)**: Daily request volume
- **Total Errors (24h)**: Daily error count
- **Request Volume Heatmap**: Visual traffic patterns

## 🚀 Setup Instructions

### Option 1: Docker Compose (Recommended)

```bash
cd /Users/diegofuego/Desktop/NIS_Protocol/monitoring

# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Verify services are running
docker-compose -f docker-compose.monitoring.yml ps
```

**Access Points**:
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- Alertmanager: http://localhost:9093

**Default Credentials**:
- Username: `admin`
- Password: `nisprotocol`

### Option 2: Manual Import

1. Open Grafana at http://localhost:3000
2. Navigate to **Dashboards** → **Import**
3. Upload `grafana-dev-dashboard.json`
4. Select Prometheus as the data source
5. Click **Import**

## 📈 Available Metrics

All metrics are exposed at `/metrics/prometheus` endpoint on the NIS Protocol backend (port 8000).

### Request Metrics
- `nis_requests_total` - Total request count by endpoint, method, status
- `nis_request_latency_seconds` - Request latency histogram

### LLM Metrics
- `nis_llm_requests_total` - LLM requests by provider, model, success
- `nis_llm_tokens_total` - Token usage by provider and type
- `nis_llm_latency_seconds` - LLM response latency
- `nis_llm_fallbacks_total` - Provider failover events

### System Metrics
- `nis_active_conversations` - Active chat sessions
- `nis_active_agents` - Registered AI agents
- `nis_cache_hits_total` - Cache hits by type
- `nis_cache_misses_total` - Cache misses by type

### Consciousness Metrics
- `nis_consciousness_level` - Consciousness level (0-1)
- `nis_ethics_score` - Ethics score (0-1)

### Error Metrics
- `nis_errors_total` - Errors by type and endpoint

## 🔍 Usage Tips

### Quick Health Check
1. Check **System Overview** row for any red indicators
2. Verify **Error Rate** is below 1 error/sec
3. Confirm **P95 Latency** is under 5 seconds
4. Check **Cache Hit Rate** is above 80%

### LLM Performance Analysis
1. Review **LLM Requests by Provider** for load distribution
2. Check **LLM Success Rate** - should be >95% for all providers
3. Monitor **Provider Fallbacks** for stability issues
4. Track **Token Usage** for cost optimization

### Debugging Issues
1. **High Error Rate**: Check **Errors by Type** and **Errors by Endpoint**
2. **Slow Responses**: Review **Request Latency Percentiles**
3. **LLM Issues**: Check **LLM Success Rate** and **LLM Latency**
4. **Cache Problems**: Analyze **Cache Hits vs Misses**

### Time Range Selection
- **Real-time monitoring**: Last 15 minutes
- **Recent issues**: Last 1-6 hours
- **Trend analysis**: Last 24 hours - 7 days
- **Historical review**: Last 30 days

## 🎨 Customization

### Adding Custom Panels

```json
{
  "title": "Your Custom Panel",
  "type": "timeseries",
  "gridPos": {"x": 0, "y": 0, "w": 12, "h": 6},
  "targets": [
    {
      "expr": "your_prometheus_query",
      "legendFormat": "{{label}}"
    }
  ]
}
```

### Creating Alerts

1. Click on any panel
2. Select **Edit**
3. Go to **Alert** tab
4. Configure alert conditions
5. Set notification channels

## 🔔 Alert Configuration

### Recommended Alerts

**High Error Rate**:
```promql
sum(rate(nis_errors_total[5m])) > 1
```

**High Latency**:
```promql
histogram_quantile(0.95, rate(nis_request_latency_seconds_bucket[5m])) > 5
```

**Low Cache Hit Rate**:
```promql
sum(rate(nis_cache_hits_total[5m])) / (sum(rate(nis_cache_hits_total[5m])) + sum(rate(nis_cache_misses_total[5m]))) < 0.5
```

**LLM Provider Down**:
```promql
sum by (provider) (rate(nis_llm_requests_total{success='False'}[5m])) / sum by (provider) (rate(nis_llm_requests_total[5m])) > 0.1
```

## 📝 Maintenance

### Updating the Dashboard

```bash
# Edit the dashboard JSON
vim monitoring/grafana-dev-dashboard.json

# Restart Grafana to reload
docker-compose -f monitoring/docker-compose.monitoring.yml restart grafana
```

### Backup Dashboard

```bash
# Export current dashboard
curl -u admin:nisprotocol http://localhost:3000/api/dashboards/uid/nis-dev-team > backup.json

# Import from backup
curl -X POST -H "Content-Type: application/json" \
  -u admin:nisprotocol \
  -d @backup.json \
  http://localhost:3000/api/dashboards/db
```

### Prometheus Data Retention

Default retention: 15 days (configured in docker-compose.monitoring.yml)

To change:
```yaml
command:
  - '--storage.tsdb.retention.time=30d'  # Change to 30 days
```

## 🐛 Troubleshooting

### Dashboard Not Loading
```bash
# Check Grafana logs
docker logs nis-grafana

# Check Prometheus connection
curl http://localhost:9090/api/v1/query?query=up
```

### No Data Showing
```bash
# Verify NIS Protocol is exposing metrics
curl http://localhost:8000/metrics/prometheus

# Check Prometheus targets
open http://localhost:9090/targets
```

### Metrics Not Updating
```bash
# Restart Prometheus
docker-compose -f monitoring/docker-compose.monitoring.yml restart prometheus

# Check scrape config
cat monitoring/prometheus.yml
```

## 🔐 Security Notes

**⚠️ IMPORTANT**: This dashboard is for **development team only**

- Not exposed to end users
- Contains internal system metrics
- Includes sensitive performance data
- Access controlled via Grafana authentication

### Production Deployment

For production, update credentials:

```yaml
environment:
  - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
  - GF_USERS_ALLOW_SIGN_UP=false
  - GF_AUTH_ANONYMOUS_ENABLED=false
```

## 📚 Additional Resources

- [Prometheus Query Language](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/dashboards/)
- [NIS Protocol Metrics Documentation](../src/monitoring/prometheus_metrics.py)

## 🤝 Contributing

To add new metrics:

1. Update `src/monitoring/prometheus_metrics.py`
2. Add metric to dashboard JSON
3. Test with sample data
4. Update this README
5. Submit PR for review

## 📞 Support

For dashboard issues or feature requests:
- Create an issue in the NIS Protocol repository
- Tag with `monitoring` label
- Include dashboard version and Grafana version

---

**Dashboard Version**: 1.0.0  
**Last Updated**: December 2024  
**Maintained By**: NIS Protocol Dev Team

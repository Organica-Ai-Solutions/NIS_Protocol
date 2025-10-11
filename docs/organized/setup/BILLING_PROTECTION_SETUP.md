# 🛡️ Billing Protection Setup Guide

**Prevent unexpected API charges with NIS Protocol v3.2's built-in protection**

## 🚨 Quick Start - Safe Mode

**For Development/Testing (ZERO billing risk):**
```bash
git clone https://github.com/Organica-Ai-Solutions/NIS_Protocol.git
cd NIS_Protocol
./start_safe.sh  # 🛡️ Mock responses only - NO API CHARGES
```

## ⚠️ Production Mode (Billing Risk)

**Only use when you need real AI responses and understand the costs:**

### 1. Configure Real API Keys
```bash
# Copy example configuration
cp .env.example .env

# Edit .env file and:
# - Add your real API keys
# - Set FORCE_MOCK_MODE=false
# - Set DISABLE_REAL_API_CALLS=false
```

### 2. Set Up Billing Protection
```bash
# Set up Google Cloud billing alerts
gcloud billing budgets create \
  --billing-account=YOUR_BILLING_ACCOUNT \
  --display-name="NIS Protocol Alert" \
  --budget-amount=10 \
  --threshold-rule=percent=0.5,basis=current-spend

# Enable monitoring
python3 scripts/utilities/billing_monitor.py &
```

### 3. Start with Monitoring
```bash
./start.sh  # ⚠️ BILLING RISK - Monitor closely
```

## 🛡️ Protection Features

### Environment Variables
- `FORCE_MOCK_MODE=true` - Forces mock responses (default)
- `DISABLE_REAL_API_CALLS=true` - Prevents real API calls (default)

### Scripts
- `./start_safe.sh` - Safe start with mock responses
- `./scripts/emergency/emergency_shutdown.sh` - Emergency stop
- `./scripts/utilities/billing_monitor.py` - Usage monitoring

### Billing Alerts
- Google Cloud budgets with early warnings
- Email notifications at spending thresholds
- Automatic shutdown after time limits

## 📊 Monitoring Your Usage

### Check Running Containers
```bash
docker ps | grep nis
```

### Monitor Google Cloud Billing
```bash
gcloud billing accounts list
gcloud billing budgets list --billing-account=YOUR_ACCOUNT
```

### Emergency Shutdown
```bash
./scripts/emergency/emergency_shutdown.sh
```

## 💰 Cost Estimates

### Mock Mode (Safe)
- **Cost**: $0.00
- **Functionality**: Full system with simulated responses
- **Use for**: Development, testing, demos

### Real API Mode (Production)
- **Google Gemini**: ~$0.075 per 1K input tokens
- **OpenAI GPT-4**: ~$0.03 per 1K input tokens
- **Anthropic Claude**: ~$0.015 per 1K input tokens
- **Continuous usage**: Can generate significant charges

### Example Costs
- **1 hour of chat**: $1-5 depending on usage
- **24 hours continuous**: $25-100+ 
- **Heavy development**: $10-50 per day

## 🚨 Best Practices

### ✅ DO:
- Always use `./start_safe.sh` for development
- Set up billing alerts before using real APIs
- Monitor container uptime regularly
- Stop containers when not in use
- Use mock mode for testing and demos

### ❌ DON'T:
- Leave containers running overnight without monitoring
- Use real API keys without billing alerts
- Ignore billing notification emails
- Assume "testing" won't cost money
- Forget to run `docker-compose down` when done

## 🆘 Emergency Procedures

### If You See Unexpected Charges:
1. **IMMEDIATE**: `./scripts/emergency/emergency_shutdown.sh`
2. **Verify**: `docker ps` (should show no nis containers)
3. **Check**: Google Cloud Console billing
4. **Disable**: API keys in cloud console if needed

### Daily Monitoring Checklist:
- [ ] Check `docker ps` for running containers
- [ ] Review billing dashboard
- [ ] Confirm containers stopped when not needed
- [ ] Monitor email for billing alerts

## 📞 Support

For billing protection issues:
- Emergency shutdown: `./scripts/emergency/emergency_shutdown.sh`
- Documentation: `README_BILLING_PROTECTION.md`
- Google Cloud Console: https://console.cloud.google.com/billing

**Remember: Your budget is important! Always use safe mode for development.** 🏠💰

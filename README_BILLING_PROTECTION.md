# 🛡️ BILLING PROTECTION GUIDE

**NEVER GET SURPRISED BY API CHARGES AGAIN!**

## 🚨 What Happened?

Your NIS Protocol ran for 48+ hours with **REAL Google API keys**, generating **$323+ in charges** from continuous Gemini API calls.

## 🛡️ Protection Measures Implemented

### 1. **Emergency Shutdown**
```bash
# IMMEDIATE STOP - Use if you see unexpected charges
./scripts/emergency/emergency_shutdown.sh
```

### 2. **Safe Start Mode**
```bash
# ALWAYS use this for development/testing
./start_safe.sh
```
- ✅ Mock responses only
- ✅ No real API calls
- ✅ No billing risk

### 3. **Billing Alerts Set Up**
- 🚨 $5 monthly budget with early warnings
- 📧 Email alerts at 25%, 50%, 75%, 100%
- 🛑 Automatic notifications

### 4. **Code-Level Protection**
- 🛡️ `FORCE_MOCK_MODE=true` by default
- ⚠️ Loud warnings when real APIs are enabled
- 🔒 Environment variables control real API access

## 🚀 Safe Usage Guide

### For Development/Testing:
```bash
./start_safe.sh  # SAFE - No billing risk
```

### For Production (CAREFUL!):
```bash
# 1. Edit .env manually
# 2. Set FORCE_MOCK_MODE=false
# 3. Add real API keys
# 4. Set strict daily limits
./start.sh
```

## 🚨 Emergency Procedures

### If You See Unexpected Charges:
1. **IMMEDIATE**: Run `./scripts/emergency/emergency_shutdown.sh`
2. **Check**: `docker ps` (should show no running containers)
3. **Verify**: Google Cloud Console billing page
4. **Disable**: API keys in Google Cloud Console

### Daily Monitoring:
```bash
# Check if anything is running
docker ps

# Check Google Cloud billing
gcloud billing accounts list
```

## 🔧 Configuration Files

### `.env.safe` - No Billing Risk
- All API keys commented out
- `FORCE_MOCK_MODE=true`
- Mock responses only

### `.env` - Production (DANGEROUS)
- Real API keys (use carefully)
- Set `FORCE_MOCK_MODE=false` only when needed
- Monitor billing closely

## 📊 Monitoring Tools

### Billing Monitor (Auto-shutdown)
```bash
# Monitors containers and auto-shuts down after 4 hours
python3 scripts/utilities/billing_monitor.py
```

### Manual Checks
```bash
# Check running containers
docker ps | grep nis

# Check Google Cloud quotas
gcloud compute project-info describe --project=organicaaisolutions
```

## 🎯 Best Practices

### ✅ DO:
- Always use `./start_safe.sh` for development
- Set up billing alerts
- Monitor container uptime
- Use mock responses for testing
- Check billing daily during active development

### ❌ DON'T:
- Leave containers running overnight
- Use real API keys without monitoring
- Forget to run `docker-compose down` when done
- Ignore billing alert emails
- Assume "it's just testing" won't cost money

## 🆘 Emergency Contacts

### Immediate Actions:
1. `./scripts/emergency/emergency_shutdown.sh`
2. `docker-compose down`
3. `docker stop $(docker ps -q)`

### Billing Issues:
- Google Cloud Console: https://console.cloud.google.com/billing
- Disable API keys: https://console.cloud.google.com/apis/credentials

## 💡 Remember

**"Better safe than sorry"** - Always assume real APIs will cost money and protect accordingly!

Your budget is important! 🏠💰

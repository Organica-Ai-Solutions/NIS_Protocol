# NIS Protocol - Tiered Access System

**Date**: December 27, 2025  
**Status**: ✅ Implemented

---

## 🎯 **User Tiers**

NIS Protocol now supports **6 user tiers** with different rate limits and access levels:

| Tier | Rate Limit | API Key Prefix | Use Case |
|------|------------|----------------|----------|
| **Free** | 10 req/min | `sk-free-` | Testing, personal projects |
| **Premium** | 60 req/min | `sk-prem-` | Small applications |
| **Pro** | 300 req/min | `sk-pro-` | Professional applications |
| **Enterprise** | 1000 req/min | `sk-ent-` | Large-scale deployments |
| **Admin** | 5000 req/min | `sk-admin-` | System administration |
| **Master** | Unlimited | `sk-master-` | Full system access |

---

## 🔑 **API Key Format**

API keys are prefixed to indicate tier level:

```
sk-free-abc123xyz...     # Free tier
sk-prem-abc123xyz...     # Premium tier
sk-pro-abc123xyz...      # Pro tier
sk-ent-abc123xyz...      # Enterprise tier
sk-admin-abc123xyz...    # Admin tier
sk-master-abc123xyz...   # Master tier
```

---

## 📊 **Rate Limiting**

### **How It Works**

1. **Tier Detection**: System reads API key prefix to determine tier
2. **Rate Calculation**: Applies tier-specific rate limit
3. **Sliding Window**: Tracks requests over 60-second window
4. **Response Headers**: Returns tier info in headers

### **Response Headers**

```http
X-RateLimit-Limit: 300
X-RateLimit-Remaining: 287
X-RateLimit-Reset: 45
X-RateLimit-Tier: pro
```

### **Rate Limit Exceeded Response**

```json
{
  "error": "Rate limit exceeded",
  "retry_after": 45,
  "tier": "free",
  "message": "Rate limit for free tier exceeded. Upgrade for higher limits."
}
```

---

## 🚀 **Usage Examples**

### **Free Tier (10 req/min)**

```bash
curl -H "X-API-Key: sk-free-your-key-here" \
  http://localhost:8000/chat \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello"}'
```

### **Pro Tier (300 req/min)**

```bash
curl -H "X-API-Key: sk-pro-your-key-here" \
  http://localhost:8000/research/deep \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"query":"AI research"}'
```

### **Master Tier (Unlimited)**

```bash
curl -H "X-API-Key: sk-master-your-key-here" \
  http://localhost:8000/autonomous/plan \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"goal":"Deploy system"}'
```

---

## 🔧 **Implementation Details**

### **Tier Detection**

```python
from src.security.auth import UserTier, rate_limiter

# Detect tier from API key
tier = rate_limiter.get_tier_from_api_key("sk-pro-abc123")
# Returns: UserTier.PRO

# Get rate limit for tier
limit = rate_limiter.get_rate_limit_for_tier(tier)
# Returns: 300
```

### **Rate Limit Check**

```python
# Check if request is allowed
allowed, remaining, reset, tier = rate_limiter.check_limit(
    identifier="192.168.1.1",
    api_key="sk-pro-abc123"
)

if not allowed:
    # Rate limit exceeded
    print(f"Try again in {reset} seconds")
else:
    # Request allowed
    print(f"{remaining} requests remaining")
```

---

## 💰 **Tier Comparison**

### **Free Tier**
- ✅ 10 requests/minute
- ✅ All endpoints accessible
- ✅ Perfect for testing
- ❌ Limited for production

### **Premium Tier**
- ✅ 60 requests/minute
- ✅ 6x more than free
- ✅ Good for small apps
- ⚠️ May need upgrade for growth

### **Pro Tier**
- ✅ 300 requests/minute
- ✅ 30x more than free
- ✅ Professional applications
- ✅ Suitable for most use cases

### **Enterprise Tier**
- ✅ 1000 requests/minute
- ✅ 100x more than free
- ✅ Large-scale deployments
- ✅ High-traffic applications

### **Admin Tier**
- ✅ 5000 requests/minute
- ✅ 500x more than free
- ✅ System administration
- ✅ Internal tools

### **Master Tier**
- ✅ Unlimited requests
- ✅ No rate limiting
- ✅ Full system access
- ✅ Development & testing

---

## 🔐 **Security Features**

### **Automatic Tier Detection**
- No manual configuration needed
- Tier determined by API key prefix
- Cached for performance

### **Sliding Window Algorithm**
- Precise rate limiting
- No burst abuse
- Fair distribution

### **Graceful Degradation**
- Unknown keys default to Free tier
- No API key = Free tier
- System always accessible

---

## 📈 **Monitoring**

### **Check Your Tier**

```bash
# Any request returns tier info in headers
curl -I -H "X-API-Key: sk-pro-your-key" \
  http://localhost:8000/health

# Response headers:
# X-RateLimit-Tier: pro
# X-RateLimit-Limit: 300
# X-RateLimit-Remaining: 299
```

### **Rate Limit Status**

All responses include rate limit headers showing:
- Current tier
- Total limit
- Remaining requests
- Reset time

---

## 🎓 **Best Practices**

### **For Free Tier Users**
1. Cache responses when possible
2. Batch requests together
3. Use webhooks instead of polling
4. Consider upgrading for production

### **For Pro/Enterprise Users**
1. Implement exponential backoff
2. Monitor rate limit headers
3. Distribute load across time
4. Use connection pooling

### **For Admin/Master Users**
1. Still implement rate limiting client-side
2. Don't abuse unlimited access
3. Monitor for anomalies
4. Rotate keys regularly

---

## 🔄 **Upgrading Tiers**

To upgrade your tier:

1. **Contact Admin** - Request tier upgrade
2. **Get New Key** - Receive key with new prefix
3. **Update Config** - Replace old key with new one
4. **Test** - Verify new rate limits apply

---

## 🐛 **Troubleshooting**

### **Rate Limit Exceeded**

**Problem**: Getting 429 errors  
**Solution**: 
- Check your tier in response headers
- Wait for reset time
- Upgrade tier if needed
- Implement request queuing

### **Wrong Tier Detected**

**Problem**: System shows wrong tier  
**Solution**:
- Verify API key prefix is correct
- Check for typos in key
- Ensure key is properly formatted
- Contact admin if issue persists

### **Unlimited Access Not Working**

**Problem**: Master key still rate limited  
**Solution**:
- Verify key starts with `sk-master-`
- Check `DISABLE_RATE_LIMIT` env var
- Review middleware logs
- Restart service

---

## 📝 **Configuration**

### **Environment Variables**

```bash
# Disable rate limiting entirely (testing only)
DISABLE_RATE_LIMIT=true

# Set master API key
NIS_API_KEY=sk-master-your-master-key-here
```

### **Custom Rate Limits**

Edit `src/security/auth.py`:

```python
TIER_RATE_LIMITS = {
    UserTier.FREE: 10,           # Adjust as needed
    UserTier.PREMIUM: 60,
    UserTier.PRO: 300,
    UserTier.ENTERPRISE: 1000,
    UserTier.ADMIN: 5000,
    UserTier.MASTER: float('inf')
}
```

---

## ✅ **Summary**

**Tier System Features**:
- ✅ 6 user tiers (Free to Master)
- ✅ Automatic tier detection from API key
- ✅ Sliding window rate limiting
- ✅ Graceful error messages
- ✅ Response headers with tier info
- ✅ Unlimited access for Master tier
- ✅ Easy to upgrade/downgrade

**Benefits**:
- Fair resource allocation
- Prevents abuse
- Scalable pricing model
- Clear upgrade path
- Production-ready security

**Next Steps**:
1. Generate API keys for each tier
2. Test rate limits
3. Monitor usage
4. Adjust limits as needed

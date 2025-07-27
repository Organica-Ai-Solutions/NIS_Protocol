# 📬 NIS Protocol v3 - Postman Collection Import Guide

## 🚀 Quick Import Instructions

### **Step 1: Import the Collection**
1. **Open Postman** (Download from: https://www.postman.com/downloads/)
2. **Click "Import"** (top left corner)
3. **Select "Upload Files"**
4. **Choose:** `NIS_Protocol_v3_Postman_Collection.json`
5. **Click "Import"**

### **Step 2: Configure Environment (Optional)**
The collection uses variables that you can customize:
- `base_url`: Default is `http://localhost:8000`
- `dashboard_url`: Default is `http://localhost:5000`

To change these:
1. **Click the ⚙️ gear icon** (top right)
2. **Select "Manage Environments"**
3. **Create new environment** or use globals
4. **Set variables** if you're using different ports

## 📋 Collection Overview

### **🏠 System Information**
- **Root - System Info** (`GET /`) - Basic system information
- **Health Check** (`GET /health`) - System health status

### **🧠 Core Processing**
- **Process Input - Text Analysis** (`POST /process`) - Analyze complex topics
- **Process Input - Creative Writing** (`POST /process`) - Generate creative content  
- **Process Input - Problem Solving** (`POST /process`) - Solve engineering problems

### **🧠 Consciousness & Status**
- **Consciousness Status** (`GET /consciousness/status`) - Cognitive state monitoring
- **Infrastructure Status** (`GET /infrastructure/status`) - Kafka, Redis, PostgreSQL status

### **📊 Monitoring & Metrics**
- **System Metrics** (`GET /metrics`) - Performance metrics
- **Real-Time Dashboard** (`GET :5000/`) - Live monitoring interface

### **🔧 Administration**
- **Restart Services** (`POST /admin/restart`) - Restart infrastructure services

### **🧪 Advanced Testing**
- **Stress Test** - Load testing with random data
- **Large Payload Test** - Test capacity limits
- **Invalid Payload Test** - Error handling validation

## 🎯 Testing Strategy

### **Recommended Testing Order:**

1. **🏥 Health Check First**
   ```
   GET /health
   ```
   **Expected:** `200 OK` with system status

2. **🏠 System Info**
   ```
   GET /
   ```
   **Expected:** Welcome message and system details

3. **🏗️ Infrastructure Status**
   ```
   GET /infrastructure/status
   ```
   **Expected:** Kafka, Redis, PostgreSQL connection status

4. **🧠 Consciousness Monitoring**
   ```
   GET /consciousness/status
   ```
   **Expected:** Cognitive state and agent status

5. **📊 Performance Metrics**
   ```
   GET /metrics
   ```
   **Expected:** Detailed performance data

6. **🧠 Core Processing Tests**
   ```
   POST /process
   ```
   **Test with different payloads** (analysis, creative, problem-solving)

## 🔧 Customizing Requests

### **Modify Processing Requests:**
You can customize the `/process` endpoint payloads:

```json
{
  "text": "Your custom text here",
  "context": "your_context",
  "processing_type": "analysis|generation|problem_solving"
}
```

### **Add Custom Headers:**
For authenticated endpoints (if implemented):
```
Authorization: Bearer your-token-here
```

## 🚨 Troubleshooting

### **Connection Refused?**
- ✅ Check if Docker containers are running: `docker ps`
- ✅ Verify ports: `8000` (API), `5000` (Dashboard)
- ✅ Try: `http://localhost:8000/health` in browser first

### **500 Internal Server Error?**
- ✅ Check container logs: `docker logs nis-main-app`
- ✅ Verify API keys in `.env` file
- ✅ Check infrastructure status endpoint

### **Timeout Errors?**
- ✅ Some processing requests may take 10-30 seconds
- ✅ Increase Postman timeout in settings
- ✅ Check system resources with `/metrics`

## 📈 Expected Response Times

| Endpoint | Typical Response Time |
|----------|---------------------|
| `/health` | < 100ms |
| `/` | < 200ms |
| `/infrastructure/status` | < 500ms |
| `/consciousness/status` | < 1000ms |
| `/metrics` | < 1000ms |
| `/process` | 5-30 seconds (depends on complexity) |

## 🎉 Success Indicators

### **✅ System is Working When:**
- Health check returns `200 OK`
- Infrastructure status shows all services connected
- Process requests return meaningful responses
- Metrics show reasonable performance data
- Dashboard accessible at `:5000`

### **❌ System Needs Attention When:**
- Health check fails or times out
- Infrastructure status shows disconnected services
- Process requests return errors or hang
- Metrics show resource exhaustion
- Dashboard not accessible

## 🔄 Automated Testing

The collection includes **automated tests** that run after each request:
- ✅ **Status Code Validation** (200/201/202)
- ✅ **Response Time Check** (< 30 seconds)
- ✅ **JSON/Content Validation**

To see test results:
1. **Run a request**
2. **Check "Test Results" tab** below the response
3. **Green = Pass, Red = Fail**

## 🚀 Ready to Test!

Your NIS Protocol v3 system should be running at:
- **🌐 API:** http://localhost:8000
- **📊 Dashboard:** http://localhost:5000

**Start with the Health Check** and work your way through the collection! 🎯 
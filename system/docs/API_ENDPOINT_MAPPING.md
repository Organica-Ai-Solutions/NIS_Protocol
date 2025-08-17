# 🔌 HTML Files API Endpoint Mapping

**Connecting HTML interfaces to NIS Protocol v3.2 working APIs**

## 📊 **Current Status Analysis**

### ✅ **Working Connections**
- **enhanced_agent_chat.html** → `/chat` ✅ **Perfect!**
- **chat_console.html** → `/health` ✅ **Working!**
- **chat_console.html** → `/research/deep` ✅ **Working!**

### ❌ **Needs Updates**

#### **modern_chat.html**
- `/tools/run` → **Needs mapping to working endpoint**
- `/chat/stream` → **Map to `/chat/enhanced`**

#### **chat_console.html** (Multiple endpoints need mapping)
- `/vision/analyze` → **Map to working agent endpoint**
- `/document/analyze` → **Map to working research endpoint** 
- `/reasoning/collaborative` → **Map to working agent endpoint**
- `/visualization/*` → **Map to working endpoints**
- `/image/generate` → **Map to working endpoint**
- `/pipeline/*` → **Map to working system endpoints**

## 🎯 **Endpoint Mapping Strategy**

### **Our 32 Working Endpoints:**
1. **System**: `/`, `/health`, `/status`, `/docs`, `/openapi.json`
2. **Physics**: `/physics/capabilities`, `/physics/validate`, `/physics/constants`, `/physics/pinn/solve`
3. **NVIDIA NeMo**: `/nvidia/nemo/status`, `/nvidia/nemo/enterprise/showcase`, `/nvidia/nemo/cosmos/demo`, `/nvidia/nemo/toolkit/status`, `/nvidia/nemo/physics/simulate`, `/nvidia/nemo/orchestrate`, `/nvidia/nemo/toolkit/test`
4. **Research**: `/research/capabilities`, `/research/deep`, `/research/arxiv`, `/research/analyze`
5. **Agents**: `/agents/status`, `/agents/consciousness/analyze`, `/agents/memory/store`, `/agents/planning/create`, `/agents/capabilities`
6. **MCP**: `/api/mcp/demo`, `/api/langgraph/status`, `/api/langgraph/invoke`
7. **Chat**: `/chat`, `/chat/enhanced`, `/chat/sessions`, `/chat/memory/{session_id}`

## 🔄 **Mapping Plan**

### **Non-working → Working Mappings:**
- `/tools/run` → `/api/langgraph/invoke`
- `/chat/stream` → `/chat/enhanced`
- `/vision/analyze` → `/agents/consciousness/analyze`
- `/document/analyze` → `/research/analyze`
- `/reasoning/collaborative` → `/agents/planning/create`
- `/image/generate` → `/nvidia/nemo/cosmos/demo`
- `/visualization/*` → `/physics/constants` (for data)
- `/pipeline/*` → `/status` (for monitoring)

## ✅ **Action Items**
1. Update modern_chat.html API calls
2. Update chat_console.html API calls  
3. Test all connections work properly
4. Ensure graceful fallbacks for missing features

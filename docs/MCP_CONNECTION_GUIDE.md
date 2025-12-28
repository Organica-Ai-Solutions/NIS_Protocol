# 🔌 Connect NIS Protocol to ChatGPT & Claude

## ✅ **Your MCP Server is Already Running!**

The NIS Protocol MCP server is active and ready to connect to AI assistants.

---

## 🤖 **Connect to ChatGPT**

### **Method 1: ChatGPT Desktop App (Recommended)**

1. **Install ChatGPT Desktop** (if not already installed)
   - Download from: https://chatgpt.com/

2. **Open ChatGPT Settings**
   - Click your profile → Settings → Developer Mode

3. **Add MCP Server**
   - Enable "Developer Mode"
   - Click "Add MCP Server"
   - Copy the contents from `chatgpt_mcp_config.json`
   - Save

4. **Restart ChatGPT**
   - Close and reopen the app
   - You should see "nis_protocol" in the MCP servers list

### **Method 2: Manual Configuration**

Copy the config file to ChatGPT's config directory:

```bash
# macOS
mkdir -p ~/Library/Application\ Support/ChatGPT/
cp chatgpt_mcp_config.json ~/Library/Application\ Support/ChatGPT/mcp_config.json

# Linux
mkdir -p ~/.config/chatgpt/
cp chatgpt_mcp_config.json ~/.config/chatgpt/mcp_config.json
```

---

## 🧠 **Connect to Claude Desktop**

### **Setup Steps:**

1. **Install Claude Desktop** (if not already installed)
   - Download from: https://claude.ai/download

2. **Copy Config File**

```bash
# macOS
mkdir -p ~/Library/Application\ Support/Claude/
cp claude_mcp_config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Linux
mkdir -p ~/.config/claude/
cp claude_mcp_config.json ~/.config/claude/claude_desktop_config.json
```

3. **Restart Claude Desktop**
   - Close and reopen the app
   - The NIS Protocol tools will be available

---

## 🧪 **Test the Connection**

### **In ChatGPT or Claude, try:**

**Test 1: List Capabilities**
```
Use the nis.list_capabilities tool to show me what NIS Protocol can do
```

**Test 2: Robotics Control**
```
Use nis.robotics_control to calculate forward kinematics for a drone with motor speeds [5000, 5000, 5000, 5000]
```

**Test 3: Physics Constants**
```
Get physics constants from NIS Protocol
```

---

## 🎯 **Available NIS Tools**

Once connected, you can use these tools:

1. **nis.run_pipeline** - Execute Laplace→KAN→PINN→LLM pipeline
2. **nis.robotics_control** - Control drones/manipulators (FK/IK/Trajectory)
3. **nis.job_status** - Get status of running jobs
4. **nis.get_artifact** - Fetch outputs (trajectories, reports)
5. **nis.cost_report** - Track GPU usage and costs
6. **nis.list_capabilities** - List all agents, pipelines, providers

---

## 🚨 **Troubleshooting**

### **MCP Server Not Found?**

1. **Check if server is running:**
   ```bash
   ps aux | grep standalone_server
   ```

2. **Restart the server:**
   ```bash
   cd /Users/diegofuego/Desktop/NIS_Protocol
   export NIS_PROJECT_ROOT=$(pwd)
   export PYTHONPATH=$NIS_PROJECT_ROOT
   python -m src.mcp.standalone_server
   ```

### **Tools Not Appearing?**

1. **Verify config file location:**
   ```bash
   # ChatGPT
   cat ~/Library/Application\ Support/ChatGPT/mcp_config.json
   
   # Claude
   cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

2. **Check logs:**
   ```bash
   # The MCP server logs to stdout
   tail -f logs/application/nis_protocol.log
   ```

### **Permission Denied?**

Make sure the config files have correct permissions:
```bash
chmod 644 ~/Library/Application\ Support/ChatGPT/mcp_config.json
chmod 644 ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

---

## 🎉 **What You Can Do Now**

With NIS Protocol connected to ChatGPT/Claude:

✅ **Robotics Control**
- Calculate drone motor thrust and moments
- Solve inverse kinematics for robotic arms
- Plan minimum-jerk trajectories

✅ **Physics Validation**
- Access fundamental constants
- Validate physical constraints
- Run PINN-based simulations

✅ **Research**
- Deep multi-source research
- ArXiv paper analysis
- Fact-checking and bias detection

✅ **Multi-Agent Coordination**
- Orchestrate 47 specialized agents
- Track agent metrics and performance
- Coordinate complex AI workflows

✅ **AGI Foundation**
- Local + cloud hybrid intelligence
- BitNet SEED model integration
- Consciousness-driven processing

---

## 📚 **More Information**

- **Full Setup Guide:** `docs/MCP_CHATGPT_CLAUDE_SETUP.md`
- **API Documentation:** `docs/README.md`
- **Robotics Guide:** `system/docs/ROBOTICS_INTEGRATION.md`
- **AWS Deployment:** `docs/AWS_DEPLOYMENT_GUIDE.md`

---

**🚀 Your Complete AI Operating System is now available to ChatGPT and Claude!**


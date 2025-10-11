# 🤖 Connect NIS Protocol to ChatGPT & Claude via MCP

**Make NIS Protocol tools available to ChatGPT and Claude (Anthropic) using the Model Context Protocol (MCP).**

---

## ✅ What You Have

Your NIS Protocol **already includes**:

1. **Full MCP Server** → `src/mcp/server.py` with 24+ tools
2. **Multi-Provider Support** → OpenAI (GPT-4o), Anthropic (Claude), Google (Gemini), DeepSeek, NVIDIA
3. **Physics-Validated AI** → Laplace → KAN → PINN → LLM pipeline
4. **Robotics Control** → FK/IK/Trajectory planning
5. **Real Agent Intelligence** → No mocks or placeholders

---

## 🚀 Quick Setup (5 minutes)

### **For ChatGPT (OpenAI Developer Mode)**

1. **Start the MCP Server:**
   ```bash
   # Local development
   cd /path/to/NIS_Protocol
   export NIS_PROJECT_ROOT=$(pwd)
   python -m src.mcp.standalone_server
   
   # AWS/Production (uses environment variables)
   # NIS_PROJECT_ROOT=/opt/nis-protocol python -m src.mcp.standalone_server
   ```

2. **Configure ChatGPT:**
   - Enable **Developer Mode** in ChatGPT settings
   - Add MCP Server:
     - Name: `NIS Protocol`
     - Command: `python`
     - Args: `-m src.mcp.standalone_server`
     - Working Directory: Set `NIS_PROJECT_ROOT` env var (default: `/opt/nis-protocol`)
   - Or use the config file: `mcp_chatgpt_config.json`

3. **Test it:**
   ```
   Ask ChatGPT: "List NIS Protocol capabilities"
   ```

---

### **For Claude (Anthropic)**

Claude Desktop supports MCP natively:

1. **Add to Claude Config** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
   ```json
   {
     "mcpServers": {
       "nis_protocol": {
         "command": "python",
         "args": ["-m", "src.mcp.standalone_server"],
         "env": {
           "PYTHONPATH": "${WORKSPACE_PATH}",
           "NIS_MCP_MODE": "standalone",
           "NIS_BACKEND_URL": "http://localhost"
         }
       }
     }
   }
   ```

2. **Restart Claude Desktop**

3. **Test it:**
   ```
   Ask Claude: "Use nis.list_capabilities to show what you can do"
   ```

---

### **For Cursor (Already Working!)**

You already have this in `.cursor/mcp.json` - it works now!

---

## 🛠️ Available Tools

Once connected, ChatGPT/Claude can use these tools:

| Tool | Description |
|------|-------------|
| `nis.run_pipeline` | Execute Laplace→KAN→PINN→LLM pipeline |
| `nis.job_status` | Get job status and metrics |
| `nis.get_artifact` | Fetch outputs (trajectories, physics validation, etc.) |
| `nis.cost_report` | GPU usage and cost tracking |
| `nis.list_capabilities` | List all agents, pipelines, providers |
| `nis.robotics_control` | Control drones/manipulators (FK/IK/Trajectory) |

---

## 📊 Example Usage

### In ChatGPT/Claude:

**1. List Capabilities:**
```
User: "What can NIS Protocol do?"
Assistant: *calls nis.list_capabilities*
```

**2. Control a Robot:**
```
User: "Calculate forward kinematics for a 6-DOF manipulator at [0, 0.785, 1.57, 0, 0.785, 0]"
Assistant: *calls nis.robotics_control with FK parameters*
```

**3. Run Physics Validation:**
```
User: "Validate this trajectory with PINN physics constraints"
Assistant: *calls nis.run_pipeline with physics_validation type*
```

**4. Get Cost Report:**
```
User: "How much compute have I used in the last 24 hours?"
Assistant: *calls nis.cost_report(window="24h")*
```

---

## 🔒 Security & Guardrails

The MCP server includes **production-ready safeguards**:

✅ **Budget Caps** → Max $100/job, configurable
✅ **Rate Limits** → 60 requests/minute, 5 concurrent jobs
✅ **Auth** → API key via `NIS_MCP_API_KEY` env var (optional in dev)
✅ **Auditing** → All requests logged
✅ **Timeouts** → Jobs auto-terminate after 1 hour

Set in `mcp_chatgpt_config.json`:
```json
{
  "guardrails": {
    "max_budget_per_job_usd": 100,
    "max_job_duration_seconds": 3600
  },
  "rate_limits": {
    "requests_per_minute": 60,
    "concurrent_jobs": 5
  }
}
```

---

## 🧪 Test the MCP Server Locally

Before connecting to ChatGPT/Claude, test it works:

```bash
# Set environment (local dev)
export NIS_PROJECT_ROOT=$(pwd)
export PYTHONPATH=$NIS_PROJECT_ROOT

# AWS/Production - already set in environment
# export NIS_PROJECT_ROOT=/opt/nis-protocol
# export NIS_BACKEND_URL=https://api.yourcompany.com

# Terminal 1: Start server
python -m src.mcp.standalone_server

# Terminal 2: Send test request
echo '{"jsonrpc":"2.0","id":"1","method":"nis.list_capabilities","params":{}}' | python -m src.mcp.standalone_server
```

You should see a JSON response with all capabilities listed.

---

## 🔗 Connect to Real Backend (Docker)

The MCP server can also connect to your running Docker backend:

```bash
# Set backend URL
export NIS_BACKEND_URL=http://localhost

# Start MCP server
python -m src.mcp.standalone_server
```

Now all `nis.*` tool calls will hit your real NIS Protocol backend!

---

## 📖 Resources

- **MCP Spec:** https://spec.modelcontextprotocol.io/
- **ChatGPT Developer Mode:** https://platform.openai.com/docs/guides/mcp
- **Claude Desktop MCP:** https://docs.anthropic.com/en/docs/build-with-claude/mcp
- **Cursor MCP:** https://docs.cursor.com/context/context-mcp

---

## 🎯 Next Steps

1. ✅ Start the MCP server
2. ✅ Configure ChatGPT or Claude
3. ✅ Test with `nis.list_capabilities`
4. ✅ Run a real robotics control command
5. ✅ Check cost tracking with `nis.cost_report`

---

## 🚨 Troubleshooting

**Server won't start?**
```bash
# Local: Set Python path to project root
export NIS_PROJECT_ROOT=$(pwd)
export PYTHONPATH=$NIS_PROJECT_ROOT

# AWS/Production: Use standard paths
# export NIS_PROJECT_ROOT=/opt/nis-protocol
# export PYTHONPATH=/opt/nis-protocol

# Check dependencies
pip install -r requirements.txt
```

**ChatGPT/Claude can't see tools?**
- Verify MCP server is running (`ps aux | grep standalone_server`)
- Check logs for errors
- Restart ChatGPT/Claude Desktop

**Tools timing out?**
- Ensure Docker backend is running (`docker ps | grep nis-backend`)
- Check backend health: `curl http://localhost/health`

---

## 💬 Support

Questions? Open an issue or ping @Organica-Ai-Solutions

**Built with honest engineering. No mocks. No hype. Just real AGI.**


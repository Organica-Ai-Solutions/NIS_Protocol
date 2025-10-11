# NIS Protocol – MCP Integration

This guide explains how to connect MCP servers to the NIS Protocol stack so IDE agents (Cursor, Gemini Code Assist, GitHub Copilot Agent mode, etc.) understand multiprovider routing, orchestrator state, BitNet offline bundles, and have browser automation capabilities.

## Available MCP Servers

1. **NIS Protocol Server** - Access to orchestrator status and provider configuration
2. **Puppeteer Browser Server** - Web browsing and automation capabilities

## Prerequisites
- Dart SDK ≥ 3.9 / Flutter ≥ 3.35
- NIS Protocol repository cloned locally
- MCP-capable client (Cursor, Gemini CLI, VS Code with Dart MCP support)

## Launch sequence
1. Start the NIS backend (`python start_safe.sh` or your preferred entry point).
2. Make the helper script executable:
   ```bash
   chmod +x scripts/utilities/run_dart_mcp_server.sh
   ```
3. Launch the MCP server from repo root:
   ```bash
   ./scripts/utilities/run_dart_mcp_server.sh
   ```
   You’ll see `Starting Dart MCP server (force roots fallback)…` when it’s live.
4. Open your IDE / MCP client; it will discover the running server automatically when configured (see below).

## Cursor configuration (`.cursor/mcp.json`)
```json
{
  "mcpServers": {
    "nis_protocol": {
      "command": "python",
      "args": ["scripts/utilities/orchestrator_status.py"],
      "workingDirectory": "/Users/diegofuego/Desktop/NIS_Protocol",
      "env": {
        "PYTHONPATH": "/Users/diegofuego/Desktop/NIS_Protocol",
        "NIS_PROVIDER_REGISTRY": "configs/provider_registry.yaml",
        "NIS_PROTOCOL_ROUTING": "configs/protocol_routing.json"
      }
    },
    "puppeteer": {
      "command": "npx",
      "args": ["-y", "@hisma/server-puppeteer"],
      "env": {
        "PUPPETEER_HEADLESS": "true"
      }
    }
  },
  "resources": {
    "providerRegistry": {
      "path": "configs/provider_registry.yaml",
      "description": "NIS multiprovider map"
    },
    "protocolRouting": {
      "path": "configs/protocol_routing.json",
      "description": "Third-party protocol routing"
    }
  },
  "tools": {
    "nisOrchestratorStatus": {
      "command": "python",
      "args": ["scripts/utilities/orchestrator_status.py"],
      "workingDirectory": "."
    }
  }
}
```
This exposes:
- **NIS Protocol Server**: Multiprovider definitions, protocol routing, and orchestrator status
- **Puppeteer Browser Server**: Web page navigation, scraping, screenshots, and browser automation

### Browser Capabilities Available:
- `puppeteer_navigate` - Navigate to URLs
- `puppeteer_screenshot` - Capture page screenshots
- `puppeteer_click` - Click elements on pages
- `puppeteer_fill` - Fill form inputs
- `puppeteer_evaluate` - Execute JavaScript on pages
- `puppeteer_content` - Get page content/HTML

## Gemini CLI / Firebase Studio
Add the following to `~/.gemini/settings.json` or `.gemini/settings.json` in your workspace:
```json
{
  "mcpServers": {
    "dart": {
      "command": "dart",
      "args": ["mcp-server", "--force-roots-fallback"],
      "env": {
        "NIS_PROVIDER_REGISTRY": "configs/provider_registry.yaml",
        "NIS_PROTOCOL_ROUTING": "configs/protocol_routing.json"
      }
    }
  }
}
```
Restart the environment and verify with `/mcp` in the agent console.

## VS Code (GitHub Copilot Agent mode)
Ensure you run the repo-level MCP helper script in a terminal, then set:
```json
"dart.mcpServer": true
```
inside user or workspace settings. The Dart extension will connect to the running server.

## Using the integration
- **Analyzer & tests:** Ask your MCP client to run `flutter analyze`, `flutter test`, or `dart format` while it has full project context.
- **Multiprovider insight:** The assistant can inspect `provider_registry.yaml`, adjust routing in `protocol_routing.json`, or query orchestrator status with the custom tool.
- **BitNet workflows:** Because the MCP server starts from repo root, it can reach scripts that package offline bundles or report BitNet status—keeping the Flutter app and backend automation in sync.
- **Browser automation:** Ask the assistant to navigate to websites, extract content, take screenshots, or interact with web pages. Example: "Navigate to the NIS Protocol documentation and extract the main features."

## Troubleshooting

### NIS Protocol Server
- Ensure `dart --version` succeeds inside your shell before launching the helper script.
- If the IDE cannot find roots, confirm `--force-roots-fallback` is passed (already set in `run_dart_mcp_server.sh`).
- Update environment variables if you relocate config files.

### Puppeteer Browser Server
- Ensure Node.js and npm are installed: `node --version && npm --version`
- The first run will download Chromium automatically (may take a few minutes)
- If browser automation fails, check that `PUPPETEER_HEADLESS=true` is set for headless mode
- For debugging, set `PUPPETEER_HEADLESS=false` to see browser actions visually

With this setup, Dart MCP-aware assistants work alongside NIS Protocol’s multiprovider architecture, enabling automation-first UX iterations directly from your IDE.

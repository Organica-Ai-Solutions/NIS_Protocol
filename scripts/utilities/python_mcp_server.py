#!/usr/bin/env python3
"""
Simple Python-based MCP server for NIS Protocol integration.
Exposes NIS configuration and orchestrator status to MCP clients.
"""

import json
import sys
import os
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from src.core.agent_orchestrator import NISAgentOrchestrator
    from src.utils.env_config import EnvConfig
    import yaml
except ImportError as e:
    print(f"Warning: Could not import NIS modules: {e}", file=sys.stderr)


class MCPHandler(BaseHTTPRequestHandler):
    """Handle MCP protocol requests"""
    
    def do_GET(self):
        """Handle GET requests for MCP resources"""
        if self.path == "/mcp/resources":
            self.send_mcp_resources()
        elif self.path == "/mcp/tools":
            self.send_mcp_tools()
        elif self.path == "/mcp/status":
            self.send_orchestrator_status()
        elif self.path.startswith("/mcp/resource/"):
            resource_name = self.path.split("/")[-1]
            self.send_resource_content(resource_name)
        else:
            self.send_error(404, "Not Found")
    
    def do_POST(self):
        """Handle POST requests for MCP tool execution"""
        if self.path == "/mcp/execute":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            try:
                request = json.loads(body)
                tool_name = request.get("tool")
                if tool_name == "nisOrchestratorStatus":
                    self.execute_orchestrator_status()
                else:
                    self.send_error(400, f"Unknown tool: {tool_name}")
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
        else:
            self.send_error(404, "Not Found")
    
    def send_mcp_resources(self):
        """Send list of available MCP resources"""
        resources = {
            "resources": [
                {
                    "uri": "file://" + str(ROOT / "configs" / "provider_registry.yaml"),
                    "name": "providerRegistry",
                    "description": "NIS Protocol multiprovider configuration",
                    "mimeType": "application/x-yaml"
                },
                {
                    "uri": "file://" + str(ROOT / "configs" / "protocol_routing.json"),
                    "name": "protocolRouting",
                    "description": "Routing and third-party protocol map",
                    "mimeType": "application/json"
                }
            ]
        }
        self.send_json_response(resources)
    
    def send_mcp_tools(self):
        """Send list of available MCP tools"""
        tools = {
            "tools": [
                {
                    "name": "nisOrchestratorStatus",
                    "description": "Get NIS Protocol orchestrator status, agent definitions, and provider configuration",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ]
        }
        self.send_json_response(tools)
    
    def send_resource_content(self, resource_name):
        """Send content of a specific resource"""
        resource_map = {
            "providerRegistry": ROOT / "configs" / "provider_registry.yaml",
            "protocolRouting": ROOT / "configs" / "protocol_routing.json"
        }
        
        if resource_name not in resource_map:
            self.send_error(404, f"Resource not found: {resource_name}")
            return
        
        resource_path = resource_map[resource_name]
        if not resource_path.exists():
            self.send_error(404, f"Resource file not found: {resource_path}")
            return
        
        with open(resource_path, 'r') as f:
            content = f.read()
        
        self.send_response(200)
        if resource_name == "providerRegistry":
            self.send_header('Content-Type', 'application/x-yaml')
        else:
            self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(content.encode())
    
    def send_orchestrator_status(self):
        """Send orchestrator status"""
        try:
            status = self.get_orchestrator_status()
            self.send_json_response(status)
        except Exception as e:
            self.send_error(500, f"Error getting status: {str(e)}")
    
    def execute_orchestrator_status(self):
        """Execute orchestrator status tool"""
        try:
            status = self.get_orchestrator_status()
            result = {
                "tool": "nisOrchestratorStatus",
                "result": status
            }
            self.send_json_response(result)
        except Exception as e:
            self.send_error(500, f"Error executing tool: {str(e)}")
    
    def get_orchestrator_status(self):
        """Get orchestrator status data"""
        try:
            orchestrator = NISAgentOrchestrator()
            orchestrator.load_agents()
            
            agents_info = {
                agent_id: {
                    "name": definition.name,
                    "type": definition.agent_type.value,
                    "priority": definition.priority,
                    "description": definition.description,
                    "status": definition.status.value,
                    "activation_trigger": definition.activation_trigger.value
                }
                for agent_id, definition in orchestrator.agents.items()
            }
        except Exception as e:
            agents_info = {"error": f"Could not load agents: {str(e)}"}
        
        try:
            registry_path = ROOT / "configs" / "provider_registry.yaml"
            if registry_path.exists():
                with open(registry_path) as f:
                    providers = yaml.safe_load(f) or {}
            else:
                providers = {"error": "provider_registry.yaml not found"}
        except Exception as e:
            providers = {"error": f"Could not load providers: {str(e)}"}
        
        try:
            env = EnvConfig()
            llm_config = env.get_llm_config()
            config_info = {
                "default_provider": llm_config.get("agent_llm_config", {}).get("default_provider"),
                "fallback_to_mock": llm_config.get("agent_llm_config", {}).get("fallback_to_mock"),
                "providers": list(llm_config.get("providers", {}).keys())
            }
        except Exception as e:
            config_info = {"error": f"Could not load config: {str(e)}"}
        
        return {
            "orchestrator": {
                "agents": agents_info,
                "total_agents": len(agents_info)
            },
            "providers": providers,
            "llm_config": config_info,
            "timestamp": str(Path(__file__).stat().st_mtime)
        }
    
    def send_json_response(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def log_message(self, format, *args):
        """Override to reduce logging noise"""
        pass  # Silent mode


def run_server(port=3333):
    """Run the MCP server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, MCPHandler)
    print(f"🚀 NIS Protocol MCP Server running on http://localhost:{port}")
    print(f"📂 Workspace root: {ROOT}")
    print(f"🔗 Resources:")
    print(f"   - Provider Registry: {ROOT / 'configs' / 'provider_registry.yaml'}")
    print(f"   - Protocol Routing: {ROOT / 'configs' / 'protocol_routing.json'}")
    print(f"🛠️  Tools:")
    print(f"   - nisOrchestratorStatus: Get agent orchestration status")
    print(f"\n✅ Server ready. Press Ctrl+C to stop.\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down MCP server...")
        httpd.shutdown()


if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", "3333"))
    run_server(port)

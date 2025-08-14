"""
Official mcp-ui SDK Integration for NIS Protocol

Integrates the official @mcp-ui/server and @mcp-ui/client packages
with our Deep Agents + MCP server implementation.
"""

import json
import time
from typing import Dict, Any, Optional, Union


class MCPUIServerIntegration:
    """
    Integration with official @mcp-ui/server package.
    
    Converts our UI resources to official mcp-ui format and ensures
    100% compatibility with @mcp-ui/client components.
    """
    
    def __init__(self):
        self.resource_counter = 0
        
    def create_ui_resource(self, uri: str, content: Dict[str, Any], 
                          encoding: str = "text") -> Dict[str, Any]:
        """
        Create UI resource in official mcp-ui format.
        
        Args:
            uri: Unique identifier (e.g., 'ui://component/123')
            content: Content configuration
            encoding: 'text' or 'blob'
            
        Returns:
            Official UIResource format
        """
        # Convert our content to official mcp-ui format
        if content.get("type") == "rawHtml":
            return {
                "type": "resource",
                "resource": {
                    "uri": uri,
                    "mimeType": "text/html",
                    "text": content["htmlString"] if encoding == "text" else None,
                    "blob": self._encode_base64(content["htmlString"]) if encoding == "blob" else None
                }
            }
            
        elif content.get("type") == "externalUrl":
            return {
                "type": "resource", 
                "resource": {
                    "uri": uri,
                    "mimeType": "text/uri-list",
                    "text": content["iframeUrl"] if encoding == "text" else None,
                    "blob": self._encode_base64(content["iframeUrl"]) if encoding == "blob" else None
                }
            }
            
        elif content.get("type") == "remoteDom":
            framework = content.get("framework", "react")
            return {
                "type": "resource",
                "resource": {
                    "uri": uri,
                    "mimeType": f"application/vnd.mcp-ui.remote-dom+javascript; framework={framework}",
                    "text": content["script"] if encoding == "text" else None,
                    "blob": self._encode_base64(content["script"]) if encoding == "blob" else None
                }
            }
            
        else:
            # Fallback to HTML
            html_content = self._convert_to_html(content)
            return {
                "type": "resource",
                "resource": {
                    "uri": uri,
                    "mimeType": "text/html", 
                    "text": html_content if encoding == "text" else None,
                    "blob": self._encode_base64(html_content) if encoding == "blob" else None
                }
            }
            
    def _encode_base64(self, content: str) -> str:
        """Encode content as base64."""
        import base64
        return base64.b64encode(content.encode('utf-8')).decode('utf-8')
        
    def _convert_to_html(self, content: Any) -> str:
        """Convert any content to HTML format."""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict) or isinstance(content, list):
            return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Data Viewer</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; }}
        .json-viewer {{ background: #f6f8fa; padding: 16px; border-radius: 6px; font-family: Monaco, 'Courier New', monospace; white-space: pre-wrap; }}
    </style>
</head>
<body>
    <div class="json-viewer">{json.dumps(content, indent=2)}</div>
</body>
</html>"""
        else:
            return f"<p>{str(content)}</p>"


class MCPUIClientHandler:
    """
    Handler for official @mcp-ui/client events and actions.
    
    Processes UI actions according to official mcp-ui specification.
    """
    
    def __init__(self, mcp_server):
        self.mcp_server = mcp_server
        
    async def handle_ui_action(self, action: Dict[str, Any], message_id: str = None) -> Dict[str, Any]:
        """
        Handle UI action from @mcp-ui/client.
        
        Args:
            action: UI action in official format
            message_id: Optional message ID for async responses
            
        Returns:
            Response for the action
        """
        action_type = action.get("type")
        payload = action.get("payload", {})
        
        if action_type == "tool":
            # Tool call from UI
            tool_name = payload.get("toolName")
            params = payload.get("params", {})
            
            # Execute via our MCP server
            response = await self.mcp_server.handle_request({
                "type": "tool",
                "tool_name": tool_name,
                "parameters": params,
                "request_id": f"ui_action_{message_id or int(time.time() * 1000)}"
            })
            
            # Return in expected format
            return {
                "success": response.get("success", False),
                "result": response.get("data"),
                "ui_resource": response.get("ui_resource"),
                "message_id": message_id
            }
            
        elif action_type == "intent":
            # Generic intent from UI
            intent = payload.get("intent")
            params = payload.get("params", {})
            
            # Handle via our intent validator
            response = await self.mcp_server.intent_validator.handle_intent(
                "intent", {"intent": intent, "params": params}, message_id
            )
            
            return {
                "success": True,
                "result": response,
                "message_id": message_id
            }
            
        elif action_type == "prompt":
            # Prompt from UI
            prompt = payload.get("prompt")
            
            # Process via agent
            response = await self.mcp_server.agent.process_request({
                "action": "process_prompt",
                "data": {"prompt": prompt, "source": "mcp_ui"},
                "metadata": {"message_id": message_id}
            })
            
            return {
                "success": True,
                "result": response,
                "message_id": message_id
            }
            
        elif action_type == "notify":
            # Notification from UI
            message = payload.get("message")
            
            # Log the notification
            import logging
            logging.info(f"UI Notification: {message}")
            
            return {
                "success": True,
                "result": {"acknowledged": True, "message": message},
                "message_id": message_id
            }
            
        elif action_type == "link":
            # Link click from UI
            url = payload.get("url")
            
            # Validate and log
            is_safe = self.mcp_server.intent_validator.validate_intent("link", payload)
            
            return {
                "success": is_safe[0],
                "result": {"url": url, "safe": is_safe[0]},
                "message_id": message_id
            }
            
        else:
            return {
                "success": False,
                "error": f"Unknown action type: {action_type}",
                "message_id": message_id
            }


class OfficialMCPUIAdapter:
    """
    Complete adapter for official mcp-ui SDK integration.
    
    Bridges our Deep Agents + MCP server with official mcp-ui packages.
    """
    
    def __init__(self, mcp_server):
        self.mcp_server = mcp_server
        self.server_integration = MCPUIServerIntegration()
        self.client_handler = MCPUIClientHandler(mcp_server)
        
    def convert_ui_resource_to_official(self, ui_resource: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert our UI resource to official mcp-ui format.
        
        Args:
            ui_resource: Our UI resource format
            
        Returns:
            Official mcp-ui UIResource format
        """
        if ui_resource.get("type") == "resource":
            # Already in correct format
            return ui_resource
            
        # Convert from our internal format
        resource = ui_resource.get("resource", {})
        uri = resource.get("uri", f"ui://converted/{int(time.time() * 1000)}")
        html_content = resource.get("text", "")
        
        return self.server_integration.create_ui_resource(
            uri=uri,
            content={"type": "rawHtml", "htmlString": html_content},
            encoding="text"
        )
        
    async def create_remote_dom_resource(self, uri: str, script: str, 
                                       framework: str = "react") -> Dict[str, Any]:
        """
        Create a Remote DOM resource using official format.
        
        Args:
            uri: Resource URI
            script: JavaScript/React script
            framework: 'react' or 'webcomponents'
            
        Returns:
            Official Remote DOM UIResource
        """
        return self.server_integration.create_ui_resource(
            uri=uri,
            content={
                "type": "remoteDom",
                "script": script,
                "framework": framework
            },
            encoding="text"
        )
        
    async def create_interactive_button_example(self) -> Dict[str, Any]:
        """Create an example interactive button using Remote DOM."""
        script = """
        const button = document.createElement('ui-button');
        button.setAttribute('label', 'Search Datasets');
        button.setAttribute('variant', 'primary');
        button.addEventListener('press', () => {
            window.parent.postMessage({
                type: 'tool',
                payload: {
                    toolName: 'dataset.search',
                    params: { query: 'weather data', limit: 10 }
                }
            }, '*');
        });
        root.appendChild(button);
        """
        
        return await self.create_remote_dom_resource(
            uri="ui://interactive/search-button",
            script=script,
            framework="react"
        )
        
    async def create_data_viewer_remote_dom(self, data: Any) -> Dict[str, Any]:
        """Create a data viewer using Remote DOM."""
        script = f"""
        const container = document.createElement('div');
        container.style.padding = '20px';
        container.style.fontFamily = 'system-ui, sans-serif';
        
        const title = document.createElement('h3');
        title.textContent = 'Interactive Data Viewer';
        container.appendChild(title);
        
        const dataDisplay = document.createElement('pre');
        dataDisplay.style.background = '#f6f8fa';
        dataDisplay.style.padding = '16px';
        dataDisplay.style.borderRadius = '8px';
        dataDisplay.style.overflow = 'auto';
        dataDisplay.textContent = {json.dumps(json.dumps(data, indent=2))};
        container.appendChild(dataDisplay);
        
        const actionButton = document.createElement('ui-button');
        actionButton.setAttribute('label', 'Refresh Data');
        actionButton.addEventListener('press', () => {{
            window.parent.postMessage({{
                type: 'intent',
                payload: {{
                    intent: 'refresh',
                    params: {{ component: 'data-viewer' }}
                }}
            }}, '*');
        }});
        container.appendChild(actionButton);
        
        root.appendChild(container);
        """
        
        return await self.create_remote_dom_resource(
            uri=f"ui://data-viewer/{int(time.time() * 1000)}",
            script=script,
            framework="react"
        )
        
    def get_supported_content_types(self) -> list:
        """Get supported content types for @mcp-ui/client."""
        return ["rawHtml", "externalUrl", "remoteDom"]
        
    def get_supported_frameworks(self) -> list:
        """Get supported frameworks for Remote DOM."""
        return ["react", "webcomponents"]
        
    def get_example_ui_resource_renderer_props(self) -> Dict[str, Any]:
        """Get example props for UIResourceRenderer component."""
        return {
            "supportedContentTypes": self.get_supported_content_types(),
            "autoResizeIframe": True,
            "style": {"width": "100%", "minHeight": "400px"},
            "iframeProps": {"sandbox": "allow-scripts allow-same-origin"},
            "remoteDomProps": {
                "library": "basicComponentLibrary"
            }
        }


# Integration functions for easy setup
def setup_official_mcp_ui_integration(mcp_server):
    """Setup official mcp-ui integration with existing MCP server."""
    return OfficialMCPUIAdapter(mcp_server)


def create_official_ui_resource(uri: str, content_type: str, content: str, 
                               framework: str = "react") -> Dict[str, Any]:
    """
    Create UI resource in official mcp-ui format.
    
    Args:
        uri: Resource URI (e.g., 'ui://component/123')
        content_type: 'html', 'url', or 'remote-dom'
        content: HTML string, URL, or Remote DOM script
        framework: For Remote DOM resources
        
    Returns:
        Official UIResource format
    """
    integration = MCPUIServerIntegration()
    
    if content_type == "html":
        return integration.create_ui_resource(
            uri=uri,
            content={"type": "rawHtml", "htmlString": content},
            encoding="text"
        )
    elif content_type == "url":
        return integration.create_ui_resource(
            uri=uri,
            content={"type": "externalUrl", "iframeUrl": content},
            encoding="text"
        )
    elif content_type == "remote-dom":
        return integration.create_ui_resource(
            uri=uri,
            content={"type": "remoteDom", "script": content, "framework": framework},
            encoding="text"
        )
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


# Example usage for frontend integration
def get_frontend_integration_example():
    """Get example code for frontend integration."""
    return {
        "react_example": """
import React from 'react';
import { UIResourceRenderer } from '@mcp-ui/client';

function MCPUIDemo({ mcpResource }) {
  const handleUIAction = (action) => {
    console.log('UI Action:', action);
    
    // Send to backend MCP server
    fetch('/api/mcp/ui-action', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(action)
    });
  };

  if (mcpResource?.type === 'resource') {
    return (
      <UIResourceRenderer
        resource={mcpResource.resource}
        onUIAction={handleUIAction}
        supportedContentTypes={['rawHtml', 'externalUrl', 'remoteDom']}
        autoResizeIframe={true}
        style={{ width: '100%', minHeight: '400px' }}
      />
    );
  }
  
  return <div>No UI resource available</div>;
}
        """,
        
        "web_component_example": """
<ui-resource-renderer
  resource='{"uri":"ui://demo/123","mimeType":"text/html","text":"<h2>Hello from mcp-ui!</h2>"}'
  supported-content-types='["rawHtml", "externalUrl", "remoteDom"]'
  auto-resize-iframe="true">
</ui-resource-renderer>

<script>
const renderer = document.querySelector('ui-resource-renderer');
renderer.addEventListener('onUIAction', (event) => {
  console.log('UI Action:', event.detail);
  
  // Send to backend
  fetch('/api/mcp/ui-action', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(event.detail)
  });
});
</script>
        """
    }

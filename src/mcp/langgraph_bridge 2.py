"""
LangGraph Bridge for MCP Integration

Bridges the NIS Protocol MCP server with LangGraph Agent Chat UI.
Provides compatibility layer and adapter patterns.
"""

import json
import time
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass

from .server import MCPServer
from .integration import MCPIntegration


@dataclass
class LangGraphMessage:
    """Represents a LangGraph message format."""
    id: str
    type: str
    content: Any
    timestamp: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LangGraphMCPBridge:
    """
    Bridge between NIS Protocol MCP Server and LangGraph Agent Chat UI.
    
    Adapts MCP tool calls and UI resources to LangGraph message format
    for seamless integration with the Agent Chat UI.
    """
    
    def __init__(self, mcp_integration: MCPIntegration):
        self.mcp_integration = mcp_integration
        self.active_sessions = {}
        self.message_history = {}
        
    async def handle_chat_message(self, message: str, session_id: str = None, 
                                 user_id: str = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle a chat message and yield streaming responses in LangGraph format.
        
        Args:
            message: User message
            session_id: Session identifier
            user_id: User identifier
            
        Yields:
            LangGraph-compatible message chunks
        """
        session_id = session_id or f"session_{int(time.time() * 1000)}"
        
        # Parse message for tool calls or direct queries
        tool_request = await self._parse_message_for_tools(message)
        
        if tool_request:
            # Handle as tool execution
            async for chunk in self._handle_tool_execution(tool_request, session_id, user_id):
                yield chunk
        else:
            # Handle as general agent query
            async for chunk in self._handle_agent_query(message, session_id, user_id):
                yield chunk
                
    async def _parse_message_for_tools(self, message: str) -> Optional[Dict[str, Any]]:
        """Parse message to extract tool calls."""
        # Simple pattern matching for tool calls
        # In production, use more sophisticated NLP parsing
        
        tool_patterns = {
            r"search (?:for )?datasets? (?:about |with |containing )?(.+)": {
                "tool_name": "dataset.search",
                "param_mapping": {"query": 1}
            },
            r"run (?:a )?pipeline (?:called |named )?(.+)": {
                "tool_name": "pipeline.run", 
                "param_mapping": {"pipeline_id": 1}
            },
            r"create (?:a )?research plan (?:for |about )?(.+)": {
                "tool_name": "research.plan",
                "param_mapping": {"goal": 1}
            },
            r"review (?:the )?code (?:in |at )?(.+)": {
                "tool_name": "code.review",
                "param_mapping": {"file_path": 1}
            },
            r"show (?:me )?(?:the )?audit (?:trail|log) (?:for )?(.*)": {
                "tool_name": "audit.view",
                "param_mapping": {"component": 1}
            }
        }
        
        import re
        for pattern, config in tool_patterns.items():
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                parameters = {}
                for param, group_idx in config["param_mapping"].items():
                    if group_idx <= len(match.groups()):
                        value = match.group(group_idx).strip()
                        if value:
                            parameters[param] = value
                            
                return {
                    "tool_name": config["tool_name"],
                    "parameters": parameters,
                    "original_message": message
                }
                
        return None
        
    async def _handle_tool_execution(self, tool_request: Dict[str, Any], 
                                   session_id: str, user_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle tool execution and stream results."""
        # Yield thinking message
        yield self._create_message(
            "assistant",
            f"I'll {tool_request['tool_name'].replace('.', ' ')} for you...",
            session_id,
            {"type": "thinking"}
        )
        
        # Execute the tool
        mcp_request = {
            "type": "tool",
            "tool_name": tool_request["tool_name"],
            "parameters": tool_request["parameters"],
            "session_id": session_id,
            "user_id": user_id
        }
        
        try:
            response = await self.mcp_integration.handle_mcp_request(mcp_request)
            
            if response["success"]:
                # Yield data summary
                data = response.get("data", {})
                summary = self._create_data_summary(tool_request["tool_name"], data)
                
                yield self._create_message(
                    "assistant",
                    summary,
                    session_id,
                    {"type": "tool_result"}
                )
                
                # Yield UI resource if available
                if response.get("ui_resource"):
                    yield self._create_artifact_message(
                        response["ui_resource"],
                        tool_request["tool_name"],
                        session_id
                    )
                    
            else:
                yield self._create_message(
                    "assistant",
                    f"Sorry, I encountered an error: {response.get('error', 'Unknown error')}",
                    session_id,
                    {"type": "error"}
                )
                
        except Exception as e:
            yield self._create_message(
                "assistant",
                f"I encountered an unexpected error: {str(e)}",
                session_id,
                {"type": "error"}
            )
            
    async def _handle_agent_query(self, message: str, session_id: str, 
                                user_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle general agent queries."""
        # Use the underlying agent for general conversation
        yield self._create_message(
            "assistant", 
            "Let me think about that...",
            session_id,
            {"type": "thinking"}
        )
        
        # Process with NIS agent
        if self.mcp_integration.agent:
            try:
                agent_response = await self.mcp_integration.agent.process_request({
                    "action": "chat",
                    "data": {"message": message, "session_id": session_id},
                    "metadata": {"user_id": user_id}
                })
                
                content = agent_response.get("content", "I'm not sure how to help with that.")
                yield self._create_message("assistant", content, session_id)
                
            except Exception as e:
                yield self._create_message(
                    "assistant",
                    "I'm having trouble processing your request. Could you try rephrasing?",
                    session_id,
                    {"type": "error"}
                )
        else:
            yield self._create_message(
                "assistant",
                "I can help you with datasets, pipelines, research, code review, and auditing. What would you like to do?",
                session_id
            )
            
    def _create_message(self, role: str, content: str, session_id: str, 
                       metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a LangGraph-compatible message."""
        message_id = f"msg_{int(time.time() * 1000)}_{hash(content) % 10000}"
        
        return {
            "id": message_id,
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "session_id": session_id,
            "metadata": metadata or {}
        }
        
    def _create_artifact_message(self, ui_resource: Dict[str, Any], 
                                tool_name: str, session_id: str) -> Dict[str, Any]:
        """Create an artifact message for UI resources."""
        artifact_id = f"artifact_{int(time.time() * 1000)}"
        
        # Convert mcp-ui resource to LangGraph artifact format
        artifact_content = {
            "type": "artifact",
            "artifact": {
                "id": artifact_id,
                "type": "ui_component",
                "title": self._get_artifact_title(tool_name),
                "content": ui_resource["resource"]["text"],
                "mimeType": ui_resource["resource"]["mimeType"],
                "metadata": {
                    "tool_name": tool_name,
                    "uri": ui_resource["resource"]["uri"]
                }
            }
        }
        
        return {
            "id": f"msg_{artifact_id}",
            "role": "assistant", 
            "content": f"Here's an interactive view of the {tool_name.replace('.', ' ')} results:",
            "timestamp": time.time(),
            "session_id": session_id,
            "metadata": artifact_content
        }
        
    def _get_artifact_title(self, tool_name: str) -> str:
        """Get appropriate title for artifact based on tool."""
        titles = {
            "dataset.search": "Dataset Search Results",
            "dataset.preview": "Dataset Preview",
            "pipeline.run": "Pipeline Execution Monitor",
            "pipeline.status": "Pipeline Status",
            "research.plan": "Research Plan",
            "research.search": "Research Results", 
            "audit.view": "Audit Timeline",
            "audit.analyze": "Performance Analysis",
            "code.review": "Code Review",
            "code.edit": "Code Changes"
        }
        return titles.get(tool_name, "Interactive Results")
        
    def _create_data_summary(self, tool_name: str, data: Dict[str, Any]) -> str:
        """Create a human-readable summary of tool results."""
        if tool_name == "dataset.search":
            items = data.get("items", [])
            total = data.get("total", 0)
            return f"Found {len(items)} datasets (total: {total} available). Click below to explore the interactive results."
            
        elif tool_name == "dataset.preview":
            stats = data.get("statistics", {})
            rows = stats.get("total_rows", "unknown")
            cols = stats.get("total_columns", "unknown")
            return f"Dataset preview ready: {rows} rows, {cols} columns. View the interactive preview below."
            
        elif tool_name == "pipeline.run":
            run_id = data.get("run_id", "unknown")
            status = data.get("status", "unknown")
            return f"Pipeline started (ID: {run_id}, Status: {status}). Monitor progress below."
            
        elif tool_name == "pipeline.status":
            progress = data.get("progress", 0)
            status = data.get("status", "unknown")
            return f"Pipeline status: {status} ({progress}% complete). View detailed status below."
            
        elif tool_name == "research.plan":
            objectives = data.get("objectives", [])
            return f"Research plan created with {len(objectives)} objectives. View the interactive plan below."
            
        elif tool_name == "research.search":
            results = data.get("results", [])
            total = data.get("total_found", 0)
            return f"Found {len(results)} research papers (total: {total} available). Explore results below."
            
        elif tool_name == "audit.view":
            events = data.get("timeline", [])
            return f"Audit trail loaded with {len(events)} events. View timeline below."
            
        elif tool_name == "code.review":
            score = data.get("overall_score", "N/A")
            issues = len(data.get("issues", []))
            return f"Code review complete (Score: {score}, Issues: {issues}). View detailed review below."
            
        else:
            return f"Operation completed successfully. View results below."
            
    async def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get message history for a session."""
        return self.message_history.get(session_id, [])
        
    async def clear_session(self, session_id: str):
        """Clear session data."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        if session_id in self.message_history:
            del self.message_history[session_id]
            
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools for the chat interface."""
        if not self.mcp_integration.is_initialized:
            return []
            
        tools = self.mcp_integration.get_tool_registry()
        
        # Convert to chat-friendly format
        chat_tools = []
        for tool_name, tool_def in tools.items():
            chat_tools.append({
                "name": tool_name,
                "description": tool_def["description"],
                "category": tool_def["skill"],
                "examples": self._get_tool_examples(tool_name)
            })
            
        return chat_tools
        
    def _get_tool_examples(self, tool_name: str) -> List[str]:
        """Get example phrases for invoking a tool."""
        examples = {
            "dataset.search": [
                "search for weather datasets",
                "find datasets about climate data",
                "search datasets containing temperature"
            ],
            "pipeline.run": [
                "run the data processing pipeline",
                "execute pipeline weather_analysis",
                "start the ETL pipeline"
            ],
            "research.plan": [
                "create a research plan for AI safety",
                "plan research about climate change",
                "generate research plan for machine learning"
            ],
            "code.review": [
                "review the code in main.py",
                "analyze code quality in src/agents/",
                "check the authentication module"
            ],
            "audit.view": [
                "show me the audit trail",
                "view system logs for today",
                "display audit timeline"
            ]
        }
        return examples.get(tool_name, [f"Use {tool_name}"])


def create_langgraph_adapter(mcp_integration: MCPIntegration) -> LangGraphMCPBridge:
    """Create a LangGraph adapter for the MCP integration."""
    return LangGraphMCPBridge(mcp_integration)


# FastAPI endpoint example for LangGraph Agent Chat UI
async def create_langgraph_endpoint(mcp_integration: MCPIntegration):
    """Create FastAPI endpoints compatible with LangGraph Agent Chat UI."""
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import StreamingResponse
    import json
    
    app = FastAPI()
    bridge = create_langgraph_adapter(mcp_integration)
    
    @app.post("/invoke")
    async def invoke_agent(request: dict):
        """Main invoke endpoint for agent chat."""
        message = request.get("input", {}).get("messages", [])[-1].get("content", "")
        session_id = request.get("config", {}).get("configurable", {}).get("thread_id")
        
        responses = []
        async for chunk in bridge.handle_chat_message(message, session_id):
            responses.append(chunk)
            
        # Return final response in LangGraph format
        return {
            "output": {
                "messages": responses
            },
            "metadata": {
                "run_id": f"run_{int(time.time() * 1000)}",
                "thread_id": session_id
            }
        }
        
    @app.post("/stream")
    async def stream_agent(request: dict):
        """Streaming endpoint for real-time responses."""
        message = request.get("input", {}).get("messages", [])[-1].get("content", "")
        session_id = request.get("config", {}).get("configurable", {}).get("thread_id")
        
        async def generate():
            async for chunk in bridge.handle_chat_message(message, session_id):
                yield f"data: {json.dumps(chunk)}\n\n"
                
        return StreamingResponse(generate(), media_type="text/plain")
        
    @app.get("/tools")
    async def get_tools():
        """Get available tools for the chat interface."""
        return bridge.get_available_tools()
        
    return app

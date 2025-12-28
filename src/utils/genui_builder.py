"""
GenUI Response Builder for NIS Protocol Backend

Converts standard backend responses into GenUI-formatted responses.
Provides helper functions to wrap responses with GenUI specifications.
"""

import time
from typing import Any, Dict, Optional, List


def build_chat_response_genui(
    response_text: str,
    provider: str = "unknown",
    model: str = "unknown",
    tokens_used: int = 0,
    status: str = "success",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build GenUI response for chat endpoints.
    
    Wraps a chat response with a GenUI card specification for visual display.
    """
    return {
        "status": status,
        "data": {
            "response": response_text,
            "provider": provider,
            "model": model,
            "tokens_used": tokens_used,
        },
        "genui": {
            "type": "chat-response-card",
            "props": {
                "title": "Response",
                "description": response_text,
                "status": status,
                "metrics": {
                    "provider": provider,
                    "model": model,
                    "tokens": tokens_used,
                }
            },
            "actions": [
                {
                    "id": "copy_response",
                    "label": "Copy",
                    "endpoint": "/action/copy",
                    "method": "POST",
                    "payload": {"text": response_text}
                },
                {
                    "id": "regenerate",
                    "label": "Regenerate",
                    "endpoint": "/chat/simple",
                    "method": "POST",
                }
            ]
        },
        "metadata": {
            "timestamp": time.time(),
            "processing_time": metadata.get("processing_time", 0) if metadata else 0,
            "agent": "chat_service"
        }
    }


def build_consciousness_status_genui(
    status: str,
    phases_active: int,
    capabilities: List[str],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build GenUI response for consciousness status endpoints.
    
    Displays system consciousness status with phase indicators and capabilities.
    """
    return {
        "status": "success",
        "data": {
            "consciousness_status": status,
            "phases_active": phases_active,
            "capabilities": capabilities,
        },
        "genui": {
            "type": "consciousness-status-dashboard",
            "props": {
                "status": status,
                "phases_active": phases_active,
                "capabilities": capabilities,
            },
            "actions": [
                {
                    "id": "evolve",
                    "label": "Trigger Evolution",
                    "endpoint": "/v4/consciousness/evolve",
                    "method": "POST",
                    "payload": {"reason": "user_triggered"}
                },
                {
                    "id": "genesis",
                    "label": "Create Agent",
                    "endpoint": "/v4/consciousness/genesis",
                    "method": "POST",
                    "payload": {"capability": "specialized"}
                },
                {
                    "id": "refresh",
                    "label": "Refresh Status",
                    "endpoint": "/v4/consciousness/status",
                    "method": "GET",
                }
            ]
        },
        "metadata": {
            "timestamp": time.time(),
            "agent": "consciousness_service"
        }
    }


def build_code_executor_genui(
    status: str,
    output: str = "",
    error: Optional[str] = None,
    execution_time_ms: int = 0,
) -> Dict[str, Any]:
    """
    Build GenUI response for code execution results.
    
    Displays code output or errors with execution metrics.
    """
    return {
        "status": status,
        "data": {
            "output": output,
            "error": error,
            "execution_time_ms": execution_time_ms,
        },
        "genui": {
            "type": "code-executor-output",
            "props": {
                "status": status,
                "output": output,
                "error": error,
                "execution_time_ms": execution_time_ms,
            },
            "actions": [
                {
                    "id": "copy_output",
                    "label": "Copy Output",
                    "endpoint": "/action/copy",
                    "method": "POST",
                    "payload": {"text": output or error or ""}
                },
                {
                    "id": "save_result",
                    "label": "Save Result",
                    "endpoint": "/action/save",
                    "method": "POST",
                }
            ]
        },
        "metadata": {
            "timestamp": time.time(),
            "execution_time_ms": execution_time_ms,
            "agent": "code_executor"
        }
    }


def build_web_search_genui(
    query: str,
    results: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Build GenUI response for web search results.
    
    Displays search results with titles, snippets, and links.
    """
    return {
        "status": "success",
        "data": {
            "query": query,
            "results": results,
            "result_count": len(results),
        },
        "genui": {
            "type": "web-search-results",
            "props": {
                "query": query,
                "results": results,
            },
            "actions": [
                {
                    "id": "new_search",
                    "label": "New Search",
                    "endpoint": "/protocol/mcp/execute",
                    "method": "POST",
                    "payload": {"tool_name": "web_search"}
                }
            ]
        },
        "metadata": {
            "timestamp": time.time(),
            "result_count": len(results),
            "agent": "web_search_tool"
        }
    }


def build_generic_mcp_tool_genui(
    tool_name: str,
    status: str,
    output: Any,
    execution_time_ms: int = 0,
) -> Dict[str, Any]:
    """
    Build generic GenUI response for MCP tool execution.
    
    Fallback for tools without specific GenUI handlers.
    """
    return {
        "status": status,
        "data": {
            "tool": tool_name,
            "output": output,
            "execution_time_ms": execution_time_ms,
        },
        "genui": {
            "type": "card",
            "props": {
                "title": f"{tool_name.replace('_', ' ').title()} Result",
                "description": str(output),
                "status": status,
                "metrics": {
                    "tool": tool_name,
                    "time_ms": execution_time_ms,
                }
            },
            "actions": []
        },
        "metadata": {
            "timestamp": time.time(),
            "tool": tool_name,
            "execution_time_ms": execution_time_ms,
            "agent": "mcp_executor"
        }
    }


def build_error_response_genui(
    error_message: str,
    endpoint: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build GenUI response for errors.
    
    Displays error information with retry actions.
    """
    return {
        "status": "error",
        "data": {
            "error": error_message,
            "endpoint": endpoint,
            "details": details or {},
        },
        "genui": {
            "type": "card",
            "props": {
                "title": "Error",
                "description": error_message,
                "status": "error",
            },
            "actions": [
                {
                    "id": "retry",
                    "label": "Retry",
                    "endpoint": endpoint,
                    "method": "POST",
                }
            ]
        },
        "metadata": {
            "timestamp": time.time(),
            "error": error_message,
            "agent": "error_handler"
        }
    }

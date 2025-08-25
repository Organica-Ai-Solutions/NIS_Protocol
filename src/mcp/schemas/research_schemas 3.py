"""
Research Tool Schemas for MCP Integration

JSON schemas for research-related MCP tools.
"""

from typing import Dict, Any


class ResearchSchemas:
    """Schema definitions for research tools."""
    
    def get_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get all research tool schemas."""
        return {
            "plan": {
                "description": "Create a research plan for a given goal",
                "input": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "Research goal or question"},
                        "constraints": {
                            "type": "object",
                            "properties": {
                                "timeline": {"type": "string"},
                                "resources": {"type": "array", "items": {"type": "string"}},
                                "scope": {"type": "string"}
                            }
                        },
                        "depth": {"type": "string", "enum": ["overview", "detailed", "comprehensive"], "default": "detailed"}
                    },
                    "required": ["goal"]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "plan_id": {"type": "string"},
                        "goal": {"type": "string"},
                        "objectives": {"type": "array"},
                        "methodology": {"type": "object"},
                        "timeline": {"type": "object"}
                    }
                }
            },
            "search": {
                "description": "Search literature and knowledge sources",
                "input": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "sources": {"type": "array", "items": {"type": "string"}},
                        "timeframe": {"type": "string"},
                        "max_results": {"type": "number", "default": 20}
                    },
                    "required": ["query"]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "results": {"type": "array"},
                        "total_found": {"type": "number"}
                    }
                }
            },
            "synthesize": {
                "description": "Synthesize research findings",
                "input": {
                    "type": "object",
                    "properties": {
                        "findings": {"type": "array", "items": {"type": "object"}},
                        "focus": {"type": "string"},
                        "format": {"type": "string", "enum": ["summary", "detailed", "report"], "default": "summary"}
                    },
                    "required": ["findings"]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "synthesis_id": {"type": "string"},
                        "key_themes": {"type": "array"},
                        "insights": {"type": "array"},
                        "conclusions": {"type": "array"}
                    }
                }
            },
            "analyze": {
                "description": "Analyze a research topic",
                "input": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "analysis_type": {"type": "string", "enum": ["trend", "gap", "impact", "feasibility"]},
                        "context": {"type": "object"}
                    },
                    "required": ["topic", "analysis_type"]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "analysis_type": {"type": "string"},
                        "results": {"type": "object"}
                    }
                }
            }
        }

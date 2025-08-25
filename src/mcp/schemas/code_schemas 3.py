"""
Code Tool Schemas for MCP Integration

JSON schemas for code-related MCP tools.
"""

from typing import Dict, Any


class CodeSchemas:
    """Schema definitions for code tools."""
    
    def get_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get all code tool schemas."""
        return {
            "edit": {
                "description": "Edit code files with specified changes",
                "input": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to file to edit"},
                        "changes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "line_start": {"type": "number"},
                                    "line_end": {"type": "number"},
                                    "old_content": {"type": "string"},
                                    "new_content": {"type": "string"},
                                    "description": {"type": "string"}
                                },
                                "required": ["old_content", "new_content"]
                            }
                        },
                        "reason": {"type": "string", "description": "Reason for changes"}
                    },
                    "required": ["file_path", "changes"]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "edit_id": {"type": "string"},
                        "file_path": {"type": "string"},
                        "changes_applied": {"type": "number"},
                        "diff": {"type": "array"},
                        "validation": {"type": "object"}
                    }
                }
            },
            "review": {
                "description": "Review code for quality and best practices",
                "input": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to file to review"},
                        "code_content": {"type": "string", "description": "Code content to review"},
                        "review_type": {"type": "string", "enum": ["security", "performance", "quality", "style", "comprehensive"]},
                        "standards": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "review_id": {"type": "string"},
                        "overall_score": {"type": "number"},
                        "categories": {"type": "object"},
                        "issues": {"type": "array"},
                        "recommendations": {"type": "array"}
                    }
                }
            },
            "analyze": {
                "description": "Analyze code structure and metrics",
                "input": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "File path or code to analyze"},
                        "analysis_type": {"type": "string", "enum": ["complexity", "dependencies", "patterns", "metrics"]},
                        "language": {"type": "string", "description": "Programming language"}
                    },
                    "required": ["target", "analysis_type"]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "analysis_id": {"type": "string"},
                        "target": {"type": "string"},
                        "analysis_type": {"type": "string"},
                        "results": {"type": "object"},
                        "insights": {"type": "array"}
                    }
                }
            },
            "generate": {
                "description": "Generate code based on specification",
                "input": {
                    "type": "object",
                    "properties": {
                        "specification": {"type": "string", "description": "What to generate"},
                        "language": {"type": "string", "description": "Programming language"},
                        "template": {"type": "string", "description": "Template to follow"},
                        "constraints": {"type": "object"}
                    },
                    "required": ["specification", "language"]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "generation_id": {"type": "string"},
                        "specification": {"type": "string"},
                        "language": {"type": "string"},
                        "code": {"type": "string"},
                        "structure": {"type": "object"}
                    }
                }
            },
            "refactor": {
                "description": "Refactor code for better structure",
                "input": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "refactor_type": {"type": "string", "enum": ["extract_method", "rename", "optimize", "modernize"]},
                        "target_element": {"type": "string", "description": "What to refactor"},
                        "options": {"type": "object"}
                    },
                    "required": ["file_path", "refactor_type"]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "refactor_id": {"type": "string"},
                        "file_path": {"type": "string"},
                        "refactor_type": {"type": "string"},
                        "changes": {"type": "array"},
                        "improvements": {"type": "object"}
                    }
                }
            }
        }

"""
Dataset Tool Schemas for MCP Integration

JSON schemas for dataset-related MCP tools.
"""

from typing import Dict, Any


class DatasetSchemas:
    """Schema definitions for dataset tools."""
    
    def get_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get all dataset tool schemas."""
        return {
            "search": {
                "description": "Search for datasets by criteria",
                "input": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "filters": {
                            "type": "object",
                            "properties": {
                                "format": {"type": "string"},
                                "size_min": {"type": "number"},
                                "size_max": {"type": "number"},
                                "created_after": {"type": "string"},
                                "tags": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "limit": {"type": "number", "default": 20},
                        "offset": {"type": "number", "default": 0}
                    },
                    "required": ["query"]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array"},
                        "total": {"type": "number"},
                        "limit": {"type": "number"},
                        "offset": {"type": "number"}
                    }
                }
            },
            "preview": {
                "description": "Get preview of dataset structure and sample data",
                "input": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "Dataset identifier"},
                        "sample_size": {"type": "number", "default": 100},
                        "include_schema": {"type": "boolean", "default": True},
                        "include_stats": {"type": "boolean", "default": True}
                    },
                    "required": ["dataset_id"]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string"},
                        "schema": {"type": "object"},
                        "sample_data": {"type": "array"},
                        "statistics": {"type": "object"}
                    }
                }
            },
            "analyze": {
                "description": "Analyze dataset quality and characteristics",
                "input": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "Dataset identifier"},
                        "analysis_type": {
                            "type": "string",
                            "enum": ["quality", "drift", "distribution", "correlation"],
                            "default": "quality"
                        },
                        "reference_dataset": {"type": "string", "description": "Reference for comparison"}
                    },
                    "required": ["dataset_id"]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string"},
                        "analysis_type": {"type": "string"},
                        "results": {"type": "object"}
                    }
                }
            },
            "list": {
                "description": "List available datasets",
                "input": {
                    "type": "object",
                    "properties": {
                        "project": {"type": "string"},
                        "limit": {"type": "number", "default": 50},
                        "offset": {"type": "number", "default": 0}
                    }
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "datasets": {"type": "array"},
                        "total": {"type": "number"}
                    }
                }
            }
        }

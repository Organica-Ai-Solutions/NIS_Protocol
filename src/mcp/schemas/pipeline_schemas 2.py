"""
Pipeline Tool Schemas for MCP Integration

JSON schemas for pipeline-related MCP tools.
"""

from typing import Dict, Any


class PipelineSchemas:
    """Schema definitions for pipeline tools."""
    
    def get_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get all pipeline tool schemas."""
        return {
            "run": {
                "description": "Execute a data processing pipeline",
                "input": {
                    "type": "object",
                    "properties": {
                        "pipeline_id": {"type": "string", "description": "Pipeline identifier"},
                        "config": {
                            "type": "object",
                            "properties": {
                                "input_dataset": {"type": "string"},
                                "output_path": {"type": "string"},
                                "parameters": {"type": "object"},
                                "resources": {
                                    "type": "object",
                                    "properties": {
                                        "cpu_cores": {"type": "number"},
                                        "memory_gb": {"type": "number"},
                                        "gpu_count": {"type": "number"}
                                    }
                                }
                            }
                        },
                        "async": {"type": "boolean", "default": True},
                        "priority": {"type": "string", "enum": ["low", "normal", "high"], "default": "normal"}
                    },
                    "required": ["pipeline_id", "config"]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "status": {"type": "string"},
                        "started_at": {"type": "string"}
                    }
                }
            },
            "status": {
                "description": "Get status of a pipeline run",
                "input": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "Pipeline run identifier"}
                    },
                    "required": ["run_id"]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "status": {"type": "string"},
                        "progress": {"type": "number"},
                        "current_step": {"type": "string"}
                    }
                }
            },
            "configure": {
                "description": "Configure pipeline parameters",
                "input": {
                    "type": "object",
                    "properties": {
                        "pipeline_id": {"type": "string"},
                        "configuration": {"type": "object"},
                        "validate_only": {"type": "boolean", "default": False}
                    },
                    "required": ["pipeline_id", "configuration"]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "pipeline_id": {"type": "string"},
                        "validation": {"type": "object"},
                        "applied": {"type": "boolean"}
                    }
                }
            },
            "cancel": {
                "description": "Cancel a running pipeline",
                "input": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "Pipeline run identifier"}
                    },
                    "required": ["run_id"]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "status": {"type": "string"},
                        "message": {"type": "string"}
                    }
                }
            },
            "artifacts": {
                "description": "Get pipeline execution artifacts",
                "input": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "Pipeline run identifier"},
                        "artifact_type": {"type": "string", "enum": ["logs", "outputs", "metrics", "all"]}
                    },
                    "required": ["run_id"]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "artifacts": {"type": "object"}
                    }
                }
            },
            "list": {
                "description": "List available pipelines and recent runs",
                "input": {
                    "type": "object",
                    "properties": {
                        "project": {"type": "string"},
                        "status": {"type": "string", "enum": ["running", "completed", "failed", "cancelled"]},
                        "limit": {"type": "number", "default": 20}
                    }
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "pipelines": {"type": "array"},
                        "recent_runs": {"type": "array"}
                    }
                }
            }
        }

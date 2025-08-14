"""
Audit Tool Schemas for MCP Integration

JSON schemas for audit-related MCP tools.
"""

from typing import Dict, Any


class AuditSchemas:
    """Schema definitions for audit tools."""
    
    def get_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get all audit tool schemas."""
        return {
            "view": {
                "description": "View audit trail for runs or timeframes",
                "input": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "Specific run to audit"},
                        "timeframe": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "string"},
                                "end": {"type": "string"}
                            }
                        },
                        "component": {"type": "string", "description": "Specific component to audit"},
                        "level": {"type": "string", "enum": ["summary", "detailed", "full"], "default": "summary"}
                    }
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "audit_id": {"type": "string"},
                        "timeline": {"type": "array"},
                        "summary": {"type": "object"},
                        "metrics": {"type": "object"}
                    }
                }
            },
            "analyze": {
                "description": "Analyze execution performance and behavior",
                "input": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "What to analyze"},
                        "metrics": {"type": "array", "items": {"type": "string"}},
                        "comparison_baseline": {"type": "string"}
                    },
                    "required": ["target"]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "analysis_id": {"type": "string"},
                        "performance": {"type": "object"},
                        "patterns": {"type": "array"},
                        "recommendations": {"type": "array"}
                    }
                }
            },
            "compliance": {
                "description": "Check compliance against frameworks",
                "input": {
                    "type": "object",
                    "properties": {
                        "framework": {"type": "string", "description": "Compliance framework"},
                        "scope": {"type": "string", "enum": ["system", "data", "operations", "all"]},
                        "level": {"type": "string", "enum": ["basic", "standard", "comprehensive"]}
                    },
                    "required": ["framework"]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "compliance_id": {"type": "string"},
                        "framework": {"type": "string"},
                        "overall_score": {"type": "number"},
                        "status": {"type": "string"},
                        "areas": {"type": "array"}
                    }
                }
            },
            "risk": {
                "description": "Assess risk across dimensions",
                "input": {
                    "type": "object",
                    "properties": {
                        "assessment_type": {"type": "string", "enum": ["security", "operational", "data", "comprehensive"]},
                        "scope": {"type": "string"},
                        "severity_threshold": {"type": "string", "enum": ["low", "medium", "high", "critical"]}
                    },
                    "required": ["assessment_type"]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "assessment_id": {"type": "string"},
                        "overall_risk_level": {"type": "string"},
                        "risk_score": {"type": "number"},
                        "risks": {"type": "array"}
                    }
                }
            },
            "report": {
                "description": "Generate audit reports",
                "input": {
                    "type": "object",
                    "properties": {
                        "report_type": {"type": "string", "enum": ["summary", "detailed", "compliance", "risk"]},
                        "timeframe": {"type": "object"},
                        "format": {"type": "string", "enum": ["json", "html", "pdf"], "default": "json"},
                        "recipients": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["report_type"]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "report_id": {"type": "string"},
                        "type": {"type": "string"},
                        "executive_summary": {"type": "string"},
                        "sections": {"type": "array"}
                    }
                }
            }
        }

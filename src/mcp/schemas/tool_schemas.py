"""
Tool Schemas for MCP Integration

Centralizes all JSON schemas for MCP tools exposed by NIS Protocol.
Each tool has input/output schemas for validation and documentation.
"""

from typing import Dict, Any
import json

from .dataset_schemas import DatasetSchemas
from .pipeline_schemas import PipelineSchemas
from .research_schemas import ResearchSchemas
from .audit_schemas import AuditSchemas
from .code_schemas import CodeSchemas


class ToolSchemas:
    """
    Centralized schema registry for all MCP tools.
    
    Provides JSON schemas for validation, documentation, and client generation.
    Organized by skill domain (dataset, pipeline, research, audit, code).
    """
    
    def __init__(self):
        self.dataset = DatasetSchemas()
        self.pipeline = PipelineSchemas()
        self.research = ResearchSchemas()
        self.audit = AuditSchemas()
        self.code = CodeSchemas()
        
    def get_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get all tool definitions with their schemas."""
        tools = {}
        
        # Dataset tools
        for tool_name, schema in self.dataset.get_schemas().items():
            tools[f"dataset.{tool_name}"] = {
                "name": f"dataset.{tool_name}",
                "description": schema.get("description", f"Dataset {tool_name} operation"),
                "input_schema": schema["input"],
                "output_schema": schema.get("output", {}),
                "skill": "dataset"
            }
            
        # Pipeline tools
        for tool_name, schema in self.pipeline.get_schemas().items():
            tools[f"pipeline.{tool_name}"] = {
                "name": f"pipeline.{tool_name}",
                "description": schema.get("description", f"Pipeline {tool_name} operation"),
                "input_schema": schema["input"],
                "output_schema": schema.get("output", {}),
                "skill": "pipeline"
            }
            
        # Research tools
        for tool_name, schema in self.research.get_schemas().items():
            tools[f"research.{tool_name}"] = {
                "name": f"research.{tool_name}",
                "description": schema.get("description", f"Research {tool_name} operation"),
                "input_schema": schema["input"],
                "output_schema": schema.get("output", {}),
                "skill": "research"
            }
            
        # Audit tools
        for tool_name, schema in self.audit.get_schemas().items():
            tools[f"audit.{tool_name}"] = {
                "name": f"audit.{tool_name}",
                "description": schema.get("description", f"Audit {tool_name} operation"),
                "input_schema": schema["input"],
                "output_schema": schema.get("output", {}),
                "skill": "audit"
            }
            
        # Code tools
        for tool_name, schema in self.code.get_schemas().items():
            tools[f"code.{tool_name}"] = {
                "name": f"code.{tool_name}",
                "description": schema.get("description", f"Code {tool_name} operation"),
                "input_schema": schema["input"],
                "output_schema": schema.get("output", {}),
                "skill": "code"
            }
            
        return tools
        
    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get schema for a specific tool."""
        tools = self.get_all_tools()
        return tools.get(tool_name, {})
        
    def validate_tool_input(self, tool_name: str, input_data: Dict[str, Any]) -> tuple[bool, list]:
        """
        Validate input data against tool schema.
        
        Returns:
            (is_valid, errors)
        """
        tool_schema = self.get_tool_schema(tool_name)
        if not tool_schema:
            return False, [f"Unknown tool: {tool_name}"]
            
        input_schema = tool_schema.get("input_schema", {})
        return self._validate_against_schema(input_data, input_schema)
        
    def get_mcp_tool_definitions(self) -> list:
        """
        Get tool definitions in MCP format for registration.
        
        Returns list of tool definitions suitable for MCP server registration.
        """
        mcp_tools = []
        tools = self.get_all_tools()
        
        for tool_name, tool_def in tools.items():
            mcp_tool = {
                "name": tool_name,
                "description": tool_def["description"],
                "inputSchema": {
                    "type": "object",
                    "properties": tool_def["input_schema"].get("properties", {}),
                    "required": tool_def["input_schema"].get("required", [])
                }
            }
            mcp_tools.append(mcp_tool)
            
        return mcp_tools
        
    def _validate_against_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, list]:
        """Basic JSON schema validation."""
        errors = []
        
        try:
            # Check required fields
            required = schema.get("required", [])
            for field in required:
                if field not in data:
                    errors.append(f"Missing required field: {field}")
                    
            # Basic type checking for properties
            properties = schema.get("properties", {})
            for field, value in data.items():
                if field in properties:
                    field_schema = properties[field]
                    expected_type = field_schema.get("type")
                    
                    if expected_type == "string" and not isinstance(value, str):
                        errors.append(f"Field '{field}' must be string")
                    elif expected_type == "number" and not isinstance(value, (int, float)):
                        errors.append(f"Field '{field}' must be number")
                    elif expected_type == "boolean" and not isinstance(value, bool):
                        errors.append(f"Field '{field}' must be boolean")
                    elif expected_type == "array" and not isinstance(value, list):
                        errors.append(f"Field '{field}' must be array")
                    elif expected_type == "object" and not isinstance(value, dict):
                        errors.append(f"Field '{field}' must be object")
                        
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]
            
    def export_schemas(self, format: str = "json") -> str:
        """Export all schemas in specified format."""
        tools = self.get_all_tools()
        
        if format == "json":
            return json.dumps(tools, indent=2)
        elif format == "mcp":
            return json.dumps(self.get_mcp_tool_definitions(), indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def get_schema_summary(self) -> Dict[str, Any]:
        """Get summary of all available tools and their capabilities."""
        tools = self.get_all_tools()
        
        summary = {
            "total_tools": len(tools),
            "skills": {},
            "tools_by_skill": {}
        }
        
        for tool_name, tool_def in tools.items():
            skill = tool_def["skill"]
            if skill not in summary["skills"]:
                summary["skills"][skill] = 0
                summary["tools_by_skill"][skill] = []
                
            summary["skills"][skill] += 1
            summary["tools_by_skill"][skill].append(tool_name)
            
        return summary

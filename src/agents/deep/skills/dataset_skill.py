"""
Dataset Skill for Deep Agents

Handles dataset operations like search, preview, analysis.
Maps to MCP tools: dataset.search, dataset.preview, dataset.analyze
"""

from typing import Dict, Any, List
import json

from .base_skill import BaseSkill


class DatasetSkill(BaseSkill):
    """
    Skill for dataset operations within NIS Protocol.
    
    Provides capabilities for:
    - Searching datasets by criteria
    - Previewing dataset structure and samples
    - Analyzing dataset properties and quality
    """
    
    def __init__(self, agent, memory_manager, config=None):
        super().__init__(agent, memory_manager, config)
        self.core_services = None  # Will be injected
        
    async def execute(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a dataset action."""
        if not self._validate_parameters(action, parameters):
            return self._format_error(f"Invalid parameters for action '{action}'", "ValidationError")
            
        try:
            if action == "search":
                result = await self._search_datasets(parameters)
            elif action == "preview":
                result = await self._preview_dataset(parameters)
            elif action == "analyze":
                result = await self._analyze_dataset(parameters)
            elif action == "list":
                result = await self._list_datasets(parameters)
            else:
                return self._format_error(f"Unknown action '{action}'", "ActionError")
                
            await self._store_result(action, parameters, result)
            return self._format_success(result)
            
        except Exception as e:
            return self._format_error(str(e), "ExecutionError")
            
    def get_available_actions(self) -> List[str]:
        """Get available dataset actions."""
        return ["search", "preview", "analyze", "list"]
        
    def _get_action_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get schemas for dataset actions."""
        return {
            "search": {
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
            "preview": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset identifier"},
                    "sample_size": {"type": "number", "default": 100},
                    "include_schema": {"type": "boolean", "default": True},
                    "include_stats": {"type": "boolean", "default": True}
                },
                "required": ["dataset_id"]
            },
            "analyze": {
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
            "list": {
                "type": "object",
                "properties": {
                    "project": {"type": "string"},
                    "limit": {"type": "number", "default": 50},
                    "offset": {"type": "number", "default": 0}
                }
            }
        }
        
    async def _search_datasets(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Search for datasets based on criteria."""
        query = parameters["query"]
        filters = parameters.get("filters", {})
        limit = parameters.get("limit", 20)
        offset = parameters.get("offset", 0)
        
        # Use agent to search datasets
        prompt = f"""
Search for datasets matching this query: {query}

Filters: {json.dumps(filters, indent=2)}

Return results in this format:
{{
    "items": [
        {{
            "id": "dataset_id",
            "name": "Dataset Name",
            "description": "Description",
            "format": "csv|json|parquet",
            "size_bytes": 1024,
            "rows": 1000,
            "columns": 10,
            "created_at": "2025-01-19T10:00:00Z",
            "tags": ["tag1", "tag2"],
            "quality_score": 0.85
        }}
    ],
    "total": 42,
    "limit": {limit},
    "offset": {offset}
}}
"""
        
        response = await self._call_agent(prompt, {"action": "search_datasets"})
        
        # Parse and validate response
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
                
            return {
                "items": content.get("items", []),
                "total": content.get("total", 0),
                "limit": limit,
                "offset": offset,
                "search_query": query,
                "filters_applied": filters
            }
        except Exception:
            # Fallback response
            return {
                "items": [],
                "total": 0,
                "limit": limit,
                "offset": offset,
                "search_query": query,
                "error": "Failed to parse search results"
            }
            
    async def _preview_dataset(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get a preview of a dataset."""
        dataset_id = parameters["dataset_id"]
        sample_size = parameters.get("sample_size", 100)
        include_schema = parameters.get("include_schema", True)
        include_stats = parameters.get("include_stats", True)
        
        prompt = f"""
Generate a preview for dataset: {dataset_id}

Requirements:
- Sample size: {sample_size} rows
- Include schema: {include_schema}
- Include statistics: {include_stats}

Return data in this format:
{{
    "dataset_id": "{dataset_id}",
    "schema": {{
        "columns": [
            {{
                "name": "column_name",
                "type": "string|number|boolean|date",
                "nullable": true,
                "description": "Column description"
            }}
        ]
    }},
    "sample_data": [
        {{"column1": "value1", "column2": 123}},
        {{"column1": "value2", "column2": 456}}
    ],
    "statistics": {{
        "total_rows": 10000,
        "total_columns": 5,
        "missing_values": {{"column1": 10, "column2": 0}},
        "data_types": {{"string": 2, "number": 2, "boolean": 1}},
        "quality_score": 0.92
    }}
}}
"""
        
        response = await self._call_agent(prompt, {"action": "preview_dataset"})
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
                
            return content
        except Exception:
            return {
                "dataset_id": dataset_id,
                "schema": {"columns": []},
                "sample_data": [],
                "statistics": {},
                "error": "Failed to generate preview"
            }
            
    async def _analyze_dataset(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dataset properties and quality."""
        dataset_id = parameters["dataset_id"]
        analysis_type = parameters.get("analysis_type", "quality")
        reference_dataset = parameters.get("reference_dataset")
        
        prompt = f"""
Perform {analysis_type} analysis on dataset: {dataset_id}
{f"Reference dataset: {reference_dataset}" if reference_dataset else ""}

Analysis type: {analysis_type}
- quality: Data quality metrics, completeness, consistency
- drift: Data drift detection compared to reference
- distribution: Statistical distribution analysis
- correlation: Feature correlation analysis

Return analysis results in structured format appropriate for the analysis type.
"""
        
        response = await self._call_agent(prompt, {"action": "analyze_dataset"})
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
                
            return {
                "dataset_id": dataset_id,
                "analysis_type": analysis_type,
                "reference_dataset": reference_dataset,
                "results": content,
                "timestamp": self._get_timestamp()
            }
        except Exception:
            return {
                "dataset_id": dataset_id,
                "analysis_type": analysis_type,
                "results": {},
                "error": "Failed to perform analysis"
            }
            
    async def _list_datasets(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """List available datasets."""
        project = parameters.get("project")
        limit = parameters.get("limit", 50)
        offset = parameters.get("offset", 0)
        
        prompt = f"""
List available datasets.
{f"Project filter: {project}" if project else "All projects"}

Return in format:
{{
    "datasets": [
        {{
            "id": "dataset_id",
            "name": "Dataset Name",
            "project": "project_name",
            "format": "csv",
            "size_bytes": 1024,
            "created_at": "2025-01-19T10:00:00Z"
        }}
    ],
    "total": 10
}}
"""
        
        response = await self._call_agent(prompt, {"action": "list_datasets"})
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
                
            return {
                "datasets": content.get("datasets", []),
                "total": content.get("total", 0),
                "project": project,
                "limit": limit,
                "offset": offset
            }
        except Exception:
            return {
                "datasets": [],
                "total": 0,
                "project": project,
                "limit": limit,
                "offset": offset,
                "error": "Failed to list datasets"
            }
            
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"

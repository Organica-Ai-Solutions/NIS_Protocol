"""
Pipeline Skill for Deep Agents

Handles pipeline operations like run, monitor, configure.
Maps to MCP tools: pipeline.run, pipeline.status, pipeline.configure
"""

from typing import Dict, Any, List
import json
import asyncio
import uuid

from .base_skill import BaseSkill


class PipelineSkill(BaseSkill):
    """
    Skill for pipeline operations within NIS Protocol.
    
    Provides capabilities for:
    - Running data processing pipelines
    - Monitoring pipeline execution
    - Configuring pipeline parameters
    - Managing pipeline artifacts
    """
    
    def __init__(self, agent, memory_manager, config=None):
        super().__init__(agent, memory_manager, config)
        self.active_runs = {}
        
    async def execute(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a pipeline action."""
        if not self._validate_parameters(action, parameters):
            return self._format_error(f"Invalid parameters for action '{action}'", "ValidationError")
            
        try:
            if action == "run":
                result = await self._run_pipeline(parameters)
            elif action == "status":
                result = await self._get_pipeline_status(parameters)
            elif action == "configure":
                result = await self._configure_pipeline(parameters)
            elif action == "cancel":
                result = await self._cancel_pipeline(parameters)
            elif action == "artifacts":
                result = await self._get_artifacts(parameters)
            elif action == "list":
                result = await self._list_pipelines(parameters)
            else:
                return self._format_error(f"Unknown action '{action}'", "ActionError")
                
            await self._store_result(action, parameters, result)
            return self._format_success(result)
            
        except Exception as e:
            return self._format_error(str(e), "ExecutionError")
            
    def get_available_actions(self) -> List[str]:
        """Get available pipeline actions."""
        return ["run", "status", "configure", "cancel", "artifacts", "list"]
        
    def _get_action_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get schemas for pipeline actions."""
        return {
            "run": {
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
            "status": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string", "description": "Pipeline run identifier"}
                },
                "required": ["run_id"]
            },
            "configure": {
                "type": "object",
                "properties": {
                    "pipeline_id": {"type": "string"},
                    "configuration": {"type": "object"},
                    "validate_only": {"type": "boolean", "default": False}
                },
                "required": ["pipeline_id", "configuration"]
            },
            "cancel": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string", "description": "Pipeline run identifier"}
                },
                "required": ["run_id"]
            },
            "artifacts": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string", "description": "Pipeline run identifier"},
                    "artifact_type": {"type": "string", "enum": ["logs", "outputs", "metrics", "all"]}
                },
                "required": ["run_id"]
            },
            "list": {
                "type": "object",
                "properties": {
                    "project": {"type": "string"},
                    "status": {"type": "string", "enum": ["running", "completed", "failed", "cancelled"]},
                    "limit": {"type": "number", "default": 20}
                }
            }
        }
        
    async def _run_pipeline(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run a data processing pipeline."""
        pipeline_id = parameters["pipeline_id"]
        config = parameters["config"]
        is_async = parameters.get("async", True)
        priority = parameters.get("priority", "normal")
        
        run_id = str(uuid.uuid4())
        
        # Store run information
        run_info = {
            "run_id": run_id,
            "pipeline_id": pipeline_id,
            "config": config,
            "status": "initializing",
            "priority": priority,
            "started_at": self._get_timestamp(),
            "progress": 0,
            "logs": [],
            "artifacts": {}
        }
        
        self.active_runs[run_id] = run_info
        
        # Use agent to execute pipeline
        prompt = f"""
Execute pipeline: {pipeline_id}
Configuration: {json.dumps(config, indent=2)}
Priority: {priority}
Run ID: {run_id}

Simulate pipeline execution with realistic progress updates.
Generate execution steps and progress for a typical data processing pipeline.

Return execution status in this format:
{{
    "run_id": "{run_id}",
    "status": "running",
    "progress": 25,
    "current_step": "data_validation",
    "estimated_completion": "2025-01-19T10:30:00Z",
    "steps": [
        {{"name": "data_loading", "status": "completed", "duration_seconds": 30}},
        {{"name": "data_validation", "status": "running", "progress": 60}},
        {{"name": "processing", "status": "pending"}},
        {{"name": "output_generation", "status": "pending"}}
    ],
    "resource_usage": {{
        "cpu_percent": 45,
        "memory_mb": 1024,
        "gpu_percent": 80
    }}
}}
"""
        
        response = await self._call_agent(prompt, {"action": "run_pipeline"})
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
                
            # Update run info with response
            run_info.update(content)
            run_info["status"] = "running"
            
            if is_async:
                # Start background monitoring
                asyncio.create_task(self._monitor_pipeline_execution(run_id))
                
            return {
                "run_id": run_id,
                "pipeline_id": pipeline_id,
                "status": "running" if is_async else "completed",
                "async": is_async,
                "started_at": run_info["started_at"],
                "progress": content.get("progress", 0),
                "estimated_completion": content.get("estimated_completion"),
                "monitoring_url": f"/pipeline/status/{run_id}"
            }
            
        except Exception as e:
            run_info["status"] = "failed"
            run_info["error"] = str(e)
            return {
                "run_id": run_id,
                "status": "failed",
                "error": str(e)
            }
            
    async def _get_pipeline_status(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get status of a pipeline run."""
        run_id = parameters["run_id"]
        
        if run_id not in self.active_runs:
            return {
                "run_id": run_id,
                "status": "not_found",
                "error": "Pipeline run not found"
            }
            
        run_info = self.active_runs[run_id]
        
        # Simulate status updates for demo
        if run_info["status"] == "running":
            # Update progress
            current_progress = run_info.get("progress", 0)
            if current_progress < 100:
                run_info["progress"] = min(100, current_progress + 5)
                
            if run_info["progress"] >= 100:
                run_info["status"] = "completed"
                run_info["completed_at"] = self._get_timestamp()
                
        return {
            "run_id": run_id,
            "pipeline_id": run_info["pipeline_id"],
            "status": run_info["status"],
            "progress": run_info.get("progress", 0),
            "started_at": run_info["started_at"],
            "completed_at": run_info.get("completed_at"),
            "current_step": run_info.get("current_step"),
            "steps": run_info.get("steps", []),
            "resource_usage": run_info.get("resource_usage", {}),
            "artifacts": list(run_info.get("artifacts", {}).keys()),
            "logs_count": len(run_info.get("logs", []))
        }
        
    async def _configure_pipeline(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Configure pipeline parameters."""
        pipeline_id = parameters["pipeline_id"]
        configuration = parameters["configuration"]
        validate_only = parameters.get("validate_only", False)
        
        prompt = f"""
Configure pipeline: {pipeline_id}
Configuration: {json.dumps(configuration, indent=2)}
Validate only: {validate_only}

Validate the configuration and return validation results.
If not validate_only, also apply the configuration.

Return result in format:
{{
    "pipeline_id": "{pipeline_id}",
    "validation": {{
        "valid": true,
        "errors": [],
        "warnings": ["Optional warning messages"]
    }},
    "applied": {str(not validate_only).lower()},
    "configuration": {configuration}
}}
"""
        
        response = await self._call_agent(prompt, {"action": "configure_pipeline"})
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
                
            return content
        except Exception:
            return {
                "pipeline_id": pipeline_id,
                "validation": {
                    "valid": False,
                    "errors": ["Failed to validate configuration"],
                    "warnings": []
                },
                "applied": False,
                "configuration": configuration
            }
            
    async def _cancel_pipeline(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel a running pipeline."""
        run_id = parameters["run_id"]
        
        if run_id not in self.active_runs:
            return {
                "run_id": run_id,
                "status": "not_found",
                "error": "Pipeline run not found"
            }
            
        run_info = self.active_runs[run_id]
        
        if run_info["status"] in ["completed", "failed", "cancelled"]:
            return {
                "run_id": run_id,
                "status": run_info["status"],
                "message": f"Pipeline is already {run_info['status']}"
            }
            
        # Cancel the pipeline
        run_info["status"] = "cancelled"
        run_info["cancelled_at"] = self._get_timestamp()
        
        return {
            "run_id": run_id,
            "status": "cancelled",
            "cancelled_at": run_info["cancelled_at"],
            "message": "Pipeline execution cancelled successfully"
        }
        
    async def _get_artifacts(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get pipeline execution artifacts."""
        run_id = parameters["run_id"]
        artifact_type = parameters.get("artifact_type", "all")
        
        if run_id not in self.active_runs:
            return {
                "run_id": run_id,
                "error": "Pipeline run not found"
            }
            
        run_info = self.active_runs[run_id]
        
        artifacts = {
            "logs": run_info.get("logs", []),
            "outputs": run_info.get("artifacts", {}).get("outputs", []),
            "metrics": run_info.get("artifacts", {}).get("metrics", {}),
            "metadata": {
                "run_id": run_id,
                "pipeline_id": run_info["pipeline_id"],
                "status": run_info["status"],
                "started_at": run_info["started_at"]
            }
        }
        
        if artifact_type != "all":
            artifacts = {artifact_type: artifacts.get(artifact_type, {})}
            
        return {
            "run_id": run_id,
            "artifact_type": artifact_type,
            "artifacts": artifacts
        }
        
    async def _list_pipelines(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """List available pipelines and recent runs."""
        project = parameters.get("project")
        status_filter = parameters.get("status")
        limit = parameters.get("limit", 20)
        
        prompt = f"""
List available pipelines and recent runs.
{f"Project filter: {project}" if project else "All projects"}
{f"Status filter: {status_filter}" if status_filter else "All statuses"}

Return in format:
{{
    "pipelines": [
        {{
            "id": "pipeline_id",
            "name": "Pipeline Name",
            "description": "Description",
            "project": "project_name",
            "created_at": "2025-01-19T10:00:00Z"
        }}
    ],
    "recent_runs": [
        {{
            "run_id": "run_id",
            "pipeline_id": "pipeline_id",
            "status": "completed",
            "started_at": "2025-01-19T10:00:00Z",
            "completed_at": "2025-01-19T10:30:00Z"
        }}
    ]
}}
"""
        
        response = await self._call_agent(prompt, {"action": "list_pipelines"})
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
                
            # Add active runs from memory
            active_runs = []
            for run_id, run_info in self.active_runs.items():
                if not status_filter or run_info["status"] == status_filter:
                    active_runs.append({
                        "run_id": run_id,
                        "pipeline_id": run_info["pipeline_id"],
                        "status": run_info["status"],
                        "started_at": run_info["started_at"],
                        "progress": run_info.get("progress", 0)
                    })
                    
            content["recent_runs"] = active_runs + content.get("recent_runs", [])
            content["recent_runs"] = content["recent_runs"][:limit]
            
            return content
        except Exception:
            return {
                "pipelines": [],
                "recent_runs": []
            }
            
    async def _monitor_pipeline_execution(self, run_id: str):
        """Background task to monitor pipeline execution."""
        while run_id in self.active_runs:
            run_info = self.active_runs[run_id]
            
            if run_info["status"] in ["completed", "failed", "cancelled"]:
                break
                
            # Simulate progress updates
            await asyncio.sleep(5)
            current_progress = run_info.get("progress", 0)
            if current_progress < 100:
                run_info["progress"] = min(100, current_progress + 2)
                
            if run_info["progress"] >= 100:
                run_info["status"] = "completed"
                run_info["completed_at"] = self._get_timestamp()
                
        # Clean up completed runs after some time
        await asyncio.sleep(300)  # Keep for 5 minutes
        if run_id in self.active_runs:
            del self.active_runs[run_id]
            
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"

"""
Audit Skill for Deep Agents

Handles audit operations like view history, analyze runs, compliance checks.
Maps to MCP tools: audit.view, audit.analyze, audit.compliance
"""

from typing import Dict, Any, List
import json

from .base_skill import BaseSkill


class AuditSkill(BaseSkill):
    """
    Skill for audit operations within NIS Protocol.
    
    Provides capabilities for:
    - Viewing execution history and audit trails
    - Analyzing system performance and behavior
    - Compliance checking and reporting
    - Risk assessment and monitoring
    """
    
    def __init__(self, agent, memory_manager, config=None):
        super().__init__(agent, memory_manager, config)
        
    async def execute(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an audit action."""
        if not self._validate_parameters(action, parameters):
            return self._format_error(f"Invalid parameters for action '{action}'", "ValidationError")
            
        try:
            if action == "view":
                result = await self._view_audit_trail(parameters)
            elif action == "analyze":
                result = await self._analyze_execution(parameters)
            elif action == "compliance":
                result = await self._check_compliance(parameters)
            elif action == "risk":
                result = await self._assess_risk(parameters)
            elif action == "report":
                result = await self._generate_report(parameters)
            else:
                return self._format_error(f"Unknown action '{action}'", "ActionError")
                
            await self._store_result(action, parameters, result)
            return self._format_success(result)
            
        except Exception as e:
            return self._format_error(str(e), "ExecutionError")
            
    def get_available_actions(self) -> List[str]:
        """Get available audit actions."""
        return ["view", "analyze", "compliance", "risk", "report"]
        
    def _get_action_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get schemas for audit actions."""
        return {
            "view": {
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
            "analyze": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "What to analyze (run_id, component, timeframe)"},
                    "metrics": {"type": "array", "items": {"type": "string"}},
                    "comparison_baseline": {"type": "string"}
                },
                "required": ["target"]
            },
            "compliance": {
                "type": "object",
                "properties": {
                    "framework": {"type": "string", "description": "Compliance framework to check against"},
                    "scope": {"type": "string", "enum": ["system", "data", "operations", "all"]},
                    "level": {"type": "string", "enum": ["basic", "standard", "comprehensive"]}
                },
                "required": ["framework"]
            },
            "risk": {
                "type": "object",
                "properties": {
                    "assessment_type": {"type": "string", "enum": ["security", "operational", "data", "comprehensive"]},
                    "scope": {"type": "string"},
                    "severity_threshold": {"type": "string", "enum": ["low", "medium", "high", "critical"]}
                },
                "required": ["assessment_type"]
            },
            "report": {
                "type": "object",
                "properties": {
                    "report_type": {"type": "string", "enum": ["summary", "detailed", "compliance", "risk"]},
                    "timeframe": {"type": "object"},
                    "format": {"type": "string", "enum": ["json", "html", "pdf"], "default": "json"},
                    "recipients": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["report_type"]
            }
        }
        
    async def _view_audit_trail(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """View audit trail for a specific run or timeframe."""
        run_id = parameters.get("run_id")
        timeframe = parameters.get("timeframe", {})
        component = parameters.get("component")
        level = parameters.get("level", "summary")
        
        prompt = f"""
Generate audit trail view:
Run ID: {run_id if run_id else "Not specified"}
Timeframe: {json.dumps(timeframe) if timeframe else "Not specified"}
Component: {component if component else "All components"}
Detail level: {level}

Return audit trail in this format:
{{
    "audit_id": "audit_123",
    "scope": {{
        "run_id": "{run_id}",
        "timeframe": {timeframe},
        "component": "{component}"
    }},
    "timeline": [
        {{
            "timestamp": "2025-01-19T10:00:00Z",
            "event": "system_start",
            "component": "core",
            "details": "System initialization completed",
            "user": "system",
            "metadata": {{}}
        }}
    ],
    "summary": {{
        "total_events": 25,
        "event_types": {{"info": 20, "warning": 4, "error": 1}},
        "components_involved": ["core", "agents", "memory"],
        "duration": "30 minutes"
    }},
    "metrics": {{
        "performance": {{"avg_response_time": 150}},
        "errors": {{"count": 1, "types": ["timeout"]}},
        "resource_usage": {{"peak_memory": "2.5GB", "cpu_avg": "45%"}}
    }}
}}
"""
        
        response = await self._call_agent(prompt, {"action": "view_audit_trail"})
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
                
            return {
                "generated_at": self._get_timestamp(),
                "level": level,
                **content
            }
        except Exception:
            return {
                "audit_id": "failed_audit",
                "scope": {"run_id": run_id, "component": component},
                "timeline": [],
                "summary": {},
                "metrics": {},
                "error": "Failed to generate audit trail"
            }
            
    async def _analyze_execution(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze execution performance and behavior."""
        target = parameters["target"]
        metrics = parameters.get("metrics", [])
        baseline = parameters.get("comparison_baseline")
        
        prompt = f"""
Analyze execution for target: {target}
Metrics to analyze: {metrics if metrics else "all available"}
Comparison baseline: {baseline if baseline else "none"}

Provide detailed analysis including:
1. Performance metrics
2. Behavioral patterns
3. Anomalies or issues
4. Recommendations for improvement

Return in this format:
{{
    "analysis_id": "analysis_123",
    "target": "{target}",
    "metrics_analyzed": {metrics},
    "performance": {{
        "response_times": {{"avg": 150, "p95": 300, "p99": 500}},
        "throughput": {{"requests_per_sec": 10}},
        "resource_usage": {{"cpu": {{"avg": 45, "peak": 80}}, "memory": {{"avg": "1.5GB", "peak": "2.1GB"}}}}
    }},
    "patterns": [
        {{"pattern": "periodic_spikes", "description": "CPU usage spikes every 5 minutes", "impact": "medium"}}
    ],
    "anomalies": [
        {{"anomaly": "timeout_cluster", "timestamp": "2025-01-19T10:15:00Z", "severity": "high"}}
    ],
    "recommendations": [
        {{"action": "increase_timeout", "priority": "high", "rationale": "Reduce timeout errors"}}
    ],
    "comparison": {{
        "baseline": "{baseline}",
        "improvements": ["response_time_20_percent_better"],
        "regressions": ["memory_usage_increased"]
    }}
}}
"""
        
        response = await self._call_agent(prompt, {"action": "analyze_execution"})
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
                
            return {
                "generated_at": self._get_timestamp(),
                **content
            }
        except Exception:
            return {
                "analysis_id": "failed_analysis",
                "target": target,
                "performance": {},
                "patterns": [],
                "anomalies": [],
                "recommendations": [],
                "error": "Failed to analyze execution"
            }
            
    async def _check_compliance(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance against specified framework."""
        framework = parameters["framework"]
        scope = parameters.get("scope", "all")
        level = parameters.get("level", "standard")
        
        prompt = f"""
Perform compliance check:
Framework: {framework}
Scope: {scope}
Level: {level}

Check compliance across relevant areas and provide detailed results.

Return in this format:
{{
    "compliance_id": "compliance_123",
    "framework": "{framework}",
    "scope": "{scope}",
    "level": "{level}",
    "overall_score": 85,
    "status": "compliant",
    "areas": [
        {{
            "area": "data_protection",
            "score": 90,
            "status": "compliant",
            "requirements_met": 18,
            "requirements_total": 20,
            "gaps": ["encryption_at_rest", "key_rotation"]
        }}
    ],
    "recommendations": [
        {{"priority": "high", "action": "implement_encryption", "area": "data_protection"}}
    ],
    "next_review": "2025-04-19T10:00:00Z"
}}
"""
        
        response = await self._call_agent(prompt, {"action": "check_compliance"})
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
                
            return {
                "checked_at": self._get_timestamp(),
                **content
            }
        except Exception:
            return {
                "compliance_id": "failed_compliance",
                "framework": framework,
                "scope": scope,
                "overall_score": 0,
                "status": "unknown",
                "areas": [],
                "recommendations": [],
                "error": "Failed to check compliance"
            }
            
    async def _assess_risk(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk across specified dimensions."""
        assessment_type = parameters["assessment_type"]
        scope = parameters.get("scope", "system")
        threshold = parameters.get("severity_threshold", "medium")
        
        prompt = f"""
Perform risk assessment:
Type: {assessment_type}
Scope: {scope}
Severity threshold: {threshold}

Assess risks and provide detailed analysis.

Return in this format:
{{
    "assessment_id": "risk_123",
    "type": "{assessment_type}",
    "scope": "{scope}",
    "overall_risk_level": "medium",
    "risk_score": 65,
    "risks": [
        {{
            "id": "risk_001",
            "category": "security",
            "description": "Unencrypted data transmission",
            "severity": "high",
            "probability": "medium",
            "impact": "high",
            "risk_score": 8.5,
            "mitigation": "Implement TLS encryption",
            "owner": "security_team"
        }}
    ],
    "summary": {{
        "total_risks": 5,
        "by_severity": {{"critical": 0, "high": 2, "medium": 2, "low": 1}},
        "by_category": {{"security": 3, "operational": 2}}
    }},
    "recommendations": [
        {{"priority": "immediate", "action": "address_high_risks"}}
    ]
}}
"""
        
        response = await self._call_agent(prompt, {"action": "assess_risk"})
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
                
            return {
                "assessed_at": self._get_timestamp(),
                "threshold": threshold,
                **content
            }
        except Exception:
            return {
                "assessment_id": "failed_risk_assessment",
                "type": assessment_type,
                "scope": scope,
                "overall_risk_level": "unknown",
                "risk_score": 0,
                "risks": [],
                "summary": {},
                "recommendations": [],
                "error": "Failed to assess risk"
            }
            
    async def _generate_report(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate audit report."""
        report_type = parameters["report_type"]
        timeframe = parameters.get("timeframe", {})
        format_type = parameters.get("format", "json")
        recipients = parameters.get("recipients", [])
        
        prompt = f"""
Generate {report_type} audit report:
Timeframe: {json.dumps(timeframe) if timeframe else "current period"}
Format: {format_type}
Recipients: {recipients}

Create comprehensive report with executive summary, findings, and recommendations.

Return report structure in this format:
{{
    "report_id": "report_123",
    "type": "{report_type}",
    "generated_at": "2025-01-19T10:00:00Z",
    "timeframe": {timeframe},
    "executive_summary": "High-level summary of findings",
    "sections": [
        {{
            "title": "System Performance",
            "content": "Detailed findings",
            "metrics": {{}},
            "recommendations": []
        }}
    ],
    "key_findings": ["finding 1", "finding 2"],
    "action_items": [
        {{"action": "implement_monitoring", "priority": "high", "owner": "ops_team", "due_date": "2025-02-01"}}
    ],
    "appendices": ["detailed_metrics", "compliance_checklist"]
}}
"""
        
        response = await self._call_agent(prompt, {"action": "generate_report"})
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
                
            return {
                "format": format_type,
                "recipients": recipients,
                **content
            }
        except Exception:
            return {
                "report_id": "failed_report",
                "type": report_type,
                "generated_at": self._get_timestamp(),
                "executive_summary": "Report generation failed",
                "sections": [],
                "key_findings": [],
                "action_items": [],
                "error": "Failed to generate report"
            }
            
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"

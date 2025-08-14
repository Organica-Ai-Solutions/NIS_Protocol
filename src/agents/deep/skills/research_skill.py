"""
Research Skill for Deep Agents

Handles research operations like plan generation, literature search, analysis.
Maps to MCP tools: research.plan, research.search, research.synthesize
"""

from typing import Dict, Any, List
import json

from .base_skill import BaseSkill


class ResearchSkill(BaseSkill):
    """
    Skill for research operations within NIS Protocol.
    
    Provides capabilities for:
    - Generating research plans
    - Searching literature and knowledge bases
    - Synthesizing research findings
    - Managing research workflows
    """
    
    def __init__(self, agent, memory_manager, config=None):
        super().__init__(agent, memory_manager, config)
        
    async def execute(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a research action."""
        if not self._validate_parameters(action, parameters):
            return self._format_error(f"Invalid parameters for action '{action}'", "ValidationError")
            
        try:
            if action == "plan":
                result = await self._create_research_plan(parameters)
            elif action == "search":
                result = await self._search_literature(parameters)
            elif action == "synthesize":
                result = await self._synthesize_findings(parameters)
            elif action == "analyze":
                result = await self._analyze_topic(parameters)
            else:
                return self._format_error(f"Unknown action '{action}'", "ActionError")
                
            await self._store_result(action, parameters, result)
            return self._format_success(result)
            
        except Exception as e:
            return self._format_error(str(e), "ExecutionError")
            
    def get_available_actions(self) -> List[str]:
        """Get available research actions."""
        return ["plan", "search", "synthesize", "analyze"]
        
    def _get_action_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get schemas for research actions."""
        return {
            "plan": {
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
            "search": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "sources": {"type": "array", "items": {"type": "string"}},
                    "timeframe": {"type": "string"},
                    "max_results": {"type": "number", "default": 20}
                },
                "required": ["query"]
            },
            "synthesize": {
                "type": "object",
                "properties": {
                    "findings": {"type": "array", "items": {"type": "object"}},
                    "focus": {"type": "string"},
                    "format": {"type": "string", "enum": ["summary", "detailed", "report"], "default": "summary"}
                },
                "required": ["findings"]
            },
            "analyze": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "analysis_type": {"type": "string", "enum": ["trend", "gap", "impact", "feasibility"]},
                    "context": {"type": "object"}
                },
                "required": ["topic", "analysis_type"]
            }
        }
        
    async def _create_research_plan(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a research plan for a given goal."""
        goal = parameters["goal"]
        constraints = parameters.get("constraints", {})
        depth = parameters.get("depth", "detailed")
        
        prompt = f"""
Create a research plan for this goal: {goal}

Constraints: {json.dumps(constraints, indent=2)}
Depth level: {depth}

Generate a structured research plan with:
1. Research objectives
2. Key research questions
3. Methodology
4. Timeline and milestones
5. Required resources
6. Success criteria

Return in this format:
{{
    "goal": "{goal}",
    "objectives": ["objective 1", "objective 2"],
    "research_questions": ["question 1", "question 2"],
    "methodology": {{
        "approach": "literature review + analysis",
        "steps": ["step 1", "step 2"]
    }},
    "timeline": {{
        "phases": [
            {{"name": "phase 1", "duration": "2 weeks", "deliverables": ["deliverable 1"]}}
        ]
    }},
    "resources": ["resource 1", "resource 2"],
    "success_criteria": ["criteria 1", "criteria 2"]
}}
"""
        
        response = await self._call_agent(prompt, {"action": "create_research_plan"})
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
                
            return {
                "plan_id": f"research_plan_{int(self._get_current_time())}",
                "created_at": self._get_timestamp(),
                "depth": depth,
                **content
            }
        except Exception:
            return {
                "goal": goal,
                "objectives": [],
                "research_questions": [],
                "methodology": {},
                "timeline": {},
                "resources": [],
                "success_criteria": [],
                "error": "Failed to generate research plan"
            }
            
    async def _search_literature(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Search literature and knowledge sources."""
        query = parameters["query"]
        sources = parameters.get("sources", [])
        timeframe = parameters.get("timeframe")
        max_results = parameters.get("max_results", 20)
        
        prompt = f"""
Search for literature and knowledge on: {query}

Sources to search: {sources if sources else "all available"}
Timeframe: {timeframe if timeframe else "any"}
Max results: {max_results}

Return search results in this format:
{{
    "query": "{query}",
    "results": [
        {{
            "title": "Paper Title",
            "authors": ["Author 1", "Author 2"],
            "source": "Journal/Conference",
            "year": 2024,
            "abstract": "Abstract text",
            "relevance_score": 0.95,
            "key_findings": ["finding 1", "finding 2"],
            "url": "https://example.com"
        }}
    ],
    "total_found": 50,
    "search_metadata": {{
        "sources_searched": ["source1", "source2"],
        "search_time": "2.3s"
    }}
}}
"""
        
        response = await self._call_agent(prompt, {"action": "search_literature"})
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
                
            return content
        except Exception:
            return {
                "query": query,
                "results": [],
                "total_found": 0,
                "search_metadata": {},
                "error": "Failed to search literature"
            }
            
    async def _synthesize_findings(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize research findings."""
        findings = parameters["findings"]
        focus = parameters.get("focus", "")
        format_type = parameters.get("format", "summary")
        
        prompt = f"""
Synthesize these research findings:
{json.dumps(findings, indent=2)}

Focus area: {focus if focus else "general synthesis"}
Output format: {format_type}

Create a synthesis that:
1. Identifies key themes and patterns
2. Highlights important insights
3. Notes contradictions or gaps
4. Provides actionable conclusions

Return in this format:
{{
    "synthesis_id": "synthesis_123",
    "key_themes": ["theme 1", "theme 2"],
    "insights": [
        {{"insight": "insight text", "supporting_evidence": ["evidence 1"]}}
    ],
    "contradictions": ["contradiction 1"],
    "gaps": ["gap 1"],
    "conclusions": ["conclusion 1"],
    "recommendations": ["recommendation 1"]
}}
"""
        
        response = await self._call_agent(prompt, {"action": "synthesize_findings"})
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
                
            return {
                "created_at": self._get_timestamp(),
                "focus": focus,
                "format": format_type,
                "findings_count": len(findings),
                **content
            }
        except Exception:
            return {
                "synthesis_id": "failed_synthesis",
                "key_themes": [],
                "insights": [],
                "contradictions": [],
                "gaps": [],
                "conclusions": [],
                "recommendations": [],
                "error": "Failed to synthesize findings"
            }
            
    async def _analyze_topic(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a research topic."""
        topic = parameters["topic"]
        analysis_type = parameters["analysis_type"]
        context = parameters.get("context", {})
        
        prompt = f"""
Perform {analysis_type} analysis on topic: {topic}

Context: {json.dumps(context, indent=2)}

Analysis type: {analysis_type}
- trend: Analyze current trends and future directions
- gap: Identify research gaps and opportunities
- impact: Assess potential impact and implications
- feasibility: Evaluate feasibility and challenges

Return analysis results appropriate for the analysis type.
"""
        
        response = await self._call_agent(prompt, {"action": "analyze_topic"})
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
                
            return {
                "topic": topic,
                "analysis_type": analysis_type,
                "context": context,
                "results": content,
                "timestamp": self._get_timestamp()
            }
        except Exception:
            return {
                "topic": topic,
                "analysis_type": analysis_type,
                "results": {},
                "error": "Failed to perform analysis"
            }
            
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"
        
    def _get_current_time(self) -> float:
        """Get current time as float."""
        import time
        return time.time()

"""
Code Skill for Deep Agents

Handles code operations like edit, review, analyze, generate.
Maps to MCP tools: code.edit, code.review, code.analyze
"""

from typing import Dict, Any, List
import json

from .base_skill import BaseSkill


class CodeSkill(BaseSkill):
    """
    Skill for code operations within NIS Protocol.
    
    Provides capabilities for:
    - Editing and modifying code files
    - Code review and quality analysis
    - Code generation and templating
    - Refactoring and optimization
    """
    
    def __init__(self, agent, memory_manager, config=None):
        super().__init__(agent, memory_manager, config)
        
    async def execute(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a code action."""
        if not self._validate_parameters(action, parameters):
            return self._format_error(f"Invalid parameters for action '{action}'", "ValidationError")
            
        try:
            if action == "edit":
                result = await self._edit_code(parameters)
            elif action == "review":
                result = await self._review_code(parameters)
            elif action == "analyze":
                result = await self._analyze_code(parameters)
            elif action == "generate":
                result = await self._generate_code(parameters)
            elif action == "refactor":
                result = await self._refactor_code(parameters)
            else:
                return self._format_error(f"Unknown action '{action}'", "ActionError")
                
            await self._store_result(action, parameters, result)
            return self._format_success(result)
            
        except Exception as e:
            return self._format_error(str(e), "ExecutionError")
            
    def get_available_actions(self) -> List[str]:
        """Get available code actions."""
        return ["edit", "review", "analyze", "generate", "refactor"]
        
    def _get_action_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get schemas for code actions."""
        return {
            "edit": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to edit"},
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
                    "reason": {"type": "string", "description": "Reason for the changes"}
                },
                "required": ["file_path", "changes"]
            },
            "review": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to file to review"},
                    "code_content": {"type": "string", "description": "Code content to review"},
                    "review_type": {"type": "string", "enum": ["security", "performance", "quality", "style", "comprehensive"]},
                    "standards": {"type": "array", "items": {"type": "string"}}
                }
            },
            "analyze": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "File path or code snippet to analyze"},
                    "analysis_type": {"type": "string", "enum": ["complexity", "dependencies", "patterns", "metrics"]},
                    "language": {"type": "string", "description": "Programming language"}
                },
                "required": ["target", "analysis_type"]
            },
            "generate": {
                "type": "object",
                "properties": {
                    "specification": {"type": "string", "description": "What to generate"},
                    "language": {"type": "string", "description": "Programming language"},
                    "template": {"type": "string", "description": "Template or pattern to follow"},
                    "constraints": {"type": "object"}
                },
                "required": ["specification", "language"]
            },
            "refactor": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "refactor_type": {"type": "string", "enum": ["extract_method", "rename", "optimize", "modernize"]},
                    "target_element": {"type": "string", "description": "What to refactor"},
                    "options": {"type": "object"}
                },
                "required": ["file_path", "refactor_type"]
            }
        }
        
    async def _edit_code(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Edit code with specified changes."""
        file_path = parameters["file_path"]
        changes = parameters["changes"]
        reason = parameters.get("reason", "Code modification")
        
        prompt = f"""
Edit code file: {file_path}
Reason: {reason}

Changes to apply:
{json.dumps(changes, indent=2)}

Generate a diff view and validate the changes.

Return in this format:
{{
    "edit_id": "edit_123",
    "file_path": "{file_path}",
    "changes_applied": {len(changes)},
    "diff": [
        {{
            "change_id": "change_1",
            "type": "modification",
            "line_start": 10,
            "line_end": 12,
            "old_content": "old code",
            "new_content": "new code",
            "status": "success"
        }}
    ],
    "validation": {{
        "syntax_valid": true,
        "warnings": [],
        "errors": []
    }},
    "backup_created": true,
    "backup_path": "/backup/file.py.bak"
}}
"""
        
        response = await self._call_agent(prompt, {"action": "edit_code"})
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
                
            return {
                "timestamp": self._get_timestamp(),
                "reason": reason,
                **content
            }
        except Exception:
            return {
                "edit_id": "failed_edit",
                "file_path": file_path,
                "changes_applied": 0,
                "diff": [],
                "validation": {"syntax_valid": False, "errors": ["Edit failed"]},
                "error": "Failed to edit code"
            }
            
    async def _review_code(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Review code for quality, security, and best practices."""
        file_path = parameters.get("file_path")
        code_content = parameters.get("code_content")
        review_type = parameters.get("review_type", "comprehensive")
        standards = parameters.get("standards", [])
        
        prompt = f"""
Perform {review_type} code review:
File: {file_path if file_path else "inline code"}
Standards: {standards if standards else "general best practices"}

Code to review:
{code_content if code_content else "[file content]"}

Provide detailed review covering:
1. Code quality and maintainability
2. Security considerations
3. Performance implications
4. Best practice adherence
5. Potential improvements

Return in this format:
{{
    "review_id": "review_123",
    "file_path": "{file_path}",
    "review_type": "{review_type}",
    "overall_score": 8.5,
    "overall_grade": "B+",
    "categories": {{
        "quality": {{"score": 8, "issues": ["long_method"]}},
        "security": {{"score": 9, "issues": []}},
        "performance": {{"score": 7, "issues": ["inefficient_loop"]}},
        "style": {{"score": 9, "issues": ["minor_naming"]}}
    }},
    "issues": [
        {{
            "id": "issue_1",
            "category": "quality",
            "severity": "medium",
            "line": 45,
            "description": "Method too long, consider breaking down",
            "suggestion": "Extract helper methods"
        }}
    ],
    "recommendations": [
        {{"priority": "high", "action": "refactor_long_method", "rationale": "Improve maintainability"}}
    ],
    "summary": "Good code overall with minor improvements needed"
}}
"""
        
        response = await self._call_agent(prompt, {"action": "review_code"})
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
                
            return {
                "reviewed_at": self._get_timestamp(),
                "standards": standards,
                **content
            }
        except Exception:
            return {
                "review_id": "failed_review",
                "file_path": file_path,
                "review_type": review_type,
                "overall_score": 0,
                "categories": {},
                "issues": [],
                "recommendations": [],
                "summary": "Review failed",
                "error": "Failed to review code"
            }
            
    async def _analyze_code(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code structure and metrics."""
        target = parameters["target"]
        analysis_type = parameters["analysis_type"]
        language = parameters.get("language", "python")
        
        prompt = f"""
Perform {analysis_type} analysis on: {target}
Language: {language}

Analysis type: {analysis_type}
- complexity: Cyclomatic complexity, nesting levels
- dependencies: Import analysis, coupling metrics
- patterns: Design patterns, anti-patterns
- metrics: Lines of code, function sizes, etc.

Return detailed analysis results.

Return in this format:
{{
    "analysis_id": "analysis_123",
    "target": "{target}",
    "analysis_type": "{analysis_type}",
    "language": "{language}",
    "results": {{
        "complexity": {{"cyclomatic": 5, "cognitive": 8}},
        "metrics": {{"lines_of_code": 150, "functions": 8, "classes": 2}},
        "dependencies": {{"imports": ["os", "json"], "coupling": "low"}},
        "patterns": {{"detected": ["singleton"], "anti_patterns": ["god_object"]}}
    }},
    "insights": [
        {{"category": "complexity", "message": "Function has high complexity", "severity": "medium"}}
    ],
    "recommendations": [
        {{"action": "reduce_complexity", "target": "main_function", "priority": "medium"}}
    ]
}}
"""
        
        response = await self._call_agent(prompt, {"action": "analyze_code"})
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
                
            return {
                "analyzed_at": self._get_timestamp(),
                **content
            }
        except Exception:
            return {
                "analysis_id": "failed_analysis",
                "target": target,
                "analysis_type": analysis_type,
                "language": language,
                "results": {},
                "insights": [],
                "recommendations": [],
                "error": "Failed to analyze code"
            }
            
    async def _generate_code(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code based on specification."""
        specification = parameters["specification"]
        language = parameters["language"]
        template = parameters.get("template")
        constraints = parameters.get("constraints", {})
        
        prompt = f"""
Generate {language} code for: {specification}

Template/Pattern: {template if template else "standard"}
Constraints: {json.dumps(constraints, indent=2) if constraints else "none"}

Generate clean, well-documented code that meets the specification.

Return in this format:
{{
    "generation_id": "gen_123",
    "specification": "{specification}",
    "language": "{language}",
    "code": "generated code here",
    "structure": {{
        "functions": ["func1", "func2"],
        "classes": ["Class1"],
        "imports": ["import1", "import2"]
    }},
    "documentation": {{
        "docstrings": true,
        "comments": true,
        "readme": "README content if applicable"
    }},
    "testing": {{
        "test_cases": ["test case 1", "test case 2"],
        "test_code": "test code if requested"
    }},
    "metadata": {{
        "lines_of_code": 45,
        "complexity": "low",
        "maintainability": "high"
    }}
}}
"""
        
        response = await self._call_agent(prompt, {"action": "generate_code"})
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
                
            return {
                "generated_at": self._get_timestamp(),
                "template": template,
                "constraints": constraints,
                **content
            }
        except Exception:
            return {
                "generation_id": "failed_generation",
                "specification": specification,
                "language": language,
                "code": "",
                "structure": {},
                "documentation": {},
                "testing": {},
                "metadata": {},
                "error": "Failed to generate code"
            }
            
    async def _refactor_code(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Refactor code for better structure and maintainability."""
        file_path = parameters["file_path"]
        refactor_type = parameters["refactor_type"]
        target_element = parameters.get("target_element")
        options = parameters.get("options", {})
        
        prompt = f"""
Refactor code file: {file_path}
Refactor type: {refactor_type}
Target element: {target_element if target_element else "entire file"}
Options: {json.dumps(options, indent=2) if options else "default"}

Perform the specified refactoring while maintaining functionality.

Return in this format:
{{
    "refactor_id": "refactor_123",
    "file_path": "{file_path}",
    "refactor_type": "{refactor_type}",
    "target_element": "{target_element}",
    "changes": [
        {{
            "type": "extract_method",
            "old_code": "original code",
            "new_code": "refactored code",
            "description": "Extracted method for clarity"
        }}
    ],
    "improvements": {{
        "complexity_reduction": "15%",
        "readability_score": "+2",
        "maintainability": "improved"
    }},
    "tests_affected": ["test_function_1"],
    "validation": {{
        "syntax_valid": true,
        "functionality_preserved": true,
        "performance_impact": "none"
    }},
    "summary": "Successfully refactored with improved maintainability"
}}
"""
        
        response = await self._call_agent(prompt, {"action": "refactor_code"})
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
                
            return {
                "refactored_at": self._get_timestamp(),
                "options": options,
                **content
            }
        except Exception:
            return {
                "refactor_id": "failed_refactor",
                "file_path": file_path,
                "refactor_type": refactor_type,
                "target_element": target_element,
                "changes": [],
                "improvements": {},
                "validation": {"syntax_valid": False},
                "summary": "Refactoring failed",
                "error": "Failed to refactor code"
            }
            
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"

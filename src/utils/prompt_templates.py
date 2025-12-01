"""
Prompt Templates - NIS Protocol v4.0
Pre-built prompts for common tasks. Reduces boilerplate and improves consistency.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class TemplateCategory(Enum):
    """Template categories"""
    ANALYSIS = "analysis"
    CODING = "coding"
    WRITING = "writing"
    RESEARCH = "research"
    ROBOTICS = "robotics"
    PHYSICS = "physics"
    CREATIVE = "creative"
    BUSINESS = "business"


@dataclass
class PromptTemplate:
    """A reusable prompt template"""
    id: str
    name: str
    description: str
    category: TemplateCategory
    template: str
    variables: List[str]  # Required variables like {topic}, {code}
    examples: List[Dict[str, str]] = field(default_factory=list)
    recommended_model: Optional[str] = None
    estimated_tokens: int = 500


# =============================================================================
# BUILT-IN TEMPLATES
# =============================================================================

TEMPLATES: Dict[str, PromptTemplate] = {
    
    # --- ANALYSIS ---
    "analyze_code": PromptTemplate(
        id="analyze_code",
        name="Code Analysis",
        description="Analyze code for bugs, improvements, and best practices",
        category=TemplateCategory.CODING,
        template="""Analyze the following {language} code:

```{language}
{code}
```

Provide:
1. **Summary**: What does this code do?
2. **Issues**: Any bugs or potential problems?
3. **Improvements**: How can it be better?
4. **Security**: Any security concerns?

Be concise and specific.""",
        variables=["language", "code"],
        examples=[{"language": "python", "code": "def add(a, b): return a + b"}],
        estimated_tokens=800
    ),
    
    "explain_concept": PromptTemplate(
        id="explain_concept",
        name="Explain Concept",
        description="Explain a technical concept clearly",
        category=TemplateCategory.ANALYSIS,
        template="""Explain {concept} in simple terms.

Target audience: {audience}

Include:
1. What it is (1-2 sentences)
2. Why it matters
3. A simple example
4. Common misconceptions

Keep it under 200 words.""",
        variables=["concept", "audience"],
        examples=[{"concept": "neural networks", "audience": "beginners"}],
        estimated_tokens=400
    ),
    
    "compare_options": PromptTemplate(
        id="compare_options",
        name="Compare Options",
        description="Compare multiple options objectively",
        category=TemplateCategory.ANALYSIS,
        template="""Compare these options for {use_case}:

Options: {options}

For each option, evaluate:
- Pros
- Cons
- Best for (use case)
- Cost/effort

End with a recommendation based on: {criteria}""",
        variables=["use_case", "options", "criteria"],
        estimated_tokens=600
    ),
    
    # --- CODING ---
    "write_function": PromptTemplate(
        id="write_function",
        name="Write Function",
        description="Generate a function with tests",
        category=TemplateCategory.CODING,
        template="""Write a {language} function that {description}.

Requirements:
{requirements}

Include:
1. The function with docstring
2. Type hints (if applicable)
3. 2-3 test cases
4. Edge case handling

Keep it clean and idiomatic.""",
        variables=["language", "description", "requirements"],
        estimated_tokens=600
    ),
    
    "debug_error": PromptTemplate(
        id="debug_error",
        name="Debug Error",
        description="Help debug an error",
        category=TemplateCategory.CODING,
        template="""I'm getting this error:

```
{error}
```

In this code:
```{language}
{code}
```

What I was trying to do: {intent}

Please:
1. Explain why this error occurs
2. Show the fix
3. Explain how to prevent it""",
        variables=["error", "language", "code", "intent"],
        estimated_tokens=500
    ),
    
    "refactor_code": PromptTemplate(
        id="refactor_code",
        name="Refactor Code",
        description="Refactor code for better quality",
        category=TemplateCategory.CODING,
        template="""Refactor this {language} code to be more {goal}:

```{language}
{code}
```

Focus on:
- {goal}
- Maintaining functionality
- Clear naming
- Proper structure

Show the refactored code with brief explanations of changes.""",
        variables=["language", "code", "goal"],
        examples=[{"goal": "readable"}, {"goal": "performant"}, {"goal": "testable"}],
        estimated_tokens=700
    ),
    
    # --- WRITING ---
    "summarize": PromptTemplate(
        id="summarize",
        name="Summarize Text",
        description="Summarize text to key points",
        category=TemplateCategory.WRITING,
        template="""Summarize the following in {length}:

{text}

Format: {format}""",
        variables=["text", "length", "format"],
        examples=[
            {"length": "3 bullet points", "format": "bullet points"},
            {"length": "one paragraph", "format": "prose"},
            {"length": "5 sentences", "format": "numbered list"}
        ],
        estimated_tokens=300
    ),
    
    "write_email": PromptTemplate(
        id="write_email",
        name="Write Email",
        description="Draft a professional email",
        category=TemplateCategory.WRITING,
        template="""Write a {tone} email about: {subject}

Key points to include:
{points}

Recipient: {recipient}
Goal: {goal}

Keep it concise and professional.""",
        variables=["tone", "subject", "points", "recipient", "goal"],
        examples=[{"tone": "professional"}, {"tone": "friendly"}, {"tone": "urgent"}],
        estimated_tokens=400
    ),
    
    # --- RESEARCH ---
    "research_topic": PromptTemplate(
        id="research_topic",
        name="Research Topic",
        description="Research a topic comprehensively",
        category=TemplateCategory.RESEARCH,
        template="""Research: {topic}

Provide:
1. **Overview**: What is it?
2. **Key Facts**: 5 important facts
3. **Current State**: Latest developments
4. **Challenges**: Main problems/debates
5. **Resources**: Where to learn more

Focus on accuracy. Cite sources if known.""",
        variables=["topic"],
        estimated_tokens=800
    ),
    
    # --- ROBOTICS ---
    "plan_trajectory": PromptTemplate(
        id="plan_trajectory",
        name="Plan Robot Trajectory",
        description="Plan a physics-validated robot trajectory",
        category=TemplateCategory.ROBOTICS,
        template="""Plan a trajectory for {robot_type} to {task}.

Constraints:
- Start: {start_position}
- End: {end_position}
- Obstacles: {obstacles}
- Max velocity: {max_velocity}

Provide:
1. Waypoints (x, y, z coordinates)
2. Timing for each segment
3. Safety considerations
4. Physics validation notes""",
        variables=["robot_type", "task", "start_position", "end_position", "obstacles", "max_velocity"],
        estimated_tokens=600
    ),
    
    # --- PHYSICS ---
    "validate_physics": PromptTemplate(
        id="validate_physics",
        name="Validate Physics",
        description="Check if a scenario obeys physics laws",
        category=TemplateCategory.PHYSICS,
        template="""Validate the physics of this scenario:

{scenario}

Check for:
1. Conservation of energy
2. Conservation of momentum
3. Newton's laws compliance
4. Thermodynamic feasibility

Identify any violations and suggest corrections.""",
        variables=["scenario"],
        estimated_tokens=500
    ),
    
    # --- CREATIVE ---
    "brainstorm": PromptTemplate(
        id="brainstorm",
        name="Brainstorm Ideas",
        description="Generate creative ideas",
        category=TemplateCategory.CREATIVE,
        template="""Brainstorm {count} ideas for: {challenge}

Context: {context}

Constraints: {constraints}

For each idea:
- Name it
- Describe it (1-2 sentences)
- Rate feasibility (1-5)

Be creative but practical.""",
        variables=["count", "challenge", "context", "constraints"],
        estimated_tokens=600
    ),
    
    # --- BUSINESS ---
    "analyze_decision": PromptTemplate(
        id="analyze_decision",
        name="Analyze Decision",
        description="Analyze a business decision",
        category=TemplateCategory.BUSINESS,
        template="""Analyze this decision: {decision}

Context: {context}

Consider:
1. **Pros**: Benefits and opportunities
2. **Cons**: Risks and downsides
3. **Alternatives**: Other options
4. **Recommendation**: What would you do?

Be objective and data-driven.""",
        variables=["decision", "context"],
        estimated_tokens=500
    ),
}


class PromptTemplateManager:
    """Manage and use prompt templates"""
    
    def __init__(self, custom_templates_path: Optional[str] = None):
        self.templates = TEMPLATES.copy()
        
        # Load custom templates
        if custom_templates_path:
            self._load_custom(Path(custom_templates_path))
    
    def _load_custom(self, path: Path):
        """Load custom templates from JSON file"""
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    for t_data in data:
                        t_data["category"] = TemplateCategory(t_data["category"])
                        template = PromptTemplate(**t_data)
                        self.templates[template.id] = template
            except Exception:
                pass
    
    def get(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a template by ID"""
        return self.templates.get(template_id)
    
    def list_all(self) -> List[Dict[str, Any]]:
        """List all available templates"""
        return [
            {
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "category": t.category.value,
                "variables": t.variables
            }
            for t in self.templates.values()
        ]
    
    def list_by_category(self, category: TemplateCategory) -> List[PromptTemplate]:
        """List templates in a category"""
        return [t for t in self.templates.values() if t.category == category]
    
    def render(self, template_id: str, **variables) -> str:
        """
        Render a template with variables.
        
        Args:
            template_id: The template ID
            **variables: Variable values to fill in
            
        Returns:
            Rendered prompt string
            
        Raises:
            ValueError: If template not found or missing variables
        """
        template = self.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        # Check required variables
        missing = [v for v in template.variables if v not in variables]
        if missing:
            raise ValueError(f"Missing variables: {missing}")
        
        # Render
        try:
            return template.template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Variable error: {e}")
    
    def add_template(self, template: PromptTemplate):
        """Add a custom template"""
        self.templates[template.id] = template
    
    def search(self, query: str) -> List[PromptTemplate]:
        """Search templates by name or description"""
        query = query.lower()
        return [
            t for t in self.templates.values()
            if query in t.name.lower() or query in t.description.lower()
        ]


# Global instance
_template_manager: Optional[PromptTemplateManager] = None


def get_template_manager() -> PromptTemplateManager:
    """Get or create the global template manager"""
    global _template_manager
    if _template_manager is None:
        _template_manager = PromptTemplateManager()
    return _template_manager

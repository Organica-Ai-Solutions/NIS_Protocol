#!/usr/bin/env python3
"""
LLM-Powered Planning Agent for NIS Protocol
Replaces keyword heuristics with intelligent goal decomposition

Copyright 2025 Organica AI Solutions
Licensed under Apache License 2.0
"""

import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from src.core.multi_provider_strategy import get_multi_provider_strategy
from src.core.predict_prefetch import get_predict_prefetch_engine

logger = logging.getLogger(__name__)


@dataclass
class PlanStep:
    """Represents a single step in an execution plan."""
    agent_type: str  # research, physics, robotics, vision, multi_agent
    description: str
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = None
    dependencies: List[int] = None  # Indices of steps this depends on
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ExecutionPlan:
    """Complete execution plan for a goal."""
    goal: str
    steps: List[PlanStep]
    estimated_duration: float  # seconds
    confidence: float  # 0.0 to 1.0
    reasoning: str


class LLMPlanner:
    """
    LLM-powered planning agent that decomposes goals into executable steps.
    
    Uses LLM to:
    1. Analyze goal and context
    2. Decompose into atomic tasks
    3. Select appropriate agents and tools
    4. Determine execution order and dependencies
    5. Generate executable plan
    """
    
    def __init__(self, llm_provider=None, mcp_executor=None):
        """Initialize LLM planner with provider."""
        self.llm_provider = llm_provider
        self.multi_provider = get_multi_provider_strategy(llm_provider) if llm_provider else None
        self.prefetch_engine = None
        if mcp_executor:
            self.prefetch_engine = get_predict_prefetch_engine(mcp_executor)
        self.available_agents = {
            "research": {
                "description": "Web search, information gathering, fact checking, file operations",
                "tools": ["web_search", "memory_store", "memory_retrieve", "code_execute", "file_read", "file_write", "file_list"],
                "capabilities": ["search web", "gather information", "validate claims", "analyze data", "read files", "write files", "create documents"]
            },
            "physics": {
                "description": "Solve physics equations using neural networks (PINNs)",
                "tools": ["physics_solve", "memory_store", "code_execute"],
                "capabilities": ["solve PDEs", "heat equation", "wave equation", "laplace equation"]
            },
            "robotics": {
                "description": "Compute kinematics, plan trajectories, motion planning",
                "tools": ["robotics_kinematics", "physics_solve", "memory_store"],
                "capabilities": ["forward kinematics", "inverse kinematics", "trajectory planning", "motion validation"]
            },
            "vision": {
                "description": "Analyze images, object detection, visual understanding",
                "tools": ["vision_analyze", "web_search", "memory_store"],
                "capabilities": ["image analysis", "object detection", "visual reasoning", "context research"]
            }
        }
        
        self.available_tools = [
            "code_execute", "web_search", "physics_solve", 
            "robotics_kinematics", "vision_analyze",
            "memory_store", "memory_retrieve",
            "consciousness_genesis", "llm_chat",
            "file_read", "file_write", "file_list", "file_exists"
        ]
    
    async def create_plan(self, goal: str, context: Optional[Dict[str, Any]] = None, enable_prefetch: bool = True) -> ExecutionPlan:
        """
        Create an execution plan for the given goal using LLM.
        
        Args:
            goal: User's goal/objective
            context: Optional context (previous results, constraints, etc.)
            enable_prefetch: Enable predict-and-prefetch optimization
        
        Returns:
            ExecutionPlan with steps, reasoning, and confidence
        """
        logger.info(f"🎯 Creating LLM-powered plan for: {goal}")
        
        # Use predict-and-prefetch if enabled
        if enable_prefetch and self.prefetch_engine:
            plan, prefetch_results = await self.prefetch_engine.plan_with_prefetch(
                llm_planner=self,
                goal=goal,
                context=context
            )
            logger.info(f"✅ Generated plan with {len(plan.steps)} steps (confidence: {plan.confidence:.2f}) + {len(prefetch_results)} prefetches")
            return plan
        
        # Standard planning without prefetch
        # Build planning prompt
        prompt = self._build_planning_prompt(goal, context)
        
        # Call LLM for plan generation
        try:
            response = await self._call_llm_for_planning(prompt)
            plan = self._parse_llm_response(response, goal)
            
            logger.info(f"✅ Generated plan with {len(plan.steps)} steps (confidence: {plan.confidence:.2f})")
            return plan
            
        except Exception as e:
            logger.error(f"LLM planning failed: {e}, falling back to heuristic")
            return self._fallback_heuristic_plan(goal, context)
    
    def _build_planning_prompt(self, goal: str, context: Optional[Dict[str, Any]]) -> str:
        """Build the planning prompt for the LLM."""
        
        # Format available agents
        agents_desc = "\n".join([
            f"- **{name}**: {info['description']}\n  Tools: {', '.join(info['tools'])}\n  Capabilities: {', '.join(info['capabilities'])}"
            for name, info in self.available_agents.items()
        ])
        
        # Format available tools
        tools_desc = ", ".join(self.available_tools)
        
        prompt = f"""You are an intelligent task planner for an autonomous agent system.

**Goal**: {goal}

**Available Agents**:
{agents_desc}

**Available Tools**: {tools_desc}

**Your Task**: Decompose the goal into a sequence of executable steps. Each step should:
1. Use ONE specific agent type (research, physics, robotics, vision)
2. Have a clear, actionable description
3. Specify which tool(s) to use
4. List any dependencies on previous steps

**Output Format** (JSON):
{{
    "reasoning": "Brief explanation of your planning approach",
    "confidence": 0.95,
    "estimated_duration": 15.0,
    "steps": [
        {{
            "agent_type": "research",
            "description": "Search for information about X",
            "tool_name": "web_search",
            "parameters": {{"query": "X"}},
            "dependencies": []
        }},
        {{
            "agent_type": "physics",
            "description": "Solve equation Y using results from step 1",
            "tool_name": "physics_solve",
            "parameters": {{"equation_type": "heat-equation"}},
            "dependencies": [0]
        }}
    ]
}}

**Few-Shot Examples**:

Example 1 - Goal: "Research quantum computing and solve a wave equation"
{{
    "reasoning": "First gather information about quantum computing, then solve the wave equation which is relevant to quantum mechanics",
    "confidence": 0.90,
    "estimated_duration": 20.0,
    "steps": [
        {{
            "agent_type": "research",
            "description": "Search for latest quantum computing developments",
            "tool_name": "web_search",
            "parameters": {{"query": "quantum computing 2025"}},
            "dependencies": []
        }},
        {{
            "agent_type": "physics",
            "description": "Solve wave equation relevant to quantum systems",
            "tool_name": "physics_solve",
            "parameters": {{"equation_type": "wave-equation", "domain_size": 10, "time_steps": 5}},
            "dependencies": [0]
        }},
        {{
            "agent_type": "research",
            "description": "Store combined results",
            "tool_name": "memory_store",
            "parameters": {{"key": "quantum_research_results"}},
            "dependencies": [0, 1]
        }}
    ]
}}

Example 2 - Goal: "Plan robot motion to target position"
{{
    "reasoning": "Compute inverse kinematics to find joint angles, then plan trajectory",
    "confidence": 0.95,
    "estimated_duration": 10.0,
    "steps": [
        {{
            "agent_type": "robotics",
            "description": "Compute inverse kinematics for target position",
            "tool_name": "robotics_kinematics",
            "parameters": {{"operation": "inverse_kinematics", "target_position": [1.0, 0.5, 0.3]}},
            "dependencies": []
        }},
        {{
            "agent_type": "robotics",
            "description": "Store motion plan",
            "tool_name": "memory_store",
            "parameters": {{"key": "robot_motion_plan"}},
            "dependencies": [0]
        }}
    ]
}}

Now create a plan for the given goal. Output ONLY valid JSON, no other text.
"""
        
        return prompt
    
    async def _call_llm_for_planning(self, prompt: str) -> str:
        """Call LLM provider to generate plan with multi-provider fallback."""
        if not self.llm_provider:
            raise Exception("No LLM provider configured")
        
        messages = [
            {"role": "system", "content": "You are an expert task planner. Output only valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        # Use multi-provider strategy with automatic fallback
        if self.multi_provider:
            response, provider_used = await self.multi_provider.call_with_fallback(
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
                strategy="round_robin"  # Rotate through all providers
            )
            
            if response:
                logger.info(f"✅ Plan generated using provider: {provider_used}")
                return response.get("content", "")
            else:
                raise Exception("All LLM providers failed")
        else:
            # Fallback to single provider
            response = await self.llm_provider.generate_response(
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            return response.get("content", "")
    
    def _parse_llm_response(self, response: str, goal: str) -> ExecutionPlan:
        """Parse LLM JSON response into ExecutionPlan."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code blocks
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])
            
            data = json.loads(response)
            
            # Parse steps
            steps = []
            for step_data in data.get("steps", []):
                step = PlanStep(
                    agent_type=step_data["agent_type"],
                    description=step_data["description"],
                    tool_name=step_data.get("tool_name"),
                    parameters=step_data.get("parameters", {}),
                    dependencies=step_data.get("dependencies", [])
                )
                steps.append(step)
            
            return ExecutionPlan(
                goal=goal,
                steps=steps,
                estimated_duration=data.get("estimated_duration", 30.0),
                confidence=data.get("confidence", 0.8),
                reasoning=data.get("reasoning", "LLM-generated plan")
            )
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise
    
    def _fallback_heuristic_plan(self, goal: str, context: Optional[Dict[str, Any]]) -> ExecutionPlan:
        """Fallback to simple heuristic planning if LLM fails."""
        logger.info("Using fallback heuristic planning")
        
        goal_lower = goal.lower()
        steps = []
        
        # Simple keyword-based heuristics
        if "research" in goal_lower or "search" in goal_lower or "find" in goal_lower:
            steps.append(PlanStep(
                agent_type="research",
                description=f"Research: {goal}",
                tool_name="web_search",
                parameters={"query": goal}
            ))
        
        if "physics" in goal_lower or "equation" in goal_lower or "solve" in goal_lower:
            equation_type = "heat-equation"
            if "wave" in goal_lower:
                equation_type = "wave-equation"
            elif "laplace" in goal_lower:
                equation_type = "laplace"
            
            steps.append(PlanStep(
                agent_type="physics",
                description=f"Solve {equation_type}",
                tool_name="physics_solve",
                parameters={"equation_type": equation_type, "parameters": {"domain_size": 10, "time_steps": 5}}
            ))
        
        if "robot" in goal_lower or "motion" in goal_lower or "kinematics" in goal_lower:
            steps.append(PlanStep(
                agent_type="robotics",
                description="Plan robot motion",
                tool_name="robotics_kinematics",
                parameters={"operation": "forward_kinematics"}
            ))
        
        if "image" in goal_lower or "vision" in goal_lower or "analyze" in goal_lower:
            steps.append(PlanStep(
                agent_type="vision",
                description="Analyze image",
                tool_name="vision_analyze",
                parameters={}
            ))
        
        # Default to research if no keywords matched
        if not steps:
            steps.append(PlanStep(
                agent_type="research",
                description=f"Research: {goal}",
                tool_name="web_search",
                parameters={"query": goal}
            ))
        
        return ExecutionPlan(
            goal=goal,
            steps=steps,
            estimated_duration=len(steps) * 10.0,
            confidence=0.6,  # Lower confidence for heuristic
            reasoning="Fallback heuristic planning (keyword-based)"
        )


# Global instance
_llm_planner: Optional[LLMPlanner] = None


def get_llm_planner(llm_provider=None, mcp_executor=None) -> LLMPlanner:
    """Get or create LLM planner instance."""
    global _llm_planner
    if _llm_planner is None:
        _llm_planner = LLMPlanner(llm_provider=llm_provider, mcp_executor=mcp_executor)
    return _llm_planner

"""
Deep Agent Planner

Core planner for multi-step reasoning and sub-agent orchestration.
Handles complex workflows by breaking them down into manageable steps
and delegating to specialized skills/sub-agents.
"""

import json
import time
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

from ...core.agent import NISAgent
from ...memory.memory_manager import MemoryManager


class PlanStepStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """Represents a single step in an execution plan."""
    id: str
    skill: str
    action: str
    parameters: Dict[str, Any]
    description: str
    status: PlanStepStatus = PlanStepStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    dependencies: List[str] = None
    max_retries: int = 3
    retry_count: int = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ExecutionPlan:
    """Represents a complete execution plan."""
    id: str
    goal: str
    context: Dict[str, Any]
    steps: List[PlanStep]
    status: str = "pending"
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.metadata is None:
            self.metadata = {}


class DeepAgentPlanner:
    """
    Deep Agent Planner for complex multi-step workflows.
    
    This planner breaks down complex goals into executable steps,
    manages dependencies, handles retries, and coordinates with
    specialized skill agents.
    """
    
    def __init__(self, agent: NISAgent, memory_manager: MemoryManager):
        self.agent = agent
        self.memory = memory_manager
        self.skills = {}
        self.active_plans = {}
        
    def register_skill(self, name: str, skill_instance):
        """Register a skill/sub-agent with the planner."""
        self.skills[name] = skill_instance
        
    async def create_plan(self, goal: str, context: Dict[str, Any] = None) -> ExecutionPlan:
        """
        Create an execution plan for a given goal.
        
        Args:
            goal: The high-level goal to achieve
            context: Additional context for planning
            
        Returns:
            ExecutionPlan with steps and dependencies
        """
        context = context or {}
        plan_id = f"plan_{int(time.time() * 1000)}"
        
        # Use the agent to generate a plan
        planning_prompt = self._build_planning_prompt(goal, context)
        response = await self.agent.process_request({
            "action": "plan",
            "data": {"prompt": planning_prompt, "goal": goal, "context": context}
        })
        
        # Parse the plan from the response
        steps = self._parse_plan_response(response)
        
        plan = ExecutionPlan(
            id=plan_id,
            goal=goal,
            context=context,
            steps=steps,
            metadata={"created_by": "deep_agent_planner"}
        )
        
        self.active_plans[plan_id] = plan
        
        # Store in memory for persistence
        await self.memory.store({
            "type": "execution_plan",
            "plan_id": plan_id,
            "data": asdict(plan)
        })
        
        return plan
        
    async def execute_plan(self, plan_id: str) -> Dict[str, Any]:
        """
        Execute a plan step by step.
        
        Args:
            plan_id: ID of the plan to execute
            
        Returns:
            Execution results and final status
        """
        if plan_id not in self.active_plans:
            raise ValueError(f"Plan {plan_id} not found")
            
        plan = self.active_plans[plan_id]
        plan.status = "executing"
        plan.started_at = time.time()
        
        results = []
        
        try:
            # Execute steps in dependency order
            execution_order = self._resolve_dependencies(plan.steps)
            
            for step in execution_order:
                step_result = await self._execute_step(step)
                results.append(step_result)
                
                # Stop execution if critical step failed
                if step.status == PlanStepStatus.FAILED and step.parameters.get("critical", False):
                    plan.status = "failed"
                    break
                    
            # Determine final status
            if plan.status != "failed":
                failed_steps = [s for s in plan.steps if s.status == PlanStepStatus.FAILED]
                if failed_steps:
                    plan.status = "partial_success"
                else:
                    plan.status = "completed"
                    
            plan.completed_at = time.time()
            
            # Update memory
            await self.memory.store({
                "type": "execution_plan",
                "plan_id": plan_id,
                "data": asdict(plan)
            })
            
        except Exception as e:
            plan.status = "error"
            plan.completed_at = time.time()
            results.append({"error": str(e), "step": "execution"})
            
        return {
            "plan_id": plan_id,
            "status": plan.status,
            "results": results,
            "execution_time": plan.completed_at - plan.started_at if plan.completed_at else None
        }
        
    async def _execute_step(self, step: PlanStep) -> Dict[str, Any]:
        """Execute a single plan step."""
        step.status = PlanStepStatus.IN_PROGRESS
        step.started_at = time.time()
        
        try:
            # Get the appropriate skill
            if step.skill not in self.skills:
                raise ValueError(f"Skill '{step.skill}' not registered")
                
            skill = self.skills[step.skill]
            
            # Execute the skill action
            result = await skill.execute(step.action, step.parameters)
            
            step.status = PlanStepStatus.COMPLETED
            step.result = result
            step.completed_at = time.time()
            
            return {
                "step_id": step.id,
                "status": "success",
                "result": result,
                "execution_time": step.completed_at - step.started_at
            }
            
        except Exception as e:
            step.status = PlanStepStatus.FAILED
            step.error = str(e)
            step.completed_at = time.time()
            
            # Retry logic
            if step.retry_count < step.max_retries:
                step.retry_count += 1
                step.status = PlanStepStatus.PENDING
                # Could implement exponential backoff here
                await asyncio.sleep(1 * step.retry_count)
                return await self._execute_step(step)
                
            return {
                "step_id": step.id,
                "status": "failed",
                "error": str(e),
                "retry_count": step.retry_count
            }
            
    def _build_planning_prompt(self, goal: str, context: Dict[str, Any]) -> str:
        """Build a prompt for the agent to create a plan."""
        available_skills = list(self.skills.keys())
        
        prompt = f"""
Create an execution plan to achieve this goal: {goal}

Available skills: {', '.join(available_skills)}

Context: {json.dumps(context, indent=2)}

Please create a step-by-step plan with the following format:
1. Each step should specify: skill, action, parameters, description
2. Include dependencies between steps where needed
3. Mark critical steps that would cause plan failure if they fail
4. Consider error handling and alternative paths

Respond with a JSON array of steps in this format:
[
  {{
    "id": "step_1",
    "skill": "dataset",
    "action": "search",
    "parameters": {{"query": "example"}},
    "description": "Search for relevant datasets",
    "dependencies": [],
    "critical": true
  }}
]
"""
        return prompt
        
    def _parse_plan_response(self, response: Dict[str, Any]) -> List[PlanStep]:
        """Parse the agent's response into PlanStep objects."""
        try:
            # Extract steps from response
            content = response.get("content", "")
            if isinstance(content, dict) and "steps" in content:
                steps_data = content["steps"]
            else:
                # Try to parse JSON from string content
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    steps_data = json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse plan from response")
                    
            steps = []
            for step_data in steps_data:
                step = PlanStep(
                    id=step_data.get("id", f"step_{len(steps)}"),
                    skill=step_data.get("skill", "unknown"),
                    action=step_data.get("action", "execute"),
                    parameters=step_data.get("parameters", {}),
                    description=step_data.get("description", ""),
                    dependencies=step_data.get("dependencies", []),
                    max_retries=step_data.get("max_retries", 3)
                )
                steps.append(step)
                
            return steps
            
        except Exception as e:
            # Fallback: create a simple single-step plan
            return [PlanStep(
                id="fallback_step",
                skill="agent",
                action="execute",
                parameters={"original_response": response},
                description="Fallback execution of original request"
            )]
            
    def _resolve_dependencies(self, steps: List[PlanStep]) -> List[PlanStep]:
        """Resolve step dependencies and return execution order."""
        # Simple topological sort
        executed = set()
        execution_order = []
        
        while len(execution_order) < len(steps):
            progress_made = False
            
            for step in steps:
                if step.id in executed:
                    continue
                    
                # Check if all dependencies are satisfied
                deps_satisfied = all(dep in executed for dep in step.dependencies)
                
                if deps_satisfied:
                    execution_order.append(step)
                    executed.add(step.id)
                    progress_made = True
                    
            if not progress_made:
                # Circular dependency or missing dependency - add remaining steps
                remaining = [s for s in steps if s.id not in executed]
                execution_order.extend(remaining)
                break
                
        return execution_order
        
    def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """Get the current status of a plan."""
        if plan_id not in self.active_plans:
            return {"error": "Plan not found"}
            
        plan = self.active_plans[plan_id]
        
        step_statuses = {}
        for step in plan.steps:
            step_statuses[step.id] = {
                "status": step.status.value,
                "description": step.description,
                "result": step.result,
                "error": step.error
            }
            
        return {
            "plan_id": plan_id,
            "goal": plan.goal,
            "status": plan.status,
            "steps": step_statuses,
            "created_at": plan.created_at,
            "started_at": plan.started_at,
            "completed_at": plan.completed_at
        }

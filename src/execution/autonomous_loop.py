"""
Autonomous Execution Loop - NIS Protocol v4.0
The missing piece that makes the system truly autonomous.

This connects: LLM → Code Executor → Output → LLM (iterate until done)
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

from .code_executor import get_code_executor, ExecutionOutput

logger = logging.getLogger("nis.autonomous_loop")


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_FOR_CODE = "waiting_for_code"
    EXECUTING_CODE = "executing_code"
    ANALYZING_RESULT = "analyzing_result"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class AutonomousStep:
    """A single step in the autonomous execution"""
    step_number: int
    action: str  # "think", "code", "execute", "analyze"
    input_data: str
    output_data: str
    execution_result: Optional[ExecutionOutput] = None
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0


@dataclass
class AutonomousTask:
    """A complete autonomous task with history"""
    task_id: str
    original_request: str
    status: TaskStatus = TaskStatus.PENDING
    steps: List[AutonomousStep] = field(default_factory=list)
    final_result: Optional[str] = None
    artifacts: List[Dict[str, Any]] = field(default_factory=list)  # plots, files, etc.
    total_iterations: int = 0
    total_time_ms: float = 0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "original_request": self.original_request,
            "status": self.status.value,
            "steps": [
                {
                    "step": s.step_number,
                    "action": s.action,
                    "input": s.input_data[:200] + "..." if len(s.input_data) > 200 else s.input_data,
                    "output": s.output_data[:200] + "..." if len(s.output_data) > 200 else s.output_data,
                    "duration_ms": s.duration_ms
                }
                for s in self.steps
            ],
            "final_result": self.final_result,
            "artifacts": self.artifacts,
            "total_iterations": self.total_iterations,
            "total_time_ms": self.total_time_ms,
            "error": self.error
        }


class AutonomousExecutor:
    """
    The autonomous execution loop.
    
    Flow:
    1. User gives task
    2. LLM analyzes and decides if code is needed
    3. If code needed: LLM generates code
    4. Code executor runs code
    5. LLM sees results (stdout, plots, errors)
    6. LLM decides: done, or iterate
    7. Repeat until complete or max iterations
    """
    
    def __init__(
        self,
        llm_callback: Optional[Callable] = None,
        max_iterations: int = 10,
        timeout_seconds: int = 300
    ):
        self.llm_callback = llm_callback  # Function to call LLM
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.executor = get_code_executor()
        self.tasks: Dict[str, AutonomousTask] = {}
    
    async def execute_task(
        self,
        request: str,
        llm_callback: Optional[Callable] = None
    ) -> AutonomousTask:
        """
        Execute an autonomous task.
        
        Args:
            request: User's request (e.g., "Analyze this data and create a chart")
            llm_callback: Async function to call LLM: async def callback(prompt) -> str
            
        Returns:
            AutonomousTask with full execution history
        """
        callback = llm_callback or self.llm_callback
        if not callback:
            raise ValueError("LLM callback required for autonomous execution")
        
        task_id = str(uuid.uuid4())[:8]
        task = AutonomousTask(
            task_id=task_id,
            original_request=request,
            status=TaskStatus.RUNNING
        )
        self.tasks[task_id] = task
        
        start_time = time.time()
        
        try:
            await self._run_loop(task, callback)
        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error = f"Task timed out after {self.timeout_seconds}s"
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"Autonomous task {task_id} failed: {e}")
        
        task.total_time_ms = (time.time() - start_time) * 1000
        return task
    
    async def _run_loop(self, task: AutonomousTask, llm_callback: Callable):
        """Run the autonomous loop"""
        
        iteration = 0
        context = f"User request: {task.original_request}\n\n"
        
        while iteration < self.max_iterations:
            iteration += 1
            task.total_iterations = iteration
            
            # Step 1: Ask LLM what to do
            task.status = TaskStatus.WAITING_FOR_CODE
            
            analysis_prompt = self._build_analysis_prompt(context, iteration)
            step_start = time.time()
            
            llm_response = await llm_callback(analysis_prompt)
            
            step = AutonomousStep(
                step_number=iteration,
                action="analyze",
                input_data=analysis_prompt,
                output_data=llm_response,
                duration_ms=(time.time() - step_start) * 1000
            )
            task.steps.append(step)
            
            # Parse LLM response
            action, content = self._parse_llm_response(llm_response)
            
            if action == "DONE":
                task.status = TaskStatus.COMPLETED
                task.final_result = content
                return
            
            elif action == "CODE":
                # Execute the code
                task.status = TaskStatus.EXECUTING_CODE
                exec_start = time.time()
                
                exec_result = await self.executor.execute(content)
                
                exec_step = AutonomousStep(
                    step_number=iteration,
                    action="execute",
                    input_data=content,
                    output_data=exec_result.stdout or exec_result.error or "No output",
                    execution_result=exec_result,
                    duration_ms=(time.time() - exec_start) * 1000
                )
                task.steps.append(exec_step)
                
                # Capture artifacts
                if exec_result.plots:
                    for plot in exec_result.plots:
                        task.artifacts.append({
                            "type": "plot",
                            "name": plot["name"],
                            "base64": plot["base64"],
                            "iteration": iteration
                        })
                
                if exec_result.dataframes:
                    for df in exec_result.dataframes:
                        task.artifacts.append({
                            "type": "dataframe",
                            "name": df["name"],
                            "shape": df["shape"],
                            "preview": df.get("preview"),
                            "iteration": iteration
                        })
                
                # Update context with execution results
                context += f"\n--- Iteration {iteration} ---\n"
                context += f"Code executed:\n```python\n{content}\n```\n"
                context += f"Output: {exec_result.stdout}\n"
                if exec_result.error:
                    context += f"\n⚠️ ERROR DETECTED - SELF-CORRECTION REQUIRED:\n"
                    context += f"Error: {exec_result.error}\n"
                    context += f"Please analyze this error and fix the code. Common fixes:\n"
                    context += f"- ImportError: Check if library is available (numpy, pandas, matplotlib, scipy, sympy)\n"
                    context += f"- NameError: Variable not defined, check spelling\n"
                    context += f"- TypeError: Wrong argument types\n"
                    context += f"- SyntaxError: Check code syntax\n"
                    context += f"You MUST fix this error in your next CODE action.\n"
                if exec_result.plots:
                    context += f"✅ Generated {len(exec_result.plots)} plot(s)\n"
                if exec_result.dataframes:
                    context += f"✅ Created {len(exec_result.dataframes)} dataframe(s)\n"
                if exec_result.success and not exec_result.error:
                    context += f"✅ Code executed successfully\n"
            
            elif action == "THINK":
                # LLM is thinking, add to context
                context += f"\n--- Thought {iteration} ---\n{content}\n"
            
            else:
                # Unknown action, treat as thinking
                context += f"\n--- Response {iteration} ---\n{llm_response}\n"
        
        # Max iterations reached
        task.status = TaskStatus.MAX_ITERATIONS
        task.error = f"Reached maximum iterations ({self.max_iterations})"
    
    def _build_analysis_prompt(self, context: str, iteration: int) -> str:
        """Build the prompt for LLM analysis"""
        return f"""You are an autonomous AI agent that can execute Python code to complete tasks.

{context}

Based on the above, decide your next action. You MUST respond in one of these formats:

1. If you need to run Python code:
ACTION: CODE
```python
# your code here
```

2. If you need to think/plan:
ACTION: THINK
Your thoughts here...

3. If the task is complete:
ACTION: DONE
Final summary of what was accomplished...

Available libraries: numpy, pandas, matplotlib, scipy, sympy, math, json, datetime
You can generate plots with matplotlib - they will be captured automatically.
You can create DataFrames - they will be captured automatically.

Current iteration: {iteration}/{self.max_iterations}

What is your next action?"""
    
    def _parse_llm_response(self, response: str) -> tuple:
        """Parse LLM response to extract action and content"""
        response = response.strip()
        
        # Check for ACTION: markers
        if "ACTION: DONE" in response:
            content = response.split("ACTION: DONE", 1)[-1].strip()
            return ("DONE", content)
        
        if "ACTION: CODE" in response:
            # Extract code block
            content = response.split("ACTION: CODE", 1)[-1].strip()
            if "```python" in content:
                start = content.find("```python") + 9
                end = content.find("```", start)
                if end > start:
                    content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                if end > start:
                    content = content[start:end].strip()
            return ("CODE", content)
        
        if "ACTION: THINK" in response:
            content = response.split("ACTION: THINK", 1)[-1].strip()
            return ("THINK", content)
        
        # Try to detect code without explicit marker
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return ("CODE", response[start:end].strip())
        
        # Default to thinking
        return ("THINK", response)
    
    def get_task(self, task_id: str) -> Optional[AutonomousTask]:
        """Get a task by ID"""
        return self.tasks.get(task_id)
    
    def list_tasks(self) -> List[Dict[str, Any]]:
        """List all tasks"""
        return [
            {
                "task_id": t.task_id,
                "status": t.status.value,
                "iterations": t.total_iterations,
                "artifacts": len(t.artifacts)
            }
            for t in self.tasks.values()
        ]


# Global instance
_autonomous_executor: Optional[AutonomousExecutor] = None


def get_autonomous_executor() -> AutonomousExecutor:
    """Get or create the global autonomous executor"""
    global _autonomous_executor
    if _autonomous_executor is None:
        _autonomous_executor = AutonomousExecutor()
    return _autonomous_executor


async def run_autonomous_task(request: str, llm_callback: Callable) -> AutonomousTask:
    """Convenience function to run an autonomous task"""
    executor = get_autonomous_executor()
    return await executor.execute_task(request, llm_callback)

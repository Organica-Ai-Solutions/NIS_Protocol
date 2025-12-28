#!/usr/bin/env python3
"""
Streaming Execution Engine for NIS Protocol
Provides real-time progress updates via Server-Sent Events (SSE)

Copyright 2025 Organica AI Solutions
Licensed under Apache License 2.0
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """Represents a single streaming event."""
    event_type: str  # plan_created, step_started, step_completed, level_completed, execution_completed
    timestamp: float
    data: Dict[str, Any]
    
    def to_sse(self) -> str:
        """Convert to Server-Sent Events format."""
        return f"event: {self.event_type}\ndata: {json.dumps(self.data)}\n\n"


class StreamingExecutor:
    """
    Executes autonomous tasks with real-time streaming updates.
    
    Emits events via Server-Sent Events (SSE) for:
    - Plan creation
    - Step execution start/completion
    - Level completion
    - Overall execution status
    - Performance metrics
    """
    
    def __init__(self, orchestrator):
        """Initialize streaming executor with orchestrator reference."""
        self.orchestrator = orchestrator
    
    async def execute_with_streaming(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        parallel: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Execute autonomous task with streaming progress updates.
        
        Args:
            goal: User's goal/objective
            context: Optional context
            parallel: Enable parallel execution
        
        Yields:
            Server-Sent Events formatted strings
        """
        start_time = time.time()
        
        try:
            # Event 1: Execution started
            yield StreamEvent(
                event_type="execution_started",
                timestamp=time.time(),
                data={
                    "goal": goal,
                    "parallel": parallel,
                    "status": "planning"
                }
            ).to_sse()
            
            # Create plan with LLM
            logger.info(f"🎯 Creating plan for: {goal}")
            execution_plan = await self.orchestrator.llm_planner.create_plan(goal, context)
            
            # Event 2: Plan created
            yield StreamEvent(
                event_type="plan_created",
                timestamp=time.time(),
                data={
                    "steps": [{
                        "index": i,
                        "agent_type": s.agent_type,
                        "description": s.description,
                        "tool_name": s.tool_name,
                        "dependencies": s.dependencies
                    } for i, s in enumerate(execution_plan.steps)],
                    "reasoning": execution_plan.reasoning,
                    "confidence": execution_plan.confidence,
                    "estimated_duration": execution_plan.estimated_duration,
                    "total_steps": len(execution_plan.steps)
                }
            ).to_sse()
            
            # Execute with streaming
            if parallel and len(execution_plan.steps) > 1:
                # Parallel execution with streaming
                async for event in self._execute_parallel_with_streaming(execution_plan):
                    yield event
            else:
                # Sequential execution with streaming
                async for event in self._execute_sequential_with_streaming(execution_plan):
                    yield event
            
            # Event: Execution completed
            execution_time = time.time() - start_time
            yield StreamEvent(
                event_type="execution_completed",
                timestamp=time.time(),
                data={
                    "status": "completed",
                    "execution_time": execution_time,
                    "total_steps": len(execution_plan.steps)
                }
            ).to_sse()
            
        except Exception as e:
            logger.error(f"Streaming execution error: {e}")
            yield StreamEvent(
                event_type="execution_error",
                timestamp=time.time(),
                data={
                    "error": str(e),
                    "status": "failed"
                }
            ).to_sse()
    
    async def _execute_parallel_with_streaming(
        self,
        execution_plan: Any
    ) -> AsyncGenerator[str, None]:
        """Execute plan in parallel with streaming updates."""
        from src.core.parallel_executor import DependencyGraph
        
        # Build dependency graph
        dep_graph = DependencyGraph(execution_plan.steps)
        parallel_batches = dep_graph.get_parallel_batches()
        
        # Event: Parallelization info
        summary = dep_graph.get_execution_summary()
        yield StreamEvent(
            event_type="parallelization_info",
            timestamp=time.time(),
            data={
                "total_levels": len(parallel_batches),
                "max_parallelism": summary["max_parallelism"],
                "parallelization_ratio": summary["parallelization_ratio"]
            }
        ).to_sse()
        
        # Execute batches level by level
        all_results = {}
        
        for level_idx, batch in enumerate(parallel_batches):
            batch_start = time.time()
            
            # Event: Level started
            yield StreamEvent(
                event_type="level_started",
                timestamp=time.time(),
                data={
                    "level": level_idx + 1,
                    "total_levels": len(parallel_batches),
                    "steps_in_level": len(batch),
                    "parallel": len(batch) > 1
                }
            ).to_sse()
            
            # Execute steps in this level
            if len(batch) > 1:
                # Parallel execution
                tasks = []
                for node in batch:
                    task = self._execute_step_with_events(node, all_results, level_idx)
                    tasks.append(task)
                
                # Gather results and events
                async for event in self._gather_with_streaming(tasks):
                    yield event
                    
            else:
                # Single step
                node = batch[0]
                async for event in self._execute_step_with_events(node, all_results, level_idx):
                    yield event
            
            # Event: Level completed
            batch_time = time.time() - batch_start
            yield StreamEvent(
                event_type="level_completed",
                timestamp=time.time(),
                data={
                    "level": level_idx + 1,
                    "execution_time": batch_time,
                    "steps_completed": len(batch)
                }
            ).to_sse()
    
    async def _execute_sequential_with_streaming(
        self,
        execution_plan: Any
    ) -> AsyncGenerator[str, None]:
        """Execute plan sequentially with streaming updates."""
        for idx, step in enumerate(execution_plan.steps):
            step_start = time.time()
            
            # Event: Step started
            yield StreamEvent(
                event_type="step_started",
                timestamp=time.time(),
                data={
                    "step_index": idx,
                    "total_steps": len(execution_plan.steps),
                    "agent_type": step.agent_type,
                    "description": step.description,
                    "tool_name": step.tool_name
                }
            ).to_sse()
            
            # Execute step
            task = {
                "type": step.agent_type,
                "description": step.description,
                "parameters": step.parameters or {}
            }
            
            result = await self.orchestrator.execute_autonomous_task(task)
            
            # Event: Step completed
            step_time = time.time() - step_start
            yield StreamEvent(
                event_type="step_completed",
                timestamp=time.time(),
                data={
                    "step_index": idx,
                    "status": result.get("status", "unknown"),
                    "execution_time": step_time,
                    "tools_used": result.get("tools_used", [])
                }
            ).to_sse()
            
            # Stop if failed
            if result.get("status") != "completed":
                break
    
    async def _execute_step_with_events(
        self,
        node: Any,
        previous_results: Dict[int, Any],
        level: int
    ) -> AsyncGenerator[str, None]:
        """Execute a single step and yield events."""
        step_start = time.time()
        
        # Event: Step started
        yield StreamEvent(
            event_type="step_started",
            timestamp=time.time(),
            data={
                "step_index": node.step_index,
                "level": level + 1,
                "agent_type": node.agent_type,
                "description": node.description,
                "tool_name": node.tool_name,
                "dependencies": node.dependencies
            }
        ).to_sse()
        
        # Execute step
        task = {
            "type": node.agent_type,
            "description": node.description,
            "parameters": node.parameters or {}
        }
        
        try:
            result = await self.orchestrator.execute_autonomous_task(task)
            previous_results[node.step_index] = result
            
            # Event: Step completed
            step_time = time.time() - step_start
            yield StreamEvent(
                event_type="step_completed",
                timestamp=time.time(),
                data={
                    "step_index": node.step_index,
                    "level": level + 1,
                    "status": result.get("status", "unknown"),
                    "execution_time": step_time,
                    "tools_used": result.get("tools_used", [])
                }
            ).to_sse()
            
        except Exception as e:
            logger.error(f"Step {node.step_index} failed: {e}")
            yield StreamEvent(
                event_type="step_error",
                timestamp=time.time(),
                data={
                    "step_index": node.step_index,
                    "error": str(e)
                }
            ).to_sse()
    
    async def _gather_with_streaming(
        self,
        tasks: list
    ) -> AsyncGenerator[str, None]:
        """Gather async generators and yield all events."""
        # Convert tasks to list of async generators
        pending = {asyncio.create_task(task.__anext__()): task for task in tasks}
        
        while pending:
            done, pending_tasks = await asyncio.wait(
                pending.keys(),
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in done:
                generator = pending.pop(task)
                try:
                    event = task.result()
                    yield event
                    
                    # Schedule next event from this generator
                    pending[asyncio.create_task(generator.__anext__())] = generator
                    
                except StopAsyncIteration:
                    # Generator finished
                    pass
                except Exception as e:
                    logger.error(f"Error in streaming task: {e}")


def get_streaming_executor(orchestrator) -> StreamingExecutor:
    """Get or create streaming executor instance."""
    return StreamingExecutor(orchestrator=orchestrator)

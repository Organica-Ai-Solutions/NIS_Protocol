"""
Autonomous Agent Orchestrator

Coordinates autonomous agents with MCP tool execution.
Agents can plan, execute, and adapt using real tools.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from src.core.mcp_tool_executor import get_mcp_executor
from src.core.llm_planner import get_llm_planner, ExecutionPlan
from src.core.parallel_executor import get_parallel_executor
from src.core.streaming_executor import get_streaming_executor
from src.core.backup_agents import get_backup_agent_executor
from src.core.agent_competition import get_agent_competition_system
from src.core.branching_strategies import get_branching_strategies_system
from src.core.ml_prediction_engine import get_ml_prediction_engine
from src.core.llm_judge import get_llm_judge
from src.core.multi_critic_review import get_multi_critic_review_system
from src.core.pipeline_processor import get_pipeline_processor
from src.core.shared_workspace import get_shared_workspace
from src.agents.autonomous_agent_mixin import (
    AutonomousResearchAgent,
    AutonomousPhysicsAgent,
    AutonomousRoboticsAgent,
    AutonomousVisionAgent
)

logger = logging.getLogger("nis.autonomous_orchestrator")


class AutonomousOrchestrator:
    """
    Orchestrates autonomous agents with real tool execution.
    
    This is the brain that coordinates multiple autonomous agents
    to accomplish complex tasks using MCP tools.
    """
    
    def __init__(self, llm_provider=None, enable_speed_optimizations=True, enable_ai_enhancements=True):
        self.mcp_executor = get_mcp_executor()
        self.llm_planner = get_llm_planner(llm_provider=llm_provider, mcp_executor=self.mcp_executor)
        self.parallel_executor = get_parallel_executor(orchestrator=self)
        self.streaming_executor = get_streaming_executor(orchestrator=self)
        
        # Speed optimization systems
        self.enable_speed_optimizations = enable_speed_optimizations
        self.enable_ai_enhancements = enable_ai_enhancements
        
        if enable_speed_optimizations:
            self.backup_executor = get_backup_agent_executor(num_backups=3)
            if self.llm_planner.multi_provider:
                self.competition_system = get_agent_competition_system(self.llm_planner.multi_provider)
            else:
                self.competition_system = None
            self.branching_system = get_branching_strategies_system(self.llm_planner)
        else:
            self.backup_executor = None
            self.competition_system = None
            self.branching_system = None
        
        # AI/ML enhancement systems
        if enable_ai_enhancements and llm_provider:
            self.ml_prediction = get_ml_prediction_engine(llm_provider, self.mcp_executor)
            self.llm_judge = get_llm_judge(llm_provider)
            self.multi_critic = get_multi_critic_review_system(llm_provider)
            self.pipeline_processor = get_pipeline_processor(self)
            self.shared_workspace = get_shared_workspace(llm_provider)
        else:
            self.ml_prediction = None
            self.llm_judge = None
            self.multi_critic = None
            self.pipeline_processor = None
            self.shared_workspace = None
        self.agents = {
            "research": AutonomousResearchAgent(),
            "physics": AutonomousPhysicsAgent(),
            "robotics": AutonomousRoboticsAgent(),
            "vision": AutonomousVisionAgent()
        }
        self.task_history = []
        
        optimization_status = "enabled" if enable_speed_optimizations else "disabled"
        ai_status = "enabled" if enable_ai_enhancements else "disabled"
        logger.info(f"🎭 Autonomous Orchestrator initialized with 4 agents + LLM planner + parallel + streaming + speed optimizations ({optimization_status}) + AI enhancements ({ai_status})")
    
    async def execute_autonomous_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an autonomous task by routing to appropriate agents.
        
        Args:
            task: Task specification with type and parameters
            
        Returns:
            Task execution result with tool usage trace
        """
        task_type = task.get("type", "general")
        task_description = task.get("description", "")
        parameters = task.get("parameters", {})
        
        logger.info(f"🎭 Executing autonomous task: {task_type}")
        
        result = {
            "task_type": task_type,
            "description": task_description,
            "status": "executing",
            "tools_used": [],
            "agent_actions": []
        }
        
        try:
            if task_type == "research":
                agent_result = await self.agents["research"].research_topic(
                    parameters.get("topic", task_description)
                )
                result["agent_result"] = agent_result
                result["tools_used"] = agent_result.get("tools_used", [])
                result["status"] = "completed"
                
            elif task_type == "physics":
                agent_result = await self.agents["physics"].solve_and_validate(
                    parameters.get("equation_type", "heat-equation"),
                    parameters.get("equation_params", {})
                )
                result["agent_result"] = agent_result
                result["tools_used"] = agent_result.get("tools_used", [])
                result["status"] = "completed"
                
            elif task_type == "robotics":
                agent_result = await self.agents["robotics"].plan_and_execute_motion(
                    parameters.get("target_position", [1.0, 0.5, 0.3])
                )
                result["agent_result"] = agent_result
                result["tools_used"] = agent_result.get("tools_used", [])
                result["status"] = "completed"
                
            elif task_type == "vision":
                agent_result = await self.agents["vision"].analyze_and_research(
                    parameters.get("image_data", ""),
                    parameters.get("context", "")
                )
                result["agent_result"] = agent_result
                result["tools_used"] = agent_result.get("tools_used", [])
                result["status"] = "completed"
                
            elif task_type == "multi_agent":
                # Multi-agent collaboration
                agent_results = await self._execute_multi_agent_task(task)
                result["agent_results"] = agent_results
                result["status"] = "completed"
                
            else:
                # General task - use research agent as default
                agent_result = await self.agents["research"].research_topic(task_description)
                result["agent_result"] = agent_result
                result["tools_used"] = agent_result.get("tools_used", [])
                result["status"] = "completed"
            
            # Track in history
            self.task_history.append(result)
            
        except Exception as e:
            logger.error(f"Autonomous task execution error: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
        
        return result
    
    async def _execute_multi_agent_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute a task requiring multiple agents.
        
        Example: Research a physics problem, solve it, and validate with code
        """
        subtasks = task.get("subtasks", [])
        results = []
        
        for subtask in subtasks:
            agent_type = subtask.get("agent", "research")
            if agent_type in self.agents:
                agent = self.agents[agent_type]
                
                # Execute based on agent type
                if agent_type == "research":
                    result = await agent.research_topic(subtask.get("topic", ""))
                elif agent_type == "physics":
                    result = await agent.solve_and_validate(
                        subtask.get("equation_type", "heat-equation"),
                        subtask.get("parameters", {})
                    )
                elif agent_type == "robotics":
                    result = await agent.plan_and_execute_motion(
                        subtask.get("target_position", [0, 0, 0])
                    )
                elif agent_type == "vision":
                    result = await agent.analyze_and_research(
                        subtask.get("image_data", ""),
                        subtask.get("context", "")
                    )
                
                results.append({
                    "agent": agent_type,
                    "subtask": subtask,
                    "result": result
                })
        
        return results
    
    async def plan_and_execute(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        parallel: bool = True,
        use_branching: bool = False,
        use_competition: bool = False,
        use_backup: bool = False
    ) -> Dict[str, Any]:
        """
        Autonomous planning and execution using LLM.
        
        The orchestrator:
        1. Uses LLM to analyze goal and create intelligent plan
        2. Executes plan steps sequentially
        3. Handles dependencies between steps
        4. Validates results
        
        Args:
            goal: User's goal/objective
            context: Optional context
            parallel: Use parallel execution for independent steps
            use_branching: Generate multiple strategies and pick best
            use_competition: Run multiple providers in competition
            use_backup: Use backup agents for reliability
        
        Returns:
            Execution result with plan and step results
        """
        logger.info(f"🎯 Planning and executing: {goal}")
        start_time = time.time()
        
        # Step 1: Create plan using LLM (with optional branching)
        if use_branching and self.branching_system:
            logger.info("🌳 Using branching strategies")
            branching_result = await self.branching_system.generate_strategies(
                goal=goal,
                context=context,
                num_strategies=3
            )
            if branching_result["success"]:
                execution_plan = branching_result["plan"]
                logger.info(f"✅ Selected {branching_result['winner']['strategy']} strategy")
            else:
                logger.warning("⚠️ Branching failed, using standard planning")
                execution_plan = await self.llm_planner.create_plan(goal, context)
        else:
            execution_plan = await self.llm_planner.create_plan(goal, context)
        
        logger.info(f"📋 Plan created: {len(execution_plan.steps)} steps, confidence: {execution_plan.confidence:.2f}")
        logger.info(f"💭 Reasoning: {execution_plan.reasoning}")
        
        # Choose execution mode
        if parallel and len(execution_plan.steps) > 1:
            # Parallel execution
            logger.info("🚀 Using parallel execution mode")
            parallel_result = await self.parallel_executor.execute_plan_parallel(execution_plan)
            
            execution_time = time.time() - start_time
            
            return {
                "goal": goal,
                "plan": {
                    "steps": [{
                        "agent_type": s.agent_type,
                        "description": s.description,
                        "tool_name": s.tool_name,
                        "dependencies": s.dependencies
                    } for s in execution_plan.steps],
                    "reasoning": execution_plan.reasoning,
                    "confidence": execution_plan.confidence,
                    "estimated_duration": execution_plan.estimated_duration
                },
                "execution_results": parallel_result["execution_results"],
                "status": parallel_result["status"],
                "execution_mode": "parallel",
                "execution_time": execution_time,
                "parallelization": parallel_result.get("parallelization", {}),
                "autonomous": True,
                "llm_powered": True
            }
        else:
            # Sequential execution (original logic)
            logger.info("⏭️ Using sequential execution mode")
            execution_results = []
            step_outputs = {}
            
            for idx, step in enumerate(execution_plan.steps):
                logger.info(f"▶️ Executing step {idx + 1}/{len(execution_plan.steps)}: {step.description}")
                
                task = {
                    "type": step.agent_type,
                    "description": step.description,
                    "parameters": step.parameters or {}
                }
                
                result = await self.execute_autonomous_task(task)
                execution_results.append({
                    "step_index": idx,
                    "step": step,
                    "result": result
                })
                
                step_outputs[idx] = result
                
                if result.get("status") != "completed":
                    logger.warning(f"⚠️ Execution stopped at step {idx + 1}: {step.description}")
                    break
            
            execution_time = time.time() - start_time
            
            return {
                "goal": goal,
                "plan": {
                    "steps": [{
                        "agent_type": s.agent_type,
                        "description": s.description,
                        "tool_name": s.tool_name,
                        "dependencies": s.dependencies
                    } for s in execution_plan.steps],
                    "reasoning": execution_plan.reasoning,
                    "confidence": execution_plan.confidence,
                    "estimated_duration": execution_plan.estimated_duration
                },
                "execution_results": execution_results,
                "status": "completed" if all(r["result"].get("status") == "completed" for r in execution_results) else "partial",
                "execution_mode": "sequential",
                "execution_time": execution_time,
                "autonomous": True,
                "llm_powered": True
            }
    
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all autonomous agents"""
        status = {
            "orchestrator": "active",
            "agents": {
                name: agent.get_autonomous_status()
                for name, agent in self.agents.items()
            },
            "tasks_completed": len(self.task_history),
            "mcp_tools_available": len(self.mcp_executor.get_available_tools()),
            "parallel_execution": {
                "enabled": True,
                "stats": self.parallel_executor.get_stats()
            }
        }
        
        # Add multi-provider stats if available
        if hasattr(self.llm_planner, 'multi_provider') and self.llm_planner.multi_provider:
            status["llm_providers"] = self.llm_planner.multi_provider.get_stats_summary()
        
        return status
    
    def get_task_history(self) -> List[Dict[str, Any]]:
        """Get history of executed tasks"""
        return self.task_history.copy()


# Global orchestrator instance
_global_orchestrator: Optional[AutonomousOrchestrator] = None


def get_autonomous_orchestrator() -> AutonomousOrchestrator:
    """Get or create global autonomous orchestrator"""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = AutonomousOrchestrator()
    return _global_orchestrator

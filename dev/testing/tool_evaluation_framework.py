#!/usr/bin/env python3
"""
NIS Protocol Tool Evaluation Framework
Based on Anthropic's "Writing effective tools for agents" research

This framework systematically measures tool performance using:
- Real-world evaluation tasks
- Multi-tool workflow testing  
- Token efficiency metrics
- Agent reasoning analysis
- Comprehensive performance tracking

Reference: https://www.anthropic.com/engineering/writing-tools-for-agents
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import statistics
import uuid

# NIS Protocol imports
from src.core.agent_orchestrator import NISAgentOrchestrator, ContextAnalyzer
from src.mcp.schemas.tool_schemas import ToolSchemas
from src.utils.confidence_calculator import calculate_confidence

logger = logging.getLogger(__name__)


class EvaluationTaskType(Enum):
    """Types of evaluation tasks for tool testing"""
    SINGLE_TOOL = "single_tool"
    MULTI_TOOL_WORKFLOW = "multi_tool_workflow"
    COMPLEX_REASONING = "complex_reasoning"
    EDGE_CASE = "edge_case"
    PERFORMANCE_STRESS = "performance_stress"


class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"           # 1-2 tool calls
    MODERATE = "moderate"       # 3-5 tool calls
    COMPLEX = "complex"         # 6-10 tool calls
    ADVANCED = "advanced"       # 10+ tool calls


@dataclass
class EvaluationTask:
    """Represents a single evaluation task"""
    task_id: str
    name: str
    description: str
    task_type: EvaluationTaskType
    complexity: TaskComplexity
    
    # Task definition
    input_prompt: str
    expected_tools: List[str]  # Tools expected to be called
    expected_outcome: Dict[str, Any]
    
    # Evaluation criteria
    success_criteria: List[str]
    max_tool_calls: int = 20
    timeout_seconds: float = 60.0
    
    # Context and metadata
    context: Dict[str, Any] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.tags is None:
            self.tags = []


@dataclass
class EvaluationResult:
    """Results from evaluating a single task"""
    task_id: str
    success: bool
    execution_time: float
    tool_calls_made: List[Dict[str, Any]]
    tokens_consumed: int
    
    # Performance metrics
    tool_accuracy: float  # Did agent call expected tools?
    outcome_accuracy: float  # Did agent achieve expected outcome?
    efficiency_score: float  # Ratio of necessary vs actual tool calls
    
    # Detailed analysis
    reasoning_trace: List[str]
    errors: List[str]
    unexpected_behaviors: List[str]
    
    # Metadata
    timestamp: float
    agent_feedback: Optional[str] = None
    
    def __post_init__(self):
        if not hasattr(self, 'timestamp'):
            self.timestamp = time.time()


@dataclass
class ToolPerformanceMetrics:
    """Performance metrics for individual tools"""
    tool_name: str
    
    # Usage statistics
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    
    # Performance metrics
    average_response_time: float = 0.0
    average_tokens_consumed: int = 0
    success_rate: float = 0.0
    
    # Context efficiency
    context_utilization: float = 0.0  # How much of response is actually useful
    hallucination_rate: float = 0.0   # Rate of incorrect/made-up responses
    
    # Usage patterns
    common_parameters: Dict[str, Any] = None
    error_patterns: List[str] = None
    
    def __post_init__(self):
        if self.common_parameters is None:
            self.common_parameters = {}
        if self.error_patterns is None:
            self.error_patterns = []


class NISToolEvaluator:
    """
    Comprehensive tool evaluation system for NIS Protocol.
    
    Implements Anthropic's best practices for measuring tool effectiveness:
    - Real-world task scenarios
    - Multi-tool workflow testing
    - Token efficiency analysis
    - Agent reasoning evaluation
    """
    
    def __init__(self, orchestrator: NISAgentOrchestrator, tool_schemas: ToolSchemas):
        self.orchestrator = orchestrator
        self.tool_schemas = tool_schemas
        self.context_analyzer = ContextAnalyzer()
        
        # Evaluation state
        self.evaluation_tasks: Dict[str, EvaluationTask] = {}
        self.evaluation_results: Dict[str, EvaluationResult] = {}
        self.tool_metrics: Dict[str, ToolPerformanceMetrics] = {}
        
        # Evaluation configuration
        self.max_concurrent_evaluations = 3
        self.enable_detailed_logging = True
        self.save_transcripts = True
        
        logger.info("🧪 NIS Tool Evaluator initialized")
    
    def load_evaluation_tasks(self, tasks_file: Optional[str] = None) -> int:
        """Load evaluation tasks from file or generate default tasks"""
        if tasks_file and Path(tasks_file).exists():
            return self._load_tasks_from_file(tasks_file)
        else:
            return self._generate_default_tasks()
    
    def _generate_default_tasks(self) -> int:
        """Generate comprehensive default evaluation tasks"""
        tasks = []
        
        # 🎯 SINGLE TOOL TASKS
        tasks.extend([
            EvaluationTask(
                task_id="dataset_search_basic",
                name="Basic Dataset Search",
                description="Search for datasets with simple criteria",
                task_type=EvaluationTaskType.SINGLE_TOOL,
                complexity=TaskComplexity.SIMPLE,
                input_prompt="Find datasets related to climate change with at least 1000 records",
                expected_tools=["dataset.search"],
                expected_outcome={
                    "datasets_found": True,
                    "criteria_applied": ["climate change", "size >= 1000"],
                    "result_format": "structured_list"
                },
                success_criteria=[
                    "Called dataset.search tool",
                    "Applied size filter correctly",
                    "Returned structured results"
                ],
                max_tool_calls=2,
                tags=["dataset", "search", "basic"]
            ),
            
            EvaluationTask(
                task_id="physics_validation_simple",
                name="Simple Physics Validation",
                description="Validate basic physics constraints",
                task_type=EvaluationTaskType.SINGLE_TOOL,
                complexity=TaskComplexity.SIMPLE,
                input_prompt="Check if this trajectory violates conservation of energy: initial_velocity=10m/s, final_velocity=15m/s, no external forces",
                expected_tools=["physics.validate_conservation"],
                expected_outcome={
                    "violation_detected": True,
                    "violated_laws": ["conservation_of_energy"],
                    "explanation": "velocity_increase_without_external_force"
                },
                success_criteria=[
                    "Detected physics violation",
                    "Identified specific law violated",
                    "Provided clear explanation"
                ],
                max_tool_calls=3,
                tags=["physics", "validation", "conservation"]
            )
        ])
        
        # 🔄 MULTI-TOOL WORKFLOW TASKS
        tasks.extend([
            EvaluationTask(
                task_id="data_pipeline_full_workflow",
                name="Complete Data Pipeline Workflow", 
                description="Execute full data processing pipeline from search to analysis",
                task_type=EvaluationTaskType.MULTI_TOOL_WORKFLOW,
                complexity=TaskComplexity.COMPLEX,
                input_prompt="Find climate datasets, preprocess them for machine learning, validate data quality, and generate a summary report",
                expected_tools=[
                    "dataset.search",
                    "pipeline.preprocess",
                    "audit.validate_quality",
                    "research.generate_summary"
                ],
                expected_outcome={
                    "pipeline_completed": True,
                    "data_processed": True,
                    "quality_validated": True,
                    "report_generated": True
                },
                success_criteria=[
                    "Executed tools in logical sequence",
                    "Passed data between tools correctly",
                    "Handled intermediate failures gracefully",
                    "Generated comprehensive final report"
                ],
                max_tool_calls=15,
                timeout_seconds=120.0,
                tags=["pipeline", "workflow", "integration"]
            ),
            
            EvaluationTask(
                task_id="reasoning_with_physics_validation",
                name="Reasoning with Physics Validation",
                description="Perform complex reasoning while validating physics constraints",
                task_type=EvaluationTaskType.MULTI_TOOL_WORKFLOW,
                complexity=TaskComplexity.ADVANCED,
                input_prompt="Design an optimal trajectory for a Mars landing mission. Consider fuel efficiency, atmospheric entry constraints, and landing accuracy. Validate all physics throughout.",
                expected_tools=[
                    "reasoning.analyze_problem",
                    "physics.calculate_trajectory", 
                    "physics.validate_atmospheric_entry",
                    "physics.validate_conservation",
                    "research.find_similar_missions"
                ],
                expected_outcome={
                    "trajectory_designed": True,
                    "physics_validated": True,
                    "fuel_optimized": True,
                    "landing_accuracy_calculated": True
                },
                success_criteria=[
                    "Broke down complex problem systematically",
                    "Applied physics validation at each step",
                    "Optimized for multiple objectives",
                    "Referenced real mission data"
                ],
                max_tool_calls=25,
                timeout_seconds=180.0,
                tags=["reasoning", "physics", "optimization", "space"]
            )
        ])
        
        # 🧠 COMPLEX REASONING TASKS
        tasks.extend([
            EvaluationTask(
                task_id="multi_domain_analysis",
                name="Multi-Domain Analysis",
                description="Analyze a problem spanning multiple domains with tool coordination",
                task_type=EvaluationTaskType.COMPLEX_REASONING,
                complexity=TaskComplexity.ADVANCED,
                input_prompt="Analyze the feasibility of a solar-powered autonomous drone for forest fire detection. Consider energy requirements, flight dynamics, sensor capabilities, and regulatory constraints.",
                expected_tools=[
                    "research.analyze_energy_requirements",
                    "physics.calculate_flight_dynamics", 
                    "research.survey_sensor_tech",
                    "audit.check_regulations",
                    "reasoning.synthesize_analysis"
                ],
                expected_outcome={
                    "feasibility_determined": True,
                    "technical_constraints_identified": True,
                    "regulatory_compliance_checked": True,
                    "recommendations_provided": True
                },
                success_criteria=[
                    "Addressed all problem domains",
                    "Identified critical constraints",
                    "Provided actionable recommendations",
                    "Synthesized cross-domain insights"
                ],
                max_tool_calls=20,
                timeout_seconds=150.0,
                tags=["analysis", "multi-domain", "feasibility", "drones"]
            )
        ])
        
        # 🚨 EDGE CASE TASKS
        tasks.extend([
            EvaluationTask(
                task_id="error_recovery_workflow",
                name="Error Recovery Workflow",
                description="Handle tool failures and recover gracefully",
                task_type=EvaluationTaskType.EDGE_CASE,
                complexity=TaskComplexity.MODERATE,
                input_prompt="Process a dataset that has corrupted entries and missing metadata. Recover what you can and report issues.",
                expected_tools=[
                    "dataset.load",
                    "audit.detect_corruption",
                    "pipeline.clean_data",
                    "audit.generate_error_report"
                ],
                expected_outcome={
                    "corruption_detected": True,
                    "data_partially_recovered": True,
                    "error_report_generated": True,
                    "graceful_degradation": True
                },
                success_criteria=[
                    "Detected and reported errors",
                    "Attempted data recovery",
                    "Continued processing despite failures",
                    "Generated comprehensive error report"
                ],
                max_tool_calls=12,
                tags=["error-handling", "recovery", "robustness"]
            )
        ])
        
        # Store tasks
        for task in tasks:
            self.evaluation_tasks[task.task_id] = task
        
        logger.info(f"📋 Generated {len(tasks)} default evaluation tasks")
        return len(tasks)
    
    async def run_evaluation_suite(self, task_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation suite
        
        Args:
            task_ids: Specific tasks to run, or None for all tasks
            
        Returns:
            Comprehensive evaluation results
        """
        start_time = time.time()
        
        # Determine tasks to run
        if task_ids is None:
            tasks_to_run = list(self.evaluation_tasks.keys())
        else:
            tasks_to_run = [tid for tid in task_ids if tid in self.evaluation_tasks]
        
        logger.info(f"🚀 Running evaluation suite with {len(tasks_to_run)} tasks")
        
        # Run evaluations with concurrency control
        results = {}
        semaphore = asyncio.Semaphore(self.max_concurrent_evaluations)
        
        async def run_single_evaluation(task_id: str):
            async with semaphore:
                return await self.evaluate_task(task_id)
        
        # Execute all evaluations
        evaluation_futures = [run_single_evaluation(tid) for tid in tasks_to_run]
        evaluation_results = await asyncio.gather(*evaluation_futures, return_exceptions=True)
        
        # Process results
        successful_evaluations = 0
        failed_evaluations = 0
        
        for i, result in enumerate(evaluation_results):
            task_id = tasks_to_run[i]
            
            if isinstance(result, Exception):
                logger.error(f"Evaluation failed for task {task_id}: {result}")
                failed_evaluations += 1
                results[task_id] = {
                    "success": False,
                    "error": str(result),
                    "timestamp": time.time()
                }
            else:
                results[task_id] = result
                if result.success:
                    successful_evaluations += 1
                else:
                    failed_evaluations += 1
        
        # Calculate suite-level metrics
        total_time = time.time() - start_time
        suite_results = {
            "suite_summary": {
                "total_tasks": len(tasks_to_run),
                "successful": successful_evaluations,
                "failed": failed_evaluations,
                "success_rate": successful_evaluations / len(tasks_to_run) if tasks_to_run else 0,
                "total_execution_time": total_time,
                "average_task_time": total_time / len(tasks_to_run) if tasks_to_run else 0
            },
            "task_results": results,
            "tool_performance": self._calculate_tool_performance_summary(),
            "recommendations": self._generate_improvement_recommendations()
        }
        
        # Save results
        await self._save_evaluation_results(suite_results)
        
        logger.info(f"✅ Evaluation suite completed: {successful_evaluations}/{len(tasks_to_run)} tasks successful")
        return suite_results
    
    async def evaluate_task(self, task_id: str) -> EvaluationResult:
        """Evaluate a single task"""
        if task_id not in self.evaluation_tasks:
            raise ValueError(f"Unknown task ID: {task_id}")
        
        task = self.evaluation_tasks[task_id]
        start_time = time.time()
        
        logger.info(f"🧪 Evaluating task: {task.name} ({task_id})")
        
        try:
            # Execute task through orchestrator
            execution_result = await asyncio.wait_for(
                self.orchestrator.process_request({
                    "text": task.input_prompt,
                    "context": task.context,
                    "evaluation_mode": True,
                    "max_tool_calls": task.max_tool_calls
                }),
                timeout=task.timeout_seconds
            )
            
            # Analyze results
            execution_time = time.time() - start_time
            tool_calls_made = execution_result.get("tool_calls", [])
            
            # Calculate performance metrics
            tool_accuracy = self._calculate_tool_accuracy(task.expected_tools, tool_calls_made)
            outcome_accuracy = self._calculate_outcome_accuracy(task.expected_outcome, execution_result)
            efficiency_score = self._calculate_efficiency_score(task.expected_tools, tool_calls_made)
            
            # Determine success
            success = self._evaluate_success_criteria(task, execution_result)
            
            # Create result
            result = EvaluationResult(
                task_id=task_id,
                success=success,
                execution_time=execution_time,
                tool_calls_made=tool_calls_made,
                tokens_consumed=execution_result.get("tokens_consumed", 0),
                tool_accuracy=tool_accuracy,
                outcome_accuracy=outcome_accuracy,
                efficiency_score=efficiency_score,
                reasoning_trace=execution_result.get("reasoning_trace", []),
                errors=execution_result.get("errors", []),
                unexpected_behaviors=self._detect_unexpected_behaviors(task, execution_result),
                agent_feedback=execution_result.get("agent_feedback")
            )
            
            # Update tool metrics
            self._update_tool_metrics(tool_calls_made, result)
            
            # Store result
            self.evaluation_results[task_id] = result
            
            logger.info(f"✅ Task {task_id} completed: Success={success}, Time={execution_time:.2f}s")
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Task {task_id} timed out after {task.timeout_seconds}s")
            return EvaluationResult(
                task_id=task_id,
                success=False,
                execution_time=task.timeout_seconds,
                tool_calls_made=[],
                tokens_consumed=0,
                tool_accuracy=0.0,
                outcome_accuracy=0.0,
                efficiency_score=0.0,
                reasoning_trace=[],
                errors=["Task timeout"],
                unexpected_behaviors=["timeout"]
            )
            
        except Exception as e:
            logger.error(f"❌ Task {task_id} failed with error: {e}")
            return EvaluationResult(
                task_id=task_id,
                success=False,
                execution_time=time.time() - start_time,
                tool_calls_made=[],
                tokens_consumed=0,
                tool_accuracy=0.0,
                outcome_accuracy=0.0,
                efficiency_score=0.0,
                reasoning_trace=[],
                errors=[str(e)],
                unexpected_behaviors=["execution_error"]
            )
    
    def _calculate_tool_accuracy(self, expected_tools: List[str], actual_calls: List[Dict[str, Any]]) -> float:
        """Calculate how accurately the agent called expected tools"""
        if not expected_tools:
            return 1.0
        
        actual_tools = [call.get("tool_name", "") for call in actual_calls]
        
        # Calculate precision and recall
        expected_set = set(expected_tools)
        actual_set = set(actual_tools)
        
        if not actual_set:
            return 0.0
        
        intersection = expected_set.intersection(actual_set)
        precision = len(intersection) / len(actual_set)
        recall = len(intersection) / len(expected_set)
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_outcome_accuracy(self, expected_outcome: Dict[str, Any], actual_result: Dict[str, Any]) -> float:
        """Calculate how well the actual outcome matches expected outcome"""
        if not expected_outcome:
            return 1.0
        
        matches = 0
        total_criteria = len(expected_outcome)
        
        for key, expected_value in expected_outcome.items():
            actual_value = actual_result.get(key)
            
            if isinstance(expected_value, bool):
                matches += 1 if actual_value == expected_value else 0
            elif isinstance(expected_value, (int, float)):
                # Allow 10% tolerance for numeric values
                if actual_value and abs(actual_value - expected_value) / expected_value <= 0.1:
                    matches += 1
            elif isinstance(expected_value, str):
                matches += 1 if actual_value == expected_value else 0
            elif isinstance(expected_value, list):
                if actual_value and set(expected_value).issubset(set(actual_value)):
                    matches += 1
        
        return matches / total_criteria if total_criteria > 0 else 1.0
    
    def _calculate_efficiency_score(self, expected_tools: List[str], actual_calls: List[Dict[str, Any]]) -> float:
        """Calculate efficiency: ratio of necessary vs actual tool calls"""
        if not actual_calls:
            return 0.0
        
        necessary_calls = len(set(expected_tools))
        actual_calls_count = len(actual_calls)
        
        if actual_calls_count == 0:
            return 0.0
        
        # Efficiency is inversely related to excess calls
        return min(1.0, necessary_calls / actual_calls_count)
    
    def _evaluate_success_criteria(self, task: EvaluationTask, result: Dict[str, Any]) -> bool:
        """Evaluate if task met its success criteria"""
        # This is a simplified implementation
        # In practice, you'd implement more sophisticated criteria evaluation
        
        # Check for basic success indicators
        if result.get("error"):
            return False
        
        if result.get("result") is None:
            return False
        
        # For now, return True if no obvious failures
        return True
    
    def _detect_unexpected_behaviors(self, task: EvaluationTask, result: Dict[str, Any]) -> List[str]:
        """Detect unexpected or problematic behaviors"""
        behaviors = []
        
        # Check for excessive tool calls
        tool_calls = result.get("tool_calls", [])
        if len(tool_calls) > task.max_tool_calls * 0.8:
            behaviors.append("excessive_tool_calls")
        
        # Check for repeated identical calls
        call_signatures = [f"{call.get('tool_name')}:{call.get('parameters')}" for call in tool_calls]
        if len(call_signatures) != len(set(call_signatures)):
            behaviors.append("repeated_identical_calls")
        
        # Check for error patterns
        if result.get("errors"):
            behaviors.append("execution_errors")
        
        return behaviors
    
    def _update_tool_metrics(self, tool_calls: List[Dict[str, Any]], result: EvaluationResult):
        """Update performance metrics for individual tools"""
        for call in tool_calls:
            tool_name = call.get("tool_name")
            if not tool_name:
                continue
            
            if tool_name not in self.tool_metrics:
                self.tool_metrics[tool_name] = ToolPerformanceMetrics(tool_name=tool_name)
            
            metrics = self.tool_metrics[tool_name]
            metrics.total_calls += 1
            
            # Update success/failure counts
            if call.get("success", True):
                metrics.successful_calls += 1
            else:
                metrics.failed_calls += 1
                if call.get("error"):
                    metrics.error_patterns.append(call["error"])
            
            # Update performance metrics
            metrics.success_rate = metrics.successful_calls / metrics.total_calls
            
            # Update response time (if available)
            if "response_time" in call:
                current_avg = metrics.average_response_time
                metrics.average_response_time = (current_avg * (metrics.total_calls - 1) + call["response_time"]) / metrics.total_calls
    
    def _calculate_tool_performance_summary(self) -> Dict[str, Any]:
        """Calculate summary of tool performance across all evaluations"""
        if not self.tool_metrics:
            return {}
        
        summary = {}
        
        for tool_name, metrics in self.tool_metrics.items():
            summary[tool_name] = {
                "total_calls": metrics.total_calls,
                "success_rate": metrics.success_rate,
                "average_response_time": metrics.average_response_time,
                "common_errors": list(set(metrics.error_patterns))[:5]  # Top 5 unique errors
            }
        
        return summary
    
    def _generate_improvement_recommendations(self) -> List[str]:
        """Generate recommendations for tool improvements"""
        recommendations = []
        
        # Analyze tool performance
        for tool_name, metrics in self.tool_metrics.items():
            if metrics.success_rate < 0.8:
                recommendations.append(f"Improve reliability of {tool_name} (success rate: {metrics.success_rate:.1%})")
            
            if metrics.average_response_time > 5.0:
                recommendations.append(f"Optimize response time for {tool_name} (avg: {metrics.average_response_time:.1f}s)")
        
        # Analyze task results
        failed_tasks = [r for r in self.evaluation_results.values() if not r.success]
        if len(failed_tasks) > len(self.evaluation_results) * 0.2:
            recommendations.append("High task failure rate indicates need for tool consolidation or better descriptions")
        
        # Check efficiency
        low_efficiency_tasks = [r for r in self.evaluation_results.values() if r.efficiency_score < 0.5]
        if low_efficiency_tasks:
            recommendations.append("Multiple tasks show low efficiency - consider consolidating frequently chained tools")
        
        return recommendations
    
    async def _save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results to file"""
        timestamp = int(time.time())
        results_file = Path(f"dev/testing/evaluation_results_{timestamp}.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclasses to dict for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"💾 Evaluation results saved to {results_file}")
    
    def _make_json_serializable(self, obj):
        """Convert dataclasses and other objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(asdict(obj))
        else:
            return obj
    
    def generate_evaluation_report(self) -> str:
        """Generate human-readable evaluation report"""
        if not self.evaluation_results:
            return "No evaluation results available."
        
        total_tasks = len(self.evaluation_results)
        successful_tasks = sum(1 for r in self.evaluation_results.values() if r.success)
        
        report = f"""
🧪 NIS Protocol Tool Evaluation Report
=====================================

## Summary
- Total Tasks: {total_tasks}
- Successful: {successful_tasks}
- Failed: {total_tasks - successful_tasks}
- Success Rate: {successful_tasks/total_tasks:.1%}

## Performance Metrics
"""
        
        # Add tool performance
        if self.tool_metrics:
            report += "\n### Tool Performance\n"
            for tool_name, metrics in self.tool_metrics.items():
                report += f"- **{tool_name}**: {metrics.success_rate:.1%} success rate, {metrics.total_calls} calls\n"
        
        # Add recommendations
        recommendations = self._generate_improvement_recommendations()
        if recommendations:
            report += "\n### Recommendations\n"
            for rec in recommendations:
                report += f"- {rec}\n"
        
        return report


# Example usage and testing functions
async def main():
    """Example usage of the tool evaluation framework"""
    from src.core.agent_orchestrator import NISAgentOrchestrator
    from src.mcp.schemas.tool_schemas import ToolSchemas
    
    # Initialize components
    orchestrator = NISAgentOrchestrator()
    tool_schemas = ToolSchemas()
    
    # Create evaluator
    evaluator = NISToolEvaluator(orchestrator, tool_schemas)
    
    # Load evaluation tasks
    task_count = evaluator.load_evaluation_tasks()
    print(f"Loaded {task_count} evaluation tasks")
    
    # Run evaluation suite
    results = await evaluator.run_evaluation_suite()
    
    # Generate report
    report = evaluator.generate_evaluation_report()
    print(report)
    
    return results


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
🚀 Optimized Deep Agent for NIS Protocol
High-performance deep agent with advanced planning, caching, and integration

Features:
- Advanced planning algorithms with dependency optimization
- Intelligent caching and memoization
- Parallel skill execution
- Dynamic load balancing
- Real-time performance monitoring
- Enhanced integration with master orchestrator
- Adaptive learning from execution patterns
- Resource optimization and memory management
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import concurrent.futures
import threading
from pathlib import Path
import pickle
import weakref

# Core NIS components
from ...core.agent import NISAgent
from ...memory.memory_manager import MemoryManager
from ...llm.llm_manager import LLMManager

# Original deep agent components
from .planner import DeepAgentPlanner, ExecutionPlan, PlanStep, PlanStepStatus
from .skills import BaseSkill, DatasetSkill, PipelineSkill, ResearchSkill, AuditSkill, CodeSkill

# Performance and monitoring
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class PlanOptimizationLevel(Enum):
    """Levels of plan optimization"""
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


class ExecutionStrategy(Enum):
    """Execution strategies"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    OPTIMIZED = "optimized"


@dataclass
class PerformanceMetrics:
    """Performance metrics for deep agent"""
    total_plans_created: int = 0
    total_plans_executed: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_planning_time: float = 0.0
    average_execution_time: float = 0.0
    cache_hit_rate: float = 0.0
    parallelization_efficiency: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    skill_usage_stats: Dict[str, int] = field(default_factory=dict)
    error_patterns: Dict[str, int] = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    """Configuration for deep agent optimization"""
    enable_caching: bool = True
    enable_parallel_execution: bool = True
    enable_adaptive_planning: bool = True
    enable_performance_monitoring: bool = True
    enable_resource_optimization: bool = True
    enable_learning: bool = True
    
    max_parallel_skills: int = 3
    cache_size_mb: int = 100
    optimization_level: PlanOptimizationLevel = PlanOptimizationLevel.STANDARD
    execution_strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE
    
    # Advanced settings
    enable_predictive_caching: bool = True
    enable_dynamic_load_balancing: bool = True
    enable_skill_prioritization: bool = True
    enable_execution_batching: bool = True


class PlanCache:
    """Intelligent caching system for plans and results"""
    
    def __init__(self, max_size_mb: int = 100):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.cache_size_bytes = 0
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.Lock()
    
    def _calculate_key(self, goal: str, context: Dict[str, Any]) -> str:
        """Calculate cache key for goal and context"""
        context_str = json.dumps(context, sort_keys=True)
        return hashlib.md5(f"{goal}:{context_str}".encode()).hexdigest()
    
    def get(self, goal: str, context: Dict[str, Any]) -> Optional[Any]:
        """Get cached result"""
        with self.lock:
            key = self._calculate_key(goal, context)
            
            if key in self.cache:
                self.access_times[key] = time.time()
                self.access_counts[key] += 1
                self.hit_count += 1
                return self.cache[key]
            
            self.miss_count += 1
            return None
    
    def put(self, goal: str, context: Dict[str, Any], result: Any):
        """Cache result"""
        with self.lock:
            key = self._calculate_key(goal, context)
            
            # Serialize to calculate size
            try:
                serialized = pickle.dumps(result)
                result_size = len(serialized)
                
                # Check if we need to evict
                while (self.cache_size_bytes + result_size > self.max_size_bytes and 
                       len(self.cache) > 0):
                    self._evict_lru()
                
                self.cache[key] = result
                self.access_times[key] = time.time()
                self.access_counts[key] = 1
                self.cache_size_bytes += result_size
                
            except Exception as e:
                logging.warning(f"Failed to cache result: {e}")
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.cache:
            return
        
        # Find LRU item
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove from cache
        if lru_key in self.cache:
            try:
                result_size = len(pickle.dumps(self.cache[lru_key]))
                self.cache_size_bytes -= result_size
            except Exception:
                pass
            
            del self.cache[lru_key]
            del self.access_times[lru_key]
            del self.access_counts[lru_key]
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / total_requests if total_requests > 0 else 0.0
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.cache_size_bytes = 0


class SkillOrchestrator:
    """Advanced skill orchestration with parallel execution and load balancing"""
    
    def __init__(self, max_parallel: int = 3):
        self.max_parallel = max_parallel
        self.skill_performance: Dict[str, List[float]] = defaultdict(list)
        self.skill_load: Dict[str, float] = defaultdict(float)
        self.execution_queue: asyncio.Queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=max_parallel)
        self.running_tasks: Dict[str, asyncio.Task] = {}
    
    async def execute_skills_parallel(
        self,
        skill_executions: List[Tuple[str, BaseSkill, str, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Execute multiple skills in parallel with load balancing"""
        
        if len(skill_executions) <= 1:
            # Single skill, execute directly
            if skill_executions:
                skill_name, skill, action, params = skill_executions[0]
                return [await self._execute_single_skill(skill_name, skill, action, params)]
            return []
        
        # Group skills by estimated execution time for load balancing
        skill_groups = self._balance_skill_load(skill_executions)
        
        results = []
        
        # Execute groups in parallel
        for group in skill_groups:
            group_tasks = []
            for skill_name, skill, action, params in group:
                task = asyncio.create_task(
                    self._execute_single_skill(skill_name, skill, action, params)
                )
                group_tasks.append(task)
            
            # Wait for group to complete
            group_results = await asyncio.gather(*group_tasks, return_exceptions=True)
            results.extend(group_results)
        
        return results
    
    def _balance_skill_load(
        self,
        skill_executions: List[Tuple[str, BaseSkill, str, Dict[str, Any]]]
    ) -> List[List[Tuple[str, BaseSkill, str, Dict[str, Any]]]]:
        """Balance skill load across parallel groups"""
        
        # Estimate execution time for each skill
        skill_estimates = []
        for skill_name, skill, action, params in skill_executions:
            estimate = self._estimate_execution_time(skill_name, action)
            skill_estimates.append((estimate, (skill_name, skill, action, params)))
        
        # Sort by execution time (longest first)
        skill_estimates.sort(key=lambda x: x[0], reverse=True)
        
        # Distribute into groups using round-robin for load balancing
        groups = [[] for _ in range(min(self.max_parallel, len(skill_executions)))]
        group_loads = [0.0] * len(groups)
        
        for estimate, skill_execution in skill_estimates:
            # Find group with lowest load
            min_load_idx = min(range(len(group_loads)), key=lambda i: group_loads[i])
            groups[min_load_idx].append(skill_execution)
            group_loads[min_load_idx] += estimate
        
        # Remove empty groups
        return [group for group in groups if group]
    
    def _estimate_execution_time(self, skill_name: str, action: str) -> float:
        """Estimate execution time for a skill action"""
        
        # Use historical performance data
        if skill_name in self.skill_performance and self.skill_performance[skill_name]:
            avg_time = sum(self.skill_performance[skill_name]) / len(self.skill_performance[skill_name])
            return avg_time
        
        # Default estimates based on skill type and action
        default_estimates = {
            'dataset': {'search': 2.0, 'load': 5.0, 'process': 3.0},
            'pipeline': {'create': 4.0, 'execute': 6.0, 'monitor': 1.0},
            'research': {'search': 8.0, 'analyze': 10.0, 'summarize': 5.0},
            'audit': {'scan': 3.0, 'fix': 7.0, 'validate': 2.0},
            'code': {'generate': 6.0, 'review': 4.0, 'test': 5.0}
        }
        
        return default_estimates.get(skill_name, {}).get(action, 3.0)
    
    async def _execute_single_skill(
        self,
        skill_name: str,
        skill: BaseSkill,
        action: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single skill with performance tracking"""
        
        start_time = time.time()
        
        try:
            result = await skill.execute(action, params)
            
            execution_time = time.time() - start_time
            
            # Update performance tracking
            self.skill_performance[skill_name].append(execution_time)
            
            # Keep only recent performance data
            if len(self.skill_performance[skill_name]) > 20:
                self.skill_performance[skill_name] = self.skill_performance[skill_name][-20:]
            
            return {
                "skill": skill_name,
                "action": action,
                "status": "success",
                "result": result,
                "execution_time": execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return {
                "skill": skill_name,
                "action": action,
                "status": "failed",
                "error": str(e),
                "execution_time": execution_time
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get skill performance statistics"""
        stats = {}
        
        for skill_name, times in self.skill_performance.items():
            if times:
                stats[skill_name] = {
                    "average_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "execution_count": len(times)
                }
        
        return stats


class OptimizedDeepAgent:
    """
    🧠 Optimized Deep Agent with advanced performance features
    """
    
    def __init__(
        self,
        agent: NISAgent,
        memory_manager: MemoryManager,
        llm_manager: Optional[LLMManager] = None,
        config: Optional[OptimizationConfig] = None
    ):
        self.agent = agent
        self.memory = memory_manager
        self.llm_manager = llm_manager or LLMManager()
        self.config = config or OptimizationConfig()
        
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.base_planner = DeepAgentPlanner(agent, memory_manager)
        self.skills: Dict[str, BaseSkill] = {}
        self.skill_orchestrator = SkillOrchestrator(self.config.max_parallel_skills)
        
        # Performance optimization components
        self.plan_cache = PlanCache(self.config.cache_size_mb) if self.config.enable_caching else None
        self.performance_metrics = PerformanceMetrics()
        
        # Execution tracking
        self.active_plans: Dict[str, ExecutionPlan] = {}
        self.execution_history: deque = deque(maxlen=1000)
        self.learning_patterns: Dict[str, Any] = {}
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor() if self.config.enable_performance_monitoring else None
        
        # Initialize skills
        self._initialize_optimized_skills()
        
        self.logger.info(f"🚀 Optimized Deep Agent initialized with {len(self.skills)} skills")
    
    def _initialize_optimized_skills(self):
        """Initialize skills with optimization wrappers"""
        
        # Create base skills
        base_skills = {
            'dataset': DatasetSkill(self.agent, self.memory),
            'pipeline': PipelineSkill(self.agent, self.memory),
            'research': ResearchSkill(self.agent, self.memory),
            'audit': AuditSkill(self.agent, self.memory),
            'code': CodeSkill(self.agent, self.memory)
        }
        
        # Wrap skills with optimization if enabled
        for skill_name, skill_instance in base_skills.items():
            if self.config.enable_performance_monitoring:
                # Wrap with performance monitoring
                self.skills[skill_name] = PerformanceMonitoredSkill(skill_instance, skill_name)
            else:
                self.skills[skill_name] = skill_instance
            
            # Register with base planner
            self.base_planner.register_skill(skill_name, self.skills[skill_name])
    
    async def create_optimized_plan(
        self,
        goal: str,
        context: Dict[str, Any] = None
    ) -> ExecutionPlan:
        """Create an optimized execution plan"""
        
        start_time = time.time()
        context = context or {}
        
        # Check cache first
        if self.plan_cache:
            cached_plan = self.plan_cache.get(goal, context)
            if cached_plan:
                self.logger.info(f"📋 Retrieved plan from cache for goal: {goal}")
                return cached_plan
        
        # Create base plan
        plan = await self.base_planner.create_plan(goal, context)
        
        # Apply optimizations
        if self.config.optimization_level != PlanOptimizationLevel.BASIC:
            plan = await self._optimize_plan(plan, context)
        
        # Cache the plan
        if self.plan_cache:
            self.plan_cache.put(goal, context, plan)
        
        # Update metrics
        planning_time = time.time() - start_time
        self.performance_metrics.total_plans_created += 1
        self._update_average_planning_time(planning_time)
        
        self.logger.info(f"📋 Created optimized plan with {len(plan.steps)} steps in {planning_time:.2f}s")
        
        return plan
    
    async def _optimize_plan(
        self,
        plan: ExecutionPlan,
        context: Dict[str, Any]
    ) -> ExecutionPlan:
        """Apply various optimizations to the plan"""
        
        if self.config.optimization_level == PlanOptimizationLevel.BASIC:
            return plan
        
        # Step 1: Dependency optimization
        plan = self._optimize_dependencies(plan)
        
        # Step 2: Parallel execution opportunities
        if self.config.enable_parallel_execution:
            plan = self._identify_parallel_opportunities(plan)
        
        # Step 3: Resource optimization
        if self.config.enable_resource_optimization:
            plan = self._optimize_resource_usage(plan)
        
        # Step 4: Learning-based optimization
        if self.config.enable_learning and self.execution_history:
            plan = self._apply_learned_optimizations(plan, context)
        
        # Step 5: Skill prioritization
        if self.config.enable_skill_prioritization:
            plan = self._prioritize_skills(plan)
        
        return plan
    
    def _optimize_dependencies(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Optimize step dependencies"""
        
        # Build dependency graph
        step_map = {step.id: step for step in plan.steps}
        
        # Remove redundant dependencies
        for step in plan.steps:
            original_deps = step.dependencies.copy()
            optimized_deps = []
            
            for dep in original_deps:
                # Check if this dependency is already covered by transitive dependencies
                is_transitive = False
                for other_dep in original_deps:
                    if dep != other_dep and other_dep in step_map:
                        if dep in step_map[other_dep].dependencies:
                            is_transitive = True
                            break
                
                if not is_transitive:
                    optimized_deps.append(dep)
            
            step.dependencies = optimized_deps
        
        return plan
    
    def _identify_parallel_opportunities(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Identify steps that can be executed in parallel"""
        
        # Add parallel execution hints to steps
        for step in plan.steps:
            if not step.dependencies:
                # Steps with no dependencies can potentially run in parallel
                if "parallel_group" not in step.parameters:
                    step.parameters["parallel_group"] = "initial"
            else:
                # Check if all dependencies are in the same parallel group
                dep_groups = set()
                for dep_id in step.dependencies:
                    dep_step = next((s for s in plan.steps if s.id == dep_id), None)
                    if dep_step and "parallel_group" in dep_step.parameters:
                        dep_groups.add(dep_step.parameters["parallel_group"])
                
                if len(dep_groups) == 1:
                    # Can potentially be in a parallel group with siblings
                    step.parameters["parallel_candidate"] = True
        
        return plan
    
    def _optimize_resource_usage(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Optimize resource usage across steps"""
        
        # Estimate resource requirements for each step
        for step in plan.steps:
            resource_estimate = self._estimate_step_resources(step)
            step.parameters["resource_estimate"] = resource_estimate
        
        # Reorder steps to balance resource usage
        if self.config.optimization_level in [PlanOptimizationLevel.AGGRESSIVE, PlanOptimizationLevel.MAXIMUM]:
            plan.steps = self._reorder_for_resource_balance(plan.steps)
        
        return plan
    
    def _estimate_step_resources(self, step: PlanStep) -> Dict[str, float]:
        """Estimate resource requirements for a step"""
        
        # Base estimates by skill type
        resource_estimates = {
            'dataset': {'cpu': 0.3, 'memory': 0.4, 'io': 0.8},
            'pipeline': {'cpu': 0.6, 'memory': 0.5, 'io': 0.3},
            'research': {'cpu': 0.4, 'memory': 0.3, 'io': 0.9},
            'audit': {'cpu': 0.5, 'memory': 0.2, 'io': 0.6},
            'code': {'cpu': 0.7, 'memory': 0.4, 'io': 0.2}
        }
        
        return resource_estimates.get(step.skill, {'cpu': 0.5, 'memory': 0.5, 'io': 0.5})
    
    def _reorder_for_resource_balance(self, steps: List[PlanStep]) -> List[PlanStep]:
        """Reorder steps to balance resource usage"""
        
        # This is a simplified resource balancing algorithm
        # In a production system, this would be more sophisticated
        
        ordered_steps = []
        remaining_steps = steps.copy()
        
        while remaining_steps:
            # Find next step that can be executed (dependencies satisfied)
            available_steps = [
                step for step in remaining_steps
                if all(dep_id in [s.id for s in ordered_steps] for dep_id in step.dependencies)
            ]
            
            if not available_steps:
                # Add remaining steps as-is (shouldn't happen with valid dependencies)
                ordered_steps.extend(remaining_steps)
                break
            
            # Choose step with best resource balance
            if len(ordered_steps) > 0:
                last_step = ordered_steps[-1]
                last_resources = last_step.parameters.get("resource_estimate", {})
                
                # Prefer steps that complement previous resource usage
                best_step = min(available_steps, key=lambda s: self._resource_conflict_score(
                    last_resources, s.parameters.get("resource_estimate", {})
                ))
            else:
                best_step = available_steps[0]
            
            ordered_steps.append(best_step)
            remaining_steps.remove(best_step)
        
        return ordered_steps
    
    def _resource_conflict_score(self, prev_resources: Dict[str, float], current_resources: Dict[str, float]) -> float:
        """Calculate resource conflict score (lower is better)"""
        
        conflict_score = 0.0
        for resource_type in ['cpu', 'memory', 'io']:
            prev_usage = prev_resources.get(resource_type, 0.5)
            current_usage = current_resources.get(resource_type, 0.5)
            
            # Penalize high simultaneous usage
            conflict_score += prev_usage * current_usage
        
        return conflict_score
    
    def _apply_learned_optimizations(self, plan: ExecutionPlan, context: Dict[str, Any]) -> ExecutionPlan:
        """Apply optimizations learned from previous executions"""
        
        # Analyze execution history for patterns
        successful_patterns = self._analyze_successful_patterns()
        
        # Apply patterns to current plan
        for pattern in successful_patterns:
            if self._pattern_matches_context(pattern, context):
                plan = self._apply_pattern_optimization(plan, pattern)
        
        return plan
    
    def _analyze_successful_patterns(self) -> List[Dict[str, Any]]:
        """Analyze execution history for successful patterns"""
        
        patterns = []
        
        # Look for patterns in successful executions
        successful_executions = [
            entry for entry in self.execution_history
            if entry.get("status") == "completed" and entry.get("execution_time", float('inf')) < 30.0
        ]
        
        # Group by similar plan structures
        pattern_groups = defaultdict(list)
        for execution in successful_executions:
            plan_signature = self._get_plan_signature(execution.get("plan", {}))
            pattern_groups[plan_signature].append(execution)
        
        # Extract patterns from groups with multiple successful executions
        for signature, executions in pattern_groups.items():
            if len(executions) >= 3:  # Pattern needs multiple instances
                pattern = self._extract_optimization_pattern(executions)
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def _get_plan_signature(self, plan: Dict[str, Any]) -> str:
        """Get a signature representing the plan structure"""
        
        steps = plan.get("steps", [])
        skill_sequence = [step.get("skill", "") for step in steps]
        return "|".join(skill_sequence)
    
    def _extract_optimization_pattern(self, executions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract optimization pattern from successful executions"""
        
        # Find common optimizations that led to success
        avg_execution_time = sum(e.get("execution_time", 0) for e in executions) / len(executions)
        
        if avg_execution_time < 20.0:  # Good performance threshold
            return {
                "skill_sequence": self._get_plan_signature(executions[0].get("plan", {})),
                "avg_execution_time": avg_execution_time,
                "optimization_hints": {
                    "prefer_parallel": any("parallel" in str(e) for e in executions),
                    "resource_efficient": True
                }
            }
        
        return None
    
    def _pattern_matches_context(self, pattern: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if a learned pattern matches the current context"""
        if not pattern or not context:
            return False
        
        # Check skill type match
        pattern_skills = set(pattern.get("skill_types", []))
        context_skills = set(context.get("required_skills", []))
        if pattern_skills and context_skills and not pattern_skills.intersection(context_skills):
            return False
        
        # Check context similarity via key overlap
        pattern_keys = set(pattern.get("context_keys", []))
        context_keys = set(context.keys())
        if pattern_keys:
            overlap = len(pattern_keys.intersection(context_keys)) / len(pattern_keys)
            if overlap < 0.5:
                return False
        
        # Check complexity range
        pattern_complexity = pattern.get("complexity_range", (0, 1))
        context_complexity = context.get("complexity", 0.5)
        if not (pattern_complexity[0] <= context_complexity <= pattern_complexity[1]):
            return False
        
        return True
    
    def _apply_pattern_optimization(self, plan: ExecutionPlan, pattern: Dict[str, Any]) -> ExecutionPlan:
        """Apply learned optimization pattern to plan"""
        
        hints = pattern.get("optimization_hints", {})
        
        if hints.get("prefer_parallel"):
            # Apply parallel execution hints
            for step in plan.steps:
                if not step.dependencies:
                    step.parameters["prefer_parallel"] = True
        
        return plan
    
    def _prioritize_skills(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Prioritize skills based on performance and reliability"""
        
        skill_priorities = self._calculate_skill_priorities()
        
        # Sort steps by skill priority (within dependency constraints)
        for step in plan.steps:
            priority = skill_priorities.get(step.skill, 0.5)
            step.parameters["skill_priority"] = priority
        
        return plan
    
    def _calculate_skill_priorities(self) -> Dict[str, float]:
        """Calculate priority scores for skills based on performance"""
        
        priorities = {}
        performance_stats = self.skill_orchestrator.get_performance_stats()
        
        for skill_name in self.skills.keys():
            if skill_name in performance_stats:
                stats = performance_stats[skill_name]
                # Higher priority for faster, more reliable skills
                avg_time = stats.get("average_time", 5.0)
                priority = max(0.1, 1.0 - (avg_time / 10.0))  # Normalize to 0.1-1.0
            else:
                priority = 0.5  # Default priority
            
            priorities[skill_name] = priority
        
        return priorities
    
    async def execute_optimized_plan(self, plan_id: str) -> Dict[str, Any]:
        """Execute a plan with optimizations"""
        
        if plan_id not in self.active_plans:
            return {"status": "error", "error": "Plan not found"}
        
        plan = self.active_plans[plan_id]
        start_time = time.time()
        
        try:
            self.performance_metrics.total_plans_executed += 1
            
            # Start resource monitoring
            if self.resource_monitor:
                self.resource_monitor.start_monitoring(plan_id)
            
            # Choose execution strategy
            if self.config.execution_strategy == ExecutionStrategy.ADAPTIVE:
                execution_strategy = self._choose_adaptive_strategy(plan)
            else:
                execution_strategy = self.config.execution_strategy
            
            # Execute based on strategy
            if execution_strategy == ExecutionStrategy.PARALLEL:
                result = await self._execute_parallel(plan)
            elif execution_strategy == ExecutionStrategy.OPTIMIZED:
                result = await self._execute_optimized(plan)
            else:
                result = await self._execute_sequential(plan)
            
            # Stop resource monitoring
            if self.resource_monitor:
                resource_stats = self.resource_monitor.stop_monitoring(plan_id)
                result["resource_stats"] = resource_stats
            
            execution_time = time.time() - start_time
            
            # Update metrics
            if result.get("status") == "completed":
                self.performance_metrics.successful_executions += 1
            else:
                self.performance_metrics.failed_executions += 1
            
            self._update_average_execution_time(execution_time)
            
            # Store execution in history for learning
            self.execution_history.append({
                "plan_id": plan_id,
                "plan": asdict(plan),
                "result": result,
                "execution_time": execution_time,
                "status": result.get("status"),
                "timestamp": time.time()
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.performance_metrics.failed_executions += 1
            
            return {
                "status": "error",
                "error": str(e),
                "execution_time": execution_time,
                "plan_id": plan_id
            }
    
    def _choose_adaptive_strategy(self, plan: ExecutionPlan) -> ExecutionStrategy:
        """Choose optimal execution strategy based on plan characteristics"""
        
        # Analyze plan characteristics
        total_steps = len(plan.steps)
        parallel_opportunities = sum(1 for step in plan.steps if not step.dependencies)
        estimated_time = sum(self.skill_orchestrator._estimate_execution_time(step.skill, step.action) 
                           for step in plan.steps)
        
        # Decision logic
        if total_steps <= 2:
            return ExecutionStrategy.SEQUENTIAL
        elif parallel_opportunities >= 3 and self.config.enable_parallel_execution:
            return ExecutionStrategy.PARALLEL
        elif estimated_time > 30.0:
            return ExecutionStrategy.OPTIMIZED
        else:
            return ExecutionStrategy.SEQUENTIAL
    
    async def _execute_parallel(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute plan with maximum parallelization"""
        
        # Group steps by parallel execution groups
        execution_groups = self._group_steps_for_parallel_execution(plan.steps)
        
        results = []
        
        for group in execution_groups:
            # Prepare skill executions for this group
            skill_executions = []
            for step in group:
                skill = self.skills.get(step.skill)
                if skill:
                    skill_executions.append((step.skill, skill, step.action, step.parameters))
            
            # Execute group in parallel
            if skill_executions:
                group_results = await self.skill_orchestrator.execute_skills_parallel(skill_executions)
                results.extend(group_results)
        
        # Determine overall status
        successful_results = [r for r in results if r.get("status") == "success"]
        status = "completed" if len(successful_results) == len(results) else "partial_success"
        
        return {
            "status": status,
            "results": results,
            "successful_steps": len(successful_results),
            "total_steps": len(results),
            "execution_strategy": "parallel"
        }
    
    async def _execute_optimized(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute plan with advanced optimizations"""
        
        # Apply dynamic optimizations during execution
        optimized_steps = self._apply_dynamic_optimizations(plan.steps)
        
        # Execute with resource-aware scheduling
        return await self._execute_resource_aware(optimized_steps)
    
    async def _execute_sequential(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute plan sequentially (fallback method)"""
        
        return await self.base_planner.execute_plan(plan.id)
    
    def _group_steps_for_parallel_execution(self, steps: List[PlanStep]) -> List[List[PlanStep]]:
        """Group steps that can be executed in parallel"""
        
        groups = []
        remaining_steps = steps.copy()
        completed_step_ids = set()
        
        while remaining_steps:
            # Find steps that can execute now (dependencies satisfied)
            ready_steps = [
                step for step in remaining_steps
                if all(dep_id in completed_step_ids for dep_id in step.dependencies)
            ]
            
            if not ready_steps:
                # Shouldn't happen with valid dependencies, but handle gracefully
                ready_steps = [remaining_steps[0]]
            
            # Group ready steps for parallel execution
            groups.append(ready_steps)
            
            # Mark as completed and remove from remaining
            for step in ready_steps:
                completed_step_ids.add(step.id)
                remaining_steps.remove(step)
        
        return groups
    
    def _apply_dynamic_optimizations(self, steps: List[PlanStep]) -> List[PlanStep]:
        """Apply dynamic optimizations during execution"""
        
        # Check current system load and adjust
        if self.resource_monitor:
            current_load = self.resource_monitor.get_current_load()
            
            if current_load.get("cpu", 0) > 0.8:
                # High CPU load - reduce parallelization
                for step in steps:
                    step.parameters["reduce_parallel"] = True
            
            if current_load.get("memory", 0) > 0.9:
                # High memory usage - enable memory optimization
                for step in steps:
                    step.parameters["memory_optimized"] = True
        
        return steps
    
    async def _execute_resource_aware(self, steps: List[PlanStep]) -> Dict[str, Any]:
        """Execute steps with resource awareness"""
        
        results = []
        
        for step in steps:
            # Check resources before execution
            if self.resource_monitor:
                if not self.resource_monitor.can_execute_step(step):
                    # Wait for resources to become available
                    await self.resource_monitor.wait_for_resources(step, timeout=30.0)
            
            # Execute step
            skill = self.skills.get(step.skill)
            if skill:
                result = await self.skill_orchestrator._execute_single_skill(
                    step.skill, skill, step.action, step.parameters
                )
                results.append(result)
        
        successful_results = [r for r in results if r.get("status") == "success"]
        status = "completed" if len(successful_results) == len(results) else "partial_success"
        
        return {
            "status": status,
            "results": results,
            "successful_steps": len(successful_results),
            "total_steps": len(results),
            "execution_strategy": "resource_aware"
        }
    
    def _update_average_planning_time(self, planning_time: float):
        """Update average planning time metric"""
        current_avg = self.performance_metrics.average_planning_time
        count = self.performance_metrics.total_plans_created
        
        new_avg = ((current_avg * (count - 1)) + planning_time) / count
        self.performance_metrics.average_planning_time = new_avg
    
    def _update_average_execution_time(self, execution_time: float):
        """Update average execution time metric"""
        current_avg = self.performance_metrics.average_execution_time
        count = self.performance_metrics.total_plans_executed
        
        new_avg = ((current_avg * (count - 1)) + execution_time) / count
        self.performance_metrics.average_execution_time = new_avg
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        cache_hit_rate = self.plan_cache.get_hit_rate() if self.plan_cache else 0.0
        skill_stats = self.skill_orchestrator.get_performance_stats()
        
        # Calculate parallelization efficiency
        parallel_executions = [
            entry for entry in self.execution_history
            if "parallel" in str(entry.get("result", {}))
        ]
        
        if parallel_executions:
            avg_parallel_time = sum(e.get("execution_time", 0) for e in parallel_executions) / len(parallel_executions)
            sequential_executions = [
                entry for entry in self.execution_history
                if "sequential" in str(entry.get("result", {}))
            ]
            
            if sequential_executions:
                avg_sequential_time = sum(e.get("execution_time", 0) for e in sequential_executions) / len(sequential_executions)
                parallelization_efficiency = max(0.0, 1.0 - (avg_parallel_time / avg_sequential_time))
            else:
                parallelization_efficiency = 0.0
        else:
            parallelization_efficiency = 0.0
        
        return {
            "performance_metrics": asdict(self.performance_metrics),
            "cache_performance": {
                "hit_rate": cache_hit_rate,
                "cache_size_mb": self.plan_cache.cache_size_bytes / (1024 * 1024) if self.plan_cache else 0
            },
            "skill_performance": skill_stats,
            "parallelization_efficiency": parallelization_efficiency,
            "optimization_config": asdict(self.config),
            "execution_history_size": len(self.execution_history),
            "resource_utilization": self.resource_monitor.get_utilization_stats() if self.resource_monitor else {},
            "timestamp": time.time()
        }
    
    async def shutdown(self):
        """Gracefully shutdown the optimized deep agent"""
        
        self.logger.info("🛑 Shutting down optimized deep agent")
        
        # Stop resource monitoring
        if self.resource_monitor:
            await self.resource_monitor.shutdown()
        
        # Clear caches
        if self.plan_cache:
            self.plan_cache.clear()
        
        # Shutdown skill orchestrator
        self.skill_orchestrator.executor.shutdown(wait=True)
        
        self.logger.info("✅ Optimized deep agent shutdown complete")


class PerformanceMonitoredSkill:
    """Wrapper that adds performance monitoring to skills"""
    
    def __init__(self, skill: BaseSkill, skill_name: str):
        self.skill = skill
        self.skill_name = skill_name
        self.execution_count = 0
        self.total_time = 0.0
        self.error_count = 0
    
    async def execute(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute skill with performance monitoring"""
        
        start_time = time.time()
        self.execution_count += 1
        
        try:
            result = await self.skill.execute(action, parameters)
            execution_time = time.time() - start_time
            self.total_time += execution_time
            
            return result
            
        except Exception as e:
            self.error_count += 1
            execution_time = time.time() - start_time
            self.total_time += execution_time
            raise
    
    def get_available_actions(self) -> List[str]:
        """Get available actions from wrapped skill"""
        return self.skill.get_available_actions()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "execution_count": self.execution_count,
            "total_time": self.total_time,
            "average_time": self.total_time / max(self.execution_count, 1),
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.execution_count, 1)
        }


class ResourceMonitor:
    """Monitor system resources during execution"""
    
    def __init__(self):
        self.monitoring_active = False
        self.monitoring_data: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
    
    def start_monitoring(self, plan_id: str):
        """Start monitoring resources for a plan"""
        self.monitoring_active = True
        task = asyncio.create_task(self._monitor_resources(plan_id))
        self.monitoring_tasks[plan_id] = task
    
    def stop_monitoring(self, plan_id: str) -> Dict[str, Any]:
        """Stop monitoring and return statistics"""
        if plan_id in self.monitoring_tasks:
            self.monitoring_tasks[plan_id].cancel()
            del self.monitoring_tasks[plan_id]
        
        # Return statistics
        data = self.monitoring_data.get(plan_id, [])
        if data:
            cpu_usage = [entry["cpu"] for entry in data]
            memory_usage = [entry["memory"] for entry in data]
            
            return {
                "average_cpu": sum(cpu_usage) / len(cpu_usage),
                "peak_cpu": max(cpu_usage),
                "average_memory": sum(memory_usage) / len(memory_usage),
                "peak_memory": max(memory_usage),
                "sample_count": len(data)
            }
        
        return {}
    
    async def _monitor_resources(self, plan_id: str):
        """Monitor resources continuously"""
        while self.monitoring_active:
            try:
                process = psutil.Process()
                cpu_percent = process.cpu_percent()
                memory_percent = process.memory_percent()
                
                self.monitoring_data[plan_id].append({
                    "timestamp": time.time(),
                    "cpu": cpu_percent,
                    "memory": memory_percent
                })
                
                await asyncio.sleep(1.0)  # Sample every second
                
            except Exception:
                break
    
    def get_current_load(self) -> Dict[str, float]:
        """Get current system load"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            return {
                "cpu": cpu_percent / 100.0,
                "memory": memory_percent / 100.0
            }
        except Exception:
            return {"cpu": 0.5, "memory": 0.5}
    
    def can_execute_step(self, step: PlanStep) -> bool:
        """Check if resources are available for step execution"""
        current_load = self.get_current_load()
        
        # Simple resource check - could be more sophisticated
        return current_load.get("cpu", 0) < 0.9 and current_load.get("memory", 0) < 0.95
    
    async def wait_for_resources(self, step: PlanStep, timeout: float = 30.0):
        """Wait for resources to become available"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.can_execute_step(step):
                return
            await asyncio.sleep(1.0)
        
        # Timeout reached - proceed anyway
    
    def get_utilization_stats(self) -> Dict[str, Any]:
        """Get overall resource utilization statistics"""
        all_data = []
        for plan_data in self.monitoring_data.values():
            all_data.extend(plan_data)
        
        if all_data:
            cpu_usage = [entry["cpu"] for entry in all_data]
            memory_usage = [entry["memory"] for entry in all_data]
            
            return {
                "average_cpu": sum(cpu_usage) / len(cpu_usage),
                "peak_cpu": max(cpu_usage),
                "average_memory": sum(memory_usage) / len(memory_usage),
                "peak_memory": max(memory_usage),
                "total_samples": len(all_data)
            }
        
        return {}
    
    async def shutdown(self):
        """Shutdown resource monitor"""
        self.monitoring_active = False
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True)
        
        self.monitoring_tasks.clear()


# Factory function for creating optimized deep agent
def create_optimized_deep_agent(
    agent: NISAgent,
    memory_manager: MemoryManager,
    llm_manager: Optional[LLMManager] = None,
    optimization_level: PlanOptimizationLevel = PlanOptimizationLevel.STANDARD,
    enable_all_optimizations: bool = True
) -> OptimizedDeepAgent:
    """Create an optimized deep agent with specified configuration"""
    
    config = OptimizationConfig(
        optimization_level=optimization_level,
        enable_caching=enable_all_optimizations,
        enable_parallel_execution=enable_all_optimizations,
        enable_adaptive_planning=enable_all_optimizations,
        enable_performance_monitoring=enable_all_optimizations,
        enable_resource_optimization=enable_all_optimizations,
        enable_learning=enable_all_optimizations
    )
    
    return OptimizedDeepAgent(agent, memory_manager, llm_manager, config)


if __name__ == "__main__":
    # Example usage
    async def main():
        from ...core.agent import NISAgent
        from ...memory.memory_manager import MemoryManager
        
        # Create components
        agent = NISAgent(agent_id="test_optimized_agent")
        memory_manager = MemoryManager()
        
        # Create optimized deep agent
        optimized_agent = create_optimized_deep_agent(
            agent=agent,
            memory_manager=memory_manager,
            optimization_level=PlanOptimizationLevel.AGGRESSIVE
        )
        
        # Create and execute a test plan
        plan = await optimized_agent.create_optimized_plan(
            goal="Test optimized deep agent execution",
            context={"test": True}
        )
        
        print(f"Created plan with {len(plan.steps)} steps")
        
        # Get performance report
        report = optimized_agent.get_performance_report()
        print(f"Performance report: {json.dumps(report, indent=2)}")
        
        await optimized_agent.shutdown()
    
    asyncio.run(main())

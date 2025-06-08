"""
NIS Protocol Goal Priority Manager

This module manages the prioritization and dynamic adjustment of goals
across the AGI system, ensuring optimal resource allocation and focus.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import heapq

from ...memory.memory_manager import MemoryManager


class PriorityLevel(Enum):
    """Priority levels for goals"""
    CRITICAL = "critical"      # Must be addressed immediately
    HIGH = "high"             # Important, should be addressed soon
    MEDIUM = "medium"         # Standard priority
    LOW = "low"              # Can be deferred
    BACKGROUND = "background" # Ongoing, low-resource activities


@dataclass
class PriorityFactors:
    """Factors that influence goal priority"""
    urgency: float           # Time-sensitive nature (0.0-1.0)
    importance: float        # Strategic importance (0.0-1.0)
    resource_availability: float  # Available resources (0.0-1.0)
    alignment_score: float   # Alignment with main objectives (0.0-1.0)
    success_probability: float    # Likelihood of success (0.0-1.0)
    emotional_motivation: float   # Emotional drive (0.0-1.0)


@dataclass
class PrioritizedGoal:
    """A goal with computed priority information"""
    goal_id: str
    goal_data: Dict[str, Any]
    priority_score: float
    priority_level: PriorityLevel
    priority_factors: PriorityFactors
    last_updated: float
    dependencies: List[str]
    resource_requirements: Dict[str, float]


class GoalPriorityManager:
    """Manages dynamic prioritization of goals across the AGI system.
    
    This manager provides:
    - Multi-criteria goal prioritization
    - Dynamic priority adjustment based on context
    - Resource-aware priority management
    - Goal scheduling and ordering
    """
    
    def __init__(self):
        """Initialize the goal priority manager."""
        self.logger = logging.getLogger("nis.goal_priority_manager")
        self.memory = MemoryManager()
        
        # Priority management
        self.prioritized_goals: Dict[str, PrioritizedGoal] = {}
        self.priority_queue: List[Tuple[float, str]] = []  # (negative_priority, goal_id)
        self.resource_allocations: Dict[str, Dict[str, float]] = {}
        
        # Priority configuration
        self.priority_weights = {
            "urgency": 0.25,
            "importance": 0.30,
            "resource_availability": 0.20,
            "alignment_score": 0.15,
            "success_probability": 0.05,
            "emotional_motivation": 0.05
        }
        
        # Dynamic adjustment parameters
        self.context_sensitivity = 0.8
        self.priority_decay_rate = 0.05
        self.recomputation_interval = 300.0  # 5 minutes
        
        self.logger.info("GoalPriorityManager initialized")
    
    def add_goal(
        self,
        goal_id: str,
        goal_data: Dict[str, Any],
        initial_factors: Optional[PriorityFactors] = None
    ) -> PrioritizedGoal:
        """Add a new goal to the priority management system.
        
        Args:
            goal_id: Unique identifier for the goal
            goal_data: Goal information and metadata
            initial_factors: Initial priority factors
            
        Returns:
            Prioritized goal object
        """
        # TODO: Implement sophisticated goal analysis
        # Should analyze:
        # - Goal type and characteristics
        # - Resource requirements
        # - Dependencies and prerequisites
        # - Strategic alignment
        # - Context relevance
        
        self.logger.info(f"Adding goal to priority management: {goal_id}")
        
        # Calculate or use provided priority factors
        if initial_factors is None:
            initial_factors = self._analyze_goal_factors(goal_data)
        
        # Compute priority score
        priority_score = self._compute_priority_score(initial_factors)
        priority_level = self._determine_priority_level(priority_score)
        
        # Create prioritized goal
        prioritized_goal = PrioritizedGoal(
            goal_id=goal_id,
            goal_data=goal_data,
            priority_score=priority_score,
            priority_level=priority_level,
            priority_factors=initial_factors,
            last_updated=time.time(),
            dependencies=goal_data.get("dependencies", []),
            resource_requirements=goal_data.get("resource_requirements", {})
        )
        
        # Add to management system
        self.prioritized_goals[goal_id] = prioritized_goal
        heapq.heappush(self.priority_queue, (-priority_score, goal_id))
        
        self.logger.info(f"Goal {goal_id} added with priority {priority_level.value}")
        return prioritized_goal
    
    def update_goal_priority(
        self,
        goal_id: str,
        context_changes: Dict[str, Any],
        force_recompute: bool = False
    ) -> Optional[PrioritizedGoal]:
        """Update the priority of a specific goal based on context changes.
        
        Args:
            goal_id: ID of goal to update
            context_changes: Changes in context that affect priority
            force_recompute: Force complete priority recomputation
            
        Returns:
            Updated prioritized goal or None if not found
        """
        # TODO: Implement sophisticated priority update
        # Should consider:
        # - Context changes and their impact
        # - Resource availability changes
        # - Goal progress and status
        # - External factors and events
        # - Time-based priority adjustments
        
        if goal_id not in self.prioritized_goals:
            self.logger.warning(f"Goal {goal_id} not found for priority update")
            return None
        
        prioritized_goal = self.prioritized_goals[goal_id]
        
        self.logger.info(f"Updating priority for goal: {goal_id}")
        
        # Update priority factors based on context
        updated_factors = self._update_priority_factors(
            prioritized_goal.priority_factors,
            context_changes
        )
        
        # Recompute priority score
        new_priority_score = self._compute_priority_score(updated_factors)
        new_priority_level = self._determine_priority_level(new_priority_score)
        
        # Update goal
        prioritized_goal.priority_factors = updated_factors
        prioritized_goal.priority_score = new_priority_score
        prioritized_goal.priority_level = new_priority_level
        prioritized_goal.last_updated = time.time()
        
        # Update priority queue
        self._rebuild_priority_queue()
        
        return prioritized_goal
    
    def get_next_priority_goal(
        self,
        available_resources: Dict[str, float],
        context: Dict[str, Any]
    ) -> Optional[PrioritizedGoal]:
        """Get the next highest priority goal that can be executed.
        
        Args:
            available_resources: Currently available resources
            context: Current execution context
            
        Returns:
            Next goal to execute or None if none available
        """
        # TODO: Implement sophisticated goal selection
        # Should consider:
        # - Resource requirements vs availability
        # - Goal dependencies and prerequisites
        # - Context compatibility
        # - Execution constraints
        # - Multi-goal coordination
        
        self.logger.info("Selecting next priority goal for execution")
        
        # Check goals in priority order
        temp_queue = self.priority_queue.copy()
        
        while temp_queue:
            neg_priority, goal_id = heapq.heappop(temp_queue)
            
            if goal_id not in self.prioritized_goals:
                continue
            
            prioritized_goal = self.prioritized_goals[goal_id]
            
            # Check if goal can be executed
            if self._can_execute_goal(prioritized_goal, available_resources, context):
                self.logger.info(f"Selected goal for execution: {goal_id}")
                return prioritized_goal
        
        self.logger.info("No executable goals found with current resources")
        return None
    
    def get_priority_ordered_goals(
        self,
        limit: Optional[int] = None,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[PrioritizedGoal]:
        """Get goals ordered by priority with optional filtering.
        
        Args:
            limit: Maximum number of goals to return
            filter_criteria: Criteria to filter goals
            
        Returns:
            Priority-ordered list of goals
        """
        # TODO: Implement sophisticated filtering and ordering
        # Should support:
        # - Various filter criteria
        # - Flexible ordering options
        # - Resource-based filtering
        # - Context-based filtering
        
        self.logger.info(f"Getting priority-ordered goals (limit: {limit})")
        
        # Get all goals ordered by priority
        ordered_goals = []
        temp_queue = self.priority_queue.copy()
        
        while temp_queue and (limit is None or len(ordered_goals) < limit):
            neg_priority, goal_id = heapq.heappop(temp_queue)
            
            if goal_id in self.prioritized_goals:
                goal = self.prioritized_goals[goal_id]
                
                # Apply filters if provided
                if filter_criteria is None or self._matches_criteria(goal, filter_criteria):
                    ordered_goals.append(goal)
        
        return ordered_goals
    
    def recompute_all_priorities(
        self,
        current_context: Dict[str, Any],
        available_resources: Dict[str, float]
    ) -> None:
        """Recompute priorities for all goals based on current context.
        
        Args:
            current_context: Current system context
            available_resources: Currently available resources
        """
        # TODO: Implement comprehensive priority recomputation
        # Should consider:
        # - Global context changes
        # - Resource availability changes
        # - Goal interactions and dependencies
        # - System-wide optimization
        # - Performance metrics
        
        self.logger.info("Recomputing all goal priorities")
        
        updated_goals = []
        
        for goal_id, prioritized_goal in self.prioritized_goals.items():
            # Update factors based on current context
            updated_factors = self._recompute_factors_for_context(
                prioritized_goal,
                current_context,
                available_resources
            )
            
            # Update priority score and level
            new_priority_score = self._compute_priority_score(updated_factors)
            new_priority_level = self._determine_priority_level(new_priority_score)
            
            # Update goal
            prioritized_goal.priority_factors = updated_factors
            prioritized_goal.priority_score = new_priority_score
            prioritized_goal.priority_level = new_priority_level
            prioritized_goal.last_updated = time.time()
            
            updated_goals.append(prioritized_goal)
        
        # Rebuild priority queue
        self._rebuild_priority_queue()
        
        self.logger.info(f"Recomputed priorities for {len(updated_goals)} goals")
    
    def remove_goal(self, goal_id: str) -> bool:
        """Remove a goal from priority management.
        
        Args:
            goal_id: ID of goal to remove
            
        Returns:
            True if goal was removed, False if not found
        """
        if goal_id in self.prioritized_goals:
            del self.prioritized_goals[goal_id]
            self._rebuild_priority_queue()
            self.logger.info(f"Removed goal from priority management: {goal_id}")
            return True
        return False
    
    def get_priority_statistics(self) -> Dict[str, Any]:
        """Get statistics about current goal priorities.
        
        Returns:
            Priority statistics and metrics
        """
        # TODO: Implement comprehensive statistics
        # Should include:
        # - Priority distribution
        # - Resource allocation analysis
        # - Goal completion rates
        # - Priority adjustment frequency
        # - System performance metrics
        
        if not self.prioritized_goals:
            return {"total_goals": 0}
        
        goals = list(self.prioritized_goals.values())
        
        # Basic statistics
        priority_scores = [goal.priority_score for goal in goals]
        
        return {
            "total_goals": len(goals),
            "average_priority": sum(priority_scores) / len(priority_scores),
            "max_priority": max(priority_scores),
            "min_priority": min(priority_scores),
            "priority_distribution": self._get_priority_distribution(),
            "resource_utilization": self._calculate_resource_utilization(),
            "last_recomputation": max(goal.last_updated for goal in goals) if goals else 0
        }
    
    def _analyze_goal_factors(self, goal_data: Dict[str, Any]) -> PriorityFactors:
        """Analyze a goal to determine its priority factors.
        
        Args:
            goal_data: Goal information
            
        Returns:
            Computed priority factors
        """
        # TODO: Implement sophisticated factor analysis
        # Should analyze:
        # - Goal type and characteristics
        # - Time constraints and deadlines
        # - Strategic importance
        # - Resource requirements
        # - Success likelihood
        # - Emotional significance
        
        # Placeholder implementation
        return PriorityFactors(
            urgency=goal_data.get("urgency", 0.5),
            importance=goal_data.get("importance", 0.5),
            resource_availability=0.8,  # TODO: Calculate actual availability
            alignment_score=goal_data.get("alignment", 0.7),
            success_probability=goal_data.get("success_probability", 0.6),
            emotional_motivation=goal_data.get("emotional_drive", 0.4)
        )
    
    def _compute_priority_score(self, factors: PriorityFactors) -> float:
        """Compute priority score from priority factors.
        
        Args:
            factors: Priority factors
            
        Returns:
            Computed priority score (0.0-1.0)
        """
        score = (
            factors.urgency * self.priority_weights["urgency"] +
            factors.importance * self.priority_weights["importance"] +
            factors.resource_availability * self.priority_weights["resource_availability"] +
            factors.alignment_score * self.priority_weights["alignment_score"] +
            factors.success_probability * self.priority_weights["success_probability"] +
            factors.emotional_motivation * self.priority_weights["emotional_motivation"]
        )
        
        return min(max(score, 0.0), 1.0)
    
    def _determine_priority_level(self, priority_score: float) -> PriorityLevel:
        """Determine priority level from score.
        
        Args:
            priority_score: Computed priority score
            
        Returns:
            Priority level
        """
        if priority_score >= 0.9:
            return PriorityLevel.CRITICAL
        elif priority_score >= 0.7:
            return PriorityLevel.HIGH
        elif priority_score >= 0.5:
            return PriorityLevel.MEDIUM
        elif priority_score >= 0.3:
            return PriorityLevel.LOW
        else:
            return PriorityLevel.BACKGROUND
    
    def _update_priority_factors(
        self,
        current_factors: PriorityFactors,
        context_changes: Dict[str, Any]
    ) -> PriorityFactors:
        """Update priority factors based on context changes.
        
        Args:
            current_factors: Current priority factors
            context_changes: Changes in context
            
        Returns:
            Updated priority factors
        """
        # TODO: Implement sophisticated factor updates
        # Should handle various types of context changes
        
        # Placeholder implementation
        updated_factors = PriorityFactors(
            urgency=current_factors.urgency,
            importance=current_factors.importance,
            resource_availability=context_changes.get("resource_change", current_factors.resource_availability),
            alignment_score=current_factors.alignment_score,
            success_probability=current_factors.success_probability,
            emotional_motivation=current_factors.emotional_motivation
        )
        
        return updated_factors
    
    def _can_execute_goal(
        self,
        goal: PrioritizedGoal,
        available_resources: Dict[str, float],
        context: Dict[str, Any]
    ) -> bool:
        """Check if a goal can be executed with available resources.
        
        Args:
            goal: Goal to check
            available_resources: Available resources
            context: Current context
            
        Returns:
            True if goal can be executed
        """
        # TODO: Implement sophisticated executability check
        # Should check:
        # - Resource requirements vs availability
        # - Prerequisites and dependencies
        # - Context compatibility
        # - Execution constraints
        
        # Simple resource check (placeholder)
        for resource, required in goal.resource_requirements.items():
            if available_resources.get(resource, 0) < required:
                return False
        
        return True
    
    def _matches_criteria(
        self,
        goal: PrioritizedGoal,
        criteria: Dict[str, Any]
    ) -> bool:
        """Check if goal matches filter criteria.
        
        Args:
            goal: Goal to check
            criteria: Filter criteria
            
        Returns:
            True if goal matches criteria
        """
        # TODO: Implement comprehensive criteria matching
        # Should support various filter types
        
        # Simple matching (placeholder)
        for key, value in criteria.items():
            if key == "priority_level" and goal.priority_level != value:
                return False
            elif key == "min_priority" and goal.priority_score < value:
                return False
        
        return True
    
    def _rebuild_priority_queue(self) -> None:
        """Rebuild the priority queue from current goals."""
        self.priority_queue = [
            (-goal.priority_score, goal_id)
            for goal_id, goal in self.prioritized_goals.items()
        ]
        heapq.heapify(self.priority_queue)
    
    def _get_priority_distribution(self) -> Dict[str, int]:
        """Get distribution of goals by priority level."""
        distribution = {level.value: 0 for level in PriorityLevel}
        
        for goal in self.prioritized_goals.values():
            distribution[goal.priority_level.value] += 1
        
        return distribution
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate current resource utilization by goals."""
        # TODO: Implement actual resource utilization calculation
        return {"cpu": 0.6, "memory": 0.7, "network": 0.3}
    
    def _recompute_factors_for_context(
        self,
        goal: PrioritizedGoal,
        context: Dict[str, Any],
        resources: Dict[str, float]
    ) -> PriorityFactors:
        """Recompute priority factors for current context.
        
        Args:
            goal: Goal to recompute factors for
            context: Current context
            resources: Available resources
            
        Returns:
            Recomputed priority factors
        """
        # TODO: Implement context-aware factor recomputation
        # For now, return existing factors (placeholder)
        return goal.priority_factors 
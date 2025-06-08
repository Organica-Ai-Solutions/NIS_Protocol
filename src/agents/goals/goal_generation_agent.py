"""
NIS Protocol Goal Generation Agent

This agent autonomously generates goals based on curiosity, context, emotional state,
and environmental conditions. It represents the drive and motivation system of the AGI.
"""

import time
import random
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from ...core.agent import NISAgent, NISLayer
from ...memory.memory_manager import MemoryManager


class GoalType(Enum):
    """Types of goals the system can generate"""
    EXPLORATION = "exploration"
    LEARNING = "learning"
    OPTIMIZATION = "optimization"
    PROBLEM_SOLVING = "problem_solving"
    CREATIVITY = "creativity"
    MAINTENANCE = "maintenance"
    SOCIAL = "social"


class GoalPriority(Enum):
    """Priority levels for goals"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class Goal:
    """Represents a generated goal"""
    goal_id: str
    goal_type: GoalType
    priority: GoalPriority
    description: str
    target_outcome: str
    success_criteria: Dict[str, Any]
    estimated_effort: float
    deadline: Optional[float]
    dependencies: List[str]
    context: Dict[str, Any]
    created_timestamp: float
    emotional_motivation: Dict[str, float]


class GoalGenerationAgent(NISAgent):
    """Agent that autonomously generates goals for the system.
    
    This agent:
    - Monitors system state and context
    - Generates new goals based on curiosity and needs
    - Prioritizes goals based on multiple factors
    - Maintains goal lifecycle and dependencies
    """
    
    def __init__(
        self,
        agent_id: str = "goal_generation_agent",
        description: str = "Autonomous goal generation and management agent"
    ):
        super().__init__(agent_id, NISLayer.REASONING, description)
        self.logger = logging.getLogger(f"nis.{agent_id}")
        
        # Initialize memory and goal tracking
        self.memory = MemoryManager()
        self.active_goals: Dict[str, Goal] = {}
        self.completed_goals: List[Goal] = []
        self.goal_history: List[Dict[str, Any]] = []
        
        # Goal generation parameters
        self.curiosity_threshold = 0.6
        self.max_active_goals = 10
        self.goal_generation_cooldown = 30.0  # seconds
        self.last_goal_generation = 0.0
        
        # Context tracking
        self.current_context: Dict[str, Any] = {}
        self.interest_areas: Set[str] = set()
        self.knowledge_gaps: List[str] = []
        
        self.logger.info(f"Initialized {self.__class__.__name__}")
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process goal generation requests and management operations.
        
        Args:
            message: Input message with goal operation
            
        Returns:
            Goal generation results
        """
        start_time = self._start_processing_timer()
        
        try:
            operation = message.get("operation", "generate_goals")
            
            if operation == "generate_goals":
                result = self._generate_goals(message)
            elif operation == "update_context":
                result = self._update_context(message)
            elif operation == "evaluate_progress":
                result = self._evaluate_goal_progress(message)
            elif operation == "complete_goal":
                result = self._complete_goal(message)
            elif operation == "prioritize_goals":
                result = self._prioritize_goals(message)
            elif operation == "get_active_goals":
                result = self._get_active_goals(message)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            # Update emotional state based on goal generation
            emotional_state = self._assess_goal_emotional_impact(result)
            
            response = self._create_response(
                "success",
                result,
                {"operation": operation, "active_goals": len(self.active_goals)},
                emotional_state
            )
            
        except Exception as e:
            self.logger.error(f"Error in goal generation: {str(e)}")
            response = self._create_response(
                "error",
                {"error": str(e)},
                {"operation": operation}
            )
        
        finally:
            self._end_processing_timer(start_time)
        
        return response
    
    def _generate_goals(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Generate new goals based on current context and state.
        
        Args:
            message: Message with generation parameters
            
        Returns:
            Generated goals and analysis
        """
        # Check if we should generate goals (cooldown and capacity)
        current_time = time.time()
        if (current_time - self.last_goal_generation < self.goal_generation_cooldown or
            len(self.active_goals) >= self.max_active_goals):
            return {
                "goals_generated": [],
                "reason": "Generation cooldown or capacity limit reached"
            }
        
        # Update context from message
        context_update = message.get("context", {})
        emotional_state = message.get("emotional_state", {})
        
        # Analyze current situation for goal generation opportunities
        generation_triggers = self._analyze_generation_triggers(context_update, emotional_state)
        
        # Generate goals based on triggers
        new_goals = []
        for trigger in generation_triggers:
            goal = self._create_goal_from_trigger(trigger, emotional_state)
            if goal:
                new_goals.append(goal)
                self.active_goals[goal.goal_id] = goal
        
        # Update generation timestamp
        self.last_goal_generation = current_time
        
        # Store goals in memory
        for goal in new_goals:
            self._store_goal_in_memory(goal)
        
        return {
            "goals_generated": [goal.__dict__ for goal in new_goals],
            "generation_triggers": generation_triggers,
            "total_active_goals": len(self.active_goals)
        }
    
    def _analyze_generation_triggers(
        self,
        context: Dict[str, Any],
        emotional_state: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Analyze context and emotional state for goal generation triggers.
        
        Args:
            context: Current context information
            emotional_state: Current emotional state
            
        Returns:
            List of goal generation triggers
        """
        triggers = []
        
        # Curiosity-driven triggers
        curiosity_level = emotional_state.get("curiosity", 0.0)
        if curiosity_level > self.curiosity_threshold:
            triggers.append({
                "type": "curiosity",
                "strength": curiosity_level,
                "focus": context.get("current_topic", "general"),
                "goal_type": GoalType.EXPLORATION
            })
        
        # Knowledge gap triggers
        unknown_concepts = context.get("unknown_concepts", [])
        if unknown_concepts:
            triggers.append({
                "type": "knowledge_gap",
                "strength": min(len(unknown_concepts) / 5.0, 1.0),
                "focus": unknown_concepts[0] if unknown_concepts else "general",
                "goal_type": GoalType.LEARNING
            })
        
        # Problem-solving triggers
        problems = context.get("detected_problems", [])
        if problems:
            triggers.append({
                "type": "problem_detected",
                "strength": 0.8,
                "focus": problems[0] if problems else "general",
                "goal_type": GoalType.PROBLEM_SOLVING
            })
        
        # Optimization triggers
        inefficiencies = context.get("inefficiencies", [])
        if inefficiencies:
            triggers.append({
                "type": "optimization_opportunity",
                "strength": 0.7,
                "focus": inefficiencies[0] if inefficiencies else "performance",
                "goal_type": GoalType.OPTIMIZATION
            })
        
        # Creative triggers (when boredom is high)
        boredom = emotional_state.get("boredom", 0.0)
        if boredom > 0.7:
            triggers.append({
                "type": "creative_stimulation",
                "strength": boredom,
                "focus": "creativity",
                "goal_type": GoalType.CREATIVITY
            })
        
        # Maintenance triggers (when satisfaction is low)
        satisfaction = emotional_state.get("satisfaction", 0.5)
        if satisfaction < 0.3:
            triggers.append({
                "type": "maintenance_need",
                "strength": 1.0 - satisfaction,
                "focus": "system_health",
                "goal_type": GoalType.MAINTENANCE
            })
        
        return triggers
    
    def _create_goal_from_trigger(
        self,
        trigger: Dict[str, Any],
        emotional_state: Dict[str, float]
    ) -> Optional[Goal]:
        """Create a specific goal from a generation trigger.
        
        Args:
            trigger: Goal generation trigger
            emotional_state: Current emotional state
            
        Returns:
            Generated goal or None if creation failed
        """
        goal_type = trigger["goal_type"]
        focus = trigger["focus"]
        strength = trigger["strength"]
        
        # Generate unique goal ID
        goal_id = f"{goal_type.value}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Determine priority based on trigger strength and type
        if strength > 0.8 or goal_type == GoalType.MAINTENANCE:
            priority = GoalPriority.HIGH
        elif strength > 0.6:
            priority = GoalPriority.MEDIUM
        else:
            priority = GoalPriority.LOW
        
        # Create goal-specific content
        if goal_type == GoalType.EXPLORATION:
            goal = self._create_exploration_goal(goal_id, focus, priority, emotional_state)
        elif goal_type == GoalType.LEARNING:
            goal = self._create_learning_goal(goal_id, focus, priority, emotional_state)
        elif goal_type == GoalType.PROBLEM_SOLVING:
            goal = self._create_problem_solving_goal(goal_id, focus, priority, emotional_state)
        elif goal_type == GoalType.OPTIMIZATION:
            goal = self._create_optimization_goal(goal_id, focus, priority, emotional_state)
        elif goal_type == GoalType.CREATIVITY:
            goal = self._create_creativity_goal(goal_id, focus, priority, emotional_state)
        elif goal_type == GoalType.MAINTENANCE:
            goal = self._create_maintenance_goal(goal_id, focus, priority, emotional_state)
        else:
            return None
        
        return goal
    
    def _create_exploration_goal(
        self,
        goal_id: str,
        focus: str,
        priority: GoalPriority,
        emotional_state: Dict[str, float]
    ) -> Goal:
        """Create an exploration goal."""
        return Goal(
            goal_id=goal_id,
            goal_type=GoalType.EXPLORATION,
            priority=priority,
            description=f"Explore and discover new information about {focus}",
            target_outcome=f"Increased knowledge and understanding of {focus}",
            success_criteria={
                "new_concepts_learned": 3,
                "exploration_depth": 0.7,
                "satisfaction_increase": 0.2
            },
            estimated_effort=0.5,
            deadline=time.time() + 3600,  # 1 hour
            dependencies=[],
            context={"focus_area": focus, "exploration_type": "knowledge_discovery"},
            created_timestamp=time.time(),
            emotional_motivation=emotional_state.copy()
        )
    
    def _create_learning_goal(
        self,
        goal_id: str,
        focus: str,
        priority: GoalPriority,
        emotional_state: Dict[str, float]
    ) -> Goal:
        """Create a learning goal."""
        return Goal(
            goal_id=goal_id,
            goal_type=GoalType.LEARNING,
            priority=priority,
            description=f"Learn and master the concept of {focus}",
            target_outcome=f"Comprehensive understanding of {focus}",
            success_criteria={
                "concept_mastery": 0.8,
                "can_explain": True,
                "can_apply": True
            },
            estimated_effort=0.7,
            deadline=time.time() + 7200,  # 2 hours
            dependencies=[],
            context={"subject": focus, "learning_type": "concept_mastery"},
            created_timestamp=time.time(),
            emotional_motivation=emotional_state.copy()
        )
    
    def _create_problem_solving_goal(
        self,
        goal_id: str,
        focus: str,
        priority: GoalPriority,
        emotional_state: Dict[str, float]
    ) -> Goal:
        """Create a problem-solving goal."""
        return Goal(
            goal_id=goal_id,
            goal_type=GoalType.PROBLEM_SOLVING,
            priority=priority,
            description=f"Solve the identified problem: {focus}",
            target_outcome=f"Resolution or significant progress on {focus}",
            success_criteria={
                "problem_understood": True,
                "solution_identified": True,
                "solution_effectiveness": 0.8
            },
            estimated_effort=0.8,
            deadline=time.time() + 5400,  # 1.5 hours
            dependencies=[],
            context={"problem": focus, "solving_approach": "systematic"},
            created_timestamp=time.time(),
            emotional_motivation=emotional_state.copy()
        )
    
    def _create_optimization_goal(
        self,
        goal_id: str,
        focus: str,
        priority: GoalPriority,
        emotional_state: Dict[str, float]
    ) -> Goal:
        """Create an optimization goal."""
        return Goal(
            goal_id=goal_id,
            goal_type=GoalType.OPTIMIZATION,
            priority=priority,
            description=f"Optimize and improve {focus}",
            target_outcome=f"Enhanced performance and efficiency of {focus}",
            success_criteria={
                "performance_improvement": 0.2,
                "efficiency_gain": 0.15,
                "resource_optimization": 0.1
            },
            estimated_effort=0.6,
            deadline=time.time() + 10800,  # 3 hours
            dependencies=[],
            context={"optimization_target": focus, "approach": "iterative"},
            created_timestamp=time.time(),
            emotional_motivation=emotional_state.copy()
        )
    
    def _create_creativity_goal(
        self,
        goal_id: str,
        focus: str,
        priority: GoalPriority,
        emotional_state: Dict[str, float]
    ) -> Goal:
        """Create a creativity goal."""
        return Goal(
            goal_id=goal_id,
            goal_type=GoalType.CREATIVITY,
            priority=priority,
            description=f"Generate creative solutions or ideas related to {focus}",
            target_outcome=f"Novel and innovative approaches to {focus}",
            success_criteria={
                "novelty_score": 0.7,
                "ideas_generated": 5,
                "originality": 0.6
            },
            estimated_effort=0.4,
            deadline=time.time() + 1800,  # 30 minutes
            dependencies=[],
            context={"creative_domain": focus, "approach": "divergent_thinking"},
            created_timestamp=time.time(),
            emotional_motivation=emotional_state.copy()
        )
    
    def _create_maintenance_goal(
        self,
        goal_id: str,
        focus: str,
        priority: GoalPriority,
        emotional_state: Dict[str, float]
    ) -> Goal:
        """Create a maintenance goal."""
        return Goal(
            goal_id=goal_id,
            goal_type=GoalType.MAINTENANCE,
            priority=GoalPriority.HIGH,  # Maintenance is always high priority
            description=f"Maintain and ensure proper functioning of {focus}",
            target_outcome=f"Optimal health and performance of {focus}",
            success_criteria={
                "system_health": 0.9,
                "performance_stability": 0.85,
                "error_reduction": 0.5
            },
            estimated_effort=0.3,
            deadline=time.time() + 900,  # 15 minutes
            dependencies=[],
            context={"maintenance_area": focus, "urgency": "high"},
            created_timestamp=time.time(),
            emotional_motivation=emotional_state.copy()
        )
    
    def _update_context(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Update the agent's context understanding."""
        new_context = message.get("context", {})
        self.current_context.update(new_context)
        
        # Update interest areas
        interests = new_context.get("interests", [])
        self.interest_areas.update(interests)
        
        # Update knowledge gaps
        gaps = new_context.get("knowledge_gaps", [])
        self.knowledge_gaps.extend(gaps)
        
        return {
            "context_updated": True,
            "current_context": self.current_context,
            "interest_areas": list(self.interest_areas),
            "knowledge_gaps": self.knowledge_gaps
        }
    
    def _evaluate_goal_progress(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate progress on active goals."""
        goal_id = message.get("goal_id")
        progress_data = message.get("progress", {})
        
        if goal_id not in self.active_goals:
            raise ValueError(f"Goal {goal_id} not found in active goals")
        
        goal = self.active_goals[goal_id]
        
        # Evaluate progress against success criteria
        progress_score = self._calculate_progress_score(goal, progress_data)
        
        # Check if goal is complete
        is_complete = progress_score >= 0.9
        
        if is_complete:
            self._complete_goal_internal(goal_id, progress_data)
        
        return {
            "goal_id": goal_id,
            "progress_score": progress_score,
            "is_complete": is_complete,
            "next_steps": self._suggest_next_steps(goal, progress_score)
        }
    
    def _complete_goal(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Mark a goal as completed."""
        goal_id = message.get("goal_id")
        completion_data = message.get("completion_data", {})
        
        return self._complete_goal_internal(goal_id, completion_data)
    
    def _complete_goal_internal(self, goal_id: str, completion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Internal method to complete a goal."""
        if goal_id not in self.active_goals:
            raise ValueError(f"Goal {goal_id} not found in active goals")
        
        goal = self.active_goals.pop(goal_id)
        self.completed_goals.append(goal)
        
        # Store completion in memory
        self.memory.store(
            f"completed_goal_{goal_id}",
            {
                "goal": goal.__dict__,
                "completion_data": completion_data,
                "completion_time": time.time()
            }
        )
        
        return {
            "goal_completed": True,
            "goal_id": goal_id,
            "completion_time": time.time(),
            "lessons_learned": self._extract_lessons_learned(goal, completion_data)
        }
    
    def _prioritize_goals(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Re-prioritize active goals based on current context."""
        context = message.get("context", {})
        emotional_state = message.get("emotional_state", {})
        
        # Re-evaluate goal priorities
        priority_updates = {}
        for goal_id, goal in self.active_goals.items():
            new_priority = self._calculate_goal_priority(goal, context, emotional_state)
            if new_priority != goal.priority:
                priority_updates[goal_id] = {
                    "old_priority": goal.priority.value,
                    "new_priority": new_priority.value
                }
                goal.priority = new_priority
        
        # Sort goals by priority
        sorted_goals = sorted(
            self.active_goals.values(),
            key=lambda g: g.priority.value
        )
        
        return {
            "priority_updates": priority_updates,
            "prioritized_goals": [goal.goal_id for goal in sorted_goals],
            "top_priority_goal": sorted_goals[0].goal_id if sorted_goals else None
        }
    
    def _get_active_goals(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Get list of active goals."""
        include_details = message.get("include_details", True)
        
        if include_details:
            goals_data = [goal.__dict__ for goal in self.active_goals.values()]
        else:
            goals_data = [
                {
                    "goal_id": goal.goal_id,
                    "goal_type": goal.goal_type.value,
                    "priority": goal.priority.value,
                    "description": goal.description
                }
                for goal in self.active_goals.values()
            ]
        
        return {
            "active_goals": goals_data,
            "total_count": len(self.active_goals),
            "by_priority": self._group_goals_by_priority(),
            "by_type": self._group_goals_by_type()
        }
    
    def _store_goal_in_memory(self, goal: Goal) -> None:
        """Store a goal in memory for future reference."""
        self.memory.store(
            f"goal_{goal.goal_id}",
            goal.__dict__,
            ttl=86400  # Store for 24 hours
        )
    
    def _assess_goal_emotional_impact(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Assess emotional impact of goal generation activities."""
        goals_generated = len(result.get("goals_generated", []))
        
        # Goal generation increases curiosity and satisfaction
        curiosity = min(0.8 + (goals_generated * 0.1), 1.0)
        satisfaction = min(0.6 + (goals_generated * 0.1), 1.0)
        
        return {
            "curiosity": curiosity,
            "satisfaction": satisfaction,
            "motivation": 0.8 if goals_generated > 0 else 0.5
        }
    
    # Helper methods (simplified implementations)
    def _calculate_progress_score(self, goal: Goal, progress_data: Dict[str, Any]) -> float:
        """Calculate progress score for a goal."""
        # Simple implementation - would be more sophisticated in practice
        completed_criteria = 0
        total_criteria = len(goal.success_criteria)
        
        for criterion, target_value in goal.success_criteria.items():
            actual_value = progress_data.get(criterion, 0)
            if isinstance(target_value, bool):
                if actual_value == target_value:
                    completed_criteria += 1
            elif isinstance(target_value, (int, float)):
                if actual_value >= target_value:
                    completed_criteria += 1
        
        return completed_criteria / total_criteria if total_criteria > 0 else 0.0
    
    def _suggest_next_steps(self, goal: Goal, progress_score: float) -> List[str]:
        """Suggest next steps for goal completion."""
        if progress_score < 0.3:
            return ["Focus on understanding requirements", "Break down into smaller tasks"]
        elif progress_score < 0.7:
            return ["Continue current approach", "Monitor progress closely"]
        else:
            return ["Push for completion", "Prepare for goal closure"]
    
    def _extract_lessons_learned(self, goal: Goal, completion_data: Dict[str, Any]) -> List[str]:
        """Extract lessons learned from goal completion."""
        return [
            f"Goal type {goal.goal_type.value} completed successfully",
            "Emotional motivation was effective",
            "Success criteria were appropriate"
        ]
    
    def _calculate_goal_priority(
        self,
        goal: Goal,
        context: Dict[str, Any],
        emotional_state: Dict[str, float]
    ) -> GoalPriority:
        """Calculate goal priority based on current context."""
        # Simple priority calculation - would be more sophisticated
        base_priority = goal.priority.value
        
        # Adjust based on emotional state
        urgency = emotional_state.get("urgency", 0.5)
        if urgency > 0.8:
            base_priority = max(1, base_priority - 1)
        
        # Adjust based on context relevance
        relevance = context.get("relevance", {}).get(goal.context.get("focus_area", ""), 0.5)
        if relevance > 0.8:
            base_priority = max(1, base_priority - 1)
        
        return GoalPriority(base_priority)
    
    def _group_goals_by_priority(self) -> Dict[str, int]:
        """Group active goals by priority."""
        groups = {}
        for goal in self.active_goals.values():
            priority_name = goal.priority.name
            groups[priority_name] = groups.get(priority_name, 0) + 1
        return groups
    
    def _group_goals_by_type(self) -> Dict[str, int]:
        """Group active goals by type."""
        groups = {}
        for goal in self.active_goals.values():
            type_name = goal.goal_type.value
            groups[type_name] = groups.get(type_name, 0) + 1
        return groups 
"""
NIS Protocol Curiosity Engine

This module implements the curiosity drive mechanism that motivates exploration,
learning, and knowledge acquisition in the AGI system.
"""

import time
import random
import logging
import math
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from ...memory.memory_manager import MemoryManager


class CuriosityType(Enum):
    """Types of curiosity that can drive exploration"""
    EPISTEMIC = "epistemic"           # Knowledge-seeking curiosity
    DIVERSIVE = "diversive"           # Novelty-seeking curiosity
    SPECIFIC = "specific"             # Specific question-driven curiosity
    PERCEPTUAL = "perceptual"         # Sensory exploration curiosity
    EMPATHIC = "empathic"            # Understanding others' perspectives
    CREATIVE = "creative"             # Creative exploration curiosity


@dataclass
class CuriositySignal:
    """Represents a curiosity signal driving exploration"""
    signal_id: str
    curiosity_type: CuriosityType
    intensity: float
    focus_area: str
    specific_questions: List[str]
    context: Dict[str, Any]
    timestamp: float
    decay_rate: float
    satisfaction_threshold: float


@dataclass
class ExplorationTarget:
    """Represents a target for curiosity-driven exploration"""
    target_id: str
    domain: str
    description: str
    novelty_score: float
    complexity_score: float
    potential_value: float
    exploration_cost: float
    prerequisites: List[str]


class CuriosityEngine:
    """Engine that generates and manages curiosity-driven exploration.
    
    This engine provides:
    - Curiosity signal generation based on knowledge gaps
    - Novelty detection and assessment
    - Exploration target identification
    - Curiosity satisfaction tracking
    """
    
    def __init__(self):
        """Initialize the curiosity engine."""
        self.logger = logging.getLogger("nis.curiosity_engine")
        self.memory = MemoryManager()
        
        # Curiosity state
        self.active_curiosity_signals: Dict[str, CuriositySignal] = {}
        self.exploration_targets: Dict[str, ExplorationTarget] = {}
        self.knowledge_map: Dict[str, Any] = {}
        
        # Curiosity parameters
        self.base_curiosity_level = 0.6
        self.novelty_threshold = 0.7
        self.complexity_preference = 0.8
        self.exploration_budget = 1.0  # Resource allocation for exploration
        
        # Learning and satisfaction tracking
        self.curiosity_satisfaction_history: List[Dict[str, Any]] = []
        self.knowledge_growth_rate = 0.0
        
        self.logger.info("CuriosityEngine initialized")
    
    def detect_knowledge_gaps(
        self,
        current_context: Dict[str, Any],
        knowledge_base: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect gaps in knowledge that could drive curiosity.
        
        Args:
            current_context: Current situational context
            knowledge_base: Available knowledge
            
        Returns:
            List of detected knowledge gaps
        """
        # TODO: Implement sophisticated knowledge gap detection
        # Should identify:
        # - Missing knowledge in current domain
        # - Inconsistencies in knowledge
        # - Unexplored related areas
        # - Questions without answers
        # - Incomplete understanding chains
        
        self.logger.info("Detecting knowledge gaps")
        
        # Placeholder implementation
        gaps = [
            {
                "gap_type": "missing_knowledge",
                "domain": "archaeology",
                "description": "Unknown cultural significance of symbol patterns",
                "importance": 0.8,
                "explorability": 0.9
            },
            {
                "gap_type": "incomplete_understanding",
                "domain": "methodology",
                "description": "Limited understanding of preservation techniques",
                "importance": 0.7,
                "explorability": 0.8
            }
        ]
        
        return gaps
    
    def assess_novelty(
        self,
        item: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Assess the novelty of an item or concept.
        
        Args:
            item: Item to assess for novelty
            context: Contextual information
            
        Returns:
            Novelty score (0.0 to 1.0)
        """
        # TODO: Implement novelty assessment
        # Should consider:
        # - Similarity to known items
        # - Frequency of occurrence
        # - Unexpectedness
        # - Surprise factor
        # - Semantic distance from known concepts
        
        self.logger.debug(f"Assessing novelty for: {item.get('name', 'unknown')}")
        
        # Placeholder implementation
        # In practice, this would use embeddings, similarity measures, etc.
        return random.uniform(0.4, 0.9)
    
    def generate_curiosity_signal(
        self,
        trigger: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[CuriositySignal]:
        """Generate a curiosity signal based on a trigger.
        
        Args:
            trigger: Event or condition that triggers curiosity
            context: Current context
            
        Returns:
            Generated curiosity signal or None
        """
        # TODO: Implement sophisticated curiosity signal generation
        # Should consider:
        # - Type of trigger (gap, novelty, inconsistency)
        # - Current knowledge state
        # - Emotional state
        # - Available resources
        # - Previous exploration outcomes
        
        trigger_type = trigger.get("type", "unknown")
        self.logger.info(f"Generating curiosity signal for trigger: {trigger_type}")
        
        # Determine curiosity type based on trigger
        if trigger_type == "knowledge_gap":
            curiosity_type = CuriosityType.EPISTEMIC
        elif trigger_type == "novel_item":
            curiosity_type = CuriosityType.DIVERSIVE
        elif trigger_type == "specific_question":
            curiosity_type = CuriosityType.SPECIFIC
        else:
            curiosity_type = CuriosityType.EPISTEMIC
        
        # Calculate intensity based on various factors
        intensity = self._calculate_curiosity_intensity(trigger, context)
        
        if intensity < 0.3:  # Below threshold
            return None
        
        # Generate signal
        signal = CuriositySignal(
            signal_id=f"curiosity_{int(time.time())}_{random.randint(1000, 9999)}",
            curiosity_type=curiosity_type,
            intensity=intensity,
            focus_area=trigger.get("domain", "general"),
            specific_questions=trigger.get("questions", []),
            context=context.copy(),
            timestamp=time.time(),
            decay_rate=0.1,  # How quickly curiosity fades
            satisfaction_threshold=0.8
        )
        
        # Store active signal
        self.active_curiosity_signals[signal.signal_id] = signal
        
        return signal
    
    def identify_exploration_targets(
        self,
        curiosity_signal: CuriositySignal,
        available_resources: Dict[str, Any]
    ) -> List[ExplorationTarget]:
        """Identify potential targets for exploration based on curiosity.
        
        Args:
            curiosity_signal: Active curiosity signal
            available_resources: Available resources for exploration
            
        Returns:
            List of potential exploration targets
        """
        # TODO: Implement target identification
        # Should identify:
        # - Relevant domains to explore
        # - Specific concepts or items to investigate
        # - Learning pathways and sequences
        # - Resource-efficient exploration options
        # - High-value exploration opportunities
        
        self.logger.info(f"Identifying exploration targets for {curiosity_signal.focus_area}")
        
        # Placeholder implementation
        targets = [
            ExplorationTarget(
                target_id=f"target_{int(time.time())}_1",
                domain=curiosity_signal.focus_area,
                description=f"Explore {curiosity_signal.focus_area} fundamentals",
                novelty_score=0.8,
                complexity_score=0.6,
                potential_value=0.9,
                exploration_cost=0.5,
                prerequisites=[]
            ),
            ExplorationTarget(
                target_id=f"target_{int(time.time())}_2",
                domain=curiosity_signal.focus_area,
                description=f"Deep dive into {curiosity_signal.focus_area} advanced concepts",
                novelty_score=0.9,
                complexity_score=0.8,
                potential_value=0.8,
                exploration_cost=0.7,
                prerequisites=["fundamentals"]
            )
        ]
        
        return targets
    
    def prioritize_exploration(
        self,
        targets: List[ExplorationTarget],
        current_goals: List[Dict[str, Any]],
        resources: Dict[str, Any]
    ) -> List[ExplorationTarget]:
        """Prioritize exploration targets based on multiple factors.
        
        Args:
            targets: Available exploration targets
            current_goals: Current system goals
            resources: Available resources
            
        Returns:
            Prioritized list of exploration targets
        """
        # TODO: Implement sophisticated prioritization
        # Should consider:
        # - Curiosity intensity and type
        # - Resource availability
        # - Goal alignment
        # - Potential learning value
        # - Prerequisites and dependencies
        # - Risk vs. reward
        
        self.logger.info(f"Prioritizing {len(targets)} exploration targets")
        
        # Placeholder implementation
        # In practice, this would use multi-criteria decision making
        def priority_score(target: ExplorationTarget) -> float:
            return (
                target.novelty_score * 0.3 +
                target.potential_value * 0.4 +
                (1.0 - target.exploration_cost) * 0.2 +
                target.complexity_score * self.complexity_preference * 0.1
            )
        
        return sorted(targets, key=priority_score, reverse=True)
    
    def track_exploration_outcome(
        self,
        target_id: str,
        outcome: Dict[str, Any],
        satisfaction_level: float
    ) -> None:
        """Track the outcome of an exploration activity.
        
        Args:
            target_id: ID of the exploration target
            outcome: Results of the exploration
            satisfaction_level: How well the exploration satisfied curiosity
        """
        # TODO: Implement outcome tracking
        # Should track:
        # - Knowledge gained
        # - Curiosity satisfaction
        # - Resource usage
        # - Unexpected discoveries
        # - Learning efficiency
        
        self.logger.info(f"Tracking exploration outcome for target: {target_id}")
        
        # Update satisfaction history
        self.curiosity_satisfaction_history.append({
            "target_id": target_id,
            "satisfaction": satisfaction_level,
            "outcome": outcome,
            "timestamp": time.time()
        })
        
        # Update knowledge map
        if "knowledge_gained" in outcome:
            self._update_knowledge_map(outcome["knowledge_gained"])
        
        # Update curiosity signals based on satisfaction
        self._update_curiosity_satisfaction(satisfaction_level)
    
    def decay_curiosity_signals(self) -> None:
        """Apply decay to active curiosity signals over time."""
        # TODO: Implement sophisticated decay
        # Should consider:
        # - Time-based decay
        # - Satisfaction-based decay
        # - Context changes
        # - Priority shifts
        
        current_time = time.time()
        signals_to_remove = []
        
        for signal_id, signal in self.active_curiosity_signals.items():
            # Apply time-based decay
            time_elapsed = current_time - signal.timestamp
            decay_factor = math.exp(-signal.decay_rate * time_elapsed)
            signal.intensity *= decay_factor
            
            # Remove signals below threshold
            if signal.intensity < 0.1:
                signals_to_remove.append(signal_id)
        
        # Remove decayed signals
        for signal_id in signals_to_remove:
            del self.active_curiosity_signals[signal_id]
    
    def get_current_curiosity_state(self) -> Dict[str, Any]:
        """Get current state of curiosity and exploration.
        
        Returns:
            Current curiosity state summary
        """
        # TODO: Implement comprehensive state summary
        # Should include:
        # - Active curiosity signals
        # - Knowledge growth metrics
        # - Exploration effectiveness
        # - Resource utilization
        # - Satisfaction trends
        
        self.decay_curiosity_signals()  # Update state first
        
        return {
            "active_signals": len(self.active_curiosity_signals),
            "average_intensity": self._calculate_average_intensity(),
            "dominant_curiosity_type": self._get_dominant_curiosity_type(),
            "knowledge_growth_rate": self.knowledge_growth_rate,
            "exploration_effectiveness": self._calculate_exploration_effectiveness(),
            "resource_utilization": self._calculate_resource_utilization()
        }
    
    def suggest_curiosity_driven_goals(
        self,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Suggest goals driven by current curiosity state.
        
        Args:
            context: Current context
            
        Returns:
            List of curiosity-driven goal suggestions
        """
        # TODO: Implement goal suggestion
        # Should suggest:
        # - Exploration goals based on active curiosity
        # - Learning goals for knowledge gaps
        # - Investigation goals for novel items
        # - Creative goals for self-expression
        
        self.logger.info("Generating curiosity-driven goal suggestions")
        
        suggestions = []
        
        for signal in self.active_curiosity_signals.values():
            if signal.intensity > 0.5:  # Only strong curiosity signals
                goal_suggestion = {
                    "goal_type": "exploration",
                    "curiosity_type": signal.curiosity_type.value,
                    "description": f"Explore {signal.focus_area} to satisfy curiosity",
                    "priority": signal.intensity,
                    "focus_area": signal.focus_area,
                    "specific_questions": signal.specific_questions,
                    "estimated_satisfaction": 0.8
                }
                suggestions.append(goal_suggestion)
        
        return suggestions
    
    def _calculate_curiosity_intensity(
        self,
        trigger: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Calculate curiosity intensity based on trigger and context.
        
        Args:
            trigger: Curiosity trigger
            context: Current context
            
        Returns:
            Calculated intensity (0.0 to 1.0)
        """
        # TODO: Implement sophisticated intensity calculation
        # Should consider:
        # - Novelty of trigger
        # - Personal relevance
        # - Knowledge gap size
        # - Emotional state
        # - Available resources
        
        base_intensity = self.base_curiosity_level
        novelty_factor = trigger.get("novelty", 0.5)
        importance_factor = trigger.get("importance", 0.5)
        
        intensity = base_intensity * (0.5 + 0.3 * novelty_factor + 0.2 * importance_factor)
        return min(intensity, 1.0)
    
    def _calculate_average_intensity(self) -> float:
        """Calculate average intensity of active curiosity signals."""
        if not self.active_curiosity_signals:
            return 0.0
        
        total_intensity = sum(
            signal.intensity for signal in self.active_curiosity_signals.values()
        )
        return total_intensity / len(self.active_curiosity_signals)
    
    def _get_dominant_curiosity_type(self) -> str:
        """Get the dominant curiosity type among active signals."""
        if not self.active_curiosity_signals:
            return "none"
        
        type_counts = {}
        for signal in self.active_curiosity_signals.values():
            curiosity_type = signal.curiosity_type.value
            type_counts[curiosity_type] = type_counts.get(curiosity_type, 0) + signal.intensity
        
        return max(type_counts, key=type_counts.get) if type_counts else "none"
    
    def _calculate_exploration_effectiveness(self) -> float:
        """Calculate effectiveness of recent explorations."""
        if not self.curiosity_satisfaction_history:
            return 0.5  # Default neutral value
        
        recent_history = self.curiosity_satisfaction_history[-10:]  # Last 10 explorations
        if not recent_history:
            return 0.5
        
        average_satisfaction = sum(
            entry["satisfaction"] for entry in recent_history
        ) / len(recent_history)
        
        return average_satisfaction
    
    def _calculate_resource_utilization(self) -> float:
        """Calculate current resource utilization for curiosity-driven activities."""
        # TODO: Implement actual resource tracking
        # For now, return a placeholder value
        return 0.6
    
    def _update_knowledge_map(self, new_knowledge: Dict[str, Any]) -> None:
        """Update the internal knowledge map with new knowledge.
        
        Args:
            new_knowledge: New knowledge to integrate
        """
        # TODO: Implement sophisticated knowledge integration
        # Should:
        # - Update knowledge graph
        # - Identify new connections
        # - Calculate knowledge growth
        # - Update understanding metrics
        
        domain = new_knowledge.get("domain", "general")
        if domain not in self.knowledge_map:
            self.knowledge_map[domain] = {}
        
        # Simple integration (placeholder)
        concepts = new_knowledge.get("concepts", [])
        for concept in concepts:
            self.knowledge_map[domain][concept] = new_knowledge.get("details", {})
    
    def _update_curiosity_satisfaction(self, satisfaction_level: float) -> None:
        """Update curiosity satisfaction metrics.
        
        Args:
            satisfaction_level: Level of satisfaction achieved
        """
        # TODO: Implement satisfaction-based curiosity adjustment
        # Should adjust:
        # - Future curiosity thresholds
        # - Exploration strategies
        # - Resource allocation
        # - Signal generation sensitivity
        
        # Simple adjustment (placeholder)
        if satisfaction_level > 0.8:
            self.base_curiosity_level = min(self.base_curiosity_level * 1.05, 1.0)
        elif satisfaction_level < 0.4:
            self.base_curiosity_level = max(self.base_curiosity_level * 0.95, 0.1) 
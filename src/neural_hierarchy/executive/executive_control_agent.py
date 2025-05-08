from typing import Optional, Dict, Any, List, Tuple
from ..base_neural_agent import NeuralAgent, NeuralLayer, NeuralSignal
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np

class ActionType(Enum):
    RESPOND = "respond"
    QUERY = "query"
    ANALYZE = "analyze"
    REFLECT = "reflect"
    PLAN = "plan"
    EXECUTE = "execute"

@dataclass
class Action:
    action_type: ActionType
    parameters: Dict[str, Any]
    priority: float
    context: Dict[str, Any]
    timestamp: datetime = datetime.now()

@dataclass
class Goal:
    description: str
    priority: float
    deadline: Optional[datetime] = None
    subgoals: List['Goal'] = None
    status: str = "pending"
    completion: float = 0.0

class ExecutiveControlAgent(NeuralAgent):
    """Agent for executive control, decision making, and planning"""
    
    def __init__(
        self,
        agent_id: str = "executive_control",
        max_active_goals: int = 5,
        planning_horizon: int = 3
    ):
        super().__init__(
            agent_id=agent_id,
            layer=NeuralLayer.EXECUTIVE,
            description="Handles executive control and decision making"
        )
        
        self.max_active_goals = max_active_goals
        self.planning_horizon = planning_horizon
        
        # Executive state
        self.active_goals: List[Goal] = []
        self.action_history: List[Action] = []
        self.current_plan: List[Action] = []
        self.emotional_context: Dict[str, Any] = {}
        
        # Decision thresholds
        self.confidence_threshold = 0.7
        self.urgency_threshold = 0.8
    
    def process_signal(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Process incoming signal and make executive decisions"""
        if not isinstance(signal.content, dict):
            return None
            
        # Update emotional context
        if 'emotional_state' in signal.content:
            self.emotional_context = signal.content
        
        # Generate actions based on current state
        actions = self._generate_actions(signal.content)
        
        # Prioritize actions
        prioritized_actions = self._prioritize_actions(actions)
        
        # Update current plan
        self._update_plan(prioritized_actions)
        
        # Select next action
        next_action = self._select_next_action()
        
        if next_action:
            # Add to history
            self.action_history.append(next_action)
            if len(self.action_history) > 100:
                self.action_history = self.action_history[-100:]
            
            # Generate signal for motor layer
            return NeuralSignal(
                source_layer=self.layer,
                target_layer=NeuralLayer.MOTOR,
                content={
                    'action': {
                        'type': next_action.action_type.value,
                        'parameters': next_action.parameters,
                        'context': next_action.context
                    },
                    'emotional_context': self.emotional_context,
                    'execution_priority': next_action.priority
                },
                priority=next_action.priority
            )
        
        return None
    
    def _generate_actions(self, content: Dict[str, Any]) -> List[Action]:
        """Generate possible actions based on current state"""
        actions = []
        
        # Get emotional state
        emotional_state = content.get('emotional_state', {})
        emotional_context = content.get('emotional_context', {})
        
        # Generate response action if needed
        if self._should_respond(emotional_state):
            actions.append(Action(
                action_type=ActionType.RESPOND,
                parameters={
                    'style': emotional_state.get('primary_emotion', 'neutral'),
                    'intensity': emotional_state.get('arousal', 0.5)
                },
                priority=0.8,
                context={'emotional_state': emotional_state}
            ))
        
        # Generate analysis action if stability is low
        stability = emotional_context.get('emotional_stability', 1.0)
        if stability < 0.7:
            actions.append(Action(
                action_type=ActionType.ANALYZE,
                parameters={
                    'focus': 'emotional_state',
                    'depth': 'deep' if stability < 0.5 else 'shallow'
                },
                priority=0.7,
                context={'stability': stability}
            ))
        
        # Generate reflection action periodically
        if len(self.action_history) % 10 == 0:
            actions.append(Action(
                action_type=ActionType.REFLECT,
                parameters={
                    'scope': 'recent_actions',
                    'count': 5
                },
                priority=0.5,
                context={'action_history': len(self.action_history)}
            ))
        
        return actions
    
    def _prioritize_actions(self, actions: List[Action]) -> List[Action]:
        """Prioritize actions based on current context"""
        # Sort by base priority
        prioritized = sorted(
            actions,
            key=lambda a: a.priority,
            reverse=True
        )
        
        # Adjust priorities based on emotional context
        if self.emotional_context:
            emotional_state = self.emotional_context.get('emotional_state', {})
            arousal = emotional_state.get('arousal', 0.5)
            
            for action in prioritized:
                # Increase priority of response actions when arousal is high
                if action.action_type == ActionType.RESPOND:
                    action.priority = min(1.0, action.priority + arousal * 0.2)
                
                # Increase priority of analysis when emotion is intense
                if action.action_type == ActionType.ANALYZE:
                    intensity = abs(emotional_state.get('valence', 0))
                    action.priority = min(1.0, action.priority + intensity * 0.15)
        
        return sorted(prioritized, key=lambda a: a.priority, reverse=True)
    
    def _update_plan(self, new_actions: List[Action]):
        """Update current action plan"""
        # Remove completed actions
        self.current_plan = [
            action for action in self.current_plan
            if action.action_type not in [a.action_type for a in self.action_history[-5:]]
        ]
        
        # Add new high-priority actions
        for action in new_actions:
            if action.priority > 0.7 and len(self.current_plan) < self.planning_horizon:
                self.current_plan.append(action)
        
        # Sort plan by priority
        self.current_plan.sort(key=lambda a: a.priority, reverse=True)
    
    def _select_next_action(self) -> Optional[Action]:
        """Select the next action to execute"""
        if not self.current_plan:
            return None
        
        # Get highest priority action
        action = self.current_plan[0]
        
        # Check if action meets confidence threshold
        if action.priority < self.confidence_threshold:
            return None
        
        # Remove from plan
        self.current_plan.pop(0)
        
        return action
    
    def _should_respond(self, emotional_state: Dict[str, Any]) -> bool:
        """Determine if a response is needed"""
        if not emotional_state:
            return False
            
        # Respond if emotion is intense
        arousal = emotional_state.get('arousal', 0)
        valence = abs(emotional_state.get('valence', 0))
        
        return arousal > 0.7 or valence > 0.6
    
    def add_goal(self, goal: Goal):
        """Add a new goal to track"""
        if len(self.active_goals) < self.max_active_goals:
            self.active_goals.append(goal)
            self.active_goals.sort(key=lambda g: g.priority, reverse=True)
    
    def update_goal_status(self, goal_description: str, completion: float):
        """Update goal completion status"""
        for goal in self.active_goals:
            if goal.description == goal_description:
                goal.completion = completion
                if completion >= 1.0:
                    goal.status = "completed"
                break
    
    def get_executive_state(self) -> Dict[str, Any]:
        """Get current executive state"""
        return {
            'active_goals': [
                {
                    'description': g.description,
                    'priority': g.priority,
                    'status': g.status,
                    'completion': g.completion
                }
                for g in self.active_goals
            ],
            'current_plan': [
                {
                    'type': a.action_type.value,
                    'priority': a.priority,
                    'timestamp': a.timestamp.isoformat()
                }
                for a in self.current_plan
            ],
            'emotional_context': self.emotional_context,
            'action_history_length': len(self.action_history)
        }
    
    def reset(self):
        """Reset executive state"""
        super().reset()
        self.active_goals = []
        self.current_plan = []
        self.emotional_context = {}
        # Keep action history for learning 
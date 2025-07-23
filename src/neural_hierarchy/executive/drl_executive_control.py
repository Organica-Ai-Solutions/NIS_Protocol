"""
DRL-Enhanced Executive Control Agent for NIS Protocol

This module provides Deep Reinforcement Learning capabilities for sophisticated
executive control and decision making. The DRL system learns optimal policies for:
- Multi-objective optimization (speed, accuracy, resource usage)
- Dynamic priority management and task scheduling
- Intelligent action selection based on system state
- Adaptive control strategies that balance competing objectives
- Context-aware decision making with learned preferences

Enhanced Features:
- Multi-objective Actor-Critic networks for complex decision making
- Dynamic priority learning with temporal consistency
- Resource-aware action selection with efficiency optimization
- Adaptive threshold learning for different system states
- Integration with existing Neural Hierarchy Executive Control
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio

# Import DRL foundation
from src.agents.learning.drl_foundation import (
    DRLPolicyNetwork, DRLEnvironment, DRLAgent, RewardSignal,
    DRLMetrics, TrainingConfig, ExperienceBuffer
)

# Import existing Executive Control infrastructure
from src.neural_hierarchy.executive.executive_control_agent import (
    ExecutiveControlAgent, ActionType, Action, Goal
)
from src.neural_hierarchy.base_neural_agent import NeuralAgent, NeuralLayer, NeuralSignal

# Integration and infrastructure
from src.infrastructure.integration_coordinator import InfrastructureCoordinator
from src.infrastructure.caching_system import CacheStrategy

# Self-audit and integrity
from src.utils.self_audit import self_audit_engine, ViolationType, IntegrityViolation
from src.utils.integrity_metrics import (
    calculate_confidence, create_default_confidence_factors
)

# Configure logging
logger = logging.getLogger("drl_executive_control")


class ExecutiveAction(Enum):
    """Enhanced executive actions for DRL decision making"""
    IMMEDIATE_RESPONSE = "immediate_response"
    DETAILED_ANALYSIS = "detailed_analysis"
    COLLABORATIVE_PROCESSING = "collaborative_processing"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    QUALITY_ENHANCEMENT = "quality_enhancement"
    SPEED_PRIORITIZATION = "speed_prioritization"
    STRATEGIC_PLANNING = "strategic_planning"
    ADAPTIVE_LEARNING = "adaptive_learning"


class ExecutiveObjective(Enum):
    """Multi-objective optimization targets"""
    MAXIMIZE_SPEED = "maximize_speed"
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    MINIMIZE_RESOURCES = "minimize_resources"
    BALANCE_ALL = "balance_all"
    ADAPTIVE_OPTIMIZATION = "adaptive_optimization"


@dataclass
class ExecutiveState:
    """Comprehensive state representation for executive decision making"""
    # Task characteristics
    task_urgency: float              # How urgent is the current task (0-1)
    task_complexity: float           # Estimated complexity (0-1)
    task_importance: float           # Strategic importance (0-1)
    quality_requirements: float      # Required quality level (0-1)
    resource_constraints: float      # Available resources (0-1)
    
    # System state
    cpu_utilization: float           # Current CPU usage (0-1)
    memory_utilization: float        # Current memory usage (0-1)
    network_utilization: float       # Current network usage (0-1)
    agent_availability: List[float]  # Availability of other agents
    system_load: float               # Overall system load (0-1)
    
    # Performance context
    recent_response_times: List[float]     # Recent task response times
    recent_quality_scores: List[float]     # Recent quality outcomes
    recent_resource_efficiency: List[float] # Recent resource efficiency
    success_rate_trend: float              # Recent success rate trend
    
    # Emotional and cognitive context
    emotional_arousal: float         # Current emotional intensity (0-1)
    emotional_valence: float         # Current emotional valence (-1 to 1)
    cognitive_load: float            # Current cognitive processing load (0-1)
    attention_focus: float           # Current attention focus level (0-1)
    
    # Strategic context
    active_goals_count: int          # Number of active goals
    goal_completion_rate: float      # Recent goal completion rate (0-1)
    strategic_alignment: float       # How well current actions align with goals (0-1)
    long_term_planning_horizon: float # Planning horizon consideration (0-1)
    
    # Learning context
    exploration_tendency: float      # Current exploration vs exploitation (0-1)
    confidence_level: float          # Confidence in decision making (0-1)
    learning_progress: float         # Recent learning progress (0-1)
    adaptation_rate: float           # How quickly to adapt to changes (0-1)


@dataclass
class ExecutiveReward:
    """Multi-dimensional reward for executive decision making"""
    speed_performance: float         # How quickly was the task completed
    accuracy_performance: float      # How accurate was the outcome
    resource_efficiency: float       # How efficiently were resources used
    strategic_alignment: float       # How well did action align with goals
    learning_value: float           # How much was learned from the action
    user_satisfaction: float        # Estimated user satisfaction
    
    # Computed multi-objective reward
    total_reward: float = 0.0
    
    def compute_multi_objective_reward(self, 
                                     objective: ExecutiveObjective,
                                     weights: Optional[Dict[str, float]] = None) -> float:
        """Compute reward based on current objective and weights"""
        
        if objective == ExecutiveObjective.MAXIMIZE_SPEED:
            objective_weights = {
                'speed_performance': 0.5,
                'accuracy_performance': 0.2,
                'resource_efficiency': 0.1,
                'strategic_alignment': 0.1,
                'learning_value': 0.05,
                'user_satisfaction': 0.05
            }
        elif objective == ExecutiveObjective.MAXIMIZE_ACCURACY:
            objective_weights = {
                'speed_performance': 0.1,
                'accuracy_performance': 0.5,
                'resource_efficiency': 0.1,
                'strategic_alignment': 0.15,
                'learning_value': 0.1,
                'user_satisfaction': 0.05
            }
        elif objective == ExecutiveObjective.MINIMIZE_RESOURCES:
            objective_weights = {
                'speed_performance': 0.15,
                'accuracy_performance': 0.2,
                'resource_efficiency': 0.4,
                'strategic_alignment': 0.1,
                'learning_value': 0.1,
                'user_satisfaction': 0.05
            }
        elif objective == ExecutiveObjective.BALANCE_ALL:
            objective_weights = {
                'speed_performance': 0.2,
                'accuracy_performance': 0.2,
                'resource_efficiency': 0.2,
                'strategic_alignment': 0.15,
                'learning_value': 0.15,
                'user_satisfaction': 0.1
            }
        else:  # ADAPTIVE_OPTIMIZATION
            objective_weights = weights or {
                'speed_performance': 0.2,
                'accuracy_performance': 0.2,
                'resource_efficiency': 0.2,
                'strategic_alignment': 0.15,
                'learning_value': 0.15,
                'user_satisfaction': 0.1
            }
        
        self.total_reward = (
            objective_weights['speed_performance'] * self.speed_performance +
            objective_weights['accuracy_performance'] * self.accuracy_performance +
            objective_weights['resource_efficiency'] * self.resource_efficiency +
            objective_weights['strategic_alignment'] * self.strategic_alignment +
            objective_weights['learning_value'] * self.learning_value +
            objective_weights['user_satisfaction'] * self.user_satisfaction
        )
        
        return self.total_reward


class MultiObjectiveExecutivePolicyNetwork(nn.Module):
    """Multi-objective Actor-Critic network for executive control"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 num_objectives: int = 5,
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_objectives = num_objectives
        
        # Shared feature extraction layers
        self.feature_layers = nn.ModuleList()
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            self.feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.feature_layers.append(nn.ReLU())
            self.feature_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Multi-objective policy heads
        self.action_policy_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Objective selection head
        self.objective_selection_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], num_objectives),
            nn.Softmax(dim=-1)
        )
        
        # Priority adjustment head
        self.priority_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )
        
        # Resource allocation head
        self.resource_allocation_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 3),  # CPU, Memory, Network
            nn.Softmax(dim=-1)
        )
        
        # Threshold adaptation head
        self.threshold_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 4),  # confidence, urgency, quality, exploration
            nn.Sigmoid()
        )
        
        # Multi-objective value heads (one for each objective)
        self.value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], 1)
            ) for _ in range(num_objectives)
        ])
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through multi-objective network"""
        # Extract shared features
        x = state
        for layer in self.feature_layers:
            x = layer(x)
        
        # Compute all outputs
        action_probs = self.action_policy_head(x)
        objective_probs = self.objective_selection_head(x)
        priority_adjustment = self.priority_head(x)
        resource_allocation = self.resource_allocation_head(x)
        thresholds = self.threshold_head(x)
        
        # Compute multi-objective values
        objective_values = torch.stack([value_head(x) for value_head in self.value_heads], dim=-1)
        
        return action_probs, objective_probs, priority_adjustment, resource_allocation, thresholds, objective_values


class DRLEnhancedExecutiveControl:
    """
    DRL-Enhanced Executive Control Agent with multi-objective optimization
    
    This agent learns sophisticated control policies that balance multiple
    objectives including speed, accuracy, resource efficiency, and strategic
    alignment through deep reinforcement learning.
    """
    
    def __init__(self,
                 state_dim: int = 60,  # Comprehensive executive state
                 action_dim: int = 8,  # Number of executive actions
                 num_objectives: int = 5,  # Number of optimization objectives
                 learning_rate: float = 0.0001,
                 gamma: float = 0.95,  # Discount factor
                 tau: float = 0.005,   # Target network update rate
                 buffer_size: int = 5000,
                 batch_size: int = 32,
                 update_frequency: int = 4,
                 infrastructure_coordinator: Optional[InfrastructureCoordinator] = None,
                 enable_self_audit: bool = True):
        """Initialize DRL-Enhanced Executive Control"""
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_objectives = num_objectives
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.infrastructure = infrastructure_coordinator
        
        # Initialize networks
        self.policy_network = MultiObjectiveExecutivePolicyNetwork(
            state_dim, action_dim, num_objectives
        )
        self.target_network = MultiObjectiveExecutivePolicyNetwork(
            state_dim, action_dim, num_objectives
        )
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Copy parameters to target network
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
        # Experience buffer
        self.experience_buffer = ExperienceBuffer(buffer_size)
        
        # Executive state and metrics
        self.current_state: Optional[ExecutiveState] = None
        self.current_objective: ExecutiveObjective = ExecutiveObjective.BALANCE_ALL
        self.executive_metrics = {
            'total_decisions': 0,
            'successful_decisions': 0,
            'average_speed_performance': 0.0,
            'average_accuracy_performance': 0.0,
            'average_resource_efficiency': 0.0,
            'average_strategic_alignment': 0.0,
            'objective_switches': 0,
            'adaptive_threshold_changes': 0,
            'learning_episodes': 0,
            'policy_loss': 0.0,
            'value_loss': 0.0
        }
        
        # Training state
        self.training_step = 0
        self.episode_rewards: deque = deque(maxlen=1000)
        self.episode_lengths: deque = deque(maxlen=1000)
        
        # Available executive actions
        self.executive_actions = list(ExecutiveAction)
        self.executive_objectives = list(ExecutiveObjective)
        
        # Performance tracking
        self.decision_history: deque = deque(maxlen=500)
        self.objective_performance: Dict[ExecutiveObjective, deque] = {
            obj: deque(maxlen=100) for obj in self.executive_objectives
        }
        
        # Adaptive thresholds (learned)
        self.adaptive_thresholds = {
            'confidence_threshold': 0.7,
            'urgency_threshold': 0.8,
            'quality_threshold': 0.7,
            'exploration_threshold': 0.2
        }
        
        # Self-audit integration
        self.enable_self_audit = enable_self_audit
        self.integrity_violations: deque = deque(maxlen=100)
        self.auto_correction_enabled = True
        
        # Multi-objective reward weights (adaptive)
        self.objective_weights = {
            ExecutiveObjective.MAXIMIZE_SPEED: {
                'speed_performance': 0.5, 'accuracy_performance': 0.2,
                'resource_efficiency': 0.1, 'strategic_alignment': 0.1,
                'learning_value': 0.05, 'user_satisfaction': 0.05
            },
            ExecutiveObjective.MAXIMIZE_ACCURACY: {
                'speed_performance': 0.1, 'accuracy_performance': 0.5,
                'resource_efficiency': 0.1, 'strategic_alignment': 0.15,
                'learning_value': 0.1, 'user_satisfaction': 0.05
            },
            ExecutiveObjective.MINIMIZE_RESOURCES: {
                'speed_performance': 0.15, 'accuracy_performance': 0.2,
                'resource_efficiency': 0.4, 'strategic_alignment': 0.1,
                'learning_value': 0.1, 'user_satisfaction': 0.05
            },
            ExecutiveObjective.BALANCE_ALL: {
                'speed_performance': 0.2, 'accuracy_performance': 0.2,
                'resource_efficiency': 0.2, 'strategic_alignment': 0.15,
                'learning_value': 0.15, 'user_satisfaction': 0.1
            },
            ExecutiveObjective.ADAPTIVE_OPTIMIZATION: {
                'speed_performance': 0.2, 'accuracy_performance': 0.2,
                'resource_efficiency': 0.2, 'strategic_alignment': 0.15,
                'learning_value': 0.15, 'user_satisfaction': 0.1
            }
        }
        
        # Redis caching for decision data
        self.cache_ttl = 1800  # 30 minutes
        
        logger.info("DRL-Enhanced Executive Control initialized with multi-objective optimization")
    
    def extract_executive_state(self, neural_signal: Dict[str, Any], 
                              system_context: Dict[str, Any]) -> ExecutiveState:
        """Extract comprehensive executive state from neural signal and system context"""
        
        # Task characteristics
        content = neural_signal.get('content', {})
        task_urgency = content.get('urgency', 0.5)
        task_complexity = self._estimate_task_complexity(content)
        task_importance = content.get('importance', 0.5)
        quality_requirements = content.get('quality_requirements', 0.7)
        resource_constraints = system_context.get('resource_constraints', 0.7)
        
        # System state
        cpu_utilization = system_context.get('cpu_usage', 0.5)
        memory_utilization = system_context.get('memory_usage', 0.5)
        network_utilization = system_context.get('network_usage', 0.3)
        
        # Agent availability (simplified)
        agent_availability = system_context.get('agent_availability', [0.8] * 5)
        agent_availability = (agent_availability + [0.8] * 5)[:5]  # Ensure 5 agents
        
        system_load = (cpu_utilization + memory_utilization + network_utilization) / 3.0
        
        # Performance context
        recent_decisions = list(self.decision_history)[-10:] if self.decision_history else []
        recent_response_times = [d.get('response_time', 1.0) for d in recent_decisions]
        recent_response_times = (recent_response_times + [1.0] * 10)[:10]
        
        recent_quality_scores = [d.get('quality_score', 0.5) for d in recent_decisions]
        recent_quality_scores = (recent_quality_scores + [0.5] * 10)[:10]
        
        recent_resource_efficiency = [d.get('resource_efficiency', 0.5) for d in recent_decisions]
        recent_resource_efficiency = (recent_resource_efficiency + [0.5] * 10)[:10]
        
        success_rate_trend = np.mean([d.get('success', 0.5) for d in recent_decisions]) if recent_decisions else 0.5
        
        # Emotional and cognitive context
        emotional_state = content.get('emotional_state', {})
        emotional_arousal = emotional_state.get('arousal', 0.5)
        emotional_valence = emotional_state.get('valence', 0.0)
        
        cognitive_load = content.get('cognitive_load', 0.5)
        attention_focus = content.get('attention_focus', 0.7)
        
        # Strategic context
        active_goals_count = content.get('active_goals_count', 2)
        goal_completion_rate = content.get('goal_completion_rate', 0.7)
        strategic_alignment = content.get('strategic_alignment', 0.6)
        long_term_planning_horizon = content.get('planning_horizon', 0.5)
        
        # Learning context
        exploration_tendency = self._calculate_exploration_tendency()
        confidence_level = self._calculate_confidence_level()
        learning_progress = self._calculate_learning_progress()
        adaptation_rate = system_context.get('adaptation_rate', 0.5)
        
        state = ExecutiveState(
            task_urgency=task_urgency,
            task_complexity=task_complexity,
            task_importance=task_importance,
            quality_requirements=quality_requirements,
            resource_constraints=resource_constraints,
            cpu_utilization=cpu_utilization,
            memory_utilization=memory_utilization,
            network_utilization=network_utilization,
            agent_availability=agent_availability,
            system_load=system_load,
            recent_response_times=recent_response_times,
            recent_quality_scores=recent_quality_scores,
            recent_resource_efficiency=recent_resource_efficiency,
            success_rate_trend=success_rate_trend,
            emotional_arousal=emotional_arousal,
            emotional_valence=emotional_valence,
            cognitive_load=cognitive_load,
            attention_focus=attention_focus,
            active_goals_count=active_goals_count,
            goal_completion_rate=goal_completion_rate,
            strategic_alignment=strategic_alignment,
            long_term_planning_horizon=long_term_planning_horizon,
            exploration_tendency=exploration_tendency,
            confidence_level=confidence_level,
            learning_progress=learning_progress,
            adaptation_rate=adaptation_rate
        )
        
        return state
    
    def state_to_tensor(self, state: ExecutiveState) -> torch.Tensor:
        """Convert executive state to tensor for neural network"""
        features = []
        
        # Task characteristics (5 features)
        features.extend([
            state.task_urgency,
            state.task_complexity,
            state.task_importance,
            state.quality_requirements,
            state.resource_constraints
        ])
        
        # System state (9 features: 3 individual + 5 agent availability + 1 system load)
        features.extend([
            state.cpu_utilization,
            state.memory_utilization,
            state.network_utilization,
            state.system_load
        ])
        features.extend(state.agent_availability)  # 5 features
        
        # Performance context (31 features: 10+10+10+1)
        features.extend(state.recent_response_times)  # 10 features
        features.extend(state.recent_quality_scores)  # 10 features
        features.extend(state.recent_resource_efficiency)  # 10 features
        features.append(state.success_rate_trend)  # 1 feature
        
        # Emotional and cognitive context (4 features)
        features.extend([
            state.emotional_arousal,
            state.emotional_valence + 1.0,  # Normalize -1,1 to 0,2
            state.cognitive_load,
            state.attention_focus
        ])
        
        # Strategic context (4 features)
        features.extend([
            min(1.0, state.active_goals_count / 10.0),  # Normalize goal count
            state.goal_completion_rate,
            state.strategic_alignment,
            state.long_term_planning_horizon
        ])
        
        # Learning context (4 features)
        features.extend([
            state.exploration_tendency,
            state.confidence_level,
            state.learning_progress,
            state.adaptation_rate
        ])
        
        # Ensure exactly state_dim features
        features = features[:self.state_dim]
        features.extend([0.0] * (self.state_dim - len(features)))
        
        return torch.FloatTensor(features).unsqueeze(0)
    
    async def make_executive_decision(self, neural_signal: Dict[str, Any], 
                                    system_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make executive decision using DRL policy"""
        
        if system_context is None:
            system_context = {}
        
        # Extract state
        state = self.extract_executive_state(neural_signal, system_context)
        self.current_state = state
        
        # Convert to tensor
        state_tensor = self.state_to_tensor(state)
        
        # Get policy predictions
        with torch.no_grad():
            action_probs, objective_probs, priority_adjustment, resource_allocation, thresholds, objective_values = \
                self.policy_network(state_tensor)
        
        # Sample action from policy
        action_dist = torch.distributions.Categorical(action_probs)
        action_idx = action_dist.sample()
        selected_action = self.executive_actions[action_idx.item()]
        
        # Sample objective
        objective_dist = torch.distributions.Categorical(objective_probs)
        objective_idx = objective_dist.sample()
        current_objective = self.executive_objectives[objective_idx.item()]
        
        # Update current objective if different
        if current_objective != self.current_objective:
            self.current_objective = current_objective
            self.executive_metrics['objective_switches'] += 1
        
        # Update adaptive thresholds
        threshold_values = thresholds[0].cpu().numpy()
        old_thresholds = self.adaptive_thresholds.copy()
        
        self.adaptive_thresholds = {
            'confidence_threshold': threshold_values[0],
            'urgency_threshold': threshold_values[1],
            'quality_threshold': threshold_values[2],
            'exploration_threshold': threshold_values[3]
        }
        
        # Track threshold changes
        threshold_change = sum(abs(old_thresholds[k] - self.adaptive_thresholds[k]) 
                             for k in self.adaptive_thresholds)
        if threshold_change > 0.1:
            self.executive_metrics['adaptive_threshold_changes'] += 1
        
        # Create executive decision
        executive_decision = {
            'action': selected_action.value,
            'objective': current_objective.value,
            'priority_adjustment': priority_adjustment.item(),
            'resource_allocation': {
                'cpu': resource_allocation[0][0].item(),
                'memory': resource_allocation[0][1].item(),
                'network': resource_allocation[0][2].item()
            },
            'adaptive_thresholds': self.adaptive_thresholds,
            'confidence': torch.max(action_probs).item(),
            'estimated_values': objective_values[0].cpu().numpy().tolist(),
            'state_features': state_tensor[0].tolist(),
            'decision_id': f"exec_{int(time.time() * 1000)}",
            'timestamp': time.time()
        }
        
        # Store decision for tracking
        self.decision_history.append({
            'decision': executive_decision,
            'state': state,
            'timestamp': time.time()
        })
        
        # Cache decision for performance tracking
        if self.infrastructure and self.infrastructure.redis_manager:
            cache_key = f"drl_executive_decision:{executive_decision['decision_id']}"
            await self.infrastructure.cache_data(
                cache_key, executive_decision,
                agent_id="drl_executive", ttl=self.cache_ttl
            )
        
        # Apply self-audit monitoring
        if self.enable_self_audit:
            executive_decision = self._monitor_decision_integrity(executive_decision, state)
        
        # Update metrics
        self.executive_metrics['total_decisions'] += 1
        
        logger.info(f"DRL executive decision: {selected_action.value} with objective {current_objective.value}")
        
        return executive_decision
    
    async def process_decision_outcome(self, decision_id: str, outcome: Dict[str, Any]) -> None:
        """Process decision outcome for DRL learning"""
        
        # Calculate multi-objective reward
        reward = self._calculate_executive_reward(outcome)
        
        # Store experience for training
        if hasattr(self, 'last_state_tensor') and hasattr(self, 'last_action'):
            experience = {
                'state': self.last_state_tensor,
                'action': self.last_action,
                'reward': reward.total_reward,
                'next_state': None,  # Will be filled on next decision
                'done': True,
                'objective': self.current_objective
            }
            self.experience_buffer.add(experience)
        
        # Update objective performance tracking
        objective_reward = reward.total_reward
        self.objective_performance[self.current_objective].append(objective_reward)
        
        # Update metrics
        if outcome.get('success', False):
            self.executive_metrics['successful_decisions'] += 1
        
        self.executive_metrics['average_speed_performance'] = (
            self.executive_metrics['average_speed_performance'] * 0.9 + 
            reward.speed_performance * 0.1
        )
        self.executive_metrics['average_accuracy_performance'] = (
            self.executive_metrics['average_accuracy_performance'] * 0.9 + 
            reward.accuracy_performance * 0.1
        )
        self.executive_metrics['average_resource_efficiency'] = (
            self.executive_metrics['average_resource_efficiency'] * 0.9 + 
            reward.resource_efficiency * 0.1
        )
        self.executive_metrics['average_strategic_alignment'] = (
            self.executive_metrics['average_strategic_alignment'] * 0.9 + 
            reward.strategic_alignment * 0.1
        )
        
        # Add to episode tracking
        self.episode_rewards.append(reward.total_reward)
        
        # Train if enough experiences
        if len(self.experience_buffer) >= self.batch_size and self.training_step % self.update_frequency == 0:
            await self._train_executive_policy()
        
        self.training_step += 1
        
        logger.debug(f"Processed executive outcome for decision {decision_id}, reward: {reward.total_reward:.3f}")
    
    def _calculate_executive_reward(self, outcome: Dict[str, Any]) -> ExecutiveReward:
        """Calculate multi-dimensional reward from decision outcome"""
        
        # Speed performance component
        response_time = outcome.get('response_time', 5.0)
        speed_performance = max(0.0, 1.0 - (response_time / 10.0))  # Normalize to 10s max
        
        # Accuracy performance component
        accuracy_performance = outcome.get('accuracy_score', 0.5)
        
        # Resource efficiency component
        cpu_used = outcome.get('cpu_usage', 0.5)
        memory_used = outcome.get('memory_usage', 0.5)
        network_used = outcome.get('network_usage', 0.3)
        resource_efficiency = 1.0 - ((cpu_used + memory_used + network_used) / 3.0)
        
        # Strategic alignment component
        strategic_alignment = outcome.get('strategic_alignment', 0.5)
        
        # Learning value component
        learning_value = outcome.get('learning_progress', 0.3)
        
        # User satisfaction component
        user_satisfaction = outcome.get('user_satisfaction', accuracy_performance)
        
        reward = ExecutiveReward(
            speed_performance=speed_performance,
            accuracy_performance=accuracy_performance,
            resource_efficiency=resource_efficiency,
            strategic_alignment=strategic_alignment,
            learning_value=learning_value,
            user_satisfaction=user_satisfaction
        )
        
        # Compute reward based on current objective
        reward.compute_multi_objective_reward(
            self.current_objective,
            self.objective_weights.get(self.current_objective)
        )
        
        return reward
    
    async def _train_executive_policy(self) -> None:
        """Train the DRL executive policy network"""
        
        if len(self.experience_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = self.experience_buffer.sample(self.batch_size)
        
        states = torch.stack([exp['state'] for exp in batch])
        actions = torch.tensor([exp['action'] for exp in batch], dtype=torch.long)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32)
        objectives = [exp['objective'] for exp in batch]
        
        # Forward pass
        action_probs, objective_probs, priority_adjustment, resource_allocation, thresholds, objective_values = \
            self.policy_network(states)
        
        # Multi-objective value calculation
        batch_objective_values = []
        for i, obj in enumerate(objectives):
            obj_idx = self.executive_objectives.index(obj)
            batch_objective_values.append(objective_values[i, obj_idx])
        
        batch_objective_values = torch.stack(batch_objective_values)
        
        # Calculate advantages
        advantages = rewards - batch_objective_values
        
        # Action policy loss
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze()
        action_policy_loss = -(action_log_probs * advantages.detach()).mean()
        
        # Objective policy loss
        objective_targets = torch.zeros(len(batch), len(self.executive_objectives))
        for i, obj in enumerate(objectives):
            obj_idx = self.executive_objectives.index(obj)
            objective_targets[i, obj_idx] = 1.0
        
        objective_policy_loss = F.cross_entropy(objective_probs, torch.argmax(objective_targets, dim=1))
        
        # Value loss
        value_loss = F.mse_loss(batch_objective_values, rewards)
        
        # Priority adjustment loss (encourage adaptive priorities)
        priority_target = torch.clamp(rewards, 0.0, 1.0).unsqueeze(1)
        priority_loss = F.mse_loss(priority_adjustment, priority_target)
        
        # Threshold adaptation loss (encourage stable but adaptive thresholds)
        threshold_target = torch.clamp(rewards.unsqueeze(1).repeat(1, 4), 0.0, 1.0)
        threshold_loss = F.mse_loss(thresholds, threshold_target)
        
        # Total loss
        total_loss = (action_policy_loss + 
                     0.3 * objective_policy_loss +
                     0.5 * value_loss +
                     0.1 * priority_loss +
                     0.1 * threshold_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self._soft_update_target_network()
        
        # Update metrics
        self.executive_metrics['policy_loss'] = action_policy_loss.item()
        self.executive_metrics['value_loss'] = value_loss.item()
        self.executive_metrics['learning_episodes'] += 1
        
        logger.debug(f"Executive policy training - Loss: {total_loss.item():.4f}")
    
    def _soft_update_target_network(self) -> None:
        """Soft update of target network parameters"""
        for target_param, param in zip(self.target_network.parameters(), 
                                     self.policy_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def _estimate_task_complexity(self, content: Dict[str, Any]) -> float:
        """Estimate task complexity from content"""
        # Simplified complexity estimation
        complexity_indicators = [
            'analysis', 'synthesis', 'reasoning', 'planning', 'coordination',
            'multi-step', 'complex', 'detailed', 'comprehensive'
        ]
        
        text_content = str(content.get('description', '')) + str(content.get('prompt', ''))
        complexity_score = sum(1 for indicator in complexity_indicators 
                             if indicator in text_content.lower())
        
        return min(1.0, complexity_score / len(complexity_indicators))
    
    def _calculate_exploration_tendency(self) -> float:
        """Calculate current exploration vs exploitation tendency"""
        # Base exploration rate that decreases with experience
        base_exploration = 0.3
        experience_factor = min(1.0, len(self.decision_history) / 1000.0)
        
        # Adjust based on recent performance
        recent_rewards = list(self.episode_rewards)[-20:] if self.episode_rewards else [0.0]
        performance_variance = np.var(recent_rewards) if len(recent_rewards) > 1 else 0.5
        
        # Higher variance -> more exploration needed
        exploration_adjustment = min(0.3, performance_variance)
        
        exploration_tendency = base_exploration * (1.0 - experience_factor) + exploration_adjustment
        return max(0.05, min(0.8, exploration_tendency))
    
    def _calculate_confidence_level(self) -> float:
        """Calculate current confidence level in decision making"""
        if len(self.episode_rewards) < 10:
            return 0.5
        
        recent_rewards = list(self.episode_rewards)[-10:]
        mean_performance = np.mean(recent_rewards)
        std_performance = np.std(recent_rewards)
        
        # Higher mean and lower std -> higher confidence
        confidence = mean_performance * (1.0 - std_performance)
        return max(0.1, min(0.95, confidence))
    
    def _calculate_learning_progress(self) -> float:
        """Calculate recent learning progress"""
        if len(self.episode_rewards) < 20:
            return 0.3
        
        recent_rewards = list(self.episode_rewards)[-20:]
        early_performance = np.mean(recent_rewards[:10])
        late_performance = np.mean(recent_rewards[10:])
        
        progress = (late_performance - early_performance) / 2.0 + 0.5
        return max(0.0, min(1.0, progress))
    
    def _monitor_decision_integrity(self, executive_decision: Dict[str, Any],
                                  state: ExecutiveState) -> Dict[str, Any]:
        """Monitor executive decision integrity"""
        try:
            violations = []
            
            # Check threshold bounds
            for threshold_name, threshold_value in executive_decision.get('adaptive_thresholds', {}).items():
                if threshold_value < 0.0 or threshold_value > 1.0:
                    violations.append(IntegrityViolation(
                        violation_type=ViolationType.INVALID_METRIC,
                        description=f"Threshold {threshold_name} = {threshold_value} outside valid range [0,1]",
                        severity="HIGH",
                        context={"threshold": threshold_name, "value": threshold_value}
                    ))
            
            # Check resource allocation sums to ~1
            resource_allocation = executive_decision.get('resource_allocation', {})
            total_allocation = sum(resource_allocation.values())
            if abs(total_allocation - 1.0) > 0.1:
                violations.append(IntegrityViolation(
                    violation_type=ViolationType.PERFORMANCE_CLAIM,
                    description=f"Resource allocation sum {total_allocation} deviates from 1.0",
                    severity="MEDIUM",
                    context={"total_allocation": total_allocation}
                ))
            
            # Apply corrections
            if violations:
                self.integrity_violations.extend(violations)
                
                if self.auto_correction_enabled:
                    # Clamp thresholds
                    corrected_thresholds = {}
                    for name, value in executive_decision.get('adaptive_thresholds', {}).items():
                        corrected_thresholds[name] = max(0.0, min(1.0, value))
                    executive_decision['adaptive_thresholds'] = corrected_thresholds
                    
                    # Normalize resource allocation
                    if total_allocation > 0:
                        normalized_allocation = {
                            k: v / total_allocation for k, v in resource_allocation.items()
                        }
                        executive_decision['resource_allocation'] = normalized_allocation
            
            return executive_decision
            
        except Exception as e:
            logger.error(f"Error in decision integrity monitoring: {e}")
            return executive_decision
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            **self.executive_metrics,
            'episode_rewards_mean': np.mean(list(self.episode_rewards)) if self.episode_rewards else 0.0,
            'episode_rewards_std': np.std(list(self.episode_rewards)) if self.episode_rewards else 0.0,
            'current_objective': self.current_objective.value,
            'adaptive_thresholds': self.adaptive_thresholds,
            'objective_performance_averages': {
                obj.value: np.mean(list(perf_history)) if perf_history else 0.0
                for obj, perf_history in self.objective_performance.items()
            },
            'exploration_tendency': self._calculate_exploration_tendency(),
            'confidence_level': self._calculate_confidence_level(),
            'learning_progress': self._calculate_learning_progress(),
            'integrity_violations': len(self.integrity_violations),
            'experience_buffer_size': len(self.experience_buffer),
            'decision_history_length': len(self.decision_history)
        }
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model"""
        try:
            torch.save({
                'policy_network_state_dict': self.policy_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'executive_metrics': self.executive_metrics,
                'training_step': self.training_step,
                'current_objective': self.current_objective,
                'adaptive_thresholds': self.adaptive_thresholds,
                'objective_weights': self.objective_weights,
                'decision_history': list(self.decision_history)
            }, filepath)
            
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model"""
        try:
            checkpoint = torch.load(filepath)
            
            self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.executive_metrics = checkpoint['executive_metrics']
            self.training_step = checkpoint['training_step']
            self.current_objective = checkpoint['current_objective']
            self.adaptive_thresholds = checkpoint['adaptive_thresholds']
            self.objective_weights = checkpoint['objective_weights']
            
            # Restore decision history
            history_data = checkpoint.get('decision_history', [])
            self.decision_history = deque(history_data, maxlen=500)
            
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False 
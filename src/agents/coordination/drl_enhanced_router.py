"""
DRL-Enhanced Agent Router for NIS Protocol

This module provides Deep Reinforcement Learning capabilities for intelligent
agent routing and selection. The DRL system learns optimal policies for:
- Agent selection based on task characteristics and performance history
- Dynamic load balancing across available agents
- Multi-objective optimization (speed, accuracy, resource usage)
- Adaptive routing strategies that improve over time

Enhanced Features:
- DRL-based agent selection policies using Actor-Critic networks
- Performance feedback learning with reward engineering
- Dynamic strategy adaptation based on system state
- Multi-objective reward functions for balanced optimization
- Integration with existing AgentRouter infrastructure
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
from src.agents/learning.drl_foundation import (
    DRLPolicyNetwork, DRLEnvironment, DRLAgent, RewardSignal,
    DRLMetrics, TrainingConfig, ExperienceBuffer
)

# Import existing router infrastructure
from src.agents.agent_router import (
    EnhancedAgentRouter, RoutingDecision, TaskPriority,
    AgentCapability, RoutingStrategy
)

# Integration and infrastructure
from src.infrastructure.integration_coordinator import (
    InfrastructureCoordinator, ServiceHealth
)
from src.infrastructure.caching_system import CacheStrategy

# Self-audit and integrity
from src.utils.self_audit import self_audit_engine, ViolationType, IntegrityViolation
from src.utils.integrity_metrics import (
    calculate_confidence, create_default_confidence_factors
)

# Configure logging
logger = logging.getLogger("drl_enhanced_router")


class AgentRoutingAction(Enum):
    """Actions the DRL agent can take for routing decisions"""
    SELECT_SINGLE_SPECIALIST = "select_single_specialist"
    SELECT_MULTI_AGENT_TEAM = "select_multi_agent_team"
    ROUTE_TO_BEST_PERFORMER = "route_to_best_performer"
    LOAD_BALANCE_DISTRIBUTE = "load_balance_distribute"
    ESCALATE_TO_COORDINATOR = "escalate_to_coordinator"
    DEFER_TO_QUEUE = "defer_to_queue"


@dataclass 
class RoutingState:
    """State representation for DRL routing decisions"""
    # Task characteristics
    task_complexity: float
    task_priority: float
    task_type_embedding: List[float]  # One-hot or learned embedding
    estimated_duration: float
    resource_requirements: List[float]
    
    # System state
    agent_availability: List[float]  # Availability scores for each agent
    agent_performance_history: List[float]  # Recent performance scores
    agent_load_levels: List[float]  # Current load for each agent
    system_resource_usage: List[float]  # CPU, memory, network usage
    
    # Context features  
    time_of_day: float  # Normalized 0-1
    recent_task_outcomes: List[float]  # Success rates for recent tasks
    consensus_requirements: float  # Whether task needs consensus
    
    # Performance metrics
    average_response_time: float
    system_throughput: float
    error_rate: float


@dataclass
class RoutingReward:
    """Reward components for DRL training"""
    task_success: float          # Did the task complete successfully?
    response_time: float         # How fast was the response?
    resource_efficiency: float   # How efficiently were resources used?
    quality_score: float         # Quality of the output
    load_balance: float          # How well balanced was the load?
    cost_efficiency: float       # Cost vs. benefit ratio
    
    # Computed total reward
    total_reward: float = 0.0
    
    def compute_total_reward(self, weights: Dict[str, float] = None) -> float:
        """Compute weighted total reward"""
        if weights is None:
            weights = {
                'task_success': 0.3,
                'response_time': 0.2, 
                'resource_efficiency': 0.15,
                'quality_score': 0.15,
                'load_balance': 0.1,
                'cost_efficiency': 0.1
            }
        
        self.total_reward = (
            weights['task_success'] * self.task_success +
            weights['response_time'] * self.response_time +
            weights['resource_efficiency'] * self.resource_efficiency +
            weights['quality_score'] * self.quality_score +
            weights['load_balance'] * self.load_balance +
            weights['cost_efficiency'] * self.cost_efficiency
        )
        
        return self.total_reward


class RoutingPolicyNetwork(nn.Module):
    """Actor-Critic network for routing policy learning"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extraction layers
        self.feature_layers = nn.ModuleList()
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            self.feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.feature_layers.append(nn.ReLU())
            self.feature_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Agent selection head (which specific agents to choose)
        self.agent_selection_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 20),  # Max 20 agents
            nn.Sigmoid()  # Probability of selecting each agent
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the network"""
        # Extract features
        x = state
        for layer in self.feature_layers:
            x = layer(x)
        
        # Compute outputs
        action_probs = self.actor_head(x)
        state_value = self.critic_head(x)
        agent_selection_probs = self.agent_selection_head(x)
        
        return action_probs, state_value, agent_selection_probs


class DRLEnhancedRouter:
    """
    DRL-Enhanced Agent Router with intelligent policy learning
    
    This router learns optimal agent selection and routing strategies through
    deep reinforcement learning, continuously improving based on task outcomes
    and system performance.
    """
    
    def __init__(self,
                 state_dim: int = 50,  # Comprehensive state representation
                 action_dim: int = 6,  # Number of routing actions
                 learning_rate: float = 0.0003,
                 gamma: float = 0.99,  # Discount factor
                 tau: float = 0.005,   # Target network update rate
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 update_frequency: int = 4,
                 infrastructure_coordinator: Optional[InfrastructureCoordinator] = None,
                 enable_self_audit: bool = True):
        """Initialize DRL-Enhanced Router"""
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.infrastructure = infrastructure_coordinator
        
        # Initialize networks
        self.policy_network = RoutingPolicyNetwork(state_dim, action_dim)
        self.target_network = RoutingPolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Copy parameters to target network
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
        # Experience buffer for training
        self.experience_buffer = ExperienceBuffer(buffer_size)
        
        # Routing state and metrics
        self.current_state: Optional[RoutingState] = None
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.routing_metrics = {
            'total_routes': 0,
            'successful_routes': 0,
            'average_response_time': 0.0,
            'average_quality_score': 0.0,
            'load_balance_score': 0.0,
            'learning_episodes': 0,
            'policy_loss': 0.0,
            'value_loss': 0.0
        }
        
        # Training state
        self.training_step = 0
        self.episode_rewards: deque = deque(maxlen=1000)
        self.episode_lengths: deque = deque(maxlen=1000)
        
        # Available routing actions
        self.routing_actions = list(AgentRoutingAction)
        
        # Agent capabilities and availability tracking
        self.available_agents: List[str] = []
        self.agent_capabilities: Dict[str, List[AgentCapability]] = {}
        self.agent_load_tracking: Dict[str, float] = {}
        
        # Self-audit integration
        self.enable_self_audit = enable_self_audit
        self.integrity_violations: deque = deque(maxlen=100)
        self.auto_correction_enabled = True
        
        # Reward engineering parameters
        self.reward_weights = {
            'task_success': 0.3,
            'response_time': 0.2,
            'resource_efficiency': 0.15,
            'quality_score': 0.15,
            'load_balance': 0.1,
            'cost_efficiency': 0.1
        }
        
        # Redis caching for performance data
        self.cache_ttl = 3600  # 1 hour
        
        logger.info("DRL-Enhanced Router initialized with intelligent policy learning")
    
    def register_agent(self, agent_id: str, capabilities: List[AgentCapability], 
                      current_load: float = 0.0) -> bool:
        """Register an agent with the router"""
        self.agent_registry[agent_id] = {
            'capabilities': capabilities,
            'current_load': current_load,
            'last_seen': time.time(),
            'performance_score': 0.5,  # Initial neutral score
            'availability': True
        }
        
        if agent_id not in self.available_agents:
            self.available_agents.append(agent_id)
        
        self.agent_capabilities[agent_id] = capabilities
        self.agent_load_tracking[agent_id] = current_load
        
        logger.info(f"Registered agent {agent_id} with DRL router")
        return True
    
    def extract_routing_state(self, task: Dict[str, Any], 
                            system_context: Dict[str, Any]) -> RoutingState:
        """Extract comprehensive state representation for DRL decision making"""
        
        # Task characteristics
        task_complexity = self._estimate_task_complexity(task)
        task_priority = task.get('priority', 0.5)
        task_type_embedding = self._encode_task_type(task.get('type', 'general'))
        estimated_duration = self._estimate_task_duration(task)
        resource_requirements = self._estimate_resource_requirements(task)
        
        # System state
        agent_availability = [
            self.agent_registry.get(agent_id, {}).get('availability', 0) * 
            (1.0 - self.agent_load_tracking.get(agent_id, 0))
            for agent_id in self.available_agents[:20]  # Limit to 20 agents
        ]
        
        # Pad or truncate to fixed size
        agent_availability = (agent_availability + [0.0] * 20)[:20]
        
        agent_performance_history = [
            np.mean(list(self.performance_history[agent_id])) if self.performance_history[agent_id] else 0.5
            for agent_id in self.available_agents[:20]
        ]
        agent_performance_history = (agent_performance_history + [0.5] * 20)[:20]
        
        agent_load_levels = [
            self.agent_load_tracking.get(agent_id, 0.0)
            for agent_id in self.available_agents[:20]
        ]
        agent_load_levels = (agent_load_levels + [0.0] * 20)[:20]
        
        # System resource usage (simplified)
        system_resource_usage = [
            system_context.get('cpu_usage', 0.5),
            system_context.get('memory_usage', 0.5),
            system_context.get('network_usage', 0.5)
        ]
        
        # Context features
        time_of_day = (time.time() % 86400) / 86400  # Normalized time of day
        
        recent_outcomes = list(self.episode_rewards)[-10:] if self.episode_rewards else [0.0]
        recent_task_outcomes = (recent_outcomes + [0.0] * 10)[:10]
        
        consensus_requirements = float(task.get('requires_consensus', False))
        
        # Performance metrics
        avg_response_time = self.routing_metrics.get('average_response_time', 1.0)
        system_throughput = len(self.episode_rewards) / max(time.time() - getattr(self, 'start_time', time.time()), 1)
        error_rate = 1.0 - (self.routing_metrics.get('successful_routes', 0) / 
                           max(self.routing_metrics.get('total_routes', 1), 1))
        
        state = RoutingState(
            task_complexity=task_complexity,
            task_priority=task_priority,
            task_type_embedding=task_type_embedding,
            estimated_duration=estimated_duration,
            resource_requirements=resource_requirements,
            agent_availability=agent_availability,
            agent_performance_history=agent_performance_history,
            agent_load_levels=agent_load_levels,
            system_resource_usage=system_resource_usage,
            time_of_day=time_of_day,
            recent_task_outcomes=recent_task_outcomes,
            consensus_requirements=consensus_requirements,
            average_response_time=avg_response_time,
            system_throughput=system_throughput,
            error_rate=error_rate
        )
        
        return state
    
    def state_to_tensor(self, state: RoutingState) -> torch.Tensor:
        """Convert routing state to tensor for neural network"""
        features = []
        
        # Task characteristics
        features.extend([
            state.task_complexity,
            state.task_priority,
            state.estimated_duration,
            state.consensus_requirements
        ])
        
        # Task type embedding (assume 5-dim)
        features.extend(state.task_type_embedding[:5] if len(state.task_type_embedding) >= 5 
                       else state.task_type_embedding + [0.0] * (5 - len(state.task_type_embedding)))
        
        # Resource requirements (assume 3-dim: CPU, memory, network)
        features.extend(state.resource_requirements[:3] if len(state.resource_requirements) >= 3
                       else state.resource_requirements + [0.0] * (3 - len(state.resource_requirements)))
        
        # Agent states (20 agents x 3 features = 60 features)
        features.extend(state.agent_availability)  # 20 features
        features.extend(state.agent_performance_history)  # 20 features
        features.extend(state.agent_load_levels)  # 20 features
        
        # System state
        features.extend(state.system_resource_usage)  # 3 features
        features.extend([
            state.time_of_day,
            state.average_response_time,
            state.system_throughput,
            state.error_rate
        ])
        
        # Recent task outcomes
        features.extend(state.recent_task_outcomes)  # 10 features
        
        # Ensure exactly state_dim features
        features = features[:self.state_dim]
        features.extend([0.0] * (self.state_dim - len(features)))
        
        return torch.FloatTensor(features).unsqueeze(0)
    
    async def route_task_with_drl(self, task: Dict[str, Any], 
                                system_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Route task using DRL policy"""
        
        if system_context is None:
            system_context = {}
        
        # Extract state
        state = self.extract_routing_state(task, system_context)
        self.current_state = state
        
        # Convert to tensor
        state_tensor = self.state_to_tensor(state)
        
        # Get policy predictions
        with torch.no_grad():
            action_probs, state_value, agent_selection_probs = self.policy_network(state_tensor)
        
        # Sample action from policy
        action_dist = torch.distributions.Categorical(action_probs)
        action_idx = action_dist.sample()
        action = self.routing_actions[action_idx.item()]
        
        # Select specific agents based on action and agent selection probabilities
        selected_agents = self._select_agents_from_probabilities(
            agent_selection_probs[0], action, task
        )
        
        # Create routing decision
        routing_decision = {
            'action': action.value,
            'selected_agents': selected_agents,
            'confidence': torch.max(action_probs).item(),
            'estimated_value': state_value.item(),
            'agent_selection_probs': agent_selection_probs[0].tolist(),
            'state_features': state_tensor[0].tolist(),
            'routing_strategy': self._map_action_to_strategy(action),
            'task_id': task.get('task_id', f"task_{int(time.time())}")
        }
        
        # Cache decision for performance tracking
        if self.infrastructure and self.infrastructure.redis_manager:
            cache_key = f"drl_routing_decision:{routing_decision['task_id']}"
            await self.infrastructure.cache_data(
                cache_key, routing_decision, 
                agent_id="drl_router", ttl=self.cache_ttl
            )
        
        # Apply self-audit monitoring
        if self.enable_self_audit:
            routing_decision = self._monitor_routing_integrity(routing_decision, state)
        
        # Update metrics
        self.routing_metrics['total_routes'] += 1
        
        logger.info(f"DRL routing decision: {action.value} -> {selected_agents}")
        
        return routing_decision
    
    def _select_agents_from_probabilities(self, agent_probs: torch.Tensor, 
                                        action: AgentRoutingAction, 
                                        task: Dict[str, Any]) -> List[str]:
        """Select specific agents based on probabilities and action type"""
        
        available_count = len(self.available_agents)
        if available_count == 0:
            return []
        
        # Get probabilities for available agents
        probs = agent_probs[:available_count].cpu().numpy()
        
        if action == AgentRoutingAction.SELECT_SINGLE_SPECIALIST:
            # Select single best agent
            best_idx = np.argmax(probs)
            return [self.available_agents[best_idx]]
        
        elif action == AgentRoutingAction.SELECT_MULTI_AGENT_TEAM:
            # Select top 2-3 agents
            top_indices = np.argsort(probs)[-3:]
            return [self.available_agents[idx] for idx in top_indices if probs[idx] > 0.3]
        
        elif action == AgentRoutingAction.ROUTE_TO_BEST_PERFORMER:
            # Select based on performance history
            performance_scores = [
                np.mean(list(self.performance_history[agent_id])) if self.performance_history[agent_id] else 0.5
                for agent_id in self.available_agents[:available_count]
            ]
            best_performer_idx = np.argmax(performance_scores)
            return [self.available_agents[best_performer_idx]]
        
        elif action == AgentRoutingAction.LOAD_BALANCE_DISTRIBUTE:
            # Select agents with lowest load
            load_scores = [
                self.agent_load_tracking.get(agent_id, 0.0)
                for agent_id in self.available_agents[:available_count]
            ]
            sorted_indices = np.argsort(load_scores)
            return [self.available_agents[idx] for idx in sorted_indices[:2]]
        
        elif action == AgentRoutingAction.ESCALATE_TO_COORDINATOR:
            # Route to coordinator agents
            coordinator_agents = [
                agent_id for agent_id in self.available_agents
                if 'coordinator' in agent_id.lower() or 'manager' in agent_id.lower()
            ]
            return coordinator_agents[:1] if coordinator_agents else [self.available_agents[0]]
        
        else:  # DEFER_TO_QUEUE
            # No immediate agent selection
            return []
    
    def _map_action_to_strategy(self, action: AgentRoutingAction) -> RoutingStrategy:
        """Map DRL action to traditional routing strategy"""
        mapping = {
            AgentRoutingAction.SELECT_SINGLE_SPECIALIST: RoutingStrategy.CAPABILITY_BASED,
            AgentRoutingAction.SELECT_MULTI_AGENT_TEAM: RoutingStrategy.MULTI_AGENT,
            AgentRoutingAction.ROUTE_TO_BEST_PERFORMER: RoutingStrategy.PERFORMANCE_BASED,
            AgentRoutingAction.LOAD_BALANCE_DISTRIBUTE: RoutingStrategy.LOAD_BALANCED,
            AgentRoutingAction.ESCALATE_TO_COORDINATOR: RoutingStrategy.HIERARCHICAL,
            AgentRoutingAction.DEFER_TO_QUEUE: RoutingStrategy.ROUND_ROBIN
        }
        return mapping.get(action, RoutingStrategy.CAPABILITY_BASED)
    
    async def process_task_outcome(self, task_id: str, outcome: Dict[str, Any]) -> None:
        """Process task outcome for DRL learning"""
        
        # Calculate reward components
        reward = self._calculate_reward(outcome)
        
        # Update performance history for agents involved
        selected_agents = outcome.get('selected_agents', [])
        for agent_id in selected_agents:
            performance_score = outcome.get('quality_score', 0.5)
            self.performance_history[agent_id].append(performance_score)
        
        # Store experience for training
        if hasattr(self, 'last_state_tensor') and hasattr(self, 'last_action'):
            experience = {
                'state': self.last_state_tensor,
                'action': self.last_action,
                'reward': reward.total_reward,
                'next_state': None,  # Will be filled on next decision
                'done': True
            }
            self.experience_buffer.add(experience)
        
        # Update metrics
        if outcome.get('success', False):
            self.routing_metrics['successful_routes'] += 1
        
        response_time = outcome.get('response_time', 1.0)
        self.routing_metrics['average_response_time'] = (
            self.routing_metrics['average_response_time'] * 0.9 + response_time * 0.1
        )
        
        quality_score = outcome.get('quality_score', 0.5)
        self.routing_metrics['average_quality_score'] = (
            self.routing_metrics['average_quality_score'] * 0.9 + quality_score * 0.1
        )
        
        # Add to episode tracking
        self.episode_rewards.append(reward.total_reward)
        
        # Train if enough experiences
        if len(self.experience_buffer) >= self.batch_size and self.training_step % self.update_frequency == 0:
            await self._train_policy()
        
        self.training_step += 1
        
        logger.debug(f"Processed outcome for task {task_id}, reward: {reward.total_reward:.3f}")
    
    def _calculate_reward(self, outcome: Dict[str, Any]) -> RoutingReward:
        """Calculate reward signal from task outcome"""
        
        # Task success component
        task_success = 1.0 if outcome.get('success', False) else -0.5
        
        # Response time component (faster is better)
        response_time = outcome.get('response_time', 10.0)
        response_time_reward = max(0.0, 1.0 - (response_time / 30.0))  # Normalize to 30s max
        
        # Resource efficiency component
        cpu_usage = outcome.get('cpu_usage', 0.5)
        memory_usage = outcome.get('memory_usage', 0.5) 
        resource_efficiency = 1.0 - ((cpu_usage + memory_usage) / 2.0)
        
        # Quality score component
        quality_score = outcome.get('quality_score', 0.5)
        
        # Load balance component
        agent_loads = outcome.get('agent_loads', [0.5])
        load_variance = np.var(agent_loads) if len(agent_loads) > 1 else 0.0
        load_balance = max(0.0, 1.0 - load_variance)
        
        # Cost efficiency component
        cost = outcome.get('cost', 0.1)
        benefit = outcome.get('benefit', quality_score)
        cost_efficiency = benefit / max(cost, 0.01)  # Avoid division by zero
        cost_efficiency = min(1.0, cost_efficiency)  # Cap at 1.0
        
        reward = RoutingReward(
            task_success=task_success,
            response_time=response_time_reward,
            resource_efficiency=resource_efficiency,
            quality_score=quality_score,
            load_balance=load_balance,
            cost_efficiency=cost_efficiency
        )
        
        reward.compute_total_reward(self.reward_weights)
        
        return reward
    
    async def _train_policy(self) -> None:
        """Train the DRL policy network"""
        
        if len(self.experience_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = self.experience_buffer.sample(self.batch_size)
        
        states = torch.stack([exp['state'] for exp in batch])
        actions = torch.tensor([exp['action'] for exp in batch], dtype=torch.long)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32)
        
        # Forward pass
        action_probs, state_values, _ = self.policy_network(states)
        
        # Calculate advantages
        advantages = rewards - state_values.squeeze()
        
        # Policy loss (REINFORCE with baseline)
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze()
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        
        # Value loss
        value_loss = F.mse_loss(state_values.squeeze(), rewards)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self._soft_update_target_network()
        
        # Update metrics
        self.routing_metrics['policy_loss'] = policy_loss.item()
        self.routing_metrics['value_loss'] = value_loss.item()
        self.routing_metrics['learning_episodes'] += 1
        
        logger.debug(f"Policy training - Loss: {total_loss.item():.4f}")
    
    def _soft_update_target_network(self) -> None:
        """Soft update of target network parameters"""
        for target_param, param in zip(self.target_network.parameters(), 
                                     self.policy_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def _estimate_task_complexity(self, task: Dict[str, Any]) -> float:
        """Estimate task complexity from task description"""
        # Simplified complexity estimation
        description = task.get('description', '')
        
        complexity_indicators = [
            'analyze', 'complex', 'multi-step', 'reasoning', 'synthesis',
            'comprehensive', 'detailed', 'research', 'evaluation'
        ]
        
        complexity_score = sum(1 for indicator in complexity_indicators 
                             if indicator in description.lower())
        
        return min(1.0, complexity_score / len(complexity_indicators))
    
    def _encode_task_type(self, task_type: str) -> List[float]:
        """Encode task type as embedding vector"""
        # Simplified one-hot encoding for common task types
        task_types = [
            'reasoning', 'analysis', 'generation', 'classification', 
            'coordination', 'general'
        ]
        
        encoding = [0.0] * len(task_types)
        if task_type.lower() in task_types:
            idx = task_types.index(task_type.lower())
            encoding[idx] = 1.0
        else:
            encoding[-1] = 1.0  # Default to 'general'
        
        return encoding
    
    def _estimate_task_duration(self, task: Dict[str, Any]) -> float:
        """Estimate task duration in normalized units"""
        # Simplified duration estimation based on complexity and type
        complexity = self._estimate_task_complexity(task)
        base_duration = task.get('estimated_duration', 10.0)
        
        # Normalize to 0-1 range assuming max 60 seconds
        duration_score = min(1.0, (base_duration * (1 + complexity)) / 60.0)
        
        return duration_score
    
    def _estimate_resource_requirements(self, task: Dict[str, Any]) -> List[float]:
        """Estimate resource requirements [CPU, Memory, Network]"""
        complexity = self._estimate_task_complexity(task)
        
        # Simple resource estimation
        cpu_req = 0.3 + complexity * 0.4  # Base 0.3, up to 0.7
        memory_req = 0.2 + complexity * 0.3  # Base 0.2, up to 0.5
        network_req = 0.1 + complexity * 0.2  # Base 0.1, up to 0.3
        
        return [cpu_req, memory_req, network_req]
    
    def _monitor_routing_integrity(self, routing_decision: Dict[str, Any], 
                                 state: RoutingState) -> Dict[str, Any]:
        """Monitor routing decision integrity"""
        try:
            violations = []
            
            # Check confidence bounds
            confidence = routing_decision.get('confidence', 0.0)
            if confidence < 0.0 or confidence > 1.0:
                violations.append(IntegrityViolation(
                    violation_type=ViolationType.INVALID_METRIC,
                    description=f"Confidence {confidence} outside valid range [0,1]",
                    severity="HIGH",
                    context={"confidence": confidence}
                ))
            
            # Check if agents are actually available
            selected_agents = routing_decision.get('selected_agents', [])
            unavailable_agents = [agent for agent in selected_agents 
                                if agent not in self.available_agents]
            
            if unavailable_agents:
                violations.append(IntegrityViolation(
                    violation_type=ViolationType.PERFORMANCE_CLAIM,
                    description=f"Selected unavailable agents: {unavailable_agents}",
                    severity="MEDIUM",
                    context={"unavailable_agents": unavailable_agents}
                ))
            
            # Apply corrections
            if violations:
                self.integrity_violations.extend(violations)
                
                if self.auto_correction_enabled:
                    # Clamp confidence
                    routing_decision['confidence'] = max(0.0, min(1.0, confidence))
                    
                    # Remove unavailable agents
                    routing_decision['selected_agents'] = [
                        agent for agent in selected_agents 
                        if agent in self.available_agents
                    ]
            
            return routing_decision
            
        except Exception as e:
            logger.error(f"Error in routing integrity monitoring: {e}")
            return routing_decision
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            **self.routing_metrics,
            'episode_rewards_mean': np.mean(list(self.episode_rewards)) if self.episode_rewards else 0.0,
            'episode_rewards_std': np.std(list(self.episode_rewards)) if self.episode_rewards else 0.0,
            'available_agents': len(self.available_agents),
            'integrity_violations': len(self.integrity_violations),
            'experience_buffer_size': len(self.experience_buffer),
            'agent_performance_scores': {
                agent_id: np.mean(list(history)) if history else 0.5
                for agent_id, history in self.performance_history.items()
            }
        }
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model"""
        try:
            torch.save({
                'policy_network_state_dict': self.policy_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'routing_metrics': self.routing_metrics,
                'training_step': self.training_step
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
            self.routing_metrics = checkpoint['routing_metrics']
            self.training_step = checkpoint['training_step']
            
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False 
"""
DRL-Enhanced Multi-LLM Agent for NIS Protocol

This module provides Deep Reinforcement Learning capabilities for intelligent
LLM provider selection and orchestration. The DRL system learns optimal policies for:
- Dynamic provider selection based on task type, cost, and quality
- Intelligent strategy selection (consensus vs. specialist vs. ensemble)
- Cost-quality optimization through learned provider combinations
- Adaptive orchestration strategies that improve over time

Enhanced Features:
- DRL-based provider selection policies using Multi-Head Actor-Critic networks
- Dynamic strategy learning with multi-objective rewards (cost, quality, speed)
- Adaptive consensus mechanisms with learned thresholds
- Provider performance learning with temporal patterns
- Integration with existing Multi-LLM infrastructure
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

# Import existing Multi-LLM infrastructure
from src.agents.coordination.multi_llm_agent import (
    EnhancedMultiLLMAgent, LLMOrchestrationStrategy, LLMCoordinationMode,
    LLMOrchestrationResult, LLMOrchestrationTask
)

from src.llm.providers.llm_provider_manager import (
    LLMProviderManager, TaskType, LLMProvider, ResponseConfidence
)

# Integration and infrastructure
from src.infrastructure.integration_coordinator import InfrastructureCoordinator
from src.infrastructure.caching_system import CacheStrategy

# Self-audit and integrity
from src.utils.self_audit import self_audit_engine, ViolationType, IntegrityViolation
from src.utils.integrity_metrics import (
    calculate_confidence, create_default_confidence_factors
)

# Configure logging
logger = logging.getLogger("drl_enhanced_multi_llm")


class LLMProviderAction(Enum):
    """Actions for LLM provider selection and orchestration"""
    SELECT_SINGLE_BEST = "select_single_best"
    SELECT_DIVERSE_PAIR = "select_diverse_pair"
    SELECT_CONSENSUS_TRIO = "select_consensus_trio"
    SELECT_ALL_AVAILABLE = "select_all_available"
    SELECT_COST_OPTIMAL = "select_cost_optimal"
    SELECT_SPEED_OPTIMAL = "select_speed_optimal"
    SELECT_QUALITY_OPTIMAL = "select_quality_optimal"
    ADAPTIVE_SELECTION = "adaptive_selection"


@dataclass
class LLMOrchestrationState:
    """State representation for DRL LLM orchestration decisions"""
    # Task characteristics
    task_complexity: float
    task_type_embedding: List[float]  # One-hot encoding of task type
    prompt_length: float  # Normalized prompt length
    context_richness: float  # Amount of context provided
    quality_requirements: float  # Required quality level
    speed_requirements: float  # Speed urgency
    cost_constraints: float  # Cost budget constraints
    
    # Provider availability and performance
    provider_availability: List[float]  # Which providers are available
    provider_performance_history: List[float]  # Recent performance scores
    provider_cost_history: List[float]  # Recent cost patterns
    provider_speed_history: List[float]  # Recent response times
    provider_specializations: List[List[float]]  # Provider capabilities matrix
    
    # System state
    current_load: List[float]  # Current load on each provider
    system_cost_budget: float  # Remaining budget
    recent_provider_usage: List[float]  # Recent usage distribution
    consensus_success_rate: float  # Recent consensus success rate
    
    # Context features
    time_of_day: float
    recent_task_outcomes: List[float]  # Success rates for recent tasks
    system_performance_trend: float  # Improving/declining performance
    user_satisfaction_trend: float  # User satisfaction trend


@dataclass
class LLMOrchestrationReward:
    """Reward components for DRL training"""
    quality_score: float         # Quality of the final response
    cost_efficiency: float       # Cost vs. quality ratio
    response_time: float         # Speed of response generation
    consensus_coherence: float   # How well providers agreed
    provider_utilization: float  # Balanced provider usage
    user_satisfaction: float     # Estimated user satisfaction
    
    # Computed total reward
    total_reward: float = 0.0
    
    def compute_total_reward(self, weights: Dict[str, float] = None) -> float:
        """Compute weighted total reward"""
        if weights is None:
            weights = {
                'quality_score': 0.3,
                'cost_efficiency': 0.2,
                'response_time': 0.15,
                'consensus_coherence': 0.15,
                'provider_utilization': 0.1,
                'user_satisfaction': 0.1
            }
        
        self.total_reward = (
            weights['quality_score'] * self.quality_score +
            weights['cost_efficiency'] * self.cost_efficiency +
            weights['response_time'] * self.response_time +
            weights['consensus_coherence'] * self.consensus_coherence +
            weights['provider_utilization'] * self.provider_utilization +
            weights['user_satisfaction'] * self.user_satisfaction
        )
        
        return self.total_reward


class MultiHeadLLMPolicyNetwork(nn.Module):
    """Multi-head Actor-Critic network for LLM orchestration"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 num_providers: int = 5,
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_providers = num_providers
        
        # Shared feature extraction
        self.feature_layers = nn.ModuleList()
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            self.feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.feature_layers.append(nn.ReLU())
            self.feature_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Multi-head outputs for different aspects
        
        # Provider selection head
        self.provider_selection_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], num_providers),
            nn.Sigmoid()  # Probability of selecting each provider
        )
        
        # Strategy selection head
        self.strategy_selection_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim),
            nn.Softmax(dim=-1)  # Strategy probabilities
        )
        
        # Quality threshold head
        self.quality_threshold_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()  # Quality threshold for consensus
        )
        
        # Cost budget allocation head
        self.cost_allocation_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], num_providers),
            nn.Softmax(dim=-1)  # Cost budget allocation per provider
        )
        
        # Value function head
        self.value_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through multi-head network"""
        # Extract shared features
        x = state
        for layer in self.feature_layers:
            x = layer(x)
        
        # Compute all outputs
        provider_probs = self.provider_selection_head(x)
        strategy_probs = self.strategy_selection_head(x)
        quality_threshold = self.quality_threshold_head(x)
        cost_allocation = self.cost_allocation_head(x)
        state_value = self.value_head(x)
        
        return provider_probs, strategy_probs, quality_threshold, cost_allocation, state_value


class DRLEnhancedMultiLLM:
    """
    DRL-Enhanced Multi-LLM Agent with intelligent orchestration learning
    
    This agent learns optimal LLM provider selection and orchestration strategies
    through deep reinforcement learning, continuously adapting based on task
    outcomes, cost efficiency, and user satisfaction.
    """
    
    def __init__(self,
                 state_dim: int = 80,  # Comprehensive state representation
                 action_dim: int = 8,  # Number of orchestration strategies
                 num_providers: int = 5,  # Number of LLM providers
                 learning_rate: float = 0.0001,
                 gamma: float = 0.95,  # Discount factor
                 tau: float = 0.005,   # Target network update rate
                 buffer_size: int = 5000,
                 batch_size: int = 32,
                 update_frequency: int = 4,
                 infrastructure_coordinator: Optional[InfrastructureCoordinator] = None,
                 enable_self_audit: bool = True):
        """Initialize DRL-Enhanced Multi-LLM Agent"""
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_providers = num_providers
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.infrastructure = infrastructure_coordinator
        
        # Initialize networks
        self.policy_network = MultiHeadLLMPolicyNetwork(state_dim, action_dim, num_providers)
        self.target_network = MultiHeadLLMPolicyNetwork(state_dim, action_dim, num_providers)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Copy parameters to target network
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
        # Experience buffer
        self.experience_buffer = ExperienceBuffer(buffer_size)
        
        # Provider tracking
        self.provider_names = ["anthropic", "openai", "deepseek", "bitnet", "mock"]
        self.provider_performance: Dict[str, deque] = {
            provider: deque(maxlen=100) for provider in self.provider_names
        }
        self.provider_costs: Dict[str, deque] = {
            provider: deque(maxlen=100) for provider in self.provider_names
        }
        self.provider_speeds: Dict[str, deque] = {
            provider: deque(maxlen=100) for provider in self.provider_names
        }
        
        # Orchestration state and metrics
        self.current_state: Optional[LLMOrchestrationState] = None
        self.orchestration_metrics = {
            'total_orchestrations': 0,
            'successful_orchestrations': 0,
            'average_quality_score': 0.0,
            'average_cost_efficiency': 0.0,
            'average_response_time': 0.0,
            'consensus_success_rate': 0.0,
            'provider_utilization': {provider: 0.0 for provider in self.provider_names},
            'learning_episodes': 0,
            'policy_loss': 0.0,
            'value_loss': 0.0
        }
        
        # Training state
        self.training_step = 0
        self.episode_rewards: deque = deque(maxlen=1000)
        self.episode_lengths: deque = deque(maxlen=1000)
        
        # Available orchestration actions
        self.orchestration_actions = list(LLMProviderAction)
        
        # Self-audit integration
        self.enable_self_audit = enable_self_audit
        self.integrity_violations: deque = deque(maxlen=100)
        self.auto_correction_enabled = True
        
        # Reward engineering parameters
        self.reward_weights = {
            'quality_score': 0.3,
            'cost_efficiency': 0.2,
            'response_time': 0.15,
            'consensus_coherence': 0.15,
            'provider_utilization': 0.1,
            'user_satisfaction': 0.1
        }
        
        # Redis caching for orchestration data
        self.cache_ttl = 1800  # 30 minutes
        
        logger.info("DRL-Enhanced Multi-LLM Agent initialized with intelligent orchestration learning")
    
    def extract_orchestration_state(self, task: Dict[str, Any], 
                                  system_context: Dict[str, Any]) -> LLMOrchestrationState:
        """Extract comprehensive state representation for DRL decision making"""
        
        # Task characteristics
        task_complexity = self._estimate_task_complexity(task)
        task_type_embedding = self._encode_task_type(task.get('type', 'general'))
        
        prompt = task.get('prompt', '')
        prompt_length = min(1.0, len(prompt) / 1000.0)  # Normalize to max 1000 chars
        
        context_richness = min(1.0, len(task.get('context', {})) / 10.0)  # Normalize context
        
        quality_requirements = task.get('quality_threshold', 0.7)
        speed_requirements = task.get('speed_priority', 0.5)
        cost_constraints = task.get('cost_budget', 1.0)
        
        # Provider availability and performance
        provider_availability = [
            1.0 if provider in system_context.get('available_providers', []) else 0.0
            for provider in self.provider_names
        ]
        
        provider_performance_history = [
            np.mean(list(self.provider_performance[provider])) if self.provider_performance[provider] else 0.5
            for provider in self.provider_names
        ]
        
        provider_cost_history = [
            np.mean(list(self.provider_costs[provider])) if self.provider_costs[provider] else 0.1
            for provider in self.provider_names
        ]
        
        provider_speed_history = [
            1.0 - min(1.0, np.mean(list(self.provider_speeds[provider])) / 10.0) if self.provider_speeds[provider] else 0.5
            for provider in self.provider_names
        ]
        
        # Provider specializations (simplified capability matrix)
        provider_specializations = self._get_provider_specializations()
        
        # System state
        current_load = [
            system_context.get('provider_loads', {}).get(provider, 0.0)
            for provider in self.provider_names
        ]
        
        system_cost_budget = system_context.get('remaining_budget', 1.0)
        
        recent_usage = [
            self.orchestration_metrics['provider_utilization'].get(provider, 0.0)
            for provider in self.provider_names
        ]
        
        consensus_success_rate = self.orchestration_metrics.get('consensus_success_rate', 0.7)
        
        # Context features
        time_of_day = (time.time() % 86400) / 86400
        
        recent_outcomes = list(self.episode_rewards)[-10:] if self.episode_rewards else [0.0]
        recent_task_outcomes = (recent_outcomes + [0.0] * 10)[:10]
        
        performance_trend = self._calculate_performance_trend()
        user_satisfaction_trend = system_context.get('user_satisfaction_trend', 0.7)
        
        state = LLMOrchestrationState(
            task_complexity=task_complexity,
            task_type_embedding=task_type_embedding,
            prompt_length=prompt_length,
            context_richness=context_richness,
            quality_requirements=quality_requirements,
            speed_requirements=speed_requirements,
            cost_constraints=cost_constraints,
            provider_availability=provider_availability,
            provider_performance_history=provider_performance_history,
            provider_cost_history=provider_cost_history,
            provider_speed_history=provider_speed_history,
            provider_specializations=provider_specializations,
            current_load=current_load,
            system_cost_budget=system_cost_budget,
            recent_provider_usage=recent_usage,
            consensus_success_rate=consensus_success_rate,
            time_of_day=time_of_day,
            recent_task_outcomes=recent_task_outcomes,
            system_performance_trend=performance_trend,
            user_satisfaction_trend=user_satisfaction_trend
        )
        
        return state
    
    def state_to_tensor(self, state: LLMOrchestrationState) -> torch.Tensor:
        """Convert orchestration state to tensor for neural network"""
        features = []
        
        # Task characteristics (7 features)
        features.extend([
            state.task_complexity,
            state.prompt_length,
            state.context_richness,
            state.quality_requirements,
            state.speed_requirements,
            state.cost_constraints,
            state.time_of_day
        ])
        
        # Task type embedding (6 features)
        features.extend(state.task_type_embedding[:6] if len(state.task_type_embedding) >= 6
                       else state.task_type_embedding + [0.0] * (6 - len(state.task_type_embedding)))
        
        # Provider states (5 providers x 5 features = 25 features)
        features.extend(state.provider_availability)  # 5 features
        features.extend(state.provider_performance_history)  # 5 features
        features.extend(state.provider_cost_history)  # 5 features
        features.extend(state.provider_speed_history)  # 5 features
        features.extend(state.current_load)  # 5 features
        
        # Provider specializations flattened (5 providers x 6 specializations = 30 features)
        for spec_vector in state.provider_specializations:
            features.extend(spec_vector[:6] if len(spec_vector) >= 6
                           else spec_vector + [0.0] * (6 - len(spec_vector)))
        
        # System and context features (7 features)
        features.extend([
            state.system_cost_budget,
            state.consensus_success_rate,
            state.system_performance_trend,
            state.user_satisfaction_trend
        ])
        features.extend(state.recent_provider_usage)  # 5 features
        
        # Recent task outcomes (10 features)
        features.extend(state.recent_task_outcomes)
        
        # Ensure exactly state_dim features
        features = features[:self.state_dim]
        features.extend([0.0] * (self.state_dim - len(features)))
        
        return torch.FloatTensor(features).unsqueeze(0)
    
    async def orchestrate_with_drl(self, task: Dict[str, Any], 
                                 system_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Orchestrate LLM task using DRL policy"""
        
        if system_context is None:
            system_context = {}
        
        # Extract state
        state = self.extract_orchestration_state(task, system_context)
        self.current_state = state
        
        # Convert to tensor
        state_tensor = self.state_to_tensor(state)
        
        # Get policy predictions
        with torch.no_grad():
            provider_probs, strategy_probs, quality_threshold, cost_allocation, state_value = \
                self.policy_network(state_tensor)
        
        # Sample strategy from policy
        strategy_dist = torch.distributions.Categorical(strategy_probs)
        strategy_idx = strategy_dist.sample()
        selected_strategy = self.orchestration_actions[strategy_idx.item()]
        
        # Select providers based on probabilities and strategy
        selected_providers = self._select_providers_from_policy(
            provider_probs[0], selected_strategy, state
        )
        
        # Determine orchestration strategy mapping
        orchestration_strategy = self._map_action_to_orchestration_strategy(selected_strategy)
        coordination_mode = self._determine_coordination_mode(selected_strategy)
        
        # Create orchestration decision
        orchestration_decision = {
            'strategy': orchestration_strategy.value,
            'coordination_mode': coordination_mode.value,
            'selected_providers': selected_providers,
            'quality_threshold': quality_threshold.item(),
            'cost_allocation': cost_allocation[0].tolist(),
            'confidence': torch.max(strategy_probs).item(),
            'estimated_value': state_value.item(),
            'provider_selection_probs': provider_probs[0].tolist(),
            'state_features': state_tensor[0].tolist(),
            'drl_action': selected_strategy.value,
            'task_id': task.get('task_id', f"llm_task_{int(time.time())}")
        }
        
        # Cache decision for performance tracking
        if self.infrastructure and self.infrastructure.redis_manager:
            cache_key = f"drl_llm_orchestration:{orchestration_decision['task_id']}"
            await self.infrastructure.cache_data(
                cache_key, orchestration_decision,
                agent_id="drl_multi_llm", ttl=self.cache_ttl
            )
        
        # Apply self-audit monitoring
        if self.enable_self_audit:
            orchestration_decision = self._monitor_orchestration_integrity(orchestration_decision, state)
        
        # Update metrics
        self.orchestration_metrics['total_orchestrations'] += 1
        
        logger.info(f"DRL orchestration: {selected_strategy.value} -> {selected_providers}")
        
        return orchestration_decision
    
    def _select_providers_from_policy(self, provider_probs: torch.Tensor,
                                    strategy: LLMProviderAction,
                                    state: LLMOrchestrationState) -> List[str]:
        """Select specific providers based on policy probabilities and strategy"""
        
        probs = provider_probs.cpu().numpy()
        available_providers = [
            self.provider_names[i] for i, available in enumerate(state.provider_availability)
            if available > 0.5
        ]
        
        if not available_providers:
            return [self.provider_names[0]]  # Fallback
        
        if strategy == LLMProviderAction.SELECT_SINGLE_BEST:
            # Select highest probability available provider
            available_indices = [i for i, name in enumerate(self.provider_names) if name in available_providers]
            best_idx = max(available_indices, key=lambda i: probs[i])
            return [self.provider_names[best_idx]]
        
        elif strategy == LLMProviderAction.SELECT_DIVERSE_PAIR:
            # Select two providers with highest probabilities but different specializations
            available_indices = [i for i, name in enumerate(self.provider_names) if name in available_providers]
            sorted_indices = sorted(available_indices, key=lambda i: probs[i], reverse=True)
            return [self.provider_names[i] for i in sorted_indices[:2]]
        
        elif strategy == LLMProviderAction.SELECT_CONSENSUS_TRIO:
            # Select three providers for consensus
            available_indices = [i for i, name in enumerate(self.provider_names) if name in available_providers]
            sorted_indices = sorted(available_indices, key=lambda i: probs[i], reverse=True)
            return [self.provider_names[i] for i in sorted_indices[:3]]
        
        elif strategy == LLMProviderAction.SELECT_ALL_AVAILABLE:
            # Use all available providers
            return available_providers
        
        elif strategy == LLMProviderAction.SELECT_COST_OPTIMAL:
            # Select based on cost efficiency
            cost_efficiency_scores = [
                (probs[i] / max(state.provider_cost_history[i], 0.01))
                for i, name in enumerate(self.provider_names) if name in available_providers
            ]
            best_cost_idx = max(range(len(cost_efficiency_scores)), key=lambda i: cost_efficiency_scores[i])
            available_indices = [i for i, name in enumerate(self.provider_names) if name in available_providers]
            return [self.provider_names[available_indices[best_cost_idx]]]
        
        elif strategy == LLMProviderAction.SELECT_SPEED_OPTIMAL:
            # Select fastest providers
            available_indices = [i for i, name in enumerate(self.provider_names) if name in available_providers]
            fastest_idx = max(available_indices, key=lambda i: state.provider_speed_history[i])
            return [self.provider_names[fastest_idx]]
        
        elif strategy == LLMProviderAction.SELECT_QUALITY_OPTIMAL:
            # Select highest quality providers
            available_indices = [i for i, name in enumerate(self.provider_names) if name in available_providers]
            quality_idx = max(available_indices, key=lambda i: state.provider_performance_history[i])
            return [self.provider_names[quality_idx]]
        
        else:  # ADAPTIVE_SELECTION
            # Adaptive selection based on current state
            threshold = 0.3
            selected = [
                self.provider_names[i] for i, prob in enumerate(probs)
                if prob > threshold and self.provider_names[i] in available_providers
            ]
            return selected if selected else [available_providers[0]]
    
    def _map_action_to_orchestration_strategy(self, action: LLMProviderAction) -> LLMOrchestrationStrategy:
        """Map DRL action to orchestration strategy"""
        mapping = {
            LLMProviderAction.SELECT_SINGLE_BEST: LLMOrchestrationStrategy.SPECIALIST,
            LLMProviderAction.SELECT_DIVERSE_PAIR: LLMOrchestrationStrategy.VALIDATION,
            LLMProviderAction.SELECT_CONSENSUS_TRIO: LLMOrchestrationStrategy.CONSENSUS,
            LLMProviderAction.SELECT_ALL_AVAILABLE: LLMOrchestrationStrategy.ENSEMBLE,
            LLMProviderAction.SELECT_COST_OPTIMAL: LLMOrchestrationStrategy.PHYSICS_INFORMED,
            LLMProviderAction.SELECT_SPEED_OPTIMAL: LLMOrchestrationStrategy.SPECIALIST,
            LLMProviderAction.SELECT_QUALITY_OPTIMAL: LLMOrchestrationStrategy.SPECIALIST,
            LLMProviderAction.ADAPTIVE_SELECTION: LLMOrchestrationStrategy.COLLABORATIVE
        }
        return mapping.get(action, LLMOrchestrationStrategy.PHYSICS_INFORMED)
    
    def _determine_coordination_mode(self, action: LLMProviderAction) -> LLMCoordinationMode:
        """Determine coordination mode based on action"""
        if action in [LLMProviderAction.SELECT_CONSENSUS_TRIO, LLMProviderAction.SELECT_ALL_AVAILABLE]:
            return LLMCoordinationMode.PARALLEL
        elif action == LLMProviderAction.ADAPTIVE_SELECTION:
            return LLMCoordinationMode.ADAPTIVE
        else:
            return LLMCoordinationMode.PARALLEL
    
    async def process_orchestration_outcome(self, task_id: str, outcome: Dict[str, Any]) -> None:
        """Process orchestration outcome for DRL learning"""
        
        # Calculate reward components
        reward = self._calculate_orchestration_reward(outcome)
        
        # Update provider performance tracking
        selected_providers = outcome.get('selected_providers', [])
        for provider in selected_providers:
            if provider in self.provider_names:
                quality_score = outcome.get('quality_score', 0.5)
                cost = outcome.get('cost_per_provider', {}).get(provider, 0.1)
                response_time = outcome.get('response_time_per_provider', {}).get(provider, 1.0)
                
                self.provider_performance[provider].append(quality_score)
                self.provider_costs[provider].append(cost)
                self.provider_speeds[provider].append(response_time)
                
                # Update utilization metrics
                self.orchestration_metrics['provider_utilization'][provider] = (
                    self.orchestration_metrics['provider_utilization'][provider] * 0.9 + 0.1
                )
        
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
            self.orchestration_metrics['successful_orchestrations'] += 1
        
        quality_score = outcome.get('quality_score', 0.5)
        self.orchestration_metrics['average_quality_score'] = (
            self.orchestration_metrics['average_quality_score'] * 0.9 + quality_score * 0.1
        )
        
        cost_efficiency = reward.cost_efficiency
        self.orchestration_metrics['average_cost_efficiency'] = (
            self.orchestration_metrics['average_cost_efficiency'] * 0.9 + cost_efficiency * 0.1
        )
        
        response_time = outcome.get('response_time', 1.0)
        self.orchestration_metrics['average_response_time'] = (
            self.orchestration_metrics['average_response_time'] * 0.9 + response_time * 0.1
        )
        
        consensus_success = outcome.get('consensus_achieved', False)
        self.orchestration_metrics['consensus_success_rate'] = (
            self.orchestration_metrics['consensus_success_rate'] * 0.9 + 
            (1.0 if consensus_success else 0.0) * 0.1
        )
        
        # Add to episode tracking
        self.episode_rewards.append(reward.total_reward)
        
        # Train if enough experiences
        if len(self.experience_buffer) >= self.batch_size and self.training_step % self.update_frequency == 0:
            await self._train_orchestration_policy()
        
        self.training_step += 1
        
        logger.debug(f"Processed orchestration outcome for task {task_id}, reward: {reward.total_reward:.3f}")
    
    def _calculate_orchestration_reward(self, outcome: Dict[str, Any]) -> LLMOrchestrationReward:
        """Calculate reward signal from orchestration outcome"""
        
        # Quality score component
        quality_score = outcome.get('quality_score', 0.5)
        
        # Cost efficiency component
        total_cost = outcome.get('total_cost', 0.1)
        cost_efficiency = min(1.0, quality_score / max(total_cost, 0.01))
        
        # Response time component (faster is better)
        response_time = outcome.get('response_time', 10.0)
        response_time_reward = max(0.0, 1.0 - (response_time / 30.0))
        
        # Consensus coherence component
        consensus_score = outcome.get('consensus_score', 0.5)
        consensus_coherence = consensus_score
        
        # Provider utilization component (balanced usage is better)
        used_providers = len(outcome.get('selected_providers', []))
        provider_utilization = min(1.0, used_providers / 3.0)  # Optimal is 2-3 providers
        
        # User satisfaction component
        user_satisfaction = outcome.get('user_satisfaction', quality_score)
        
        reward = LLMOrchestrationReward(
            quality_score=quality_score,
            cost_efficiency=cost_efficiency,
            response_time=response_time_reward,
            consensus_coherence=consensus_coherence,
            provider_utilization=provider_utilization,
            user_satisfaction=user_satisfaction
        )
        
        reward.compute_total_reward(self.reward_weights)
        
        return reward
    
    async def _train_orchestration_policy(self) -> None:
        """Train the DRL orchestration policy network"""
        
        if len(self.experience_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = self.experience_buffer.sample(self.batch_size)
        
        states = torch.stack([exp['state'] for exp in batch])
        actions = torch.tensor([exp['action'] for exp in batch], dtype=torch.long)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32)
        
        # Forward pass
        provider_probs, strategy_probs, quality_threshold, cost_allocation, state_values = \
            self.policy_network(states)
        
        # Calculate advantages
        advantages = rewards - state_values.squeeze()
        
        # Multi-head losses
        
        # Strategy policy loss
        strategy_log_probs = torch.log(strategy_probs.gather(1, actions.unsqueeze(1))).squeeze()
        strategy_policy_loss = -(strategy_log_probs * advantages.detach()).mean()
        
        # Provider selection loss (simplified)
        provider_selection_loss = F.mse_loss(
            provider_probs.mean(dim=1), 
            torch.ones_like(provider_probs.mean(dim=1)) * 0.5
        )
        
        # Value loss
        value_loss = F.mse_loss(state_values.squeeze(), rewards)
        
        # Quality threshold loss (encourage adaptive thresholds)
        quality_target = rewards.unsqueeze(1)  # Use reward as quality target
        quality_loss = F.mse_loss(quality_threshold, quality_target)
        
        # Total loss
        total_loss = (strategy_policy_loss + 
                     0.1 * provider_selection_loss +
                     0.5 * value_loss +
                     0.1 * quality_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self._soft_update_target_network()
        
        # Update metrics
        self.orchestration_metrics['policy_loss'] = strategy_policy_loss.item()
        self.orchestration_metrics['value_loss'] = value_loss.item()
        self.orchestration_metrics['learning_episodes'] += 1
        
        logger.debug(f"Orchestration policy training - Loss: {total_loss.item():.4f}")
    
    def _soft_update_target_network(self) -> None:
        """Soft update of target network parameters"""
        for target_param, param in zip(self.target_network.parameters(), 
                                     self.policy_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def _estimate_task_complexity(self, task: Dict[str, Any]) -> float:
        """Estimate task complexity from task description"""
        prompt = task.get('prompt', '')
        
        complexity_indicators = [
            'analyze', 'compare', 'synthesize', 'evaluate', 'reason',
            'complex', 'detailed', 'comprehensive', 'multi-step'
        ]
        
        complexity_score = sum(1 for indicator in complexity_indicators 
                             if indicator in prompt.lower())
        
        return min(1.0, complexity_score / len(complexity_indicators))
    
    def _encode_task_type(self, task_type: str) -> List[float]:
        """Encode task type as embedding vector"""
        task_types = [
            'reasoning', 'analysis', 'generation', 'translation',
            'summarization', 'creative', 'technical', 'general'
        ]
        
        encoding = [0.0] * len(task_types)
        if task_type.lower() in task_types:
            idx = task_types.index(task_type.lower())
            encoding[idx] = 1.0
        else:
            encoding[-1] = 1.0  # Default to 'general'
        
        return encoding
    
    def _get_provider_specializations(self) -> List[List[float]]:
        """Get provider specialization vectors"""
        # Simplified specialization matrix
        specializations = {
            'anthropic': [0.9, 0.8, 0.7, 0.6, 0.8, 0.9],  # reasoning, analysis, etc.
            'openai': [0.8, 0.7, 0.9, 0.8, 0.7, 0.8],
            'deepseek': [0.9, 0.9, 0.6, 0.7, 0.9, 0.7],
            'bitnet': [0.6, 0.7, 0.8, 0.9, 0.6, 0.7],
            'mock': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        }
        
        return [specializations.get(provider, [0.5] * 6) for provider in self.provider_names]
    
    def _calculate_performance_trend(self) -> float:
        """Calculate recent performance trend"""
        if len(self.episode_rewards) < 10:
            return 0.5
        
        recent_rewards = list(self.episode_rewards)[-10:]
        early_avg = np.mean(recent_rewards[:5])
        late_avg = np.mean(recent_rewards[5:])
        
        # Normalize trend to 0-1 range
        trend = (late_avg - early_avg) / 2.0 + 0.5
        return max(0.0, min(1.0, trend))
    
    def _monitor_orchestration_integrity(self, orchestration_decision: Dict[str, Any],
                                       state: LLMOrchestrationState) -> Dict[str, Any]:
        """Monitor orchestration decision integrity"""
        try:
            violations = []
            
            # Check quality threshold bounds
            quality_threshold = orchestration_decision.get('quality_threshold', 0.5)
            if quality_threshold < 0.0 or quality_threshold > 1.0:
                violations.append(IntegrityViolation(
                    violation_type=ViolationType.INVALID_METRIC,
                    description=f"Quality threshold {quality_threshold} outside valid range [0,1]",
                    severity="HIGH",
                    context={"quality_threshold": quality_threshold}
                ))
            
            # Check if selected providers are available
            selected_providers = orchestration_decision.get('selected_providers', [])
            unavailable_providers = [
                provider for provider in selected_providers
                if provider not in self.provider_names or 
                state.provider_availability[self.provider_names.index(provider)] < 0.5
            ]
            
            if unavailable_providers:
                violations.append(IntegrityViolation(
                    violation_type=ViolationType.PERFORMANCE_CLAIM,
                    description=f"Selected unavailable providers: {unavailable_providers}",
                    severity="MEDIUM",
                    context={"unavailable_providers": unavailable_providers}
                ))
            
            # Apply corrections
            if violations:
                self.integrity_violations.extend(violations)
                
                if self.auto_correction_enabled:
                    # Clamp quality threshold
                    orchestration_decision['quality_threshold'] = max(0.0, min(1.0, quality_threshold))
                    
                    # Remove unavailable providers
                    available_providers = [
                        provider for provider in selected_providers
                        if provider in self.provider_names and 
                        state.provider_availability[self.provider_names.index(provider)] >= 0.5
                    ]
                    
                    if not available_providers:
                        # Fallback to first available provider
                        for i, provider in enumerate(self.provider_names):
                            if state.provider_availability[i] >= 0.5:
                                available_providers = [provider]
                                break
                    
                    orchestration_decision['selected_providers'] = available_providers
            
            return orchestration_decision
            
        except Exception as e:
            logger.error(f"Error in orchestration integrity monitoring: {e}")
            return orchestration_decision
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            **self.orchestration_metrics,
            'episode_rewards_mean': np.mean(list(self.episode_rewards)) if self.episode_rewards else 0.0,
            'episode_rewards_std': np.std(list(self.episode_rewards)) if self.episode_rewards else 0.0,
            'provider_performance_averages': {
                provider: np.mean(list(history)) if history else 0.5
                for provider, history in self.provider_performance.items()
            },
            'provider_cost_averages': {
                provider: np.mean(list(history)) if history else 0.1
                for provider, history in self.provider_costs.items()
            },
            'provider_speed_averages': {
                provider: np.mean(list(history)) if history else 1.0
                for provider, history in self.provider_speeds.items()
            },
            'integrity_violations': len(self.integrity_violations),
            'experience_buffer_size': len(self.experience_buffer)
        }
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model"""
        try:
            torch.save({
                'policy_network_state_dict': self.policy_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'orchestration_metrics': self.orchestration_metrics,
                'training_step': self.training_step,
                'provider_performance': dict(self.provider_performance),
                'provider_costs': dict(self.provider_costs),
                'provider_speeds': dict(self.provider_speeds)
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
            self.orchestration_metrics = checkpoint['orchestration_metrics']
            self.training_step = checkpoint['training_step']
            
            # Restore provider histories
            for provider, history in checkpoint.get('provider_performance', {}).items():
                self.provider_performance[provider] = deque(history, maxlen=100)
            for provider, history in checkpoint.get('provider_costs', {}).items():
                self.provider_costs[provider] = deque(history, maxlen=100)
            for provider, history in checkpoint.get('provider_speeds', {}).items():
                self.provider_speeds[provider] = deque(history, maxlen=100)
            
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False 
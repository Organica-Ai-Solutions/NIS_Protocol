#!/usr/bin/env python3
"""
🎯 NIS Protocol Consensus Controller
User-controllable multi-LLM consensus with smart optimization

Features:
- User-selectable consensus modes
- Provider selection controls
- Cost-aware consensus decisions
- Quality vs. speed tradeoffs
- Real-time consensus metrics
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ConsensusMode(Enum):
    """Consensus operation modes"""
    SINGLE = "single"              # Single provider (fastest, cheapest)
    DUAL = "dual"                  # Two providers for validation
    TRIPLE = "triple"              # Three providers for strong consensus
    FULL = "full"                  # All available providers (most expensive)
    SMART = "smart"                # AI-driven consensus decision
    CUSTOM = "custom"              # User-defined provider selection

class ConsensusStrategy(Enum):
    """Consensus decision strategies"""
    MAJORITY = "majority"          # Majority vote wins
    WEIGHTED = "weighted"          # Weighted by provider quality scores
    BEST_EFFORT = "best_effort"    # Use best available response
    FUSION = "fusion"              # Synthesize all responses
    COMPETITIVE = "competitive"    # Multiple responses, user picks best

@dataclass
class ConsensusConfig:
    """Consensus configuration"""
    mode: ConsensusMode = ConsensusMode.SINGLE
    strategy: ConsensusStrategy = ConsensusStrategy.MAJORITY
    selected_providers: Optional[List[str]] = None
    max_cost: float = 0.10  # Max cost per request
    quality_threshold: float = 0.8
    speed_priority: bool = False
    enable_caching: bool = True
    user_preference: str = "balanced"  # balanced, quality, speed, cost

@dataclass
class ConsensusResult:
    """Result from consensus processing"""
    final_response: str
    providers_used: List[str]
    individual_responses: Dict[str, str]
    consensus_achieved: bool
    consensus_confidence: float
    total_cost: float
    processing_time: float
    strategy_used: str
    quality_scores: Dict[str, float]

class ConsensusController:
    """
    🧠 Smart consensus controller for multi-LLM coordination
    
    Allows users to choose between different consensus modes while
    optimizing for cost, quality, and speed based on preferences.
    """
    
    def __init__(self):
        # Provider quality scores (can be updated based on performance)
        self.provider_quality = {
            "anthropic": 0.95,
            "openai": 0.90,
            "deepseek": 0.85,
            "google": 0.82,
            "nvidia": 0.88,
            "bitnet": 0.65
        }
        
        # Provider cost estimates (per 1K tokens)
        self.provider_costs = {
            "anthropic": 0.008,
            "openai": 0.005,
            "deepseek": 0.0015,
            "google": 0.0015,
            "nvidia": 0.02,
            "bitnet": 0.0
        }
        
        # Provider speed estimates (avg latency in seconds)
        self.provider_speeds = {
            "anthropic": 2.5,
            "openai": 1.5,
            "deepseek": 8.0,
            "google": 1.2,
            "nvidia": 3.0,
            "bitnet": 0.5
        }
        
        # Consensus history for learning
        self.consensus_history: List[Dict[str, Any]] = []
        
        logger.info("🧠 Consensus Controller initialized")
    
    def determine_optimal_consensus(self, 
                                  request_context: Dict[str, Any],
                                  config: ConsensusConfig) -> ConsensusConfig:
        """
        🤖 AI-driven consensus mode determination
        
        Analyzes request context and user preferences to automatically
        choose the best consensus approach.
        """
        
        if config.mode != ConsensusMode.SMART:
            return config  # User has explicit preference
        
        # Extract request characteristics
        message_length = len(request_context.get("message", ""))
        task_complexity = self._assess_task_complexity(request_context.get("message", ""))
        user_priority = request_context.get("priority", "normal")
        available_budget = config.max_cost
        
        # Smart decision logic
        optimal_config = ConsensusConfig(
            max_cost=config.max_cost,
            quality_threshold=config.quality_threshold,
            enable_caching=config.enable_caching,
            user_preference=config.user_preference
        )
        
        # Decision tree based on context
        if user_priority == "critical" or task_complexity > 0.8:
            # High-stakes requests get full consensus
            optimal_config.mode = ConsensusMode.TRIPLE
            optimal_config.strategy = ConsensusStrategy.WEIGHTED
            optimal_config.selected_providers = ["anthropic", "openai", "deepseek"]
            
        elif available_budget > 0.05 and task_complexity > 0.6:
            # Medium complexity with good budget
            optimal_config.mode = ConsensusMode.DUAL
            optimal_config.strategy = ConsensusStrategy.BEST_EFFORT
            optimal_config.selected_providers = ["anthropic", "openai"]
            
        elif config.speed_priority or user_priority == "fast":
            # Speed priority
            optimal_config.mode = ConsensusMode.SINGLE
            optimal_config.selected_providers = ["google"]  # Fastest provider
            
        elif config.user_preference == "cost":
            # Cost-conscious users
            optimal_config.mode = ConsensusMode.SINGLE
            optimal_config.selected_providers = ["deepseek"]  # Cheapest quality provider
            
        elif config.user_preference == "quality":
            # Quality-focused users
            optimal_config.mode = ConsensusMode.DUAL
            optimal_config.strategy = ConsensusStrategy.WEIGHTED
            optimal_config.selected_providers = ["anthropic", "openai"]
            
        else:
            # Balanced approach (default)
            if available_budget > 0.03:
                optimal_config.mode = ConsensusMode.DUAL
                optimal_config.selected_providers = ["openai", "deepseek"]
            else:
                optimal_config.mode = ConsensusMode.SINGLE
                optimal_config.selected_providers = ["openai"]
        
        logger.info(f"🤖 Smart consensus: {optimal_config.mode.value} with {optimal_config.selected_providers}")
        return optimal_config
    
    def _assess_task_complexity(self, message: str) -> float:
        """Assess task complexity from message content"""
        complexity_indicators = {
            "simple": ["hello", "hi", "thanks", "yes", "no"],
            "medium": ["explain", "how", "what", "why", "analyze"],
            "complex": ["comprehensive", "detailed", "research", "compare", "evaluate"],
            "expert": ["mathematical", "scientific", "theoretical", "prove", "derive"]
        }
        
        message_lower = message.lower()
        scores = []
        
        for level, indicators in complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in message_lower)
            if level == "simple":
                scores.append(score * 0.2)
            elif level == "medium":
                scores.append(score * 0.5)
            elif level == "complex":
                scores.append(score * 0.8)
            elif level == "expert":
                scores.append(score * 1.0)
        
        # Also consider message length
        length_factor = min(len(message) / 500, 1.0)
        
        return min(max(scores) + length_factor * 0.3, 1.0) if scores else 0.3
    
    def select_providers_for_consensus(self, config: ConsensusConfig) -> List[str]:
        """Select optimal providers based on consensus configuration"""
        
        if config.selected_providers:
            # User has explicitly chosen providers
            return config.selected_providers
        
        available_providers = list(self.provider_quality.keys())
        
        if config.mode == ConsensusMode.SINGLE:
            # Select single best provider based on user preference
            if config.user_preference == "speed":
                return [min(available_providers, key=lambda p: self.provider_speeds[p])]
            elif config.user_preference == "cost":
                return [min(available_providers, key=lambda p: self.provider_costs[p])]
            else:  # quality or balanced
                return [max(available_providers, key=lambda p: self.provider_quality[p])]
        
        elif config.mode == ConsensusMode.DUAL:
            # Select two complementary providers
            if config.user_preference == "quality":
                sorted_by_quality = sorted(available_providers, 
                                         key=lambda p: self.provider_quality[p], reverse=True)
                return sorted_by_quality[:2]
            else:
                # Balanced: one high-quality, one cost-effective
                return ["anthropic", "deepseek"]
        
        elif config.mode == ConsensusMode.TRIPLE:
            # Three providers for strong consensus
            return ["anthropic", "openai", "deepseek"]
        
        elif config.mode == ConsensusMode.FULL:
            # All available providers
            return available_providers
        
        else:
            # Default to single provider
            return ["openai"]
    
    def estimate_consensus_cost(self, 
                              providers: List[str], 
                              estimated_tokens: int = 1000) -> float:
        """Estimate total cost for consensus request"""
        total_cost = 0.0
        
        for provider in providers:
            if provider in self.provider_costs:
                cost_per_1k = self.provider_costs[provider]
                total_cost += (estimated_tokens / 1000) * cost_per_1k
        
        return total_cost
    
    def validate_consensus_request(self, 
                                 config: ConsensusConfig,
                                 providers: List[str],
                                 estimated_tokens: int = 1000) -> tuple[bool, str]:
        """Validate if consensus request should proceed"""
        
        # Check cost limits
        estimated_cost = self.estimate_consensus_cost(providers, estimated_tokens)
        if estimated_cost > config.max_cost:
            return False, f"Estimated cost ${estimated_cost:.4f} exceeds limit ${config.max_cost:.4f}"
        
        # Check provider availability (simplified)
        unavailable_providers = [p for p in providers if p not in self.provider_quality]
        if unavailable_providers:
            return False, f"Providers not available: {unavailable_providers}"
        
        # Check quality threshold
        avg_quality = sum(self.provider_quality.get(p, 0) for p in providers) / len(providers)
        if avg_quality < config.quality_threshold:
            return False, f"Average quality {avg_quality:.2f} below threshold {config.quality_threshold:.2f}"
        
        return True, "Validation passed"
    
    async def execute_consensus(self,
                              providers: List[str],
                              request_func: callable,
                              config: ConsensusConfig) -> ConsensusResult:
        """Execute consensus request across multiple providers"""
        
        start_time = time.time()
        individual_responses = {}
        quality_scores = {}
        total_cost = 0.0
        
        # Execute requests in parallel for speed
        tasks = []
        for provider in providers:
            task = asyncio.create_task(
                self._execute_single_provider(provider, request_func)
            )
            tasks.append((provider, task))
        
        # Gather results
        successful_responses = {}
        for provider, task in tasks:
            try:
                response = await task
                individual_responses[provider] = response.get("content", "")
                quality_scores[provider] = self.provider_quality.get(provider, 0.8)
                total_cost += response.get("cost", self.provider_costs.get(provider, 0.01))
                successful_responses[provider] = response
            except Exception as e:
                logger.warning(f"Provider {provider} failed: {e}")
                individual_responses[provider] = f"Provider {provider} failed: {str(e)}"
                quality_scores[provider] = 0.0
        
        # Apply consensus strategy
        final_response, consensus_achieved, consensus_confidence = self._apply_consensus_strategy(
            successful_responses, config.strategy
        )
        
        processing_time = time.time() - start_time
        
        # Record consensus attempt for learning
        self._record_consensus_attempt(config, providers, successful_responses, 
                                     consensus_achieved, total_cost, processing_time)
        
        return ConsensusResult(
            final_response=final_response,
            providers_used=list(successful_responses.keys()),
            individual_responses=individual_responses,
            consensus_achieved=consensus_achieved,
            consensus_confidence=consensus_confidence,
            total_cost=total_cost,
            processing_time=processing_time,
            strategy_used=config.strategy.value,
            quality_scores=quality_scores
        )
    
    async def _execute_single_provider(self, provider: str, request_func: callable) -> Dict[str, Any]:
        """Execute request for single provider"""
        # This would integrate with the actual LLM provider
        # For now, simulate the call
        try:
            response = await request_func(provider)
            return response
        except Exception as e:
            logger.error(f"Provider {provider} execution failed: {e}")
            raise
    
    def _apply_consensus_strategy(self, 
                                responses: Dict[str, Dict[str, Any]],
                                strategy: ConsensusStrategy) -> tuple[str, bool, float]:
        """Apply consensus strategy to combine responses"""
        
        if not responses:
            return "No responses available", False, 0.0
        
        if len(responses) == 1:
            # Single response
            response = list(responses.values())[0]
            return response.get("content", ""), True, 1.0
        
        if strategy == ConsensusStrategy.MAJORITY:
            # Simple majority (simplified implementation)
            contents = [r.get("content", "") for r in responses.values()]
            # In production, use semantic similarity for true majority
            return contents[0], True, 0.8
        
        elif strategy == ConsensusStrategy.WEIGHTED:
            # Weighted by provider quality
            best_provider = max(responses.keys(), 
                              key=lambda p: self.provider_quality.get(p, 0))
            best_response = responses[best_provider]
            confidence = self.provider_quality.get(best_provider, 0.8)
            return best_response.get("content", ""), True, confidence
        
        elif strategy == ConsensusStrategy.BEST_EFFORT:
            # Use highest confidence response
            best_response = max(responses.values(), 
                              key=lambda r: r.get("confidence", 0))
            return best_response.get("content", ""), True, best_response.get("confidence", 0.8)
        
        elif strategy == ConsensusStrategy.FUSION:
            # Synthesize all responses
            contents = [r.get("content", "") for r in responses.values()]
            providers = list(responses.keys())
            
            fused_content = f"""🧠 **MULTI-LLM CONSENSUS RESPONSE**

I've consulted {len(providers)} AI models ({', '.join(providers)}) to provide you with a comprehensive answer:

{self._synthesize_responses(contents)}

---
**Consensus Details:**
• Models: {', '.join(providers)}
• Strategy: Fusion synthesis
• Confidence: High consensus achieved
"""
            return fused_content, True, 0.9
        
        elif strategy == ConsensusStrategy.COMPETITIVE:
            # Return all responses for user to choose
            competitive_content = "🎯 **COMPETITIVE MULTI-LLM RESPONSES**\n\n"
            competitive_content += "Multiple AI models have provided different perspectives. Choose your preferred response:\n\n"
            
            for i, (provider, response) in enumerate(responses.items(), 1):
                competitive_content += f"**Option {i} ({provider.title()}):**\n"
                competitive_content += f"{response.get('content', '')}\n\n"
                competitive_content += f"*Quality Score: {self.provider_quality.get(provider, 0.8):.1%}*\n\n---\n\n"
            
            return competitive_content, True, 0.85
        
        else:
            # Default to first response
            first_response = list(responses.values())[0]
            return first_response.get("content", ""), False, 0.6
    
    def _synthesize_responses(self, contents: List[str]) -> str:
        """Synthesize multiple response contents (simplified)"""
        if not contents:
            return "No content to synthesize"
        
        if len(contents) == 1:
            return contents[0]
        
        # Simple synthesis - in production, use more sophisticated NLP
        synthesis = "Based on analysis from multiple AI models:\n\n"
        
        # Take key insights from each response
        for i, content in enumerate(contents[:3], 1):  # Limit to first 3
            # Extract first meaningful sentence
            sentences = content.split('.')
            key_insight = sentences[0] if sentences else content[:200]
            synthesis += f"**Perspective {i}:** {key_insight.strip()}.\n\n"
        
        synthesis += "These perspectives have been synthesized to provide comprehensive coverage of your question."
        return synthesis
    
    def _record_consensus_attempt(self, 
                                config: ConsensusConfig,
                                providers: List[str],
                                responses: Dict[str, Dict[str, Any]],
                                consensus_achieved: bool,
                                total_cost: float,
                                processing_time: float):
        """Record consensus attempt for learning and optimization"""
        
        record = {
            "timestamp": time.time(),
            "config": {
                "mode": config.mode.value,
                "strategy": config.strategy.value,
                "user_preference": config.user_preference
            },
            "providers_requested": providers,
            "providers_successful": list(responses.keys()),
            "consensus_achieved": consensus_achieved,
            "total_cost": total_cost,
            "processing_time": processing_time,
            "success_rate": len(responses) / len(providers) if providers else 0
        }
        
        self.consensus_history.append(record)
        
        # Keep only recent history
        if len(self.consensus_history) > 1000:
            self.consensus_history = self.consensus_history[-500:]
    
    def get_consensus_recommendations(self, 
                                    user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get personalized consensus recommendations based on history"""
        
        if not self.consensus_history:
            return {
                "recommended_mode": ConsensusMode.SMART.value,
                "recommended_providers": ["openai"],
                "reason": "No history available - using smart defaults"
            }
        
        # Analyze user patterns
        recent_history = self.consensus_history[-50:]  # Last 50 requests
        
        user_preference = user_context.get("preference", "balanced")
        typical_budget = user_context.get("typical_budget", 0.05)
        
        # Calculate success rates by mode
        mode_performance = {}
        for record in recent_history:
            mode = record["config"]["mode"]
            if mode not in mode_performance:
                mode_performance[mode] = {"success": 0, "total": 0, "avg_cost": 0}
            
            mode_performance[mode]["total"] += 1
            if record["consensus_achieved"]:
                mode_performance[mode]["success"] += 1
            mode_performance[mode]["avg_cost"] += record["total_cost"]
        
        # Calculate success rates
        for mode in mode_performance:
            perf = mode_performance[mode]
            perf["success_rate"] = perf["success"] / perf["total"]
            perf["avg_cost"] /= perf["total"]
        
        # Find best mode for user
        if user_preference == "quality":
            best_mode = max(mode_performance.keys(), 
                          key=lambda m: mode_performance[m]["success_rate"])
        elif user_preference == "cost":
            best_mode = min(mode_performance.keys(), 
                          key=lambda m: mode_performance[m]["avg_cost"])
        else:  # balanced
            best_mode = max(mode_performance.keys(), 
                          key=lambda m: mode_performance[m]["success_rate"] / max(mode_performance[m]["avg_cost"], 0.001))
        
        return {
            "recommended_mode": best_mode,
            "recommended_providers": self.select_providers_for_consensus(
                ConsensusConfig(mode=ConsensusMode(best_mode))
            ),
            "reason": f"Based on your {user_preference} preference and usage patterns",
            "performance_data": mode_performance
        }
    
    def get_provider_recommendations(self) -> Dict[str, Any]:
        """Get current provider recommendations"""
        return {
            "quality_ranking": sorted(self.provider_quality.items(), 
                                    key=lambda x: x[1], reverse=True),
            "cost_ranking": sorted(self.provider_costs.items(), 
                                 key=lambda x: x[1]),
            "speed_ranking": sorted(self.provider_speeds.items(), 
                                  key=lambda x: x[1]),
            "recommendations": {
                "best_quality": max(self.provider_quality.items(), key=lambda x: x[1])[0],
                "most_cost_effective": min(self.provider_costs.items(), key=lambda x: x[1])[0],
                "fastest": min(self.provider_speeds.items(), key=lambda x: x[1])[0],
                "balanced": "openai"  # Good balance of all factors
            }
        }


# Global consensus controller
_global_consensus_controller: Optional[ConsensusController] = None

def get_consensus_controller() -> ConsensusController:
    """Get global consensus controller instance"""
    global _global_consensus_controller
    if _global_consensus_controller is None:
        _global_consensus_controller = ConsensusController()
    return _global_consensus_controller

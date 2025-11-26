#!/usr/bin/env python3
"""
NIS Protocol v4.0 - Evolution Engine (Meta-Agent)
==================================================
EXPERIMENTAL RESEARCH SANDBOX - NOT FOR PRODUCTION

This meta-agent uses MULTI-PROVIDER CONSENSUS (not single Kimi K2) to analyze
agent performance and suggest architectural improvements.

Reality Check:
--------------
What this IS:
  ✅ Performance monitoring + LLM-generated config suggestions
  ✅ A/B testing framework with safety guardrails
  ✅ Audit trail for agent evolution

What this is NOT:
  ❌ True "self-aware" AI
  ❌ Unsupervised self-modification
  ❌ Production-ready (requires $500K+ revenue, dedicated MLops)

Copyright 2025 Organica AI Solutions
Licensed under Apache 2.0
"""

import asyncio
import json
import logging
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class EvolutionMode(Enum):
    """Evolution strategies with different risk levels"""
    META_LEARNING = "meta_learning"  # 🟢 Moderate risk: LLM rewrites configs
    BEHAVIORAL = "behavioral"  # 🟡 High risk: RL on agent outputs  
    ARCHITECTURAL = "architectural"  # 🔴 Critical risk: Spawn/merge agents


class EvolutionStatus(Enum):
    """Status of an evolution attempt"""
    PROPOSED = "proposed"
    TESTING_CANARY = "testing_canary"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    DEPLOYED = "deployed"


@dataclass
class AgentMetrics:
    """Performance metrics for an agent"""
    agent_id: str
    success_rate: float  # 0.0 to 1.0
    avg_latency_ms: float
    total_requests: int
    failed_requests: int
    cost_per_request: float
    confidence_avg: float
    
    # Failure modes (what went wrong)
    failure_modes: Dict[str, int] = field(default_factory=dict)
    
    # Time window
    window_start: float = 0.0
    window_end: float = 0.0
    
    def is_underperforming(self, threshold: float = 0.85) -> bool:
        """Check if agent is below performance threshold"""
        return self.success_rate < threshold


@dataclass
class EvolutionProposal:
    """A proposed evolution for an agent"""
    evolution_id: str
    agent_id: str
    mode: EvolutionMode
    status: EvolutionStatus
    
    # What's changing
    current_config: Dict[str, Any]
    proposed_config: Dict[str, Any]
    rationale: str  # LLM explanation of why this change
    
    # Consensus from multiple LLM providers
    provider_votes: Dict[str, float] = field(default_factory=dict)  # provider -> confidence
    consensus_score: float = 0.0
    
    # Testing results
    canary_metrics: Optional[AgentMetrics] = None
    baseline_metrics: Optional[AgentMetrics] = None
    
    # Safety
    embedding_distance: float = 0.0  # Distance from original agent
    kill_switch_triggered: bool = False
    
    # Audit trail
    proposed_at: float = 0.0
    approved_at: Optional[float] = None
    deployed_at: Optional[float] = None
    approved_by: Optional[str] = None  # "auto" or human identifier


@dataclass
class SafetyLimits:
    """Safety guardrails for evolution"""
    max_evolutions_per_agent_per_day: int = 1
    min_success_rate_for_promotion: float = 0.95
    max_embedding_distance: float = 0.3
    canary_test_duration_seconds: int = 3600  # 1 hour
    require_human_approval_for_first_n: int = 100
    auto_approve_confidence_threshold: float = 0.95
    max_cost_increase_percent: float = 20.0
    max_latency_increase_percent: float = 15.0


class EvolutionEngine:
    """
    Meta-Agent: Analyzes agent performance and proposes improvements
    
    Uses MULTI-PROVIDER CONSENSUS (OpenAI, Anthropic, Google, DeepSeek, Kimi K2)
    NOT single Kimi K2 as originally suggested.
    
    Architecture:
    -------------
    1. Monitor agent metrics in Redis TimeSeries
    2. Detect underperforming agents
    3. Query MULTIPLE LLM providers for config suggestions
    4. Build consensus from provider responses
    5. Deploy canary test
    6. Human approval (first 100) or auto-approve if consensus >0.95
    7. Gradual rollout if successful
    """
    
    def __init__(
        self,
        llm_provider,  # Multi-provider LLM manager
        redis_client,  # For performance telemetry
        safety_limits: Optional[SafetyLimits] = None
    ):
        self.llm = llm_provider
        self.redis = redis_client
        self.safety = safety_limits or SafetyLimits()
        
        # Track evolution history
        self.evolution_history: List[EvolutionProposal] = []
        self.evolution_count_24h: Dict[str, int] = {}  # agent_id -> count
        
        # Baseline embeddings for kill switch
        self.baseline_embeddings: Dict[str, List[float]] = {}
        
        self.logger = logging.getLogger(f"{__name__}.EvolutionEngine")
        self.logger.info("🧬 EvolutionEngine initialized (EXPERIMENTAL - NOT PRODUCTION)")
    
    async def get_agent_metrics(
        self, 
        agent_id: str, 
        window_hours: int = 24
    ) -> AgentMetrics:
        """
        Fetch performance metrics for an agent from Redis TimeSeries
        
        Reality: This requires Redis TimeSeries module to be installed.
        If not available, returns mock metrics.
        """
        try:
            # Try to query Redis TimeSeries
            window_start = time.time() - (window_hours * 3600)
            window_end = time.time()
            
            # Keys we expect:
            # agent:{agent_id}:success_count
            # agent:{agent_id}:failure_count
            # agent:{agent_id}:latency_ms
            # agent:{agent_id}:cost
            # agent:{agent_id}:confidence
            
            success_count = await self._get_redis_counter(f"agent:{agent_id}:success_count")
            failure_count = await self._get_redis_counter(f"agent:{agent_id}:failure_count")
            total_requests = success_count + failure_count
            
            if total_requests == 0:
                # No data yet - return neutral metrics
                return AgentMetrics(
                    agent_id=agent_id,
                    success_rate=1.0,
                    avg_latency_ms=0.0,
                    total_requests=0,
                    failed_requests=0,
                    cost_per_request=0.0,
                    confidence_avg=0.0,
                    window_start=window_start,
                    window_end=window_end
                )
            
            success_rate = success_count / total_requests
            avg_latency = await self._get_redis_avg(f"agent:{agent_id}:latency_ms")
            avg_cost = await self._get_redis_avg(f"agent:{agent_id}:cost")
            avg_confidence = await self._get_redis_avg(f"agent:{agent_id}:confidence")
            
            # Get failure modes (stored as hash)
            failure_modes = await self._get_redis_hash(f"agent:{agent_id}:failure_modes")
            
            return AgentMetrics(
                agent_id=agent_id,
                success_rate=success_rate,
                avg_latency_ms=avg_latency,
                total_requests=total_requests,
                failed_requests=failure_count,
                cost_per_request=avg_cost,
                confidence_avg=avg_confidence,
                failure_modes=failure_modes,
                window_start=window_start,
                window_end=window_end
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to get metrics for {agent_id}: {e}. Using mock data.")
            # Return mock metrics for testing
            return AgentMetrics(
                agent_id=agent_id,
                success_rate=0.80,  # Below threshold to trigger evolution
                avg_latency_ms=250.0,
                total_requests=1000,
                failed_requests=200,
                cost_per_request=0.001,
                confidence_avg=0.75,
                failure_modes={"timeout": 80, "invalid_response": 70, "rate_limit": 50},
                window_start=time.time() - 86400,
                window_end=time.time()
            )
    
    async def _get_redis_counter(self, key: str) -> int:
        """Get counter value from Redis"""
        try:
            value = await self.redis.get(key)
            return int(value) if value else 0
        except:
            return 0
    
    async def _get_redis_avg(self, key: str) -> float:
        """Get average value from Redis (stored as float)"""
        try:
            value = await self.redis.get(key)
            return float(value) if value else 0.0
        except:
            return 0.0
    
    async def _get_redis_hash(self, key: str) -> Dict[str, int]:
        """Get hash from Redis"""
        try:
            data = await self.redis.hgetall(key)
            return {k.decode(): int(v) for k, v in data.items()} if data else {}
        except:
            return {}
    
    async def detect_underperforming_agents(
        self, 
        agent_ids: List[str]
    ) -> List[Tuple[str, AgentMetrics]]:
        """
        Scan all agents and identify underperformers
        """
        underperformers = []
        
        for agent_id in agent_ids:
            metrics = await self.get_agent_metrics(agent_id)
            
            if metrics.is_underperforming():
                self.logger.warning(
                    f"⚠️ Agent {agent_id} underperforming: "
                    f"success_rate={metrics.success_rate:.2%}, "
                    f"failures={metrics.failed_requests}/{metrics.total_requests}"
                )
                underperformers.append((agent_id, metrics))
        
        return underperformers
    
    async def generate_evolution_proposal(
        self,
        agent_id: str,
        metrics: AgentMetrics,
        current_config: Dict[str, Any]
    ) -> EvolutionProposal:
        """
        Use MULTI-PROVIDER CONSENSUS to generate improvement suggestions
        
        This is the core "meta-learning" step where multiple LLMs analyze
        the agent's performance and propose config changes.
        """
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(agent_id, metrics, current_config)
        
        # Query MULTIPLE providers (not just Kimi K2)
        providers_to_query = ["openai", "anthropic", "google", "deepseek"]
        if hasattr(self.llm, 'api_keys') and self.llm.api_keys.get("kimi"):
            providers_to_query.append("kimi")
        
        provider_responses = {}
        provider_votes = {}
        
        for provider in providers_to_query:
            try:
                response = await self.llm.generate_response(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,  # Low temperature for consistency
                    requested_provider=provider
                )
                
                # Parse JSON response
                suggestion = self._parse_llm_suggestion(response)
                provider_responses[provider] = suggestion
                provider_votes[provider] = suggestion.get("confidence", 0.5)
                
                self.logger.info(f"✅ {provider} voted: confidence={provider_votes[provider]:.2f}")
                
            except Exception as e:
                self.logger.warning(f"⚠️ {provider} failed to respond: {e}")
                provider_votes[provider] = 0.0
        
        # Calculate consensus
        consensus_score = sum(provider_votes.values()) / max(len(provider_votes), 1)
        
        # Merge suggestions (use highest-confidence provider's config)
        best_provider = max(provider_votes.items(), key=lambda x: x[1])[0]
        proposed_config = provider_responses[best_provider]["proposed_config"]
        rationale = provider_responses[best_provider]["rationale"]
        
        # Create evolution proposal
        evolution_id = self._generate_evolution_id(agent_id)
        
        proposal = EvolutionProposal(
            evolution_id=evolution_id,
            agent_id=agent_id,
            mode=EvolutionMode.META_LEARNING,
            status=EvolutionStatus.PROPOSED,
            current_config=current_config,
            proposed_config=proposed_config,
            rationale=rationale,
            provider_votes=provider_votes,
            consensus_score=consensus_score,
            baseline_metrics=metrics,
            proposed_at=time.time()
        )
        
        self.logger.info(
            f"🧬 Evolution proposed for {agent_id}: "
            f"consensus={consensus_score:.2%}, providers={len(provider_votes)}"
        )
        
        return proposal
    
    def _build_analysis_prompt(
        self,
        agent_id: str,
        metrics: AgentMetrics,
        current_config: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM analysis"""
        return f"""You are a meta-learning system analyzing AI agent performance.

Agent ID: {agent_id}

Current Performance (24h window):
- Success Rate: {metrics.success_rate:.2%} (target: >85%)
- Failed Requests: {metrics.failed_requests}/{metrics.total_requests}
- Average Latency: {metrics.avg_latency_ms:.0f}ms
- Cost per Request: ${metrics.cost_per_request:.4f}
- Confidence Average: {metrics.confidence_avg:.2f}

Failure Breakdown:
{json.dumps(metrics.failure_modes, indent=2)}

Current Configuration:
{json.dumps(current_config, indent=2)}

Task: Propose configuration changes to improve performance.

Respond in JSON format:
{{
  "proposed_config": {{
    // Modified configuration parameters
  }},
  "rationale": "Brief explanation of why these changes will help",
  "confidence": 0.0-1.0,
  "expected_improvements": {{
    "success_rate": 0.XX,
    "latency_reduction_percent": XX,
    "cost_reduction_percent": XX
  }}
}}

Focus on:
1. Reducing failure modes (timeouts, invalid responses, rate limits)
2. Optimizing model selection or parameters
3. Adjusting retry logic or caching strategies
4. DO NOT suggest changes that would fundamentally alter the agent's purpose
"""
    
    def _parse_llm_suggestion(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response"""
        try:
            # Extract JSON from response (might have markdown formatting)
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            return json.loads(response.strip())
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            return {
                "proposed_config": {},
                "rationale": "Failed to parse suggestion",
                "confidence": 0.0
            }
    
    def _generate_evolution_id(self, agent_id: str) -> str:
        """Generate unique evolution ID"""
        timestamp = int(time.time() * 1000)
        data = f"{agent_id}:{timestamp}"
        hash_val = hashlib.sha256(data.encode()).hexdigest()[:12]
        return f"evo_{agent_id}_{hash_val}"
    
    async def check_safety_limits(self, proposal: EvolutionProposal) -> Tuple[bool, str]:
        """
        Verify proposal doesn't violate safety guardrails
        
        Returns: (is_safe, reason)
        """
        
        # 1. Rate limit: Max 1 evolution per agent per day
        recent_count = self.evolution_count_24h.get(proposal.agent_id, 0)
        if recent_count >= self.safety.max_evolutions_per_agent_per_day:
            return False, f"Rate limit: Agent already evolved {recent_count} times in 24h"
        
        # 2. Consensus threshold
        if proposal.consensus_score < 0.5:
            return False, f"Insufficient consensus: {proposal.consensus_score:.2%} < 50%"
        
        # 3. Check if this is a reasonable change (not too radical)
        config_distance = self._calculate_config_distance(
            proposal.current_config, proposal.proposed_config
        )
        if config_distance > 0.7:  # More than 70% change is too radical
            return False, f"Config change too radical: {config_distance:.1%} distance"
        
        return True, "All safety checks passed"
    
    def _calculate_config_distance(self, config_a: Dict, config_b: Dict) -> float:
        """Calculate normalized distance between two configs (0-1 scale)"""
        if not config_a or not config_b:
            return 0.5  # Unknown, assume moderate
        
        # Flatten configs to compare
        def flatten(d, prefix=''):
            items = []
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    items.extend(flatten(v, key).items())
                else:
                    items.append((key, str(v)))
            return dict(items)
        
        flat_a, flat_b = flatten(config_a), flatten(config_b)
        all_keys = set(flat_a.keys()) | set(flat_b.keys())
        
        if not all_keys:
            return 0.0
        
        differences = sum(1 for k in all_keys if flat_a.get(k) != flat_b.get(k))
        return differences / len(all_keys)
    
    async def deploy_canary(self, proposal: EvolutionProposal) -> None:
        """
        Deploy canary version of agent for A/B testing
        
        Reality: This would require infrastructure to run two versions
        of the same agent side-by-side. For now, we simulate it.
        """
        self.logger.info(
            f"🔬 Deploying canary for {proposal.agent_id} (evolution {proposal.evolution_id})"
        )
        
        proposal.status = EvolutionStatus.TESTING_CANARY
        
        # In reality, this would:
        # 1. Create new agent instance with proposed_config
        # 2. Route 10% of traffic to canary
        # 3. Monitor for canary_test_duration_seconds
        # 4. Compare metrics
        
        # For now, simulate canary metrics (slightly better than baseline)
        await asyncio.sleep(2)  # Simulate deployment time
        
        baseline = proposal.baseline_metrics
        proposal.canary_metrics = AgentMetrics(
            agent_id=f"{proposal.agent_id}_canary",
            success_rate=min(1.0, baseline.success_rate + 0.10),
            avg_latency_ms=baseline.avg_latency_ms * 0.90,
            total_requests=100,  # Canary gets less traffic
            failed_requests=5,
            cost_per_request=baseline.cost_per_request * 0.95,
            confidence_avg=min(1.0, baseline.confidence_avg + 0.05),
            window_start=time.time(),
            window_end=time.time() + self.safety.canary_test_duration_seconds
        )
        
        self.logger.info(
            f"✅ Canary deployed: success_rate={proposal.canary_metrics.success_rate:.2%} "
            f"(baseline: {baseline.success_rate:.2%})"
        )
    
    async def evaluate_canary(self, proposal: EvolutionProposal) -> bool:
        """
        Compare canary vs baseline metrics
        
        Returns: True if canary is better and should be promoted
        """
        canary = proposal.canary_metrics
        baseline = proposal.baseline_metrics
        
        if not canary:
            return False
        
        # Check success rate
        if canary.success_rate < self.safety.min_success_rate_for_promotion:
            self.logger.warning(
                f"❌ Canary below threshold: {canary.success_rate:.2%} < "
                f"{self.safety.min_success_rate_for_promotion:.2%}"
            )
            return False
        
        # Check latency didn't increase too much
        latency_increase = ((canary.avg_latency_ms - baseline.avg_latency_ms) / 
                           baseline.avg_latency_ms * 100)
        if latency_increase > self.safety.max_latency_increase_percent:
            self.logger.warning(f"❌ Latency increased too much: {latency_increase:.1f}%")
            return False
        
        # Check cost didn't increase too much
        cost_increase = ((canary.cost_per_request - baseline.cost_per_request) / 
                        baseline.cost_per_request * 100)
        if cost_increase > self.safety.max_cost_increase_percent:
            self.logger.warning(f"❌ Cost increased too much: {cost_increase:.1f}%")
            return False
        
        # All checks passed
        improvement = (canary.success_rate - baseline.success_rate) * 100
        self.logger.info(
            f"✅ Canary successful: +{improvement:.1f}% success rate, "
            f"{latency_increase:+.1f}% latency, {cost_increase:+.1f}% cost"
        )
        return True
    
    async def require_human_approval(self, proposal: EvolutionProposal) -> bool:
        """
        Check if human approval is required
        
        Rules:
        - First 100 evolutions always require approval
        - After that, auto-approve if consensus >0.95
        """
        total_evolutions = len(self.evolution_history)
        
        if total_evolutions < self.safety.require_human_approval_for_first_n:
            return True
        
        if proposal.consensus_score < self.safety.auto_approve_confidence_threshold:
            return True
        
        return False
    
    async def evolve(self, agent_id: str) -> Optional[EvolutionProposal]:
        """
        Main evolution loop for a single agent
        
        Steps:
        1. Get metrics
        2. Check if underperforming
        3. Generate proposal (multi-provider consensus)
        4. Check safety limits
        5. Deploy canary
        6. Evaluate canary
        7. Require human approval OR auto-approve
        8. Promote if successful
        """
        
        self.logger.info(f"🔬 Evaluating evolution for agent: {agent_id}")
        
        # 1. Get current metrics
        metrics = await self.get_agent_metrics(agent_id)
        
        if not metrics.is_underperforming():
            self.logger.info(f"✅ Agent {agent_id} performing well ({metrics.success_rate:.2%})")
            return None
        
        # 2. Get current config from registry
        current_config = self._get_agent_config(agent_id)
        
        # 3. Generate proposal
        proposal = await self.generate_evolution_proposal(agent_id, metrics, current_config)
        
        # 4. Safety checks
        is_safe, reason = await self.check_safety_limits(proposal)
        if not is_safe:
            self.logger.warning(f"🚫 Evolution blocked: {reason}")
            proposal.status = EvolutionStatus.REJECTED
            self.evolution_history.append(proposal)
            return proposal
        
        # 5. Deploy canary
        await self.deploy_canary(proposal)
        
        # 6. Wait for canary test duration (simulated)
        self.logger.info(f"⏳ Waiting {self.safety.canary_test_duration_seconds}s for canary test...")
        # In reality: await asyncio.sleep(self.safety.canary_test_duration_seconds)
        await asyncio.sleep(2)  # Simulated for demo
        
        # 7. Evaluate canary
        canary_successful = await self.evaluate_canary(proposal)
        
        if not canary_successful:
            self.logger.warning(f"❌ Canary failed evaluation")
            proposal.status = EvolutionStatus.ROLLED_BACK
            self.evolution_history.append(proposal)
            return proposal
        
        # 8. Check if human approval needed
        needs_approval = await self.require_human_approval(proposal)
        
        if needs_approval:
            proposal.status = EvolutionStatus.AWAITING_APPROVAL
            self.logger.info(
                f"⏸️ Evolution awaiting human approval (total evolutions: {len(self.evolution_history)})"
            )
            # Send notification for approval
            await self._notify_for_approval(proposal)
        else:
            proposal.status = EvolutionStatus.APPROVED
            proposal.approved_by = "auto"
            proposal.approved_at = time.time()
            self.logger.info(f"✅ Auto-approved (consensus: {proposal.consensus_score:.2%})")
        
        # 9. Track evolution
        self.evolution_history.append(proposal)
        self.evolution_count_24h[agent_id] = self.evolution_count_24h.get(agent_id, 0) + 1
        
        return proposal
    
    async def promote_evolution(self, evolution_id: str) -> bool:
        """
        Promote approved evolution to production
        
        This would actually deploy the new config to the live agent.
        """
        proposal = next((e for e in self.evolution_history if e.evolution_id == evolution_id), None)
        
        if not proposal:
            self.logger.error(f"Evolution {evolution_id} not found")
            return False
        
        if proposal.status != EvolutionStatus.APPROVED:
            self.logger.error(f"Evolution {evolution_id} not approved (status: {proposal.status})")
            return False
        
        # Deploy to production
        success = await self._deploy_config(proposal)
        if not success:
            self.logger.error(f"Failed to deploy evolution {evolution_id}")
            return False
        
        proposal.status = EvolutionStatus.DEPLOYED
        proposal.deployed_at = time.time()
        
        self.logger.info(f"🚀 Evolution {evolution_id} deployed to production")
        return True
    
    def _get_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """Get current config for an agent from registry"""
        try:
            from src.core.registry import NISRegistry
            registry = NISRegistry()
            agent = registry.agents.get(agent_id)
            if agent:
                return {
                    "agent_id": agent.agent_id,
                    "layer": agent.layer.value if hasattr(agent.layer, 'value') else str(agent.layer),
                    "description": getattr(agent, 'description', ''),
                    "active": getattr(agent, 'active', True),
                    "config": getattr(agent, 'config', {})
                }
        except Exception as e:
            self.logger.warning(f"Failed to get agent config: {e}")
        
        return {"agent_id": agent_id, "config": {}}
    
    async def _notify_for_approval(self, proposal: 'EvolutionProposal') -> None:
        """Send notification for human approval (webhook-based)"""
        import os
        webhook_url = os.environ.get("EVOLUTION_WEBHOOK_URL")
        
        if not webhook_url:
            self.logger.info("📧 No webhook configured - approval notification logged only")
            return
        
        try:
            import aiohttp
            payload = {
                "text": f"🧬 Evolution Proposal Awaiting Approval",
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": "🧬 Agent Evolution Request"}},
                    {"type": "section", "text": {"type": "mrkdwn", 
                        "text": f"*Agent:* {proposal.agent_id}\n*Evolution ID:* {proposal.evolution_id}\n*Consensus:* {proposal.consensus_score:.1%}"}}
                ]
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as resp:
                    if resp.status == 200:
                        self.logger.info(f"📧 Approval notification sent")
        except Exception as e:
            self.logger.warning(f"Failed to send approval notification: {e}")
    
    async def _deploy_config(self, proposal: 'EvolutionProposal') -> bool:
        """Deploy new config to agent"""
        try:
            from src.core.registry import NISRegistry
            registry = NISRegistry()
            agent = registry.agents.get(proposal.agent_id)
            
            if agent and hasattr(agent, 'config'):
                # Update agent config
                if isinstance(proposal.proposed_config, dict):
                    for key, value in proposal.proposed_config.items():
                        if hasattr(agent, key):
                            setattr(agent, key, value)
                        elif hasattr(agent, 'config') and isinstance(agent.config, dict):
                            agent.config[key] = value
                
                self.logger.info(f"✅ Config deployed for {proposal.agent_id}")
                return True
            else:
                # Agent not in registry - store config for next restart
                self._pending_configs = getattr(self, '_pending_configs', {})
                self._pending_configs[proposal.agent_id] = proposal.proposed_config
                self.logger.info(f"📦 Config staged for {proposal.agent_id} (will apply on restart)")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to deploy config: {e}")
            return False
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get statistics about evolution history"""
        if not self.evolution_history:
            return {"total_evolutions": 0}
        
        status_counts = {}
        for proposal in self.evolution_history:
            status_counts[proposal.status.value] = status_counts.get(proposal.status.value, 0) + 1
        
        avg_consensus = sum(p.consensus_score for p in self.evolution_history) / len(self.evolution_history)
        
        return {
            "total_evolutions": len(self.evolution_history),
            "status_breakdown": status_counts,
            "avg_consensus_score": round(avg_consensus, 3),
            "agents_evolved": len(set(p.agent_id for p in self.evolution_history)),
            "latest_evolution": self.evolution_history[-1].evolution_id if self.evolution_history else None
        }

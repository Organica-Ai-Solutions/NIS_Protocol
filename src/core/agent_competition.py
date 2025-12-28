#!/usr/bin/env python3
"""
Agent Competition System for NIS Protocol
Runs multiple agents with different models/strategies, judge picks winner

Copyright 2025 Organica AI Solutions
Licensed under Apache License 2.0
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CompetitorResult:
    """Result from competing agent."""
    competitor_id: str
    provider: str
    result: Any
    execution_time: float
    success: bool
    quality_score: float = 0.0


class AgentCompetitionSystem:
    """
    Run multiple agents in competition, judge picks best result.
    
    Technique: Diversity reduces risk of bad outputs
    - Different LLM providers (Anthropic, OpenAI, Google)
    - Different prompts/strategies
    - Judge evaluates quality
    - Best result wins
    
    Honest Assessment:
    - Real parallel execution across providers
    - Simple judge (length + success heuristic, not LLM judge)
    - Real diversity benefits
    - 70% real - actual competition, simple judging
    """
    
    def __init__(self, multi_provider_strategy):
        """
        Initialize agent competition system.
        
        Args:
            multi_provider_strategy: Multi-provider strategy instance
        """
        self.multi_provider = multi_provider_strategy
        self.stats = {
            "total_competitions": 0,
            "wins_by_provider": {},
            "average_quality_improvement": 0.0
        }
        
        logger.info("🏆 Agent competition system initialized")
    
    async def compete(
        self,
        execute_func,
        providers: Optional[List[str]] = None,
        num_competitors: int = 3,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run agent competition.
        
        Args:
            execute_func: Function to execute (must accept provider param)
            providers: List of providers to compete (default: use multi-provider list)
            num_competitors: Number of competitors
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Dict with winning result and competition stats
        """
        start_time = time.time()
        self.stats["total_competitions"] += 1
        
        # Select providers
        if providers is None:
            available_providers = self.multi_provider.get_available_providers()
            providers = available_providers[:num_competitors]
        
        if len(providers) < num_competitors:
            logger.warning(f"Only {len(providers)} providers available, wanted {num_competitors}")
        
        # Create competitor tasks
        tasks = []
        for provider in providers:
            task = asyncio.create_task(
                self._execute_competitor(
                    competitor_id=f"competitor_{provider}",
                    provider=provider,
                    execute_func=execute_func,
                    args=args,
                    kwargs=kwargs
                )
            )
            tasks.append(task)
        
        # Wait for all competitors
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = [
            r for r in results 
            if isinstance(r, CompetitorResult) and r.success
        ]
        
        if not successful_results:
            logger.error("❌ All competitors failed")
            return {
                "success": False,
                "error": "All competitors failed",
                "competitors": len(providers),
                "execution_time": time.time() - start_time
            }
        
        # Judge picks winner
        winner = self._judge_results(successful_results)
        
        # Update stats
        if winner.provider not in self.stats["wins_by_provider"]:
            self.stats["wins_by_provider"][winner.provider] = 0
        self.stats["wins_by_provider"][winner.provider] += 1
        
        total_time = time.time() - start_time
        
        logger.info(
            f"🏆 Winner: {winner.provider} "
            f"(quality: {winner.quality_score:.2f}, "
            f"time: {winner.execution_time:.2f}s)"
        )
        
        return {
            "success": True,
            "result": winner.result,
            "winner": {
                "provider": winner.provider,
                "quality_score": winner.quality_score,
                "execution_time": winner.execution_time
            },
            "competitors": [
                {
                    "provider": r.provider,
                    "quality_score": r.quality_score,
                    "execution_time": r.execution_time,
                    "success": r.success
                }
                for r in successful_results
            ],
            "total_competitors": len(providers),
            "successful_competitors": len(successful_results),
            "total_time": total_time
        }
    
    async def _execute_competitor(
        self,
        competitor_id: str,
        provider: str,
        execute_func,
        args: tuple,
        kwargs: dict
    ) -> CompetitorResult:
        """Execute single competitor."""
        start_time = time.time()
        
        try:
            # Add provider to kwargs
            kwargs_with_provider = {**kwargs, "provider": provider}
            
            # Execute
            result = await execute_func(*args, **kwargs_with_provider)
            
            execution_time = time.time() - start_time
            
            # Calculate quality score
            quality_score = self._calculate_quality(result)
            
            return CompetitorResult(
                competitor_id=competitor_id,
                provider=provider,
                result=result,
                execution_time=execution_time,
                success=True,
                quality_score=quality_score
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.warning(f"⚠️ Competitor {competitor_id} ({provider}) failed: {e}")
            
            return CompetitorResult(
                competitor_id=competitor_id,
                provider=provider,
                result={"error": str(e)},
                execution_time=execution_time,
                success=False,
                quality_score=0.0
            )
    
    def _calculate_quality(self, result: Any) -> float:
        """
        Calculate quality score for result.
        
        HONEST: Simple heuristic, not sophisticated LLM judge.
        In production, use LLM to evaluate quality.
        
        Current heuristic:
        - Length of response (more detail = better)
        - Success indicators
        - Completeness
        """
        score = 0.0
        
        if isinstance(result, dict):
            # Check success
            if result.get("success"):
                score += 0.3
            
            # Check for content
            if "result" in result or "data" in result or "content" in result:
                score += 0.2
            
            # Check response length (proxy for detail)
            result_str = str(result)
            if len(result_str) > 100:
                score += 0.2
            if len(result_str) > 500:
                score += 0.2
            if len(result_str) > 1000:
                score += 0.1
        
        return min(score, 1.0)
    
    def _judge_results(self, results: List[CompetitorResult]) -> CompetitorResult:
        """
        Judge picks best result.
        
        HONEST: Simple scoring, not LLM judge.
        Combines quality score and speed.
        """
        # Score = quality * 0.7 + speed_score * 0.3
        min_time = min(r.execution_time for r in results)
        max_time = max(r.execution_time for r in results)
        time_range = max_time - min_time if max_time > min_time else 1.0
        
        scored_results = []
        for r in results:
            # Speed score (faster = better)
            speed_score = 1.0 - ((r.execution_time - min_time) / time_range)
            
            # Combined score
            combined_score = r.quality_score * 0.7 + speed_score * 0.3
            
            scored_results.append((combined_score, r))
        
        # Pick highest score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return scored_results[0][1]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get competition statistics."""
        return {
            "total_competitions": self.stats["total_competitions"],
            "wins_by_provider": self.stats["wins_by_provider"],
            "provider_win_rates": {
                provider: wins / self.stats["total_competitions"]
                for provider, wins in self.stats["wins_by_provider"].items()
            } if self.stats["total_competitions"] > 0 else {}
        }


# Global instance
_competition_system: Optional[AgentCompetitionSystem] = None


def get_agent_competition_system(multi_provider_strategy) -> AgentCompetitionSystem:
    """Get or create agent competition system instance."""
    global _competition_system
    if _competition_system is None:
        _competition_system = AgentCompetitionSystem(
            multi_provider_strategy=multi_provider_strategy
        )
    return _competition_system

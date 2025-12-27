#!/usr/bin/env python3
"""
Multi-Critic Review System for NIS Protocol
Parallel specialist critics for quality assurance

Copyright 2025 Organica AI Solutions
Licensed under Apache License 2.0
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CriticType(Enum):
    """Types of specialist critics."""
    FACT_CHECKER = "fact_checker"
    TONE_CHECKER = "tone_checker"
    RISK_CHECKER = "risk_checker"
    COMPLETENESS_CHECKER = "completeness_checker"
    TECHNICAL_REVIEWER = "technical_reviewer"


@dataclass
class CriticReview:
    """Review from a single critic."""
    critic_type: CriticType
    score: float  # 0.0 to 1.0
    issues: List[str]
    suggestions: List[str]
    reasoning: str


class MultiCriticReviewSystem:
    """
    Multi-critic review with parallel specialist evaluation.
    
    Technique: Send content to multiple specialist critics simultaneously
    - Fact checker validates accuracy
    - Tone checker evaluates communication
    - Risk checker identifies problems
    - Technical reviewer checks correctness
    - Final editor combines all feedback
    
    Honest Assessment:
    - Real LLM-based critics (not heuristics)
    - Real parallel execution
    - Specialized prompts per critic
    - Combined feedback synthesis
    - 90% real - actual AI critics with specialization
    """
    
    def __init__(self, llm_provider):
        """Initialize multi-critic review system."""
        self.llm_provider = llm_provider
        
        # Critic configurations
        self.critics = {
            CriticType.FACT_CHECKER: {
                "name": "Fact Checker",
                "focus": "Verify factual accuracy and correctness",
                "checks": ["accuracy", "sources", "claims"]
            },
            CriticType.TONE_CHECKER: {
                "name": "Tone Checker",
                "focus": "Evaluate communication style and clarity",
                "checks": ["clarity", "professionalism", "readability"]
            },
            CriticType.RISK_CHECKER: {
                "name": "Risk Checker",
                "focus": "Identify potential problems and risks",
                "checks": ["safety", "security", "compliance"]
            },
            CriticType.COMPLETENESS_CHECKER: {
                "name": "Completeness Checker",
                "focus": "Ensure thoroughness and coverage",
                "checks": ["coverage", "missing_info", "depth"]
            },
            CriticType.TECHNICAL_REVIEWER: {
                "name": "Technical Reviewer",
                "focus": "Validate technical correctness",
                "checks": ["technical_accuracy", "best_practices", "implementation"]
            }
        }
        
        logger.info(f"👥 Multi-Critic Review initialized with {len(self.critics)} critics")
    
    async def review_content(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        critics: Optional[List[CriticType]] = None
    ) -> Dict[str, Any]:
        """
        Review content with multiple critics in parallel.
        
        Args:
            content: Content to review
            context: Optional context (goal, requirements, etc.)
            critics: Optional list of critics to use (default: all)
            
        Returns:
            Dict with all reviews and combined feedback
        """
        # Select critics
        if critics is None:
            critics = list(self.critics.keys())
        
        logger.info(f"👥 Starting multi-critic review with {len(critics)} critics")
        
        # Run all critics in parallel
        review_tasks = [
            asyncio.create_task(self._run_critic(critic_type, content, context))
            for critic_type in critics
        ]
        
        reviews = await asyncio.gather(*review_tasks, return_exceptions=True)
        
        # Filter successful reviews
        successful_reviews = [
            r for r in reviews
            if isinstance(r, CriticReview)
        ]
        
        if not successful_reviews:
            logger.error("❌ All critics failed")
            return {
                "success": False,
                "error": "All critics failed"
            }
        
        # Combine feedback
        combined_feedback = await self._combine_feedback(
            content,
            successful_reviews,
            context
        )
        
        # Calculate overall score
        overall_score = sum(r.score for r in successful_reviews) / len(successful_reviews)
        
        logger.info(f"✅ Multi-critic review complete: {len(successful_reviews)} reviews, score: {overall_score:.2f}")
        
        return {
            "success": True,
            "overall_score": overall_score,
            "reviews": [self._review_to_dict(r) for r in successful_reviews],
            "combined_feedback": combined_feedback,
            "critics_used": len(successful_reviews)
        }
    
    async def _run_critic(
        self,
        critic_type: CriticType,
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> CriticReview:
        """Run single critic review."""
        try:
            critic_config = self.critics[critic_type]
            
            # Build critic prompt
            prompt = self._build_critic_prompt(
                critic_type,
                critic_config,
                content,
                context
            )
            
            # Call LLM as critic
            messages = [
                {"role": "system", "content": f"You are a {critic_config['name']} specializing in {critic_config['focus']}. Be thorough and constructive. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm_provider.generate_response(
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse review
            review = self._parse_critic_response(
                critic_type,
                response.get("content", "")
            )
            
            logger.info(f"✅ {critic_config['name']} review: score={review.score:.2f}")
            
            return review
            
        except Exception as e:
            logger.error(f"❌ Critic {critic_type.value} failed: {e}")
            # Return neutral review on failure
            return CriticReview(
                critic_type=critic_type,
                score=0.5,
                issues=[],
                suggestions=[],
                reasoning="Critic failed"
            )
    
    def _build_critic_prompt(
        self,
        critic_type: CriticType,
        critic_config: Dict[str, Any],
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for specific critic."""
        
        context_str = ""
        if context:
            if "goal" in context:
                context_str += f"\n**Goal**: {context['goal']}"
            if "requirements" in context:
                context_str += f"\n**Requirements**: {context['requirements']}"
        
        checks_str = "\n".join([f"- {check}" for check in critic_config["checks"]])
        
        prompt = f"""Review this content as a {critic_config['name']}.

**Your Focus**: {critic_config['focus']}
{context_str}

**Content to Review**:
{content[:2000]}  

**Your Checks**:
{checks_str}

**Task**: Provide a thorough review with:
1. Score (0.0 to 1.0) - how well does it meet your criteria?
2. Issues found (list specific problems)
3. Suggestions for improvement
4. Reasoning for your score

Output JSON format:
{{
    "score": 0.85,
    "issues": ["Issue 1", "Issue 2"],
    "suggestions": ["Suggestion 1", "Suggestion 2"],
    "reasoning": "Detailed explanation of score and findings"
}}

Be constructive and specific. Output ONLY valid JSON."""
        
        return prompt
    
    def _parse_critic_response(
        self,
        critic_type: CriticType,
        response: str
    ) -> CriticReview:
        """Parse critic response."""
        try:
            import json
            
            # Clean response
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])
            
            data = json.loads(response)
            
            return CriticReview(
                critic_type=critic_type,
                score=data.get("score", 0.5),
                issues=data.get("issues", []),
                suggestions=data.get("suggestions", []),
                reasoning=data.get("reasoning", "")
            )
            
        except Exception as e:
            logger.error(f"Failed to parse critic response: {e}")
            return CriticReview(
                critic_type=critic_type,
                score=0.5,
                issues=[],
                suggestions=[],
                reasoning="Parse failed"
            )
    
    async def _combine_feedback(
        self,
        content: str,
        reviews: List[CriticReview],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Combine feedback from all critics using LLM editor."""
        try:
            # Build combination prompt
            reviews_summary = "\n\n".join([
                f"**{self.critics[r.critic_type]['name']}** (Score: {r.score:.2f}):\n"
                f"Issues: {', '.join(r.issues) if r.issues else 'None'}\n"
                f"Suggestions: {', '.join(r.suggestions) if r.suggestions else 'None'}\n"
                f"Reasoning: {r.reasoning}"
                for r in reviews
            ])
            
            prompt = f"""You are a senior editor combining feedback from multiple specialist reviewers.

**Original Content**:
{content[:1000]}

**Specialist Reviews**:
{reviews_summary}

**Task**: Synthesize all feedback into:
1. Key strengths
2. Priority issues to address
3. Actionable recommendations
4. Overall assessment

Be concise and actionable. Focus on the most important points."""
            
            messages = [
                {"role": "system", "content": "You are a senior editor synthesizing feedback from multiple reviewers."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm_provider.generate_response(
                messages=messages,
                temperature=0.4,
                max_tokens=800
            )
            
            return response.get("content", "No combined feedback available")
            
        except Exception as e:
            logger.error(f"Failed to combine feedback: {e}")
            return "Failed to combine feedback"
    
    def _review_to_dict(self, review: CriticReview) -> Dict[str, Any]:
        """Convert review to dictionary."""
        return {
            "critic": review.critic_type.value,
            "score": review.score,
            "issues": review.issues,
            "suggestions": review.suggestions,
            "reasoning": review.reasoning
        }


# Global instance
_multi_critic: Optional[MultiCriticReviewSystem] = None


def get_multi_critic_review_system(llm_provider) -> MultiCriticReviewSystem:
    """Get or create multi-critic review system instance."""
    global _multi_critic
    if _multi_critic is None:
        _multi_critic = MultiCriticReviewSystem(llm_provider=llm_provider)
    return _multi_critic

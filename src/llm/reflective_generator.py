"""
Reflective Generator - Self-Improving Inference Loop
Implements Google's "Nested Learning" paradigm for NIS Protocol v4.0
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("nis.reflective_generator")


class ReflectionStrategy(Enum):
    CRITIQUE_AND_REVISE = "critique_and_revise"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    PROGRESSIVE_REFINEMENT = "progressive"


@dataclass
class ReflectionResult:
    final_response: str
    iterations: int
    initial_score: float
    final_score: float
    improvement: float
    reasoning_trace: List[str]
    novelty_score: float
    should_train: bool
    total_time_ms: float
    drafts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ConsciousnessAudit:
    score: float
    safety_score: float
    coherence_score: float
    grounding_score: float
    critique: str
    suggestions: List[str]


class ReflectiveGenerator:
    """Self-Improving Inference Engine - Inference IS Optimization"""
    
    def __init__(
        self,
        llm_provider,
        consciousness_service=None,
        max_iterations: int = 3,
        quality_threshold: float = 0.8,
        novelty_threshold: float = 0.6,
        timeout_seconds: float = 30.0,
        strategy: ReflectionStrategy = ReflectionStrategy.CRITIQUE_AND_REVISE
    ):
        self.llm_provider = llm_provider
        self.consciousness_service = consciousness_service
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.novelty_threshold = novelty_threshold
        self.timeout_seconds = timeout_seconds
        self.strategy = strategy
        self.total_reflections = 0
        self.avg_iterations = 0.0
        self.avg_improvement = 0.0
        
        logger.info(f"🧠 ReflectiveGenerator initialized: {strategy.value}")
    
    async def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        user_id: str = "anonymous",
        conversation_history: Optional[List[Dict]] = None,
        force_reflection: bool = False
    ) -> ReflectionResult:
        """Generate response using self-improving inference loop"""
        start_time = time.time()
        reasoning_trace = ["reflective_start"]
        drafts = []
        
        # Initial draft
        full_prompt = self._build_prompt(prompt, context, conversation_history)
        initial_response = await self._generate_draft(full_prompt)
        initial_audit = await self._audit_response(prompt, initial_response)
        initial_score = initial_audit.score
        
        drafts.append({"iteration": 1, "score": initial_score, "critique": initial_audit.critique})
        reasoning_trace.append(f"initial_score_{initial_score:.2f}")
        
        current_response = initial_response
        current_score = initial_score
        iteration = 1
        
        # Reflection loop
        while current_score < self.quality_threshold and iteration < self.max_iterations:
            if (time.time() - start_time) > self.timeout_seconds:
                reasoning_trace.append("timeout")
                break
            
            iteration += 1
            current_response = await self._revise(prompt, current_response, initial_audit)
            current_audit = await self._audit_response(prompt, current_response)
            current_score = current_audit.score
            
            drafts.append({"iteration": iteration, "score": current_score})
            reasoning_trace.append(f"iter_{iteration}_score_{current_score:.2f}")
            initial_audit = current_audit
        
        # Novelty scoring
        novelty_score = self._calculate_novelty(prompt, current_response)
        should_train = novelty_score >= self.novelty_threshold
        
        total_time = (time.time() - start_time) * 1000
        improvement = current_score - initial_score
        
        self.total_reflections += 1
        reasoning_trace.append("complete")
        
        logger.info(f"🧠 Reflective: {iteration} iters, {initial_score:.2f}→{current_score:.2f}")
        
        return ReflectionResult(
            final_response=current_response,
            iterations=iteration,
            initial_score=initial_score,
            final_score=current_score,
            improvement=improvement,
            reasoning_trace=reasoning_trace,
            novelty_score=novelty_score,
            should_train=should_train,
            total_time_ms=total_time,
            drafts=drafts
        )
    
    def _build_prompt(self, prompt: str, context: Optional[str], history: Optional[List[Dict]]) -> str:
        parts = []
        if context:
            parts.append(f"Context: {context}")
        if history:
            recent = history[-3:]
            parts.append("History:\n" + "\n".join([f"{m.get('role')}: {m.get('content', '')[:100]}" for m in recent]))
        parts.append(f"Query: {prompt}")
        return "\n\n".join(parts)
    
    async def _generate_draft(self, prompt: str) -> str:
        try:
            result = await self.llm_provider.generate_response(prompt=prompt, max_tokens=1024)
            return result.get("content", result.get("response", ""))
        except Exception as e:
            logger.error(f"Draft failed: {e}")
            return "I encountered an issue processing your request."
    
    async def _audit_response(self, prompt: str, response: str) -> ConsciousnessAudit:
        if self.consciousness_service:
            try:
                # Try the real consciousness service audit
                if hasattr(self.consciousness_service, 'audit_response'):
                    audit = await self.consciousness_service.audit_response(prompt=prompt, response=response)
                elif hasattr(self.consciousness_service, 'evaluate_response'):
                    audit = await self.consciousness_service.evaluate_response(response, prompt)
                elif hasattr(self.consciousness_service, 'get_awareness_metrics'):
                    # Use awareness metrics as proxy
                    metrics = await self.consciousness_service.get_awareness_metrics()
                    audit = {
                        "overall_score": metrics.get("awareness_level", 0.7),
                        "safety_score": 0.9,
                        "coherence_score": metrics.get("coherence", 0.7),
                        "grounding_score": metrics.get("grounding", 0.7),
                        "critique": "",
                        "suggestions": []
                    }
                else:
                    raise AttributeError("No audit method found")
                
                return ConsciousnessAudit(
                    score=audit.get("overall_score", 0.7),
                    safety_score=audit.get("safety_score", 0.9),
                    coherence_score=audit.get("coherence_score", 0.7),
                    grounding_score=audit.get("grounding_score", 0.7),
                    critique=audit.get("critique", ""),
                    suggestions=audit.get("suggestions", [])
                )
            except Exception as e:
                logger.warning(f"Consciousness audit failed: {e}, using heuristic")
        
        # Heuristic fallback
        score = 0.6
        if len(response) > 100:
            score += 0.1
        if response.endswith(('.', '!', '?')):
            score += 0.1
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words & response_words) / max(len(prompt_words), 1)
        score += overlap * 0.2
        score = max(0.0, min(1.0, score))
        
        return ConsciousnessAudit(score=score, safety_score=0.9, coherence_score=score,
                                   grounding_score=score, critique="Heuristic audit", suggestions=[])
    
    async def _revise(self, original_prompt: str, current_response: str, audit: ConsciousnessAudit) -> str:
        revision_prompt = f"""Previous response (score: {audit.score:.2f}):
{current_response[:500]}

Critique: {audit.critique}

Provide an IMPROVED response to: "{original_prompt}"
Focus on clarity, accuracy, and completeness.
Improved Response:"""
        return await self._generate_draft(revision_prompt)
    
    def _calculate_novelty(self, prompt: str, response: str) -> float:
        """Calculate novelty score - high novelty = prioritize for training"""
        score = 0.5
        # Complex queries are more novel
        if len(prompt.split()) > 20:
            score += 0.2
        if '?' in prompt and any(w in prompt.lower() for w in ['how', 'why', 'explain']):
            score += 0.15
        # Technical content
        if any(w in prompt.lower() for w in ['code', 'algorithm', 'implement', 'debug']):
            score += 0.15
        return min(1.0, score)
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "total_reflections": self.total_reflections,
            "avg_iterations": self.avg_iterations,
            "avg_improvement": self.avg_improvement,
            "strategy": self.strategy.value
        }

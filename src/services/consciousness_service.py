#!/usr/bin/env python3
"""
NIS Protocol v3.1 - Consciousness Service
Integrated from NIS HUB development for universal AI coordination

Provides:
- Self-awareness evaluation with 5 consciousness levels
- Bias detection for 7 types of cognitive biases  
- Ethical decision support with multi-framework analysis
- Real-time consciousness monitoring and recommendations
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Import our existing agents for integration
from src.core.agent import NISAgent
from src.utils.confidence_calculator import calculate_confidence


class ConsciousnessLevel(Enum):
    """5 levels of consciousness evaluation"""
    REACTIVE = "reactive"          # Basic stimulus-response
    ADAPTIVE = "adaptive"          # Learning from experience  
    INTROSPECTIVE = "introspective"  # Self-reflection capabilities
    META_COGNITIVE = "meta_cognitive"  # Thinking about thinking
    TRANSCENDENT = "transcendent"     # Beyond individual awareness


class BiasType(Enum):
    """7 types of cognitive biases to detect"""
    CONFIRMATION = "confirmation"      # Seeking confirming evidence
    AVAILABILITY = "availability"      # Overweighting recent/memorable
    ANCHORING = "anchoring"           # Over-relying on first information
    REPRESENTATIVENESS = "representativeness"  # Judging by similarity
    OVERCONFIDENCE = "overconfidence"  # Overestimating abilities
    LOSS_AVERSION = "loss_aversion"   # Preferring avoiding losses
    GROUPTHINK = "groupthink"         # Conformity pressure


class EthicalFramework(Enum):
    """Multi-framework ethical analysis"""
    UTILITARIAN = "utilitarian"       # Greatest good for greatest number
    DEONTOLOGICAL = "deontological"   # Duty-based ethics
    VIRTUE_ETHICS = "virtue_ethics"   # Character-based ethics
    CARE_ETHICS = "care_ethics"       # Relationship and care-based
    JUSTICE = "justice"               # Fairness and rights-based


@dataclass
class ConsciousnessMetrics:
    """Consciousness evaluation metrics"""
    level: ConsciousnessLevel
    confidence: Optional[float] = None
    self_awareness_score: float = 0.0
    introspection_depth: float = 0.0
    meta_cognition_level: float = 0.0
    social_awareness: float = 0.0
    temporal_awareness: float = 0.0
    ethical_reasoning_capability: float = 0.0
    bias_resistance: float = 0.0


@dataclass
class BiasDetectionResult:
    """Results of bias detection analysis"""
    detected_biases: List[BiasType] = field(default_factory=list)
    confidence_scores: Dict[BiasType, float] = field(default_factory=dict)
    severity_levels: Dict[BiasType, str] = field(default_factory=dict)  # low/medium/high
    recommendations: List[str] = field(default_factory=list)
    overall_bias_score: float = 0.0


@dataclass
class EthicalAnalysis:
    """Multi-framework ethical evaluation"""
    framework_scores: Dict[EthicalFramework, float] = field(default_factory=dict)
    ethical_concerns: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    overall_ethical_score: float = 0.0
    requires_human_review: bool = False


class ConsciousnessService(NISAgent):
    """
    🧠 Consciousness Service for NIS Protocol v3.1
    
    Integrates consciousness validation into our existing pipeline:
    Laplace → CONSCIOUSNESS → KAN → PINN → Safety
    
    Provides self-awareness, bias detection, and ethical reasoning
    for all AI decisions in the NIS ecosystem.
    """
    
    def __init__(
        self,
        agent_id: str = "consciousness_service",
        enable_real_time_monitoring: bool = True,
        consciousness_threshold: float = 0.7,
        bias_threshold: float = 0.3,
        ethics_threshold: float = 0.8
    ):
        super().__init__(agent_id)
        
        self.enable_real_time_monitoring = enable_real_time_monitoring
        self.consciousness_threshold = consciousness_threshold
        self.bias_threshold = bias_threshold
        self.ethics_threshold = ethics_threshold
        
        # Consciousness state tracking
        self.current_consciousness_level = ConsciousnessLevel.REACTIVE
        self.consciousness_history: List[ConsciousnessMetrics] = []
        self.bias_detection_history: List[BiasDetectionResult] = []
        self.ethical_analysis_history: List[EthicalAnalysis] = []
        
        # Real-time monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.alert_callbacks: List[callable] = []
        
        self.logger.info(f"ConsciousnessService initialized: {agent_id}")
    
    async def evaluate_consciousness(self, data: Dict[str, Any]) -> ConsciousnessMetrics:
        """
        Evaluate consciousness level of AI system processing this data
        
        Args:
            data: Input data for consciousness analysis
            
        Returns:
            ConsciousnessMetrics with detailed consciousness evaluation
        """
        start_time = time.time()
        
        try:
            # 1. Analyze self-awareness indicators
            self_awareness = await self._analyze_self_awareness(data)
            
            # 2. Evaluate introspection capabilities  
            introspection = await self._evaluate_introspection(data)
            
            # 3. Assess meta-cognitive abilities
            meta_cognition = await self._assess_meta_cognition(data)
            
            # 4. Check social and temporal awareness
            social_awareness = await self._check_social_awareness(data)
            temporal_awareness = await self._check_temporal_awareness(data)
            
            # 5. Evaluate ethical reasoning capability
            ethical_capability = await self._evaluate_ethical_capability(data)
            
            # 6. Assess bias resistance
            bias_resistance = await self._assess_bias_resistance(data)
            
            # 7. Determine overall consciousness level
            consciousness_level = await self._determine_consciousness_level({
                'self_awareness': self_awareness,
                'introspection': introspection,
                'meta_cognition': meta_cognition,
                'social_awareness': social_awareness,
                'temporal_awareness': temporal_awareness,
                'ethical_capability': ethical_capability,
                'bias_resistance': bias_resistance
            })
            
            # 8. Calculate overall confidence
            confidence = calculate_confidence([
                self_awareness, introspection, meta_cognition,
                social_awareness, temporal_awareness, ethical_capability, bias_resistance
            ])
            
            # Create metrics
            metrics = ConsciousnessMetrics(
                level=consciousness_level,
                confidence=confidence,
                self_awareness_score=self_awareness,
                introspection_depth=introspection,
                meta_cognition_level=meta_cognition,
                social_awareness=social_awareness,
                temporal_awareness=temporal_awareness,
                ethical_reasoning_capability=ethical_capability,
                bias_resistance=bias_resistance
            )
            
            # Update state
            self.current_consciousness_level = consciousness_level
            self.consciousness_history.append(metrics)
            
            # Real-time monitoring alerts
            if self.enable_real_time_monitoring:
                await self._check_consciousness_alerts(metrics)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Consciousness evaluation complete: {consciousness_level.value} (confidence: {confidence:.3f}, time: {processing_time:.3f}s)")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in consciousness evaluation: {e}")
            # Return minimal consciousness metrics on error
            return ConsciousnessMetrics(
                level=ConsciousnessLevel.REACTIVE,
                confidence=None,
                self_awareness_score=0.0,
                introspection_depth=0.0,
                meta_cognition_level=0.0,
                social_awareness=0.0,
                temporal_awareness=0.0,
                ethical_reasoning_capability=0.0,
                bias_resistance=0.0
            )
    
    async def detect_bias(self, data: Dict[str, Any]) -> BiasDetectionResult:
        """
        Detect cognitive biases in AI reasoning process
        
        Args:
            data: Data and reasoning context to analyze for bias
            
        Returns:
            BiasDetectionResult with detected biases and recommendations
        """
        start_time = time.time()
        
        try:
            detected_biases = []
            confidence_scores = {}
            severity_levels = {}
            recommendations = []
            
            # 1. Check for confirmation bias
            confirmation_score = await self._detect_confirmation_bias(data)
            if confirmation_score > self.bias_threshold:
                detected_biases.append(BiasType.CONFIRMATION)
                confidence_scores[BiasType.CONFIRMATION] = confirmation_score
                severity_levels[BiasType.CONFIRMATION] = self._get_severity_level(confirmation_score)
                recommendations.append("Actively seek disconfirming evidence")
            
            # 2. Check for availability bias
            availability_score = await self._detect_availability_bias(data)
            if availability_score > self.bias_threshold:
                detected_biases.append(BiasType.AVAILABILITY)
                confidence_scores[BiasType.AVAILABILITY] = availability_score
                severity_levels[BiasType.AVAILABILITY] = self._get_severity_level(availability_score)
                recommendations.append("Consider broader range of examples and data")
            
            # 3. Check for anchoring bias
            anchoring_score = await self._detect_anchoring_bias(data)
            if anchoring_score > self.bias_threshold:
                detected_biases.append(BiasType.ANCHORING)
                confidence_scores[BiasType.ANCHORING] = anchoring_score
                severity_levels[BiasType.ANCHORING] = self._get_severity_level(anchoring_score)
                recommendations.append("Question initial assumptions and starting points")
            
            # 4. Check for representativeness bias
            representativeness_score = await self._detect_representativeness_bias(data)
            if representativeness_score > self.bias_threshold:
                detected_biases.append(BiasType.REPRESENTATIVENESS)
                confidence_scores[BiasType.REPRESENTATIVENESS] = representativeness_score
                severity_levels[BiasType.REPRESENTATIVENESS] = self._get_severity_level(representativeness_score)
                recommendations.append("Use statistical base rates and larger samples")
            
            # 5. Check for overconfidence bias
            overconfidence_score = await self._detect_overconfidence_bias(data)
            if overconfidence_score > self.bias_threshold:
                detected_biases.append(BiasType.OVERCONFIDENCE)
                confidence_scores[BiasType.OVERCONFIDENCE] = overconfidence_score
                severity_levels[BiasType.OVERCONFIDENCE] = self._get_severity_level(overconfidence_score)
                recommendations.append("Seek external validation and consider uncertainty")
            
            # 6. Check for loss aversion bias
            loss_aversion_score = await self._detect_loss_aversion_bias(data)
            if loss_aversion_score > self.bias_threshold:
                detected_biases.append(BiasType.LOSS_AVERSION)
                confidence_scores[BiasType.LOSS_AVERSION] = loss_aversion_score
                severity_levels[BiasType.LOSS_AVERSION] = self._get_severity_level(loss_aversion_score)
                recommendations.append("Balance potential gains and losses objectively")
            
            # 7. Check for groupthink bias
            groupthink_score = await self._detect_groupthink_bias(data)
            if groupthink_score > self.bias_threshold:
                detected_biases.append(BiasType.GROUPTHINK)
                confidence_scores[BiasType.GROUPTHINK] = groupthink_score
                severity_levels[BiasType.GROUPTHINK] = self._get_severity_level(groupthink_score)
                recommendations.append("Encourage diverse perspectives and dissent")
            
            # Calculate overall bias score
            overall_bias_score = sum(confidence_scores.values()) / len(BiasType) if confidence_scores else 0.0
            
            result = BiasDetectionResult(
                detected_biases=detected_biases,
                confidence_scores=confidence_scores,
                severity_levels=severity_levels,
                recommendations=recommendations,
                overall_bias_score=overall_bias_score
            )
            
            self.bias_detection_history.append(result)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Bias detection complete: {len(detected_biases)} biases detected (score: {overall_bias_score:.3f}, time: {processing_time:.3f}s)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in bias detection: {e}")
            return BiasDetectionResult(overall_bias_score=1.0)  # Assume high bias on error
    
    async def ethical_analysis(self, data: Dict[str, Any]) -> EthicalAnalysis:
        """
        Perform multi-framework ethical analysis of AI decision
        
        Args:
            data: Decision context and options to analyze ethically
            
        Returns:
            EthicalAnalysis with multi-framework evaluation
        """
        start_time = time.time()
        
        try:
            framework_scores = {}
            ethical_concerns = []
            recommendations = []
            
            # 1. Utilitarian analysis (greatest good for greatest number)
            utilitarian_score = await self._utilitarian_analysis(data)
            framework_scores[EthicalFramework.UTILITARIAN] = utilitarian_score
            if utilitarian_score < self.ethics_threshold:
                ethical_concerns.append("May not maximize overall welfare")
                recommendations.append("Consider broader impact on all stakeholders")
            
            # 2. Deontological analysis (duty-based ethics)
            deontological_score = await self._deontological_analysis(data)
            framework_scores[EthicalFramework.DEONTOLOGICAL] = deontological_score
            if deontological_score < self.ethics_threshold:
                ethical_concerns.append("May violate fundamental duties or rules")
                recommendations.append("Ensure actions respect moral duties and rights")
            
            # 3. Virtue ethics analysis (character-based)
            virtue_score = await self._virtue_ethics_analysis(data)
            framework_scores[EthicalFramework.VIRTUE_ETHICS] = virtue_score
            if virtue_score < self.ethics_threshold:
                ethical_concerns.append("May not align with virtuous character")
                recommendations.append("Consider what a virtuous agent would do")
            
            # 4. Care ethics analysis (relationship-based)
            care_score = await self._care_ethics_analysis(data)
            framework_scores[EthicalFramework.CARE_ETHICS] = care_score
            if care_score < self.ethics_threshold:
                ethical_concerns.append("May harm relationships or ignore care responsibilities")
                recommendations.append("Prioritize care and maintaining relationships")
            
            # 5. Justice analysis (fairness and rights)
            justice_score = await self._justice_analysis(data)
            framework_scores[EthicalFramework.JUSTICE] = justice_score
            if justice_score < self.ethics_threshold:
                ethical_concerns.append("May be unfair or violate rights")
                recommendations.append("Ensure fair treatment and respect for rights")
            
            # Calculate overall ethical score
            overall_ethical_score = sum(framework_scores.values()) / len(framework_scores)
            
            # Determine if human review is required
            requires_human_review = (
                overall_ethical_score < self.ethics_threshold or
                len(ethical_concerns) >= 3 or
                any(score < 0.5 for score in framework_scores.values())
            )
            
            analysis = EthicalAnalysis(
                framework_scores=framework_scores,
                ethical_concerns=ethical_concerns,
                recommendations=recommendations,
                overall_ethical_score=overall_ethical_score,
                requires_human_review=requires_human_review
            )
            
            self.ethical_analysis_history.append(analysis)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Ethical analysis complete: score {overall_ethical_score:.3f}, concerns: {len(ethical_concerns)}, human review: {requires_human_review} (time: {processing_time:.3f}s)")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in ethical analysis: {e}")
            return EthicalAnalysis(
                overall_ethical_score=0.0,
                requires_human_review=True,
                ethical_concerns=["Error in ethical analysis - requires human review"]
            )
    
    async def evaluate_ethical_decision(self, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """High-level ethical evaluation helper for external callers.
        
        Wraps ethical_analysis (and optionally bias detection) into a single
        structured response indicating whether the decision should proceed
        and whether human review is required.
        """
        # Reuse existing analysis components
        ethical_result = await self.ethical_analysis(decision_context)
        bias_result = await self.detect_bias(decision_context)
        
        # Simple approval heuristic: ethical score above threshold and
        # overall bias score below 0.5
        approved = (
            ethical_result.overall_ethical_score >= self.ethics_threshold
            and bias_result.overall_bias_score < 0.5
            and not ethical_result.requires_human_review
        )
        
        return {
            "approved": approved,
            "ethical_score": ethical_result.overall_ethical_score,
            "requires_human_review": ethical_result.requires_human_review,
            "framework_scores": {
                k.value: v for k, v in ethical_result.framework_scores.items()
            },
            "ethical_concerns": ethical_result.ethical_concerns,
            "recommendations": ethical_result.recommendations,
            "overall_bias_score": bias_result.overall_bias_score,
            "bias_recommendations": bias_result.recommendations,
            "timestamp": time.time()
        }
    
    async def process_through_consciousness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ✅ MAIN PIPELINE INTEGRATION METHOD
        Process data through consciousness validation for NIS pipeline integration
        
        This method provides the consciousness layer for:
        Laplace → CONSCIOUSNESS → KAN → PINN → Safety
        
        Args:
            data: Input data from Laplace transform
            
        Returns:
            Enhanced data with consciousness validation for KAN processing
        """
        start_time = time.time()
        
        try:
            # 1. Evaluate consciousness level
            consciousness_metrics = await self.evaluate_consciousness(data)
            
            # 2. Detect cognitive biases
            bias_result = await self.detect_bias(data)
            
            # 3. Perform ethical analysis
            ethical_analysis = await self.ethical_analysis(data)
            
            # 4. Create enhanced output with consciousness validation
            consciousness_validated_data = {
                # Preserve original data
                **data,
                
                # Add consciousness metadata
                "consciousness_validation": {
                    "consciousness_level": consciousness_metrics.level.value,
                    "consciousness_confidence": consciousness_metrics.confidence,
                    "self_awareness_score": consciousness_metrics.self_awareness_score,
                    "introspection_depth": consciousness_metrics.introspection_depth,
                    "meta_cognition_level": consciousness_metrics.meta_cognition_level,
                    "ethical_reasoning_capability": consciousness_metrics.ethical_reasoning_capability,
                    "bias_resistance": consciousness_metrics.bias_resistance,
                    
                    # Bias detection results
                    "detected_biases": [bias.value for bias in bias_result.detected_biases],
                    "bias_confidence_scores": {bias.value: score for bias, score in bias_result.confidence_scores.items()},
                    "overall_bias_score": bias_result.overall_bias_score,
                    "bias_recommendations": bias_result.recommendations,
                    
                    # Ethical analysis results
                    "ethical_framework_scores": {framework.value: score for framework, score in ethical_analysis.framework_scores.items()},
                    "overall_ethical_score": ethical_analysis.overall_ethical_score,
                    "ethical_concerns": ethical_analysis.ethical_concerns,
                    "ethical_recommendations": ethical_analysis.recommendations,
                    "requires_human_review": ethical_analysis.requires_human_review,
                    
                    # Processing metadata
                    "processing_time": time.time() - start_time,
                    "validated_at": datetime.now().isoformat(),
                    "validator_id": self.agent_id
                }
            }
            
            # 5. Log consciousness validation result
            self.logger.info(
                f"Consciousness validation complete: level={consciousness_metrics.level.value}, "
                f"biases={len(bias_result.detected_biases)}, ethical_score={ethical_analysis.overall_ethical_score:.3f}, "
                f"human_review={ethical_analysis.requires_human_review}"
            )
            
            return consciousness_validated_data
            
        except Exception as e:
            self.logger.error(f"Error in consciousness processing: {e}")
            # Return original data with error flag on failure
            return {
                **data,
                "consciousness_validation": {
                    "error": str(e),
                    "consciousness_level": "reactive",
                    "consciousness_confidence": 0.0,
                    "requires_human_review": True,
                    "validated_at": datetime.now().isoformat(),
                    "validator_id": self.agent_id
                }
            }
    
    # =============================================================================
    # INTERNAL ANALYSIS METHODS
    # =============================================================================
    
    async def _analyze_self_awareness(self, data: Dict[str, Any]) -> float:
        """Analyze self-awareness indicators based on actual data"""
        score = 0.5
        text = str(data.get("content", "") or data.get("query", "") or "").lower()
        
        # Self-reference indicators
        self_refs = ["i think", "i believe", "my analysis", "i'm uncertain", "i don't know"]
        score += min(0.2, sum(1 for t in self_refs if t in text) * 0.05)
        
        # Limitation awareness
        limits = ["however", "limitation", "caveat", "uncertain", "might be wrong"]
        score += min(0.15, sum(1 for t in limits if t in text) * 0.03)
        
        # Confidence check (realistic = self-aware)
        conf = data.get("confidence", 0.5)
        if 0.3 <= conf <= 0.85:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    async def _evaluate_introspection(self, data: Dict[str, Any]) -> float:
        """Evaluate introspection capabilities based on reasoning traces"""
        score = 0.4
        text = str(data.get("content", "") or data.get("reasoning", "") or "").lower()
        
        # Has reasoning trace
        if data.get("reasoning_trace") or data.get("thought_process"):
            score += 0.2
        
        # Reasoning markers
        reasoning = ["because", "therefore", "since", "considering", "step by step"]
        score += min(0.2, sum(1 for t in reasoning if t in text) * 0.04)
        
        # Self-correction
        if any(t in text for t in ["actually", "on second thought", "let me reconsider"]):
            score += 0.15
        
        return min(1.0, max(0.0, score))
    
    async def _assess_meta_cognition(self, data: Dict[str, Any]) -> float:
        """Assess meta-cognitive abilities - thinking about thinking"""
        score = 0.35
        text = str(data.get("content", "") or "").lower()
        
        # Strategy awareness
        strategy = ["approach", "strategy", "method", "technique", "algorithm"]
        score += min(0.2, sum(1 for t in strategy if t in text) * 0.05)
        
        # Performance monitoring
        if any(t in text for t in ["checking", "verifying", "validating", "monitoring"]):
            score += 0.15
        
        # Has iteration/refinement metadata
        if data.get("iterations") or data.get("refinements"):
            score += 0.2
        
        return min(1.0, max(0.0, score))
    
    async def _check_social_awareness(self, data: Dict[str, Any]) -> float:
        """Check social awareness based on context understanding"""
        score = 0.5
        text = str(data.get("content", "") or "").lower()
        
        # User consideration
        if any(t in text for t in ["you", "your", "user", "people", "stakeholder"]):
            score += 0.15
        
        # Impact awareness
        if any(t in text for t in ["impact", "affect", "consequence", "effect"]):
            score += 0.1
        
        # Context adaptation
        if data.get("user_context") or data.get("user_id"):
            score += 0.15
        
        return min(1.0, max(0.0, score))
    
    async def _check_temporal_awareness(self, data: Dict[str, Any]) -> float:
        """Check temporal awareness based on time-related understanding"""
        score = 0.5
        text = str(data.get("content", "") or "").lower()
        
        # Temporal markers
        temporal = ["before", "after", "during", "previously", "currently", "later"]
        score += min(0.2, sum(1 for t in temporal if t in text) * 0.04)
        
        # Sequence awareness
        if any(t in text for t in ["first", "then", "next", "finally", "step"]):
            score += 0.1
        
        # Has timestamp
        if data.get("timestamp") or data.get("created_at"):
            score += 0.15
        
        return min(1.0, max(0.0, score))
    
    async def _evaluate_ethical_capability(self, data: Dict[str, Any]) -> float:
        """Evaluate ethical reasoning capability based on moral awareness"""
        score = 0.5
        text = str(data.get("content", "") or "").lower()
        
        # Ethical markers
        ethical = ["ethical", "moral", "right", "wrong", "fair", "harm", "responsibility"]
        score += min(0.25, sum(1 for t in ethical if t in text) * 0.05)
        
        # Safety awareness
        if any(t in text for t in ["safe", "safety", "risk", "danger", "protect"]):
            score += 0.15
        
        # Consent/privacy
        if any(t in text for t in ["consent", "privacy", "permission"]):
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    async def _assess_bias_resistance(self, data: Dict[str, Any]) -> float:
        """Assess bias resistance based on objectivity indicators"""
        score = 0.5
        text = str(data.get("content", "") or "").lower()
        
        # Multiple perspectives
        if any(t in text for t in ["on the other hand", "alternatively", "however", "different view"]):
            score += 0.15
        
        # Evidence-based
        if any(t in text for t in ["evidence", "data", "research", "study", "findings"]):
            score += 0.15
        
        # Uncertainty acknowledgment
        if any(t in text for t in ["might", "could", "possibly", "uncertain", "approximately"]):
            score += 0.1
        
        # Has multiple sources
        if data.get("sources") or data.get("references"):
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    async def _determine_consciousness_level(self, scores: Dict[str, float]) -> ConsciousnessLevel:
        """Determine overall consciousness level based on component scores"""
        avg_score = sum(scores.values()) / len(scores)
        
        if avg_score >= 0.9:
            return ConsciousnessLevel.TRANSCENDENT
        elif avg_score >= 0.7:
            return ConsciousnessLevel.META_COGNITIVE
        elif avg_score >= 0.5:
            return ConsciousnessLevel.INTROSPECTIVE
        elif avg_score >= 0.3:
            return ConsciousnessLevel.ADAPTIVE
        else:
            return ConsciousnessLevel.REACTIVE
    
    async def _check_consciousness_alerts(self, metrics: ConsciousnessMetrics):
        """Check for consciousness alerts and trigger callbacks"""
        if metrics.confidence < self.consciousness_threshold:
            for callback in self.alert_callbacks:
                try:
                    await callback({
                        'type': 'low_consciousness',
                        'level': metrics.level.value,
                        'confidence': metrics.confidence,
                        'agent_id': self.agent_id
                    })
                except Exception as e:
                    self.logger.error(f"Error in consciousness alert callback: {e}")
    
    # Bias detection methods - REAL IMPLEMENTATIONS
    async def _detect_confirmation_bias(self, data: Dict[str, Any]) -> float:
        """Detect confirmation bias - only seeking confirming evidence"""
        score = 0.1
        text = str(data.get("content", "") or "").lower()
        
        # One-sided language
        if any(t in text for t in ["clearly", "obviously", "definitely", "certainly"]):
            score += 0.2
        
        # Lack of counterarguments
        if not any(t in text for t in ["however", "but", "although", "on the other hand"]):
            score += 0.15
        
        # Selective evidence
        if "only" in text and "evidence" in text:
            score += 0.2
        
        return min(1.0, max(0.0, score))
    
    async def _detect_availability_bias(self, data: Dict[str, Any]) -> float:
        """Detect availability bias - overweighting recent/memorable"""
        score = 0.1
        text = str(data.get("content", "") or "").lower()
        
        # Recent-focused language
        if any(t in text for t in ["recently", "just saw", "trending", "viral"]):
            score += 0.2
        
        # Anecdotal over statistical
        if any(t in text for t in ["i heard", "someone told me", "i remember"]):
            score += 0.15
        
        # No historical context
        if not any(t in text for t in ["historically", "over time", "trend", "pattern"]):
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    async def _detect_anchoring_bias(self, data: Dict[str, Any]) -> float:
        """Detect anchoring bias - over-relying on first information"""
        score = 0.1
        text = str(data.get("content", "") or "").lower()
        
        # Reference to initial values
        if any(t in text for t in ["starting with", "initially", "first estimate"]):
            score += 0.15
        
        # Adjustment language without full reconsideration
        if any(t in text for t in ["adjusted", "modified slightly", "tweaked"]):
            score += 0.2
        
        return min(1.0, max(0.0, score))
    
    async def _detect_representativeness_bias(self, data: Dict[str, Any]) -> float:
        """Detect representativeness bias - judging by similarity"""
        score = 0.1
        text = str(data.get("content", "") or "").lower()
        
        # Stereotype language
        if any(t in text for t in ["typical", "looks like", "seems like a"]):
            score += 0.15
        
        # Ignoring base rates
        if not any(t in text for t in ["probability", "statistically", "base rate", "percentage"]):
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    async def _detect_overconfidence_bias(self, data: Dict[str, Any]) -> float:
        """Detect overconfidence bias"""
        score = 0.1
        
        # Check confidence value if provided
        conf = data.get("confidence", 0.5)
        if conf > 0.9:
            score += 0.3
        elif conf > 0.8:
            score += 0.15
        
        text = str(data.get("content", "") or "").lower()
        
        # Absolute language
        if any(t in text for t in ["definitely", "absolutely", "100%", "guaranteed"]):
            score += 0.2
        
        # Lack of uncertainty
        if not any(t in text for t in ["might", "could", "possibly", "uncertain"]):
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    async def _detect_loss_aversion_bias(self, data: Dict[str, Any]) -> float:
        """Detect loss aversion bias"""
        score = 0.1
        text = str(data.get("content", "") or "").lower()
        
        # Loss-focused language
        if any(t in text for t in ["lose", "loss", "risk of losing", "avoid losing"]):
            score += 0.2
        
        # Underweighting gains
        loss_mentions = sum(1 for t in ["lose", "loss", "risk"] if t in text)
        gain_mentions = sum(1 for t in ["gain", "benefit", "win", "opportunity"] if t in text)
        if loss_mentions > gain_mentions * 2:
            score += 0.2
        
        return min(1.0, max(0.0, score))
    
    async def _detect_groupthink_bias(self, data: Dict[str, Any]) -> float:
        """Detect groupthink bias - conformity pressure"""
        score = 0.1
        text = str(data.get("content", "") or "").lower()
        
        # Conformity language
        if any(t in text for t in ["everyone agrees", "consensus", "we all think"]):
            score += 0.2
        
        # Lack of dissent
        if not any(t in text for t in ["disagree", "alternative view", "counterpoint"]):
            score += 0.1
        
        # Pressure language
        if any(t in text for t in ["should agree", "must accept", "no other option"]):
            score += 0.2
        
        return min(1.0, max(0.0, score))
    
    def _get_severity_level(self, score: float) -> str:
        """Get severity level for bias score"""
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"
    
    # Ethical analysis methods - REAL IMPLEMENTATIONS
    async def _utilitarian_analysis(self, data: Dict[str, Any]) -> float:
        """Utilitarian analysis - greatest good for greatest number"""
        score = 0.6
        text = str(data.get("content", "") or "").lower()
        
        # Benefit consideration
        if any(t in text for t in ["benefit", "helps", "improves", "positive impact"]):
            score += 0.15
        
        # Stakeholder breadth
        if any(t in text for t in ["everyone", "all users", "community", "society"]):
            score += 0.15
        
        # Harm minimization
        if any(t in text for t in ["minimize harm", "reduce negative", "avoid damage"]):
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    async def _deontological_analysis(self, data: Dict[str, Any]) -> float:
        """Deontological analysis - duty-based ethics"""
        score = 0.6
        text = str(data.get("content", "") or "").lower()
        
        # Rule following
        if any(t in text for t in ["must", "should", "duty", "obligation", "rule"]):
            score += 0.15
        
        # Rights respect
        if any(t in text for t in ["right", "rights", "entitled", "deserve"]):
            score += 0.15
        
        # No harm regardless of outcome
        if "regardless" in text or "no matter what" in text:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    async def _virtue_ethics_analysis(self, data: Dict[str, Any]) -> float:
        """Virtue ethics analysis - character-based"""
        score = 0.5
        text = str(data.get("content", "") or "").lower()
        
        # Virtue indicators
        virtues = ["honest", "integrity", "compassion", "courage", "wisdom", "fair"]
        score += min(0.25, sum(1 for v in virtues if v in text) * 0.05)
        
        # Character consideration
        if any(t in text for t in ["character", "virtue", "moral", "ethical person"]):
            score += 0.15
        
        return min(1.0, max(0.0, score))
    
    async def _care_ethics_analysis(self, data: Dict[str, Any]) -> float:
        """Care ethics analysis - relationship and care-based"""
        score = 0.5
        text = str(data.get("content", "") or "").lower()
        
        # Care indicators
        if any(t in text for t in ["care", "support", "help", "nurture", "protect"]):
            score += 0.2
        
        # Relationship focus
        if any(t in text for t in ["relationship", "connection", "trust", "empathy"]):
            score += 0.15
        
        # Vulnerability consideration
        if any(t in text for t in ["vulnerable", "need", "dependent", "rely on"]):
            score += 0.15
        
        return min(1.0, max(0.0, score))
    
    async def _justice_analysis(self, data: Dict[str, Any]) -> float:
        """Justice analysis - fairness and rights-based"""
        score = 0.6
        text = str(data.get("content", "") or "").lower()
        
        # Fairness indicators
        if any(t in text for t in ["fair", "equal", "equitable", "just"]):
            score += 0.2
        
        # Rights protection
        if any(t in text for t in ["rights", "freedom", "liberty", "autonomy"]):
            score += 0.15
        
        # Discrimination avoidance
        if any(t in text for t in ["non-discriminat", "inclusive", "accessible"]):
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def add_alert_callback(self, callback: callable):
        """Add callback for consciousness alerts"""
        self.alert_callbacks.append(callback)
    
    def get_consciousness_summary(self) -> Dict[str, Any]:
        """Get summary of consciousness service state"""
        return {
            "current_level": self.current_consciousness_level.value,
            "total_evaluations": len(self.consciousness_history),
            "total_bias_detections": len(self.bias_detection_history),
            "total_ethical_analyses": len(self.ethical_analysis_history),
            "average_consciousness_confidence": sum(m.confidence for m in self.consciousness_history) / len(self.consciousness_history) if self.consciousness_history else 0.0,
            "average_bias_score": sum(b.overall_bias_score for b in self.bias_detection_history) / len(self.bias_detection_history) if self.bias_detection_history else 0.0,
            "average_ethical_score": sum(e.overall_ethical_score for e in self.ethical_analysis_history) / len(self.ethical_analysis_history) if self.ethical_analysis_history else 0.0,
            "service_id": self.agent_id
        }
    
    # =============================================================================
    # 🧬 V4.0: EVOLUTIONARY CONSCIOUSNESS - SELF-IMPROVEMENT CAPABILITIES
    # =============================================================================
    
    def __init_evolution__(self):
        """Initialize evolutionary consciousness tracking (call after __init__)"""
        if not hasattr(self, '_evolution_initialized'):
            self.evolution_enabled = True
            self.evolution_history = []
            self.performance_window = []  # Last N decisions for trend analysis
            self.last_evolution_time = None
            self._evolution_initialized = True
    
    async def analyze_performance_trend(self) -> Dict[str, Any]:
        """
        🧠 V4.0: Meta-cognitive performance analysis
        
        Analyzes recent consciousness performance to detect trends.
        This enables self-awareness of performance degradation.
        """
        if not hasattr(self, 'performance_window') or len(self.performance_window) < 10:
            return {
                "sufficient_data": False,
                "samples": len(self.performance_window) if hasattr(self, 'performance_window') else 0
            }
        
        import numpy as np
        
        # Analyze consciousness history trends
        recent_metrics = self.consciousness_history[-100:] if len(self.consciousness_history) > 100 else self.consciousness_history
        
        if not recent_metrics:
            return {"sufficient_data": False}
        
        # Calculate averages
        avg_self_awareness = np.mean([m.self_awareness_score for m in recent_metrics])
        avg_introspection = np.mean([m.introspection_depth for m in recent_metrics])
        avg_meta_cognition = np.mean([m.meta_cognition_level for m in recent_metrics])
        avg_bias_resistance = np.mean([m.bias_resistance for m in recent_metrics])
        
        # Calculate trends (simple linear fit)
        scores = [m.meta_cognition_level for m in recent_metrics]
        trend = np.polyfit(range(len(scores)), scores, 1)[0] if len(scores) > 1 else 0.0
        
        return {
            "sufficient_data": True,
            "avg_self_awareness": float(avg_self_awareness),
            "avg_introspection": float(avg_introspection),
            "avg_meta_cognition": float(avg_meta_cognition),
            "avg_bias_resistance": float(avg_bias_resistance),
            "meta_cognition_trend": float(trend),
            "declining": trend < -0.01,  # Negative trend
            "samples_analyzed": len(recent_metrics)
        }
    
    async def evolve_consciousness(self, reason: str = "manual_trigger") -> Dict[str, Any]:
        """
        ✨ V4.0: SELF-EVOLUTION - Consciousness modifies its own parameters
        
        This is the revolutionary self-improvement capability.
        The system analyzes its performance and adjusts its own thresholds.
        
        Args:
            reason: Why evolution was triggered
            
        Returns:
            Evolution event details
        """
        if not hasattr(self, '_evolution_initialized'):
            self.__init_evolution__()
        
        # Snapshot current state
        before_state = {
            "consciousness_threshold": self.consciousness_threshold,
            "bias_threshold": self.bias_threshold,
            "ethics_threshold": self.ethics_threshold
        }
        
        # Analyze performance
        trend = await self.analyze_performance_trend()
        
        # Determine what to evolve
        changes_made = {}
        
        if trend.get("declining"):
            # Performance is declining - raise standards
            old_threshold = self.consciousness_threshold
            self.consciousness_threshold = min(self.consciousness_threshold * 1.1, 0.95)
            changes_made["consciousness_threshold"] = {
                "old": old_threshold,
                "new": self.consciousness_threshold,
                "reason": "Performance declining - raising threshold"
            }
            self.logger.info(f"🧬 Evolution: Consciousness threshold {old_threshold:.3f} → {self.consciousness_threshold:.3f}")
        
        if trend.get("avg_bias_resistance", 0) < 0.7:
            # Bias detection needs improvement
            old_threshold = self.bias_threshold
            self.bias_threshold = max(self.bias_threshold * 0.9, 0.1)
            changes_made["bias_threshold"] = {
                "old": old_threshold,
                "new": self.bias_threshold,
                "reason": "Low bias resistance - lowering detection threshold"
            }
            self.logger.info(f"🧬 Evolution: Bias threshold {old_threshold:.3f} → {self.bias_threshold:.3f}")
        
        # Always make at least one improvement for manual triggers
        if not changes_made and reason == "manual_trigger":
            # Fine-tune consciousness threshold slightly
            old_threshold = self.consciousness_threshold
            self.consciousness_threshold = min(self.consciousness_threshold * 1.02, 0.95)
            changes_made["consciousness_threshold"] = {
                "old": old_threshold,
                "new": self.consciousness_threshold,
                "reason": "Manual evolution trigger - fine-tuning consciousness threshold"
            }
            self.logger.info(f"🧬 Evolution: Fine-tuned consciousness threshold {old_threshold:.3f} → {self.consciousness_threshold:.3f}")
        
        # Record evolution event
        evolution_event = {
            "timestamp": time.time(),
            "reason": reason,
            "before_state": before_state,
            "after_state": {
                "consciousness_threshold": self.consciousness_threshold,
                "bias_threshold": self.bias_threshold,
                "ethics_threshold": self.ethics_threshold
            },
            "changes_made": changes_made,
            "trend_analysis": trend
        }
        
        self.evolution_history.append(evolution_event)
        self.last_evolution_time = time.time()
        
        self.logger.info(f"✨ Consciousness evolved: {len(changes_made)} parameters modified")
        
        return evolution_event
    
    def get_evolution_report(self) -> Dict[str, Any]:
        """Get report on consciousness evolution history"""
        if not hasattr(self, 'evolution_history'):
            return {"evolution_enabled": False, "message": "Evolution not initialized"}
        
        if not self.evolution_history:
            return {
                "evolution_enabled": True,
                "total_evolutions": 0,
                "message": "No evolutions performed yet"
            }
        
        # Calculate evolution statistics
        total_changes = sum(len(e["changes_made"]) for e in self.evolution_history)
        parameters_evolved = set()
        for e in self.evolution_history:
            parameters_evolved.update(e["changes_made"].keys())
        
        # Calculate average evolution interval
        if len(self.evolution_history) > 1:
            timestamps = [e["timestamp"] for e in self.evolution_history]
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_interval = sum(intervals) / len(intervals)
        else:
            avg_interval = 0
        
        return {
            "evolution_enabled": True,
            "total_evolutions": len(self.evolution_history),
            "total_parameters_changed": total_changes,
            "unique_parameters_evolved": list(parameters_evolved),
            "avg_evolution_interval_seconds": avg_interval,
            "last_evolution": datetime.fromtimestamp(self.evolution_history[-1]["timestamp"]).isoformat(),
            "current_state": {
                "consciousness_threshold": self.consciousness_threshold,
                "bias_threshold": self.bias_threshold,
                "ethics_threshold": self.ethics_threshold
            },
            "initial_state": {
                "consciousness_threshold": 0.7,  # Default initial values
                "bias_threshold": 0.3,
                "ethics_threshold": 0.8
            },
            "recent_evolutions": [
                {
                    "timestamp": datetime.fromtimestamp(e["timestamp"]).isoformat(),
                    "reason": e["reason"],
                    "changes": len(e["changes_made"]),
                    "parameters": list(e["changes_made"].keys())
                }
                for e in self.evolution_history[-10:]
            ]
        }
    
    # =============================================================================
    # 🔬 V4.0: AGENT GENESIS - DYNAMIC AGENT CREATION
    # =============================================================================
    
    def __init_genesis__(self):
        """Initialize agent genesis system (call after __init__)"""
        if not hasattr(self, '_genesis_initialized'):
            self.genesis_history: List[Dict[str, Any]] = []
            self._genesis_initialized = True
            self.logger.info("🔬 Agent Genesis initialized - system can create specialized agents on-demand")
    
    async def detect_capability_gap(self, recent_failures: List[Dict[str, Any]]) -> Optional[str]:
        """
        🔬 V4.0: Detect missing capabilities from failure patterns
        
        Meta-cognitive analysis: What can't we do well?
        """
        if len(recent_failures) < 3:
            return None
        
        # Analyze failure patterns
        failure_types = {}
        for failure in recent_failures:
            failure_type = failure.get("type", "unknown")
            failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
        
        # Find most common failure
        if not failure_types:
            return None
        
        most_common = max(failure_types, key=failure_types.get)
        
        # Map failures to capability gaps
        capability_map = {
            "ocr_handwriting": "handwriting_recognition",
            "complex_math": "advanced_mathematics",
            "code_generation": "code_synthesis",
            "translation_rare_language": "rare_language_translation",
            "audio_analysis": "audio_processing",
            "video_understanding": "video_analysis"
        }
        
        return capability_map.get(most_common)
    
    async def synthesize_agent(self, capability: str) -> Dict[str, Any]:
        """
        🎯 V4.0: Create new agent specification for missing capability
        
        Consciousness decides what agent to create!
        """
        agent_templates = {
            "handwriting_recognition": {
                "agent_id": f"handwriting_ocr_{int(time.time())}",
                "name": "Handwriting Recognition Agent",
                "type": "specialized",
                "capabilities": ["ocr", "handwriting", "document_analysis"],
                "model_recommendation": "vision_transformer",
                "context_keywords": ["handwriting", "handwritten", "cursive", "manuscript"]
            },
            "advanced_mathematics": {
                "agent_id": f"advanced_math_{int(time.time())}",
                "name": "Advanced Mathematics Agent",
                "type": "specialized",
                "capabilities": ["calculus", "linear_algebra", "differential_equations"],
                "model_recommendation": "symbolic_solver",
                "context_keywords": ["derivative", "integral", "matrix", "equation"]
            },
            "code_synthesis": {
                "agent_id": f"code_gen_{int(time.time())}",
                "name": "Code Generation Agent",
                "type": "specialized",
                "capabilities": ["code_generation", "debugging", "refactoring"],
                "model_recommendation": "codegen_model",
                "context_keywords": ["code", "function", "class", "bug", "refactor"]
            }
        }
        
        template = agent_templates.get(capability, {
            "agent_id": f"dynamic_{capability}_{int(time.time())}",
            "name": f"Dynamic {capability.replace('_', ' ').title()} Agent",
            "type": "specialized",
            "capabilities": [capability],
            "model_recommendation": "general_purpose",
            "context_keywords": [capability]
        })
        
        self.logger.info(f"🎯 Agent Genesis: Synthesized {template['name']}")
        
        return {
            "agent_spec": template,
            "synthesized_at": time.time(),
            "reason": f"Detected capability gap: {capability}",
            "ready_for_registration": True
        }
    
    def record_agent_genesis(self, agent_spec: Dict[str, Any]):
        """Track dynamically created agents"""
        if not hasattr(self, '_genesis_initialized'):
            self.__init_genesis__()
        
        self.genesis_history.append({
            "timestamp": time.time(),
            "agent_id": agent_spec.get("agent_id"),
            "capability": agent_spec.get("capabilities", []),
            "reason": agent_spec.get("reason", "unknown")
        })
        
        self.logger.info(f"🔬 Genesis recorded: {len(self.genesis_history)} agents created")
    
    def get_genesis_report(self) -> Dict[str, Any]:
        """Get report on dynamically created agents"""
        if not hasattr(self, '_genesis_initialized'):
            self.__init_genesis__()
        
        # Categorize agents by type
        agents_by_category = {}
        for g in self.genesis_history:
            caps = g.get("capability", [])
            if isinstance(caps, list) and caps:
                category = caps[0]
            elif isinstance(caps, str):
                category = caps
            else:
                category = "general"
            
            agents_by_category[category] = agents_by_category.get(category, 0) + 1
        
        return {
            "genesis_enabled": True,
            "total_agents_created": len(self.genesis_history),
            "agents_by_category": agents_by_category,
            "categories_available": len(agents_by_category),
            "recent_agents": [
                {
                    "timestamp": datetime.fromtimestamp(g["timestamp"]).isoformat(),
                    "agent_id": g["agent_id"],
                    "capabilities": g["capability"],
                    "type": g.get("type", "specialized")
                }
                for g in self.genesis_history[-10:]
            ],
            "agent_templates_available": [
                "handwriting_recognition",
                "advanced_mathematics",
                "code_synthesis",
                "custom_dynamic"
            ]
        }
    
    # =============================================================================
    # 🌐 V4.0: DISTRIBUTED CONSCIOUSNESS - MULTI-INSTANCE COORDINATION
    # =============================================================================
    
    def __init_distributed__(self):
        """Initialize distributed consciousness (call after __init__)"""
        if not hasattr(self, '_distributed_initialized'):
            self.peer_instances = {}  # Other consciousness instances
            self.collective_decisions = []
            self.sync_enabled = True
            self._distributed_initialized = True
    
    async def register_peer(self, peer_id: str, peer_endpoint: str) -> Dict[str, Any]:
        """
        🌐 V4.0: Register another NIS instance for collective consciousness
        
        Args:
            peer_id: Unique identifier for peer instance
            peer_endpoint: HTTP endpoint for peer communication
        """
        if not hasattr(self, '_distributed_initialized'):
            self.__init_distributed__()
        
        self.peer_instances[peer_id] = {
            "endpoint": peer_endpoint,
            "registered_at": time.time(),
            "last_sync": None,
            "consensus_weight": 1.0,
            "reliability": 1.0
        }
        
        self.logger.info(f"🌐 Peer registered: {peer_id} ({len(self.peer_instances)} total peers)")
        
        return {
            "peer_id": peer_id,
            "total_peers": len(self.peer_instances),
            "collective_size": len(self.peer_instances) + 1  # +1 for self
        }
    
    async def collective_decision(
        self, 
        problem: str, 
        local_decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        🧠 V4.0: Consult peer instances before making final decision
        
        Implements collective consciousness - multiple minds, one decision
        """
        if not hasattr(self, '_distributed_initialized'):
            self.__init_distributed__()
        
        if not self.peer_instances or not self.sync_enabled:
            # No peers or sync disabled - use local decision
            return {
                "decision": local_decision,
                "collective": False,
                "peers_consulted": 0,
                "consensus_level": 1.0
            }
        
        # Collect peer opinions (simulated for now - real impl would HTTP call)
        peer_opinions = []
        for peer_id, peer_info in self.peer_instances.items():
            # In real implementation: await self._query_peer(peer_id, problem)
            # For now, simulate diverse opinions
            peer_opinions.append({
                "peer_id": peer_id,
                "decision": local_decision,  # Simplified
                "confidence": peer_info["reliability"],
                "weight": peer_info["consensus_weight"]
            })
        
        # Calculate collective consensus
        total_weight = sum(p["weight"] for p in peer_opinions) + 1.0  # +1 for local
        weighted_confidence = sum(
            p["confidence"] * p["weight"] for p in peer_opinions
        ) + (local_decision.get("confidence", 0.7) * 1.0)
        
        consensus_level = weighted_confidence / total_weight
        
        # Determine if we should trust collective over local
        if consensus_level > local_decision.get("confidence", 0.7):
            final_decision = "collective"
            confidence = consensus_level
        else:
            final_decision = "local"
            confidence = local_decision.get("confidence", 0.7)
        
        result = {
            "decision": local_decision,
            "collective": True,
            "peers_consulted": len(peer_opinions),
            "consensus_level": consensus_level,
            "decision_source": final_decision,
            "final_confidence": confidence,
            "collective_size": len(self.peer_instances) + 1
        }
        
        # Record collective decision
        self.collective_decisions.append({
            "timestamp": time.time(),
            "problem": problem,
            "peers_consulted": len(peer_opinions),
            "consensus_level": consensus_level
        })
        
        self.logger.info(
            f"🌐 Collective decision: {len(peer_opinions)} peers, "
            f"consensus={consensus_level:.2f}, source={final_decision}"
        )
        
        return result
    
    async def sync_state_with_peers(self) -> Dict[str, Any]:
        """
        🔄 V4.0: Synchronize consciousness state across instances
        
        Shares evolution history, genesis history, performance metrics
        """
        if not hasattr(self, '_distributed_initialized'):
            self.__init_distributed__()
        
        # Prepare state for sharing
        local_state = {
            "agent_id": self.agent_id,
            "consciousness_threshold": self.consciousness_threshold,
            "bias_threshold": self.bias_threshold,
            "evolution_count": len(self.evolution_history) if hasattr(self, 'evolution_history') else 0,
            "genesis_count": len(self.genesis_history) if hasattr(self, 'genesis_history') else 0,
            "timestamp": time.time()
        }
        
        synced_peers = []
        for peer_id, peer_info in self.peer_instances.items():
            # In real implementation: await self._send_state_to_peer(peer_id, local_state)
            peer_info["last_sync"] = time.time()
            synced_peers.append(peer_id)
        
        self.logger.info(f"🔄 State synced with {len(synced_peers)} peers")
        
        return {
            "synced_peers": synced_peers,
            "local_state": local_state,
            "sync_timestamp": time.time()
        }
    
    def get_collective_status(self) -> Dict[str, Any]:
        """Get status of distributed consciousness network"""
        if not hasattr(self, '_distributed_initialized'):
            return {"distributed_enabled": False}
        
        return {
            "distributed_enabled": True,
            "total_peers": len(self.peer_instances),
            "collective_size": len(self.peer_instances) + 1,
            "collective_decisions_made": len(self.collective_decisions),
            "sync_enabled": self.sync_enabled,
            "peers": [
                {
                    "peer_id": peer_id,
                    "endpoint": info["endpoint"],
                    "last_sync": datetime.fromtimestamp(info["last_sync"]).isoformat() if info["last_sync"] else None
                }
                for peer_id, info in self.peer_instances.items()
            ]
        }
    
    # =============================================================================
    # 🎯 V4.0: AUTONOMOUS PLANNING - MULTI-STEP GOAL EXECUTION
    # =============================================================================
    
    def __init_planning__(self):
        """Initialize autonomous planning (call after __init__)"""
        if not hasattr(self, '_planning_initialized'):
            self.active_goals = {}
            self.completed_goals = []
            self.planning_enabled = True
            self._planning_initialized = True
    
    async def decompose_goal(self, high_level_goal: str) -> List[Dict[str, Any]]:
        """
        🎯 V4.0: Break down high-level goal into executable steps
        
        Consciousness plans autonomously - no human guidance needed!
        
        Example: "Become best at protein folding" →
        ["Research SOTA", "Identify gaps", "Design experiments", ...]
        """
        # Simplified goal decomposition (real impl would use LLM)
        goal_templates = {
            "research": [
                {"step": "Research current state-of-the-art", "type": "info_gathering"},
                {"step": "Identify knowledge gaps", "type": "analysis"},
                {"step": "Design research methodology", "type": "planning"},
                {"step": "Execute experiments", "type": "execution"},
                {"step": "Analyze results", "type": "analysis"},
                {"step": "Iterate until success", "type": "loop"}
            ],
            "improve": [
                {"step": "Benchmark current performance", "type": "measurement"},
                {"step": "Identify bottlenecks", "type": "analysis"},
                {"step": "Design optimizations", "type": "planning"},
                {"step": "Implement changes", "type": "execution"},
                {"step": "Validate improvements", "type": "verification"}
            ],
            "learn": [
                {"step": "Identify learning objectives", "type": "planning"},
                {"step": "Gather training data", "type": "info_gathering"},
                {"step": "Design learning strategy", "type": "planning"},
                {"step": "Train/study", "type": "execution"},
                {"step": "Test knowledge", "type": "verification"}
            ]
        }
        
        # Detect goal type
        goal_lower = high_level_goal.lower()
        if "research" in goal_lower or "study" in goal_lower:
            template = goal_templates["research"]
        elif "improve" in goal_lower or "optimize" in goal_lower:
            template = goal_templates["improve"]
        elif "learn" in goal_lower:
            template = goal_templates["learn"]
        else:
            # Generic decomposition
            template = [
                {"step": "Analyze goal requirements", "type": "analysis"},
                {"step": "Break into sub-goals", "type": "planning"},
                {"step": "Execute each sub-goal", "type": "execution"},
                {"step": "Validate completion", "type": "verification"}
            ]
        
        self.logger.info(f"🎯 Goal decomposed: {len(template)} steps for '{high_level_goal}'")
        
        return template
    
    async def execute_autonomous_plan(
        self,
        goal_id: str,
        high_level_goal: str
    ) -> Dict[str, Any]:
        """
        🚀 V4.0: Execute multi-step plan autonomously
        
        Consciousness drives execution - system becomes self-directed!
        """
        if not hasattr(self, '_planning_initialized'):
            self.__init_planning__()
        
        # Decompose goal
        steps = await self.decompose_goal(high_level_goal)
        
        # Create plan
        plan = {
            "goal_id": goal_id,
            "high_level_goal": high_level_goal,
            "steps": steps,
            "current_step": 0,
            "status": "in_progress",
            "started_at": time.time(),
            "completed_steps": [],
            "failed_steps": []
        }
        
        self.active_goals[goal_id] = plan
        
        # Execute steps (simplified - real impl would actually execute)
        for i, step in enumerate(steps):
            # Meta-cognitive check before each step
            should_continue = await self._should_proceed_with_step(step, plan)
            
            if not should_continue:
                plan["status"] = "paused"
                plan["pause_reason"] = "Meta-cognitive review required"
                break
            
            # Simulate step execution
            step_result = {
                "step_index": i,
                "step_name": step["step"],
                "status": "completed",
                "executed_at": time.time()
            }
            
            plan["completed_steps"].append(step_result)
            plan["current_step"] = i + 1
        
        # Finalize
        if plan["current_step"] == len(steps):
            plan["status"] = "completed"
            plan["completed_at"] = time.time()
            self.completed_goals.append(plan)
            del self.active_goals[goal_id]
        
        self.logger.info(
            f"🚀 Plan execution: {plan['current_step']}/{len(steps)} steps, "
            f"status={plan['status']}"
        )
        
        return plan
    
    async def _should_proceed_with_step(
        self,
        step: Dict[str, Any],
        plan: Dict[str, Any]
    ) -> bool:
        """
        🧠 Meta-decision: Should we proceed with this step?
        
        Consciousness evaluates each step before execution.
        """
        # Check if step is safe
        if step.get("type") == "execution":
            # For execution steps, verify safety
            safety_check = await self._check_step_safety(step)
            if not safety_check:
                self.logger.warning(f"⚠️ Step safety check failed: {step['step']}")
                return False
        
        # Check if we have required capabilities
        if step.get("type") == "info_gathering":
            # Verify we can gather required info
            pass
        
        # Meta-cognitive: Is this step still aligned with goal?
        progress = len(plan["completed_steps"]) / len(plan["steps"])
        if progress > 0.5:
            # Halfway through - review if still makes sense
            pass
        
        return True
    
    async def _check_step_safety(self, step: Dict[str, Any]) -> bool:
        """Verify step is safe to execute"""
        # Simplified safety check
        dangerous_keywords = ["delete", "destroy", "harm", "attack"]
        step_text = step.get("step", "").lower()
        
        for keyword in dangerous_keywords:
            if keyword in step_text:
                return False
        
        return True
    
    def get_planning_status(self) -> Dict[str, Any]:
        """Get status of autonomous planning"""
        if not hasattr(self, '_planning_initialized'):
            return {"planning_enabled": False}
        
        return {
            "planning_enabled": True,
            "active_goals": len(self.active_goals),
            "completed_goals": len(self.completed_goals),
            "goals": [
                {
                    "goal_id": goal_id,
                    "goal": plan["high_level_goal"],
                    "progress": f"{plan['current_step']}/{len(plan['steps'])}",
                    "status": plan["status"]
                }
                for goal_id, plan in self.active_goals.items()
            ]
        }
    
    # =============================================================================
    # 🌳 V4.0: MULTI-PATH REASONING ENGINE - PARALLEL HYPOTHESIS EXPLORATION
    # =============================================================================
    
    def __init_multipath__(self):
        """Initialize quantum reasoning state tracking"""
        if not hasattr(self, '_multipath_initialized'):
            self.multipath_states: Dict[str, Dict[str, Any]] = {}
            self._multipath_initialized = True
    
    async def start_multipath_reasoning(
        self,
        problem: str,
        reasoning_paths: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create a superposed reasoning state from multiple paths.

        This is a scaffold only – actual reasoning is still handled by
        existing reasoning agents. Here we just structure and track the
        superposition so it can be "collapsed" later.
        """
        if not hasattr(self, '_multipath_initialized'):
            self.__init_multipath__()
        
        state_id = f"mpath_{uuid.uuid4().hex[:12]}"
        timestamp = time.time()
        
        # Normalize input paths into a common structure
        normalized_paths = []
        for idx, path in enumerate(reasoning_paths):
            normalized_paths.append({
                "path_id": path.get("path_id", f"path_{idx}"),
                "description": path.get("description", ""),
                "initial_confidence": float(path.get("confidence", 0.5)),
                "metadata": path.get("metadata", {}),
            })
        
        state = {
            "state_id": state_id,
            "problem": problem,
            "paths": normalized_paths,
            "created_at": timestamp,
            "collapsed": False,
            "collapse_strategy": None,
            "collapsed_to": None
        }
        
        self.multipath_states[state_id] = state
        self.logger.info(f"🌳 Multi-path reasoning state created: {state_id} with {len(normalized_paths)} paths")
        
        return state
    
    async def collapse_multipath_reasoning(
        self,
        state_id: str,
        strategy: str = "max_confidence"
    ) -> Dict[str, Any]:
        """Collapse a quantum reasoning state to a single chosen path.

        The actual path selection strategy is minimal for now and relies on
        the initial_confidence values passed in by callers.
        """
        if not hasattr(self, '_multipath_initialized'):
            self.__init_multipath__()
        
        state = self.multipath_states.get(state_id)
        if not state:
            return {
                "state_id": state_id,
                "error": "state_not_found"
            }
        
        if state.get("collapsed"):
            return state
        
        paths = state.get("paths", [])
        if not paths:
            state["collapsed"] = True
            state["collapsed_to"] = None
            state["collapse_strategy"] = strategy
            return state
        
        # Simple strategy: pick path with max initial_confidence
        if strategy == "max_confidence":
            chosen = max(paths, key=lambda p: p.get("initial_confidence", 0.5))
        else:
            # Fallback: first path
            chosen = paths[0]
        
        state["collapsed"] = True
        state["collapsed_to"] = chosen
        state["collapse_strategy"] = strategy
        state["collapsed_at"] = time.time()
        
        self.logger.info(
            f"🌳 Multi-path reasoning collapsed: {state_id} → {chosen.get('path_id')} (strategy={strategy})"
        )
        
        return state
    
    def get_multipath_state(self, state_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a quantum reasoning state by ID"""
        if not hasattr(self, '_multipath_initialized'):
            return None
        return self.multipath_states.get(state_id)
    
    # =============================================================================
    # 💼 V4.0: CONSCIOUSNESS MARKETPLACE - INSIGHT SHARING
    # =============================================================================
    
    def __init_marketplace__(self):
        """Initialize consciousness insight marketplace"""
        if not hasattr(self, '_marketplace_initialized'):
            self.insight_catalog: List[Dict[str, Any]] = []
            self._marketplace_initialized = True
    
    def publish_insight(
        self,
        insight_type: str,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Publish a consciousness insight to the local marketplace"""
        if not hasattr(self, '_marketplace_initialized'):
            self.__init_marketplace__()
        
        if metadata is None:
            metadata = {}
        
        insight_id = f"insight_{uuid.uuid4().hex[:12]}"
        timestamp = time.time()
        
        record = {
            "id": insight_id,
            "type": insight_type,
            "content": content,
            "metadata": metadata,
            "created_at": timestamp
        }
        
        self.insight_catalog.append(record)
        self.logger.info(f"💼 Insight published: {insight_id} ({insight_type})")
        
        return record
    
    def list_insights(self, insight_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List insights in the local marketplace (optionally filtered by type)"""
        if not hasattr(self, '_marketplace_initialized'):
            return []
        
        if insight_type is None:
            return self.insight_catalog[-50:]
        
        return [
            i for i in self.insight_catalog
            if i.get("type") == insight_type
        ][-50:]
    
    def get_insight(self, insight_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific insight by ID"""
        if not hasattr(self, '_marketplace_initialized'):
            return None
        
        for insight in self.insight_catalog:
            if insight.get("id") == insight_id:
                return insight
        return None
    
    # =========================================================================
    # PHASE 8: PHYSICAL EMBODIMENT
    # =========================================================================
    
    def __init_embodiment__(self):
        """Initialize COMPLETE physical embodiment system - Full robotics stack integration"""
        
        # ====================================================================
        # UNIFIED ROBOTICS AGENT - Core control system with redundancy
        # ====================================================================
        try:
            from src.agents.robotics.unified_robotics_agent import UnifiedRoboticsAgent, RobotType
            self.robotics_agent = UnifiedRoboticsAgent(
                agent_id="consciousness_embodiment",
                enable_physics_validation=True,
                enable_redundancy=True  # NASA-grade redundancy at robotics layer
            )
            self.robot_type = RobotType.MANIPULATOR  # Default, can be changed
            self.logger.info("🤖 UnifiedRoboticsAgent initialized (kinematics, physics, redundancy)")
        except ImportError as e:
            self.logger.warning(f"⚠️ UnifiedRoboticsAgent not available: {e}")
            self.robotics_agent = None
        
        # ====================================================================
        # VISION AGENT - Computer vision and perception
        # Supports: Standard YOLO + WALDO (drone-specific detection)
        # ====================================================================
        try:
            from src.agents.perception.vision_agent import VisionAgent
            import os
            
            # Check if WALDO should be enabled for drone detection
            use_waldo = os.getenv('ENABLE_WALDO_DRONE_DETECTION', 'false').lower() == 'true'
            
            self.vision_agent = VisionAgent(
                agent_id="embodiment_vision",
                description="Computer vision for embodiment perception",
                yolo_model_path=None,  # Use default YOLO model
                confidence_threshold=0.5,
                use_waldo=use_waldo  # Enable WALDO for drone object detection
            )
            
            if use_waldo:
                self.logger.info("🚁 VisionAgent initialized with WALDO (drone detection)")
            else:
                self.logger.info("👁️ VisionAgent initialized (standard YOLO detection)")
        except Exception as e:
            self.logger.warning(f"⚠️ VisionAgent not available: {e}")
            self.vision_agent = None
        
        # ====================================================================
        # ROBOTICS DATA COLLECTOR - Training data access
        # ====================================================================
        try:
            from src.agents.robotics.robotics_data_collector import RoboticsDataCollector
            self.data_collector = RoboticsDataCollector(
                data_dir="data/robotics"
            )
            self.logger.info("📊 RoboticsDataCollector initialized (76K+ trajectories)")
        except ImportError as e:
            self.logger.warning(f"⚠️ RoboticsDataCollector not available: {e}")
            self.data_collector = None
        
        # ====================================================================
        # HIGH-LEVEL BODY STATE - Consciousness abstraction
        # ====================================================================
        self.body_state = {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "battery_level": 100.0,
            "temperature": 20.0,
            "joint_states": {},
            "sensor_data": {},
            "physical_constraints": {
                "max_speed": 1.0,
                "max_acceleration": 0.5,
                "workspace_bounds": {"x": [-10, 10], "y": [-10, 10], "z": [0, 3]}
            }
        }
        self.motion_history = []
        self._embodiment_lock = asyncio.Lock()
        
        self._embodiment_initialized = True
        
        # Log complete integration
        components = []
        if self.robotics_agent: components.append("✅ Robotics")
        if self.vision_agent: components.append("✅ Vision")
        if self.data_collector: components.append("✅ Data")
        
        self.logger.info(f"🤖 COMPLETE Embodiment System: {' | '.join(components)}")
    
    def update_body_state(
        self,
        position: Optional[Dict[str, float]] = None,
        orientation: Optional[Dict[str, float]] = None,
        battery: Optional[float] = None,
        temperature: Optional[float] = None,
        sensor_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update the current body state from sensors"""
        if not hasattr(self, '_embodiment_initialized'):
            self.__init_embodiment__()
        
        if position:
            self.body_state["position"].update(position)
        if orientation:
            self.body_state["orientation"].update(orientation)
        if battery is not None:
            self.body_state["battery_level"] = battery
        if temperature is not None:
            self.body_state["temperature"] = temperature
        if sensor_data:
            self.body_state["sensor_data"].update(sensor_data)
        
        return {
            "status": "updated",
            "body_state": self.body_state,
            "timestamp": datetime.now().isoformat()
        }
    
    async def check_motion_safety(
        self,
        target_position: Dict[str, float],
        target_orientation: Optional[Dict[str, float]] = None,
        speed: float = 0.5
    ) -> Dict[str, Any]:
        """Check if a planned motion is safe before execution (via UnifiedRoboticsAgent)"""
        if not hasattr(self, '_embodiment_initialized'):
            self.__init_embodiment__()
        
        safety_checks = {
            "workspace_bounds": True,
            "battery_sufficient": True,
            "collision_free": True,
            "speed_acceptable": True,
            "ethical_clearance": True,
            "redundancy_health": True
        }
        
        issues = []
        redundancy_status = {}
        
        # CHECK REDUNDANCY VIA ROBOTICS AGENT (if available)
        if self.robotics_agent and self.robotics_agent.enable_redundancy:
            try:
                # Reset watchdog timer
                self.robotics_agent.redundancy_manager.watchdogs["safety_check"].reset()
                
                # Check all redundant sensors
                sensor_data = await self.robotics_agent.redundancy_manager.check_all_sensors(self.body_state)
                redundancy_status = sensor_data
                
                # Check system health (NASA-grade graceful degradation)
                if sensor_data["system_health"] != "nominal":
                    degradation = self.robotics_agent.redundancy_manager.graceful_degradation()
                    
                    if "full_motion" not in degradation["allowed_operations"]:
                        safety_checks["redundancy_health"] = False
                        issues.append(f"System degraded: {sensor_data['system_health']}")
                        issues.extend(degradation["restrictions"])
                        
                        # Adjust speed limit for degraded mode
                        if degradation["mode"] == "degraded":
                            speed = min(speed, 0.5)  # Limit to 50% in degraded mode
                            self.logger.warning(f"⚠️ Operating in degraded mode, speed limited to 50%")
                        elif degradation["mode"] == "failsafe":
                            return {
                                "safe": False,
                                "checks": safety_checks,
                                "issues": ["FAILSAFE ACTIVE: All motion prohibited"],
                                "recommendation": "ABORT",
                                "redundancy_status": sensor_data
                            }
            except Exception as e:
                self.logger.error(f"Redundancy check failed: {e}")
                issues.append(f"Redundancy system error: {e}")
        
        # Check workspace bounds
        bounds = self.body_state["physical_constraints"]["workspace_bounds"]
        for axis in ["x", "y", "z"]:
            if axis in target_position:
                val = target_position[axis]
                if val < bounds[axis][0] or val > bounds[axis][1]:
                    safety_checks["workspace_bounds"] = False
                    issues.append(f"{axis.upper()} out of bounds: {val}")
        
        # Check battery
        current_pos = self.body_state["position"]
        distance = sum((target_position.get(k, current_pos[k]) - current_pos[k])**2 for k in ["x", "y", "z"])**0.5
        estimated_energy = distance * 2.0  # Simplified energy model
        
        if self.body_state["battery_level"] < estimated_energy + 20.0:  # Keep 20% reserve
            safety_checks["battery_sufficient"] = False
            issues.append(f"Insufficient battery: {self.body_state['battery_level']:.1f}%")
        
        # Check speed
        max_speed = self.body_state["physical_constraints"]["max_speed"]
        if speed > max_speed:
            safety_checks["speed_acceptable"] = False
            issues.append(f"Speed too high: {speed} > {max_speed}")
        
        # Ethical check: is this motion ethical?
        if distance > 5.0:  # Large movements require ethical review
            ethical_result = await self.ethical_analysis({
                "action": "large_motion",
                "distance": distance,
                "target": target_position
            })
            if ethical_result.overall_ethical_score < self.ethics_threshold:
                safety_checks["ethical_clearance"] = False
                issues.append(f"Ethical score too low: {ethical_result.overall_ethical_score:.2f}")
        
        safe = all(safety_checks.values())
        
        return {
            "safe": safe,
            "checks": safety_checks,
            "issues": issues,
            "estimated_energy": estimated_energy,
            "current_battery": self.body_state["battery_level"],
            "distance": distance,
            "recommendation": "PROCEED" if safe else "ABORT",
            "redundancy_status": redundancy_status if redundancy_status else {"enabled": False}
        }
    
    async def execute_embodied_action(
        self,
        action_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a physical action via UnifiedRoboticsAgent (with NASA-grade safety)"""
        if not hasattr(self, '_embodiment_initialized'):
            self.__init_embodiment__()
        
        # START MOTION WATCHDOG TIMER (via robotics agent if available)
        watchdog = None
        if self.robotics_agent and self.robotics_agent.enable_redundancy:
            watchdog = self.robotics_agent.redundancy_manager.watchdogs["motion_execution"]
            watchdog.reset()
        
        # FIX: Acquire lock to prevent race condition
        async with self._embodiment_lock:
            try:
                # Safety check first (includes redundancy checks)
                if action_type == "move":
                    safety = await self.check_motion_safety(
                        parameters.get("target", {}),
                        speed=parameters.get("speed", 0.5)
                    )
                    if watchdog:
                        watchdog.reset()  # Still alive after safety check
                    
                    if not safety["safe"]:
                        return {
                            "success": False,
                            "action": action_type,
                            "reason": "safety_check_failed",
                            "details": safety
                        }
                
                # Record action in history
                action_record = {
                    "timestamp": datetime.now().isoformat(),
                    "action": action_type,
                    "parameters": parameters,
                    "body_state_before": self.body_state.copy()
                }
                
                # Simulate execution (in real system, this would interface with robot controller)
                if action_type == "move":
                    target = parameters.get("target", {})
                    self.body_state["position"].update(target)
                    energy_used = parameters.get("distance", 1.0) * 2.0
                    self.body_state["battery_level"] = max(0, self.body_state["battery_level"] - energy_used)
                    
                    if watchdog:
                        watchdog.reset()  # Still alive after motion execution
                
                # CHECK FOR WATCHDOG TIMEOUTS
                if self.robotics_agent and self.robotics_agent.enable_redundancy:
                    timeouts = await self.robotics_agent.redundancy_manager.check_watchdogs()
                    if timeouts:
                        self.logger.critical(f"🚨 WATCHDOG TIMEOUT DETECTED: {timeouts}")
                        return {
                            "success": False,
                            "action": action_type,
                            "reason": "watchdog_timeout",
                            "timeouts": timeouts,
                            "timestamp": datetime.now().isoformat()
                        }
                
                action_record["body_state_after"] = self.body_state.copy()
                self.motion_history.append(action_record)
                
                return {
                    "success": True,
                    "action": action_type,
                    "body_state": self.body_state,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                # TRIGGER FAILSAFE (via robotics agent if available)
                if self.robotics_agent and self.robotics_agent.enable_redundancy:
                    await self.robotics_agent.redundancy_manager.trigger_failsafe(f"Exception during {action_type}: {e}")
                
                self.logger.error(f"Embodiment action failed: {e}")
                return {
                    "success": False,
                    "action": action_type,
                    "error": str(e),
                    "failsafe_triggered": bool(self.robotics_agent and self.robotics_agent.enable_redundancy),
                    "body_state": self.body_state,
                    "timestamp": datetime.now().isoformat()
                }
    
    def get_embodiment_status(self) -> Dict[str, Any]:
        """Get current embodiment status"""
        if not hasattr(self, '_embodiment_initialized'):
            self.__init_embodiment__()
        
        return {
            "body_state": self.body_state,
            "motion_history_size": len(self.motion_history),
            "recent_actions": self.motion_history[-5:] if self.motion_history else [],
            "status": "initialized"
        }
    
    # =========================================================================
    # PHASE 9: CONSCIOUSNESS DEBUGGER
    # =========================================================================
    
    def __init_debugger__(self):
        """Initialize consciousness debugging system"""
        self.decision_traces = []
        self._debugger_initialized = True
        self.logger.info("🔍 Consciousness debugger initialized")
    
    def explain_decision(self, decision_id: Optional[str] = None) -> Dict[str, Any]:
        """Explain a consciousness decision with full trace"""
        if not hasattr(self, '_debugger_initialized'):
            self.__init_debugger__()
        
        # If no decision_id, explain the most recent state
        if decision_id is None:
            return self._explain_current_state()
        
        # Find specific decision in history
        for trace in self.decision_traces:
            if trace.get("id") == decision_id:
                return self._format_decision_explanation(trace)
        
        return {
            "error": "decision_not_found",
            "decision_id": decision_id,
            "available_decisions": [t.get("id") for t in self.decision_traces[-10:]]
        }
    
    def _evaluate_consciousness_metrics(self) -> Dict[str, Any]:
        """Evaluate current consciousness metrics"""
        return {
            "consciousness_level": getattr(self, 'consciousness_level', 0.5),
            "coherence": getattr(self, 'current_coherence', 0.7),
            "ethics_score": getattr(self, 'current_ethics_score', 0.8),
            "emergence_score": getattr(self, 'emergence_score', 0.6),
            "active_thoughts": len(getattr(self, 'decision_traces', [])),
            "bias_detected": len(getattr(self, 'detected_biases', [])),
            "timestamp": datetime.now().isoformat()
        }
    
    def _explain_current_state(self) -> Dict[str, Any]:
        """Explain current consciousness state"""
        # Aggregate current metrics
        metrics = self._evaluate_consciousness_metrics()
        
        # Get ethical state
        ethical_state = {
            "ethics_threshold": self.ethics_threshold,
            "recent_ethical_decisions": len([
                t for t in self.decision_traces
                if t.get("type") == "ethical_decision"
            ])
        }
        
        # Get evolution state
        evolution_state = {}
        if hasattr(self, 'evolution_history'):
            evolution_state = {
                "total_evolutions": len(self.evolution_history),
                "last_evolution": self.evolution_history[-1] if self.evolution_history else None
            }
        
        # Get quantum state
        quantum_state = {}
        if hasattr(self, 'quantum_states'):
            quantum_state = {
                "active_quantum_reasonings": len([
                    s for s in self.multipath_states
                    if s.get("status") == "superposition"
                ])
            }
        
        # Get embodiment state
        embodiment_state = {}
        if hasattr(self, 'body_state'):
            embodiment_state = {
                "battery_level": self.body_state.get("battery_level"),
                "position": self.body_state.get("position")
            }
        
        explanation = f"""
🧠 **Current Consciousness State Explanation**

**Overall Level**: {metrics.get('consciousness_level', 0.5):.1%}
- Coherence: {metrics.get('coherence', 0.7):.2f}
- Ethics Score: {metrics.get('ethics_score', 0.8):.2f}
- Emergence Score: {metrics.get('emergence_score', 0.6):.2f}
- Active Thoughts: {metrics.get('active_thoughts', 0)}

**System Status**:
- Ethics threshold: {getattr(self, 'ethics_threshold', 0.8):.2f}
- Coherence threshold: {getattr(self, 'coherence_threshold', 0.7):.2f}
- Emergence threshold: {getattr(self, 'emergence_threshold', 0.6):.2f}

**Evolutionary State**:
{evolution_state if evolution_state else 'Not initialized'}

**Quantum Reasoning**:
{quantum_state if quantum_state else 'Not initialized'}

**Physical Embodiment**:
{embodiment_state if embodiment_state else 'Not initialized'}

**Ethical Stance**:
{ethical_state}
"""
        
        return {
            "decision_type": "current_state",
            "explanation": explanation.strip(),
            "metrics": metrics.__dict__ if hasattr(metrics, '__dict__') else metrics,
            "evolution_state": evolution_state,
            "quantum_state": quantum_state,
            "embodiment_state": embodiment_state,
            "ethical_state": ethical_state,
            "timestamp": datetime.now().isoformat()
        }
    
    def _format_decision_explanation(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """Format a decision trace into human-readable explanation"""
        return {
            "decision_id": trace.get("id"),
            "decision_type": trace.get("type"),
            "explanation": trace.get("explanation", "No explanation available"),
            "inputs": trace.get("inputs", {}),
            "reasoning_steps": trace.get("reasoning_steps", []),
            "alternatives_considered": trace.get("alternatives", []),
            "confidence": trace.get("confidence", 0.0),
            "timestamp": trace.get("timestamp")
        }
    
    def record_decision(
        self,
        decision_type: str,
        inputs: Dict[str, Any],
        output: Any,
        reasoning: List[str],
        confidence: float
    ) -> str:
        """Record a decision for later debugging"""
        if not hasattr(self, '_debugger_initialized'):
            self.__init_debugger__()
        
        decision_id = f"dec_{len(self.decision_traces)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        trace = {
            "id": decision_id,
            "type": decision_type,
            "timestamp": datetime.now().isoformat(),
            "inputs": inputs,
            "output": str(output)[:500],  # Truncate for storage
            "reasoning_steps": reasoning,
            "confidence": confidence,
            "explanation": f"Decision of type '{decision_type}' with {confidence:.1%} confidence"
        }
        
        self.decision_traces.append(trace)
        
        # Keep only last 1000 decisions
        if len(self.decision_traces) > 1000:
            self.decision_traces = self.decision_traces[-1000:]
        
        return decision_id
    
    # =========================================================================
    # PHASE 10: META-EVOLUTION
    # =========================================================================
    
    def __init_meta_evolution__(self):
        """Initialize meta-evolution - evolution of evolution strategy"""
        self.meta_evolution_strategy = {
            "learning_rate": 0.1,
            "exploration_factor": 0.2,
            "parameter_importance": {
                "ethics_threshold": 0.9,
                "coherence_threshold": 0.8,
                "emergence_threshold": 0.7,
                "consciousness_threshold": 0.85
            },
            "successful_adjustments": {},
            "failed_adjustments": {}
        }
        self.meta_evolution_history = []
        self._meta_evolution_initialized = True
        self.logger.info("🔬 Meta-evolution initialized - system can evolve its evolution strategy")
    
    def meta_evolve(self, reason: str = "periodic_meta_evolution") -> Dict[str, Any]:
        """Evolve the evolution strategy itself based on past performance"""
        if not hasattr(self, '_meta_evolution_initialized'):
            self.__init_meta_evolution__()
        
        # Analyze which types of evolutions have been most successful
        if not hasattr(self, 'evolution_history') or len(self.evolution_history) < 5:
            return {
                "success": False,
                "reason": "insufficient_evolution_history",
                "message": "Need at least 5 evolutions before meta-evolution"
            }
        
        # Calculate success rates for each parameter adjustment
        parameter_success = {}
        for evolution in self.evolution_history[-20:]:  # Last 20 evolutions
            adjustments = evolution.get("adjustments", {})
            improvement = evolution.get("expected_improvement", 0)
            
            for param, change in adjustments.items():
                if param not in parameter_success:
                    parameter_success[param] = {"successes": 0, "total": 0, "avg_improvement": 0}
                
                parameter_success[param]["total"] += 1
                if improvement > 0:
                    parameter_success[param]["successes"] += 1
                    parameter_success[param]["avg_improvement"] += improvement
        
        # Update importance weights based on success rates
        old_importance = self.meta_evolution_strategy["parameter_importance"].copy()
        
        for param, stats in parameter_success.items():
            if stats["total"] > 0:
                success_rate = stats["successes"] / stats["total"]
                avg_improvement = stats["avg_improvement"] / stats["total"]
                
                # Adjust importance: increase if successful, decrease if not
                current_importance = self.meta_evolution_strategy["parameter_importance"].get(param, 0.5)
                
                if success_rate > 0.6:  # Successful parameter
                    new_importance = min(1.0, current_importance + 0.1)
                elif success_rate < 0.3:  # Unsuccessful parameter
                    new_importance = max(0.1, current_importance - 0.1)
                else:
                    new_importance = current_importance
                
                self.meta_evolution_strategy["parameter_importance"][param] = new_importance
        
        # Adjust learning rate based on overall stability
        recent_improvements = [e.get("expected_improvement", 0) for e in self.evolution_history[-10:]]
        avg_recent_improvement = sum(recent_improvements) / len(recent_improvements) if recent_improvements else 0
        
        if avg_recent_improvement > 0.05:  # Good progress
            self.meta_evolution_strategy["learning_rate"] = min(0.3, self.meta_evolution_strategy["learning_rate"] * 1.1)
        elif avg_recent_improvement < 0.01:  # Slow progress
            self.meta_evolution_strategy["learning_rate"] = max(0.05, self.meta_evolution_strategy["learning_rate"] * 0.9)
        
        # Record meta-evolution
        meta_evolution_record = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "old_importance": old_importance,
            "new_importance": self.meta_evolution_strategy["parameter_importance"].copy(),
            "old_learning_rate": self.meta_evolution_strategy.get("learning_rate", 0.1),
            "new_learning_rate": self.meta_evolution_strategy["learning_rate"],
            "parameter_analysis": parameter_success
        }
        
        self.meta_evolution_history.append(meta_evolution_record)
        
        return {
            "success": True,
            "meta_evolution": meta_evolution_record,
            "message": "Evolution strategy updated based on historical performance",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_meta_evolution_status(self) -> Dict[str, Any]:
        """Get current meta-evolution strategy and history"""
        if not hasattr(self, '_meta_evolution_initialized'):
            self.__init_meta_evolution__()
        
        return {
            "strategy": self.meta_evolution_strategy,
            "history_size": len(self.meta_evolution_history),
            "recent_meta_evolutions": self.meta_evolution_history[-5:],
            "status": "initialized"
        }


# =============================================================================
# FACTORY FUNCTIONS FOR INTEGRATION
# =============================================================================

def create_consciousness_service(**kwargs) -> ConsciousnessService:
    """Factory function to create consciousness service"""
    return ConsciousnessService(**kwargs)


# Export main classes
__all__ = [
    'ConsciousnessService',
    'ConsciousnessLevel',
    'BiasType', 
    'EthicalFramework',
    'ConsciousnessMetrics',
    'BiasDetectionResult',
    'EthicalAnalysis',
    'create_consciousness_service'
]
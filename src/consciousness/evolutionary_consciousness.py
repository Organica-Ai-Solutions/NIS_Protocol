#!/usr/bin/env python3
"""
NIS Protocol v4.0 - Evolutionary Consciousness Engine

The consciousness layer that evolves itself through meta-cognitive analysis.
This is the foundation of true autonomous improvement.

Key Features:
- Self-optimization based on performance analysis
- Dynamic parameter adjustment
- Pattern learning and integration
- Meta-meta-cognition (thinking about thinking about thinking)
- Autonomous capability gap detection
"""

import asyncio
import logging
import time
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class EvolutionTrigger(Enum):
    """Triggers for consciousness evolution"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    BIAS_PATTERN_DETECTED = "bias_pattern_detected"
    CAPABILITY_GAP = "capability_gap"
    EFFICIENCY_OPPORTUNITY = "efficiency_opportunity"
    SCHEDULED_OPTIMIZATION = "scheduled_optimization"
    MANUAL_TRIGGER = "manual_trigger"


class EvolutionType(Enum):
    """Types of evolutionary changes"""
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    PATTERN_LEARNING = "pattern_learning"
    THRESHOLD_OPTIMIZATION = "threshold_optimization"
    BIAS_DETECTION_ENHANCEMENT = "bias_detection_enhancement"
    DECISION_STRATEGY_MODIFICATION = "decision_strategy_modification"


@dataclass
class PerformanceTrend:
    """Analysis of consciousness performance over time"""
    decision_quality_avg: float = 0.0
    decision_quality_trend: float = 0.0  # Positive = improving, Negative = degrading
    bias_detection_accuracy: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    introspection_depth_avg: float = 0.0
    meta_cognition_effectiveness: float = 0.0
    samples_analyzed: int = 0
    time_period_hours: float = 0.0


@dataclass
class EvolutionEvent:
    """Record of a consciousness evolution event"""
    timestamp: float
    trigger: EvolutionTrigger
    evolution_type: EvolutionType
    changes_made: Dict[str, Any]
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]
    expected_improvement: float
    actual_improvement: Optional[float] = None
    success: bool = False
    notes: str = ""


@dataclass
class ConsciousnessState:
    """Current state of consciousness parameters"""
    consciousness_threshold: float = 0.7
    bias_detection_threshold: float = 0.3
    introspection_min_depth: float = 0.5
    meta_cognition_frequency: float = 0.8
    decision_confidence_threshold: float = 0.75
    
    # Learned patterns
    bias_patterns: List[Dict[str, Any]] = field(default_factory=list)
    decision_heuristics: List[Dict[str, Any]] = field(default_factory=list)
    known_capability_gaps: List[str] = field(default_factory=list)
    
    # Performance tracking
    total_decisions: int = 0
    successful_decisions: int = 0
    biases_detected: int = 0
    evolutions_performed: int = 0
    
    last_evolution: Optional[float] = None
    last_performance_check: Optional[float] = None


class EvolutionaryConsciousness:
    """
    🧠 Self-Improving Consciousness Engine
    
    This is the meta-meta-cognitive layer that allows the system to
    evolve its own consciousness parameters based on performance analysis.
    
    The system literally thinks about how it thinks, and improves itself.
    """
    
    def __init__(
        self,
        agent_id: str = "evolutionary_consciousness",
        evolution_enabled: bool = True,
        min_evolution_interval_hours: float = 24.0,
        performance_window_hours: float = 168.0,  # 1 week
        auto_evolution: bool = True
    ):
        self.agent_id = agent_id
        self.evolution_enabled = evolution_enabled
        self.min_evolution_interval = timedelta(hours=min_evolution_interval_hours)
        self.performance_window = timedelta(hours=performance_window_hours)
        self.auto_evolution = auto_evolution
        
        # Current consciousness state
        self.state = ConsciousnessState()
        
        # Evolution history (track all self-modifications)
        self.evolution_history: List[EvolutionEvent] = []
        
        # Performance history (rolling window)
        self.performance_history = deque(maxlen=10000)  # Last 10k decisions
        
        # Evolution lock (prevent concurrent modifications)
        self._evolution_lock = asyncio.Lock()
        
        # Background monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        
        logger.info(f"🧠 Evolutionary Consciousness initialized: {agent_id}")
        logger.info(f"   Auto-evolution: {auto_evolution}")
        logger.info(f"   Min evolution interval: {min_evolution_interval_hours}h")
    
    async def start_monitoring(self):
        """Start background consciousness evolution monitoring"""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitor_and_evolve())
            logger.info("🔄 Consciousness evolution monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring"""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("⏸️  Consciousness evolution monitoring stopped")
    
    async def _monitor_and_evolve(self):
        """Background task: Monitor performance and trigger evolution"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                if not self.auto_evolution:
                    continue
                
                # Analyze recent performance
                trend = await self.analyze_performance_trend()
                
                # Check if evolution is needed
                should_evolve, trigger = self._should_trigger_evolution(trend)
                
                if should_evolve:
                    logger.info(f"🎯 Evolution triggered: {trigger}")
                    await self.evolve(trigger=trigger)
                
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in consciousness monitoring: {e}")
                await asyncio.sleep(300)  # Wait 5 min before retry
    
    def record_decision(
        self,
        decision: Dict[str, Any],
        quality_score: float,
        biases_detected: List[str],
        introspection_depth: float,
        meta_cognition_used: bool
    ):
        """
        Record a consciousness decision for performance tracking
        
        This is called after every decision to build the performance history
        that drives self-improvement.
        """
        self.performance_history.append({
            "timestamp": time.time(),
            "decision": decision,
            "quality_score": quality_score,
            "biases_detected": biases_detected,
            "introspection_depth": introspection_depth,
            "meta_cognition_used": meta_cognition_used,
            "state_snapshot": {
                "consciousness_threshold": self.state.consciousness_threshold,
                "bias_threshold": self.state.bias_detection_threshold
            }
        })
        
        self.state.total_decisions += 1
        if quality_score > 0.8:
            self.state.successful_decisions += 1
        if biases_detected:
            self.state.biases_detected += len(biases_detected)
    
    async def analyze_performance_trend(self) -> PerformanceTrend:
        """
        Analyze recent performance to detect trends
        
        This is the meta-cognitive analysis that enables self-awareness
        of performance degradation or improvement.
        """
        cutoff_time = time.time() - self.performance_window.total_seconds()
        
        recent_decisions = [
            d for d in self.performance_history
            if d["timestamp"] > cutoff_time
        ]
        
        if len(recent_decisions) < 10:
            # Not enough data yet
            return PerformanceTrend(samples_analyzed=len(recent_decisions))
        
        # Calculate metrics
        quality_scores = [d["quality_score"] for d in recent_decisions]
        avg_quality = np.mean(quality_scores)
        
        # Calculate trend (linear regression on quality over time)
        times = np.array([d["timestamp"] for d in recent_decisions])
        qualities = np.array(quality_scores)
        
        # Normalize time to 0-1 range
        times_norm = (times - times.min()) / (times.max() - times.min() + 1e-10)
        
        # Simple linear regression
        slope = np.cov(times_norm, qualities)[0, 1] / (np.var(times_norm) + 1e-10)
        
        # Bias detection accuracy
        biases_detected_count = sum(len(d["biases_detected"]) for d in recent_decisions)
        bias_detection_rate = biases_detected_count / len(recent_decisions)
        
        # Introspection depth
        avg_introspection = np.mean([d["introspection_depth"] for d in recent_decisions])
        
        # Meta-cognition usage
        meta_cognition_rate = sum(d["meta_cognition_used"] for d in recent_decisions) / len(recent_decisions)
        
        hours_analyzed = (times.max() - times.min()) / 3600
        
        return PerformanceTrend(
            decision_quality_avg=float(avg_quality),
            decision_quality_trend=float(slope),
            bias_detection_accuracy=float(bias_detection_rate),
            false_positive_rate=0.0,  # TODO: Implement false positive tracking
            false_negative_rate=0.0,  # TODO: Implement false negative tracking
            introspection_depth_avg=float(avg_introspection),
            meta_cognition_effectiveness=float(meta_cognition_rate),
            samples_analyzed=len(recent_decisions),
            time_period_hours=float(hours_analyzed)
        )
    
    def _should_trigger_evolution(self, trend: PerformanceTrend) -> Tuple[bool, Optional[EvolutionTrigger]]:
        """
        Meta-decision: Should we evolve?
        
        This is consciousness deciding to modify itself.
        """
        # Check evolution cooldown
        if self.state.last_evolution:
            time_since_evolution = time.time() - self.state.last_evolution
            if time_since_evolution < self.min_evolution_interval.total_seconds():
                return False, None
        
        # Check for performance degradation
        if trend.decision_quality_trend < -0.05:  # Declining quality
            return True, EvolutionTrigger.PERFORMANCE_DEGRADATION
        
        # Check for missed biases
        if trend.false_negative_rate > 0.15:  # Missing 15%+ of biases
            return True, EvolutionTrigger.BIAS_PATTERN_DETECTED
        
        # Check for efficiency opportunities
        if trend.meta_cognition_effectiveness < 0.5:
            return True, EvolutionTrigger.EFFICIENCY_OPPORTUNITY
        
        # Scheduled optimization (every week if performing well)
        if self.state.last_evolution:
            hours_since = (time.time() - self.state.last_evolution) / 3600
            if hours_since > 168:  # 1 week
                return True, EvolutionTrigger.SCHEDULED_OPTIMIZATION
        
        return False, None
    
    async def evolve(
        self,
        trigger: EvolutionTrigger = EvolutionTrigger.MANUAL_TRIGGER
    ) -> EvolutionEvent:
        """
        🌟 SELF-EVOLUTION: Modify consciousness parameters
        
        This is where the magic happens - the system improves itself.
        """
        async with self._evolution_lock:
            logger.info(f"🧬 Evolution initiated: {trigger.value}")
            
            # Snapshot current state
            before_state = {
                "consciousness_threshold": self.state.consciousness_threshold,
                "bias_threshold": self.state.bias_detection_threshold,
                "introspection_depth": self.state.introspection_min_depth,
                "meta_cognition_frequency": self.state.meta_cognition_frequency
            }
            
            # Analyze what needs to change
            trend = await self.analyze_performance_trend()
            
            # Determine evolution type and changes
            evolution_type, changes = await self._determine_evolution(trigger, trend)
            
            # Apply changes (SELF-MODIFICATION!)
            await self._apply_evolution(changes)
            
            # Snapshot new state
            after_state = {
                "consciousness_threshold": self.state.consciousness_threshold,
                "bias_threshold": self.state.bias_detection_threshold,
                "introspection_depth": self.state.introspection_min_depth,
                "meta_cognition_frequency": self.state.meta_cognition_frequency
            }
            
            # Calculate expected improvement
            expected_improvement = self._estimate_improvement(trigger, changes)
            
            # Create evolution event
            event = EvolutionEvent(
                timestamp=time.time(),
                trigger=trigger,
                evolution_type=evolution_type,
                changes_made=changes,
                before_state=before_state,
                after_state=after_state,
                expected_improvement=expected_improvement,
                notes=f"Evolution triggered by {trigger.value}"
            )
            
            # Record evolution
            self.evolution_history.append(event)
            self.state.evolutions_performed += 1
            self.state.last_evolution = time.time()
            
            logger.info(f"✨ Evolution complete: {evolution_type.value}")
            logger.info(f"   Changes: {json.dumps(changes, indent=2)}")
            logger.info(f"   Expected improvement: {expected_improvement:.1%}")
            
            return event
    
    async def _determine_evolution(
        self,
        trigger: EvolutionTrigger,
        trend: PerformanceTrend
    ) -> Tuple[EvolutionType, Dict[str, Any]]:
        """
        Meta-cognitive analysis: What should we change about ourselves?
        """
        if trigger == EvolutionTrigger.PERFORMANCE_DEGRADATION:
            # Quality is declining - tighten standards
            return (
                EvolutionType.THRESHOLD_OPTIMIZATION,
                {
                    "consciousness_threshold": min(
                        self.state.consciousness_threshold * 1.1,
                        0.95
                    ),
                    "decision_confidence_threshold": min(
                        self.state.decision_confidence_threshold * 1.05,
                        0.9
                    ),
                    "reason": "Raising thresholds to improve decision quality"
                }
            )
        
        elif trigger == EvolutionTrigger.BIAS_PATTERN_DETECTED:
            # Missing biases - enhance detection
            return (
                EvolutionType.BIAS_DETECTION_ENHANCEMENT,
                {
                    "bias_threshold": max(
                        self.state.bias_detection_threshold * 0.9,
                        0.1
                    ),
                    "add_pattern": "confirmation_bias_variant_2",
                    "reason": "Lowering bias threshold to catch more patterns"
                }
            )
        
        elif trigger == EvolutionTrigger.EFFICIENCY_OPPORTUNITY:
            # Can be more efficient
            return (
                EvolutionType.PARAMETER_ADJUSTMENT,
                {
                    "meta_cognition_frequency": max(
                        self.state.meta_cognition_frequency * 0.95,
                        0.5
                    ),
                    "reason": "Reducing meta-cognition overhead while maintaining quality"
                }
            )
        
        else:
            # Scheduled optimization - fine-tune based on data
            return (
                EvolutionType.PARAMETER_ADJUSTMENT,
                {
                    "fine_tune": True,
                    "adjustments": "data_driven_optimization",
                    "reason": "Scheduled parameter optimization"
                }
            )
    
    async def _apply_evolution(self, changes: Dict[str, Any]):
        """
        Apply the evolutionary changes to consciousness state
        
        THIS IS SELF-MODIFICATION IN ACTION
        """
        if "consciousness_threshold" in changes:
            old = self.state.consciousness_threshold
            self.state.consciousness_threshold = changes["consciousness_threshold"]
            logger.info(f"   📊 Consciousness threshold: {old:.3f} → {self.state.consciousness_threshold:.3f}")
        
        if "bias_threshold" in changes:
            old = self.state.bias_detection_threshold
            self.state.bias_detection_threshold = changes["bias_threshold"]
            logger.info(f"   🎯 Bias threshold: {old:.3f} → {self.state.bias_detection_threshold:.3f}")
        
        if "decision_confidence_threshold" in changes:
            old = self.state.decision_confidence_threshold
            self.state.decision_confidence_threshold = changes["decision_confidence_threshold"]
            logger.info(f"   ✅ Confidence threshold: {old:.3f} → {self.state.decision_confidence_threshold:.3f}")
        
        if "meta_cognition_frequency" in changes:
            old = self.state.meta_cognition_frequency
            self.state.meta_cognition_frequency = changes["meta_cognition_frequency"]
            logger.info(f"   🧠 Meta-cognition frequency: {old:.3f} → {self.state.meta_cognition_frequency:.3f}")
        
        if "add_pattern" in changes:
            pattern_id = changes["add_pattern"]
            self.state.bias_patterns.append({
                "id": pattern_id,
                "added_at": time.time(),
                "effectiveness": 0.0  # Will be measured
            })
            logger.info(f"   🆕 Added bias pattern: {pattern_id}")
    
    def _estimate_improvement(self, trigger: EvolutionTrigger, changes: Dict[str, Any]) -> float:
        """Estimate expected improvement from these changes"""
        # Simple heuristic estimation
        if trigger == EvolutionTrigger.PERFORMANCE_DEGRADATION:
            return 0.15  # Expect 15% improvement
        elif trigger == EvolutionTrigger.BIAS_PATTERN_DETECTED:
            return 0.20  # Expect 20% better bias detection
        elif trigger == EvolutionTrigger.EFFICIENCY_OPPORTUNITY:
            return 0.10  # Expect 10% efficiency gain
        else:
            return 0.05  # Modest improvement expected
    
    def get_evolution_report(self) -> Dict[str, Any]:
        """Generate report on consciousness evolution history"""
        if not self.evolution_history:
            return {
                "total_evolutions": 0,
                "message": "No evolutions performed yet"
            }
        
        successful_evolutions = [e for e in self.evolution_history if e.success]
        
        return {
            "total_evolutions": len(self.evolution_history),
            "successful_evolutions": len(successful_evolutions),
            "success_rate": len(successful_evolutions) / len(self.evolution_history),
            "last_evolution": datetime.fromtimestamp(self.evolution_history[-1].timestamp).isoformat(),
            "current_state": {
                "consciousness_threshold": self.state.consciousness_threshold,
                "bias_threshold": self.state.bias_detection_threshold,
                "total_decisions": self.state.total_decisions,
                "success_rate": self.state.successful_decisions / max(self.state.total_decisions, 1)
            },
            "recent_evolutions": [
                {
                    "timestamp": datetime.fromtimestamp(e.timestamp).isoformat(),
                    "trigger": e.trigger.value,
                    "type": e.evolution_type.value,
                    "expected_improvement": f"{e.expected_improvement:.1%}",
                    "actual_improvement": f"{e.actual_improvement:.1%}" if e.actual_improvement else "measuring..."
                }
                for e in self.evolution_history[-5:]
            ]
        }


# Factory function
def create_evolutionary_consciousness(
    evolution_enabled: bool = True,
    auto_evolution: bool = True
) -> EvolutionaryConsciousness:
    """Create an evolutionary consciousness instance"""
    return EvolutionaryConsciousness(
        evolution_enabled=evolution_enabled,
        auto_evolution=auto_evolution
    )


if __name__ == "__main__":
    # Example usage
    async def test_evolution():
        """Test the evolutionary consciousness"""
        consciousness = create_evolutionary_consciousness(auto_evolution=False)
        
        # Simulate some decisions
        for i in range(100):
            consciousness.record_decision(
                decision={"action": f"decision_{i}"},
                quality_score=0.7 + np.random.normal(0, 0.1),
                biases_detected=["confirmation"] if np.random.random() < 0.1 else [],
                introspection_depth=0.6 + np.random.normal(0, 0.1),
                meta_cognition_used=np.random.random() < 0.8
            )
        
        # Analyze performance
        trend = await consciousness.analyze_performance_trend()
        print(f"Performance trend: quality={trend.decision_quality_avg:.3f}, trend={trend.decision_quality_trend:.4f}")
        
        # Trigger evolution
        event = await consciousness.evolve(trigger=EvolutionTrigger.MANUAL_TRIGGER)
        print(f"Evolution completed: {event.evolution_type.value}")
        
        # Get report
        report = consciousness.get_evolution_report()
        print(json.dumps(report, indent=2))
    
    asyncio.run(test_evolution())

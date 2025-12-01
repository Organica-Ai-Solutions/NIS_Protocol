"""
Self-Modification System - NIS Protocol v4.0
Allows the system to adapt its own behavior based on performance.
Implements the "Self-Modifying Architecture" concept from Nested Learning.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import hashlib

logger = logging.getLogger("nis.self_modifier")


@dataclass
class Modification:
    """A single modification to the system"""
    id: str
    target: str  # What's being modified (prompt, parameter, behavior)
    modification_type: str  # "prompt_template", "parameter", "routing", "threshold"
    original_value: Any
    new_value: Any
    reason: str
    performance_before: float
    performance_after: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    reverted: bool = False
    success: bool = False


@dataclass
class PerformanceMetric:
    """Track performance over time"""
    metric_name: str
    values: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    
    def add(self, value: float):
        self.values.append(value)
        self.timestamps.append(time.time())
        # Keep last 1000
        if len(self.values) > 1000:
            self.values = self.values[-1000:]
            self.timestamps = self.timestamps[-1000:]
    
    def get_trend(self, window: int = 50) -> float:
        """Get trend: positive = improving, negative = declining"""
        n = len(self.values)
        if n < 2:
            return 0.0
        
        # For small samples, use linear regression slope
        if n < window:
            if n < 3:
                return self.values[-1] - self.values[0]
            # Simple linear regression
            x_mean = (n - 1) / 2
            y_mean = sum(self.values) / n
            numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(self.values))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            if denominator == 0:
                return 0.0
            return numerator / denominator
        
        # For larger samples, compare recent vs older windows
        recent = self.values[-window:]
        older = self.values[-window*2:-window] if n >= window*2 else self.values[:window]
        if not older:
            return 0.0
        return (sum(recent) / len(recent)) - (sum(older) / len(older))
    
    def get_average(self, window: int = 50) -> float:
        if not self.values:
            return 0.0
        recent = self.values[-window:]
        return sum(recent) / len(recent)


class SelfModifier:
    """
    Self-Modification Engine
    
    Monitors system performance and automatically adapts:
    - Prompt templates (improve based on what works)
    - Routing thresholds (optimize query routing)
    - Model parameters (temperature, max_tokens)
    - Behavior patterns (learn from success/failure)
    
    Safety: All modifications are logged and can be reverted.
    """
    
    def __init__(
        self,
        storage_path: str = "data/modifications",
        min_samples_before_modify: int = 20,
        performance_threshold: float = 0.7,
        improvement_threshold: float = 0.05,
        max_modifications_per_hour: int = 5,
        auto_revert_on_decline: bool = True
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.min_samples = min_samples_before_modify
        self.performance_threshold = performance_threshold
        self.improvement_threshold = improvement_threshold
        self.max_modifications_per_hour = max_modifications_per_hour
        self.auto_revert = auto_revert_on_decline
        
        # Track modifications
        self.modifications: List[Modification] = []
        self.modification_count_this_hour = 0
        self.last_hour_reset = time.time()
        
        # Performance tracking
        self.metrics: Dict[str, PerformanceMetric] = {
            "response_quality": PerformanceMetric("response_quality"),
            "user_satisfaction": PerformanceMetric("user_satisfaction"),
            "latency_ms": PerformanceMetric("latency_ms"),
            "reflection_improvement": PerformanceMetric("reflection_improvement"),
            "novelty_detection": PerformanceMetric("novelty_detection")
        }
        
        # Modifiable components
        self.prompt_templates: Dict[str, str] = {}
        self.parameters: Dict[str, Any] = {
            "default_temperature": 0.7,
            "reflection_threshold": 0.75,
            "novelty_threshold": 0.6,
            "max_reflection_iterations": 3,
            "context_window_size": 5
        }
        self.routing_rules: Dict[str, float] = {
            "simple_query_threshold": 0.3,
            "complex_query_threshold": 0.7,
            "reflection_trigger_threshold": 0.6
        }
        
        # Load saved state
        self._load_state()
        
        logger.info(f"🔧 SelfModifier initialized: {len(self.modifications)} historical modifications")
    
    def _generate_id(self) -> str:
        return hashlib.sha256(f"{time.time()}:{len(self.modifications)}".encode()).hexdigest()[:12]
    
    def _load_state(self):
        """Load saved modifications and parameters"""
        state_path = self.storage_path / "self_modifier_state.json"
        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                    self.parameters = state.get("parameters", self.parameters)
                    self.routing_rules = state.get("routing_rules", self.routing_rules)
                    self.prompt_templates = state.get("prompt_templates", {})
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
    
    def _save_state(self):
        """Save current state"""
        state_path = self.storage_path / "self_modifier_state.json"
        try:
            state = {
                "parameters": self.parameters,
                "routing_rules": self.routing_rules,
                "prompt_templates": self.prompt_templates,
                "last_saved": time.time()
            }
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")
    
    def record_metric(self, metric_name: str, value: float):
        """Record a performance metric"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = PerformanceMetric(metric_name)
        self.metrics[metric_name].add(value)
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a modifiable parameter"""
        return self.parameters.get(name, default)
    
    def get_routing_rule(self, name: str, default: float = 0.5) -> float:
        """Get a routing rule threshold"""
        return self.routing_rules.get(name, default)
    
    def get_prompt_template(self, name: str, default: str = "") -> str:
        """Get a prompt template"""
        return self.prompt_templates.get(name, default)
    
    async def propose_modification(
        self,
        target: str,
        modification_type: str,
        new_value: Any,
        reason: str
    ) -> Optional[Modification]:
        """Propose a modification based on performance analysis"""
        # Rate limiting
        if time.time() - self.last_hour_reset > 3600:
            self.modification_count_this_hour = 0
            self.last_hour_reset = time.time()
        
        if self.modification_count_this_hour >= self.max_modifications_per_hour:
            logger.warning("Modification rate limit reached")
            return None
        
        # Get current value
        if modification_type == "parameter":
            original_value = self.parameters.get(target)
        elif modification_type == "routing":
            original_value = self.routing_rules.get(target)
        elif modification_type == "prompt_template":
            original_value = self.prompt_templates.get(target, "")
        else:
            original_value = None
        
        # Get current performance
        main_metric = self.metrics.get("response_quality", PerformanceMetric("response_quality"))
        performance_before = main_metric.get_average()
        
        # Create modification
        mod = Modification(
            id=self._generate_id(),
            target=target,
            modification_type=modification_type,
            original_value=original_value,
            new_value=new_value,
            reason=reason,
            performance_before=performance_before
        )
        
        self.modifications.append(mod)
        self.modification_count_this_hour += 1
        
        logger.info(f"🔧 Proposed modification: {target} ({modification_type})")
        return mod
    
    async def apply_modification(self, mod: Modification) -> bool:
        """Apply a proposed modification"""
        try:
            if mod.modification_type == "parameter":
                self.parameters[mod.target] = mod.new_value
            elif mod.modification_type == "routing":
                self.routing_rules[mod.target] = mod.new_value
            elif mod.modification_type == "prompt_template":
                self.prompt_templates[mod.target] = mod.new_value
            
            self._save_state()
            logger.info(f"✅ Applied modification: {mod.id} - {mod.target}")
            return True
        except Exception as e:
            logger.error(f"Failed to apply modification: {e}")
            return False
    
    async def evaluate_modification(self, mod: Modification, wait_samples: int = 20) -> bool:
        """Evaluate if a modification improved performance"""
        # Wait for enough samples
        main_metric = self.metrics.get("response_quality", PerformanceMetric("response_quality"))
        
        # Get performance after modification
        mod.performance_after = main_metric.get_average()
        
        improvement = (mod.performance_after or 0) - mod.performance_before
        mod.success = improvement >= self.improvement_threshold
        
        if not mod.success and self.auto_revert:
            await self.revert_modification(mod)
            logger.warning(f"⚠️ Modification {mod.id} reverted - no improvement")
        else:
            logger.info(f"✅ Modification {mod.id} successful: +{improvement:.3f}")
        
        return mod.success
    
    async def revert_modification(self, mod: Modification) -> bool:
        """Revert a modification"""
        try:
            if mod.modification_type == "parameter":
                self.parameters[mod.target] = mod.original_value
            elif mod.modification_type == "routing":
                self.routing_rules[mod.target] = mod.original_value
            elif mod.modification_type == "prompt_template":
                if mod.original_value:
                    self.prompt_templates[mod.target] = mod.original_value
                else:
                    self.prompt_templates.pop(mod.target, None)
            
            mod.reverted = True
            self._save_state()
            logger.info(f"↩️ Reverted modification: {mod.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to revert modification: {e}")
            return False
    
    async def auto_optimize(self) -> List[Modification]:
        """Automatically propose optimizations based on performance trends"""
        proposed = []
        
        # Check response quality trend
        quality_metric = self.metrics.get("response_quality")
        if quality_metric and len(quality_metric.values) >= self.min_samples:
            trend = quality_metric.get_trend()
            avg = quality_metric.get_average()
            
            # If quality is declining, try adjustments
            if trend < -0.05 and avg < self.performance_threshold:
                # Try increasing reflection threshold
                current_threshold = self.parameters.get("reflection_threshold", 0.75)
                if current_threshold < 0.9:
                    mod = await self.propose_modification(
                        target="reflection_threshold",
                        modification_type="parameter",
                        new_value=min(current_threshold + 0.05, 0.9),
                        reason=f"Quality declining (trend={trend:.3f}), increasing reflection threshold"
                    )
                    if mod:
                        await self.apply_modification(mod)
                        proposed.append(mod)
        
        # Check latency trend
        latency_metric = self.metrics.get("latency_ms")
        if latency_metric and len(latency_metric.values) >= self.min_samples:
            avg_latency = latency_metric.get_average()
            
            # If latency is too high, reduce iterations
            if avg_latency > 5000:  # 5 seconds
                current_iters = self.parameters.get("max_reflection_iterations", 3)
                if current_iters > 1:
                    mod = await self.propose_modification(
                        target="max_reflection_iterations",
                        modification_type="parameter",
                        new_value=max(current_iters - 1, 1),
                        reason=f"Latency too high ({avg_latency:.0f}ms), reducing iterations"
                    )
                    if mod:
                        await self.apply_modification(mod)
                        proposed.append(mod)
        
        # Check reflection improvement
        reflection_metric = self.metrics.get("reflection_improvement")
        if reflection_metric and len(reflection_metric.values) >= self.min_samples:
            avg_improvement = reflection_metric.get_average()
            
            # If reflection isn't helping much, lower threshold to trigger less
            if avg_improvement < 0.02:
                current_trigger = self.routing_rules.get("reflection_trigger_threshold", 0.6)
                if current_trigger < 0.85:
                    mod = await self.propose_modification(
                        target="reflection_trigger_threshold",
                        modification_type="routing",
                        new_value=min(current_trigger + 0.1, 0.85),
                        reason=f"Reflection not improving much ({avg_improvement:.3f}), raising trigger threshold"
                    )
                    if mod:
                        await self.apply_modification(mod)
                        proposed.append(mod)
        
        return proposed
    
    def get_status(self) -> Dict[str, Any]:
        """Get self-modifier status"""
        return {
            "total_modifications": len(self.modifications),
            "successful_modifications": sum(1 for m in self.modifications if m.success),
            "reverted_modifications": sum(1 for m in self.modifications if m.reverted),
            "modifications_this_hour": self.modification_count_this_hour,
            "current_parameters": self.parameters,
            "current_routing_rules": self.routing_rules,
            "metrics_summary": {
                name: {
                    "average": metric.get_average(),
                    "trend": metric.get_trend(),
                    "samples": len(metric.values)
                }
                for name, metric in self.metrics.items()
            }
        }
    
    def get_modification_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent modification history"""
        recent = self.modifications[-limit:]
        return [
            {
                "id": m.id,
                "target": m.target,
                "type": m.modification_type,
                "reason": m.reason,
                "performance_before": m.performance_before,
                "performance_after": m.performance_after,
                "success": m.success,
                "reverted": m.reverted,
                "timestamp": m.timestamp
            }
            for m in reversed(recent)
        ]


# Global instance
_self_modifier: Optional[SelfModifier] = None


def get_self_modifier() -> SelfModifier:
    """Get or create the global self-modifier"""
    global _self_modifier
    if _self_modifier is None:
        _self_modifier = SelfModifier()
    return _self_modifier

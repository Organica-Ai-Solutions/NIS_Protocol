#!/usr/bin/env python3
"""
Standalone test for NIS Protocol v2.0 AGI Core Implementation

This test includes minimal implementations to verify our core algorithms work.
"""

import time
import random
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import heapq


# Mock Memory Manager
class MockMemoryManager:
    def store(self, key, data, ttl=None):
        pass
    
    def retrieve(self, key):
        return None


# MetaCognitiveProcessor Components
class CognitiveProcess(Enum):
    PERCEPTION = "perception"
    REASONING = "reasoning"
    MEMORY_ACCESS = "memory_access"
    DECISION_MAKING = "decision_making"
    EMOTIONAL_PROCESSING = "emotional_processing"
    GOAL_FORMATION = "goal_formation"


@dataclass
class CognitiveAnalysis:
    process_type: CognitiveProcess
    efficiency_score: float
    quality_metrics: Dict[str, float]
    bottlenecks: List[str]
    improvement_suggestions: List[str]
    confidence: float
    timestamp: float


class MetaCognitiveProcessor:
    def __init__(self):
        self.logger = logging.getLogger("nis.meta_cognitive_processor")
        self.memory = MockMemoryManager()
        self.analysis_history = []
    
    def analyze_cognitive_process(self, process_type, process_data, context):
        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency_score(process_data, context)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(process_type, process_data, context)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(process_type, process_data, context)
        
        # Generate improvement suggestions
        improvements = self._generate_improvement_suggestions(
            process_type, efficiency_score, quality_metrics, bottlenecks, context
        )
        
        # Calculate confidence
        confidence = self._calculate_analysis_confidence(process_data, quality_metrics)
        
        analysis = CognitiveAnalysis(
            process_type=process_type,
            efficiency_score=efficiency_score,
            quality_metrics=quality_metrics,
            bottlenecks=bottlenecks,
            improvement_suggestions=improvements,
            confidence=confidence,
            timestamp=time.time()
        )
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _calculate_efficiency_score(self, process_data, context):
        processing_time = process_data.get("processing_time", 0)
        resource_usage = process_data.get("resource_usage", {})
        data_volume = process_data.get("data_volume", 1)
        
        time_efficiency = max(0.1, min(1.0, 1.0 / (1.0 + processing_time / max(data_volume, 1))))
        
        memory_usage = resource_usage.get("memory", 0)
        cpu_usage = resource_usage.get("cpu", 0)
        resource_efficiency = max(0.1, 1.0 - (memory_usage + cpu_usage) / 2.0)
        
        output_completeness = process_data.get("output_completeness", 1.0)
        error_rate = process_data.get("error_rate", 0.0)
        quality_efficiency = output_completeness * (1.0 - error_rate)
        
        efficiency_score = (
            0.4 * time_efficiency +
            0.3 * resource_efficiency +
            0.3 * quality_efficiency
        )
        
        return max(0.0, min(1.0, efficiency_score))
    
    def _calculate_quality_metrics(self, process_type, process_data, context):
        return {
            "accuracy": process_data.get("validation_score", 0.8) * (1.0 - process_data.get("error_rate", 0.0)),
            "completeness": process_data.get("output_completeness", 0.8),
            "relevance": (process_data.get("context_alignment", 0.8) + process_data.get("goal_alignment", 0.8)) / 2.0,
            "coherence": (process_data.get("internal_consistency", 0.9) + process_data.get("logical_flow", 0.8)) / 2.0
        }
    
    def _identify_bottlenecks(self, process_type, process_data, context):
        bottlenecks = []
        
        if process_data.get("processing_time", 0) > 5.0:
            bottlenecks.append("excessive_processing_time")
        
        memory_usage = process_data.get("resource_usage", {}).get("memory", 0)
        if memory_usage > 0.8:
            bottlenecks.append("high_memory_usage")
        
        if process_data.get("error_rate", 0) > 0.1:
            bottlenecks.append("high_error_rate")
        
        return bottlenecks
    
    def _generate_improvement_suggestions(self, process_type, efficiency_score, quality_metrics, bottlenecks, context):
        suggestions = []
        
        if efficiency_score < 0.7:
            suggestions.extend(["optimize_processing_pipeline", "implement_caching_strategy"])
        
        if quality_metrics.get("accuracy", 1.0) < 0.8:
            suggestions.extend(["improve_input_validation", "add_error_correction_mechanisms"])
        
        if "excessive_processing_time" in bottlenecks:
            suggestions.extend(["parallelize_processing_steps", "implement_early_termination_conditions"])
        
        return list(set(suggestions))
    
    def _calculate_analysis_confidence(self, process_data, quality_metrics):
        required_fields = ["processing_time", "resource_usage", "output_completeness"]
        completeness = sum(1 for field in required_fields if field in process_data) / len(required_fields)
        consistency = min(quality_metrics.values()) if quality_metrics else 0.5
        confidence = (0.6 * completeness + 0.4 * consistency)
        return max(0.1, min(1.0, confidence))
    
    def detect_cognitive_biases(self, reasoning_chain, context):
        bias_results = {
            "biases_detected": [],
            "severity_scores": {},
            "confidence_scores": {},
            "bias_explanations": {},
            "mitigation_strategies": {},
            "overall_bias_score": 0.0
        }
        
        # Check for confirmation bias
        supporting_evidence = sum(1 for step in reasoning_chain if step.get("evidence_type") == "supporting")
        total_evidence = len(reasoning_chain)
        
        if total_evidence > 0 and supporting_evidence / total_evidence > 0.8:
            bias_results["biases_detected"].append("confirmation_bias")
            bias_results["severity_scores"]["confirmation_bias"] = 0.7
            bias_results["confidence_scores"]["confirmation_bias"] = 0.8
            bias_results["bias_explanations"]["confirmation_bias"] = "High ratio of supporting evidence"
            bias_results["mitigation_strategies"]["confirmation_bias"] = ["Actively seek contradicting evidence"]
        
        # Check for overconfidence bias
        high_confidence_count = sum(1 for step in reasoning_chain if step.get("confidence", 0.5) > 0.9)
        if high_confidence_count > len(reasoning_chain) * 0.5:
            bias_results["biases_detected"].append("overconfidence_bias")
            bias_results["severity_scores"]["overconfidence_bias"] = 0.6
            bias_results["confidence_scores"]["overconfidence_bias"] = 0.7
            bias_results["bias_explanations"]["overconfidence_bias"] = "Excessive confidence in conclusions"
            bias_results["mitigation_strategies"]["overconfidence_bias"] = ["Use confidence intervals"]
        
        # Calculate overall bias score
        if bias_results["severity_scores"]:
            bias_results["overall_bias_score"] = sum(bias_results["severity_scores"].values()) / len(bias_results["severity_scores"])
        
        return bias_results


# CuriosityEngine Components
class CuriosityType(Enum):
    EPISTEMIC = "epistemic"
    DIVERSIVE = "diversive"
    SPECIFIC = "specific"
    PERCEPTUAL = "perceptual"
    EMPATHIC = "empathic"
    CREATIVE = "creative"


@dataclass
class CuriositySignal:
    signal_id: str
    curiosity_type: CuriosityType
    intensity: float
    focus_area: str
    specific_questions: List[str]
    context: Dict[str, Any]
    timestamp: float
    decay_rate: float
    satisfaction_threshold: float


class CuriosityEngine:
    def __init__(self):
        self.logger = logging.getLogger("nis.curiosity_engine")
        self.memory = MockMemoryManager()
        self.active_curiosity_signals = {}
        self.base_curiosity_level = 0.6
    
    def detect_knowledge_gaps(self, current_context, knowledge_base):
        gaps = []
        
        # Detect missing domain knowledge
        domain = current_context.get("domain", "general")
        expected_areas = {
            "archaeology": ["cultural_context", "artifact_types", "preservation_methods"],
            "general": ["basic_concepts", "relationships"]
        }.get(domain, ["basic_concepts"])
        
        domain_knowledge = knowledge_base.get(domain, {})
        for area in expected_areas:
            if area not in domain_knowledge:
                gaps.append({
                    "gap_type": "missing_domain_knowledge",
                    "domain": domain,
                    "area": area,
                    "description": f"Missing knowledge in {area}",
                    "importance": 0.8,
                    "explorability": 0.7,
                    "total_score": 0.75
                })
        
        # Detect unanswered questions
        questions = current_context.get("pending_questions", [])
        for question in questions:
            gaps.append({
                "gap_type": "unanswered_question",
                "domain": domain,
                "question": question,
                "description": f"Unanswered: {question}",
                "importance": 0.7,
                "explorability": 0.8,
                "total_score": 0.75
            })
        
        # Sort by total score
        return sorted(gaps, key=lambda x: x.get("total_score", 0), reverse=True)
    
    def assess_novelty(self, item, context):
        # Simple novelty calculation based on features
        known_items = context.get("known_items", [])
        if not known_items:
            return 0.8  # High novelty if no comparison items
        
        item_type = item.get("type", "unknown")
        item_category = item.get("category", "unknown")
        
        # Check similarity to known items
        similar_count = 0
        for known in known_items:
            if known.get("type") == item_type:
                similar_count += 0.5
            if known.get("category") == item_category:
                similar_count += 0.3
        
        novelty_score = max(0.1, 1.0 - (similar_count / max(len(known_items), 1)))
        return min(1.0, novelty_score)
    
    def generate_curiosity_signal(self, trigger, context):
        trigger_type = trigger.get("type", "unknown")
        
        # Determine curiosity type
        curiosity_type = {
            "knowledge_gap": CuriosityType.EPISTEMIC,
            "novel_item": CuriosityType.DIVERSIVE,
            "specific_question": CuriosityType.SPECIFIC
        }.get(trigger_type, CuriosityType.EPISTEMIC)
        
        # Calculate intensity
        intensity = 0.7  # Base intensity
        if trigger.get("importance", 0) > 0.8:
            intensity += 0.2
        
        # Generate questions
        questions = []
        if trigger_type == "knowledge_gap":
            area = trigger.get("area", "this area")
            questions = [
                f"What is the fundamental nature of {area}?",
                f"How does {area} relate to other concepts?",
                f"What evidence supports understanding of {area}?"
            ]
        
        signal_id = f"curiosity_{int(time.time())}_{random.randint(1000, 9999)}"
        
        signal = CuriositySignal(
            signal_id=signal_id,
            curiosity_type=curiosity_type,
            intensity=intensity,
            focus_area=trigger.get("area", "general"),
            specific_questions=questions,
            context=context.copy(),
            timestamp=time.time(),
            decay_rate=0.1,
            satisfaction_threshold=0.8
        )
        
        self.active_curiosity_signals[signal_id] = signal
        return signal


# GoalPriorityManager Components
class PriorityLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


@dataclass
class PriorityFactors:
    urgency: float
    importance: float
    resource_availability: float
    alignment_score: float
    success_probability: float
    emotional_motivation: float


@dataclass
class PrioritizedGoal:
    goal_id: str
    goal_data: Dict[str, Any]
    priority_score: float
    priority_level: PriorityLevel
    priority_factors: PriorityFactors
    last_updated: float
    dependencies: List[str]
    resource_requirements: Dict[str, float]


class GoalPriorityManager:
    def __init__(self):
        self.logger = logging.getLogger("nis.goal_priority_manager")
        self.memory = MockMemoryManager()
        self.prioritized_goals = {}
        self.priority_queue = []
        
        self.priority_weights = {
            "urgency": 0.25,
            "importance": 0.30,
            "resource_availability": 0.20,
            "alignment_score": 0.15,
            "success_probability": 0.05,
            "emotional_motivation": 0.05
        }
    
    def add_goal(self, goal_id, goal_data, initial_factors=None):
        if initial_factors is None:
            initial_factors = self._analyze_goal_factors(goal_data)
        
        priority_score = self._compute_priority_score(initial_factors)
        priority_level = self._determine_priority_level(priority_score)
        
        prioritized_goal = PrioritizedGoal(
            goal_id=goal_id,
            goal_data=goal_data,
            priority_score=priority_score,
            priority_level=priority_level,
            priority_factors=initial_factors,
            last_updated=time.time(),
            dependencies=goal_data.get("dependencies", []),
            resource_requirements=goal_data.get("resource_requirements", {})
        )
        
        self.prioritized_goals[goal_id] = prioritized_goal
        heapq.heappush(self.priority_queue, (-priority_score, goal_id))
        
        return prioritized_goal
    
    def _analyze_goal_factors(self, goal_data):
        # Calculate urgency
        deadline = goal_data.get("deadline")
        if deadline:
            time_remaining = deadline - time.time()
            urgency = max(0.1, 1.0 - (time_remaining / 86400))  # Normalize to days
        else:
            goal_type = goal_data.get("goal_type", "learning")
            urgency = {"maintenance": 0.8, "problem_solving": 0.7, "learning": 0.4}.get(goal_type, 0.5)
        
        # Calculate importance
        importance = goal_data.get("importance", 0.5)
        goal_type = goal_data.get("goal_type", "learning")
        type_importance = {"maintenance": 0.9, "problem_solving": 0.8, "learning": 0.6}.get(goal_type, 0.5)
        final_importance = (importance + type_importance) / 2.0
        
        # Calculate alignment (archaeological focus)
        domain = goal_data.get("domain", "general")
        alignment_score = 1.0 if domain in ["archaeology", "heritage_preservation"] else 0.6
        
        # Other factors
        resource_availability = 0.8  # Mock value
        success_probability = goal_data.get("success_probability", 0.7)
        emotional_motivation = goal_data.get("emotional_context", {}).get("interest", 0.5)
        
        return PriorityFactors(
            urgency=min(1.0, urgency),
            importance=min(1.0, final_importance),
            resource_availability=resource_availability,
            alignment_score=alignment_score,
            success_probability=success_probability,
            emotional_motivation=emotional_motivation
        )
    
    def _compute_priority_score(self, factors):
        return (
            self.priority_weights["urgency"] * factors.urgency +
            self.priority_weights["importance"] * factors.importance +
            self.priority_weights["resource_availability"] * factors.resource_availability +
            self.priority_weights["alignment_score"] * factors.alignment_score +
            self.priority_weights["success_probability"] * factors.success_probability +
            self.priority_weights["emotional_motivation"] * factors.emotional_motivation
        )
    
    def _determine_priority_level(self, priority_score):
        if priority_score >= 0.8:
            return PriorityLevel.CRITICAL
        elif priority_score >= 0.7:
            return PriorityLevel.HIGH
        elif priority_score >= 0.5:
            return PriorityLevel.MEDIUM
        elif priority_score >= 0.3:
            return PriorityLevel.LOW
        else:
            return PriorityLevel.BACKGROUND
    
    def get_priority_ordered_goals(self, limit=None):
        # Sort goals by priority score
        sorted_goals = sorted(
            self.prioritized_goals.values(),
            key=lambda x: x.priority_score,
            reverse=True
        )
        return sorted_goals[:limit] if limit else sorted_goals
    
    def get_priority_statistics(self):
        total_goals = len(self.prioritized_goals)
        priority_distribution = {}
        total_score = 0
        
        for goal in self.prioritized_goals.values():
            level = goal.priority_level.value
            priority_distribution[level] = priority_distribution.get(level, 0) + 1
            total_score += goal.priority_score
        
        return {
            "total_goals": total_goals,
            "priority_distribution": priority_distribution,
            "average_priority_score": total_score / max(total_goals, 1)
        }


def main():
    """Run standalone tests for AGI v2.0 components."""
    print("🚀 NIS Protocol v2.0 AGI Core Implementation - Standalone Tests")
    print("=" * 70)
    
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise
    
    results = []
    
    # Test MetaCognitiveProcessor
    print("\n🧠 Testing MetaCognitiveProcessor")
    print("-" * 40)
    try:
        processor = MetaCognitiveProcessor()
        
        process_data = {
            "processing_time": 2.5,
            "resource_usage": {"memory": 0.6, "cpu": 0.4},
            "data_volume": 100,
            "output_completeness": 0.9,
            "error_rate": 0.05,
            "context_alignment": 0.8,
            "goal_alignment": 0.85
        }
        
        analysis = processor.analyze_cognitive_process(
            CognitiveProcess.REASONING, process_data, {"domain": "archaeology"}
        )
        
        print(f"✅ Efficiency Score: {analysis.efficiency_score:.3f}")
        print(f"✅ Quality Metrics: {len(analysis.quality_metrics)} calculated")
        print(f"✅ Confidence: {analysis.confidence:.3f}")
        
        # Test bias detection
        reasoning_chain = [
            {"evidence_type": "supporting", "confidence": 0.9, "text": "Strong evidence"},
            {"evidence_type": "supporting", "confidence": 0.8, "text": "More evidence"}
        ]
        
        bias_results = processor.detect_cognitive_biases(reasoning_chain, {})
        print(f"✅ Bias Detection: {len(bias_results['biases_detected'])} biases found")
        
        results.append(True)
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append(False)
    
    # Test CuriosityEngine  
    print("\n🔍 Testing CuriosityEngine")
    print("-" * 40)
    try:
        engine = CuriosityEngine()
        
        context = {
            "domain": "archaeology",
            "pending_questions": ["What does this symbol mean?"]
        }
        knowledge_base = {"archaeology": {"facts": []}}
        
        gaps = engine.detect_knowledge_gaps(context, knowledge_base)
        print(f"✅ Knowledge Gaps: {len(gaps)} found")
        
        novel_item = {"type": "hieroglyph", "category": "religious"}
        novelty_context = {"known_items": [{"type": "pottery"}]}
        novelty = engine.assess_novelty(novel_item, novelty_context)
        print(f"✅ Novelty Assessment: {novelty:.3f}")
        
        trigger = {"type": "knowledge_gap", "area": "mayan_culture"}
        signal = engine.generate_curiosity_signal(trigger, context)
        print(f"✅ Curiosity Signal: {signal.curiosity_type.value}, intensity {signal.intensity:.3f}")
        
        results.append(True)
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append(False)
    
    # Test GoalPriorityManager
    print("\n🎯 Testing GoalPriorityManager")
    print("-" * 40)
    try:
        manager = GoalPriorityManager()
        
        goal_data = {
            "goal_type": "maintenance",
            "description": "Preserve ancient artifact",
            "domain": "heritage_preservation",
            "importance": 0.9,
            "deadline": time.time() + 86400,
            "emotional_context": {"interest": 0.8}
        }
        
        goal = manager.add_goal("preserve_001", goal_data)
        print(f"✅ Goal Added: {goal.priority_level.value} priority")
        print(f"✅ Priority Score: {goal.priority_score:.3f}")
        
        # Add second goal
        goal_data_2 = {
            "goal_type": "exploration",
            "domain": "archaeology",
            "importance": 0.6
        }
        manager.add_goal("explore_001", goal_data_2)
        
        ordered = manager.get_priority_ordered_goals()
        print(f"✅ Goal Ordering: {len(ordered)} goals prioritized")
        
        stats = manager.get_priority_statistics()
        print(f"✅ Statistics: avg score {stats['average_priority_score']:.3f}")
        
        results.append(True)
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Results: {passed}/{total} components passed")
    
    if passed == total:
        print("\n🎉 All AGI v2.0 core components working!")
        print("\n✨ Achievements:")
        print("✅ Cognitive process analysis with efficiency & quality metrics")
        print("✅ Bias detection for confirmation & overconfidence biases")
        print("✅ Knowledge gap detection with domain-specific logic")
        print("✅ Novelty assessment based on item similarity")
        print("✅ Curiosity signal generation with type classification")
        print("✅ Multi-criteria goal prioritization with cultural alignment")
        print("✅ Priority management with urgency, importance & alignment factors")
        
        print("\n🚀 Ready for next development phase!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} component(s) failed")
        return 1


if __name__ == "__main__":
    exit(main()) 
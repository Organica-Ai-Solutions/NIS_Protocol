#!/usr/bin/env python3
"""
Test script for NIS Protocol v2.0 AGI Core Implementation

This script tests the newly implemented core AGI components:
- MetaCognitiveProcessor: Cognitive analysis and bias detection
- CuriosityEngine: Knowledge gap detection and exploration
- GoalPriorityManager: Multi-criteria goal prioritization
"""

import sys
import os
import time
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.consciousness.meta_cognitive_processor import (
    MetaCognitiveProcessor, CognitiveProcess
)
from src.agents.goals.curiosity_engine import CuriosityEngine, CuriosityType
from src.agents.goals.goal_priority_manager import GoalPriorityManager


def test_meta_cognitive_processor():
    """Test the MetaCognitiveProcessor implementation."""
    print("\n🧠 Testing MetaCognitiveProcessor")
    print("=" * 50)
    
    processor = MetaCognitiveProcessor()
    
    # Test 1: Cognitive Process Analysis
    print("\n1. Testing cognitive process analysis...")
    process_data = {
        "processing_time": 2.5,
        "resource_usage": {"memory": 0.6, "cpu": 0.4},
        "data_volume": 100,
        "output_completeness": 0.9,
        "error_rate": 0.05,
        "context_alignment": 0.8,
        "goal_alignment": 0.85,
        "internal_consistency": 0.92,
        "logical_flow": 0.88
    }
    
    context = {
        "domain": "archaeology",
        "current_task": "artifact_analysis"
    }
    
    analysis = processor.analyze_cognitive_process(
        CognitiveProcess.REASONING, 
        process_data, 
        context
    )
    
    print(f"   Efficiency Score: {analysis.efficiency_score:.3f}")
    print(f"   Quality Metrics: {analysis.quality_metrics}")
    print(f"   Bottlenecks: {analysis.bottlenecks}")
    print(f"   Confidence: {analysis.confidence:.3f}")
    print(f"   Improvements: {analysis.improvement_suggestions[:3]}")
    
    # Test 2: Bias Detection
    print("\n2. Testing cognitive bias detection...")
    reasoning_chain = [
        {
            "evidence_type": "supporting",
            "confidence": 0.9,
            "text": "This definitely proves our hypothesis",
            "initial_estimate": 100,
            "reasoning_steps": 3,
            "contradiction_count": 0
        },
        {
            "evidence_type": "supporting", 
            "confidence": 0.8,
            "text": "Another piece of clearly supporting evidence",
            "estimate": 105,
            "reasoning_steps": 2,
            "contradiction_count": 0
        },
        {
            "evidence_type": "supporting",
            "confidence": 0.95,
            "text": "Obviously this confirms our view",
            "estimate": 102,
            "reasoning_steps": 1,
            "contradiction_count": 0
        }
    ]
    
    bias_context = {
        "search_terms": ["prove theory", "confirm hypothesis", "best evidence"],
        "domain": "archaeology"
    }
    
    bias_results = processor.detect_cognitive_biases(reasoning_chain, bias_context)
    
    print(f"   Biases Detected: {bias_results['biases_detected']}")
    print(f"   Overall Bias Score: {bias_results['overall_bias_score']:.3f}")
    for bias in bias_results['biases_detected']:
        severity = bias_results['severity_scores'][bias]
        print(f"   - {bias}: severity {severity:.3f}")
    
    print("✅ MetaCognitiveProcessor tests completed")


def test_curiosity_engine():
    """Test the CuriosityEngine implementation."""
    print("\n🔍 Testing CuriosityEngine")
    print("=" * 50)
    
    engine = CuriosityEngine()
    
    # Test 1: Knowledge Gap Detection
    print("\n1. Testing knowledge gap detection...")
    context = {
        "domain": "archaeology",
        "topics": ["mayan_hieroglyphs", "preservation_methods"],
        "pending_questions": [
            "What is the meaning of this symbol?",
            "How can we preserve fragile artifacts?"
        ]
    }
    
    knowledge_base = {
        "archaeology": {
            "facts": [
                {"description": "Mayan civilization existed in Mesoamerica", "explanation": "Historical evidence"},
                {"description": "Stone carvings require special preservation"}
            ],
            "relationships": [
                {"description": "Cultural artifacts reveal societal structure", "chain": ["artifact", "society"]}
            ]
        }
    }
    
    gaps = engine.detect_knowledge_gaps(context, knowledge_base)
    
    print(f"   Knowledge gaps found: {len(gaps)}")
    for gap in gaps[:3]:  # Show top 3
        print(f"   - {gap['gap_type']}: {gap['description']} (score: {gap.get('total_score', 0):.3f})")
    
    # Test 2: Novelty Assessment
    print("\n2. Testing novelty assessment...")
    novel_item = {
        "name": "Unknown Symbol",
        "type": "hieroglyph",
        "category": "religious_symbol",
        "description": "intricate carved religious ceremonial marker unknown origin"
    }
    
    novelty_context = {
        "known_items": [
            {"type": "hieroglyph", "category": "calendar", "description": "date marker calendar stone"},
            {"type": "hieroglyph", "category": "royal", "description": "king name royal title"}
        ],
        "occurrence_history": {"hieroglyph": 5, "pottery": 12},
        "expected_items": [{"type": "pottery", "category": "ceremonial"}]
    }
    
    novelty_score = engine.assess_novelty(novel_item, novelty_context)
    print(f"   Novelty Score: {novelty_score:.3f}")
    
    # Test 3: Curiosity Signal Generation
    print("\n3. Testing curiosity signal generation...")
    trigger = {
        "type": "knowledge_gap",
        "area": "mayan_religious_practices",
        "description": "Gap in understanding of religious ceremonies"
    }
    
    signal_context = {
        "domain": "archaeology",
        "emotional_state": {"curiosity": 0.8, "urgency": 0.6}
    }
    
    signal = engine.generate_curiosity_signal(trigger, signal_context)
    
    if signal:
        print(f"   Signal Type: {signal.curiosity_type.value}")
        print(f"   Intensity: {signal.intensity:.3f}")
        print(f"   Focus Area: {signal.focus_area}")
        print(f"   Questions: {signal.specific_questions[:2]}")
    else:
        print("   No signal generated (below threshold)")
    
    print("✅ CuriosityEngine tests completed")


def test_goal_priority_manager():
    """Test the GoalPriorityManager implementation."""
    print("\n🎯 Testing GoalPriorityManager")
    print("=" * 50)
    
    manager = GoalPriorityManager()
    
    # Test 1: Goal Addition and Prioritization
    print("\n1. Testing goal addition and prioritization...")
    
    goals_to_add = [
        {
            "goal_id": "preserve_artifact_001",
            "goal_data": {
                "goal_type": "maintenance",
                "description": "Preserve ancient mayan codex using traditional methods",
                "domain": "heritage_preservation",
                "importance": 0.9,
                "complexity": 0.7,
                "deadline": time.time() + 86400,  # 1 day
                "potential_impact": {
                    "knowledge_gain": 0.8,
                    "capability_gain": 0.6,
                    "system_improvement": 0.7
                },
                "ethical_considerations": {
                    "cultural_sensitivity": True,
                    "community_involvement": True,
                    "indigenous_rights": True,
                    "potential_harm": "none"
                },
                "emotional_context": {
                    "curiosity": 0.7,
                    "interest": 0.8,
                    "satisfaction_potential": 0.9
                }
            }
        },
        {
            "goal_id": "explore_new_site",
            "goal_data": {
                "goal_type": "exploration",
                "description": "Investigate newly discovered archaeological site",
                "domain": "archaeology",
                "importance": 0.6,
                "complexity": 0.8,
                "potential_impact": {
                    "knowledge_gain": 0.9,
                    "capability_gain": 0.5,
                    "system_improvement": 0.4
                },
                "emotional_context": {
                    "curiosity": 0.9,
                    "interest": 0.8,
                    "satisfaction_potential": 0.7
                }
            }
        },
        {
            "goal_id": "optimize_documentation",
            "goal_data": {
                "goal_type": "optimization",
                "description": "Improve efficiency of artifact documentation process",
                "domain": "heritage_preservation",
                "importance": 0.5,
                "complexity": 0.4,
                "resource_requirements": {
                    "computational": 0.6,
                    "time": 0.4
                },
                "potential_impact": {
                    "knowledge_gain": 0.3,
                    "capability_gain": 0.8,
                    "system_improvement": 0.9
                }
            }
        }
    ]
    
    prioritized_goals = []
    for goal_info in goals_to_add:
        prioritized_goal = manager.add_goal(
            goal_info["goal_id"],
            goal_info["goal_data"]
        )
        prioritized_goals.append(prioritized_goal)
    
    # Display prioritization results
    print(f"   Added {len(prioritized_goals)} goals to priority manager")
    
    # Test 2: Priority Ordering
    print("\n2. Testing priority ordering...")
    ordered_goals = manager.get_priority_ordered_goals()
    
    for i, goal in enumerate(ordered_goals, 1):
        factors = goal.priority_factors
        print(f"   {i}. {goal.goal_id}")
        print(f"      Priority: {goal.priority_level.value} (score: {goal.priority_score:.3f})")
        print(f"      Key factors: urgency={factors.urgency:.2f}, importance={factors.importance:.2f}, "
              f"alignment={factors.alignment_score:.2f}")
    
    # Test 3: Priority Update
    print("\n3. Testing priority update...")
    if ordered_goals:
        test_goal_id = ordered_goals[0].goal_id
        context_changes = {
            "urgency_increase": 0.3,
            "resource_availability_change": -0.2,
            "new_deadline": time.time() + 3600  # 1 hour urgency
        }
        
        updated_goal = manager.update_goal_priority(test_goal_id, context_changes)
        if updated_goal:
            print(f"   Updated {test_goal_id}")
            print(f"   New priority: {updated_goal.priority_level.value} (score: {updated_goal.priority_score:.3f})")
    
    # Test 4: Statistics
    print("\n4. Testing priority statistics...")
    stats = manager.get_priority_statistics()
    print(f"   Total goals: {stats['total_goals']}")
    print(f"   Priority distribution: {stats['priority_distribution']}")
    print(f"   Average priority score: {stats['average_priority_score']:.3f}")
    
    print("✅ GoalPriorityManager tests completed")


def main():
    """Run all AGI v2.0 component tests."""
    print("🚀 NIS Protocol v2.0 AGI Core Implementation Tests")
    print("=" * 60)
    print("Testing the Week 1-2 priority implementations:")
    print("• MetaCognitiveProcessor - Cognitive analysis and bias detection")
    print("• CuriosityEngine - Knowledge gap detection and exploration")
    print("• GoalPriorityManager - Multi-criteria goal prioritization")
    
    try:
        # Run all tests
        test_meta_cognitive_processor()
        test_curiosity_engine() 
        test_goal_priority_manager()
        
        print("\n🎉 All tests completed successfully!")
        print("\n📊 Summary:")
        print("✅ MetaCognitiveProcessor: Cognitive analysis, bias detection")
        print("✅ CuriosityEngine: Knowledge gaps, novelty assessment, signal generation") 
        print("✅ GoalPriorityManager: Multi-criteria prioritization, dynamic updates")
        
        print("\n🔄 Next Steps:")
        print("• Week 3-4: Implement Simulation & Prediction modules")
        print("• Week 5-6: Implement Alignment & Safety modules")
        print("• Week 7-8: Implement Memory Enhancement & Optimization")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 
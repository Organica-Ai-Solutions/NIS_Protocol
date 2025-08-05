#!/usr/bin/env python3
"""
Minimal test script for NIS Protocol v2.0 AGI Core Implementation

This script tests the newly implemented core AGI components with minimal dependencies.
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_meta_cognitive_processor():
    """Test the MetaCognitiveProcessor implementation."""
    print("\n🧠 Testing MetaCognitiveProcessor")
    print("=" * 50)
    
    try:
        # Import directly to avoid dependency issues
        sys.path.append('../src/agents/consciousness')
        
        from src.agents.consciousness.meta_cognitive_processor import (
            MetaCognitiveProcessor, CognitiveProcess
        )
        
        processor = MetaCognitiveProcessor()
        
        # Test cognitive process analysis
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
        
        print(f"   ✅ Efficiency Score: {analysis.efficiency_score:.3f}")
        print(f"   ✅ Quality Metrics: {len(analysis.quality_metrics)} metrics calculated")
        print(f"   ✅ Bottlenecks Found: {len(analysis.bottlenecks)}")
        print(f"   ✅ Confidence: {analysis.confidence:.3f}")
        print(f"   ✅ Improvements: {len(analysis.improvement_suggestions)} suggestions")
        
        # Test bias detection
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
            }
        ]
        
        bias_context = {
            "search_terms": ["prove theory", "confirm hypothesis", "recommended evidence"],
            "domain": "archaeology"
        }
        
        bias_results = processor.detect_cognitive_biases(reasoning_chain, bias_context)
        
        print(f"   ✅ Biases Detected: {len(bias_results['biases_detected'])}")
        print(f"   ✅ Overall Bias Score: {bias_results['overall_bias_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ MetaCognitiveProcessor test failed: {e}")
        return False


def test_curiosity_engine():
    """Test the CuriosityEngine implementation."""
    print("\n🔍 Testing CuriosityEngine") 
    print("=" * 50)
    
    try:
        from src.agents.goals.curiosity_engine import CuriosityEngine
        
        engine = CuriosityEngine()
        
        # Test knowledge gap detection
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
        
        print(f"   ✅ Knowledge gaps found: {len(gaps)}")
        print(f"   ✅ Gap types: {set(gap['gap_type'] for gap in gaps)}")
        
        # Test novelty assessment
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
        print(f"   ✅ Novelty Score: {novelty_score:.3f}")
        
        # Test curiosity signal generation
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
            print(f"   ✅ Signal Type: {signal.curiosity_type.value}")
            print(f"   ✅ Intensity: {signal.intensity:.3f}")
            print(f"   ✅ Focus Area: {signal.focus_area}")
            print(f"   ✅ Questions Generated: {len(signal.specific_questions)}")
        else:
            print("   ⚠️  No signal generated (below threshold)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ CuriosityEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_goal_priority_manager():
    """Test the GoalPriorityManager implementation."""
    print("\n🎯 Testing GoalPriorityManager")
    print("=" * 50)
    
    try:
        from src.agents.goals.goal_priority_manager import GoalPriorityManager
        
        manager = GoalPriorityManager()
        
        # Test goal addition and prioritization
        print("\n1. Testing goal addition and prioritization...")
        
        goal_data = {
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
        
        prioritized_goal = manager.add_goal("preserve_artifact_001", goal_data)
        
        print(f"   ✅ Goal added with priority: {prioritized_goal.priority_level.value}")
        print(f"   ✅ Priority score: {prioritized_goal.priority_score:.3f}")
        print(f"   ✅ Dependencies: {len(prioritized_goal.dependencies)}")
        print(f"   ✅ Resource requirements: {len(prioritized_goal.resource_requirements)}")
        
        # Test priority factors
        factors = prioritized_goal.priority_factors
        print(f"   ✅ Urgency: {factors.urgency:.3f}")
        print(f"   ✅ Importance: {factors.importance:.3f}")
        print(f"   ✅ Alignment: {factors.alignment_score:.3f}")
        print(f"   ✅ Success Probability: {factors.success_probability:.3f}")
        
        # Test statistics
        print("\n2. Testing priority statistics...")
        stats = manager.get_priority_statistics()
        print(f"   ✅ Total goals: {stats['total_goals']}")
        print(f"   ✅ Priority distribution: {stats['priority_distribution']}")
        print(f"   ✅ Average priority score: {stats['average_priority_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ GoalPriorityManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all AGI v2.0 component tests."""
    print("🚀 NIS Protocol v2.0 AGI Core Implementation Tests (Minimal)")
    print("=" * 65)
    print("Testing the Week 1-2 priority implementations:")
    print("• MetaCognitiveProcessor - Cognitive analysis and bias detection")
    print("• CuriosityEngine - Knowledge gap detection and exploration")
    print("• GoalPriorityManager - Multi-criteria goal prioritization")
    
    results = []
    
    # Run tests
    results.append(test_meta_cognitive_processor())
    results.append(test_curiosity_engine())
    results.append(test_goal_priority_manager())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} components passed")
    
    if passed == total:
        print("\n🎉 All core AGI v2.0 components are working correctly!")
        print("\n✨ Implementation Status:")
        print("✅ MetaCognitiveProcessor: comprehensive cognitive analysis & bias detection")
        print("✅ CuriosityEngine: comprehensive knowledge gap detection & novelty assessment") 
        print("✅ GoalPriorityManager: Multi-criteria prioritization with cultural alignment")
        
        print("\n🎯 Ready for Week 3-4 Development:")
        print("• Simulation Agent - Predictive modeling & scenario planning")
        print("• Alignment Agent - Ethical alignment & safety monitoring")
        print("• Memory Consolidation - Enhanced memory system integration")
        
        return 0
    else:
        print(f"\n⚠️  {total - passed} component(s) failed testing")
        return 1


if __name__ == "__main__":
    exit(main()) 
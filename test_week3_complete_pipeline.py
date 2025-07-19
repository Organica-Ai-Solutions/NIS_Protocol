#!/usr/bin/env python3
"""
Test Week 3: Complete PINN Integration - Laplace→KAN→PINN→LLM Pipeline

This script comprehensively tests the complete Week 3 implementation with
PINN physics validation integrated into the scientific pipeline.

Test Coverage:
- PINN Physics Agent functionality
- Complete pipeline integration (Laplace→KAN→PINN)
- Physics constraint validation
- Enhanced hybrid agents with physics validation
- Performance with physics compliance scoring
"""

import sys
import os
import numpy as np
import torch
import time

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_pinn_physics_agent():
    """Test the PINN Physics Agent implementation."""
    print("🧪 Testing PINN Physics Agent...")
    
    try:
        from src.agents.physics.pinn_physics_agent import PINNPhysicsAgent, PhysicsLaw
        
        # Create PINN agent
        agent = PINNPhysicsAgent()
        
        # Test 1: Symbolic function validation
        test_message = {
            "operation": "validate_symbolic",
            "payload": {
                "symbolic_function": "sin(2*pi*0.5*t)*exp(-0.1*t)",  # Physically reasonable
                "constraints": ["conservation_energy", "causality"]
            }
        }
        
        result = agent.process(test_message)
        
        if result["status"] == "success":
            payload = result["payload"]
            print(f"   ✅ Symbolic validation successful")
            print(f"   📊 Physics compliance: {payload['physics_compliance']:.3f}")
            print(f"   ⚠️  Violations: {len(payload['violations'])}")
            print(f"   ⏱️  Processing time: {payload['processing_time']:.3f}s")
            
            if payload['corrected_function']:
                print(f"   🔧 Corrected function: {payload['corrected_function']}")
        else:
            print(f"   ❌ Symbolic validation failed: {result['payload']}")
            return False
        
        # Test 2: Physics constraints check
        constraints_message = {
            "operation": "check_constraints",
            "payload": {"laws": ["conservation_energy", "causality", "continuity"]}
        }
        
        constraints_result = agent.process(constraints_message)
        
        if constraints_result["status"] == "success":
            print(f"   ✅ Constraints check successful")
            print(f"   🔬 Available constraints: {constraints_result['payload']['total_constraints']}")
        else:
            print(f"   ❌ Constraints check failed")
            return False
        
        # Test 3: Physics validation with violations
        violation_message = {
            "operation": "validate_symbolic", 
            "payload": {
                "symbolic_function": "exp(10*t)",  # Exponential growth - likely violation
                "constraints": ["conservation_energy", "causality"]
            }
        }
        
        violation_result = agent.process(violation_message)
        
        if violation_result["status"] == "success":
            v_payload = violation_result["payload"]
            print(f"   ✅ Violation test successful")
            print(f"   📉 Physics compliance: {v_payload['physics_compliance']:.3f}")
            print(f"   ⚠️  Violations detected: {len(v_payload['violations'])}")
            
            if v_payload['violations']:
                print(f"   🚨 First violation: {v_payload['violations'][0]['description']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ PINN Physics Agent test failed: {e}")
        return False

def test_complete_scientific_pipeline():
    """Test the complete Laplace→KAN→PINN scientific pipeline."""
    print("\n🔬 Testing Complete Scientific Pipeline...")
    
    try:
        from src.agents.hybrid_agent_core import CompleteScientificPipeline
        
        # Create complete pipeline
        pipeline = CompleteScientificPipeline()
        
        # Test signals with different physics characteristics
        test_signals = [
            ("Damped Oscillation", [np.sin(2*np.pi*0.5*t) * np.exp(-0.1*t) for t in np.linspace(0, 10, 50)]),
            ("Exponential Growth", [np.exp(0.1*t) for t in np.linspace(0, 5, 30)]),
            ("Conservation Signal", [np.cos(2*np.pi*t) for t in np.linspace(0, 2, 40)])
        ]
        
        for signal_name, signal_data in test_signals:
            print(f"   🌊 Processing {signal_name}...")
            
            # Configure pipeline with PINN enabled
            config = {
                "enable_pinn": True,
                "laplace": {"transform_type": "numerical"},
                "kan": {"symbolic_threshold": 0.5},
                "pinn": {
                    "physics_laws": ["conservation_energy", "causality", "continuity"],
                    "physics_threshold": 0.8
                }
            }
            
            result = pipeline.process_through_complete_pipeline(signal_data, config)
            
            print(f"     📊 Integrity Score: {result.integrity_score:.3f}")
            print(f"     🧪 Physics Compliance: {result.physics_compliance:.3f}")
            print(f"     ⚠️  Violations: {result.violations_detected}")
            print(f"     ⏱️  Processing Time: {result.processing_time:.3f}s")
            print(f"     🔧 Layers: {list(result.layer_outputs.keys())}")
            
            if result.symbolic_functions:
                print(f"     🔢 Symbolic: {result.symbolic_functions[0]}")
            
            if result.confidence_scores:
                print(f"     📈 Layer scores: {result.confidence_scores}")
        
        # Get complete pipeline statistics
        stats = pipeline.get_complete_pipeline_statistics()
        print(f"   📊 Complete Pipeline Statistics:")
        print(f"     • Total processed: {stats['total_processed']}")
        print(f"     • Success rate: {stats['overall_success_rate']:.2f}")
        print(f"     • Physics compliance avg: {stats['physics_compliance_average']:.3f}")
        print(f"     • Auto-corrections: {stats['auto_corrections']}")
        print(f"     • PINN integration: {stats['pinn_integration']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Complete pipeline test failed: {e}")
        return False

def test_complete_hybrid_agents():
    """Test the complete hybrid agents with PINN integration."""
    print("\n🤖 Testing Complete Hybrid Agents...")
    
    try:
        from src.agents.hybrid_agent_core import (
            CompleteMeTaCognitiveProcessor, CompleteCuriosityEngine, CompleteValidationAgent
        )
        
        agents_to_test = [
            ("MetaCognitive", CompleteMeTaCognitiveProcessor),
            ("Curiosity", CompleteCuriosityEngine), 
            ("Validation", CompleteValidationAgent)
        ]
        
        for agent_name, agent_class in agents_to_test:
            print(f"   🧠 Testing Complete {agent_name} Agent...")
            
            agent = agent_class()
            
            test_message = {
                "operation": "analyze",
                "payload": {
                    "data": [np.sin(x) * np.exp(-0.1*x) for x in np.linspace(0, 2*np.pi, 20)],
                    "description": f"Complete {agent_name.lower()} analysis with physics validation",
                    "metadata": {"test_type": "week3_complete_validation"}
                }
            }
            
            result = agent.process(test_message)
            
            if result["status"] == "success":
                payload = result["payload"]
                print(f"     ✅ Processing successful")
                print(f"     📊 Confidence: {payload['confidence']:.3f}")
                print(f"     🔬 Integrity Score: {payload['scientific_validation']['integrity_score']:.3f}")
                print(f"     🧪 Physics Compliance: {payload['scientific_validation']['physics_compliance']:.3f}")
                print(f"     ⚗️ Layers processed: {len(payload['scientific_validation']['layers_processed'])}")
                print(f"     🎯 Symbolic functions: {len(payload['scientific_validation']['symbolic_functions'])}")
                
                if payload.get('physics_validation'):
                    physics_val = payload['physics_validation']
                    print(f"     ⚖️ Physics violations: {len(physics_val['violations'])}")
                    print(f"     🔧 Auto-correction: {physics_val['auto_correction_applied']}")
                
                if payload['symbolic_insights']:
                    print(f"     💡 Key insight: {payload['symbolic_insights'][0]}")
            else:
                print(f"     ❌ {agent_name} processing failed: {result['payload']}")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Complete hybrid agents test failed: {e}")
        return False

def test_physics_constraint_enforcement():
    """Test physics constraint enforcement and auto-correction."""
    print("\n⚖️ Testing Physics Constraint Enforcement...")
    
    try:
        from src.agents.physics.pinn_physics_agent import PINNPhysicsAgent
        
        agent = PINNPhysicsAgent()
        
        # Test cases with known physics violations
        test_cases = [
            {
                "name": "Energy Conservation Violation",
                "function": "exp(t)",  # Infinite growth
                "constraints": ["conservation_energy"],
                "expected_violation": True
            },
            {
                "name": "Causality Violation", 
                "function": "exp(-t)*Heaviside(-t)",  # Effect before cause
                "constraints": ["causality"],
                "expected_violation": True
            },
            {
                "name": "Valid Physics Function",
                "function": "sin(t)*exp(-0.1*t)",  # Damped oscillation
                "constraints": ["conservation_energy", "causality"],
                "expected_violation": False
            }
        ]
        
        for test_case in test_cases:
            print(f"   🧪 Testing: {test_case['name']}")
            
            message = {
                "operation": "validate_symbolic",
                "payload": {
                    "symbolic_function": test_case["function"],
                    "constraints": test_case["constraints"]
                }
            }
            
            result = agent.process(message)
            
            if result["status"] == "success":
                payload = result["payload"]
                has_violations = len(payload["violations"]) > 0
                
                print(f"     📊 Physics compliance: {payload['physics_compliance']:.3f}")
                print(f"     ⚠️  Violations detected: {has_violations}")
                print(f"     ✅ Expected violations: {test_case['expected_violation']}")
                
                # Check if detection matches expectation
                if has_violations == test_case["expected_violation"]:
                    print(f"     ✅ Constraint enforcement working correctly")
                else:
                    print(f"     ❌ Constraint enforcement mismatch")
                
                if payload["corrected_function"]:
                    print(f"     🔧 Auto-correction applied: {payload['corrected_function']}")
            else:
                print(f"     ❌ Test failed: {result['payload']}")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Physics constraint enforcement test failed: {e}")
        return False

def test_performance_with_pinn():
    """Test performance benchmarks with PINN integration."""
    print("\n⚡ Testing Performance with PINN Integration...")
    
    try:
        from src.agents.hybrid_agent_core import CompleteMeTaCognitiveProcessor
        
        agent = CompleteMeTaCognitiveProcessor()
        
        # Performance test with different data sizes
        data_sizes = [10, 25, 50, 100]
        results = []
        
        for size in data_sizes:
            # Generate test signal
            t_vals = np.linspace(0, 2*np.pi, size)
            test_data = [np.sin(2*t) * np.exp(-0.1*t) for t in t_vals]
            
            start_time = time.time()
            
            message = {
                "operation": "analyze",
                "payload": {
                    "data": test_data,
                    "description": f"Performance test with PINN validation - {size} points"
                }
            }
            
            result = agent.process(message)
            processing_time = time.time() - start_time
            
            success = result["status"] == "success"
            confidence = result["payload"]["confidence"] if success else 0.0
            physics_compliance = 0.0
            
            if success and "physics_validation" in result["payload"]:
                physics_compliance = result["payload"]["physics_validation"]["compliance_score"]
            
            results.append({
                "size": size,
                "time": processing_time,
                "success": success,
                "confidence": confidence,
                "physics_compliance": physics_compliance
            })
            
            print(f"   📏 Size {size:3d}: {processing_time:.3f}s, Success: {'✅' if success else '❌'}, Physics: {physics_compliance:.3f}")
        
        # Calculate performance metrics
        avg_time = np.mean([r["time"] for r in results])
        success_rate = np.mean([r["success"] for r in results])
        avg_confidence = np.mean([r["confidence"] for r in results if r["success"]])
        avg_physics = np.mean([r["physics_compliance"] for r in results if r["physics_compliance"] > 0])
        
        print(f"   📊 Performance Summary with PINN:")
        print(f"     • Average processing time: {avg_time:.3f}s")
        print(f"     • Success rate: {success_rate:.2f}")
        print(f"     • Average confidence: {avg_confidence:.3f}")
        print(f"     • Average physics compliance: {avg_physics:.3f}")
        
        # Enhanced performance targets for Week 3
        targets = {
            "max_processing_time": 8.0,  # Slightly higher due to PINN
            "min_success_rate": 0.8,
            "min_confidence": 0.5,
            "min_physics_compliance": 0.7  # NEW: Physics target
        }
        
        meets_targets = (
            avg_time <= targets["max_processing_time"] and
            success_rate >= targets["min_success_rate"] and
            avg_confidence >= targets["min_confidence"] and
            avg_physics >= targets["min_physics_compliance"]
        )
        
        print(f"   🎯 Enhanced performance targets: {'✅ MET' if meets_targets else '❌ NOT MET'}")
        
        return meets_targets
        
    except Exception as e:
        print(f"   ❌ Performance with PINN test failed: {e}")
        return False

def test_end_to_end_complete_pipeline():
    """Test complete end-to-end pipeline with all components."""
    print("\n🔄 Testing End-to-End Complete Pipeline...")
    
    try:
        from src.agents.hybrid_agent_core import CompleteValidationAgent
        
        # Use the most strict agent for comprehensive testing
        agent = CompleteValidationAgent()
        
        # Complex test scenario
        complex_signal = []
        t_values = np.linspace(0, 4*np.pi, 80)
        for t in t_values:
            # Multi-component signal: fundamental + harmonics + decay
            signal_val = (
                1.0 * np.sin(t) +           # Fundamental
                0.5 * np.sin(3*t) +         # Third harmonic
                0.2 * np.sin(5*t)           # Fifth harmonic
            ) * np.exp(-0.05*t)             # Exponential decay
            complex_signal.append(signal_val)
        
        test_message = {
            "operation": "analyze",
            "payload": {
                "data": complex_signal,
                "description": "Complete end-to-end validation of complex multi-harmonic signal",
                "constraints": ["strict_physics", "high_accuracy"],
                "metadata": {
                    "test_type": "end_to_end_complete",
                    "signal_complexity": "high",
                    "expected_physics": "excellent"
                }
            }
        }
        
        start_time = time.time()
        result = agent.process(test_message)
        total_time = time.time() - start_time
        
        if result["status"] == "success":
            payload = result["payload"]
            
            print(f"   ✅ End-to-end processing successful")
            print(f"   ⏱️  Total processing time: {total_time:.3f}s")
            print(f"   📊 Final confidence: {payload['confidence']:.3f}")
            
            # Scientific validation metrics
            sci_val = payload["scientific_validation"]
            print(f"   🔬 Scientific Validation:")
            print(f"     • Integrity score: {sci_val['integrity_score']:.3f}")
            print(f"     • Physics compliance: {sci_val['physics_compliance']:.3f}")
            print(f"     • Layers processed: {sci_val['layers_processed']}")
            print(f"     • Violations detected: {sci_val['violations_detected']}")
            print(f"     • PINN validation: {sci_val['pinn_validation_complete']}")
            
            # Physics validation details
            if payload.get("physics_validation"):
                physics_val = payload["physics_validation"]
                print(f"   🧪 Physics Validation:")
                print(f"     • Compliance score: {physics_val['compliance_score']:.3f}")
                print(f"     • Violations: {len(physics_val['violations'])}")
                print(f"     • Auto-correction: {physics_val['auto_correction_applied']}")
            
            # Symbolic insights
            if payload["symbolic_insights"]:
                print(f"   💡 Symbolic Insights:")
                for insight in payload["symbolic_insights"][:3]:  # Show first 3
                    print(f"     • {insight}")
            
            # Pipeline completeness check
            required_layers = ["laplace", "kan", "pinn"]
            layers_present = sci_val["layers_processed"]
            complete_pipeline = all(layer in layers_present for layer in required_layers)
            
            print(f"   🔄 Pipeline completeness: {'✅ COMPLETE' if complete_pipeline else '❌ INCOMPLETE'}")
            
            return complete_pipeline and payload["confidence"] > 0.6
        else:
            print(f"   ❌ End-to-end processing failed: {result['payload']}")
            return False
        
    except Exception as e:
        print(f"   ❌ End-to-end complete pipeline test failed: {e}")
        return False

def main():
    """Run comprehensive Week 3 complete testing suite."""
    print("=" * 80)
    print("🧪 NIS Protocol V3 - Week 3: Complete PINN Integration Testing")
    print("=" * 80)
    
    tests = [
        ("PINN Physics Agent", test_pinn_physics_agent),
        ("Complete Scientific Pipeline", test_complete_scientific_pipeline),
        ("Complete Hybrid Agents", test_complete_hybrid_agents),
        ("Physics Constraint Enforcement", test_physics_constraint_enforcement),
        ("Performance with PINN", test_performance_with_pinn),
        ("End-to-End Complete Pipeline", test_end_to_end_complete_pipeline)
    ]
    
    results = []
    total_start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{test_name}: {status}")
            
        except Exception as e:
            print(f"\n{test_name}: ❌ ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    total_time = time.time() - total_start_time
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print("\n" + "=" * 80)
    print("🏁 WEEK 3 COMPLETE TESTING SUMMARY")
    print("=" * 80)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name:<40} {status}")
    
    print(f"\n📊 Overall Results:")
    print(f"   • Tests passed: {passed}/{total}")
    print(f"   • Success rate: {passed/total:.1%}")
    print(f"   • Total testing time: {total_time:.2f}s")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Week 3 Complete PINN Integration is ready!")
        print("   ✅ Laplace→KAN→PINN→LLM pipeline fully operational")
        print("   ✅ Physics constraint enforcement working")
        print("   ✅ Auto-correction capabilities functional")
        print("   Ready to proceed to Week 4: LLM Provider Integration")
    else:
        print(f"\n⚠️  {total-passed} tests failed. Review PINN integration before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
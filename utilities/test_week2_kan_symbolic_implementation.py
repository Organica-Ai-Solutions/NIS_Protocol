#!/usr/bin/env python3
"""
Test Week 2: KAN Symbolic Layer Implementation

This script comprehensively tests the enhanced KAN symbolic reasoning layer
and its integration with the Laplace→KAN→PINN scientific pipeline.

Test Coverage:
- Symbolic Bridge functionality
- Enhanced KAN reasoning agent
- Hybrid Agent Core integration
- Complete pipeline processing
- Performance validation
"""

import sys
import os
import numpy as np
import torch
import time

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_symbolic_bridge():
    """Test the Symbolic Bridge implementation."""
    print("🔬 Testing Symbolic Bridge...")
    
    try:
        from src.core.symbolic_bridge import SymbolicBridge, SymbolicType, PatternType
        from src.agents.signal_processing.laplace_processor import LaplaceSignalProcessor, LaplaceTransform, LaplaceTransformType
        
        # Create test signal
        t_values = np.linspace(0, 10, 100)
        signal = np.sin(2 * np.pi * 0.5 * t_values) * np.exp(-0.1 * t_values)
        
        # Process through Laplace
        laplace_processor = LaplaceSignalProcessor()
        
        # Create mock Laplace transform for testing
        s_values = np.linspace(-1+1j, 1+10j, 100)
        transform_values = np.fft.fft(signal)
        
        laplace_transform = LaplaceTransform(
            s_values=s_values,
            transform_values=transform_values,
            original_signal=signal,
            time_vector=t_values,
            transform_type=LaplaceTransformType.NUMERICAL
        )
        
        # Test symbolic bridge
        bridge = SymbolicBridge()
        result = bridge.transform_to_symbolic(laplace_transform)
        
        print(f"   ✅ Symbolic extraction completed")
        print(f"   📊 Function: {result.primary_function.expression}")
        print(f"   📈 Confidence: {result.extraction_confidence:.3f}")
        print(f"   🎯 Interpretability: {result.interpretability_score:.3f}")
        print(f"   ⏱️  Processing time: {result.computational_cost:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Symbolic Bridge test failed: {e}")
        return False

def test_enhanced_kan_agent():
    """Test the Enhanced KAN Reasoning Agent."""
    print("\n🧠 Testing Enhanced KAN Reasoning Agent...")
    
    try:
        from src.agents.reasoning.kan_reasoning_agent import KANReasoningAgent, SymbolicReasoningType
        
        # Create agent in symbolic mode
        agent = KANReasoningAgent(mode="symbolic")
        
        # Test symbolic extraction
        test_data = [1.0, 0.5, -0.5, 0.8, 0.2, -0.3, 0.9, 0.1, -0.8, 0.4]
        message = {
            "operation": "symbolic_extraction",
            "payload": {
                "input_data": test_data,
                "frequency_features": {
                    "dominant_frequencies": [0.5, 1.0, 2.0],
                    "magnitude_peaks": [1.0, 0.8, 0.3],
                    "spectral_centroid": 1.0,
                    "bandwidth": 2.0,
                    "energy": 5.0,
                    "pattern_complexity": 0.7
                }
            }
        }
        
        result = agent.process(message)
        
        if result["status"] == "success":
            payload = result["payload"]
            print(f"   ✅ Symbolic extraction successful")
            print(f"   🔢 Function: {payload['symbolic_function']}")
            print(f"   📊 Confidence: {payload['confidence']:.3f}")
            print(f"   💡 Interpretability: {payload['interpretability_score']:.3f}")
            print(f"   ⚙️  Reasoning type: {payload['reasoning_type']}")
        else:
            print(f"   ❌ Symbolic extraction failed: {result['payload']}")
            return False
        
        # Test frequency analysis
        freq_message = {
            "operation": "frequency_analysis",
            "payload": {
                "frequency_data": {
                    "dominant_frequencies": [0.5, 1.0, 1.5],
                    "magnitude": [1.0, 0.8, 0.3],
                    "phase": [0.0, 0.5, 1.0]
                }
            }
        }
        
        freq_result = agent.process(freq_message)
        
        if freq_result["status"] == "success":
            print(f"   ✅ Frequency analysis successful")
            pattern_analysis = freq_result["payload"]["pattern_analysis"]
            print(f"   📈 Pattern strength: {pattern_analysis['pattern_strength']:.3f}")
            print(f"   🎵 Pattern type: {pattern_analysis['pattern_type']}")
        else:
            print(f"   ❌ Frequency analysis failed")
            return False
        
        # Test statistics
        stats = agent.get_processing_statistics()
        print(f"   📊 Agent statistics: {stats['symbolic_extractions']} extractions, {stats['success_rate']:.2f} success rate")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Enhanced KAN Agent test failed: {e}")
        return False

def test_hybrid_agent_integration():
    """Test the Hybrid Agent Core integration."""
    print("\n🔗 Testing Hybrid Agent Core Integration...")
    
    try:
        from src.agents.hybrid_agent_core import HybridAgent, MetaCognitiveProcessor, CuriosityEngine, ValidationAgent
        from src.agents.hybrid_agent_core import LLMProvider
        
        # Test MetaCognitive Processor
        print("   🧠 Testing MetaCognitive Processor...")
        metacog_agent = MetaCognitiveProcessor()
        
        test_message = {
            "operation": "analyze",
            "payload": {
                "data": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8],
                "description": "Pattern analysis for metacognitive assessment",
                "metadata": {"test_type": "week2_validation"}
            }
        }
        
        result = metacog_agent.process(test_message)
        
        if result["status"] == "success":
            payload = result["payload"]
            print(f"     ✅ Processing successful")
            print(f"     📊 Confidence: {payload['confidence']:.3f}")
            print(f"     🔬 Integrity Score: {payload['scientific_validation']['integrity_score']:.3f}")
            print(f"     ⚗️ Layers processed: {len(payload['scientific_validation']['layers_processed'])}")
            print(f"     🎯 Symbolic functions: {len(payload['scientific_validation']['symbolic_functions'])}")
            
            if payload['symbolic_insights']:
                print(f"     💡 Key insight: {payload['symbolic_insights'][0]}")
        else:
            print(f"     ❌ MetaCognitive processing failed: {result['payload']}")
            return False
        
        # Test Curiosity Engine
        print("   🔍 Testing Curiosity Engine...")
        curiosity_agent = CuriosityEngine()
        
        novelty_message = {
            "operation": "analyze", 
            "payload": {
                "data": np.random.random(20).tolist(),  # Random data for novelty detection
                "description": "Novelty detection test"
            }
        }
        
        curiosity_result = curiosity_agent.process(novelty_message)
        
        if curiosity_result["status"] == "success":
            print(f"     ✅ Curiosity engine processing successful")
            print(f"     🎲 Novelty confidence: {curiosity_result['payload']['confidence']:.3f}")
        else:
            print(f"     ❌ Curiosity engine failed")
            return False
        
        # Test Validation Agent
        print("   🧪 Testing Validation Agent...")
        validation_agent = ValidationAgent()
        
        validation_message = {
            "operation": "analyze",
            "payload": {
                "data": [np.sin(x) for x in np.linspace(0, 2*np.pi, 10)],  # Clean sinusoidal for physics validation
                "description": "Physics validation test"
            }
        }
        
        validation_result = validation_agent.process(validation_message)
        
        if validation_result["status"] == "success":
            print(f"     ✅ Validation agent processing successful")
            validation_payload = validation_result["payload"]
            print(f"     ⚖️ Physics compliance: {validation_payload.get('physics_compliance', 'N/A')}")
            print(f"     🔍 Scientific validation: {validation_payload['scientific_validation']['integrity_score']:.3f}")
        else:
            print(f"     ❌ Validation agent failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Hybrid Agent integration test failed: {e}")
        return False

def test_complete_pipeline():
    """Test the complete Laplace→KAN→PINN pipeline."""
    print("\n🌊 Testing Complete Scientific Pipeline...")
    
    try:
        from src.agents.hybrid_agent_core import EnhancedScientificPipeline
        
        # Create pipeline
        pipeline = EnhancedScientificPipeline()
        
        # Test with different signal types
        test_signals = [
            ("Sinusoidal", [np.sin(2*np.pi*0.5*t) for t in np.linspace(0, 10, 50)]),
            ("Exponential Decay", [np.exp(-0.5*t) for t in np.linspace(0, 5, 30)]),
            ("Complex Wave", [np.sin(2*np.pi*t) + 0.5*np.cos(4*np.pi*t) for t in np.linspace(0, 2, 40)])
        ]
        
        for signal_name, signal_data in test_signals:
            print(f"   🌊 Processing {signal_name}...")
            
            # Configure pipeline
            config = {
                "enable_pinn": True,
                "laplace": {"transform_type": "numerical"},
                "kan": {"symbolic_threshold": 0.5},
                "pinn": {"strict_mode": False}
            }
            
            result = pipeline.process_through_pipeline(signal_data, config)
            
            print(f"     📊 Integrity Score: {result.integrity_score:.3f}")
            print(f"     ⏱️  Processing Time: {result.processing_time:.3f}s")
            print(f"     🔧 Layers: {list(result.layer_outputs.keys())}")
            
            if result.symbolic_functions:
                print(f"     🔢 Symbolic: {result.symbolic_functions[0]}")
            
            if result.confidence_scores:
                print(f"     📈 Confidence scores: {result.confidence_scores}")
        
        # Get pipeline statistics
        stats = pipeline.get_pipeline_statistics()
        print(f"   📊 Pipeline Statistics:")
        print(f"     • Total processed: {stats['total_processed']}")
        print(f"     • Success rate: {stats['overall_success_rate']:.2f}")
        print(f"     • Avg processing time: {stats['average_processing_time']:.3f}s")
        print(f"     • Layer success rates: {stats['layer_success_rates']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Complete pipeline test failed: {e}")
        return False

def test_performance_benchmarks():
    """Test performance benchmarks for the Week 2 implementation."""
    print("\n⚡ Testing Performance Benchmarks...")
    
    try:
        from src.agents.hybrid_agent_core import MetaCognitiveProcessor
        
        agent = MetaCognitiveProcessor()
        
        # Benchmark different data sizes
        data_sizes = [10, 50, 100, 200]
        results = []
        
        for size in data_sizes:
            test_data = np.random.random(size).tolist()
            
            start_time = time.time()
            
            message = {
                "operation": "analyze",
                "payload": {
                    "data": test_data,
                    "description": f"Performance test with {size} data points"
                }
            }
            
            result = agent.process(message)
            processing_time = time.time() - start_time
            
            success = result["status"] == "success"
            confidence = result["payload"]["confidence"] if success else 0.0
            
            results.append({
                "size": size,
                "time": processing_time,
                "success": success,
                "confidence": confidence
            })
            
            print(f"   📏 Size {size:3d}: {processing_time:.3f}s, Success: {'✅' if success else '❌'}, Confidence: {confidence:.3f}")
        
        # Calculate performance metrics
        avg_time = np.mean([r["time"] for r in results])
        success_rate = np.mean([r["success"] for r in results])
        avg_confidence = np.mean([r["confidence"] for r in results if r["success"]])
        
        print(f"   📊 Performance Summary:")
        print(f"     • Average processing time: {avg_time:.3f}s")
        print(f"     • Success rate: {success_rate:.2f}")
        print(f"     • Average confidence: {avg_confidence:.3f}")
        
        # Performance targets for Week 2
        targets = {
            "max_processing_time": 5.0,  # 5 seconds max
            "min_success_rate": 0.8,     # 80% success rate
            "min_confidence": 0.5         # 50% confidence
        }
        
        meets_targets = (
            avg_time <= targets["max_processing_time"] and
            success_rate >= targets["min_success_rate"] and
            avg_confidence >= targets["min_confidence"]
        )
        
        print(f"   🎯 Performance targets: {'✅ MET' if meets_targets else '❌ NOT MET'}")
        
        return meets_targets
        
    except Exception as e:
        print(f"   ❌ Performance benchmark failed: {e}")
        return False

def main():
    """Run comprehensive Week 2 testing suite."""
    print("=" * 80)
    print("🧪 NIS Protocol V3 - Week 2: KAN Symbolic Layer Testing")
    print("=" * 80)
    
    tests = [
        ("Symbolic Bridge", test_symbolic_bridge),
        ("Enhanced KAN Agent", test_enhanced_kan_agent),
        ("Hybrid Agent Integration", test_hybrid_agent_integration),
        ("Complete Pipeline", test_complete_pipeline),
        ("Performance Benchmarks", test_performance_benchmarks)
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
    print("🏁 WEEK 2 TESTING SUMMARY")
    print("=" * 80)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name:<30} {status}")
    
    print(f"\n📊 Overall Results:")
    print(f"   • Tests passed: {passed}/{total}")
    print(f"   • Success rate: {passed/total:.1%}")
    print(f"   • Total testing time: {total_time:.2f}s")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Week 2 KAN Symbolic Layer implementation is ready!")
        print("   Ready to proceed to Week 3: PINN Physics Validation")
    else:
        print(f"\n⚠️  {total-passed} tests failed. Review implementation before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
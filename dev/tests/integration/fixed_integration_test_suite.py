#!/usr/bin/env python3
"""
🧪 NIS Protocol v3 - Fixed Integration Test Suite

Comprehensive validation with proper import handling for all operational agents.
"""

import sys
import os
import time
import asyncio
import logging
import traceback
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import json

# Fix Python path - go up to project root and add src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_path = os.path.join(project_root, 'src')

# Add paths to sys.path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print(f"🔧 Fixed Python paths:")
print(f"   Project root: {project_root}")
print(f"   Source path: {src_path}")

@dataclass
class TestResult:
    test_name: str
    category: str
    status: str  # 'PASS', 'FAIL', 'SKIP'
    execution_time: float
    details: Dict[str, Any]
    error_message: Optional[str] = None

class FixedIntegrationTestSuite:
    """Fixed integration testing for NIS Protocol v3"""
    
    def __init__(self):
        self.test_results: List[TestResult] = []
        self.start_time = 0.0
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('fixed_integration_tests')
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete integration test suite with fixed imports"""
        
        print("🚀 Starting Fixed NIS Protocol v3 Integration Tests")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Test categories
        test_categories = [
            ("Import Validation", self._test_imports),
            ("Core Components", self._test_core_components),
            ("Scientific Pipeline", self._test_scientific_pipeline),
            ("Consciousness System", self._test_consciousness_system),
            ("Documentation Examples", self._test_documentation_examples),
            ("Performance Tests", self._test_performance)
        ]
        
        for category_name, test_function in test_categories:
            print(f"\n🧪 Testing: {category_name}")
            print("-" * 40)
            
            try:
                test_function()
            except Exception as e:
                self.logger.error(f"Category {category_name} failed: {e}")
                self._add_test_result(
                    f"{category_name.lower().replace(' ', '_')}_category",
                    category_name,
                    "FAIL",
                    0.0,
                    {},
                    str(e)
                )
        
        return self._generate_report()
    
    def _test_imports(self):
        """Test all critical imports work correctly"""
        
        # Test 1: Utils imports
        start_time = time.time()
        try:
            from utils.self_audit import self_audit_engine
            from utils.integrity_metrics import calculate_confidence
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "utils_imports",
                "Import Validation",
                "PASS",
                execution_time,
                {"modules": ["self_audit", "integrity_metrics"]}
            )
            print("  ✅ Utils imports successful")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "utils_imports",
                "Import Validation",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"  ❌ Utils imports failed: {e}")
        
        # Test 2: Scientific agents imports
        start_time = time.time()
        scientific_agents = {
            "Laplace": ("agents.signal_processing.enhanced_laplace_transformer", "EnhancedLaplaceTransformer"),
            "KAN": ("agents.reasoning.enhanced_kan_reasoning_agent", "EnhancedKANReasoningAgent"),
            "PINN": ("agents.physics.enhanced_pinn_physics_agent", "EnhancedPINNPhysicsAgent"),
            "Coordinator": ("meta.enhanced_scientific_coordinator", "EnhancedScientificCoordinator")
        }
        
        imported_agents = {}
        failed_imports = []
        
        for agent_name, (module_path, class_name) in scientific_agents.items():
            try:
                module = __import__(module_path, fromlist=[class_name])
                agent_class = getattr(module, class_name)
                imported_agents[agent_name] = agent_class
                print(f"    ✅ {agent_name} imported successfully")
            except Exception as e:
                failed_imports.append(f"{agent_name}: {e}")
                print(f"    ❌ {agent_name} import failed: {e}")
        
        execution_time = time.time() - start_time
        
        if not failed_imports:
            self._add_test_result(
                "scientific_agents_imports",
                "Import Validation",
                "PASS",
                execution_time,
                {"imported_agents": list(imported_agents.keys())}
            )
        else:
            self._add_test_result(
                "scientific_agents_imports",
                "Import Validation",
                "FAIL",
                execution_time,
                {"imported_agents": list(imported_agents.keys()), "failed_imports": failed_imports},
                f"Failed imports: {failed_imports}"
            )
        
        # Test 3: Consciousness agents imports
        start_time = time.time()
        try:
            from agents.consciousness.enhanced_conscious_agent import EnhancedConsciousAgent, ReflectionType, ConsciousnessLevel
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "consciousness_imports",
                "Import Validation",
                "PASS",
                execution_time,
                {"modules": ["enhanced_conscious_agent"]}
            )
            print("  ✅ Consciousness imports successful")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "consciousness_imports",
                "Import Validation",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"  ❌ Consciousness imports failed: {e}")
    
    def _test_core_components(self):
        """Test core component functionality"""
        
        # Test 1: Self-audit engine
        start_time = time.time()
        try:
            from utils.self_audit import self_audit_engine
            
            test_texts = [
                "System analysis completed with measured performance metrics",
                "comprehensive AI delivers well-suited results systematically",
                "Comprehensive evaluation yielded validated results"
            ]
            
            audit_results = []
            for text in test_texts:
                violations = self_audit_engine.audit_text(text)
                integrity_score = self_audit_engine.get_integrity_score(text)
                audit_results.append({
                    "text": text,
                    "integrity_score": integrity_score,
                    "violations": len(violations)
                })
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "self_audit_functionality",
                "Core Components",
                "PASS",
                execution_time,
                {"audit_results": audit_results}
            )
            
            avg_score = np.mean([r["integrity_score"] for r in audit_results])
            print(f"  ✅ Self-audit engine: {avg_score:.1f}/100 average score")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "self_audit_functionality",
                "Core Components",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"  ❌ Self-audit engine failed: {e}")
        
        # Test 2: Integrity metrics calculation
        start_time = time.time()
        try:
            from utils.integrity_metrics import calculate_confidence, create_default_confidence_factors
            
            factors = create_default_confidence_factors()
            
            # Test different confidence scenarios
            test_cases = [
                (1.0, 0.5, 0.9),  # Good data, medium complexity, high validation
                (0.8, 0.8, 0.7),  # Good data, high complexity, good validation
                (0.3, 0.2, 0.4)   # Poor data, low complexity, poor validation
            ]
            
            confidence_results = []
            for data_quality, complexity, validation in test_cases:
                confidence = calculate_confidence(data_quality, complexity, validation, factors)
                confidence_results.append({
                    "inputs": [data_quality, complexity, validation],
                    "confidence": confidence
                })
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "integrity_metrics_calculation",
                "Core Components",
                "PASS",
                execution_time,
                {"confidence_results": confidence_results}
            )
            print(f"  ✅ Integrity metrics: {len(confidence_results)} scenarios tested")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "integrity_metrics_calculation",
                "Core Components",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"  ❌ Integrity metrics failed: {e}")
    
    def _test_scientific_pipeline(self):
        """Test scientific pipeline components individually"""
        
        # Test 1: Laplace Transformer
        start_time = time.time()
        try:
            from agents.signal_processing.enhanced_laplace_transformer import EnhancedLaplaceTransformer
            
            transformer = EnhancedLaplaceTransformer("test_laplace", max_frequency=50.0)
            
            # Create test signal
            t = np.linspace(0, 2, 1000)
            signal = np.sin(2*np.pi*5*t) + 0.5*np.exp(-0.5*t)
            
            result = transformer.compute_laplace_transform(signal, t)
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "laplace_transformer_test",
                "Scientific Pipeline",
                "PASS",
                execution_time,
                {
                    "processing_time": result.metrics.processing_time,
                    "reconstruction_error": result.reconstruction_error,
                    "signal_quality": result.quality_assessment.value,
                    "poles_found": len(result.poles),
                    "signal_length": len(signal)
                }
            )
            print(f"  ✅ Laplace Transformer: {result.reconstruction_error:.6f} reconstruction error")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "laplace_transformer_test",
                "Scientific Pipeline",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"  ❌ Laplace Transformer failed: {e}")
        
        # Test 2: Scientific Coordinator
        start_time = time.time()
        try:
            from meta.enhanced_scientific_coordinator import EnhancedScientificCoordinator
            
            coordinator = EnhancedScientificCoordinator("test_coordinator")
            summary = coordinator.get_coordination_summary()
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "scientific_coordinator_test",
                "Scientific Pipeline",
                "PASS",
                execution_time,
                {
                    "coordinator_id": coordinator.coordinator_id,
                    "summary_available": summary is not None
                }
            )
            print(f"  ✅ Scientific Coordinator: {coordinator.coordinator_id}")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "scientific_coordinator_test",
                "Scientific Pipeline",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"  ❌ Scientific Coordinator failed: {e}")
        
        # Test 3: KAN Reasoning Agent (if available)
        start_time = time.time()
        try:
            from agents.reasoning.enhanced_kan_reasoning_agent import EnhancedKANReasoningAgent
            
            kan_agent = EnhancedKANReasoningAgent("test_kan", 4, [8, 4], 1, enable_self_audit=True)
            summary = kan_agent.get_performance_summary()
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "kan_reasoning_agent_test",
                "Scientific Pipeline",
                "PASS",
                execution_time,
                {
                    "agent_id": kan_agent.agent_id,
                    "input_dim": kan_agent.input_dim,
                    "output_dim": kan_agent.output_dim
                }
            )
            print(f"  ✅ KAN Reasoning Agent: {kan_agent.agent_id}")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "kan_reasoning_agent_test",
                "Scientific Pipeline",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"  ❌ KAN Reasoning Agent failed: {e}")
        
        # Test 4: PINN Physics Agent (if available)
        start_time = time.time()
        try:
            from agents.physics.enhanced_pinn_physics_agent import EnhancedPINNPhysicsAgent
            
            pinn_agent = EnhancedPINNPhysicsAgent("test_pinn", enable_self_audit=True)
            summary = pinn_agent.get_performance_summary()
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "pinn_physics_agent_test",
                "Scientific Pipeline",
                "PASS",
                execution_time,
                {
                    "agent_id": pinn_agent.agent_id,
                    "self_audit_enabled": pinn_agent.enable_self_audit
                }
            )
            print(f"  ✅ PINN Physics Agent: {pinn_agent.agent_id}")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "pinn_physics_agent_test",
                "Scientific Pipeline",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"  ❌ PINN Physics Agent failed: {e}")
    
    def _test_consciousness_system(self):
        """Test consciousness system functionality"""
        
        # Test 1: Enhanced Conscious Agent initialization
        start_time = time.time()
        try:
            from agents.consciousness.enhanced_conscious_agent import (
                EnhancedConsciousAgent, ReflectionType, ConsciousnessLevel
            )
            
            agent = EnhancedConsciousAgent(
                "test_consciousness",
                consciousness_level=ConsciousnessLevel.ENHANCED,
                enable_self_audit=True
            )
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "conscious_agent_initialization",
                "Consciousness System",
                "PASS",
                execution_time,
                {
                    "agent_id": agent.agent_id,
                    "consciousness_level": agent.consciousness_level.value,
                    "self_audit_enabled": agent.enable_self_audit
                }
            )
            print(f"  ✅ Enhanced Conscious Agent: {agent.agent_id}")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "conscious_agent_initialization",
                "Consciousness System",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"  ❌ Enhanced Conscious Agent failed: {e}")
            return
        
        # Test 2: Introspection capabilities
        start_time = time.time()
        try:
            introspection_tests = [
                ReflectionType.SYSTEM_HEALTH_CHECK,
                ReflectionType.INTEGRITY_ASSESSMENT,
                ReflectionType.PERFORMANCE_REVIEW
            ]
            
            introspection_results = []
            
            for reflection_type in introspection_tests:
                result = agent.perform_introspection(reflection_type)
                introspection_results.append({
                    "reflection_type": reflection_type.value,
                    "confidence": result.confidence,
                    "integrity_score": result.integrity_score,
                    "findings_count": len(result.findings),
                    "recommendations_count": len(result.recommendations)
                })
            
            execution_time = time.time() - start_time
            avg_confidence = np.mean([r["confidence"] for r in introspection_results])
            avg_integrity = np.mean([r["integrity_score"] for r in introspection_results])
            
            self._add_test_result(
                "introspection_capabilities",
                "Consciousness System",
                "PASS",
                execution_time,
                {
                    "introspection_results": introspection_results,
                    "average_confidence": avg_confidence,
                    "average_integrity_score": avg_integrity
                }
            )
            print(f"  ✅ Introspection: {avg_confidence:.3f} avg confidence, {avg_integrity:.1f}/100 integrity")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "introspection_capabilities",
                "Consciousness System",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"  ❌ Introspection capabilities failed: {e}")
        
        # Test 3: Agent monitoring
        start_time = time.time()
        try:
            # Register test agents for monitoring
            test_agents = [
                ("test_laplace", {"type": "signal_processing"}),
                ("test_coordinator", {"type": "coordination"})
            ]
            
            for agent_id, metadata in test_agents:
                agent.register_agent_for_monitoring(agent_id, metadata)
            
            # Get consciousness summary
            summary = agent.get_consciousness_summary()
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "agent_monitoring",
                "Consciousness System",
                "PASS",
                execution_time,
                {
                    "agents_registered": len(test_agents),
                    "agents_monitored": summary['consciousness_metrics']['agents_monitored'],
                    "total_reflections": summary['consciousness_metrics']['total_reflections']
                }
            )
            print(f"  ✅ Agent monitoring: {len(test_agents)} agents registered")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "agent_monitoring",
                "Consciousness System",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"  ❌ Agent monitoring failed: {e}")
    
    def _test_documentation_examples(self):
        """Test examples from documentation"""
        
        # Test 1: Basic integrity example from getting started
        start_time = time.time()
        try:
            from utils.self_audit import self_audit_engine
            
            # Example from documentation
            text = "System analysis completed with measured performance metrics"
            violations = self_audit_engine.audit_text(text)
            integrity_score = self_audit_engine.get_integrity_score(text)
            
            if violations:
                corrected_text, _ = self_audit_engine.auto_correct_text(text)
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "docs_integrity_example",
                "Documentation Examples",
                "PASS",
                execution_time,
                {
                    "original_text": text,
                    "integrity_score": integrity_score,
                    "violations_count": len(violations)
                }
            )
            print(f"  ✅ Docs integrity example: {integrity_score}/100 score")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "docs_integrity_example",
                "Documentation Examples",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"  ❌ Docs integrity example failed: {e}")
        
        # Test 2: Basic signal processing from docs
        start_time = time.time()
        try:
            from agents.signal_processing.enhanced_laplace_transformer import EnhancedLaplaceTransformer
            
            # Example from documentation
            transformer = EnhancedLaplaceTransformer("basic_integration", enable_self_audit=True)
            
            t = np.linspace(0, 2, 1000)
            signal = np.sin(2*np.pi*5*t) + 0.5*np.exp(-0.5*t)
            
            result = transformer.compute_laplace_transform(signal, t)
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "docs_signal_processing_example",
                "Documentation Examples",
                "PASS",
                execution_time,
                {
                    "processing_time": result.metrics.processing_time,
                    "reconstruction_error": result.reconstruction_error,
                    "signal_quality": result.quality_assessment.value
                }
            )
            print(f"  ✅ Docs signal processing: {result.reconstruction_error:.6f} error")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "docs_signal_processing_example",
                "Documentation Examples",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"  ❌ Docs signal processing failed: {e}")
    
    def _test_performance(self):
        """Test performance characteristics"""
        
        # Test 1: Laplace transformer performance scaling
        start_time = time.time()
        try:
            from agents.signal_processing.enhanced_laplace_transformer import EnhancedLaplaceTransformer
            
            transformer = EnhancedLaplaceTransformer("perf_test")
            
            # Test different signal sizes
            performance_data = []
            
            for size in [100, 500, 1000]:
                t = np.linspace(0, 1, size)
                signal = np.sin(2*np.pi*10*t) + 0.3*np.cos(2*np.pi*25*t)
                
                perf_start = time.time()
                result = transformer.compute_laplace_transform(signal, t)
                perf_time = time.time() - perf_start
                
                performance_data.append({
                    "signal_size": size,
                    "processing_time": perf_time,
                    "reconstruction_error": result.reconstruction_error
                })
            
            execution_time = time.time() - start_time
            avg_time = np.mean([p["processing_time"] for p in performance_data])
            
            self._add_test_result(
                "laplace_performance_scaling",
                "Performance Tests",
                "PASS",
                execution_time,
                {
                    "performance_data": performance_data,
                    "average_processing_time": avg_time
                }
            )
            print(f"  ✅ Laplace performance: {avg_time:.4f}s average processing")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "laplace_performance_scaling",
                "Performance Tests",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"  ❌ Laplace performance test failed: {e}")
        
        # Test 2: Consciousness introspection performance
        start_time = time.time()
        try:
            from agents.consciousness.enhanced_conscious_agent import (
                EnhancedConsciousAgent, ReflectionType, ConsciousnessLevel
            )
            
            agent = EnhancedConsciousAgent("perf_test_consciousness")
            
            # Test introspection speed
            introspection_times = []
            
            for _ in range(3):  # Reduced iterations for faster testing
                perf_start = time.time()
                result = agent.perform_introspection(ReflectionType.SYSTEM_HEALTH_CHECK)
                perf_time = time.time() - perf_start
                introspection_times.append(perf_time)
            
            execution_time = time.time() - start_time
            avg_introspection_time = np.mean(introspection_times)
            
            self._add_test_result(
                "consciousness_introspection_performance",
                "Performance Tests",
                "PASS",
                execution_time,
                {
                    "introspection_times": introspection_times,
                    "average_introspection_time": avg_introspection_time
                }
            )
            print(f"  ✅ Consciousness performance: {avg_introspection_time:.4f}s per introspection")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "consciousness_introspection_performance",
                "Performance Tests",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"  ❌ Consciousness performance test failed: {e}")
    
    def _add_test_result(self, test_name: str, category: str, status: str, 
                        execution_time: float, details: Dict[str, Any], 
                        error_message: Optional[str] = None):
        """Add test result to collection"""
        
        result = TestResult(
            test_name=test_name,
            category=category,
            status=status,
            execution_time=execution_time,
            details=details,
            error_message=error_message
        )
        
        self.test_results.append(result)
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        total_execution_time = time.time() - self.start_time
        
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == "PASS"])
        failed_tests = len([r for r in self.test_results if r.status == "FAIL"])
        skipped_tests = len([r for r in self.test_results if r.status == "SKIP"])
        
        # Calculate pass rate
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # Determine system status
        if pass_rate >= 0.9:
            system_status = "EXCELLENT"
        elif pass_rate >= 0.75:
            system_status = "GOOD"
        elif pass_rate >= 0.6:
            system_status = "ADEQUATE"
        else:
            system_status = "NEEDS_ATTENTION"
        
        # Generate recommendations
        recommendations = []
        
        if failed_tests == 0:
            recommendations.append("All tests passed - system is fully operational")
        else:
            recommendations.append(f"Address {failed_tests} failing tests")
        
        if pass_rate < 0.8:
            recommendations.append("Improve test pass rate before production deployment")
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "skipped_tests": skipped_tests,
                "pass_rate": pass_rate,
                "total_execution_time": total_execution_time,
                "system_status": system_status
            },
            "recommendations": recommendations,
            "test_results": self.test_results
        }


def main():
    """Run fixed integration test suite"""
    
    print("🧪 NIS Protocol v3 - Fixed Integration Test Suite")
    print("Testing with proper import handling")
    print("=" * 60)
    
    # Initialize test suite
    test_suite = FixedIntegrationTestSuite()
    
    # Run all tests
    report = test_suite.run_all_tests()
    
    # Display results
    summary = report["summary"]
    
    print(f"\n🎯 INTEGRATION TEST RESULTS")
    print("=" * 40)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"✅ Passed: {summary['passed_tests']}")
    print(f"❌ Failed: {summary['failed_tests']}")
    print(f"⏸️  Skipped: {summary['skipped_tests']}")
    print(f"⏱️  Total Time: {summary['total_execution_time']:.2f}s")
    print(f"🎯 System Status: {summary['system_status']}")
    print(f"📊 Pass Rate: {summary['pass_rate']:.1%}")
    
    # Results by category
    print(f"\n📋 Results by Category:")
    categories = {}
    for result in report["test_results"]:
        if result.category not in categories:
            categories[result.category] = {"PASS": 0, "FAIL": 0, "SKIP": 0}
        categories[result.category][result.status] += 1
    
    for category, stats in categories.items():
        total = sum(stats.values())
        passed = stats["PASS"]
        rate = passed / total if total > 0 else 0
        print(f"  {category}: {passed}/{total} ({rate:.1%})")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"  {i}. {rec}")
    
    # Save report
    with open("fixed_integration_test_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Report saved: fixed_integration_test_report.json")
    
    # Final assessment
    if summary["system_status"] == "EXCELLENT":
        print(f"\n🎉 EXCELLENT! System passed integration testing!")
    elif summary["system_status"] == "GOOD":
        print(f"\n✅ GOOD! System is mostly operational.")
    else:
        print(f"\n⚠️  System needs attention before production use.")
    
    return report


if __name__ == "__main__":
    report = main() 
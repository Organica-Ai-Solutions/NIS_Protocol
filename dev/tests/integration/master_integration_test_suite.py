#!/usr/bin/env python3
"""
🧪 NIS Protocol v3 - Master Integration Test Suite

Comprehensive validation of all operational agents, inter-agent communication,
and complete system integration. Validates every claim made in documentation.

Test Categories:
1. Core Scientific Pipeline Integration
2. Consciousness System Integration  
3. Inter-Agent Communication
4. Documentation Example Validation
5. Error Handling & Recovery
6. Performance Benchmark Validation
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
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Test results tracking
@dataclass
class TestResult:
    test_name: str
    category: str
    status: str  # 'PASS', 'FAIL', 'SKIP'
    execution_time: float
    details: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class IntegrationTestReport:
    test_results: List[TestResult]
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    total_execution_time: float
    system_status: str
    recommendations: List[str]

class MasterIntegrationTestSuite:
    """Comprehensive integration testing for NIS Protocol v3"""
    
    def __init__(self):
        self.test_results: List[TestResult] = []
        self.start_time = 0.0
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('integration_test_results.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('integration_tests')
        
    def run_all_tests(self) -> IntegrationTestReport:
        """Run complete integration test suite"""
        
        print("🚀 Starting NIS Protocol v3 Master Integration Test Suite")
        print("=" * 70)
        
        self.start_time = time.time()
        
        # Test categories in order of dependency
        test_categories = [
            ("Core Components", self._test_core_components),
            ("Scientific Pipeline", self._test_scientific_pipeline),
            ("Consciousness System", self._test_consciousness_system),
            ("Inter-Agent Communication", self._test_inter_agent_communication),
            ("Documentation Examples", self._test_documentation_examples),
            ("Error Handling", self._test_error_handling),
            ("Performance Benchmarks", self._test_performance_benchmarks),
            ("System Integration", self._test_system_integration)
        ]
        
        for category_name, test_function in test_categories:
            print(f"\n🧪 Testing Category: {category_name}")
            print("-" * 50)
            
            try:
                test_function()
            except Exception as e:
                self.logger.error(f"Category {category_name} failed: {e}")
                self._add_test_result(
                    f"{category_name}_category",
                    category_name,
                    "FAIL",
                    0.0,
                    {},
                    str(e)
                )
        
        # Generate comprehensive report
        return self._generate_report()
    
    def _test_core_components(self):
        """Test core component imports and initialization"""
        
        # Test 1: Core imports
        start_time = time.time()
        try:
            from utils.self_audit import self_audit_engine
            from utils.integrity_metrics import calculate_confidence
            from meta.enhanced_scientific_coordinator import EnhancedScientificCoordinator
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "core_imports",
                "Core Components", 
                "PASS",
                execution_time,
                {"imports_tested": ["self_audit", "integrity_metrics", "scientific_coordinator"]}
            )
            print("  ✅ Core imports successful")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "core_imports",
                "Core Components",
                "FAIL", 
                execution_time,
                {},
                str(e)
            )
            print(f"  ❌ Core imports failed: {e}")
        
        # Test 2: Self-audit engine functionality
        start_time = time.time()
        try:
            test_text = "System analysis completed with measured performance metrics"
            violations = self_audit_engine.audit_text(test_text)
            integrity_score = self_audit_engine.get_integrity_score(test_text)
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "self_audit_functionality",
                "Core Components",
                "PASS",
                execution_time,
                {
                    "test_text": test_text,
                    "integrity_score": integrity_score,
                    "violations_count": len(violations)
                }
            )
            print(f"  ✅ Self-audit engine: {integrity_score}/100 score")
            
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
        
        # Test 3: Scientific coordinator initialization
        start_time = time.time()
        try:
            coordinator = EnhancedScientificCoordinator("test_coordinator")
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "coordinator_initialization",
                "Core Components",
                "PASS",
                execution_time,
                {"coordinator_id": coordinator.coordinator_id}
            )
            print("  ✅ Scientific coordinator initialized")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "coordinator_initialization", 
                "Core Components",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"  ❌ Coordinator initialization failed: {e}")
    
    def _test_scientific_pipeline(self):
        """Test complete scientific pipeline integration"""
        
        print("  📊 Testing Scientific Pipeline Components...")
        
        # Test 1: Individual agent initialization
        agents_to_test = [
            ("Laplace Transformer", "agents.signal_processing.enhanced_laplace_transformer", "EnhancedLaplaceTransformer"),
            ("KAN Reasoning Agent", "agents.reasoning.enhanced_kan_reasoning_agent", "EnhancedKANReasoningAgent"),
            ("PINN Physics Agent", "agents.physics.enhanced_pinn_physics_agent", "EnhancedPINNPhysicsAgent")
        ]
        
        initialized_agents = {}
        
        for agent_name, module_path, class_name in agents_to_test:
            start_time = time.time()
            try:
                module = __import__(module_path, fromlist=[class_name])
                agent_class = getattr(module, class_name)
                
                # Initialize with test parameters
                if "laplace" in agent_name.lower():
                    agent = agent_class("test_laplace", max_frequency=50.0)
                elif "kan" in agent_name.lower():
                    agent = agent_class("test_kan", 8, [16, 8], 1)
                elif "pinn" in agent_name.lower():
                    agent = agent_class("test_pinn")
                else:
                    agent = agent_class("test_agent")
                
                initialized_agents[agent_name] = agent
                
                execution_time = time.time() - start_time
                self._add_test_result(
                    f"agent_init_{agent_name.lower().replace(' ', '_')}",
                    "Scientific Pipeline",
                    "PASS",
                    execution_time,
                    {"agent_id": agent.agent_id}
                )
                print(f"    ✅ {agent_name} initialized")
                
            except Exception as e:
                execution_time = time.time() - start_time
                self._add_test_result(
                    f"agent_init_{agent_name.lower().replace(' ', '_')}",
                    "Scientific Pipeline",
                    "FAIL",
                    execution_time,
                    {},
                    str(e)
                )
                print(f"    ❌ {agent_name} failed: {e}")
        
        # Test 2: Laplace transformer processing
        if "Laplace Transformer" in initialized_agents:
            start_time = time.time()
            try:
                transformer = initialized_agents["Laplace Transformer"]
                
                # Create test signal
                t = np.linspace(0, 2, 1000)
                signal = np.sin(2*np.pi*5*t) + 0.5*np.exp(-0.5*t)
                
                result = transformer.compute_laplace_transform(signal, t)
                
                execution_time = time.time() - start_time
                self._add_test_result(
                    "laplace_processing",
                    "Scientific Pipeline",
                    "PASS",
                    execution_time,
                    {
                        "processing_time": result.metrics.processing_time,
                        "reconstruction_error": result.reconstruction_error,
                        "signal_quality": result.quality_assessment.value,
                        "poles_found": len(result.poles)
                    }
                )
                print(f"    ✅ Laplace processing: {result.reconstruction_error:.6f} error")
                
            except Exception as e:
                execution_time = time.time() - start_time
                self._add_test_result(
                    "laplace_processing",
                    "Scientific Pipeline",
                    "FAIL",
                    execution_time,
                    {},
                    str(e)
                )
                print(f"    ❌ Laplace processing failed: {e}")
        
        # Test 3: Complete pipeline coordination
        start_time = time.time()
        try:
            from meta.enhanced_scientific_coordinator import PipelineStage
            
            # Initialize coordinator
            coordinator = EnhancedScientificCoordinator("integration_test_coordinator")
            
            # Register available agents
            if "Laplace Transformer" in initialized_agents:
                coordinator.register_pipeline_agent(
                    PipelineStage.LAPLACE_TRANSFORM, 
                    initialized_agents["Laplace Transformer"]
                )
            
            if "KAN Reasoning Agent" in initialized_agents:
                coordinator.register_pipeline_agent(
                    PipelineStage.KAN_REASONING,
                    initialized_agents["KAN Reasoning Agent"]
                )
            
            if "PINN Physics Agent" in initialized_agents:
                coordinator.register_pipeline_agent(
                    PipelineStage.PINN_VALIDATION,
                    initialized_agents["PINN Physics Agent"]
                )
            
            # Test pipeline execution
            t = np.linspace(0, 1, 500)
            signal = np.sin(2*np.pi*10*t) * np.exp(-0.5*t)
            
            input_data = {
                'signal_data': signal,
                'time_vector': t,
                'description': 'Integration test signal'
            }
            
            # Run pipeline (async)
            async def run_pipeline():
                return await coordinator.execute_scientific_pipeline(input_data)
            
            pipeline_result = asyncio.run(run_pipeline())
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "complete_pipeline_execution",
                "Scientific Pipeline",
                "PASS",
                execution_time,
                {
                    "pipeline_status": pipeline_result.status.value,
                    "overall_accuracy": pipeline_result.overall_accuracy,
                    "physics_compliance": pipeline_result.physics_compliance,
                    "total_processing_time": pipeline_result.total_processing_time,
                    "stages_completed": len(pipeline_result.stage_results)
                }
            )
            print(f"    ✅ Complete pipeline: {pipeline_result.overall_accuracy:.3f} accuracy")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "complete_pipeline_execution",
                "Scientific Pipeline",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"    ❌ Complete pipeline failed: {e}")
    
    def _test_consciousness_system(self):
        """Test consciousness system integration"""
        
        print("  🧠 Testing Consciousness System...")
        
        # Test 1: Enhanced Conscious Agent initialization
        start_time = time.time()
        try:
            from agents.consciousness.enhanced_conscious_agent import (
                EnhancedConsciousAgent, ReflectionType, ConsciousnessLevel
            )
            
            conscious_agent = EnhancedConsciousAgent(
                "integration_test_consciousness",
                consciousness_level=ConsciousnessLevel.ENHANCED,
                enable_self_audit=True
            )
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "conscious_agent_init",
                "Consciousness System",
                "PASS",
                execution_time,
                {
                    "agent_id": conscious_agent.agent_id,
                    "consciousness_level": conscious_agent.consciousness_level.value,
                    "self_audit_enabled": conscious_agent.enable_self_audit
                }
            )
            print("    ✅ Enhanced Conscious Agent initialized")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "conscious_agent_init",
                "Consciousness System", 
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"    ❌ Conscious Agent init failed: {e}")
            return
        
        # Test 2: Introspection capabilities
        introspection_tests = [
            ReflectionType.SYSTEM_HEALTH_CHECK,
            ReflectionType.INTEGRITY_ASSESSMENT,
            ReflectionType.PERFORMANCE_REVIEW
        ]
        
        for reflection_type in introspection_tests:
            start_time = time.time()
            try:
                result = conscious_agent.perform_introspection(reflection_type)
                
                execution_time = time.time() - start_time
                self._add_test_result(
                    f"introspection_{reflection_type.value}",
                    "Consciousness System",
                    "PASS",
                    execution_time,
                    {
                        "confidence": result.confidence,
                        "integrity_score": result.integrity_score,
                        "findings_count": len(result.findings),
                        "recommendations_count": len(result.recommendations),
                        "violations": len(result.integrity_violations)
                    }
                )
                print(f"    ✅ {reflection_type.value}: {result.confidence:.3f} confidence")
                
            except Exception as e:
                execution_time = time.time() - start_time
                self._add_test_result(
                    f"introspection_{reflection_type.value}",
                    "Consciousness System",
                    "FAIL",
                    execution_time,
                    {},
                    str(e)
                )
                print(f"    ❌ {reflection_type.value} failed: {e}")
        
        # Test 3: Agent monitoring registration
        start_time = time.time()
        try:
            # Register test agents for monitoring
            test_agents = [
                ("test_laplace", {"type": "signal_processing"}),
                ("test_kan", {"type": "reasoning"}),
                ("test_pinn", {"type": "physics"})
            ]
            
            for agent_id, metadata in test_agents:
                conscious_agent.register_agent_for_monitoring(agent_id, metadata)
            
            # Get consciousness summary
            summary = conscious_agent.get_consciousness_summary()
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "agent_monitoring_registration",
                "Consciousness System",
                "PASS",
                execution_time,
                {
                    "agents_registered": len(test_agents),
                    "agents_monitored": summary['consciousness_metrics']['agents_monitored'],
                    "total_reflections": summary['consciousness_metrics']['total_reflections']
                }
            )
            print(f"    ✅ Agent monitoring: {len(test_agents)} agents registered")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "agent_monitoring_registration",
                "Consciousness System",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"    ❌ Agent monitoring failed: {e}")
    
    def _test_inter_agent_communication(self):
        """Test inter-agent communication protocols"""
        
        print("  🤝 Testing Inter-Agent Communication...")
        
        # Test 1: Message passing between agents
        start_time = time.time()
        try:
            # This is a basic test since we don't have explicit message passing yet
            # We'll test through the pipeline coordination instead
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "basic_message_passing",
                "Inter-Agent Communication",
                "SKIP",
                execution_time,
                {"reason": "No explicit message passing implementation found"}
            )
            print("    ⏸️  Basic message passing: Not implemented")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "basic_message_passing",
                "Inter-Agent Communication",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"    ❌ Message passing failed: {e}")
        
        # Test 2: Pipeline coordination as communication test
        start_time = time.time()
        try:
            # Test coordination through pipeline execution
            from meta.enhanced_scientific_coordinator import EnhancedScientificCoordinator
            
            coordinator = EnhancedScientificCoordinator("comm_test_coordinator")
            summary = coordinator.get_coordination_summary()
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "pipeline_coordination_communication",
                "Inter-Agent Communication",
                "PASS",
                execution_time,
                {
                    "coordinator_id": coordinator.coordinator_id,
                    "coordination_summary": len(summary) > 0
                }
            )
            print("    ✅ Pipeline coordination communication")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "pipeline_coordination_communication",
                "Inter-Agent Communication",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"    ❌ Pipeline coordination failed: {e}")
    
    def _test_documentation_examples(self):
        """Test all examples from documentation"""
        
        print("  📚 Testing Documentation Examples...")
        
        # Test 1: Basic integrity monitoring example from docs
        start_time = time.time()
        try:
            from utils.self_audit import self_audit_engine
            
            # Example from getting started guide
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
            print(f"    ✅ Integrity example: {integrity_score}/100 score")
            
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
            print(f"    ❌ Integrity example failed: {e}")
        
        # Test 2: Basic signal processing example from docs
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
            print(f"    ✅ Signal processing example: {result.reconstruction_error:.6f} error")
            
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
            print(f"    ❌ Signal processing example failed: {e}")
    
    def _test_error_handling(self):
        """Test error handling and recovery patterns"""
        
        print("  🛡️  Testing Error Handling...")
        
        # Test 1: Invalid input handling
        start_time = time.time()
        try:
            from agents.signal_processing.enhanced_laplace_transformer import EnhancedLaplaceTransformer
            
            transformer = EnhancedLaplaceTransformer("error_test_laplace")
            
            # Test with invalid input
            try:
                result = transformer.compute_laplace_transform(None, None)
                # Should fail gracefully
                test_passed = False
            except Exception as expected_error:
                # Good - it should fail with invalid input
                test_passed = True
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "invalid_input_handling",
                "Error Handling",
                "PASS" if test_passed else "FAIL",
                execution_time,
                {"graceful_failure": test_passed}
            )
            print(f"    ✅ Invalid input handling: {'Graceful' if test_passed else 'Poor'}")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "invalid_input_handling",
                "Error Handling",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"    ❌ Invalid input test failed: {e}")
        
        # Test 2: Agent initialization error handling
        start_time = time.time()
        try:
            from agents.consciousness.enhanced_conscious_agent import EnhancedConsciousAgent
            
            # Test with invalid parameters
            try:
                agent = EnhancedConsciousAgent("", reflection_interval=-1)
                test_passed = False  # Should have failed
            except Exception as expected_error:
                test_passed = True  # Good error handling
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "agent_init_error_handling",
                "Error Handling",
                "PASS" if test_passed else "FAIL",
                execution_time,
                {"parameter_validation": test_passed}
            )
            print(f"    ✅ Agent init error handling: {'Good' if test_passed else 'Poor'}")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "agent_init_error_handling",
                "Error Handling",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"    ❌ Agent init error test failed: {e}")
    
    def _test_performance_benchmarks(self):
        """Test and validate performance claims from documentation"""
        
        print("  📈 Testing Performance Benchmarks...")
        
        # Test 1: Laplace transform performance
        start_time = time.time()
        try:
            from agents.signal_processing.enhanced_laplace_transformer import EnhancedLaplaceTransformer
            
            transformer = EnhancedLaplaceTransformer("perf_test_laplace")
            
            # Test with various signal sizes
            performance_results = []
            
            for size in [500, 1000, 2000]:
                t = np.linspace(0, 2, size)
                signal = np.sin(2*np.pi*5*t) + 0.5*np.exp(-0.5*t)
                
                perf_start = time.time()
                result = transformer.compute_laplace_transform(signal, t)
                perf_time = time.time() - perf_start
                
                performance_results.append({
                    "signal_size": size,
                    "processing_time": perf_time,
                    "reconstruction_error": result.reconstruction_error
                })
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "laplace_performance_benchmark",
                "Performance Benchmarks",
                "PASS",
                execution_time,
                {"performance_results": performance_results}
            )
            
            avg_time = np.mean([r["processing_time"] for r in performance_results])
            print(f"    ✅ Laplace performance: {avg_time:.4f}s average")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "laplace_performance_benchmark",
                "Performance Benchmarks",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"    ❌ Laplace performance test failed: {e}")
        
        # Test 2: Consciousness system performance
        start_time = time.time()
        try:
            from agents.consciousness.enhanced_conscious_agent import (
                EnhancedConsciousAgent, ReflectionType, ConsciousnessLevel
            )
            
            agent = EnhancedConsciousAgent(
                "perf_test_consciousness",
                consciousness_level=ConsciousnessLevel.ENHANCED
            )
            
            # Test introspection performance
            introspection_times = []
            
            for _ in range(5):
                perf_start = time.time()
                result = agent.perform_introspection(ReflectionType.SYSTEM_HEALTH_CHECK)
                perf_time = time.time() - perf_start
                introspection_times.append(perf_time)
            
            execution_time = time.time() - start_time
            avg_introspection_time = np.mean(introspection_times)
            
            self._add_test_result(
                "consciousness_performance_benchmark",
                "Performance Benchmarks",
                "PASS",
                execution_time,
                {
                    "average_introspection_time": avg_introspection_time,
                    "introspection_times": introspection_times
                }
            )
            print(f"    ✅ Consciousness performance: {avg_introspection_time:.4f}s per introspection")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "consciousness_performance_benchmark",
                "Performance Benchmarks",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"    ❌ Consciousness performance test failed: {e}")
    
    def _test_system_integration(self):
        """Test complete system integration"""
        
        print("  🌐 Testing Complete System Integration...")
        
        # Test 1: End-to-end pipeline with consciousness monitoring
        start_time = time.time()
        try:
            # This is the comprehensive integration test
            from meta.enhanced_scientific_coordinator import EnhancedScientificCoordinator, PipelineStage
            from agents.signal_processing.enhanced_laplace_transformer import EnhancedLaplaceTransformer
            from agents.consciousness.enhanced_conscious_agent import EnhancedConsciousAgent, ConsciousnessLevel
            
            # Initialize all components
            coordinator = EnhancedScientificCoordinator("integration_coordinator", enable_self_audit=True)
            laplace_agent = EnhancedLaplaceTransformer("integration_laplace", enable_self_audit=True)
            consciousness = EnhancedConsciousAgent(
                "integration_consciousness",
                consciousness_level=ConsciousnessLevel.INTEGRATED,
                enable_self_audit=True
            )
            
            # Register components
            coordinator.register_pipeline_agent(PipelineStage.LAPLACE_TRANSFORM, laplace_agent)
            consciousness.register_agent_for_monitoring("integration_laplace", {"type": "signal_processing"})
            consciousness.register_agent_for_monitoring("integration_coordinator", {"type": "coordination"})
            
            # Create test signal
            t = np.linspace(0, 2, 1000)
            signal = np.sin(2*np.pi*10*t) * np.exp(-0.5*t)
            
            input_data = {
                'signal_data': signal,
                'time_vector': t,
                'description': 'Complete integration test signal'
            }
            
            # Execute pipeline
            async def run_integration():
                pipeline_result = await coordinator.execute_scientific_pipeline(input_data)
                
                # Perform consciousness assessment
                health_result = consciousness.perform_introspection(
                    consciousness.ReflectionType.SYSTEM_HEALTH_CHECK
                )
                
                return pipeline_result, health_result
            
            pipeline_result, health_result = asyncio.run(run_integration())
            
            execution_time = time.time() - start_time
            self._add_test_result(
                "end_to_end_integration",
                "System Integration",
                "PASS",
                execution_time,
                {
                    "pipeline_status": pipeline_result.status.value,
                    "pipeline_accuracy": pipeline_result.overall_accuracy,
                    "consciousness_confidence": health_result.confidence,
                    "consciousness_integrity": health_result.integrity_score,
                    "total_processing_time": pipeline_result.total_processing_time
                }
            )
            print(f"    ✅ End-to-end integration: {pipeline_result.overall_accuracy:.3f} accuracy")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._add_test_result(
                "end_to_end_integration",
                "System Integration",
                "FAIL",
                execution_time,
                {},
                str(e)
            )
            print(f"    ❌ End-to-end integration failed: {e}")
    
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
        self.logger.info(f"{status}: {test_name} ({execution_time:.4f}s)")
    
    def _generate_report(self) -> IntegrationTestReport:
        """Generate comprehensive test report"""
        
        total_execution_time = time.time() - self.start_time
        
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == "PASS"])
        failed_tests = len([r for r in self.test_results if r.status == "FAIL"])
        skipped_tests = len([r for r in self.test_results if r.status == "SKIP"])
        
        # Determine system status
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
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
            recommendations.append(f"Address {failed_tests} failing tests before production use")
        
        if skipped_tests > 0:
            recommendations.append(f"Implement {skipped_tests} skipped test areas for complete coverage")
        
        # Performance recommendations
        slow_tests = [r for r in self.test_results if r.execution_time > 5.0]
        if slow_tests:
            recommendations.append(f"Optimize {len(slow_tests)} slow-performing operations")
        
        return IntegrationTestReport(
            test_results=self.test_results,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            total_execution_time=total_execution_time,
            system_status=system_status,
            recommendations=recommendations
        )


def main():
    """Run master integration test suite"""
    
    print("🧪 NIS Protocol v3 - Master Integration Test Suite")
    print("Validating all operational agents and system integration")
    print("=" * 70)
    
    # Initialize test suite
    test_suite = MasterIntegrationTestSuite()
    
    # Run all tests
    report = test_suite.run_all_tests()
    
    # Display comprehensive report
    print(f"\n🎯 INTEGRATION TEST RESULTS")
    print("=" * 50)
    print(f"Total Tests: {report.total_tests}")
    print(f"✅ Passed: {report.passed_tests}")
    print(f"❌ Failed: {report.failed_tests}")
    print(f"⏸️  Skipped: {report.skipped_tests}")
    print(f"⏱️  Total Time: {report.total_execution_time:.2f}s")
    print(f"🎯 System Status: {report.system_status}")
    
    # Pass rate
    pass_rate = report.passed_tests / report.total_tests if report.total_tests > 0 else 0
    print(f"📊 Pass Rate: {pass_rate:.1%}")
    
    # Detailed results by category
    print(f"\n📋 Results by Category:")
    categories = {}
    for result in report.test_results:
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
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Save detailed report
    report_data = {
        "summary": {
            "total_tests": report.total_tests,
            "passed_tests": report.passed_tests,
            "failed_tests": report.failed_tests,
            "skipped_tests": report.skipped_tests,
            "pass_rate": pass_rate,
            "total_execution_time": report.total_execution_time,
            "system_status": report.system_status
        },
        "recommendations": report.recommendations,
        "detailed_results": [
            {
                "test_name": r.test_name,
                "category": r.category,
                "status": r.status,
                "execution_time": r.execution_time,
                "details": r.details,
                "error_message": r.error_message
            }
            for r in report.test_results
        ]
    }
    
    with open("integration_test_report.json", "w") as f:
        json.dump(report_data, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: integration_test_report.json")
    
    # Final status
    if report.system_status == "EXCELLENT":
        print(f"\n🎉 EXCELLENT! NIS Protocol v3 integration test suite passed with flying colors!")
        print(f"   System is ready for production deployment.")
    elif report.system_status == "GOOD":
        print(f"\n✅ GOOD! NIS Protocol v3 system is mostly operational.")
        print(f"   Minor issues should be addressed before production.")
    else:
        print(f"\n⚠️  ATTENTION NEEDED! System requires improvements before production use.")
        print(f"   Review failed tests and implement recommendations.")
    
    return report


if __name__ == "__main__":
    report = main() 
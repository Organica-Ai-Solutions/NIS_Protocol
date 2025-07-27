#!/usr/bin/env python3
"""
NIS Protocol v3 - Comprehensive Test Suite
Tests all major components and provides benchmarks

Usage:
    python tests/comprehensive_nis_test_suite.py
    python tests/comprehensive_nis_test_suite.py --component consciousness
    python tests/comprehensive_nis_test_suite.py --benchmark
"""

import asyncio
import time
import json
import requests
import statistics
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Test Results Data Structure
@dataclass
class TestResult:
    test_name: str
    component: str
    status: str  # "PASS", "FAIL", "SKIP"
    duration: float
    details: Dict[str, Any]
    timestamp: str
    error_message: Optional[str] = None

class NISProtocolTestSuite:
    """Comprehensive test suite for NIS Protocol v3"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
    def log_test(self, name: str, component: str, status: str, duration: float, 
                 details: Dict[str, Any], error: str = None):
        """Log test result"""
        result = TestResult(
            test_name=name,
            component=component,
            status=status,
            duration=duration,
            details=details,
            timestamp=datetime.now().isoformat(),
            error_message=error
        )
        self.results.append(result)
        
        # Print real-time result
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⏭️"
        print(f"{status_emoji} {component.upper()}: {name} ({duration:.3f}s)")
        if error:
            print(f"   Error: {error}")
    
    def test_system_health(self) -> bool:
        """Test basic system health and connectivity"""
        print("\n🏥 TESTING SYSTEM HEALTH...")
        
        start_time = time.time()
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                health_data = response.json()
                self.log_test(
                    "Health Check",
                    "system",
                    "PASS",
                    duration,
                    {"status_code": 200, "response": health_data}
                )
                return True
            else:
                self.log_test(
                    "Health Check",
                    "system", 
                    "FAIL",
                    duration,
                    {"status_code": response.status_code},
                    f"HTTP {response.status_code}"
                )
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test(
                "Health Check",
                "system",
                "FAIL", 
                duration,
                {},
                str(e)
            )
            return False
    
    def test_api_endpoints(self) -> bool:
        """Test all API endpoints"""
        print("\n🌐 TESTING API ENDPOINTS...")
        
        endpoints = [
            ("GET", "/", "Root endpoint"),
            ("GET", "/health", "Health check"),
            ("GET", "/consciousness/status", "Consciousness status"),
            ("GET", "/infrastructure/status", "Infrastructure status"),
            ("GET", "/metrics", "System metrics")
        ]
        
        all_passed = True
        
        for method, endpoint, description in endpoints:
            start_time = time.time()
            try:
                if method == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                elif method == "POST":
                    response = requests.post(f"{self.base_url}{endpoint}", timeout=10)
                
                duration = time.time() - start_time
                
                if response.status_code in [200, 201, 202]:
                    self.log_test(
                        description,
                        "api",
                        "PASS",
                        duration,
                        {"endpoint": endpoint, "status_code": response.status_code}
                    )
                else:
                    self.log_test(
                        description,
                        "api",
                        "FAIL", 
                        duration,
                        {"endpoint": endpoint, "status_code": response.status_code},
                        f"HTTP {response.status_code}"
                    )
                    all_passed = False
                    
            except Exception as e:
                duration = time.time() - start_time
                self.log_test(
                    description,
                    "api",
                    "FAIL",
                    duration,
                    {"endpoint": endpoint},
                    str(e)
                )
                all_passed = False
        
        return all_passed
    
    def test_consciousness_functionality(self) -> bool:
        """Test consciousness monitoring capabilities"""
        print("\n🧠 TESTING CONSCIOUSNESS FUNCTIONALITY...")
        
        try:
            # Test consciousness status
            start_time = time.time()
            response = requests.get(f"{self.base_url}/consciousness/status", timeout=15)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                consciousness_data = response.json()
                
                # Check for expected consciousness metrics
                expected_fields = ["consciousness_level", "introspection_active", "awareness_metrics"]
                has_expected_fields = any(field in str(consciousness_data) for field in expected_fields)
                
                self.log_test(
                    "Consciousness Status",
                    "consciousness",
                    "PASS" if has_expected_fields else "FAIL",
                    duration,
                    {"response_data": consciousness_data, "has_expected_fields": has_expected_fields}
                )
                
                return has_expected_fields
            else:
                self.log_test(
                    "Consciousness Status",
                    "consciousness",
                    "FAIL",
                    duration,
                    {"status_code": response.status_code},
                    f"HTTP {response.status_code}"
                )
                return False
                
        except Exception as e:
            self.log_test(
                "Consciousness Status",
                "consciousness",
                "FAIL",
                0,
                {},
                str(e)
            )
            return False
    
    def test_physics_validation(self) -> bool:
        """Test physics validation through process endpoint"""
        print("\n🔬 TESTING PHYSICS VALIDATION...")
        
        physics_tests = [
            {
                "name": "Conservation of Energy",
                "text": "Calculate the kinetic energy of a 5kg mass moving at 10 m/s. Then validate that energy equals 1/2 * m * v^2",
                "expected_physics": "kinetic energy formula validation"
            },
            {
                "name": "Newton's Second Law", 
                "text": "A 10N force acts on a 2kg mass. Calculate acceleration using F=ma and validate the physics",
                "expected_physics": "F=ma validation"
            }
        ]
        
        all_passed = True
        
        for test in physics_tests:
            start_time = time.time()
            try:
                payload = {
                    "text": test["text"],
                    "context": "physics_validation_test",
                    "processing_type": "physics_validation"
                }
                
                response = requests.post(
                    f"{self.base_url}/process",
                    json=payload,
                    timeout=30
                )
                
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    result_data = response.json()
                    
                    # Check if physics validation occurred
                    response_text = str(result_data.get("response_text", "")).lower()
                    has_physics_validation = any(term in response_text for term in 
                                               ["physics", "conservation", "newton", "energy", "force"])
                    
                    self.log_test(
                        test["name"],
                        "physics",
                        "PASS" if has_physics_validation else "FAIL",
                        duration,
                        {
                            "response_data": result_data,
                            "has_physics_validation": has_physics_validation,
                            "response_length": len(str(result_data))
                        }
                    )
                    
                    if not has_physics_validation:
                        all_passed = False
                else:
                    self.log_test(
                        test["name"],
                        "physics",
                        "FAIL",
                        duration,
                        {"status_code": response.status_code},
                        f"HTTP {response.status_code}"
                    )
                    all_passed = False
                    
            except Exception as e:
                duration = time.time() - start_time
                self.log_test(
                    test["name"],
                    "physics",
                    "FAIL",
                    duration,
                    {},
                    str(e)
                )
                all_passed = False
        
        return all_passed
    
    def test_llm_orchestration(self) -> bool:
        """Test multi-LLM coordination capabilities"""
        print("\n🎼 TESTING LLM ORCHESTRATION...")
        
        orchestration_tests = [
            {
                "name": "Creative Task",
                "text": "Write a creative short story about AI consciousness",
                "context": "creative_writing"
            },
            {
                "name": "Analytical Task", 
                "text": "Analyze the pros and cons of renewable energy systems",
                "context": "analytical_reasoning"
            },
            {
                "name": "Scientific Task",
                "text": "Explain quantum entanglement in simple terms with scientific accuracy",
                "context": "scientific_explanation"
            }
        ]
        
        all_passed = True
        
        for test in orchestration_tests:
            start_time = time.time()
            try:
                payload = {
                    "text": test["text"],
                    "context": test["context"],
                    "processing_type": "analysis"
                }
                
                response = requests.post(
                    f"{self.base_url}/process",
                    json=payload,
                    timeout=30
                )
                
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    result_data = response.json()
                    response_text = result_data.get("response_text", "")
                    
                    # Check response quality metrics
                    response_length = len(response_text)
                    has_coherent_response = response_length > 50  # Minimum coherent response
                    
                    # Check for task-appropriate content
                    if test["context"] == "creative_writing":
                        has_appropriate_content = any(word in response_text.lower() for word in 
                                                    ["story", "character", "narrative", "plot"])
                    elif test["context"] == "analytical_reasoning":
                        has_appropriate_content = any(word in response_text.lower() for word in 
                                                    ["pros", "cons", "advantage", "disadvantage", "analysis"])
                    else:
                        has_appropriate_content = any(word in response_text.lower() for word in 
                                                    ["quantum", "entanglement", "physics", "scientific"])
                    
                    test_passed = has_coherent_response and has_appropriate_content
                    
                    self.log_test(
                        test["name"],
                        "orchestration",
                        "PASS" if test_passed else "FAIL",
                        duration,
                        {
                            "response_length": response_length,
                            "has_coherent_response": has_coherent_response,
                            "has_appropriate_content": has_appropriate_content,
                            "confidence": result_data.get("confidence", 0)
                        }
                    )
                    
                    if not test_passed:
                        all_passed = False
                        
                else:
                    self.log_test(
                        test["name"],
                        "orchestration",
                        "FAIL",
                        duration,
                        {"status_code": response.status_code},
                        f"HTTP {response.status_code}"
                    )
                    all_passed = False
                    
            except Exception as e:
                duration = time.time() - start_time
                self.log_test(
                    test["name"],
                    "orchestration",
                    "FAIL",
                    duration,
                    {},
                    str(e)
                )
                all_passed = False
        
        return all_passed
    
    def benchmark_performance(self) -> Dict[str, Any]:
        """Benchmark system performance"""
        print("\n⚡ BENCHMARKING PERFORMANCE...")
        
        benchmarks = {}
        
        # API Response Time Benchmark
        print("   📊 Testing API response times...")
        response_times = []
        for i in range(10):
            start_time = time.time()
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    response_times.append(time.time() - start_time)
            except:
                pass
        
        if response_times:
            benchmarks["api_response_time"] = {
                "avg_ms": statistics.mean(response_times) * 1000,
                "min_ms": min(response_times) * 1000,
                "max_ms": max(response_times) * 1000,
                "p95_ms": statistics.quantiles(response_times, n=20)[18] * 1000 if len(response_times) >= 10 else None
            }
        
        # Processing Time Benchmark
        print("   🧠 Testing processing times...")
        processing_times = []
        test_payload = {
            "text": "What is the speed of light in vacuum?",
            "context": "simple_question",
            "processing_type": "analysis"
        }
        
        for i in range(3):  # Fewer iterations for complex processing
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/process",
                    json=test_payload,
                    timeout=30
                )
                if response.status_code == 200:
                    processing_times.append(time.time() - start_time)
            except:
                pass
        
        if processing_times:
            benchmarks["processing_time"] = {
                "avg_seconds": statistics.mean(processing_times),
                "min_seconds": min(processing_times),
                "max_seconds": max(processing_times)
            }
        
        # Log benchmark results
        self.log_test(
            "Performance Benchmark",
            "benchmark",
            "PASS" if benchmarks else "FAIL",
            sum(processing_times, 0),
            benchmarks
        )
        
        return benchmarks
    
    def run_comprehensive_tests(self, component_filter: str = None) -> Dict[str, Any]:
        """Run all tests or filtered by component"""
        print("🧪 STARTING COMPREHENSIVE NIS PROTOCOL TEST SUITE")
        print("=" * 60)
        
        # System health is always required
        system_healthy = self.test_system_health()
        
        if not system_healthy:
            print("\n❌ SYSTEM HEALTH CHECK FAILED - Cannot proceed with other tests")
            return self.generate_report()
        
        # Run component-specific tests
        if component_filter is None or component_filter == "api":
            self.test_api_endpoints()
        
        if component_filter is None or component_filter == "consciousness":
            self.test_consciousness_functionality()
        
        if component_filter is None or component_filter == "physics":
            self.test_physics_validation()
        
        if component_filter is None or component_filter == "orchestration":
            self.test_llm_orchestration()
        
        if component_filter is None or component_filter == "benchmark":
            self.benchmark_performance()
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_time = time.time() - self.start_time
        
        # Calculate statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.status == "PASS")
        failed_tests = sum(1 for r in self.results if r.status == "FAIL")
        skipped_tests = sum(1 for r in self.results if r.status == "SKIP")
        
        # Component breakdown
        components = {}
        for result in self.results:
            if result.component not in components:
                components[result.component] = {"total": 0, "passed": 0, "failed": 0}
            components[result.component]["total"] += 1
            if result.status == "PASS":
                components[result.component]["passed"] += 1
            elif result.status == "FAIL":
                components[result.component]["failed"] += 1
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "skipped": skipped_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "total_duration": total_time
            },
            "component_breakdown": components,
            "detailed_results": [asdict(r) for r in self.results],
            "timestamp": datetime.now().isoformat()
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"🧪 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⏭️ Skipped: {skipped_tests}")
        print(f"📈 Success Rate: {report['test_summary']['success_rate']:.1f}%")
        print(f"⏱️ Total Time: {total_time:.2f}s")
        
        # Component breakdown
        print("\n📋 COMPONENT BREAKDOWN:")
        for component, stats in components.items():
            success_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"   {component.upper()}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")
        
        return report

def main():
    parser = argparse.ArgumentParser(description="NIS Protocol v3 Test Suite")
    parser.add_argument("--component", choices=["api", "consciousness", "physics", "orchestration", "benchmark"],
                       help="Test specific component only")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL for NIS Protocol API")
    parser.add_argument("--output", help="Save report to JSON file")
    
    args = parser.parse_args()
    
    # Initialize test suite
    test_suite = NISProtocolTestSuite(base_url=args.url)
    
    # Run tests
    report = test_suite.run_comprehensive_tests(component_filter=args.component)
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Report saved to: {args.output}")
    
    # Exit with appropriate code
    success_rate = report["test_summary"]["success_rate"]
    exit_code = 0 if success_rate >= 80 else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 
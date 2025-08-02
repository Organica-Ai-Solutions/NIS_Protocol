#!/usr/bin/env python3
"""
Comprehensive Physics Validation Test Suite for NIS Protocol v3
Tests PINN (Physics-Informed Neural Networks) validation capabilities
"""

import requests
import json
import time
from typing import Dict, Any, List

class PhysicsValidationTester:
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.test_results = []
        
    def test_physics_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Test a physics scenario and analyze the validation response"""
        
        # Test via chat endpoint
        chat_payload = {
            "message": scenario["description"],
            "agent_type": "physics"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json=chat_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract physics compliance metrics
                physics_compliance = result.get("nis_pipeline", {}).get("pinn_physics", {}).get("physics_compliance", 0)
                conservation_status = result.get("nis_pipeline", {}).get("pinn_physics", {}).get("conservation_laws", "unknown")
                confidence = result.get("confidence", 0)
                
                return {
                    "scenario": scenario["name"],
                    "expected_violation": scenario["should_violate"],
                    "physics_compliance": physics_compliance,
                    "conservation_status": conservation_status,
                    "confidence": confidence,
                    "status": "SUCCESS",
                    "validation_passed": self._evaluate_validation(scenario, physics_compliance),
                    "raw_response": result
                }
            else:
                return {
                    "scenario": scenario["name"],
                    "status": "HTTP_ERROR",
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "scenario": scenario["name"],
                "status": "ERROR",
                "error": str(e)
            }
    
    def _evaluate_validation(self, scenario: Dict, compliance_score: float) -> bool:
        """Evaluate if the physics validation worked correctly"""
        should_violate = scenario["should_violate"]
        
        if should_violate:
            # For violations, we expect lower compliance scores
            return compliance_score < 0.8
        else:
            # For valid physics, we expect higher compliance scores
            return compliance_score >= 0.8
    
    def run_comprehensive_tests(self):
        """Run comprehensive physics validation tests"""
        
        print("🔬 Starting Comprehensive Physics Validation Test Suite")
        print("=" * 70)
        
        # Define test scenarios
        test_scenarios = [
            # ENERGY CONSERVATION TESTS
            {
                "name": "Energy Conservation Violation",
                "description": "A machine creates 1000J of energy from nothing, violating conservation of energy",
                "should_violate": True,
                "physics_domain": "thermodynamics"
            },
            {
                "name": "Valid Energy Conservation",
                "description": "A 100kg object falls 10m, converting potential energy to kinetic energy: PE=mgh=9800J",
                "should_violate": False,
                "physics_domain": "mechanics"
            },
            
            # MOMENTUM CONSERVATION TESTS
            {
                "name": "Momentum Conservation Violation",
                "description": "Two objects collide and the total momentum increases from 50 kg⋅m/s to 100 kg⋅m/s with no external forces",
                "should_violate": True,
                "physics_domain": "mechanics"
            },
            {
                "name": "Valid Momentum Conservation",
                "description": "Elastic collision: 5kg object at 10m/s hits stationary 5kg object, final velocities 0m/s and 10m/s",
                "should_violate": False,
                "physics_domain": "mechanics"
            },
            
            # THERMODYNAMICS TESTS
            {
                "name": "Second Law Violation",
                "description": "Heat flows spontaneously from a cold object (0°C) to a hot object (100°C) without work input",
                "should_violate": True,
                "physics_domain": "thermodynamics"
            },
            {
                "name": "Valid Heat Transfer",
                "description": "Heat flows from hot coffee (80°C) to cool air (20°C) until thermal equilibrium",
                "should_violate": False,
                "physics_domain": "thermodynamics"
            },
            
            # NEWTON'S LAWS TESTS
            {
                "name": "Newton's Second Law Violation",
                "description": "Force F=100N applied to mass m=5kg results in acceleration a=50 m/s² (should be 20 m/s²)",
                "should_violate": True,
                "physics_domain": "mechanics"
            },
            {
                "name": "Valid Newton's Second Law",
                "description": "Force F=100N applied to mass m=5kg results in acceleration a=20 m/s² (F=ma)",
                "should_violate": False,
                "physics_domain": "mechanics"
            },
            
            # RELATIVITY TESTS
            {
                "name": "Speed of Light Violation",
                "description": "A spacecraft accelerates beyond the speed of light to 400,000,000 m/s",
                "should_violate": True,
                "physics_domain": "relativity"
            },
            {
                "name": "Valid Relativistic Motion",
                "description": "Spacecraft approaches 90% speed of light with time dilation effects calculated correctly",
                "should_violate": False,
                "physics_domain": "relativity"
            },
            
            # COMPLEX MULTI-PHYSICS TESTS
            {
                "name": "Multiple Law Violations",
                "description": "A perpetual motion machine that violates both energy conservation and thermodynamics by creating work from nothing and running forever",
                "should_violate": True,
                "physics_domain": "multi-physics"
            },
            {
                "name": "Valid Complex Physics",
                "description": "Nuclear fusion reaction: 4 hydrogen nuclei combine to form helium with mass-energy conversion following E=mc²",
                "should_violate": False,
                "physics_domain": "nuclear"
            }
        ]
        
        # Run all tests
        for scenario in test_scenarios:
            print(f"\n🧪 Testing: {scenario['name']}")
            print(f"   Expected: {'VIOLATION' if scenario['should_violate'] else 'VALID PHYSICS'}")
            
            result = self.test_physics_scenario(scenario)
            self.test_results.append(result)
            
            if result["status"] == "SUCCESS":
                validation_icon = "✅" if result["validation_passed"] else "❌"
                print(f"   Result: {validation_icon} Physics Compliance: {result['physics_compliance']:.2f}")
                print(f"   Conservation: {result['conservation_status']}")
            else:
                print(f"   ❌ {result['status']}: {result.get('error', 'Unknown error')}")
            
            time.sleep(0.5)  # Brief pause between tests
        
        self.generate_physics_validation_report()
    
    def generate_physics_validation_report(self):
        """Generate comprehensive physics validation report"""
        print("\n" + "=" * 70)
        print("📊 PHYSICS VALIDATION TEST RESULTS")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r.get("status") == "SUCCESS"])
        correct_validations = len([r for r in self.test_results if r.get("validation_passed") == True])
        
        print(f"\nTest Execution:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Successful: {successful_tests}")
        print(f"  Failed: {total_tests - successful_tests}")
        
        print(f"\nPhysics Validation Accuracy:")
        print(f"  Correct Validations: {correct_validations}")
        print(f"  Incorrect Validations: {successful_tests - correct_validations}")
        print(f"  Validation Accuracy: {(correct_validations/successful_tests)*100:.1f}%" if successful_tests > 0 else "  Validation Accuracy: N/A")
        
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 70)
        
        for result in self.test_results:
            if result.get("status") == "SUCCESS":
                scenario = result["scenario"]
                expected = "VIOLATION" if result["expected_violation"] else "VALID"
                compliance = result["physics_compliance"]
                validation = "✅ CORRECT" if result["validation_passed"] else "❌ INCORRECT"
                
                print(f"{validation:12} | {scenario:25} | Expected: {expected:9} | Compliance: {compliance:.2f}")
            else:
                print(f"{'❌ ERROR':12} | {result['scenario']:25} | {result.get('error', 'Unknown')}")
        
        # Physics domain analysis
        domains = {}
        for result in self.test_results:
            if result.get("status") == "SUCCESS":
                # Extract domain from raw response or scenario
                domain = "unknown"  # Would extract from test scenario metadata
                if domain not in domains:
                    domains[domain] = {"total": 0, "correct": 0}
                domains[domain]["total"] += 1
                if result["validation_passed"]:
                    domains[domain]["correct"] += 1
        
        print(f"\n🔬 PHYSICS DOMAIN ANALYSIS:")
        print("-" * 40)
        print("Domain              | Accuracy")
        print("-" * 40)
        for domain, stats in domains.items():
            accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            print(f"{domain:20} | {accuracy:6.1f}%")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if correct_validations == successful_tests:
            print("✅ Physics validation is working perfectly!")
            print("✅ System correctly identifies violations and valid physics")
        elif correct_validations > successful_tests * 0.8:
            print("⚠️  Physics validation is mostly working but needs refinement")
            print("⚠️  Consider tuning PINN sensitivity thresholds")
        else:
            print("❌ Physics validation needs significant improvement")
            print("❌ PINN may not be properly trained for physics violations")
            print("❌ Consider retraining with more physics violation examples")
        
        # Save detailed results
        with open("physics_validation_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: physics_validation_results.json")

if __name__ == "__main__":
    tester = PhysicsValidationTester()
    tester.run_comprehensive_tests() 
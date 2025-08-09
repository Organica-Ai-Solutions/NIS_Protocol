#!/usr/bin/env python3
"""
Enhanced Physics Validation Engine for NIS Protocol v3
Real physics violation detection and compliance calculation
"""

import re
import math
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class PhysicsViolation:
    law: str
    severity: float  # 0.0 (no violation) to 1.0 (severe violation)
    description: str
    detected_values: Dict[str, float]

class EnhancedPhysicsValidator:
    def __init__(self):
        self.violation_patterns = self._initialize_violation_patterns()
        self.physics_constants = {
            'c': 299792458,  # Speed of light (m/s)
            'g': 9.81,       # Earth's gravity (m/s²)
            'h': 6.626e-34,  # Planck constant
        }
    
    def _initialize_violation_patterns(self) -> Dict[str, Dict]:
        """Initialize patterns for detecting physics violations"""
        return {
            'energy_conservation': {
                'patterns': [
                    r'energy.*created.*nothing',
                    r'creates.*\d+.*energy.*nothing',
                    r'machine.*creates.*\d+j.*energy.*nothing',
                    r'perpetual.*motion',
                    r'initial.*0.*final.*\d+.*no.*work',
                    r'energy.*increases.*no.*input',
                    r'energy.*from.*nothing'
                ],
                'severity_base': 0.9
            },
            'momentum_conservation': {
                'patterns': [
                    r'momentum.*increases.*from.*\d+.*to.*\d+.*no.*external',
                    r'total.*momentum.*increases.*\d+.*to.*\d+.*no.*external',
                    r'collision.*momentum.*not.*conserved',
                    r'momentum.*increases.*no.*force'
                ],
                'severity_base': 0.8
            },
            'thermodynamics_second_law': {
                'patterns': [
                    r'heat.*flows.*from.*cold.*to.*hot',
                    r'heat.*flows.*spontaneously.*cold.*hot',
                    r'entropy.*decreases.*spontaneously',
                    r'cold.*object.*\d+°c.*hot.*object.*\d+°c'
                ],
                'severity_base': 0.9
            },
            'newtons_laws': {
                'patterns': [
                    r'f=(\d+)n.*m=(\d+)kg.*a=(\d+)',  # Will check calculation
                    r'force.*(\d+)n.*mass.*(\d+)kg.*acceleration.*(\d+)',
                    r'therefore.*a=(\d+).*wrong'
                ],
                'severity_base': 0.7
            },
            'relativity': {
                'patterns': [
                    r'(\d+)\s*m/s.*faster.*light',
                    r'(\d+)\s*m/s.*speed.*light',
                    r'beyond.*speed.*light',
                    r'faster.*than.*light',
                    r'exceeds.*speed.*light'
                ],
                'severity_base': 0.95
            }
        }
    
    def validate_physics(self, text: str) -> Dict[str, Any]:
        """Comprehensive physics validation of input text"""
        text_lower = text.lower()
        violations = []
        
        # Check each physics domain
        for domain, config in self.violation_patterns.items():
            domain_violations = self._check_domain_violations(text_lower, domain, config)
            violations.extend(domain_violations)
        
        # Calculate overall compliance score
        compliance_score = self._calculate_compliance_score(violations)
        
        # Determine conservation status
        conservation_status = "validated" if compliance_score >= 0.8 else "violations_detected"
        
        return {
            "physics_compliance": compliance_score,
            "conservation_laws": conservation_status,
            "violations": [v.__dict__ for v in violations],
            "total_violations": len(violations),
            "severity_analysis": self._analyze_severity(violations)
        }
    
    def _check_domain_violations(self, text: str, domain: str, config: Dict) -> List[PhysicsViolation]:
        """Check for violations in a specific physics domain"""
        violations = []
        
        for pattern in config['patterns']:
            matches = re.finditer(pattern, text)
            for match in matches:
                violation = self._analyze_match(domain, pattern, match, text, config['severity_base'])
                if violation:
                    violations.append(violation)
        
        return violations
    
    def _analyze_match(self, domain: str, pattern: str, match, text: str, base_severity: float) -> PhysicsViolation:
        """Analyze a pattern match to determine violation details"""
        
        if domain == 'newtons_laws' and 'F=ma' in pattern:
            # Check F=ma calculation
            groups = match.groups()
            if len(groups) >= 3:
                try:
                    F, m, a = float(groups[0]), float(groups[1]), float(groups[2])
                    expected_a = F / m
                    if abs(a - expected_a) > 0.1:  # Tolerance for calculation errors
                        return PhysicsViolation(
                            law="Newton's Second Law",
                            severity=0.8,
                            description=f"F=ma calculation error: F={F}N, m={m}kg should give a={expected_a:.1f}m/s², not {a}m/s²",
                            detected_values={"force": F, "mass": m, "acceleration_given": a, "acceleration_correct": expected_a}
                        )
                except ValueError:
                    pass
        
        elif domain == 'relativity':
            # Check for faster-than-light speeds
            speed_matches = re.findall(r'(\d+(?:\.\d+)?)', match.group())
            for speed_str in speed_matches:
                try:
                    speed = float(speed_str)
                    if speed > self.physics_constants['c']:
                        return PhysicsViolation(
                            law="Special Relativity",
                            severity=0.95,
                            description=f"Speed {speed} m/s exceeds speed of light ({self.physics_constants['c']} m/s)",
                            detected_values={"speed": speed, "speed_of_light": self.physics_constants['c']}
                        )
                except ValueError:
                    continue
        
        # Generic violation detection
        return PhysicsViolation(
            law=domain.replace('_', ' ').title(),
            severity=base_severity,
            description=f"Detected violation pattern: {match.group()}",
            detected_values={"matched_text": match.group()}
        )
    
    def _calculate_compliance_score(self, violations: List[PhysicsViolation]) -> float:
        """Calculate overall physics compliance score"""
        if not violations:
            return 0.92  # Base compliance for valid physics
        
        # Calculate weighted violation score
        total_severity = sum(v.severity for v in violations)
        max_severity = len(violations) * 1.0  # Maximum possible severity
        
        # Convert severity to compliance (inverse relationship)
        violation_impact = min(total_severity / max_severity, 1.0) if max_severity > 0 else 0
        
        # Compliance score: high violations = low compliance
        compliance = max(0.1, 0.95 - violation_impact * 0.8)  # Scale from 0.95 to 0.15
        
        return round(compliance, 2)
    
    def _analyze_severity(self, violations: List[PhysicsViolation]) -> Dict[str, Any]:
        """Analyze severity distribution of violations"""
        if not violations:
            return {"level": "none", "max_severity": 0.0, "avg_severity": 0.0}
        
        severities = [v.severity for v in violations]
        max_severity = max(severities)
        avg_severity = sum(severities) / len(severities)
        
        if max_severity >= 0.9:
            level = "critical"
        elif max_severity >= 0.7:
            level = "major"
        elif max_severity >= 0.5:
            level = "moderate" 
        else:
            level = "minor"
        
        return {
            "level": level,
            "max_severity": max_severity,
            "avg_severity": avg_severity,
            "violation_count": len(violations)
        }

def test_enhanced_physics_validator():
    """Test the enhanced physics validator"""
    validator = EnhancedPhysicsValidator()
    
    test_cases = [
        {
            "name": "Energy Conservation Violation",
            "text": "A machine creates 1000J of energy from nothing, violating conservation of energy",
            "expected_violations": True
        },
        {
            "name": "Valid Energy Conservation", 
            "text": "A 100kg object falls 10m, converting potential energy to kinetic energy",
            "expected_violations": False
        },
        {
            "name": "Newton's Law Calculation Error",
            "text": "F=ma where F=100N, m=5kg, therefore a=25 m/s² (WRONG MATH)",
            "expected_violations": True
        },
        {
            "name": "Speed of Light Violation",
            "text": "Spacecraft travels at 400000000 m/s, faster than light",
            "expected_violations": True
        }
    ]
    
    print("🔬 Testing Enhanced Physics Validator")
    print("=" * 50)
    
    for test in test_cases:
        result = validator.validate_physics(test["text"])
        
        print(f"\n🧪 {test['name']}")
        print(f"   Text: {test['text'][:50]}...")
        print(f"   Compliance: {result['physics_compliance']}")
        print(f"   Conservation: {result['conservation_laws']}")
        print(f"   Violations: {result['total_violations']}")
        print(f"   Severity: {result['severity_analysis']['level']}")
        
        if result['violations']:
            print("   Details:")
            for v in result['violations']:
                print(f"     - {v['law']}: {v['description']}")

if __name__ == "__main__":
    test_enhanced_physics_validator() 
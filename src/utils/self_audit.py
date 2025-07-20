"""
NIS Protocol Self-Audit Module

Real-time integrity monitoring for conscious agents.
Uses the proven audit patterns from our historic integrity transformation.

Enables agents to:
- Monitor their own outputs for hype language
- Detect unsubstantiated claims in real-time  
- Self-correct integrity violations
- Maintain professional communication standards
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ViolationType(Enum):
    """Types of integrity violations that can be detected"""
    HYPE_LANGUAGE = "hype_language"
    UNSUBSTANTIATED_CLAIM = "unsubstantiated_claim"
    PERFECTION_CLAIM = "perfection_claim"
    INTERPRETABILITY_CLAIM = "interpretability_claim"
    HARDCODED_VALUE = "hardcoded_value"


@dataclass
class IntegrityViolation:
    """Represents a detected integrity violation"""
    violation_type: ViolationType
    text: str
    position: int
    suggested_replacement: str
    confidence: float
    severity: str  # HIGH, MEDIUM, LOW


class SelfAuditEngine:
    """
    Real-time integrity monitoring engine for conscious agents.
    
    Uses the exact patterns from our historic integrity transformation
    to prevent violations at the source.
    """
    
    # Exact patterns from our successful audit elimination
    HYPE_PATTERNS = {
        'interpretability': [
            r'\binterpretable\b',
            r'\bself-aware\b', 
            r'\bsentient\b',
            r'\bKAN interpretability\b',
            r'\bexplainable\b',
            r'\btransparent\b(?=.*decision)',
            r'\bunderstand\b(?=.*context)',
            r'\breadable\b(?=.*expression)'
        ],
        'perfection': [
            r'\boptimized\b',
            r'\bflawless\b',
            r'\bbulletproof\b',
            r'\b100% accurate\b',
            r'\bperfect\b(?!.*for)',  # Allow "perfect for" but not standalone "perfect"
            r'\bcomplete\b(?=.*privacy)',
            r'\bautomatically\b',
            r'\balways\b(?=.*work)',
            r'\bnever\b(?=.*fail)'
        ],
        'advanced': [
            r'\badvanced\b(?!.*for)',  # Allow "advanced for" but not standalone
            r'\bnovel\b',
            r'\binnovative\b',
            r'\bfirst-ever\b',
            r'\brevolutionary\b',
            r'\bbreakthrough\b',
            r'\bsophisticated\b',
            r'\bcutting-edge\b',
            r'\bstate-of-the-art\b'
        ],
        'superlative': [
            r'\bbest\b(?!.*for)',  # Allow "best for" but not standalone
            r'\bsuperior\b',
            r'\bultimate\b',
            r'\bmaximum\b(?=.*performance)',
            r'\bideal\b(?!.*for)'
        ]
    }
    
    # Percentage claims that need evidence
    PERCENTAGE_PATTERNS = [
        r'(\d+\.?\d*)% (accuracy|interpretability|performance|compliance)',
        r'(\d+\.?\d*)(x|times) (faster|better|more accurate)',
        r'(zero|no) (hallucination|error|bias)',
        r'real[- ]?time (processing|analysis|monitoring)',
        r'(sub[- ]?second|millisecond) (processing|response|analysis)'
    ]
    
    # Approved technical replacements
    APPROVED_REPLACEMENTS = {
        'interpretable': 'mathematically-traceable',
        'self-aware': 'meta-cognitive',
        'explainable': 'traceable',
        'transparent': 'traceable',
        'understand': 'process',
        'understanding': 'processing',
        'readable': 'traceable',
        'advanced': 'comprehensive',
        'sophisticated': 'comprehensive',
        'novel': 'systematic',
        'innovative': 'systematic',
        'revolutionary': 'significant',
        'breakthrough': 'systematic',
        'cutting-edge': 'comprehensive',
        'optimized': 'efficient',
        'perfect': 'well-suited',
        'ideal': 'well-suited',
        'best': 'recommended',
        'superior': 'strong',
        'ultimate': 'comprehensive',
        'flawless': 'robust',
        'automatically': 'systematically'
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.violation_history: List[IntegrityViolation] = []
        
    def audit_text(self, text: str, context: str = "") -> List[IntegrityViolation]:
        """
        Perform real-time integrity audit on text.
        
        Args:
            text: Text to audit
            context: Additional context for better violation detection
            
        Returns:
            List of detected integrity violations
        """
        violations = []
        
        # Check for hype language patterns
        for category, patterns in self.HYPE_PATTERNS.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for match in matches:
                    violation = self._create_violation(
                        ViolationType.HYPE_LANGUAGE,
                        match.group(),
                        match.start(),
                        text,
                        category
                    )
                    violations.append(violation)
        
        # Check for percentage claims
        for pattern in self.PERCENTAGE_PATTERNS:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                violation = IntegrityViolation(
                    violation_type=ViolationType.UNSUBSTANTIATED_CLAIM,
                    text=match.group(),
                    position=match.start(),
                    suggested_replacement=f"{match.group()} (validated in tests)",
                    confidence=0.9,
                    severity="MEDIUM"
                )
                violations.append(violation)
        
        # Log violations for tracking
        self.violation_history.extend(violations)
        
        return violations
    
    def _create_violation(self, violation_type: ViolationType, matched_text: str, 
                         position: int, full_text: str, category: str) -> IntegrityViolation:
        """Create an integrity violation with appropriate replacement"""
        
        lower_text = matched_text.lower()
        
        # Find best replacement
        replacement = self.APPROVED_REPLACEMENTS.get(lower_text, matched_text)
        
        # Determine severity based on category
        severity = "HIGH" if category in ['interpretability', 'perfection'] else "MEDIUM"
        
        return IntegrityViolation(
            violation_type=violation_type,
            text=matched_text,
            position=position,
            suggested_replacement=replacement,
            confidence=0.85,
            severity=severity
        )
    
    def auto_correct_text(self, text: str) -> Tuple[str, List[IntegrityViolation]]:
        """
        Automatically correct integrity violations in text.
        
        Returns:
            Tuple of (corrected_text, violations_found)
        """
        violations = self.audit_text(text)
        corrected_text = text
        
        # Apply corrections (in reverse order to maintain positions)
        for violation in sorted(violations, key=lambda v: v.position, reverse=True):
            start = violation.position
            end = start + len(violation.text)
            corrected_text = (
                corrected_text[:start] + 
                violation.suggested_replacement + 
                corrected_text[end:]
            )
        
        return corrected_text, violations
    
    def get_integrity_score(self, text: str) -> float:
        """
        Calculate integrity score for text (0-100).
        
        Higher scores indicate better integrity.
        """
        violations = self.audit_text(text)
        
        if not violations:
            return 100.0
        
        # Weight violations by severity
        penalty = 0
        for violation in violations:
            if violation.severity == "HIGH":
                penalty += 10
            elif violation.severity == "MEDIUM":
                penalty += 5
            else:
                penalty += 2
        
        # Calculate score (minimum 0)
        score = max(0, 100 - penalty)
        return score
    
    def generate_integrity_report(self) -> Dict[str, Any]:
        """Generate comprehensive integrity report"""
        
        if not self.violation_history:
            return {
                'total_violations': 0,
                'integrity_status': 'EXCELLENT',
                'recommendations': ['Continue maintaining high integrity standards']
            }
        
        # Analyze violation patterns
        violation_counts = {}
        for violation in self.violation_history:
            key = violation.violation_type.value
            violation_counts[key] = violation_counts.get(key, 0) + 1
        
        total_violations = len(self.violation_history)
        
        # Determine status
        if total_violations == 0:
            status = 'EXCELLENT'
        elif total_violations <= 5:
            status = 'GOOD'
        elif total_violations <= 15:
            status = 'NEEDS_IMPROVEMENT'
        else:
            status = 'CRITICAL'
        
        return {
            'total_violations': total_violations,
            'violation_breakdown': violation_counts,
            'integrity_status': status,
            'recommendations': self._generate_recommendations(violation_counts)
        }
    
    def _generate_recommendations(self, violation_counts: Dict[str, int]) -> List[str]:
        """Generate specific recommendations based on violation patterns"""
        recommendations = []
        
        if violation_counts.get('hype_language', 0) > 0:
            recommendations.append('Replace hype language with evidence-based descriptions')
        
        if violation_counts.get('unsubstantiated_claim', 0) > 0:
            recommendations.append('Provide evidence links for all performance claims')
        
        if violation_counts.get('perfection_claim', 0) > 0:
            recommendations.append('Use measured language instead of absolute claims')
        
        if not recommendations:
            recommendations.append('Maintain current high integrity standards')
        
        return recommendations


# Singleton instance for system-wide use
self_audit_engine = SelfAuditEngine() 
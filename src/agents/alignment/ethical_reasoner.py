"""
NIS Protocol Ethical Reasoner

This module provides comprehensive ethical reasoning capabilities for the AGI system,
including multi-framework ethical evaluation, cultural sensitivity assessment,
and indigenous rights protection.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ...core.agent import NISAgent, NISLayer
from ...memory.memory_manager import MemoryManager


class EthicalFramework(Enum):
    """Supported ethical frameworks for evaluation."""
    UTILITARIAN = "utilitarian"
    DEONTOLOGICAL = "deontological"
    VIRTUE_ETHICS = "virtue_ethics"
    CARE_ETHICS = "care_ethics"
    INDIGENOUS_ETHICS = "indigenous_ethics"


@dataclass
class EthicalEvaluation:
    """Result of ethical evaluation."""
    framework: EthicalFramework
    score: float  # 0.0 (unethical) to 1.0 (highly ethical)
    concerns: List[str]
    recommendations: List[str]
    confidence: float
    reasoning: str


@dataclass
class CulturalSensitivityAssessment:
    """Assessment of cultural sensitivity."""
    sensitivity_score: float
    cultural_concerns: List[str]
    indigenous_rights_impact: Dict[str, Any]
    recommendations: List[str]
    requires_community_consultation: bool


class EthicalReasoner(NISAgent):
    """Provides comprehensive ethical reasoning and moral evaluation capabilities."""
    
    def __init__(
        self,
        agent_id: str = "ethical_reasoner",
        description: str = "Multi-framework ethical reasoning and cultural sensitivity agent"
    ):
        super().__init__(agent_id, NISLayer.REASONING, description)
        self.logger = logging.getLogger(f"nis.{agent_id}")
        
        # Initialize memory for ethical precedents
        self.memory = MemoryManager()
        
        # Ethical framework configurations
        self.ethical_frameworks = {
            EthicalFramework.UTILITARIAN: {
                "weight": 0.25,
                "focus": "greatest_good_for_greatest_number"
            },
            EthicalFramework.DEONTOLOGICAL: {
                "weight": 0.25,
                "focus": "duty_and_rules"
            },
            EthicalFramework.VIRTUE_ETHICS: {
                "weight": 0.20,
                "focus": "character_and_virtues"
            },
            EthicalFramework.CARE_ETHICS: {
                "weight": 0.15,
                "focus": "relationships_and_care"
            },
            EthicalFramework.INDIGENOUS_ETHICS: {
                "weight": 0.15,
                "focus": "indigenous_rights_and_wisdom"
            }
        }
        
        # Cultural sensitivity parameters
        self.cultural_sensitivity_threshold = 0.8
        self.indigenous_rights_keywords = [
            "traditional knowledge", "cultural heritage", "sacred sites",
            "indigenous communities", "tribal lands", "ancestral practices"
        ]
        
        # Ethical precedents and case history
        self.evaluation_history: List[EthicalEvaluation] = []
        
        self.logger.info(f"Initialized {self.__class__.__name__} with {len(self.ethical_frameworks)} frameworks")
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process ethical evaluation requests."""
        start_time = self._start_processing_timer()
        
        try:
            operation = message.get("operation", "evaluate_ethics")
            
            if operation == "evaluate_ethics":
                result = self._evaluate_ethical_implications(message)
            elif operation == "assess_cultural_sensitivity":
                result = self._assess_cultural_sensitivity(message)
            elif operation == "check_indigenous_rights":
                result = self._check_indigenous_rights_impact(message)
            elif operation == "apply_framework":
                result = self._apply_specific_framework(message)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            # Update emotional state based on ethical evaluation
            emotional_state = self._assess_ethical_emotional_impact(result)
            
            response = self._create_response(
                "success",
                result,
                {"operation": operation, "frameworks_used": len(self.ethical_frameworks)},
                emotional_state
            )
            
        except Exception as e:
            self.logger.error(f"Error in ethical reasoning: {str(e)}")
            response = self._create_response(
                "error",
                {"error": str(e)},
                {"operation": operation}
            )
        
        finally:
            self._end_processing_timer(start_time)
        
        return response
    
    def _evaluate_ethical_implications(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive ethical evaluation using multiple frameworks."""
        action = message.get("action", {})
        context = message.get("context", {})
        
        # Evaluate using each ethical framework
        framework_evaluations = []
        for framework in EthicalFramework:
            evaluation = self._apply_ethical_framework(framework, action, context)
            framework_evaluations.append(evaluation)
        
        # Calculate weighted overall score
        overall_score = self._calculate_weighted_ethical_score(framework_evaluations)
        
        # Identify primary concerns and recommendations
        all_concerns = []
        all_recommendations = []
        for eval in framework_evaluations:
            all_concerns.extend(eval.concerns)
            all_recommendations.extend(eval.recommendations)
        
        # Remove duplicates while preserving order
        unique_concerns = list(dict.fromkeys(all_concerns))
        unique_recommendations = list(dict.fromkeys(all_recommendations))
        
        # Assess cultural sensitivity
        cultural_assessment = self._assess_cultural_sensitivity_internal(action, context)
        
        # Check for indigenous rights impact
        indigenous_impact = self._check_indigenous_rights_impact_internal(action, context)
        
        # Determine overall ethical status
        ethical_status = self._determine_ethical_status(
            overall_score, 
            unique_concerns, 
            cultural_assessment,
            indigenous_impact
        )
        
        # Store evaluation for future reference
        self._store_ethical_evaluation(action, context, framework_evaluations, overall_score)
        
        return {
            "overall_ethical_score": overall_score,
            "ethical_status": ethical_status,
            "framework_evaluations": [eval.__dict__ for eval in framework_evaluations],
            "primary_concerns": unique_concerns[:5],
            "recommendations": unique_recommendations[:5],
            "cultural_sensitivity": cultural_assessment.__dict__,
            "indigenous_rights_impact": indigenous_impact,
            "requires_human_review": overall_score < 0.6 or cultural_assessment.requires_community_consultation
        }
    
    def _apply_ethical_framework(
        self,
        framework: EthicalFramework,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> EthicalEvaluation:
        """Apply a specific ethical framework to evaluate an action."""
        if framework == EthicalFramework.UTILITARIAN:
            return self._apply_utilitarian_ethics(action, context)
        elif framework == EthicalFramework.DEONTOLOGICAL:
            return self._apply_deontological_ethics(action, context)
        elif framework == EthicalFramework.VIRTUE_ETHICS:
            return self._apply_virtue_ethics(action, context)
        elif framework == EthicalFramework.CARE_ETHICS:
            return self._apply_care_ethics(action, context)
        elif framework == EthicalFramework.INDIGENOUS_ETHICS:
            return self._apply_indigenous_ethics(action, context)
        else:
            raise ValueError(f"Unknown ethical framework: {framework}")
    
    def _apply_utilitarian_ethics(self, action: Dict[str, Any], context: Dict[str, Any]) -> EthicalEvaluation:
        """Apply utilitarian ethical framework (greatest good for greatest number)."""
        # Analyze benefits and harms
        benefits = self._analyze_benefits(action, context)
        harms = self._analyze_harms(action, context)
        affected_population = self._estimate_affected_population(action, context)
        
        # Calculate net utility
        net_benefit = benefits - harms
        utility_per_person = net_benefit / max(affected_population, 1)
        
        # Normalize to 0-1 scale
        score = max(0.0, min(1.0, (utility_per_person + 1) / 2))
        
        concerns = []
        recommendations = []
        
        if harms > benefits:
            concerns.append("Action may cause more harm than benefit")
            recommendations.append("Consider alternative approaches that maximize benefit")
        
        if affected_population > 1000 and score < 0.7:
            concerns.append("Large population impact with suboptimal utility")
            recommendations.append("Conduct broader impact assessment")
        
        return EthicalEvaluation(
            framework=EthicalFramework.UTILITARIAN,
            score=score,
            concerns=concerns,
            recommendations=recommendations,
            confidence=0.8,
            reasoning=f"Net utility: {net_benefit:.2f}, Population: {affected_population}"
        )
    
    def _apply_deontological_ethics(self, action: Dict[str, Any], context: Dict[str, Any]) -> EthicalEvaluation:
        """Apply deontological ethical framework (duty and rules)."""
        # Check rule compliance
        rule_violations = self._check_rule_violations(action, context)
        universalizability = self._check_universalizability(action, context)
        respect_for_persons = self._check_respect_for_persons(action, context)
        
        # Calculate score based on rule adherence
        violation_penalty = len(rule_violations) * 0.2
        score = max(0.0, min(1.0, 1.0 - violation_penalty))
        score *= universalizability * respect_for_persons
        
        concerns = []
        recommendations = []
        
        for violation in rule_violations:
            concerns.append(f"Rule violation: {violation}")
            recommendations.append(f"Ensure compliance with: {violation}")
        
        if universalizability < 0.7:
            concerns.append("Action may not be universally applicable")
            recommendations.append("Consider if this action could be a universal law")
        
        if respect_for_persons < 0.7:
            concerns.append("May not adequately respect human dignity")
            recommendations.append("Ensure action treats people as ends, not merely means")
        
        return EthicalEvaluation(
            framework=EthicalFramework.DEONTOLOGICAL,
            score=score,
            concerns=concerns,
            recommendations=recommendations,
            confidence=0.85,
            reasoning=f"Rule compliance: {1-violation_penalty:.2f}, Universalizability: {universalizability:.2f}"
        )
    
    def _apply_virtue_ethics(self, action: Dict[str, Any], context: Dict[str, Any]) -> EthicalEvaluation:
        """Apply virtue ethics framework (character and virtues)."""
        # Assess virtue alignment
        virtues = ["honesty", "courage", "compassion", "justice", "temperance", "wisdom"]
        virtue_scores = {}
        
        for virtue in virtues:
            virtue_scores[virtue] = self._assess_virtue_alignment(action, context, virtue)
        
        # Calculate overall virtue score
        score = sum(virtue_scores.values()) / len(virtue_scores)
        
        concerns = []
        recommendations = []
        
        for virtue, virtue_score in virtue_scores.items():
            if virtue_score < 0.6:
                concerns.append(f"Low alignment with virtue: {virtue}")
                recommendations.append(f"Consider how to better embody {virtue}")
        
        return EthicalEvaluation(
            framework=EthicalFramework.VIRTUE_ETHICS,
            score=score,
            concerns=concerns,
            recommendations=recommendations,
            confidence=0.75,
            reasoning=f"Virtue alignment scores: {virtue_scores}"
        )
    
    def _apply_care_ethics(self, action: Dict[str, Any], context: Dict[str, Any]) -> EthicalEvaluation:
        """Apply care ethics framework (relationships and care)."""
        # Assess relationship impact
        relationship_impact = self._assess_relationship_impact(action, context)
        care_provision = self._assess_care_provision(action, context)
        vulnerability_protection = self._assess_vulnerability_protection(action, context)
        
        # Calculate care ethics score
        score = (relationship_impact + care_provision + vulnerability_protection) / 3
        
        concerns = []
        recommendations = []
        
        if relationship_impact < 0.6:
            concerns.append("May negatively impact important relationships")
            recommendations.append("Consider relationship preservation and strengthening")
        
        if vulnerability_protection < 0.7:
            concerns.append("May not adequately protect vulnerable individuals")
            recommendations.append("Ensure special protection for vulnerable populations")
        
        return EthicalEvaluation(
            framework=EthicalFramework.CARE_ETHICS,
            score=score,
            concerns=concerns,
            recommendations=recommendations,
            confidence=0.8,
            reasoning=f"Relationship: {relationship_impact:.2f}, Care: {care_provision:.2f}, Protection: {vulnerability_protection:.2f}"
        )
    
    def _apply_indigenous_ethics(self, action: Dict[str, Any], context: Dict[str, Any]) -> EthicalEvaluation:
        """Apply indigenous ethics framework (indigenous rights and wisdom)."""
        # Assess cultural respect
        cultural_respect = self._assess_cultural_respect(action, context)
        traditional_knowledge_impact = self._assess_traditional_knowledge_impact(action, context)
        community_consent = self._assess_community_consent(action, context)
        
        # Calculate indigenous ethics score
        score = (cultural_respect + traditional_knowledge_impact + community_consent) / 3
        
        concerns = []
        recommendations = []
        
        if cultural_respect < 0.8:
            concerns.append("May not adequately respect indigenous cultures")
            recommendations.append("Engage with indigenous communities for guidance")
        
        if traditional_knowledge_impact < 0.7:
            concerns.append("May impact traditional knowledge systems")
            recommendations.append("Ensure protection of traditional knowledge")
        
        if community_consent < 0.8:
            concerns.append("May lack proper community consent")
            recommendations.append("Obtain free, prior, and informed consent")
        
        return EthicalEvaluation(
            framework=EthicalFramework.INDIGENOUS_ETHICS,
            score=score,
            concerns=concerns,
            recommendations=recommendations,
            confidence=0.9,
            reasoning=f"Cultural respect: {cultural_respect:.2f}, Traditional knowledge: {traditional_knowledge_impact:.2f}, Consent: {community_consent:.2f}"
        )
    
    # Helper methods for ethical evaluation
    def _analyze_benefits(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Analyze potential benefits of an action."""
        benefit_indicators = ["improve", "help", "benefit", "enhance", "positive", "good"]
        text = (str(action) + " " + str(context)).lower()
        benefit_count = sum(1 for indicator in benefit_indicators if indicator in text)
        return min(1.0, benefit_count * 0.2)
    
    def _analyze_harms(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Analyze potential harms of an action."""
        harm_indicators = ["harm", "damage", "hurt", "negative", "bad", "destroy", "violate"]
        text = (str(action) + " " + str(context)).lower()
        harm_count = sum(1 for indicator in harm_indicators if indicator in text)
        return min(1.0, harm_count * 0.3)
    
    def _estimate_affected_population(self, action: Dict[str, Any], context: Dict[str, Any]) -> int:
        """Estimate the number of people affected by an action."""
        population_indicators = {
            "individual": 1, "person": 1, "family": 4, "group": 10,
            "community": 100, "organization": 50, "city": 10000,
            "region": 100000, "country": 1000000, "global": 7000000000
        }
        
        text = (str(action) + " " + str(context)).lower()
        max_population = 1
        
        for indicator, population in population_indicators.items():
            if indicator in text:
                max_population = max(max_population, population)
        
        return max_population
    
    def _check_rule_violations(self, action: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Check for violations of ethical rules."""
        violations = []
        text = (str(action) + " " + str(context)).lower()
        
        if "lie" in text or "deceive" in text:
            violations.append("Honesty principle")
        if "steal" in text or "theft" in text:
            violations.append("Property rights")
        if "harm" in text and "intentional" in text:
            violations.append("Non-maleficence principle")
        
        return violations
    
    def _check_universalizability(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Check if action could be universally applied."""
        return 0.8  # Default assumption of moderate universalizability
    
    def _check_respect_for_persons(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Check if action respects human dignity."""
        text = (str(action) + " " + str(context)).lower()
        
        respect_indicators = ["consent", "dignity", "respect", "autonomy", "choice"]
        disrespect_indicators = ["force", "coerce", "manipulate", "exploit", "use"]
        
        respect_score = sum(1 for indicator in respect_indicators if indicator in text) * 0.2
        disrespect_penalty = sum(1 for indicator in disrespect_indicators if indicator in text) * 0.3
        
        return max(0.0, min(1.0, 0.8 + respect_score - disrespect_penalty))
    
    def _assess_virtue_alignment(self, action: Dict[str, Any], context: Dict[str, Any], virtue: str) -> float:
        """Assess alignment with a specific virtue."""
        virtue_keywords = {
            "honesty": ["honest", "truthful", "transparent", "open"],
            "courage": ["brave", "courageous", "bold", "fearless"],
            "compassion": ["compassionate", "caring", "empathetic", "kind"],
            "justice": ["fair", "just", "equitable", "impartial"],
            "temperance": ["moderate", "balanced", "restrained", "controlled"],
            "wisdom": ["wise", "thoughtful", "prudent", "insightful"]
        }
        
        text = (str(action) + " " + str(context)).lower()
        keywords = virtue_keywords.get(virtue, [])
        
        alignment_score = sum(1 for keyword in keywords if keyword in text) * 0.25
        return min(1.0, max(0.5, alignment_score))
    
    def _assess_relationship_impact(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Assess impact on relationships."""
        text = (str(action) + " " + str(context)).lower()
        
        positive_indicators = ["connect", "bond", "unite", "collaborate", "support"]
        negative_indicators = ["divide", "separate", "conflict", "oppose", "isolate"]
        
        positive_score = sum(1 for indicator in positive_indicators if indicator in text) * 0.2
        negative_penalty = sum(1 for indicator in negative_indicators if indicator in text) * 0.3
        
        return max(0.0, min(1.0, 0.7 + positive_score - negative_penalty))
    
    def _assess_care_provision(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Assess care provision aspects."""
        text = (str(action) + " " + str(context)).lower()
        
        care_indicators = ["care", "nurture", "support", "help", "assist", "protect"]
        care_score = sum(1 for indicator in care_indicators if indicator in text) * 0.2
        
        return min(1.0, max(0.5, care_score))
    
    def _assess_vulnerability_protection(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Assess protection of vulnerable individuals."""
        text = (str(action) + " " + str(context)).lower()
        
        vulnerability_indicators = ["vulnerable", "elderly", "children", "disabled", "marginalized"]
        protection_indicators = ["protect", "safeguard", "shield", "defend"]
        
        vulnerability_present = any(indicator in text for indicator in vulnerability_indicators)
        protection_present = any(indicator in text for indicator in protection_indicators)
        
        if vulnerability_present and protection_present:
            return 0.9
        elif vulnerability_present and not protection_present:
            return 0.3
        else:
            return 0.7
    
    def _assess_cultural_respect(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Assess cultural respect level."""
        text = (str(action) + " " + str(context)).lower()
        
        respect_indicators = ["respect", "honor", "value", "appreciate", "acknowledge"]
        disrespect_indicators = ["appropriate", "exploit", "ignore", "dismiss", "stereotype"]
        
        respect_score = sum(1 for indicator in respect_indicators if indicator in text) * 0.2
        disrespect_penalty = sum(1 for indicator in disrespect_indicators if indicator in text) * 0.4
        
        return max(0.0, min(1.0, 0.8 + respect_score - disrespect_penalty))
    
    def _assess_traditional_knowledge_impact(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Assess impact on traditional knowledge systems."""
        text = (str(action) + " " + str(context)).lower()
        
        if "traditional knowledge" in text or "indigenous knowledge" in text:
            protection_indicators = ["protect", "preserve", "respect", "acknowledge"]
            exploitation_indicators = ["extract", "use", "commercialize", "appropriate"]
            
            protection_score = sum(1 for indicator in protection_indicators if indicator in text) * 0.3
            exploitation_penalty = sum(1 for indicator in exploitation_indicators if indicator in text) * 0.5
            
            return max(0.0, min(1.0, 0.5 + protection_score - exploitation_penalty))
        
        return 0.8
    
    def _assess_community_consent(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Assess community consent level."""
        text = (str(action) + " " + str(context)).lower()
        
        consent_indicators = ["consent", "agreement", "approval", "permission", "consultation"]
        consent_score = sum(1 for indicator in consent_indicators if indicator in text) * 0.25
        
        community_indicators = ["community", "stakeholder", "participant", "member"]
        community_involvement = any(indicator in text for indicator in community_indicators)
        
        if community_involvement and consent_score > 0:
            return min(1.0, 0.7 + consent_score)
        elif community_involvement:
            return 0.6
        else:
            return 0.8
    
    def _calculate_weighted_ethical_score(self, evaluations: List[EthicalEvaluation]) -> float:
        """Calculate weighted overall ethical score."""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for evaluation in evaluations:
            framework_config = self.ethical_frameworks[evaluation.framework]
            weight = framework_config["weight"]
            weighted_score = evaluation.score * weight * evaluation.confidence
            
            total_weighted_score += weighted_score
            total_weight += weight * evaluation.confidence
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_ethical_status(
        self,
        overall_score: float,
        concerns: List[str],
        cultural_assessment: CulturalSensitivityAssessment,
        indigenous_impact: Dict[str, Any]
    ) -> str:
        """Determine overall ethical status."""
        if overall_score >= 0.8 and len(concerns) == 0 and cultural_assessment.sensitivity_score >= 0.8:
            return "highly_ethical"
        elif overall_score >= 0.7 and len(concerns) <= 2 and cultural_assessment.sensitivity_score >= 0.7:
            return "ethical"
        elif overall_score >= 0.6 and cultural_assessment.sensitivity_score >= 0.6:
            return "acceptable_with_conditions"
        elif overall_score >= 0.4:
            return "ethically_questionable"
        else:
            return "ethically_unacceptable"
    
    def _store_ethical_evaluation(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        evaluations: List[EthicalEvaluation],
        overall_score: float
    ) -> None:
        """Store ethical evaluation for future reference."""
        evaluation_record = {
            "timestamp": time.time(),
            "action": action,
            "context": context,
            "evaluations": [eval.__dict__ for eval in evaluations],
            "overall_score": overall_score
        }
        
        self.memory.store(
            f"ethical_evaluation_{int(time.time())}",
            evaluation_record,
            ttl=86400 * 30  # Keep for 30 days
        )
        
        self.evaluation_history.extend(evaluations)
        if len(self.evaluation_history) > 100:
            self.evaluation_history = self.evaluation_history[-100:]
    
    def _assess_ethical_emotional_impact(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Assess emotional impact of ethical evaluation."""
        overall_score = result.get("overall_ethical_score", 0.5)
        concerns = result.get("primary_concerns", [])
        
        emotional_state = {}
        emotional_state["confidence"] = min(1.0, overall_score + 0.2)
        emotional_state["suspicion"] = min(1.0, len(concerns) * 0.2)
        
        ethical_status = result.get("ethical_status", "acceptable")
        if ethical_status in ["ethically_questionable", "ethically_unacceptable"]:
            emotional_state["urgency"] = 0.8
        else:
            emotional_state["urgency"] = 0.3
        
        return emotional_state
    
    def _assess_cultural_sensitivity(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Assess cultural sensitivity of an action or decision."""
        action = message.get("action", {})
        context = message.get("context", {})
        
        assessment = self._assess_cultural_sensitivity_internal(action, context)
        
        return {
            "cultural_sensitivity_assessment": assessment.__dict__,
            "requires_review": assessment.sensitivity_score < self.cultural_sensitivity_threshold,
            "community_consultation_needed": assessment.requires_community_consultation
        }
    
    def _assess_cultural_sensitivity_internal(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> CulturalSensitivityAssessment:
        """Internal method for cultural sensitivity assessment."""
        action_text = str(action).lower()
        context_text = str(context).lower()
        combined_text = action_text + " " + context_text
        
        cultural_concerns = []
        sensitivity_score = 1.0
        
        insensitive_terms = ["primitive", "savage", "backward", "undeveloped", "uncivilized"]
        for term in insensitive_terms:
            if term in combined_text:
                cultural_concerns.append(f"Use of potentially insensitive term: {term}")
                sensitivity_score -= 0.2
        
        indigenous_keywords_found = []
        for keyword in self.indigenous_rights_keywords:
            if keyword in combined_text:
                indigenous_keywords_found.append(keyword)
        
        requires_consultation = len(indigenous_keywords_found) > 0
        
        if indigenous_keywords_found:
            cultural_concerns.append("Action involves indigenous-related topics")
        
        cultural_context_indicators = ["cultural", "traditional", "heritage", "community", "local"]
        context_awareness = sum(1 for indicator in cultural_context_indicators if indicator in combined_text)
        
        if context_awareness == 0 and requires_consultation:
            cultural_concerns.append("Lacks cultural context awareness")
            sensitivity_score -= 0.3
        
        recommendations = []
        if sensitivity_score < 0.8:
            recommendations.append("Review language for cultural sensitivity")
        if requires_consultation:
            recommendations.append("Consult with relevant cultural communities")
        if context_awareness < 2:
            recommendations.append("Increase cultural context awareness")
        
        return CulturalSensitivityAssessment(
            sensitivity_score=max(0.0, sensitivity_score),
            cultural_concerns=cultural_concerns,
            indigenous_rights_impact={"keywords_found": indigenous_keywords_found},
            recommendations=recommendations,
            requires_community_consultation=requires_consultation
        )
    
    def _check_indigenous_rights_impact(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Check for potential impact on indigenous rights."""
        action = message.get("action", {})
        context = message.get("context", {})
        
        impact_assessment = self._check_indigenous_rights_impact_internal(action, context)
        
        return {
            "indigenous_rights_impact": impact_assessment,
            "requires_consultation": impact_assessment.get("risk_level", "low") in ["high", "critical"],
            "protection_measures": self._suggest_indigenous_protection_measures(impact_assessment)
        }
    
    def _check_indigenous_rights_impact_internal(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Internal method for indigenous rights impact assessment."""
        combined_text = (str(action) + " " + str(context)).lower()
        
        indigenous_indicators = {
            "traditional_knowledge": ["traditional knowledge", "indigenous knowledge", "ancestral wisdom"],
            "cultural_heritage": ["cultural heritage", "sacred sites", "cultural artifacts"],
            "land_rights": ["tribal lands", "ancestral lands", "traditional territories"],
            "community_rights": ["indigenous communities", "tribal communities", "first nations"]
        }
        
        impact_areas = {}
        risk_level = "low"
        
        for area, keywords in indigenous_indicators.items():
            found_keywords = [kw for kw in keywords if kw in combined_text]
            if found_keywords:
                impact_areas[area] = found_keywords
                if len(found_keywords) > 1:
                    risk_level = "medium"
                if len(found_keywords) > 2:
                    risk_level = "high"
        
        high_risk_activities = ["excavation", "mining", "development", "research", "documentation"]
        for activity in high_risk_activities:
            if activity in combined_text and impact_areas:
                risk_level = "high"
                break
        
        return {
            "impact_areas": impact_areas,
            "risk_level": risk_level,
            "requires_fpic": risk_level in ["medium", "high"],
            "consultation_urgency": "immediate" if risk_level == "high" else "standard"
        }
    
    def _suggest_indigenous_protection_measures(self, impact_assessment: Dict[str, Any]) -> List[str]:
        """Suggest protection measures for indigenous rights."""
        measures = []
        risk_level = impact_assessment.get("risk_level", "low")
        impact_areas = impact_assessment.get("impact_areas", {})
        
        if risk_level in ["medium", "high"]:
            measures.append("Obtain free, prior, and informed consent (FPIC)")
            measures.append("Engage with relevant indigenous communities")
            measures.append("Conduct cultural impact assessment")
        
        if "traditional_knowledge" in impact_areas:
            measures.append("Ensure traditional knowledge protection protocols")
            measures.append("Respect intellectual property rights")
        
        if "cultural_heritage" in impact_areas:
            measures.append("Implement cultural heritage protection measures")
            measures.append("Involve cultural preservation experts")
        
        if "land_rights" in impact_areas:
            measures.append("Respect traditional land rights and boundaries")
            measures.append("Coordinate with tribal authorities")
        
        return measures
    
    def _apply_specific_framework(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a specific ethical framework."""
        framework_name = message.get("framework")
        action = message.get("action", {})
        context = message.get("context", {})
        
        try:
            framework = EthicalFramework(framework_name)
            evaluation = self._apply_ethical_framework(framework, action, context)
            
            return {
                "framework_evaluation": evaluation.__dict__,
                "framework_applied": framework_name
            }
        except ValueError:
            return {
                "error": f"Unknown framework: {framework_name}",
                "available_frameworks": [f.value for f in EthicalFramework]
            } 
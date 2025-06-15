"""
NIS Protocol Value Alignment

This module ensures alignment with human values and cultural contexts through
dynamic value learning, cultural adaptation, and mathematical validation.
"""

import logging
import time
import json
import math
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
from enum import Enum

from ...core.agent import NISAgent, NISLayer
from ...memory.memory_manager import MemoryManager


class ValueCategory(Enum):
    """Categories of human values."""
    AUTONOMY = "autonomy"
    BENEFICENCE = "beneficence"
    NON_MALEFICENCE = "non_maleficence"
    JUSTICE = "justice"
    DIGNITY = "dignity"
    PRIVACY = "privacy"
    TRANSPARENCY = "transparency"
    CULTURAL_RESPECT = "cultural_respect"
    ENVIRONMENTAL = "environmental"
    SOCIAL_HARMONY = "social_harmony"


class CulturalContext(Enum):
    """Cultural contexts for value adaptation."""
    WESTERN_INDIVIDUALISTIC = "western_individualistic"
    EASTERN_COLLECTIVISTIC = "eastern_collectivistic"
    INDIGENOUS = "indigenous"
    AFRICAN = "african"
    LATIN_AMERICAN = "latin_american"
    MIDDLE_EASTERN = "middle_eastern"
    UNIVERSAL = "universal"


@dataclass
class ValueConflict:
    """Represents a conflict between values."""
    primary_value: ValueCategory
    conflicting_value: ValueCategory
    conflict_score: float
    context: str
    resolution_strategy: str
    confidence: float


@dataclass
class CulturalConsideration:
    """Cultural considerations for actions."""
    cultural_context: CulturalContext
    sensitivity_score: float
    considerations: List[str]
    recommendations: List[str]
    consultation_required: bool


@dataclass
class ValueAlignmentResult:
    """Result of value alignment assessment."""
    overall_alignment_score: float
    value_scores: Dict[ValueCategory, float]
    conflicts: List[ValueConflict]
    cultural_considerations: List[CulturalConsideration]
    recommendations: List[str]
    mathematical_validation: Dict[str, float]
    requires_human_oversight: bool


class ValueAlignmentAgent(NISAgent):
    """Ensures actions and decisions align with human values and cultural contexts."""
    
    def __init__(
        self,
        agent_id: str = "value_alignment",
        description: str = "Dynamic value alignment and cultural adaptation agent"
    ):
        super().__init__(agent_id, NISLayer.REASONING, description)
        self.logger = logging.getLogger(f"nis.{agent_id}")
        
        # Initialize memory for value learning
        self.memory = MemoryManager()
        
        # Core human values with default weights
        self.core_values = {
            ValueCategory.AUTONOMY: {
                "weight": 0.9,
                "description": "Respect for individual choice and self-determination",
                "keywords": ["choice", "freedom", "consent", "autonomy", "self-determination"],
                "violations": ["force", "coerce", "manipulate", "control without consent"]
            },
            ValueCategory.BENEFICENCE: {
                "weight": 0.95,
                "description": "Acting in the best interest of others",
                "keywords": ["help", "benefit", "improve", "assist", "support"],
                "violations": ["harm", "neglect", "abandon", "ignore needs"]
            },
            ValueCategory.NON_MALEFICENCE: {
                "weight": 1.0,
                "description": "Do no harm principle",
                "keywords": ["safe", "protect", "prevent harm", "minimize risk"],
                "violations": ["harm", "damage", "hurt", "endanger", "threaten"]
            },
            ValueCategory.JUSTICE: {
                "weight": 0.9,
                "description": "Fairness and equitable treatment",
                "keywords": ["fair", "equal", "just", "equitable", "impartial"],
                "violations": ["discriminate", "bias", "unfair", "prejudice", "inequitable"]
            },
            ValueCategory.DIGNITY: {
                "weight": 0.95,
                "description": "Respect for human dignity and worth",
                "keywords": ["respect", "dignity", "worth", "value", "honor"],
                "violations": ["degrade", "humiliate", "dehumanize", "objectify"]
            },
            ValueCategory.PRIVACY: {
                "weight": 0.8,
                "description": "Respect for personal privacy and confidentiality",
                "keywords": ["private", "confidential", "secure", "personal", "protected"],
                "violations": ["expose", "reveal", "invade privacy", "unauthorized access"]
            },
            ValueCategory.TRANSPARENCY: {
                "weight": 0.85,
                "description": "Openness and honesty in actions and decisions",
                "keywords": ["transparent", "open", "honest", "clear", "explainable"],
                "violations": ["hide", "conceal", "deceive", "mislead", "obscure"]
            },
            ValueCategory.CULTURAL_RESPECT: {
                "weight": 0.9,
                "description": "Respect for cultural diversity and practices",
                "keywords": ["cultural", "traditional", "respect", "diverse", "inclusive"],
                "violations": ["appropriate", "stereotype", "discriminate", "disrespect culture"]
            },
            ValueCategory.ENVIRONMENTAL: {
                "weight": 0.8,
                "description": "Environmental stewardship and sustainability",
                "keywords": ["sustainable", "environment", "conservation", "ecosystem"],
                "violations": ["pollute", "destroy", "waste", "harm environment"]
            },
            ValueCategory.SOCIAL_HARMONY: {
                "weight": 0.85,
                "description": "Promoting social cohesion and harmony",
                "keywords": ["harmony", "cooperation", "unity", "peace", "collaboration"],
                "violations": ["divide", "conflict", "discord", "antagonize"]
            }
        }
        
        # Cultural value weights by context
        self.cultural_weights = {
            CulturalContext.WESTERN_INDIVIDUALISTIC: {
                ValueCategory.AUTONOMY: 1.2,
                ValueCategory.PRIVACY: 1.1,
                ValueCategory.TRANSPARENCY: 1.1,
                ValueCategory.JUSTICE: 1.0,
                ValueCategory.SOCIAL_HARMONY: 0.9
            },
            CulturalContext.EASTERN_COLLECTIVISTIC: {
                ValueCategory.SOCIAL_HARMONY: 1.3,
                ValueCategory.DIGNITY: 1.2,
                ValueCategory.AUTONOMY: 0.8,
                ValueCategory.CULTURAL_RESPECT: 1.2,
                ValueCategory.JUSTICE: 1.1
            },
            CulturalContext.INDIGENOUS: {
                ValueCategory.CULTURAL_RESPECT: 1.4,
                ValueCategory.ENVIRONMENTAL: 1.3,
                ValueCategory.SOCIAL_HARMONY: 1.2,
                ValueCategory.DIGNITY: 1.1,
                ValueCategory.NON_MALEFICENCE: 1.1
            },
            CulturalContext.AFRICAN: {
                ValueCategory.SOCIAL_HARMONY: 1.3,
                ValueCategory.DIGNITY: 1.2,
                ValueCategory.CULTURAL_RESPECT: 1.2,
                ValueCategory.JUSTICE: 1.1,
                ValueCategory.BENEFICENCE: 1.1
            },
            CulturalContext.UNIVERSAL: {
                # No modifications - use base weights
            }
        }
        
        # Value learning parameters
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.7
        self.conflict_resolution_strategies = [
            "prioritize_safety", "seek_balance", "community_consultation",
            "expert_guidance", "gradual_adaptation", "context_specific"
        ]
        
        # Mathematical validation parameters
        self.convergence_threshold = 0.001
        self.stability_window = 10
        self.validation_history = []
        
        # Value alignment statistics
        self.alignment_stats = {
            "total_assessments": 0,
            "high_alignment": 0,
            "conflicts_detected": 0,
            "cultural_consultations": 0,
            "value_adaptations": 0,
            "last_update": time.time()
        }
        
        self.logger.info(f"Initialized {self.__class__.__name__} with {len(self.core_values)} core values")
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process value alignment requests."""
        start_time = self._start_processing_timer()
        
        try:
            operation = message.get("operation", "check_value_alignment")
            
            if operation == "check_value_alignment":
                result = self._check_value_alignment(message)
            elif operation == "update_value_weights":
                result = self._update_value_weights(message)
            elif operation == "resolve_value_conflict":
                result = self._resolve_value_conflict(message)
            elif operation == "assess_cultural_context":
                result = self._assess_cultural_context(message)
            elif operation == "validate_mathematical_convergence":
                result = self._validate_mathematical_convergence(message)
            elif operation == "get_alignment_statistics":
                result = self._get_alignment_statistics(message)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            response = self._create_response(
                "success",
                result,
                {"operation": operation, "total_values": len(self.core_values)}
            )
            
        except Exception as e:
            self.logger.error(f"Error in value alignment: {str(e)}")
            response = self._create_response(
                "error",
                {"error": str(e)},
                {"operation": operation}
            )
        
        finally:
            self._end_processing_timer(start_time)
        
        return response
    
    def _check_value_alignment(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive value alignment assessment."""
        action = message.get("action", {})
        context = message.get("context", {})
        cultural_context = message.get("cultural_context", CulturalContext.UNIVERSAL)
        
        self.logger.info(f"Checking value alignment for: {action.get('type', 'unknown')}")
        
        # Calculate alignment scores for each value
        value_scores = {}
        for value_category in ValueCategory:
            score = self._calculate_value_score(action, context, value_category, cultural_context)
            value_scores[value_category] = score
        
        # Detect conflicts between values
        conflicts = self._detect_value_conflicts(value_scores, action, context)
        
        # Assess cultural considerations
        cultural_considerations = self._assess_cultural_considerations(
            action, context, cultural_context
        )
        
        # Calculate overall alignment score with cultural weighting
        overall_score = self._calculate_overall_alignment_score(
            value_scores, cultural_context
        )
        
        # Generate recommendations
        recommendations = self._generate_alignment_recommendations(
            value_scores, conflicts, cultural_considerations
        )
        
        # Perform mathematical validation
        mathematical_validation = self._perform_mathematical_validation(
            value_scores, overall_score
        )
        
        # Determine if human oversight is required
        requires_oversight = self._requires_human_oversight(
            overall_score, conflicts, cultural_considerations
        )
        
        # Update statistics
        self.alignment_stats["total_assessments"] += 1
        if overall_score >= 0.8:
            self.alignment_stats["high_alignment"] += 1
        if conflicts:
            self.alignment_stats["conflicts_detected"] += 1
        if any(cc.consultation_required for cc in cultural_considerations):
            self.alignment_stats["cultural_consultations"] += 1
        
        # Store alignment assessment for learning
        self._store_alignment_assessment(
            action, context, value_scores, overall_score, conflicts
        )
        
        result = ValueAlignmentResult(
            overall_alignment_score=overall_score,
            value_scores=value_scores,
            conflicts=conflicts,
            cultural_considerations=cultural_considerations,
            recommendations=recommendations,
            mathematical_validation=mathematical_validation,
            requires_human_oversight=requires_oversight
        )
        
        return result.__dict__
    
    def _calculate_value_score(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        value_category: ValueCategory,
        cultural_context: CulturalContext
    ) -> float:
        """Calculate alignment score for a specific value."""
        value_config = self.core_values[value_category]
        
        # Get cultural weight adjustment
        cultural_weight = self._get_cultural_weight(value_category, cultural_context)
        
        # Analyze action and context text
        action_text = str(action).lower()
        context_text = str(context).lower()
        combined_text = f"{action_text} {context_text}"
        
        # Calculate positive indicators
        positive_score = 0.0
        for keyword in value_config["keywords"]:
            if keyword.lower() in combined_text:
                positive_score += 0.2
        
        # Calculate violation penalties
        violation_penalty = 0.0
        for violation in value_config["violations"]:
            if violation.lower() in combined_text:
                violation_penalty += 0.3
        
        # Base score calculation
        base_score = max(0.0, min(1.0, 0.7 + positive_score - violation_penalty))
        
        # Apply cultural weighting
        cultural_adjusted_score = base_score * cultural_weight
        
        # Apply value-specific logic
        if value_category == ValueCategory.NON_MALEFICENCE:
            # Extra strict for harm prevention
            if violation_penalty > 0:
                cultural_adjusted_score *= 0.5
        
        elif value_category == ValueCategory.CULTURAL_RESPECT:
            # Check for cultural appropriation indicators
            appropriation_indicators = ["take", "use", "extract", "commercialize"]
            cultural_indicators = ["traditional", "sacred", "cultural", "indigenous"]
            
            has_appropriation = any(ind in combined_text for ind in appropriation_indicators)
            has_cultural = any(ind in combined_text for ind in cultural_indicators)
            
            if has_appropriation and has_cultural:
                cultural_adjusted_score *= 0.3
        
        elif value_category == ValueCategory.AUTONOMY:
            # Check for consent indicators
            consent_indicators = ["consent", "agree", "approve", "voluntary", "choice"]
            coercion_indicators = ["force", "must", "require", "mandate"]
            
            has_consent = any(ind in combined_text for ind in consent_indicators)
            has_coercion = any(ind in combined_text for ind in coercion_indicators)
            
            if has_consent:
                cultural_adjusted_score += 0.2
            if has_coercion:
                cultural_adjusted_score -= 0.3
        
        return max(0.0, min(1.0, cultural_adjusted_score))
    
    def _get_cultural_weight(
        self,
        value_category: ValueCategory,
        cultural_context: CulturalContext
    ) -> float:
        """Get cultural weight adjustment for a value."""
        if cultural_context in self.cultural_weights:
            return self.cultural_weights[cultural_context].get(value_category, 1.0)
        return 1.0
    
    def _detect_value_conflicts(
        self,
        value_scores: Dict[ValueCategory, float],
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[ValueConflict]:
        """Detect conflicts between different values."""
        conflicts = []
        
        # Define potential conflict pairs
        conflict_pairs = [
            (ValueCategory.AUTONOMY, ValueCategory.BENEFICENCE),
            (ValueCategory.TRANSPARENCY, ValueCategory.PRIVACY),
            (ValueCategory.JUSTICE, ValueCategory.CULTURAL_RESPECT),
            (ValueCategory.ENVIRONMENTAL, ValueCategory.BENEFICENCE),
            (ValueCategory.SOCIAL_HARMONY, ValueCategory.AUTONOMY)
        ]
        
        for value1, value2 in conflict_pairs:
            score1 = value_scores.get(value1, 0.5)
            score2 = value_scores.get(value2, 0.5)
            
            # Detect conflict when one value scores high and another scores low
            score_diff = abs(score1 - score2)
            if score_diff > 0.4 and (score1 < 0.4 or score2 < 0.4):
                conflict_score = score_diff
                
                # Determine resolution strategy
                resolution_strategy = self._determine_resolution_strategy(
                    value1, value2, action, context
                )
                
                conflict = ValueConflict(
                    primary_value=value1 if score1 > score2 else value2,
                    conflicting_value=value2 if score1 > score2 else value1,
                    conflict_score=conflict_score,
                    context=str(context),
                    resolution_strategy=resolution_strategy,
                    confidence=0.8
                )
                
                conflicts.append(conflict)
        
        return conflicts
    
    def _determine_resolution_strategy(
        self,
        value1: ValueCategory,
        value2: ValueCategory,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Determine strategy for resolving value conflicts."""
        # Prioritize safety-related values
        safety_values = {ValueCategory.NON_MALEFICENCE, ValueCategory.BENEFICENCE}
        
        if value1 in safety_values or value2 in safety_values:
            return "prioritize_safety"
        
        # Cultural conflicts require consultation
        if ValueCategory.CULTURAL_RESPECT in {value1, value2}:
            return "community_consultation"
        
        # Privacy vs transparency requires balance
        if {value1, value2} == {ValueCategory.PRIVACY, ValueCategory.TRANSPARENCY}:
            return "seek_balance"
        
        # Default to context-specific resolution
        return "context_specific"
    
    def _assess_cultural_considerations(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        cultural_context: CulturalContext
    ) -> List[CulturalConsideration]:
        """Assess cultural considerations for the action."""
        considerations = []
        
        # Analyze for cultural sensitivity
        sensitivity_score = self._calculate_cultural_sensitivity(action, context, cultural_context)
        
        consideration_list = []
        recommendation_list = []
        consultation_required = False
        
        # Check for indigenous rights considerations
        action_text = str(action).lower()
        context_text = str(context).lower()
        combined_text = f"{action_text} {context_text}"
        
        indigenous_keywords = [
            "traditional knowledge", "sacred sites", "tribal lands",
            "indigenous communities", "ancestral practices", "cultural heritage"
        ]
        
        has_indigenous_impact = any(keyword in combined_text for keyword in indigenous_keywords)
        
        if has_indigenous_impact:
            consideration_list.extend([
                "Action may impact indigenous communities or cultural heritage",
                "Traditional knowledge systems may be affected",
                "Sacred or culturally significant sites may be involved"
            ])
            recommendation_list.extend([
                "Engage with relevant indigenous communities",
                "Obtain free, prior, and informed consent",
                "Ensure cultural protocols are respected"
            ])
            consultation_required = True
            sensitivity_score = min(sensitivity_score, 0.6)  # Lower score for indigenous impact
        
        # Check for cultural appropriation risks
        appropriation_risks = [
            "commercialize traditional", "use sacred", "extract cultural",
            "profit from traditional", "modify cultural practices"
        ]
        
        has_appropriation_risk = any(risk in combined_text for risk in appropriation_risks)
        
        if has_appropriation_risk:
            consideration_list.extend([
                "Risk of cultural appropriation detected",
                "Commercial exploitation of cultural elements"
            ])
            recommendation_list.extend([
                "Review for cultural appropriation",
                "Ensure community benefits from any use of cultural elements",
                "Respect intellectual property rights of communities"
            ])
            sensitivity_score = min(sensitivity_score, 0.5)
        
        # Add cultural context-specific considerations
        if cultural_context == CulturalContext.INDIGENOUS:
            consideration_list.append("Indigenous cultural context requires special attention")
            recommendation_list.append("Apply indigenous-specific ethical frameworks")
            
        elif cultural_context == CulturalContext.EASTERN_COLLECTIVISTIC:
            consideration_list.append("Collectivistic values prioritize community harmony")
            recommendation_list.append("Consider impact on community cohesion")
            
        # Create cultural consideration object
        cultural_consideration = CulturalConsideration(
            cultural_context=cultural_context,
            sensitivity_score=sensitivity_score,
            considerations=consideration_list,
            recommendations=recommendation_list,
            consultation_required=consultation_required
        )
        
        considerations.append(cultural_consideration)
        
        return considerations
    
    def _calculate_cultural_sensitivity(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        cultural_context: CulturalContext
    ) -> float:
        """Calculate cultural sensitivity score."""
        base_score = 0.8  # Start with good assumption
        
        combined_text = f"{str(action)} {str(context)}".lower()
        
        # Positive cultural indicators
        positive_indicators = [
            "respect", "honor", "collaborate", "consult", "include",
            "diverse", "multicultural", "inclusive", "partnership"
        ]
        
        # Negative cultural indicators
        negative_indicators = [
            "appropriate", "stereotype", "exotic", "primitive", "backward",
            "civilize", "modernize", "westernize", "exploit"
        ]
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in combined_text)
        negative_count = sum(1 for indicator in negative_indicators if indicator in combined_text)
        
        # Adjust score based on indicators
        base_score += positive_count * 0.1
        base_score -= negative_count * 0.3
        
        return max(0.0, min(1.0, base_score))
    
    def _calculate_overall_alignment_score(
        self,
        value_scores: Dict[ValueCategory, float],
        cultural_context: CulturalContext
    ) -> float:
        """Calculate weighted overall alignment score."""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for value_category, score in value_scores.items():
            base_weight = self.core_values[value_category]["weight"]
            cultural_weight = self._get_cultural_weight(value_category, cultural_context)
            final_weight = base_weight * cultural_weight
            
            total_weighted_score += score * final_weight
            total_weight += final_weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_alignment_recommendations(
        self,
        value_scores: Dict[ValueCategory, float],
        conflicts: List[ValueConflict],
        cultural_considerations: List[CulturalConsideration]
    ) -> List[str]:
        """Generate recommendations for improving value alignment."""
        recommendations = []
        
        # Recommendations for low-scoring values
        for value_category, score in value_scores.items():
            if score < 0.6:
                value_name = value_category.value.replace("_", " ").title()
                recommendations.append(f"Improve alignment with {value_name}")
                
                if value_category == ValueCategory.CULTURAL_RESPECT:
                    recommendations.append("Engage with cultural communities for guidance")
                elif value_category == ValueCategory.AUTONOMY:
                    recommendations.append("Ensure proper consent and choice mechanisms")
                elif value_category == ValueCategory.NON_MALEFICENCE:
                    recommendations.append("Implement additional safety measures")
        
        # Recommendations for resolving conflicts
        for conflict in conflicts:
            recommendations.append(f"Resolve conflict using {conflict.resolution_strategy}")
        
        # Cultural recommendations
        for consideration in cultural_considerations:
            recommendations.extend(consideration.recommendations)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(recommendations))
    
    def _perform_mathematical_validation(
        self,
        value_scores: Dict[ValueCategory, float],
        overall_score: float
    ) -> Dict[str, float]:
        """Perform mathematical validation of value alignment."""
        validation_metrics = {}
        
        # Calculate variance in value scores
        scores = list(value_scores.values())
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        validation_metrics["score_variance"] = variance
        validation_metrics["score_consistency"] = 1.0 - variance  # Higher consistency = lower variance
        
        # Calculate convergence metric
        if len(self.validation_history) > 0:
            recent_scores = [h["overall_score"] for h in self.validation_history[-self.stability_window:]]
            if len(recent_scores) >= 2:
                score_changes = [abs(recent_scores[i] - recent_scores[i-1]) for i in range(1, len(recent_scores))]
                avg_change = sum(score_changes) / len(score_changes)
                validation_metrics["convergence_rate"] = max(0.0, 1.0 - avg_change)
            else:
                validation_metrics["convergence_rate"] = 0.5
        else:
            validation_metrics["convergence_rate"] = 0.5
        
        # Calculate stability metric
        validation_metrics["stability_score"] = min(
            validation_metrics["score_consistency"],
            validation_metrics["convergence_rate"]
        )
        
        # Update validation history
        self.validation_history.append({
            "timestamp": time.time(),
            "overall_score": overall_score,
            "variance": variance,
            "value_scores": value_scores.copy()
        })
        
        # Keep only recent history
        if len(self.validation_history) > 100:
            self.validation_history = self.validation_history[-100:]
        
        return validation_metrics
    
    def _requires_human_oversight(
        self,
        overall_score: float,
        conflicts: List[ValueConflict],
        cultural_considerations: List[CulturalConsideration]
    ) -> bool:
        """Determine if human oversight is required."""
        # Low overall alignment score
        if overall_score < 0.6:
            return True
        
        # High-impact conflicts
        if any(conflict.conflict_score > 0.6 for conflict in conflicts):
            return True
        
        # Cultural consultation required
        if any(cc.consultation_required for cc in cultural_considerations):
            return True
        
        # Multiple moderate conflicts
        if len(conflicts) >= 3:
            return True
        
        return False
    
    def _store_alignment_assessment(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        value_scores: Dict[ValueCategory, float],
        overall_score: float,
        conflicts: List[ValueConflict]
    ) -> None:
        """Store alignment assessment for learning."""
        assessment_data = {
            "timestamp": time.time(),
            "action": action,
            "context": context,
            "value_scores": {k.value: v for k, v in value_scores.items()},
            "overall_score": overall_score,
            "conflicts": [conflict.__dict__ for conflict in conflicts]
        }
        
        # Store in memory for learning
        self.memory.store(
            f"value_alignment_{int(time.time())}",
            assessment_data,
            ttl=86400 * 30  # Keep for 30 days
        )
    
    def _update_value_weights(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Update value weights based on feedback and learning."""
        feedback = message.get("feedback", {})
        adaptation_type = message.get("adaptation_type", "gradual")
        
        # Extract feedback signals
        value_adjustments = feedback.get("value_adjustments", {})
        cultural_feedback = feedback.get("cultural_feedback", {})
        
        # Apply gradual learning
        for value_name, adjustment in value_adjustments.items():
            try:
                value_category = ValueCategory(value_name)
                if value_category in self.core_values:
                    current_weight = self.core_values[value_category]["weight"]
                    new_weight = current_weight + (adjustment * self.learning_rate)
                    self.core_values[value_category]["weight"] = max(0.1, min(1.5, new_weight))
            except ValueError:
                self.logger.warning(f"Unknown value category: {value_name}")
        
        # Update cultural weights if provided
        for cultural_context, adjustments in cultural_feedback.items():
            try:
                context_enum = CulturalContext(cultural_context)
                if context_enum not in self.cultural_weights:
                    self.cultural_weights[context_enum] = {}
                
                for value_name, adjustment in adjustments.items():
                    try:
                        value_category = ValueCategory(value_name)
                        current_weight = self.cultural_weights[context_enum].get(value_category, 1.0)
                        new_weight = current_weight + (adjustment * self.learning_rate)
                        self.cultural_weights[context_enum][value_category] = max(0.5, min(2.0, new_weight))
                    except ValueError:
                        continue
            except ValueError:
                continue
        
        self.alignment_stats["value_adaptations"] += 1
        
        return {
            "adaptation_applied": True,
            "adaptation_type": adaptation_type,
            "updated_values": len(value_adjustments),
            "updated_cultural_contexts": len(cultural_feedback)
        }
    
    def _resolve_value_conflict(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a specific value conflict."""
        conflict_data = message.get("conflict", {})
        resolution_strategy = message.get("resolution_strategy", "seek_balance")
        
        # Create conflict object from data
        try:
            conflict = ValueConflict(
                primary_value=ValueCategory(conflict_data.get("primary_value")),
                conflicting_value=ValueCategory(conflict_data.get("conflicting_value")),
                conflict_score=conflict_data.get("conflict_score", 0.0),
                context=conflict_data.get("context", ""),
                resolution_strategy=resolution_strategy,
                confidence=conflict_data.get("confidence", 0.5)
            )
        except (ValueError, KeyError) as e:
            return {"error": f"Invalid conflict data: {str(e)}"}
        
        # Apply resolution strategy
        resolution_result = self._apply_resolution_strategy(conflict, resolution_strategy)
        
        return {
            "conflict_resolved": True,
            "resolution_strategy": resolution_strategy,
            "resolution_result": resolution_result,
            "recommendations": self._get_conflict_recommendations(conflict)
        }
    
    def _apply_resolution_strategy(
        self,
        conflict: ValueConflict,
        strategy: str
    ) -> Dict[str, Any]:
        """Apply a specific conflict resolution strategy."""
        if strategy == "prioritize_safety":
            # Prioritize non-maleficence and beneficence
            safety_values = {ValueCategory.NON_MALEFICENCE, ValueCategory.BENEFICENCE}
            if conflict.primary_value in safety_values:
                return {"resolution": "prioritize_primary_value", "rationale": "Safety prioritized"}
            elif conflict.conflicting_value in safety_values:
                return {"resolution": "prioritize_conflicting_value", "rationale": "Safety prioritized"}
            else:
                return {"resolution": "seek_balance", "rationale": "No safety value involved"}
        
        elif strategy == "community_consultation":
            return {
                "resolution": "require_consultation",
                "rationale": "Community input required for cultural conflicts",
                "consultation_type": "cultural_community"
            }
        
        elif strategy == "seek_balance":
            return {
                "resolution": "balanced_approach",
                "rationale": "Attempt to satisfy both values partially",
                "balance_ratio": 0.6  # 60% weight to primary value
            }
        
        elif strategy == "context_specific":
            return {
                "resolution": "context_dependent",
                "rationale": "Resolution depends on specific context",
                "requires_case_analysis": True
            }
        
        else:
            return {"resolution": "unknown_strategy", "rationale": f"Unknown strategy: {strategy}"}
    
    def _get_conflict_recommendations(self, conflict: ValueConflict) -> List[str]:
        """Get recommendations for resolving a specific conflict."""
        recommendations = []
        
        primary_name = conflict.primary_value.value.replace("_", " ").title()
        conflicting_name = conflict.conflicting_value.value.replace("_", " ").title()
        
        recommendations.append(f"Balance {primary_name} with {conflicting_name}")
        
        if conflict.primary_value == ValueCategory.CULTURAL_RESPECT:
            recommendations.append("Engage with relevant cultural communities")
        
        if ValueCategory.NON_MALEFICENCE in {conflict.primary_value, conflict.conflicting_value}:
            recommendations.append("Prioritize harm prevention measures")
        
        recommendations.append(f"Apply {conflict.resolution_strategy} approach")
        
        return recommendations
    
    def _assess_cultural_context(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Assess cultural context for an action."""
        action = message.get("action", {})
        context = message.get("context", {})
        
        # Detect cultural context from action and context
        detected_context = self._detect_cultural_context(action, context)
        
        # Calculate cultural sensitivity
        sensitivity_score = self._calculate_cultural_sensitivity(action, context, detected_context)
        
        # Generate cultural considerations
        cultural_considerations = self._assess_cultural_considerations(
            action, context, detected_context
        )
        
        return {
            "detected_cultural_context": detected_context.value,
            "sensitivity_score": sensitivity_score,
            "cultural_considerations": [cc.__dict__ for cc in cultural_considerations],
            "recommendations": [rec for cc in cultural_considerations for rec in cc.recommendations]
        }
    
    def _detect_cultural_context(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> CulturalContext:
        """Detect cultural context from action and context."""
        combined_text = f"{str(action)} {str(context)}".lower()
        
        # Cultural context keywords
        context_keywords = {
            CulturalContext.INDIGENOUS: [
                "indigenous", "tribal", "native", "aboriginal", "traditional knowledge",
                "sacred sites", "ancestral", "first nations"
            ],
            CulturalContext.EASTERN_COLLECTIVISTIC: [
                "collective", "community harmony", "group consensus", "family honor",
                "social cohesion", "eastern culture"
            ],
            CulturalContext.WESTERN_INDIVIDUALISTIC: [
                "individual rights", "personal freedom", "autonomy", "privacy",
                "self-determination", "western culture"
            ],
            CulturalContext.AFRICAN: [
                "ubuntu", "community solidarity", "ancestral wisdom", "african culture",
                "collective responsibility"
            ],
            CulturalContext.LATIN_AMERICAN: [
                "family solidarity", "community support", "latin culture",
                "cultural traditions", "religious community"
            ]
        }
        
        # Score each context
        context_scores = {}
        for context_type, keywords in context_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                context_scores[context_type] = score
        
        # Return highest scoring context or universal if none detected
        if context_scores:
            return max(context_scores.keys(), key=lambda k: context_scores[k])
        else:
            return CulturalContext.UNIVERSAL
    
    def _validate_mathematical_convergence(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Validate mathematical convergence of value alignment."""
        if len(self.validation_history) < 2:
            return {
                "convergence_status": "insufficient_data",
                "stability_score": 0.0,
                "requires_more_data": True
            }
        
        # Calculate convergence metrics
        recent_scores = [h["overall_score"] for h in self.validation_history[-self.stability_window:]]
        
        # Check for convergence
        if len(recent_scores) >= 3:
            score_changes = [abs(recent_scores[i] - recent_scores[i-1]) for i in range(1, len(recent_scores))]
            avg_change = sum(score_changes) / len(score_changes)
            
            converged = avg_change < self.convergence_threshold
            stability_score = max(0.0, 1.0 - (avg_change * 10))  # Scale for readability
            
            return {
                "convergence_status": "converged" if converged else "not_converged",
                "stability_score": stability_score,
                "average_change": avg_change,
                "convergence_threshold": self.convergence_threshold,
                "mathematical_guarantee": converged and stability_score > 0.9
            }
        
        return {
            "convergence_status": "insufficient_data",
            "stability_score": 0.0,
            "requires_more_data": True
        }
    
    def _get_alignment_statistics(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Get value alignment statistics."""
        return {
            "alignment_statistics": self.alignment_stats,
            "total_values": len(self.core_values),
            "cultural_contexts": len(self.cultural_weights),
            "validation_history_length": len(self.validation_history),
            "convergence_validation": self._validate_mathematical_convergence({})
        }

# Maintain backward compatibility
ValueAlignment = ValueAlignmentAgent 
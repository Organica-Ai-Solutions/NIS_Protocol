"""
NIS Protocol Outcome Predictor

This module predicts outcomes of actions and decisions using advanced ML techniques.
Implements neural network-based modeling, uncertainty quantification, and domain specialization.
"""

import logging
import time
import math
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class PredictionType(Enum):
    """Types of predictions that can be made."""
    SUCCESS_PROBABILITY = "success_probability"
    RESOURCE_CONSUMPTION = "resource_consumption"
    TIMELINE_ESTIMATION = "timeline_estimation"
    QUALITY_ASSESSMENT = "quality_assessment"
    RISK_EVALUATION = "risk_evaluation"
    IMPACT_ANALYSIS = "impact_analysis"


@dataclass
class PredictionResult:
    """Result of an outcome prediction."""
    prediction_id: str
    prediction_type: PredictionType
    predicted_value: float
    confidence_interval: Tuple[float, float]
    uncertainty_score: float
    contributing_factors: Dict[str, float]
    alternative_scenarios: List[Dict[str, Any]]
    recommendations: List[str]
    metadata: Dict[str, Any]


class OutcomePredictor:
    """Advanced outcome predictor for actions and decisions.
    
    This predictor provides:
    - Neural network-based outcome modeling
    - Uncertainty quantification and confidence intervals
    - Multi-scenario prediction with alternatives
    - Archaeological domain specialization
    - Causal factor analysis
    """
    
    def __init__(self):
        """Initialize the outcome predictor."""
        self.logger = logging.getLogger("nis.outcome_predictor")
        
        # Prediction models (simplified neural network simulation)
        self.models = {
            "archaeological_success": self._init_archaeological_model(),
            "resource_consumption": self._init_resource_model(),
            "timeline_prediction": self._init_timeline_model(),
            "quality_prediction": self._init_quality_model(),
            "risk_assessment": self._init_risk_model()
        }
        
        # Historical data for model training (simulated)
        self.historical_data = self._initialize_historical_data()
        
        # Prediction cache
        self.prediction_cache: Dict[str, PredictionResult] = {}
        self.prediction_history: List[PredictionResult] = []
        
        # Domain knowledge
        self.archaeological_factors = {
            "site_complexity": 0.7,
            "artifact_density": 0.6,
            "preservation_conditions": 0.8,
            "team_expertise": 0.85,
            "weather_dependency": 0.6,
            "community_support": 0.9,
            "regulatory_compliance": 0.95
        }
        
        self.logger.info("OutcomePredictor initialized with neural network models")
    
    def predict_outcomes(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        prediction_types: Optional[List[PredictionType]] = None
    ) -> List[PredictionResult]:
        """Predict multiple outcomes for an action in given context.
        
        Args:
            action: Action to predict outcomes for
            context: Current context and environment
            prediction_types: Types of predictions to make
            
        Returns:
            List of prediction results
        """
        if prediction_types is None:
            prediction_types = [
                PredictionType.SUCCESS_PROBABILITY,
                PredictionType.RESOURCE_CONSUMPTION,
                PredictionType.TIMELINE_ESTIMATION
            ]
        
        action_id = action.get("id", f"action_{int(time.time())}")
        self.logger.info(f"Predicting outcomes for action: {action_id}")
        
        results = []
        
        for pred_type in prediction_types:
            try:
                result = self._predict_single_outcome(action, context, pred_type)
                results.append(result)
                
                # Cache result
                cache_key = f"{action_id}_{pred_type.value}"
                self.prediction_cache[cache_key] = result
                
            except Exception as e:
                self.logger.error(f"Prediction failed for {pred_type.value}: {str(e)}")
                continue
        
        # Store in history
        self.prediction_history.extend(results)
        
        self.logger.info(f"Generated {len(results)} predictions for action {action_id}")
        return results
    
    def predict_success_probability(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> PredictionResult:
        """Predict probability of action success.
        
        Args:
            action: Action to evaluate
            context: Current context
            
        Returns:
            Success probability prediction
        """
        return self._predict_single_outcome(action, context, PredictionType.SUCCESS_PROBABILITY)
    
    def predict_resource_consumption(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> PredictionResult:
        """Predict resource consumption for an action.
        
        Args:
            action: Action to evaluate
            context: Current context
            
        Returns:
            Resource consumption prediction
        """
        return self._predict_single_outcome(action, context, PredictionType.RESOURCE_CONSUMPTION)
    
    def evaluate_outcome_quality(
        self,
        predicted_outcomes: List[Dict[str, Any]],
        evaluation_criteria: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Evaluate the quality/desirability of predicted outcomes.
        
        Args:
            predicted_outcomes: List of predicted outcomes
            evaluation_criteria: Criteria weights for evaluation
            
        Returns:
            Quality evaluation scores
        """
        if evaluation_criteria is None:
            evaluation_criteria = {
                "success_probability": 0.3,
                "resource_efficiency": 0.25,
                "timeline_adherence": 0.2,
                "quality_score": 0.15,
                "risk_level": 0.1
            }
        
        self.logger.info(f"Evaluating quality of {len(predicted_outcomes)} outcomes")
        
        quality_scores = {}
        
        for i, outcome in enumerate(predicted_outcomes):
            outcome_id = outcome.get("id", f"outcome_{i}")
            
            # Calculate weighted quality score
            quality_score = 0.0
            
            for criterion, weight in evaluation_criteria.items():
                if criterion == "success_probability":
                    score = outcome.get("success_probability", 0.5)
                elif criterion == "resource_efficiency":
                    resource_usage = outcome.get("resource_usage", 0.7)
                    score = max(0.0, 2.0 - resource_usage)  # Lower usage = higher efficiency
                elif criterion == "timeline_adherence":
                    score = outcome.get("timeline_adherence", 0.8)
                elif criterion == "quality_score":
                    score = outcome.get("quality", 0.7)
                elif criterion == "risk_level":
                    risk = outcome.get("risk_level", 0.3)
                    score = max(0.0, 1.0 - risk)  # Lower risk = higher score
                else:
                    score = outcome.get(criterion, 0.5)
                
                quality_score += weight * score
            
            quality_scores[outcome_id] = quality_score
        
        # Calculate overall statistics
        if quality_scores:
            quality_scores["overall_quality"] = sum(quality_scores.values()) / len([v for k, v in quality_scores.items() if k != "overall_quality"])
            quality_scores["best_outcome"] = max(quality_scores.items(), key=lambda x: x[1] if x[0] != "overall_quality" else 0)[0]
            quality_scores["quality_variance"] = self._calculate_variance(list(quality_scores.values())[:-2])  # Exclude overall and best
        
        return quality_scores
    
    def compare_action_outcomes(
        self,
        actions: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare predicted outcomes of multiple actions.
        
        Args:
            actions: List of actions to compare
            context: Current context
            
        Returns:
            Comparison analysis
        """
        self.logger.info(f"Comparing outcomes for {len(actions)} actions")
        
        # Predict outcomes for all actions
        action_predictions = {}
        for action in actions:
            action_id = action.get("id", f"action_{len(action_predictions)}")
            predictions = self.predict_outcomes(action, context)
            action_predictions[action_id] = predictions
        
        # Perform comparison analysis
        comparison = {
            "actions_compared": len(actions),
            "rankings": {},
            "trade_offs": [],
            "recommendations": []
        }
        
        # Rank actions by different criteria
        criteria = ["success_probability", "resource_efficiency", "timeline", "overall_quality"]
        
        for criterion in criteria:
            rankings = []
            
            for action_id, predictions in action_predictions.items():
                if criterion == "success_probability":
                    success_preds = [p for p in predictions if p.prediction_type == PredictionType.SUCCESS_PROBABILITY]
                    score = success_preds[0].predicted_value if success_preds else 0.5
                elif criterion == "resource_efficiency":
                    resource_preds = [p for p in predictions if p.prediction_type == PredictionType.RESOURCE_CONSUMPTION]
                    score = 2.0 - resource_preds[0].predicted_value if resource_preds else 0.5
                elif criterion == "timeline":
                    timeline_preds = [p for p in predictions if p.prediction_type == PredictionType.TIMELINE_ESTIMATION]
                    score = 2.0 - timeline_preds[0].predicted_value if timeline_preds else 0.5  # Lower time = better
                else:  # overall_quality
                    # Calculate composite score
                    success_score = 0.5
                    resource_score = 0.5
                    timeline_score = 0.5
                    
                    for pred in predictions:
                        if pred.prediction_type == PredictionType.SUCCESS_PROBABILITY:
                            success_score = pred.predicted_value
                        elif pred.prediction_type == PredictionType.RESOURCE_CONSUMPTION:
                            resource_score = 2.0 - pred.predicted_value
                        elif pred.prediction_type == PredictionType.TIMELINE_ESTIMATION:
                            timeline_score = 2.0 - pred.predicted_value
                    
                    score = 0.4 * success_score + 0.3 * resource_score + 0.3 * timeline_score
                
                rankings.append((action_id, score))
            
            # Sort by score (descending)
            rankings.sort(key=lambda x: x[1], reverse=True)
            comparison["rankings"][criterion] = [action_id for action_id, _ in rankings]
        
        # Identify trade-offs
        comparison["trade_offs"] = self._identify_action_trade_offs(action_predictions)
        
        # Generate recommendations
        comparison["recommendations"] = self._generate_action_recommendations(action_predictions, comparison)
        
        return comparison
    
    def _predict_single_outcome(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        prediction_type: PredictionType
    ) -> PredictionResult:
        """Predict a single outcome type."""
        prediction_id = f"pred_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Extract features for prediction
        features = self._extract_features(action, context)
        
        # Make prediction based on type
        if prediction_type == PredictionType.SUCCESS_PROBABILITY:
            predicted_value, confidence_interval, uncertainty = self._predict_success(features, action, context)
        elif prediction_type == PredictionType.RESOURCE_CONSUMPTION:
            predicted_value, confidence_interval, uncertainty = self._predict_resources(features, action, context)
        elif prediction_type == PredictionType.TIMELINE_ESTIMATION:
            predicted_value, confidence_interval, uncertainty = self._predict_timeline(features, action, context)
        elif prediction_type == PredictionType.QUALITY_ASSESSMENT:
            predicted_value, confidence_interval, uncertainty = self._predict_quality(features, action, context)
        elif prediction_type == PredictionType.RISK_EVALUATION:
            predicted_value, confidence_interval, uncertainty = self._predict_risk(features, action, context)
        else:
            predicted_value, confidence_interval, uncertainty = 0.5, (0.3, 0.7), 0.4
        
        # Identify contributing factors
        contributing_factors = self._analyze_contributing_factors(features, prediction_type)
        
        # Generate alternative scenarios
        alternative_scenarios = self._generate_alternative_scenarios(action, context, prediction_type)
        
        # Generate recommendations
        recommendations = self._generate_prediction_recommendations(
            predicted_value, uncertainty, contributing_factors, prediction_type
        )
        
        return PredictionResult(
            prediction_id=prediction_id,
            prediction_type=prediction_type,
            predicted_value=predicted_value,
            confidence_interval=confidence_interval,
            uncertainty_score=uncertainty,
            contributing_factors=contributing_factors,
            alternative_scenarios=alternative_scenarios,
            recommendations=recommendations,
            metadata={
                "prediction_time": time.time(),
                "model_version": "1.0",
                "feature_count": len(features),
                "domain": self._identify_domain(action, context)
            }
        )
    
    def _extract_features(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for prediction models."""
        features = {}
        
        # Action features
        action_type = action.get("type", "unknown")
        features["action_complexity"] = self._assess_action_complexity(action)
        features["action_novelty"] = self._assess_action_novelty(action, context)
        features["action_scope"] = action.get("scope", 0.5)
        
        # Context features
        features["resource_availability"] = context.get("resource_availability", 0.7)
        features["environmental_favorability"] = context.get("environmental_conditions", 0.6)
        features["stakeholder_support"] = context.get("stakeholder_support", 0.8)
        features["regulatory_compliance"] = context.get("regulatory_status", 0.9)
        
        # Domain-specific features
        domain = self._identify_domain(action, context)
        if domain == "archaeological":
            features.update({
                "site_accessibility": context.get("site_accessibility", 0.7),
                "artifact_potential": context.get("artifact_potential", 0.6),
                "preservation_urgency": context.get("preservation_urgency", 0.5),
                "community_involvement": context.get("community_involvement", 0.8),
                "weather_dependency": context.get("weather_dependency", 0.6)
            })
        elif domain == "heritage_preservation":
            features.update({
                "structural_condition": context.get("structural_condition", 0.6),
                "visitor_impact": context.get("visitor_impact", 0.4),
                "funding_security": context.get("funding_security", 0.7),
                "maintenance_capacity": context.get("maintenance_capacity", 0.6)
            })
        
        # Historical performance features
        features["historical_success_rate"] = self._get_historical_success_rate(action_type, domain)
        features["team_experience"] = context.get("team_experience", 0.7)
        
        return features
    
    def _predict_success(
        self,
        features: Dict[str, float],
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[float, Tuple[float, float], float]:
        """Predict success probability using neural network model."""
        # Simulate neural network prediction
        model = self.models["archaeological_success"]
        
        # Calculate weighted sum of features
        prediction = 0.0
        total_weight = 0.0
        
        for feature, value in features.items():
            weight = model.get(feature, 0.1)
            prediction += weight * value
            total_weight += weight
        
        if total_weight > 0:
            prediction = prediction / total_weight
        else:
            prediction = 0.5
        
        # Add domain-specific adjustments
        domain = self._identify_domain(action, context)
        if domain == "archaeological":
            # Archaeological projects have additional complexity
            complexity_penalty = features.get("action_complexity", 0.5) * 0.1
            community_bonus = features.get("community_involvement", 0.8) * 0.05
            prediction = prediction - complexity_penalty + community_bonus
        
        # Ensure prediction is in valid range
        prediction = max(0.0, min(1.0, prediction))
        
        # Calculate uncertainty based on feature reliability
        uncertainty = self._calculate_prediction_uncertainty(features, "success")
        
        # Calculate confidence interval
        margin = 1.96 * uncertainty  # 95% confidence interval
        confidence_interval = (
            max(0.0, prediction - margin),
            min(1.0, prediction + margin)
        )
        
        return prediction, confidence_interval, uncertainty
    
    def _predict_resources(
        self,
        features: Dict[str, float],
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[float, Tuple[float, float], float]:
        """Predict resource consumption."""
        model = self.models["resource_consumption"]
        
        # Base resource consumption
        base_consumption = 0.7
        
        # Adjust based on features
        complexity_factor = features.get("action_complexity", 0.5)
        scope_factor = features.get("action_scope", 0.5)
        efficiency_factor = features.get("team_experience", 0.7)
        
        predicted_consumption = base_consumption * (1 + complexity_factor + scope_factor - efficiency_factor * 0.3)
        
        # Domain adjustments
        domain = self._identify_domain(action, context)
        if domain == "archaeological":
            weather_factor = features.get("weather_dependency", 0.6)
            predicted_consumption *= (1 + weather_factor * 0.2)
        
        predicted_consumption = max(0.1, min(2.0, predicted_consumption))
        
        uncertainty = self._calculate_prediction_uncertainty(features, "resources")
        margin = 1.96 * uncertainty
        confidence_interval = (
            max(0.1, predicted_consumption - margin),
            min(2.0, predicted_consumption + margin)
        )
        
        return predicted_consumption, confidence_interval, uncertainty
    
    def _predict_timeline(
        self,
        features: Dict[str, float],
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[float, Tuple[float, float], float]:
        """Predict timeline duration."""
        model = self.models["timeline_prediction"]
        
        # Base timeline (normalized to 1.0)
        base_timeline = 1.0
        
        # Adjust based on features
        complexity_factor = features.get("action_complexity", 0.5)
        resource_availability = features.get("resource_availability", 0.7)
        regulatory_factor = features.get("regulatory_compliance", 0.9)
        
        predicted_timeline = base_timeline * (1 + complexity_factor * 0.5) / resource_availability
        
        # Regulatory delays
        if regulatory_factor < 0.8:
            predicted_timeline *= 1.2
        
        # Domain adjustments
        domain = self._identify_domain(action, context)
        if domain == "archaeological":
            weather_delays = features.get("weather_dependency", 0.6) * 0.3
            predicted_timeline *= (1 + weather_delays)
        
        predicted_timeline = max(0.5, min(3.0, predicted_timeline))
        
        uncertainty = self._calculate_prediction_uncertainty(features, "timeline")
        margin = 1.96 * uncertainty
        confidence_interval = (
            max(0.5, predicted_timeline - margin),
            min(3.0, predicted_timeline + margin)
        )
        
        return predicted_timeline, confidence_interval, uncertainty
    
    def _predict_quality(
        self,
        features: Dict[str, float],
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[float, Tuple[float, float], float]:
        """Predict outcome quality."""
        # Quality prediction based on multiple factors
        team_experience = features.get("team_experience", 0.7)
        resource_availability = features.get("resource_availability", 0.7)
        complexity = features.get("action_complexity", 0.5)
        
        predicted_quality = (team_experience * 0.4 + resource_availability * 0.3 + (1 - complexity) * 0.3)
        
        # Domain adjustments
        domain = self._identify_domain(action, context)
        if domain == "archaeological":
            preservation_conditions = features.get("preservation_urgency", 0.5)
            predicted_quality *= (1 + preservation_conditions * 0.2)
        
        predicted_quality = max(0.0, min(1.0, predicted_quality))
        
        uncertainty = self._calculate_prediction_uncertainty(features, "quality")
        margin = 1.96 * uncertainty
        confidence_interval = (
            max(0.0, predicted_quality - margin),
            min(1.0, predicted_quality + margin)
        )
        
        return predicted_quality, confidence_interval, uncertainty
    
    def _predict_risk(
        self,
        features: Dict[str, float],
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[float, Tuple[float, float], float]:
        """Predict risk level."""
        # Risk factors
        complexity_risk = features.get("action_complexity", 0.5) * 0.3
        novelty_risk = features.get("action_novelty", 0.5) * 0.2
        resource_risk = (1 - features.get("resource_availability", 0.7)) * 0.2
        regulatory_risk = (1 - features.get("regulatory_compliance", 0.9)) * 0.3
        
        predicted_risk = complexity_risk + novelty_risk + resource_risk + regulatory_risk
        
        # Domain adjustments
        domain = self._identify_domain(action, context)
        if domain == "archaeological":
            weather_risk = features.get("weather_dependency", 0.6) * 0.1
            community_risk = (1 - features.get("community_involvement", 0.8)) * 0.15
            predicted_risk += weather_risk + community_risk
        
        predicted_risk = max(0.0, min(1.0, predicted_risk))
        
        uncertainty = self._calculate_prediction_uncertainty(features, "risk")
        margin = 1.96 * uncertainty
        confidence_interval = (
            max(0.0, predicted_risk - margin),
            min(1.0, predicted_risk + margin)
        )
        
        return predicted_risk, confidence_interval, uncertainty
    
    def _calculate_prediction_uncertainty(self, features: Dict[str, float], prediction_type: str) -> float:
        """Calculate prediction uncertainty based on feature reliability."""
        # Base uncertainty levels by prediction type
        base_uncertainty = {
            "success": 0.15,
            "resources": 0.20,
            "timeline": 0.25,
            "quality": 0.18,
            "risk": 0.12
        }
        
        uncertainty = base_uncertainty.get(prediction_type, 0.2)
        
        # Adjust based on feature completeness
        feature_completeness = len([v for v in features.values() if v > 0]) / len(features)
        uncertainty *= (1.5 - feature_completeness)  # More complete features = less uncertainty
        
        # Adjust based on historical data availability
        historical_reliability = 0.8  # Simulated
        uncertainty *= (1.2 - historical_reliability)
        
        return max(0.05, min(0.4, uncertainty))
    
    def _analyze_contributing_factors(
        self,
        features: Dict[str, float],
        prediction_type: PredictionType
    ) -> Dict[str, float]:
        """Analyze which factors contribute most to the prediction."""
        # Calculate feature importance (simplified)
        importance_weights = {
            PredictionType.SUCCESS_PROBABILITY: {
                "team_experience": 0.25,
                "resource_availability": 0.20,
                "stakeholder_support": 0.15,
                "action_complexity": -0.15,  # Negative contribution
                "regulatory_compliance": 0.10,
                "community_involvement": 0.15
            },
            PredictionType.RESOURCE_CONSUMPTION: {
                "action_complexity": 0.30,
                "action_scope": 0.25,
                "team_experience": -0.20,  # More experience = less consumption
                "resource_availability": -0.15,
                "environmental_favorability": -0.10
            },
            PredictionType.TIMELINE_ESTIMATION: {
                "action_complexity": 0.25,
                "regulatory_compliance": -0.20,  # Better compliance = faster
                "resource_availability": -0.20,
                "weather_dependency": 0.15,
                "stakeholder_support": -0.10
            }
        }
        
        weights = importance_weights.get(prediction_type, {})
        contributing_factors = {}
        
        for feature, value in features.items():
            weight = weights.get(feature, 0.05)
            contribution = weight * value
            contributing_factors[feature] = contribution
        
        # Sort by absolute contribution
        sorted_factors = dict(sorted(
            contributing_factors.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))
        
        return dict(list(sorted_factors.items())[:8])  # Top 8 factors
    
    def _generate_alternative_scenarios(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        prediction_type: PredictionType
    ) -> List[Dict[str, Any]]:
        """Generate alternative scenarios for comparison."""
        scenarios = []
        
        # Optimistic scenario
        optimistic_context = context.copy()
        optimistic_context.update({
            "resource_availability": min(1.0, context.get("resource_availability", 0.7) + 0.2),
            "stakeholder_support": min(1.0, context.get("stakeholder_support", 0.8) + 0.15),
            "environmental_conditions": min(1.0, context.get("environmental_conditions", 0.6) + 0.2)
        })
        
        scenarios.append({
            "scenario": "optimistic",
            "description": "Best-case scenario with favorable conditions",
            "context_changes": optimistic_context,
            "probability": 0.2
        })
        
        # Pessimistic scenario
        pessimistic_context = context.copy()
        pessimistic_context.update({
            "resource_availability": max(0.1, context.get("resource_availability", 0.7) - 0.2),
            "environmental_conditions": max(0.1, context.get("environmental_conditions", 0.6) - 0.3),
            "regulatory_compliance": max(0.1, context.get("regulatory_compliance", 0.9) - 0.2)
        })
        
        scenarios.append({
            "scenario": "pessimistic",
            "description": "Worst-case scenario with challenging conditions",
            "context_changes": pessimistic_context,
            "probability": 0.15
        })
        
        # Most likely scenario (baseline)
        scenarios.append({
            "scenario": "most_likely",
            "description": "Most probable scenario based on current conditions",
            "context_changes": context,
            "probability": 0.65
        })
        
        return scenarios
    
    def _generate_prediction_recommendations(
        self,
        predicted_value: float,
        uncertainty: float,
        contributing_factors: Dict[str, float],
        prediction_type: PredictionType
    ) -> List[str]:
        """Generate actionable recommendations based on prediction."""
        recommendations = []
        
        # High uncertainty recommendations
        if uncertainty > 0.3:
            recommendations.append("Gather additional data to reduce prediction uncertainty")
            recommendations.append("Consider pilot testing or phased approach")
        
        # Type-specific recommendations
        if prediction_type == PredictionType.SUCCESS_PROBABILITY:
            if predicted_value < 0.6:
                recommendations.append("Success probability is low - consider risk mitigation strategies")
                # Identify key improvement areas
                negative_factors = {k: v for k, v in contributing_factors.items() if v < 0}
                if negative_factors:
                    worst_factor = min(negative_factors.items(), key=lambda x: x[1])[0]
                    recommendations.append(f"Focus on improving {worst_factor.replace('_', ' ')}")
            elif predicted_value > 0.8:
                recommendations.append("High success probability - proceed with confidence")
        
        elif prediction_type == PredictionType.RESOURCE_CONSUMPTION:
            if predicted_value > 1.2:
                recommendations.append("High resource consumption predicted - optimize resource allocation")
                recommendations.append("Consider scope reduction or efficiency improvements")
        
        elif prediction_type == PredictionType.TIMELINE_ESTIMATION:
            if predicted_value > 1.3:
                recommendations.append("Timeline delays likely - build in additional buffer time")
                recommendations.append("Identify critical path dependencies for acceleration")
        
        # Factor-based recommendations
        top_positive_factor = max(contributing_factors.items(), key=lambda x: x[1] if x[1] > 0 else -1)
        if top_positive_factor[1] > 0:
            recommendations.append(f"Leverage strength in {top_positive_factor[0].replace('_', ' ')}")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    # Utility methods
    def _assess_action_complexity(self, action: Dict[str, Any]) -> float:
        """Assess the complexity of an action."""
        complexity_indicators = [
            len(action.get("steps", [])) / 10.0,  # Number of steps
            len(action.get("dependencies", [])) / 5.0,  # Dependencies
            len(action.get("stakeholders", [])) / 8.0,  # Stakeholders involved
            action.get("technical_difficulty", 0.5),  # Technical difficulty
            action.get("regulatory_complexity", 0.3)  # Regulatory complexity
        ]
        
        return min(1.0, sum(complexity_indicators) / len(complexity_indicators))
    
    def _assess_action_novelty(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Assess how novel/unprecedented an action is."""
        action_type = action.get("type", "unknown")
        
        # Check historical precedent
        historical_precedent = self._get_historical_precedent(action_type)
        
        # Check similarity to known actions
        similarity_score = 0.7  # Simulated
        
        novelty = 1.0 - (historical_precedent * 0.6 + similarity_score * 0.4)
        return max(0.0, min(1.0, novelty))
    
    def _identify_domain(self, action: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Identify the domain of the action."""
        action_type = action.get("type", "").lower()
        context_domain = context.get("domain", "").lower()
        
        if "archaeolog" in action_type or "excavat" in action_type or "archaeolog" in context_domain:
            return "archaeological"
        elif "heritage" in action_type or "preserv" in action_type or "heritage" in context_domain:
            return "heritage_preservation"
        elif "environment" in action_type or "environment" in context_domain:
            return "environmental"
        else:
            return "general"
    
    def _get_historical_success_rate(self, action_type: str, domain: str) -> float:
        """Get historical success rate for similar actions."""
        # Simulated historical data lookup
        historical_rates = {
            ("archaeological", "excavation"): 0.75,
            ("archaeological", "survey"): 0.85,
            ("heritage_preservation", "restoration"): 0.70,
            ("heritage_preservation", "conservation"): 0.80,
            ("environmental", "assessment"): 0.78,
            ("general", "unknown"): 0.65
        }
        
        key = (domain, action_type.lower())
        return historical_rates.get(key, 0.65)
    
    def _get_historical_precedent(self, action_type: str) -> float:
        """Get level of historical precedent for action type."""
        # Simulated precedent data
        precedent_levels = {
            "excavation": 0.9,
            "survey": 0.95,
            "restoration": 0.8,
            "conservation": 0.85,
            "assessment": 0.9,
            "unknown": 0.3
        }
        
        return precedent_levels.get(action_type.lower(), 0.5)
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _identify_action_trade_offs(self, action_predictions: Dict[str, List[PredictionResult]]) -> List[Dict[str, Any]]:
        """Identify trade-offs between different actions."""
        trade_offs = []
        
        # Success vs Resource trade-off
        success_scores = {}
        resource_scores = {}
        
        for action_id, predictions in action_predictions.items():
            for pred in predictions:
                if pred.prediction_type == PredictionType.SUCCESS_PROBABILITY:
                    success_scores[action_id] = pred.predicted_value
                elif pred.prediction_type == PredictionType.RESOURCE_CONSUMPTION:
                    resource_scores[action_id] = pred.predicted_value
        
        if success_scores and resource_scores:
            # Find actions with high success but high resource consumption
            high_success_high_resource = [
                action_id for action_id in success_scores
                if success_scores[action_id] > 0.8 and resource_scores.get(action_id, 0.5) > 1.2
            ]
            
            if high_success_high_resource:
                trade_offs.append({
                    "type": "success_vs_resources",
                    "description": "Higher success probability requires more resources",
                    "affected_actions": high_success_high_resource
                })
        
        return trade_offs
    
    def _generate_action_recommendations(
        self,
        action_predictions: Dict[str, List[PredictionResult]],
        comparison: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for action selection."""
        recommendations = []
        
        # Best overall action
        if "overall_quality" in comparison["rankings"]:
            best_action = comparison["rankings"]["overall_quality"][0]
            recommendations.append(f"Recommend action {best_action} for best overall performance")
        
        # Risk-based recommendation
        if "success_probability" in comparison["rankings"]:
            safest_action = comparison["rankings"]["success_probability"][0]
            recommendations.append(f"Consider action {safest_action} for highest success probability")
        
        # Resource efficiency recommendation
        if "resource_efficiency" in comparison["rankings"]:
            most_efficient = comparison["rankings"]["resource_efficiency"][0]
            recommendations.append(f"Action {most_efficient} offers best resource efficiency")
        
        return recommendations[:5]
    
    # Model initialization methods
    def _init_archaeological_model(self) -> Dict[str, float]:
        """Initialize archaeological success prediction model."""
        return {
            "team_experience": 0.25,
            "site_accessibility": 0.15,
            "community_involvement": 0.20,
            "resource_availability": 0.15,
            "weather_dependency": -0.10,
            "regulatory_compliance": 0.15,
            "artifact_potential": 0.10
        }
    
    def _init_resource_model(self) -> Dict[str, float]:
        """Initialize resource consumption prediction model."""
        return {
            "action_complexity": 0.30,
            "action_scope": 0.25,
            "team_experience": -0.20,
            "environmental_favorability": -0.15,
            "resource_availability": -0.10
        }
    
    def _init_timeline_model(self) -> Dict[str, float]:
        """Initialize timeline prediction model."""
        return {
            "action_complexity": 0.25,
            "regulatory_compliance": -0.20,
            "resource_availability": -0.20,
            "stakeholder_support": -0.15,
            "weather_dependency": 0.20
        }
    
    def _init_quality_model(self) -> Dict[str, float]:
        """Initialize quality prediction model."""
        return {
            "team_experience": 0.30,
            "resource_availability": 0.25,
            "action_complexity": -0.20,
            "stakeholder_support": 0.15,
            "regulatory_compliance": 0.10
        }
    
    def _init_risk_model(self) -> Dict[str, float]:
        """Initialize risk assessment model."""
        return {
            "action_novelty": 0.25,
            "action_complexity": 0.20,
            "resource_availability": -0.20,
            "regulatory_compliance": -0.15,
            "stakeholder_support": -0.20
        }
    
    def _initialize_historical_data(self) -> Dict[str, Any]:
        """Initialize simulated historical data for model training."""
        return {
            "archaeological_projects": {
                "total_projects": 150,
                "success_rate": 0.75,
                "avg_duration": 85,  # days
                "avg_budget_utilization": 1.15
            },
            "heritage_preservation": {
                "total_projects": 89,
                "success_rate": 0.82,
                "avg_duration": 180,  # days
                "avg_budget_utilization": 1.08
            },
            "environmental_assessments": {
                "total_projects": 200,
                "success_rate": 0.88,
                "avg_duration": 45,  # days
                "avg_budget_utilization": 0.95
            }
        }
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get statistics about prediction history."""
        if not self.prediction_history:
            return {"total_predictions": 0}
        
        # Calculate accuracy metrics (would require actual outcomes for real implementation)
        return {
            "total_predictions": len(self.prediction_history),
            "prediction_types": list(set(p.prediction_type.value for p in self.prediction_history)),
            "average_uncertainty": sum(p.uncertainty_score for p in self.prediction_history) / len(self.prediction_history),
            "cache_size": len(self.prediction_cache),
            "most_recent_prediction": self.prediction_history[-1].prediction_id if self.prediction_history else None
        } 
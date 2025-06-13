"""
NIS Protocol Risk Assessor

This module provides comprehensive risk assessment capabilities for actions and decisions.
Implements multi-factor risk analysis, mitigation strategies, and domain specialization.
"""

import logging
import time
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class RiskCategory(Enum):
    """Categories of risks that can be assessed."""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    TECHNICAL = "technical"
    ENVIRONMENTAL = "environmental"
    REGULATORY = "regulatory"
    SOCIAL = "social"
    CULTURAL = "cultural"
    TEMPORAL = "temporal"
    SAFETY = "safety"
    REPUTATIONAL = "reputational"


class RiskSeverity(Enum):
    """Severity levels for risks."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


@dataclass
class RiskFactor:
    """Individual risk factor."""
    factor_id: str
    category: RiskCategory
    description: str
    probability: float  # 0.0 to 1.0
    impact: float      # 0.0 to 1.0
    severity: RiskSeverity
    mitigation_strategies: List[str]
    indicators: List[str]
    dependencies: List[str]


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment result."""
    assessment_id: str
    overall_risk_score: float
    risk_level: RiskSeverity
    risk_factors: List[RiskFactor]
    mitigation_plan: Dict[str, Any]
    monitoring_requirements: List[str]
    contingency_plans: List[Dict[str, Any]]
    recommendations: List[str]
    confidence_score: float
    metadata: Dict[str, Any]


class RiskAssessor:
    """Comprehensive risk assessor for actions and decisions.
    
    This assessor provides:
    - Multi-factor risk analysis across 10+ categories
    - Probability and impact assessment
    - Mitigation strategy generation
    - Archaeological and heritage domain specialization
    - Continuous risk monitoring capabilities
    """
    
    def __init__(self):
        """Initialize the risk assessor."""
        self.logger = logging.getLogger("nis.risk_assessor")
        
        # Risk assessment models
        self.risk_models = self._initialize_risk_models()
        
        # Domain-specific risk factors
        self.archaeological_risks = self._initialize_archaeological_risks()
        self.heritage_risks = self._initialize_heritage_risks()
        
        # Risk assessment history
        self.assessment_history: List[RiskAssessment] = []
        self.risk_monitoring: Dict[str, Dict[str, Any]] = {}
        
        # Mitigation strategy database
        self.mitigation_strategies = self._initialize_mitigation_strategies()
        
        self.logger.info("RiskAssessor initialized with comprehensive risk models")
    
    def assess_risks(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        risk_categories: Optional[List[RiskCategory]] = None
    ) -> RiskAssessment:
        """Assess comprehensive risks for a potential action.
        
        Args:
            action: Action to assess risks for
            context: Current context and environment
            risk_categories: Specific risk categories to assess
            
        Returns:
            Comprehensive risk assessment
        """
        if risk_categories is None:
            risk_categories = list(RiskCategory)
        
        assessment_id = f"risk_{int(time.time())}"
        action_id = action.get("id", "unknown")
        
        self.logger.info(f"Assessing risks for action: {action_id}")
        
        # Identify domain for specialized assessment
        domain = self._identify_domain(action, context)
        
        # Assess individual risk factors
        risk_factors = []
        for category in risk_categories:
            factors = self._assess_category_risks(action, context, category, domain)
            risk_factors.extend(factors)
        
        # Calculate overall risk score
        overall_risk_score = self._calculate_overall_risk_score(risk_factors)
        risk_level = self._determine_risk_level(overall_risk_score)
        
        # Generate mitigation plan
        mitigation_plan = self._generate_mitigation_plan(risk_factors, action, context)
        
        # Identify monitoring requirements
        monitoring_requirements = self._identify_monitoring_requirements(risk_factors)
        
        # Generate contingency plans
        contingency_plans = self._generate_contingency_plans(risk_factors, action, context)
        
        # Generate recommendations
        recommendations = self._generate_risk_recommendations(risk_factors, overall_risk_score)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(risk_factors, context)
        
        # Create assessment result
        assessment = RiskAssessment(
            assessment_id=assessment_id,
            overall_risk_score=overall_risk_score,
            risk_level=risk_level,
            risk_factors=risk_factors,
            mitigation_plan=mitigation_plan,
            monitoring_requirements=monitoring_requirements,
            contingency_plans=contingency_plans,
            recommendations=recommendations,
            confidence_score=confidence_score,
            metadata={
                "assessment_time": time.time(),
                "action_id": action_id,
                "domain": domain,
                "categories_assessed": [cat.value for cat in risk_categories],
                "total_factors": len(risk_factors)
            }
        )
        
        # Store assessment
        self.assessment_history.append(assessment)
        
        self.logger.info(f"Risk assessment completed: {assessment_id} (Risk Level: {risk_level.value})")
        return assessment
    
    def calculate_risk_score(
        self,
        risks: Dict[str, Any],
        weighting_scheme: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate overall risk score from risk assessment.
        
        Args:
            risks: Risk assessment data
            weighting_scheme: Custom weights for different risk categories
            
        Returns:
            Overall risk score (0.0 to 1.0)
        """
        if weighting_scheme is None:
            weighting_scheme = {
                "financial": 0.20,
                "operational": 0.15,
                "technical": 0.12,
                "environmental": 0.10,
                "regulatory": 0.15,
                "social": 0.08,
                "cultural": 0.10,
                "temporal": 0.05,
                "safety": 0.03,
                "reputational": 0.02
            }
        
        total_score = 0.0
        total_weight = 0.0
        
        for category, weight in weighting_scheme.items():
            if category in risks:
                category_risk = risks[category]
                if isinstance(category_risk, dict):
                    # Calculate category score from probability and impact
                    probability = category_risk.get("probability", 0.5)
                    impact = category_risk.get("impact", 0.5)
                    category_score = probability * impact
                else:
                    category_score = float(category_risk)
                
                total_score += weight * category_score
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def monitor_risks(
        self,
        assessment_id: str,
        current_indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Monitor ongoing risks and update assessment.
        
        Args:
            assessment_id: ID of assessment to monitor
            current_indicators: Current risk indicator values
            
        Returns:
            Updated risk monitoring status
        """
        self.logger.info(f"Monitoring risks for assessment: {assessment_id}")
        
        # Find the assessment
        assessment = None
        for hist_assessment in self.assessment_history:
            if hist_assessment.assessment_id == assessment_id:
                assessment = hist_assessment
                break
        
        if not assessment:
            self.logger.warning(f"Assessment {assessment_id} not found for monitoring")
            return {"status": "error", "message": "Assessment not found"}
        
        # Update monitoring data
        if assessment_id not in self.risk_monitoring:
            self.risk_monitoring[assessment_id] = {
                "start_time": time.time(),
                "updates": [],
                "alerts": []
            }
        
        monitoring_data = self.risk_monitoring[assessment_id]
        
        # Check for risk threshold breaches
        alerts = []
        for factor in assessment.risk_factors:
            for indicator in factor.indicators:
                if indicator in current_indicators:
                    current_value = current_indicators[indicator]
                    
                    # Check if indicator suggests increased risk
                    if self._is_risk_threshold_breached(factor, indicator, current_value):
                        alerts.append({
                            "factor_id": factor.factor_id,
                            "indicator": indicator,
                            "current_value": current_value,
                            "severity": factor.severity.value,
                            "timestamp": time.time()
                        })
        
        # Update monitoring data
        monitoring_data["updates"].append({
            "timestamp": time.time(),
            "indicators": current_indicators,
            "alerts_generated": len(alerts)
        })
        monitoring_data["alerts"].extend(alerts)
        
        # Generate monitoring report
        return {
            "assessment_id": assessment_id,
            "monitoring_status": "active",
            "new_alerts": len(alerts),
            "total_alerts": len(monitoring_data["alerts"]),
            "last_update": time.time(),
            "risk_trend": self._calculate_risk_trend(monitoring_data),
            "recommendations": self._generate_monitoring_recommendations(alerts, assessment)
        }
    
    def _assess_category_risks(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        category: RiskCategory,
        domain: str
    ) -> List[RiskFactor]:
        """Assess risks for a specific category."""
        factors = []
        
        if category == RiskCategory.FINANCIAL:
            factors.extend(self._assess_financial_risks(action, context, domain))
        elif category == RiskCategory.OPERATIONAL:
            factors.extend(self._assess_operational_risks(action, context, domain))
        elif category == RiskCategory.TECHNICAL:
            factors.extend(self._assess_technical_risks(action, context, domain))
        elif category == RiskCategory.ENVIRONMENTAL:
            factors.extend(self._assess_environmental_risks(action, context, domain))
        elif category == RiskCategory.REGULATORY:
            factors.extend(self._assess_regulatory_risks(action, context, domain))
        elif category == RiskCategory.SOCIAL:
            factors.extend(self._assess_social_risks(action, context, domain))
        elif category == RiskCategory.CULTURAL:
            factors.extend(self._assess_cultural_risks(action, context, domain))
        elif category == RiskCategory.TEMPORAL:
            factors.extend(self._assess_temporal_risks(action, context, domain))
        elif category == RiskCategory.SAFETY:
            factors.extend(self._assess_safety_risks(action, context, domain))
        elif category == RiskCategory.REPUTATIONAL:
            factors.extend(self._assess_reputational_risks(action, context, domain))
        
        return factors
    
    def _assess_financial_risks(self, action: Dict[str, Any], context: Dict[str, Any], domain: str) -> List[RiskFactor]:
        """Assess financial risks."""
        factors = []
        
        # Budget overrun risk
        budget = action.get("budget", 100000)
        complexity = self._assess_action_complexity(action)
        
        overrun_probability = min(0.8, 0.2 + complexity * 0.4)
        overrun_impact = min(1.0, 0.3 + (budget / 50000) * 0.1)
        
        factors.append(RiskFactor(
            factor_id="budget_overrun",
            category=RiskCategory.FINANCIAL,
            description="Risk of exceeding allocated budget",
            probability=overrun_probability,
            impact=overrun_impact,
            severity=self._calculate_severity(overrun_probability, overrun_impact),
            mitigation_strategies=[
                "Implement strict budget monitoring",
                "Establish contingency fund (15-20%)",
                "Regular cost reviews and approvals",
                "Phased budget release"
            ],
            indicators=["monthly_spend_rate", "cost_variance", "scope_changes"],
            dependencies=["resource_availability", "scope_stability"]
        ))
        
        # Funding security risk
        funding_security = context.get("funding_security", 0.8)
        if funding_security < 0.9:
            factors.append(RiskFactor(
                factor_id="funding_insecurity",
                category=RiskCategory.FINANCIAL,
                description="Risk of funding withdrawal or reduction",
                probability=1.0 - funding_security,
                impact=0.9,  # High impact if funding is lost
                severity=self._calculate_severity(1.0 - funding_security, 0.9),
                mitigation_strategies=[
                    "Diversify funding sources",
                    "Secure multi-year commitments",
                    "Maintain strong stakeholder relationships",
                    "Develop alternative funding plans"
                ],
                indicators=["funder_satisfaction", "political_stability", "economic_conditions"],
                dependencies=["stakeholder_support", "project_performance"]
            ))
        
        return factors
    
    def _assess_operational_risks(self, action: Dict[str, Any], context: Dict[str, Any], domain: str) -> List[RiskFactor]:
        """Assess operational risks."""
        factors = []
        
        # Resource availability risk
        resource_availability = context.get("resource_availability", 0.7)
        if resource_availability < 0.8:
            factors.append(RiskFactor(
                factor_id="resource_shortage",
                category=RiskCategory.OPERATIONAL,
                description="Risk of insufficient resources (personnel, equipment)",
                probability=1.0 - resource_availability,
                impact=0.7,
                severity=self._calculate_severity(1.0 - resource_availability, 0.7),
                mitigation_strategies=[
                    "Develop resource contingency plans",
                    "Cross-train team members",
                    "Establish equipment backup options",
                    "Build strategic partnerships"
                ],
                indicators=["staff_availability", "equipment_status", "supplier_reliability"],
                dependencies=["budget", "market_conditions"]
            ))
        
        # Team coordination risk
        team_size = len(action.get("stakeholders", []))
        if team_size > 8:  # Large teams have coordination challenges
            coordination_risk = min(0.6, (team_size - 5) * 0.1)
            factors.append(RiskFactor(
                factor_id="team_coordination",
                category=RiskCategory.OPERATIONAL,
                description="Risk of poor team coordination and communication",
                probability=coordination_risk,
                impact=0.5,
                severity=self._calculate_severity(coordination_risk, 0.5),
                mitigation_strategies=[
                    "Implement clear communication protocols",
                    "Regular team meetings and updates",
                    "Use project management tools",
                    "Define clear roles and responsibilities"
                ],
                indicators=["communication_frequency", "conflict_incidents", "milestone_delays"],
                dependencies=["team_experience", "leadership_quality"]
            ))
        
        return factors
    
    def _assess_technical_risks(self, action: Dict[str, Any], context: Dict[str, Any], domain: str) -> List[RiskFactor]:
        """Assess technical risks."""
        factors = []
        
        # Technical complexity risk
        complexity = self._assess_action_complexity(action)
        if complexity > 0.6:
            factors.append(RiskFactor(
                factor_id="technical_complexity",
                category=RiskCategory.TECHNICAL,
                description="Risk from high technical complexity",
                probability=complexity,
                impact=0.6,
                severity=self._calculate_severity(complexity, 0.6),
                mitigation_strategies=[
                    "Break down into smaller components",
                    "Prototype critical elements",
                    "Engage technical experts",
                    "Implement rigorous testing"
                ],
                indicators=["technical_issues", "expert_confidence", "prototype_results"],
                dependencies=["team_expertise", "technology_maturity"]
            ))
        
        # Equipment failure risk (domain-specific)
        if domain == "archaeological":
            factors.append(RiskFactor(
                factor_id="equipment_failure",
                category=RiskCategory.TECHNICAL,
                description="Risk of specialized equipment failure in field conditions",
                probability=0.3,
                impact=0.4,
                severity=RiskSeverity.MEDIUM,
                mitigation_strategies=[
                    "Maintain backup equipment",
                    "Regular equipment maintenance",
                    "Train team on equipment repair",
                    "Establish rapid replacement protocols"
                ],
                indicators=["equipment_age", "maintenance_records", "failure_history"],
                dependencies=["budget", "supplier_support"]
            ))
        
        return factors
    
    def _assess_environmental_risks(self, action: Dict[str, Any], context: Dict[str, Any], domain: str) -> List[RiskFactor]:
        """Assess environmental risks."""
        factors = []
        
        # Weather dependency risk
        weather_dependency = context.get("weather_dependency", 0.5)
        if weather_dependency > 0.4:
            factors.append(RiskFactor(
                factor_id="weather_disruption",
                category=RiskCategory.ENVIRONMENTAL,
                description="Risk of weather-related disruptions",
                probability=weather_dependency,
                impact=0.5,
                severity=self._calculate_severity(weather_dependency, 0.5),
                mitigation_strategies=[
                    "Plan for seasonal weather patterns",
                    "Develop weather contingency protocols",
                    "Invest in weather protection equipment",
                    "Build schedule flexibility"
                ],
                indicators=["weather_forecasts", "seasonal_patterns", "site_exposure"],
                dependencies=["timeline_flexibility", "equipment_weatherproofing"]
            ))
        
        # Environmental impact risk
        if domain in ["archaeological", "heritage_preservation"]:
            factors.append(RiskFactor(
                factor_id="environmental_damage",
                category=RiskCategory.ENVIRONMENTAL,
                description="Risk of causing environmental damage during operations",
                probability=0.2,
                impact=0.8,  # High impact due to irreversibility
                severity=RiskSeverity.MEDIUM,
                mitigation_strategies=[
                    "Conduct environmental impact assessments",
                    "Implement strict environmental protocols",
                    "Monitor environmental indicators",
                    "Engage environmental consultants"
                ],
                indicators=["environmental_monitoring", "compliance_status", "incident_reports"],
                dependencies=["regulatory_compliance", "team_training"]
            ))
        
        return factors
    
    def _assess_regulatory_risks(self, action: Dict[str, Any], context: Dict[str, Any], domain: str) -> List[RiskFactor]:
        """Assess regulatory risks."""
        factors = []
        
        # Permit and compliance risk
        regulatory_compliance = context.get("regulatory_compliance", 0.8)
        if regulatory_compliance < 0.95:
            factors.append(RiskFactor(
                factor_id="regulatory_non_compliance",
                category=RiskCategory.REGULATORY,
                description="Risk of regulatory non-compliance or permit issues",
                probability=1.0 - regulatory_compliance,
                impact=0.9,  # High impact due to potential project shutdown
                severity=self._calculate_severity(1.0 - regulatory_compliance, 0.9),
                mitigation_strategies=[
                    "Engage regulatory experts early",
                    "Maintain comprehensive compliance documentation",
                    "Regular compliance audits",
                    "Build relationships with regulatory bodies"
                ],
                indicators=["permit_status", "compliance_audits", "regulatory_changes"],
                dependencies=["legal_expertise", "documentation_quality"]
            ))
        
        # Regulatory change risk
        if domain in ["archaeological", "heritage_preservation"]:
            factors.append(RiskFactor(
                factor_id="regulatory_changes",
                category=RiskCategory.REGULATORY,
                description="Risk of changing regulations affecting project",
                probability=0.15,
                impact=0.6,
                severity=RiskSeverity.LOW,
                mitigation_strategies=[
                    "Monitor regulatory developments",
                    "Engage with policy makers",
                    "Build flexibility into project design",
                    "Maintain legal counsel"
                ],
                indicators=["policy_discussions", "legislative_calendar", "industry_trends"],
                dependencies=["political_stability", "advocacy_efforts"]
            ))
        
        return factors
    
    def _assess_social_risks(self, action: Dict[str, Any], context: Dict[str, Any], domain: str) -> List[RiskFactor]:
        """Assess social risks."""
        factors = []
        
        # Community opposition risk
        community_support = context.get("community_support", 0.7)
        if community_support < 0.8:
            factors.append(RiskFactor(
                factor_id="community_opposition",
                category=RiskCategory.SOCIAL,
                description="Risk of community opposition or protests",
                probability=1.0 - community_support,
                impact=0.7,
                severity=self._calculate_severity(1.0 - community_support, 0.7),
                mitigation_strategies=[
                    "Engage community early and often",
                    "Address community concerns transparently",
                    "Provide community benefits",
                    "Establish community liaison roles"
                ],
                indicators=["community_meetings", "media_coverage", "petition_activity"],
                dependencies=["communication_quality", "community_benefits"]
            ))
        
        # Stakeholder conflict risk
        stakeholder_count = len(action.get("stakeholders", []))
        if stakeholder_count > 5:
            conflict_probability = min(0.5, (stakeholder_count - 3) * 0.08)
            factors.append(RiskFactor(
                factor_id="stakeholder_conflicts",
                category=RiskCategory.SOCIAL,
                description="Risk of conflicts between stakeholders",
                probability=conflict_probability,
                impact=0.5,
                severity=self._calculate_severity(conflict_probability, 0.5),
                mitigation_strategies=[
                    "Facilitate stakeholder alignment sessions",
                    "Establish conflict resolution procedures",
                    "Maintain transparent communication",
                    "Define clear decision-making processes"
                ],
                indicators=["stakeholder_satisfaction", "conflict_incidents", "meeting_attendance"],
                dependencies=["leadership_quality", "communication_protocols"]
            ))
        
        return factors
    
    def _assess_cultural_risks(self, action: Dict[str, Any], context: Dict[str, Any], domain: str) -> List[RiskFactor]:
        """Assess cultural risks."""
        factors = []
        
        if domain in ["archaeological", "heritage_preservation"]:
            # Cultural sensitivity risk
            factors.append(RiskFactor(
                factor_id="cultural_insensitivity",
                category=RiskCategory.CULTURAL,
                description="Risk of cultural insensitivity or sacred site violations",
                probability=0.2,
                impact=0.9,  # Very high impact due to irreversible cultural damage
                severity=RiskSeverity.MEDIUM,
                mitigation_strategies=[
                    "Engage indigenous and local communities",
                    "Conduct cultural sensitivity training",
                    "Establish cultural protocols",
                    "Include cultural advisors in team"
                ],
                indicators=["community_feedback", "cultural_advisor_input", "protocol_compliance"],
                dependencies=["community_engagement", "cultural_expertise"]
            ))
            
            # Artifact repatriation risk
            factors.append(RiskFactor(
                factor_id="repatriation_disputes",
                category=RiskCategory.CULTURAL,
                description="Risk of disputes over artifact ownership and repatriation",
                probability=0.15,
                impact=0.6,
                severity=RiskSeverity.LOW,
                mitigation_strategies=[
                    "Establish clear ownership agreements upfront",
                    "Follow international repatriation guidelines",
                    "Maintain transparent artifact documentation",
                    "Engage legal experts in cultural property"
                ],
                indicators=["ownership_clarity", "documentation_quality", "legal_challenges"],
                dependencies=["legal_framework", "community_agreements"]
            ))
        
        return factors
    
    def _assess_temporal_risks(self, action: Dict[str, Any], context: Dict[str, Any], domain: str) -> List[RiskFactor]:
        """Assess temporal/timeline risks."""
        factors = []
        
        # Schedule delay risk
        complexity = self._assess_action_complexity(action)
        delay_probability = min(0.7, 0.2 + complexity * 0.3)
        
        factors.append(RiskFactor(
            factor_id="schedule_delays",
            category=RiskCategory.TEMPORAL,
            description="Risk of project schedule delays",
            probability=delay_probability,
            impact=0.4,
            severity=self._calculate_severity(delay_probability, 0.4),
            mitigation_strategies=[
                "Build buffer time into schedule",
                "Identify critical path dependencies",
                "Implement milestone tracking",
                "Develop acceleration strategies"
            ],
            indicators=["milestone_progress", "critical_path_status", "resource_delays"],
            dependencies=["resource_availability", "external_dependencies"]
        ))
        
        return factors
    
    def _assess_safety_risks(self, action: Dict[str, Any], context: Dict[str, Any], domain: str) -> List[RiskFactor]:
        """Assess safety risks."""
        factors = []
        
        if domain == "archaeological":
            # Field work safety risk
            factors.append(RiskFactor(
                factor_id="fieldwork_safety",
                category=RiskCategory.SAFETY,
                description="Risk of injuries during fieldwork activities",
                probability=0.1,
                impact=0.8,  # High impact due to potential for serious injury
                severity=RiskSeverity.LOW,
                mitigation_strategies=[
                    "Implement comprehensive safety protocols",
                    "Provide safety training and equipment",
                    "Conduct regular safety inspections",
                    "Maintain emergency response procedures"
                ],
                indicators=["safety_incidents", "training_completion", "equipment_status"],
                dependencies=["safety_training", "equipment_quality"]
            ))
        
        return factors
    
    def _assess_reputational_risks(self, action: Dict[str, Any], context: Dict[str, Any], domain: str) -> List[RiskFactor]:
        """Assess reputational risks."""
        factors = []
        
        # Public perception risk
        media_attention = context.get("media_attention", 0.3)
        if media_attention > 0.5:
            factors.append(RiskFactor(
                factor_id="negative_publicity",
                category=RiskCategory.REPUTATIONAL,
                description="Risk of negative media coverage or public perception",
                probability=media_attention * 0.4,
                impact=0.5,
                severity=self._calculate_severity(media_attention * 0.4, 0.5),
                mitigation_strategies=[
                    "Develop proactive communication strategy",
                    "Engage media relations professionals",
                    "Maintain transparency in operations",
                    "Build positive community relationships"
                ],
                indicators=["media_coverage", "social_media_sentiment", "public_feedback"],
                dependencies=["communication_quality", "project_performance"]
            ))
        
        return factors
    
    def _calculate_overall_risk_score(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate overall risk score from individual factors."""
        if not risk_factors:
            return 0.0
        
        # Weight factors by category importance
        category_weights = {
            RiskCategory.FINANCIAL: 0.20,
            RiskCategory.OPERATIONAL: 0.15,
            RiskCategory.TECHNICAL: 0.12,
            RiskCategory.ENVIRONMENTAL: 0.10,
            RiskCategory.REGULATORY: 0.15,
            RiskCategory.SOCIAL: 0.08,
            RiskCategory.CULTURAL: 0.10,
            RiskCategory.TEMPORAL: 0.05,
            RiskCategory.SAFETY: 0.03,
            RiskCategory.REPUTATIONAL: 0.02
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for factor in risk_factors:
            weight = category_weights.get(factor.category, 0.05)
            factor_score = factor.probability * factor.impact
            total_score += weight * factor_score
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_risk_level(self, risk_score: float) -> RiskSeverity:
        """Determine risk level from overall score."""
        if risk_score >= 0.8:
            return RiskSeverity.CRITICAL
        elif risk_score >= 0.6:
            return RiskSeverity.HIGH
        elif risk_score >= 0.4:
            return RiskSeverity.MEDIUM
        elif risk_score >= 0.2:
            return RiskSeverity.LOW
        else:
            return RiskSeverity.NEGLIGIBLE
    
    def _calculate_severity(self, probability: float, impact: float) -> RiskSeverity:
        """Calculate severity from probability and impact."""
        risk_score = probability * impact
        return self._determine_risk_level(risk_score)
    
    def _generate_mitigation_plan(
        self,
        risk_factors: List[RiskFactor],
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive mitigation plan."""
        # Prioritize risks by severity and score
        high_priority_risks = [
            f for f in risk_factors 
            if f.severity in [RiskSeverity.CRITICAL, RiskSeverity.HIGH]
        ]
        
        mitigation_plan = {
            "immediate_actions": [],
            "short_term_strategies": [],
            "long_term_measures": [],
            "resource_requirements": {},
            "timeline": {},
            "success_metrics": []
        }
        
        for factor in high_priority_risks:
            # Categorize mitigation strategies by timeline
            for strategy in factor.mitigation_strategies:
                if any(word in strategy.lower() for word in ["immediate", "urgent", "emergency"]):
                    mitigation_plan["immediate_actions"].append({
                        "risk_factor": factor.factor_id,
                        "strategy": strategy,
                        "priority": "high"
                    })
                elif any(word in strategy.lower() for word in ["establish", "implement", "develop"]):
                    mitigation_plan["short_term_strategies"].append({
                        "risk_factor": factor.factor_id,
                        "strategy": strategy,
                        "timeline": "1-3 months"
                    })
                else:
                    mitigation_plan["long_term_measures"].append({
                        "risk_factor": factor.factor_id,
                        "strategy": strategy,
                        "timeline": "3+ months"
                    })
        
        return mitigation_plan
    
    def _identify_monitoring_requirements(self, risk_factors: List[RiskFactor]) -> List[str]:
        """Identify monitoring requirements for ongoing risk management."""
        monitoring_requirements = []
        
        # Collect all indicators from risk factors
        all_indicators = set()
        for factor in risk_factors:
            all_indicators.update(factor.indicators)
        
        # Convert to monitoring requirements
        for indicator in all_indicators:
            monitoring_requirements.append(f"Monitor {indicator} regularly")
        
        # Add general monitoring requirements
        monitoring_requirements.extend([
            "Conduct weekly risk review meetings",
            "Update risk register monthly",
            "Perform quarterly comprehensive risk assessment",
            "Maintain incident reporting system"
        ])
        
        return monitoring_requirements[:10]  # Limit to top 10
    
    def _generate_contingency_plans(
        self,
        risk_factors: List[RiskFactor],
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate contingency plans for high-impact risks."""
        contingency_plans = []
        
        # Focus on high-impact risks
        high_impact_risks = [f for f in risk_factors if f.impact >= 0.7]
        
        for factor in high_impact_risks:
            plan = {
                "risk_factor": factor.factor_id,
                "trigger_conditions": [f"High {indicator}" for indicator in factor.indicators[:2]],
                "response_actions": factor.mitigation_strategies[:3],
                "escalation_criteria": f"{factor.severity.value} severity reached",
                "resource_allocation": "TBD based on specific trigger",
                "communication_plan": "Notify all stakeholders within 24 hours"
            }
            contingency_plans.append(plan)
        
        return contingency_plans[:5]  # Limit to top 5 most critical
    
    def _generate_risk_recommendations(
        self,
        risk_factors: List[RiskFactor],
        overall_risk_score: float
    ) -> List[str]:
        """Generate actionable risk management recommendations."""
        recommendations = []
        
        # Overall risk level recommendations
        if overall_risk_score >= 0.7:
            recommendations.append("Overall risk level is high - consider project redesign or delay")
            recommendations.append("Implement immediate risk mitigation measures")
        elif overall_risk_score >= 0.5:
            recommendations.append("Moderate risk level - strengthen mitigation strategies")
        
        # Category-specific recommendations
        category_risks = {}
        for factor in risk_factors:
            if factor.category not in category_risks:
                category_risks[factor.category] = []
            category_risks[factor.category].append(factor)
        
        for category, factors in category_risks.items():
            if len(factors) > 2:  # Multiple risks in same category
                recommendations.append(f"Address multiple {category.value} risks through coordinated approach")
        
        # High-severity factor recommendations
        critical_factors = [f for f in risk_factors if f.severity == RiskSeverity.CRITICAL]
        for factor in critical_factors:
            recommendations.append(f"Critical risk: {factor.description} - immediate action required")
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def _calculate_confidence_score(
        self,
        risk_factors: List[RiskFactor],
        context: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for the risk assessment."""
        # Base confidence
        confidence = 0.7
        
        # Adjust based on data completeness
        context_completeness = len([v for v in context.values() if v is not None]) / max(1, len(context))
        confidence += (context_completeness - 0.5) * 0.2
        
        # Adjust based on number of risk factors assessed
        factor_coverage = min(1.0, len(risk_factors) / 15)  # Assume 15 is comprehensive
        confidence += (factor_coverage - 0.5) * 0.1
        
        return max(0.3, min(0.95, confidence))
    
    # Utility methods
    def _assess_action_complexity(self, action: Dict[str, Any]) -> float:
        """Assess complexity of an action."""
        complexity_indicators = [
            len(action.get("steps", [])) / 10.0,
            len(action.get("dependencies", [])) / 5.0,
            len(action.get("stakeholders", [])) / 8.0,
            action.get("technical_difficulty", 0.5),
            action.get("regulatory_complexity", 0.3)
        ]
        
        return min(1.0, sum(complexity_indicators) / len(complexity_indicators))
    
    def _identify_domain(self, action: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Identify the domain of the action."""
        action_type = action.get("type", "").lower()
        context_domain = context.get("domain", "").lower()
        
        if "archaeolog" in action_type or "excavat" in action_type:
            return "archaeological"
        elif "heritage" in action_type or "preserv" in action_type:
            return "heritage_preservation"
        elif "environment" in action_type:
            return "environmental"
        else:
            return "general"
    
    def _is_risk_threshold_breached(
        self,
        factor: RiskFactor,
        indicator: str,
        current_value: Any
    ) -> bool:
        """Check if a risk threshold has been breached."""
        # Simplified threshold checking
        if isinstance(current_value, (int, float)):
            if "high" in indicator.lower() or "increase" in indicator.lower():
                return current_value > 0.7
            elif "low" in indicator.lower() or "decrease" in indicator.lower():
                return current_value < 0.3
        
        return False
    
    def _calculate_risk_trend(self, monitoring_data: Dict[str, Any]) -> str:
        """Calculate risk trend from monitoring data."""
        updates = monitoring_data.get("updates", [])
        if len(updates) < 2:
            return "insufficient_data"
        
        recent_alerts = sum(update.get("alerts_generated", 0) for update in updates[-3:])
        earlier_alerts = sum(update.get("alerts_generated", 0) for update in updates[-6:-3])
        
        if recent_alerts > earlier_alerts * 1.2:
            return "increasing"
        elif recent_alerts < earlier_alerts * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _generate_monitoring_recommendations(
        self,
        alerts: List[Dict[str, Any]],
        assessment: RiskAssessment
    ) -> List[str]:
        """Generate recommendations based on monitoring alerts."""
        recommendations = []
        
        if len(alerts) > 3:
            recommendations.append("Multiple risk alerts - conduct immediate risk review")
        
        critical_alerts = [a for a in alerts if a.get("severity") == "critical"]
        if critical_alerts:
            recommendations.append("Critical risk alerts detected - activate contingency plans")
        
        return recommendations
    
    # Initialization methods
    def _initialize_risk_models(self) -> Dict[str, Any]:
        """Initialize risk assessment models."""
        return {
            "financial_model": {"budget_variance": 0.3, "funding_stability": 0.4},
            "operational_model": {"resource_efficiency": 0.25, "coordination_quality": 0.2},
            "technical_model": {"complexity_factor": 0.3, "innovation_risk": 0.25},
            "environmental_model": {"weather_impact": 0.4, "site_conditions": 0.3},
            "regulatory_model": {"compliance_score": 0.5, "regulatory_stability": 0.3}
        }
    
    def _initialize_archaeological_risks(self) -> Dict[str, Any]:
        """Initialize archaeological domain-specific risks."""
        return {
            "site_access": {"probability": 0.2, "impact": 0.6},
            "artifact_damage": {"probability": 0.15, "impact": 0.9},
            "weather_delays": {"probability": 0.4, "impact": 0.5},
            "community_relations": {"probability": 0.25, "impact": 0.7},
            "permit_issues": {"probability": 0.1, "impact": 0.8}
        }
    
    def _initialize_heritage_risks(self) -> Dict[str, Any]:
        """Initialize heritage preservation domain-specific risks."""
        return {
            "structural_damage": {"probability": 0.2, "impact": 0.9},
            "visitor_safety": {"probability": 0.15, "impact": 0.8},
            "funding_cuts": {"probability": 0.3, "impact": 0.7},
            "maintenance_backlog": {"probability": 0.4, "impact": 0.6},
            "tourism_impact": {"probability": 0.25, "impact": 0.5}
        }
    
    def _initialize_mitigation_strategies(self) -> Dict[str, List[str]]:
        """Initialize database of mitigation strategies."""
        return {
            "financial": [
                "Establish contingency funds",
                "Diversify funding sources",
                "Implement cost controls",
                "Regular budget reviews"
            ],
            "operational": [
                "Cross-train team members",
                "Develop backup plans",
                "Improve communication",
                "Streamline processes"
            ],
            "technical": [
                "Prototype solutions",
                "Engage experts",
                "Implement testing",
                "Document procedures"
            ],
            "environmental": [
                "Monitor conditions",
                "Develop contingencies",
                "Use protective measures",
                "Plan for seasons"
            ],
            "regulatory": [
                "Engage early",
                "Maintain compliance",
                "Monitor changes",
                "Build relationships"
            ]
        }
    
    def get_risk_statistics(self) -> Dict[str, Any]:
        """Get statistics about risk assessment history."""
        if not self.assessment_history:
            return {"total_assessments": 0}
        
        return {
            "total_assessments": len(self.assessment_history),
            "average_risk_score": sum(a.overall_risk_score for a in self.assessment_history) / len(self.assessment_history),
            "risk_level_distribution": {
                level.value: sum(1 for a in self.assessment_history if a.risk_level == level)
                for level in RiskSeverity
            },
            "most_common_risks": self._get_most_common_risk_categories(),
            "active_monitoring": len(self.risk_monitoring)
        }
    
    def _get_most_common_risk_categories(self) -> Dict[str, int]:
        """Get most common risk categories from assessment history."""
        category_counts = {}
        
        for assessment in self.assessment_history:
            for factor in assessment.risk_factors:
                category = factor.category.value
                category_counts[category] = category_counts.get(category, 0) + 1
        
        return dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]) 
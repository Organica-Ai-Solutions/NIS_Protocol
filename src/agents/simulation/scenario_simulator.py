"""
NIS Protocol Scenario Simulator

This module provides advanced scenario simulation capabilities for the AGI system.
Implements physics-based modeling, Monte Carlo simulation, and archaeological domain specialization.
"""

import logging
import time
import random
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ScenarioType(Enum):
    """Types of scenarios that can be simulated."""
    ARCHAEOLOGICAL_EXCAVATION = "archaeological_excavation"
    HERITAGE_PRESERVATION = "heritage_preservation"
    ENVIRONMENTAL_IMPACT = "environmental_impact"
    RESOURCE_ALLOCATION = "resource_allocation"
    DECISION_MAKING = "decision_making"
    RISK_MITIGATION = "risk_mitigation"
    CULTURAL_INTERACTION = "cultural_interaction"
    TEMPORAL_ANALYSIS = "temporal_analysis"


@dataclass
class SimulationParameters:
    """Parameters for scenario simulation."""
    time_horizon: float  # Simulation time in hours/days
    resolution: float    # Time step resolution
    iterations: int      # Monte Carlo iterations
    confidence_level: float  # Statistical confidence level
    environmental_factors: Dict[str, float]
    resource_constraints: Dict[str, float]
    uncertainty_factors: Dict[str, float]


@dataclass
class SimulationResult:
    """Result of a scenario simulation."""
    scenario_id: str
    scenario_type: ScenarioType
    success_probability: float
    expected_outcomes: List[Dict[str, Any]]
    risk_factors: List[Dict[str, str]]
    resource_utilization: Dict[str, float]
    timeline: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    recommendations: List[str]
    metadata: Dict[str, Any]


class ScenarioSimulator:
    """Advanced scenario simulator for decision-making and planning.
    
    This simulator provides:
    - Physics-based scenario modeling
    - Monte Carlo simulation for uncertainty handling
    - Archaeological domain specialization
    - Multi-factor outcome prediction
    - Resource optimization analysis
    """
    
    def __init__(self):
        """Initialize the scenario simulator."""
        self.logger = logging.getLogger("nis.scenario_simulator")
        
        # Simulation state
        self.active_simulations: Dict[str, Dict[str, Any]] = {}
        self.simulation_history: List[SimulationResult] = []
        
        # Domain knowledge for archaeological scenarios
        self.archaeological_factors = {
            "soil_composition": {"clay": 0.3, "sand": 0.4, "rock": 0.3},
            "weather_sensitivity": 0.7,
            "artifact_fragility": 0.8,
            "cultural_significance": 0.9,
            "community_involvement": 0.85
        }
        
        # Environmental simulation parameters
        self.environmental_models = {
            "weather": {"temperature": 20.0, "humidity": 0.6, "precipitation": 0.2},
            "geological": {"stability": 0.8, "erosion_rate": 0.1, "contamination": 0.05},
            "biological": {"vegetation_cover": 0.7, "wildlife_impact": 0.3}
        }
        
        # Resource modeling
        self.resource_types = {
            "human_resources": {"archaeologists": 5, "technicians": 3, "volunteers": 10},
            "equipment": {"excavation_tools": 20, "analysis_equipment": 5, "preservation_materials": 15},
            "financial": {"budget": 100000, "contingency": 20000},
            "time": {"available_days": 90, "weather_dependent_days": 60}
        }
        
        self.logger.info("ScenarioSimulator initialized with archaeological domain specialization")
    
    def simulate_scenario(
        self,
        scenario: Dict[str, Any],
        parameters: Optional[SimulationParameters] = None
    ) -> SimulationResult:
        """Simulate a given scenario and return comprehensive results.
        
        Args:
            scenario: Scenario definition with type, objectives, constraints
            parameters: Simulation parameters (uses defaults if None)
            
        Returns:
            Comprehensive simulation results
        """
        scenario_id = scenario.get("id", f"sim_{int(time.time())}")
        scenario_type = ScenarioType(scenario.get("type", "decision_making"))
        
        self.logger.info(f"Simulating scenario: {scenario_id} ({scenario_type.value})")
        
        # Use default parameters if none provided
        if parameters is None:
            parameters = self._get_default_parameters(scenario_type)
        
        # Initialize simulation
        simulation_state = self._initialize_simulation(scenario, parameters)
        self.active_simulations[scenario_id] = simulation_state
        
        try:
            # Run Monte Carlo simulation
            outcomes = self._run_monte_carlo_simulation(scenario, parameters)
            
            # Analyze results
            analysis = self._analyze_simulation_results(outcomes, scenario, parameters)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(analysis, scenario)
            
            # Create result object
            result = SimulationResult(
                scenario_id=scenario_id,
                scenario_type=scenario_type,
                success_probability=analysis["success_probability"],
                expected_outcomes=analysis["expected_outcomes"],
                risk_factors=analysis["risk_factors"],
                resource_utilization=analysis["resource_utilization"],
                timeline=analysis["timeline"],
                confidence_intervals=analysis["confidence_intervals"],
                recommendations=recommendations,
                metadata={
                    "simulation_time": time.time() - simulation_state["start_time"],
                    "iterations_completed": parameters.iterations,
                    "convergence_achieved": analysis.get("convergence", True),
                    "domain_specialization": scenario_type.value
                }
            )
            
            # Store result
            self.simulation_history.append(result)
            
            self.logger.info(f"Scenario simulation completed: {scenario_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Simulation failed for scenario {scenario_id}: {str(e)}")
            raise
        finally:
            # Clean up active simulation
            if scenario_id in self.active_simulations:
                del self.active_simulations[scenario_id]
    
    def create_scenario_variations(
        self,
        base_scenario: Dict[str, Any],
        variation_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Create variations of a base scenario for comprehensive analysis.
        
        Args:
            base_scenario: Base scenario to create variations from
            variation_types: Types of variations to create
            
        Returns:
            List of scenario variations
        """
        if variation_types is None:
            variation_types = ["resource", "environmental", "temporal", "risk"]
        
        variations = []
        base_id = base_scenario.get("id", "base")
        
        self.logger.info(f"Creating scenario variations for: {base_id}")
        
        for variation_type in variation_types:
            if variation_type == "resource":
                variations.extend(self._create_resource_variations(base_scenario))
            elif variation_type == "environmental":
                variations.extend(self._create_environmental_variations(base_scenario))
            elif variation_type == "temporal":
                variations.extend(self._create_temporal_variations(base_scenario))
            elif variation_type == "risk":
                variations.extend(self._create_risk_variations(base_scenario))
        
        self.logger.info(f"Created {len(variations)} scenario variations")
        return variations
    
    def compare_scenarios(
        self,
        scenarios: List[Dict[str, Any]],
        comparison_criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple scenarios and provide analysis.
        
        Args:
            scenarios: List of scenarios to compare
            comparison_criteria: Criteria for comparison
            
        Returns:
            Comparison analysis
        """
        if comparison_criteria is None:
            comparison_criteria = ["success_probability", "resource_efficiency", "risk_level", "timeline"]
        
        self.logger.info(f"Comparing {len(scenarios)} scenarios")
        
        # Simulate all scenarios
        results = []
        for scenario in scenarios:
            result = self.simulate_scenario(scenario)
            results.append(result)
        
        # Perform comparison analysis
        comparison = {
            "scenarios_compared": len(scenarios),
            "criteria": comparison_criteria,
            "rankings": {},
            "trade_offs": [],
            "recommendations": []
        }
        
        # Rank scenarios by each criterion
        for criterion in comparison_criteria:
            if criterion == "success_probability":
                ranking = sorted(results, key=lambda r: r.success_probability, reverse=True)
            elif criterion == "resource_efficiency":
                ranking = sorted(results, key=lambda r: self._calculate_resource_efficiency(r))
            elif criterion == "risk_level":
                ranking = sorted(results, key=lambda r: self._calculate_overall_risk(r))
            elif criterion == "timeline":
                ranking = sorted(results, key=lambda r: r.timeline.get("total_duration", float('inf')))
            else:
                ranking = results  # Default order
            
            comparison["rankings"][criterion] = [r.scenario_id for r in ranking]
        
        # Identify trade-offs
        comparison["trade_offs"] = self._identify_trade_offs(results)
        
        # Generate comparison recommendations
        comparison["recommendations"] = self._generate_comparison_recommendations(results, comparison)
        
        return comparison
    
    def _initialize_simulation(
        self,
        scenario: Dict[str, Any],
        parameters: SimulationParameters
    ) -> Dict[str, Any]:
        """Initialize simulation state."""
        return {
            "start_time": time.time(),
            "scenario": scenario,
            "parameters": parameters,
            "current_iteration": 0,
            "outcomes": [],
            "convergence_metrics": []
        }
    
    def _get_default_parameters(self, scenario_type: ScenarioType) -> SimulationParameters:
        """Get default simulation parameters based on scenario type."""
        if scenario_type == ScenarioType.ARCHAEOLOGICAL_EXCAVATION:
            return SimulationParameters(
                time_horizon=90.0,  # 90 days
                resolution=1.0,     # 1 day resolution
                iterations=1000,
                confidence_level=0.95,
                environmental_factors={"weather": 0.7, "seasonal": 0.6, "geological": 0.8},
                resource_constraints={"budget": 0.8, "personnel": 0.7, "equipment": 0.9},
                uncertainty_factors={"discovery": 0.5, "preservation": 0.3, "community": 0.4}
            )
        elif scenario_type == ScenarioType.HERITAGE_PRESERVATION:
            return SimulationParameters(
                time_horizon=365.0,  # 1 year
                resolution=7.0,      # 1 week resolution
                iterations=500,
                confidence_level=0.90,
                environmental_factors={"deterioration": 0.6, "climate": 0.8, "pollution": 0.5},
                resource_constraints={"funding": 0.6, "expertise": 0.8, "materials": 0.7},
                uncertainty_factors={"effectiveness": 0.4, "longevity": 0.6, "cost": 0.3}
            )
        else:
            # Default parameters
            return SimulationParameters(
                time_horizon=30.0,
                resolution=1.0,
                iterations=500,
                confidence_level=0.90,
                environmental_factors={"general": 0.5},
                resource_constraints={"general": 0.7},
                uncertainty_factors={"general": 0.4}
            )
    
    def _run_monte_carlo_simulation(
        self,
        scenario: Dict[str, Any],
        parameters: SimulationParameters
    ) -> List[Dict[str, Any]]:
        """Run Monte Carlo simulation for the scenario."""
        outcomes = []
        
        for iteration in range(parameters.iterations):
            # Generate random variables for this iteration
            random_factors = self._generate_random_factors(parameters)
            
            # Simulate single iteration
            outcome = self._simulate_single_iteration(scenario, parameters, random_factors)
            outcomes.append(outcome)
            
            # Check convergence every 100 iterations
            if iteration > 0 and iteration % 100 == 0:
                if self._check_convergence(outcomes[-100:]):
                    self.logger.info(f"Convergence achieved at iteration {iteration}")
                    break
        
        return outcomes
    
    def _simulate_single_iteration(
        self,
        scenario: Dict[str, Any],
        parameters: SimulationParameters,
        random_factors: Dict[str, float]
    ) -> Dict[str, Any]:
        """Simulate a single iteration of the scenario."""
        scenario_type = ScenarioType(scenario.get("type", "decision_making"))
        
        if scenario_type == ScenarioType.ARCHAEOLOGICAL_EXCAVATION:
            return self._simulate_archaeological_excavation(scenario, parameters, random_factors)
        elif scenario_type == ScenarioType.HERITAGE_PRESERVATION:
            return self._simulate_heritage_preservation(scenario, parameters, random_factors)
        else:
            return self._simulate_generic_scenario(scenario, parameters, random_factors)
    
    def _simulate_archaeological_excavation(
        self,
        scenario: Dict[str, Any],
        parameters: SimulationParameters,
        random_factors: Dict[str, float]
    ) -> Dict[str, Any]:
        """Simulate an archaeological excavation scenario."""
        # Initialize outcome
        outcome = {
            "success": False,
            "artifacts_discovered": 0,
            "preservation_quality": 0.0,
            "community_engagement": 0.0,
            "budget_utilization": 0.0,
            "timeline_adherence": 0.0,
            "cultural_sensitivity": 0.0,
            "scientific_value": 0.0
        }
        
        # Simulate excavation process
        weather_factor = random_factors.get("weather", 0.7)
        soil_factor = random_factors.get("soil_composition", 0.6)
        team_efficiency = random_factors.get("team_efficiency", 0.8)
        
        # Calculate discovery probability
        base_discovery_rate = scenario.get("expected_discovery_rate", 0.3)
        discovery_modifier = weather_factor * soil_factor * team_efficiency
        actual_discovery_rate = base_discovery_rate * discovery_modifier
        
        # Simulate artifact discovery
        excavation_days = int(parameters.time_horizon)
        for day in range(excavation_days):
            if random.random() < actual_discovery_rate:
                outcome["artifacts_discovered"] += 1
                # Higher quality artifacts are rarer
                if random.random() < 0.2:  # 20% chance of high-value artifact
                    outcome["scientific_value"] += 0.3
                else:
                    outcome["scientific_value"] += 0.1
        
        # Calculate preservation quality
        preservation_factors = [
            random_factors.get("humidity", 0.6),
            random_factors.get("temperature_stability", 0.7),
            random_factors.get("handling_care", 0.8)
        ]
        outcome["preservation_quality"] = sum(preservation_factors) / len(preservation_factors)
        
        # Calculate community engagement
        community_factors = [
            random_factors.get("local_involvement", 0.7),
            random_factors.get("cultural_respect", 0.9),
            random_factors.get("communication_quality", 0.8)
        ]
        outcome["community_engagement"] = sum(community_factors) / len(community_factors)
        
        # Calculate budget utilization
        base_budget = scenario.get("budget", 100000)
        unexpected_costs = random_factors.get("unexpected_costs", 0.1)
        outcome["budget_utilization"] = min(1.0, (1.0 + unexpected_costs))
        
        # Calculate timeline adherence
        delays = random_factors.get("weather_delays", 0.0) + random_factors.get("permit_delays", 0.0)
        outcome["timeline_adherence"] = max(0.0, 1.0 - delays)
        
        # Calculate cultural sensitivity score
        sensitivity_factors = [
            random_factors.get("indigenous_consultation", 0.9),
            random_factors.get("sacred_site_respect", 0.95),
            random_factors.get("artifact_repatriation", 0.8)
        ]
        outcome["cultural_sensitivity"] = sum(sensitivity_factors) / len(sensitivity_factors)
        
        # Determine overall success
        success_threshold = 0.6
        overall_score = (
            0.2 * (outcome["artifacts_discovered"] / max(1, scenario.get("target_artifacts", 10))) +
            0.15 * outcome["preservation_quality"] +
            0.2 * outcome["community_engagement"] +
            0.15 * (2.0 - outcome["budget_utilization"]) +  # Lower budget usage is better
            0.15 * outcome["timeline_adherence"] +
            0.15 * outcome["cultural_sensitivity"]
        )
        
        outcome["success"] = overall_score >= success_threshold
        outcome["overall_score"] = overall_score
        
        return outcome
    
    def _simulate_heritage_preservation(
        self,
        scenario: Dict[str, Any],
        parameters: SimulationParameters,
        random_factors: Dict[str, float]
    ) -> Dict[str, Any]:
        """Simulate a heritage preservation scenario."""
        outcome = {
            "success": False,
            "preservation_effectiveness": 0.0,
            "structural_integrity": 0.0,
            "visitor_accessibility": 0.0,
            "educational_impact": 0.0,
            "sustainability": 0.0,
            "cost_effectiveness": 0.0
        }
        
        # Simulate preservation interventions
        intervention_quality = random_factors.get("intervention_quality", 0.7)
        material_durability = random_factors.get("material_durability", 0.8)
        environmental_protection = random_factors.get("environmental_protection", 0.6)
        
        # Calculate preservation effectiveness
        outcome["preservation_effectiveness"] = (
            0.4 * intervention_quality +
            0.3 * material_durability +
            0.3 * environmental_protection
        )
        
        # Calculate structural integrity improvement
        baseline_condition = scenario.get("baseline_condition", 0.5)
        improvement_factor = random_factors.get("structural_improvement", 0.3)
        outcome["structural_integrity"] = min(1.0, baseline_condition + improvement_factor)
        
        # Calculate visitor accessibility
        accessibility_improvements = random_factors.get("accessibility_improvements", 0.6)
        safety_measures = random_factors.get("safety_measures", 0.8)
        outcome["visitor_accessibility"] = (accessibility_improvements + safety_measures) / 2
        
        # Calculate educational impact
        interpretation_quality = random_factors.get("interpretation_quality", 0.7)
        community_programs = random_factors.get("community_programs", 0.6)
        outcome["educational_impact"] = (interpretation_quality + community_programs) / 2
        
        # Calculate sustainability
        maintenance_plan = random_factors.get("maintenance_plan", 0.7)
        funding_security = random_factors.get("funding_security", 0.6)
        outcome["sustainability"] = (maintenance_plan + funding_security) / 2
        
        # Calculate cost effectiveness
        budget_efficiency = random_factors.get("budget_efficiency", 0.8)
        long_term_value = random_factors.get("long_term_value", 0.7)
        outcome["cost_effectiveness"] = (budget_efficiency + long_term_value) / 2
        
        # Determine overall success
        overall_score = (
            0.25 * outcome["preservation_effectiveness"] +
            0.2 * outcome["structural_integrity"] +
            0.15 * outcome["visitor_accessibility"] +
            0.15 * outcome["educational_impact"] +
            0.15 * outcome["sustainability"] +
            0.1 * outcome["cost_effectiveness"]
        )
        
        outcome["success"] = overall_score >= 0.65
        outcome["overall_score"] = overall_score
        
        return outcome
    
    def _simulate_generic_scenario(
        self,
        scenario: Dict[str, Any],
        parameters: SimulationParameters,
        random_factors: Dict[str, float]
    ) -> Dict[str, Any]:
        """Simulate a generic scenario."""
        outcome = {
            "success": False,
            "efficiency": random_factors.get("efficiency", 0.7),
            "quality": random_factors.get("quality", 0.8),
            "timeliness": random_factors.get("timeliness", 0.75),
            "resource_usage": random_factors.get("resource_usage", 0.6)
        }
        
        overall_score = (
            0.3 * outcome["efficiency"] +
            0.3 * outcome["quality"] +
            0.2 * outcome["timeliness"] +
            0.2 * (1.0 - outcome["resource_usage"])  # Lower resource usage is better
        )
        
        outcome["success"] = overall_score >= 0.6
        outcome["overall_score"] = overall_score
        
        return outcome
    
    def _generate_random_factors(self, parameters: SimulationParameters) -> Dict[str, float]:
        """Generate random factors for simulation iteration."""
        factors = {}
        
        # Environmental factors
        for factor, base_value in parameters.environmental_factors.items():
            uncertainty = parameters.uncertainty_factors.get(factor, 0.2)
            factors[factor] = max(0.0, min(1.0, random.gauss(base_value, uncertainty * 0.3)))
        
        # Resource factors
        for factor, base_value in parameters.resource_constraints.items():
            uncertainty = parameters.uncertainty_factors.get(factor, 0.15)
            factors[factor] = max(0.0, min(1.0, random.gauss(base_value, uncertainty * 0.3)))
        
        # Domain-specific factors
        factors.update({
            "weather": random.gauss(0.7, 0.2),
            "soil_composition": random.gauss(0.6, 0.15),
            "team_efficiency": random.gauss(0.8, 0.1),
            "humidity": random.gauss(0.6, 0.15),
            "temperature_stability": random.gauss(0.7, 0.1),
            "handling_care": random.gauss(0.8, 0.1),
            "local_involvement": random.gauss(0.7, 0.2),
            "cultural_respect": random.gauss(0.9, 0.05),
            "communication_quality": random.gauss(0.8, 0.1),
            "unexpected_costs": abs(random.gauss(0.1, 0.05)),
            "weather_delays": abs(random.gauss(0.05, 0.03)),
            "permit_delays": abs(random.gauss(0.02, 0.02)),
            "indigenous_consultation": random.gauss(0.9, 0.05),
            "sacred_site_respect": random.gauss(0.95, 0.03),
            "artifact_repatriation": random.gauss(0.8, 0.1)
        })
        
        # Clamp all values to [0, 1] range
        for key, value in factors.items():
            factors[key] = max(0.0, min(1.0, value))
        
        return factors
    
    def _check_convergence(self, recent_outcomes: List[Dict[str, Any]]) -> bool:
        """Check if simulation has converged."""
        if len(recent_outcomes) < 50:
            return False
        
        # Check convergence of success rate
        success_rates = []
        window_size = 10
        
        for i in range(len(recent_outcomes) - window_size + 1):
            window = recent_outcomes[i:i + window_size]
            success_rate = sum(1 for outcome in window if outcome.get("success", False)) / window_size
            success_rates.append(success_rate)
        
        if len(success_rates) < 5:
            return False
        
        # Check if recent success rates are stable
        recent_variance = sum((rate - success_rates[-1]) ** 2 for rate in success_rates[-5:]) / 5
        return recent_variance < 0.01  # Convergence threshold
    
    def _analyze_simulation_results(
        self,
        outcomes: List[Dict[str, Any]],
        scenario: Dict[str, Any],
        parameters: SimulationParameters
    ) -> Dict[str, Any]:
        """Analyze simulation results and generate insights."""
        if not outcomes:
            return {"success_probability": 0.0, "expected_outcomes": [], "risk_factors": []}
        
        # Calculate success probability
        success_count = sum(1 for outcome in outcomes if outcome.get("success", False))
        success_probability = success_count / len(outcomes)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(outcomes, parameters.confidence_level)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(outcomes, scenario)
        
        # Calculate resource utilization
        resource_utilization = self._calculate_resource_utilization(outcomes)
        
        # Estimate timeline
        timeline = self._estimate_timeline(outcomes, parameters)
        
        # Generate expected outcomes
        expected_outcomes = self._generate_expected_outcomes(outcomes)
        
        return {
            "success_probability": success_probability,
            "expected_outcomes": expected_outcomes,
            "risk_factors": risk_factors,
            "resource_utilization": resource_utilization,
            "timeline": timeline,
            "confidence_intervals": confidence_intervals,
            "convergence": len(outcomes) < parameters.iterations  # True if converged early
        }
    
    def _calculate_confidence_intervals(
        self,
        outcomes: List[Dict[str, Any]],
        confidence_level: float
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for key metrics."""
        intervals = {}
        
        # Success probability confidence interval
        success_count = sum(1 for outcome in outcomes if outcome.get("success", False))
        n = len(outcomes)
        p = success_count / n
        
        # Wilson score interval for binomial proportion
        z = 1.96 if confidence_level >= 0.95 else 1.645  # Simplified
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
        
        intervals["success_probability"] = (
            max(0.0, center - margin),
            min(1.0, center + margin)
        )
        
        # Other metric intervals (simplified normal approximation)
        for metric in ["overall_score", "preservation_quality", "community_engagement"]:
            values = [outcome.get(metric, 0.0) for outcome in outcomes if metric in outcome]
            if values:
                mean_val = sum(values) / len(values)
                variance = sum((v - mean_val)**2 for v in values) / len(values)
                std_error = math.sqrt(variance / len(values))
                margin = z * std_error
                
                intervals[metric] = (
                    max(0.0, mean_val - margin),
                    min(1.0, mean_val + margin)
                )
        
        return intervals
    
    def _identify_risk_factors(
        self,
        outcomes: List[Dict[str, Any]],
        scenario: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Identify key risk factors from simulation results."""
        risk_factors = []
        
        # Analyze failure patterns
        failed_outcomes = [outcome for outcome in outcomes if not outcome.get("success", False)]
        
        if failed_outcomes:
            # Budget overrun risk
            budget_overruns = [o for o in failed_outcomes if o.get("budget_utilization", 0) > 1.1]
            if len(budget_overruns) > len(failed_outcomes) * 0.3:
                risk_factors.append({
                    "type": "financial",
                    "description": "High probability of budget overruns",
                    "severity": "high" if len(budget_overruns) > len(failed_outcomes) * 0.5 else "medium"
                })
            
            # Timeline risk
            timeline_issues = [o for o in failed_outcomes if o.get("timeline_adherence", 1) < 0.8]
            if len(timeline_issues) > len(failed_outcomes) * 0.3:
                risk_factors.append({
                    "type": "temporal",
                    "description": "Significant timeline delays likely",
                    "severity": "high" if len(timeline_issues) > len(failed_outcomes) * 0.5 else "medium"
                })
            
            # Quality risk
            quality_issues = [o for o in failed_outcomes if o.get("preservation_quality", 1) < 0.6]
            if len(quality_issues) > len(failed_outcomes) * 0.3:
                risk_factors.append({
                    "type": "quality",
                    "description": "Preservation quality may be compromised",
                    "severity": "high" if len(quality_issues) > len(failed_outcomes) * 0.5 else "medium"
                })
            
            # Community engagement risk
            community_issues = [o for o in failed_outcomes if o.get("community_engagement", 1) < 0.7]
            if len(community_issues) > len(failed_outcomes) * 0.3:
                risk_factors.append({
                    "type": "social",
                    "description": "Poor community engagement may cause issues",
                    "severity": "medium"
                })
        
        return risk_factors
    
    def _calculate_resource_utilization(self, outcomes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate expected resource utilization."""
        if not outcomes:
            return {}
        
        utilization = {}
        
        # Budget utilization
        budget_values = [o.get("budget_utilization", 0.8) for o in outcomes]
        utilization["budget"] = sum(budget_values) / len(budget_values)
        
        # Time utilization (inverse of timeline adherence)
        timeline_values = [o.get("timeline_adherence", 0.9) for o in outcomes]
        utilization["time"] = 2.0 - (sum(timeline_values) / len(timeline_values))  # Convert adherence to utilization
        
        # Human resources (estimated based on efficiency)
        efficiency_values = [o.get("team_efficiency", 0.8) for o in outcomes if "team_efficiency" in o]
        if efficiency_values:
            utilization["human_resources"] = 1.0 / (sum(efficiency_values) / len(efficiency_values))
        
        return utilization
    
    def _estimate_timeline(
        self,
        outcomes: List[Dict[str, Any]],
        parameters: SimulationParameters
    ) -> Dict[str, float]:
        """Estimate project timeline based on simulation results."""
        timeline_adherence_values = [o.get("timeline_adherence", 0.9) for o in outcomes]
        avg_adherence = sum(timeline_adherence_values) / len(timeline_adherence_values)
        
        # Calculate expected duration
        base_duration = parameters.time_horizon
        expected_duration = base_duration / avg_adherence
        
        return {
            "base_duration": base_duration,
            "expected_duration": expected_duration,
            "delay_probability": 1.0 - avg_adherence,
            "timeline_risk": "high" if avg_adherence < 0.7 else "medium" if avg_adherence < 0.85 else "low"
        }
    
    def _generate_expected_outcomes(self, outcomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate expected outcomes summary."""
        if not outcomes:
            return []
        
        expected = []
        
        # Success scenario
        successful_outcomes = [o for o in outcomes if o.get("success", False)]
        if successful_outcomes:
            avg_success = {}
            for key in successful_outcomes[0].keys():
                if isinstance(successful_outcomes[0][key], (int, float)):
                    values = [o.get(key, 0) for o in successful_outcomes]
                    avg_success[key] = sum(values) / len(values)
            
            expected.append({
                "scenario": "success",
                "probability": len(successful_outcomes) / len(outcomes),
                "characteristics": avg_success
            })
        
        # Failure scenario
        failed_outcomes = [o for o in outcomes if not o.get("success", False)]
        if failed_outcomes:
            avg_failure = {}
            for key in failed_outcomes[0].keys():
                if isinstance(failed_outcomes[0][key], (int, float)):
                    values = [o.get(key, 0) for o in failed_outcomes]
                    avg_failure[key] = sum(values) / len(values)
            
            expected.append({
                "scenario": "failure",
                "probability": len(failed_outcomes) / len(outcomes),
                "characteristics": avg_failure
            })
        
        return expected
    
    def _generate_recommendations(
        self,
        analysis: Dict[str, Any],
        scenario: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on simulation analysis."""
        recommendations = []
        
        success_prob = analysis.get("success_probability", 0.5)
        risk_factors = analysis.get("risk_factors", [])
        resource_util = analysis.get("resource_utilization", {})
        
        # Success probability recommendations
        if success_prob < 0.6:
            recommendations.append("Consider revising project scope or approach - success probability is low")
        elif success_prob < 0.8:
            recommendations.append("Implement additional risk mitigation strategies to improve success probability")
        
        # Risk-specific recommendations
        for risk in risk_factors:
            if risk["type"] == "financial":
                recommendations.append("Increase budget contingency and implement stricter cost controls")
            elif risk["type"] == "temporal":
                recommendations.append("Build additional buffer time and identify critical path dependencies")
            elif risk["type"] == "quality":
                recommendations.append("Enhance quality assurance processes and preservation protocols")
            elif risk["type"] == "social":
                recommendations.append("Invest more in community engagement and stakeholder communication")
        
        # Resource utilization recommendations
        if resource_util.get("budget", 0.8) > 1.0:
            recommendations.append("Budget optimization needed - consider cost reduction strategies")
        
        if resource_util.get("time", 1.0) > 1.2:
            recommendations.append("Timeline optimization needed - identify opportunities to accelerate delivery")
        
        # Domain-specific recommendations
        scenario_type = scenario.get("type", "")
        if scenario_type == "archaeological_excavation":
            recommendations.extend([
                "Ensure adequate weather contingency planning",
                "Prioritize community and indigenous stakeholder engagement",
                "Implement robust artifact preservation protocols"
            ])
        elif scenario_type == "heritage_preservation":
            recommendations.extend([
                "Focus on long-term sustainability planning",
                "Ensure adequate funding security for maintenance",
                "Prioritize visitor safety and accessibility"
            ])
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    # Scenario variation methods
    def _create_resource_variations(self, base_scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create resource-based variations of the scenario."""
        variations = []
        base_id = base_scenario.get("id", "base")
        
        # Budget variations
        for budget_factor in [0.7, 0.85, 1.15, 1.3]:
            variation = base_scenario.copy()
            variation["id"] = f"{base_id}_budget_{budget_factor}"
            variation["budget"] = base_scenario.get("budget", 100000) * budget_factor
            variation["variation_type"] = "resource_budget"
            variations.append(variation)
        
        # Personnel variations
        for personnel_factor in [0.8, 1.2]:
            variation = base_scenario.copy()
            variation["id"] = f"{base_id}_personnel_{personnel_factor}"
            variation["personnel_availability"] = personnel_factor
            variation["variation_type"] = "resource_personnel"
            variations.append(variation)
        
        return variations
    
    def _create_environmental_variations(self, base_scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create environment-based variations of the scenario."""
        variations = []
        base_id = base_scenario.get("id", "base")
        
        # Weather variations
        weather_conditions = ["favorable", "challenging", "severe"]
        for condition in weather_conditions:
            variation = base_scenario.copy()
            variation["id"] = f"{base_id}_weather_{condition}"
            variation["weather_condition"] = condition
            variation["variation_type"] = "environmental_weather"
            variations.append(variation)
        
        return variations
    
    def _create_temporal_variations(self, base_scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create time-based variations of the scenario."""
        variations = []
        base_id = base_scenario.get("id", "base")
        
        # Timeline variations
        for timeline_factor in [0.8, 1.2, 1.5]:
            variation = base_scenario.copy()
            variation["id"] = f"{base_id}_timeline_{timeline_factor}"
            variation["timeline_pressure"] = 2.0 - timeline_factor  # Inverse relationship
            variation["variation_type"] = "temporal_timeline"
            variations.append(variation)
        
        return variations
    
    def _create_risk_variations(self, base_scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create risk-based variations of the scenario."""
        variations = []
        base_id = base_scenario.get("id", "base")
        
        # Risk level variations
        risk_levels = ["low", "medium", "high"]
        for risk_level in risk_levels:
            variation = base_scenario.copy()
            variation["id"] = f"{base_id}_risk_{risk_level}"
            variation["risk_level"] = risk_level
            variation["variation_type"] = "risk_level"
            variations.append(variation)
        
        return variations
    
    # Comparison utility methods
    def _calculate_resource_efficiency(self, result: SimulationResult) -> float:
        """Calculate resource efficiency score for comparison."""
        resource_util = result.resource_utilization
        
        # Lower utilization is better for efficiency
        budget_efficiency = 2.0 - resource_util.get("budget", 1.0)
        time_efficiency = 2.0 - resource_util.get("time", 1.0)
        
        return (budget_efficiency + time_efficiency) / 2
    
    def _calculate_overall_risk(self, result: SimulationResult) -> float:
        """Calculate overall risk score for comparison."""
        risk_score = 0.0
        
        for risk in result.risk_factors:
            if risk.get("severity") == "high":
                risk_score += 0.3
            elif risk.get("severity") == "medium":
                risk_score += 0.2
            else:
                risk_score += 0.1
        
        return min(1.0, risk_score)
    
    def _identify_trade_offs(self, results: List[SimulationResult]) -> List[Dict[str, Any]]:
        """Identify trade-offs between scenarios."""
        trade_offs = []
        
        # Success vs. Resource efficiency trade-off
        high_success = [r for r in results if r.success_probability > 0.8]
        high_efficiency = [r for r in results if self._calculate_resource_efficiency(r) > 0.8]
        
        if high_success and high_efficiency:
            overlap = set(r.scenario_id for r in high_success) & set(r.scenario_id for r in high_efficiency)
            if len(overlap) < min(len(high_success), len(high_efficiency)):
                trade_offs.append({
                    "type": "success_vs_efficiency",
                    "description": "Higher success probability may require more resources"
                })
        
        return trade_offs
    
    def _generate_comparison_recommendations(
        self,
        results: List[SimulationResult],
        comparison: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on scenario comparison."""
        recommendations = []
        
        # Find best overall scenario
        best_overall = max(results, key=lambda r: r.success_probability * self._calculate_resource_efficiency(r))
        recommendations.append(f"Recommend scenario {best_overall.scenario_id} for best overall balance")
        
        # Risk-based recommendations
        lowest_risk = min(results, key=lambda r: self._calculate_overall_risk(r))
        if lowest_risk.scenario_id != best_overall.scenario_id:
            recommendations.append(f"Consider scenario {lowest_risk.scenario_id} if risk minimization is priority")
        
        return recommendations
    
    def get_simulation_statistics(self) -> Dict[str, Any]:
        """Get statistics about simulation history."""
        if not self.simulation_history:
            return {"total_simulations": 0}
        
        return {
            "total_simulations": len(self.simulation_history),
            "average_success_probability": sum(r.success_probability for r in self.simulation_history) / len(self.simulation_history),
            "scenario_types": list(set(r.scenario_type.value for r in self.simulation_history)),
            "most_recent_simulation": self.simulation_history[-1].scenario_id,
            "active_simulations": len(self.active_simulations)
        } 
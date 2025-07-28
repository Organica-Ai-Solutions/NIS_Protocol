#!/usr/bin/env python3
"""
NIS Protocol v2.0 - Week 3-4 Simulation & Prediction Components Test

This script tests the comprehensive simulation and prediction capabilities
implemented for NIS Protocol v2.0 AGI Evolution.

Components tested:
- ScenarioSimulator: Physics-based scenario modeling with Monte Carlo simulation
- OutcomePredictor: Neural network-based outcome prediction with uncertainty quantification
- RiskAssessor: Multi-factor risk analysis with mitigation strategies

Domain focus: Archaeological excavation and heritage preservation
"""

import sys
import os
import time
import logging
from typing import Dict, Any, List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_scenario_simulator():
    """Test the ScenarioSimulator with archaeological scenarios."""
    print("\n" + "="*60)
    print("🎮 TESTING SCENARIO SIMULATOR")
    print("="*60)
    
    try:
        from agents.simulation.scenario_simulator import ScenarioSimulator, ScenarioType, SimulationParameters
        
        # Initialize simulator
        simulator = ScenarioSimulator()
        print("✅ ScenarioSimulator initialized successfully")
        
        # Test 1: Archaeological Excavation Scenario
        print("\n📋 Test 1: Archaeological Excavation Simulation")
        excavation_scenario = {
            "id": "maya_site_excavation",
            "type": "archaeological_excavation",
            "name": "Maya Temple Complex Excavation",
            "description": "Excavation of newly discovered Maya temple complex",
            "budget": 150000,
            "target_artifacts": 25,
            "expected_discovery_rate": 0.4,
            "site_location": "Guatemala",
            "duration_days": 120
        }
        
        # Custom simulation parameters
        sim_params = SimulationParameters(
            time_horizon=120.0,  # 120 days
            resolution=1.0,      # Daily resolution
            iterations=500,      # Monte Carlo iterations
            confidence_level=0.95,
            environmental_factors={"weather": 0.6, "seasonal": 0.7, "geological": 0.8},
            resource_constraints={"budget": 0.8, "personnel": 0.75, "equipment": 0.9},
            uncertainty_factors={"discovery": 0.4, "preservation": 0.3, "community": 0.2}
        )
        
        # Run simulation
        start_time = time.time()
        result = simulator.simulate_scenario(excavation_scenario, sim_params)
        simulation_time = time.time() - start_time
        
        print(f"   Scenario ID: {result.scenario_id}")
        print(f"   Scenario Type: {result.scenario_type.value}")
        print(f"   Success Probability: {result.success_probability:.3f} ({result.success_probability*100:.1f}%)")
        print(f"   Simulation Time: {simulation_time:.2f} seconds")
        print(f"   Monte Carlo Iterations: {result.metadata['iterations_completed']}")
        print(f"   Convergence Achieved: {result.metadata['convergence_achieved']}")
        
        # Display expected outcomes
        print(f"\n   📊 Expected Outcomes:")
        for outcome in result.expected_outcomes:
            print(f"      {outcome['scenario'].title()}: {outcome['probability']:.3f} probability")
            if 'characteristics' in outcome:
                key_metrics = ['artifacts_discovered', 'preservation_quality', 'community_engagement']
                for metric in key_metrics:
                    if metric in outcome['characteristics']:
                        print(f"         {metric.replace('_', ' ').title()}: {outcome['characteristics'][metric]:.3f}")
        
        # Display risk factors
        print(f"\n   ⚠️  Risk Factors ({len(result.risk_factors)} identified):")
        for risk in result.risk_factors[:3]:  # Show top 3
            print(f"      {risk['type'].title()}: {risk['description']}")
        
        # Display resource utilization
        print(f"\n   💰 Resource Utilization:")
        for resource, utilization in result.resource_utilization.items():
            print(f"      {resource.replace('_', ' ').title()}: {utilization:.3f}")
        
        # Display timeline
        print(f"\n   ⏱️  Timeline Analysis:")
        for metric, value in result.timeline.items():
            if isinstance(value, (int, float)):
                print(f"      {metric.replace('_', ' ').title()}: {value:.1f}")
            else:
                print(f"      {metric.replace('_', ' ').title()}: {value}")
        
        # Display top recommendations
        print(f"\n   💡 Top Recommendations:")
        for i, rec in enumerate(result.recommendations[:3], 1):
            print(f"      {i}. {rec}")
        
        print("✅ Archaeological excavation simulation completed successfully")
        
        # Test 2: Scenario Variations
        print("\n📋 Test 2: Scenario Variations Generation")
        variations = simulator.create_scenario_variations(
            excavation_scenario,
            variation_types=["resource", "environmental", "temporal"]
        )
        
        print(f"   Generated {len(variations)} scenario variations:")
        variation_types = {}
        for var in variations:
            var_type = var.get("variation_type", "unknown")
            variation_types[var_type] = variation_types.get(var_type, 0) + 1
        
        for var_type, count in variation_types.items():
            print(f"      {var_type.replace('_', ' ').title()}: {count} variations")
        
        print("✅ Scenario variations generated successfully")
        
        # Test 3: Heritage Preservation Scenario
        print("\n📋 Test 3: Heritage Preservation Simulation")
        heritage_scenario = {
            "id": "angkor_wat_preservation",
            "type": "heritage_preservation",
            "name": "Angkor Wat Temple Preservation",
            "description": "Comprehensive preservation of Angkor Wat temple complex",
            "budget": 500000,
            "baseline_condition": 0.6,
            "duration_days": 365,
            "visitor_impact": 0.4
        }
        
        heritage_result = simulator.simulate_scenario(heritage_scenario)
        print(f"   Heritage Preservation Success Probability: {heritage_result.success_probability:.3f}")
        print(f"   Risk Factors Identified: {len(heritage_result.risk_factors)}")
        print(f"   Recommendations Generated: {len(heritage_result.recommendations)}")
        
        print("✅ Heritage preservation simulation completed successfully")
        
        # Test 4: Scenario Comparison
        print("\n📋 Test 4: Multi-Scenario Comparison")
        scenarios_to_compare = [excavation_scenario, heritage_scenario]
        comparison = simulator.compare_scenarios(scenarios_to_compare)
        
        print(f"   Scenarios Compared: {comparison['scenarios_compared']}")
        print(f"   Comparison Criteria: {', '.join(comparison['criteria'])}")
        
        print("   📊 Rankings by Success Probability:")
        if 'success_probability' in comparison['rankings']:
            for i, scenario_id in enumerate(comparison['rankings']['success_probability'], 1):
                print(f"      {i}. {scenario_id}")
        
        print("✅ Scenario comparison completed successfully")
        
        # Display simulator statistics
        stats = simulator.get_simulation_statistics()
        print(f"\n📈 Simulator Statistics:")
        print(f"   Total Simulations: {stats['total_simulations']}")
        print(f"   Average Success Probability: {stats['average_success_probability']:.3f}")
        print(f"   Scenario Types: {', '.join(stats['scenario_types'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ ScenarioSimulator test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_outcome_predictor():
    """Test the OutcomePredictor with various prediction types."""
    print("\n" + "="*60)
    print("🔮 TESTING OUTCOME PREDICTOR")
    print("="*60)
    
    try:
        from agents.simulation.outcome_predictor import OutcomePredictor, PredictionType
        
        # Initialize predictor
        predictor = OutcomePredictor()
        print("✅ OutcomePredictor initialized successfully")
        
        # Test 1: Archaeological Action Prediction
        print("\n📋 Test 1: Archaeological Action Outcome Prediction")
        archaeological_action = {
            "id": "site_survey_action",
            "type": "archaeological_survey",
            "description": "Comprehensive site survey using ground-penetrating radar",
            "scope": 0.7,
            "budget": 75000,
            "steps": ["site_preparation", "equipment_setup", "data_collection", "analysis", "reporting"],
            "dependencies": ["weather_conditions", "equipment_availability"],
            "stakeholders": ["archaeologists", "local_community", "government", "university"],
            "technical_difficulty": 0.6
        }
        
        archaeological_context = {
            "domain": "archaeological",
            "resource_availability": 0.8,
            "environmental_conditions": 0.7,
            "stakeholder_support": 0.85,
            "regulatory_status": 0.9,
            "site_accessibility": 0.75,
            "artifact_potential": 0.6,
            "preservation_urgency": 0.4,
            "community_involvement": 0.8,
            "weather_dependency": 0.5,
            "team_experience": 0.8,
            "funding_security": 0.9
        }
        
        # Predict multiple outcome types
        prediction_types = [
            PredictionType.SUCCESS_PROBABILITY,
            PredictionType.RESOURCE_CONSUMPTION,
            PredictionType.TIMELINE_ESTIMATION,
            PredictionType.QUALITY_ASSESSMENT,
            PredictionType.RISK_EVALUATION
        ]
        
        predictions = predictor.predict_outcomes(
            archaeological_action,
            archaeological_context,
            prediction_types
        )
        
        print(f"   Generated {len(predictions)} predictions:")
        for pred in predictions:
            print(f"\n   🎯 {pred.prediction_type.value.replace('_', ' ').title()}:")
            print(f"      Predicted Value: {pred.predicted_value:.3f}")
            print(f"      Confidence Interval: ({pred.confidence_interval[0]:.3f}, {pred.confidence_interval[1]:.3f})")
            print(f"      Uncertainty Score: {pred.uncertainty_score:.3f}")
            
            # Show top contributing factors
            print(f"      Top Contributing Factors:")
            for factor, contribution in list(pred.contributing_factors.items())[:3]:
                print(f"         {factor.replace('_', ' ').title()}: {contribution:+.3f}")
            
            # Show recommendations
            if pred.recommendations:
                print(f"      Recommendations:")
                for rec in pred.recommendations[:2]:
                    print(f"         • {rec}")
        
        print("✅ Archaeological action predictions completed successfully")
        
        # Test 2: Heritage Preservation Action
        print("\n📋 Test 2: Heritage Preservation Action Prediction")
        heritage_action = {
            "id": "temple_restoration",
            "type": "heritage_restoration",
            "description": "Structural restoration of ancient temple",
            "scope": 0.9,
            "budget": 200000,
            "technical_difficulty": 0.8,
            "regulatory_complexity": 0.6
        }
        
        heritage_context = {
            "domain": "heritage_preservation",
            "structural_condition": 0.5,
            "visitor_impact": 0.6,
            "funding_security": 0.7,
            "maintenance_capacity": 0.6,
            "resource_availability": 0.7,
            "stakeholder_support": 0.8,
            "team_experience": 0.9
        }
        
        # Test specific prediction methods
        success_pred = predictor.predict_success_probability(heritage_action, heritage_context)
        resource_pred = predictor.predict_resource_consumption(heritage_action, heritage_context)
        
        print(f"   Success Probability: {success_pred.predicted_value:.3f} ± {success_pred.uncertainty_score:.3f}")
        print(f"   Resource Consumption: {resource_pred.predicted_value:.3f} ± {resource_pred.uncertainty_score:.3f}")
        
        print("✅ Heritage preservation predictions completed successfully")
        
        # Test 3: Outcome Quality Evaluation
        print("\n📋 Test 3: Outcome Quality Evaluation")
        predicted_outcomes = [
            {
                "id": "outcome_1",
                "success_probability": 0.8,
                "resource_usage": 0.7,
                "timeline_adherence": 0.9,
                "quality": 0.85,
                "risk_level": 0.2
            },
            {
                "id": "outcome_2",
                "success_probability": 0.6,
                "resource_usage": 0.5,
                "timeline_adherence": 0.7,
                "quality": 0.7,
                "risk_level": 0.4
            }
        ]
        
        quality_evaluation = predictor.evaluate_outcome_quality(predicted_outcomes)
        
        print(f"   Overall Quality Score: {quality_evaluation.get('overall_quality', 0):.3f}")
        print(f"   Best Outcome: {quality_evaluation.get('best_outcome', 'N/A')}")
        print(f"   Quality Variance: {quality_evaluation.get('quality_variance', 0):.3f}")
        
        for outcome_id, score in quality_evaluation.items():
            if outcome_id not in ['overall_quality', 'best_outcome', 'quality_variance']:
                print(f"   {outcome_id}: {score:.3f}")
        
        print("✅ Outcome quality evaluation completed successfully")
        
        # Test 4: Action Comparison
        print("\n📋 Test 4: Multi-Action Comparison")
        actions_to_compare = [archaeological_action, heritage_action]
        comparison = predictor.compare_action_outcomes(actions_to_compare, archaeological_context)
        
        print(f"   Actions Compared: {comparison['actions_compared']}")
        print(f"   Ranking Criteria: {', '.join(comparison['rankings'].keys())}")
        
        for criterion, ranking in comparison['rankings'].items():
            print(f"   {criterion.replace('_', ' ').title()} Ranking:")
            for i, action_id in enumerate(ranking, 1):
                print(f"      {i}. {action_id}")
        
        if comparison['recommendations']:
            print(f"   Comparison Recommendations:")
            for rec in comparison['recommendations']:
                print(f"      • {rec}")
        
        print("✅ Action comparison completed successfully")
        
        # Display predictor statistics
        stats = predictor.get_prediction_statistics()
        print(f"\n📈 Predictor Statistics:")
        print(f"   Total Predictions: {stats['total_predictions']}")
        print(f"   Prediction Types: {', '.join(stats['prediction_types'])}")
        print(f"   Average Uncertainty: {stats['average_uncertainty']:.3f}")
        print(f"   Cache Size: {stats['cache_size']}")
        
        return True
        
    except Exception as e:
        print(f"❌ OutcomePredictor test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_risk_assessor():
    """Test the RiskAssessor with comprehensive risk analysis."""
    print("\n" + "="*60)
    print("⚠️  TESTING RISK ASSESSOR")
    print("="*60)
    
    try:
        from agents.simulation.risk_assessor import RiskAssessor, RiskCategory, RiskSeverity
        
        # Initialize assessor
        assessor = RiskAssessor()
        print("✅ RiskAssessor initialized successfully")
        
        # Test 1: Comprehensive Risk Assessment
        print("\n📋 Test 1: Comprehensive Archaeological Risk Assessment")
        archaeological_action = {
            "id": "maya_excavation_project",
            "type": "archaeological_excavation",
            "description": "Large-scale Maya site excavation",
            "budget": 300000,
            "steps": ["site_prep", "excavation", "analysis", "preservation", "documentation"],
            "dependencies": ["permits", "weather", "community_approval"],
            "stakeholders": ["archaeologists", "local_community", "government", "university", "tourists"],
            "technical_difficulty": 0.7,
            "regulatory_complexity": 0.6
        }
        
        risk_context = {
            "domain": "archaeological",
            "resource_availability": 0.7,
            "environmental_conditions": 0.6,
            "stakeholder_support": 0.8,
            "regulatory_compliance": 0.85,
            "community_support": 0.75,
            "funding_security": 0.8,
            "weather_dependency": 0.7,
            "media_attention": 0.6,
            "site_accessibility": 0.8,
            "team_experience": 0.8
        }
        
        # Assess all risk categories
        risk_categories = [
            RiskCategory.FINANCIAL,
            RiskCategory.OPERATIONAL,
            RiskCategory.TECHNICAL,
            RiskCategory.ENVIRONMENTAL,
            RiskCategory.REGULATORY,
            RiskCategory.SOCIAL,
            RiskCategory.CULTURAL,
            RiskCategory.TEMPORAL,
            RiskCategory.SAFETY,
            RiskCategory.REPUTATIONAL
        ]
        
        assessment = assessor.assess_risks(
            archaeological_action,
            risk_context,
            risk_categories
        )
        
        print(f"   Assessment ID: {assessment.assessment_id}")
        print(f"   Overall Risk Score: {assessment.overall_risk_score:.3f}")
        print(f"   Risk Level: {assessment.risk_level.value.upper()}")
        print(f"   Confidence Score: {assessment.confidence_score:.3f}")
        print(f"   Total Risk Factors: {len(assessment.risk_factors)}")
        
        # Display risk factors by category
        print(f"\n   🎯 Risk Factors by Category:")
        risk_by_category = {}
        for factor in assessment.risk_factors:
            category = factor.category.value
            if category not in risk_by_category:
                risk_by_category[category] = []
            risk_by_category[category].append(factor)
        
        for category, factors in risk_by_category.items():
            print(f"      {category.title()}: {len(factors)} factors")
            for factor in factors[:2]:  # Show top 2 per category
                print(f"         • {factor.description}")
                print(f"           Probability: {factor.probability:.3f}, Impact: {factor.impact:.3f}")
                print(f"           Severity: {factor.severity.value}")
        
        # Display mitigation plan
        print(f"\n   🛡️  Mitigation Plan:")
        mitigation = assessment.mitigation_plan
        if mitigation.get('immediate_actions'):
            print(f"      Immediate Actions ({len(mitigation['immediate_actions'])}):")
            for action in mitigation['immediate_actions'][:2]:
                print(f"         • {action['strategy']}")
        
        if mitigation.get('short_term_strategies'):
            print(f"      Short-term Strategies ({len(mitigation['short_term_strategies'])}):")
            for strategy in mitigation['short_term_strategies'][:2]:
                print(f"         • {strategy['strategy']}")
        
        # Display monitoring requirements
        print(f"\n   📊 Monitoring Requirements ({len(assessment.monitoring_requirements)}):")
        for req in assessment.monitoring_requirements[:3]:
            print(f"      • {req}")
        
        # Display contingency plans
        print(f"\n   🚨 Contingency Plans ({len(assessment.contingency_plans)}):")
        for plan in assessment.contingency_plans[:2]:
            print(f"      Risk: {plan['risk_factor']}")
            print(f"      Trigger: {', '.join(plan['trigger_conditions'])}")
            print(f"      Response: {plan['response_actions'][0] if plan['response_actions'] else 'TBD'}")
        
        # Display recommendations
        print(f"\n   💡 Risk Management Recommendations:")
        for i, rec in enumerate(assessment.recommendations[:3], 1):
            print(f"      {i}. {rec}")
        
        print("✅ Comprehensive risk assessment completed successfully")
        
        # Test 2: Risk Score Calculation
        print("\n📋 Test 2: Risk Score Calculation")
        risk_data = {
            "financial": {"probability": 0.3, "impact": 0.7},
            "operational": {"probability": 0.4, "impact": 0.5},
            "technical": {"probability": 0.2, "impact": 0.6},
            "regulatory": {"probability": 0.1, "impact": 0.9}
        }
        
        risk_score = assessor.calculate_risk_score(risk_data)
        print(f"   Calculated Risk Score: {risk_score:.3f}")
        
        # Test with custom weighting
        custom_weights = {
            "financial": 0.3,
            "operational": 0.2,
            "technical": 0.2,
            "regulatory": 0.3
        }
        
        weighted_score = assessor.calculate_risk_score(risk_data, custom_weights)
        print(f"   Weighted Risk Score: {weighted_score:.3f}")
        
        print("✅ Risk score calculation completed successfully")
        
        # Test 3: Risk Monitoring
        print("\n📋 Test 3: Risk Monitoring")
        current_indicators = {
            "monthly_spend_rate": 0.8,
            "staff_availability": 0.6,
            "weather_forecasts": 0.4,
            "community_feedback": 0.9,
            "permit_status": 0.95
        }
        
        monitoring_result = assessor.monitor_risks(
            assessment.assessment_id,
            current_indicators
        )
        
        print(f"   Monitoring Status: {monitoring_result['monitoring_status']}")
        print(f"   New Alerts: {monitoring_result['new_alerts']}")
        print(f"   Total Alerts: {monitoring_result['total_alerts']}")
        print(f"   Risk Trend: {monitoring_result['risk_trend']}")
        
        if monitoring_result.get('recommendations'):
            print(f"   Monitoring Recommendations:")
            for rec in monitoring_result['recommendations']:
                print(f"      • {rec}")
        
        print("✅ Risk monitoring completed successfully")
        
        # Test 4: Heritage Preservation Risk Assessment
        print("\n📋 Test 4: Heritage Preservation Risk Assessment")
        heritage_action = {
            "id": "temple_conservation",
            "type": "heritage_conservation",
            "description": "Ancient temple conservation project",
            "budget": 400000,
            "stakeholders": ["conservators", "government", "tourists", "local_community"]
        }
        
        heritage_context = {
            "domain": "heritage_preservation",
            "structural_condition": 0.4,
            "visitor_impact": 0.7,
            "funding_security": 0.6,
            "regulatory_compliance": 0.9,
            "community_support": 0.8
        }
        
        heritage_assessment = assessor.assess_risks(heritage_action, heritage_context)
        
        print(f"   Heritage Risk Score: {heritage_assessment.overall_risk_score:.3f}")
        print(f"   Heritage Risk Level: {heritage_assessment.risk_level.value}")
        print(f"   Heritage Risk Factors: {len(heritage_assessment.risk_factors)}")
        
        print("✅ Heritage preservation risk assessment completed successfully")
        
        # Display assessor statistics
        stats = assessor.get_risk_statistics()
        print(f"\n📈 Risk Assessor Statistics:")
        print(f"   Total Assessments: {stats['total_assessments']}")
        print(f"   Average Risk Score: {stats['average_risk_score']:.3f}")
        print(f"   Risk Level Distribution:")
        for level, count in stats['risk_level_distribution'].items():
            if count > 0:
                print(f"      {level.title()}: {count}")
        
        if stats.get('most_common_risks'):
            print(f"   Most Common Risk Categories:")
            for category, count in list(stats['most_common_risks'].items())[:3]:
                print(f"      {category.title()}: {count}")
        
        return True
        
    except Exception as e:
        print(f"❌ RiskAssessor test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration between simulation components."""
    print("\n" + "="*60)
    print("🔗 TESTING COMPONENT INTEGRATION")
    print("="*60)
    
    try:
        from agents.simulation.scenario_simulator import ScenarioSimulator
        from agents.simulation.outcome_predictor import OutcomePredictor, PredictionType
        from agents.simulation.risk_assessor import RiskAssessor
        
        # Initialize all components
        simulator = ScenarioSimulator()
        predictor = OutcomePredictor()
        assessor = RiskAssessor()
        
        print("✅ All simulation components initialized")
        
        # Integrated workflow test
        print("\n📋 Integrated Workflow: Archaeological Project Planning")
        
        # Define project
        project_action = {
            "id": "integrated_archaeological_project",
            "type": "archaeological_excavation",
            "description": "Integrated archaeological project with full analysis",
            "budget": 200000,
            "scope": 0.8,
            "stakeholders": ["archaeologists", "community", "government", "university"]
        }
        
        project_context = {
            "domain": "archaeological",
            "resource_availability": 0.75,
            "environmental_conditions": 0.7,
            "stakeholder_support": 0.8,
            "regulatory_compliance": 0.9,
            "community_support": 0.85,
            "weather_dependency": 0.6
        }
        
        # Step 1: Risk Assessment
        print("   Step 1: Conducting risk assessment...")
        risk_assessment = assessor.assess_risks(project_action, project_context)
        print(f"      Risk Level: {risk_assessment.risk_level.value}")
        print(f"      Risk Score: {risk_assessment.overall_risk_score:.3f}")
        
        # Step 2: Outcome Prediction
        print("   Step 2: Predicting outcomes...")
        predictions = predictor.predict_outcomes(
            project_action,
            project_context,
            [PredictionType.SUCCESS_PROBABILITY, PredictionType.RESOURCE_CONSUMPTION]
        )
        
        success_pred = next((p for p in predictions if p.prediction_type == PredictionType.SUCCESS_PROBABILITY), None)
        resource_pred = next((p for p in predictions if p.prediction_type == PredictionType.RESOURCE_CONSUMPTION), None)
        
        if success_pred:
            print(f"      Success Probability: {success_pred.predicted_value:.3f}")
        if resource_pred:
            print(f"      Resource Consumption: {resource_pred.predicted_value:.3f}")
        
        # Step 3: Scenario Simulation
        print("   Step 3: Running scenario simulation...")
        scenario = {
            "id": "integrated_scenario",
            "type": "archaeological_excavation",
            "name": "Integrated Archaeological Project",
            "budget": project_action["budget"],
            "risk_level": risk_assessment.risk_level.value,
            "predicted_success": success_pred.predicted_value if success_pred else 0.7
        }
        
        sim_result = simulator.simulate_scenario(scenario)
        print(f"      Simulation Success Probability: {sim_result.success_probability:.3f}")
        print(f"      Simulation Risk Factors: {len(sim_result.risk_factors)}")
        
        # Step 4: Integrated Analysis
        print("   Step 4: Integrated analysis...")
        
        # Compare predictions vs simulation
        if success_pred:
            prediction_accuracy = abs(success_pred.predicted_value - sim_result.success_probability)
            print(f"      Prediction vs Simulation Accuracy: {1 - prediction_accuracy:.3f}")
        
        # Risk consistency check
        risk_consistency = abs(risk_assessment.overall_risk_score - (1 - sim_result.success_probability))
        print(f"      Risk Assessment Consistency: {1 - risk_consistency:.3f}")
        
        # Generate integrated recommendations
        integrated_recommendations = []
        
        # From risk assessment
        integrated_recommendations.extend(risk_assessment.recommendations[:2])
        
        # From predictions
        for pred in predictions:
            integrated_recommendations.extend(pred.recommendations[:1])
        
        # From simulation
        integrated_recommendations.extend(sim_result.recommendations[:2])
        
        print(f"   🎯 Integrated Recommendations ({len(integrated_recommendations)}):")
        for i, rec in enumerate(set(integrated_recommendations)[:5], 1):  # Remove duplicates, show top 5
            print(f"      {i}. {rec}")
        
        print("✅ Component integration test completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all simulation component tests."""
    print("🚀 NIS Protocol v2.0 - Week 3-4 Simulation & Prediction Components Test")
    print("=" * 80)
    print("Testing comprehensive simulation and prediction capabilities:")
    print("• ScenarioSimulator: Physics-based Monte Carlo simulation")
    print("• OutcomePredictor: Neural network-based prediction with uncertainty")
    print("• RiskAssessor: Multi-factor risk analysis with mitigation")
    print("• Domain Focus: Archaeological excavation & heritage preservation")
    print("=" * 80)
    
    # Track test results
    test_results = {}
    
    # Run individual component tests
    test_results['scenario_simulator'] = test_scenario_simulator()
    test_results['outcome_predictor'] = test_outcome_predictor()
    test_results['risk_assessor'] = test_risk_assessor()
    test_results['integration'] = test_integration()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL SIMULATION COMPONENTS WORKING PERFECTLY!")
        print("✅ Week 3-4 Simulation & Prediction implementation COMPLETE")
        print("\n🏆 NIS Protocol v2.0 AGI Capabilities:")
        print("   • Advanced scenario modeling with Monte Carlo simulation")
        print("   • Neural network-based outcome prediction")
        print("   • Comprehensive multi-factor risk assessment")
        print("   • Archaeological domain specialization")
        print("   • Integrated simulation workflow")
        print("   • Real-time risk monitoring")
        print("   • Uncertainty quantification")
        print("   • Mitigation strategy generation")
        
        print("\n🎯 Ready for Week 5-6: Alignment & Safety Implementation")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed - review implementation")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
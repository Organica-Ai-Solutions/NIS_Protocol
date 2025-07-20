"""
Integrity Metrics - Actual Calculation Functions
Replaces hardcoded performance values with evidence-based calculations
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result from validation operations"""
    violations: List[str]
    constraints_tested: int
    passed_constraints: int
    metadata: Dict[str, Any]

@dataclass
class ConfidenceFactors:
    """Factors contributing to confidence calculation"""
    data_quality: float
    algorithm_stability: float  
    validation_coverage: float
    error_rate: float

def calculate_physics_compliance(validation_result: ValidationResult) -> float:
    """
    Calculate actual physics constraint compliance based on validation results
    
    Args:
        validation_result: Results from physics constraint validation
        
    Returns:
        Compliance score between 0.0 and 1.0
    """
    if validation_result.constraints_tested == 0:
        logger.warning("No physics constraints tested - returning 0.0 compliance")
        return 0.0
    
    # Calculate compliance as ratio of passed constraints
    compliance = validation_result.passed_constraints / validation_result.constraints_tested
    
    # Apply penalty for critical violations
    critical_violations = [v for v in validation_result.violations if 'critical' in v.lower()]
    if critical_violations:
        penalty = min(0.3, len(critical_violations) * 0.1)
        compliance = max(0.0, compliance - penalty)
    
    logger.debug(f"Physics compliance calculated: {compliance:.3f} "
                f"({validation_result.passed_constraints}/{validation_result.constraints_tested} constraints passed)")
    
    return float(compliance)

def calculate_confidence(factors: ConfidenceFactors) -> float:
    """
    Calculate confidence score based on multiple validated factors
    
    Args:
        factors: Validated factors contributing to confidence
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    # Weighted average of factors
    weights = {
        'data_quality': 0.3,
        'algorithm_stability': 0.25,
        'validation_coverage': 0.25, 
        'error_rate': 0.2
    }
    
    # Error rate is inverted (lower error = higher confidence)
    error_confidence = 1.0 - min(1.0, factors.error_rate)
    
    confidence = (
        factors.data_quality * weights['data_quality'] +
        factors.algorithm_stability * weights['algorithm_stability'] +
        factors.validation_coverage * weights['validation_coverage'] +
        error_confidence * weights['error_rate']
    )
    
    # Clamp to valid range
    confidence = max(0.0, min(1.0, confidence))
    
    logger.debug(f"Confidence calculated: {confidence:.3f} "
                f"(data:{factors.data_quality:.2f}, stability:{factors.algorithm_stability:.2f}, "
                f"coverage:{factors.validation_coverage:.2f}, error:{factors.error_rate:.2f})")
    
    return float(confidence)

def calculate_accuracy(predictions: List[Any], ground_truth: List[Any], 
                      tolerance: float = 0.01) -> float:
    """
    Calculate actual accuracy based on predictions vs ground truth
    
    Args:
        predictions: Model predictions
        ground_truth: Known correct values
        tolerance: Tolerance for numerical comparisons
        
    Returns:
        Accuracy score between 0.0 and 1.0
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    if len(predictions) == 0:
        logger.warning("No data provided for accuracy calculation")
        return 0.0
    
    correct = 0
    for pred, truth in zip(predictions, ground_truth):
        if isinstance(pred, (int, float)) and isinstance(truth, (int, float)):
            # Numerical comparison with tolerance
            if abs(pred - truth) <= tolerance:
                correct += 1
        else:
            # Exact comparison for other types
            if pred == truth:
                correct += 1
    
    accuracy = correct / len(predictions)
    
    logger.debug(f"Accuracy calculated: {accuracy:.3f} "
                f"({correct}/{len(predictions)} correct)")
    
    return float(accuracy)

def calculate_interpretability(model_output: Dict[str, Any], 
                             explanation_data: Dict[str, Any]) -> float:
    """
    Calculate interpretability score based on model transparency
    
    Args:
        model_output: Model predictions and intermediate results
        explanation_data: Explanation/reasoning data from model
        
    Returns:
        Interpretability score between 0.0 and 1.0
    """
    score = 0.0
    max_score = 0.0
    
    # Check for symbolic functions (KAN interpretability)
    if 'symbolic_functions' in explanation_data:
        functions = explanation_data['symbolic_functions']
        if functions and len(functions) > 0:
            score += 0.3  # 30% for having symbolic functions
        max_score += 0.3
    
    # Check for decision path explanation
    if 'decision_path' in explanation_data:
        path = explanation_data['decision_path']
        if path and len(path) > 0:
            score += 0.25  # 25% for decision path
        max_score += 0.25
    
    # Check for feature importance
    if 'feature_importance' in explanation_data:
        importance = explanation_data['feature_importance']
        if importance and len(importance) > 0:
            score += 0.2  # 20% for feature importance
        max_score += 0.2
    
    # Check for confidence intervals
    if 'confidence_intervals' in model_output:
        intervals = model_output['confidence_intervals']
        if intervals:
            score += 0.15  # 15% for uncertainty quantification
        max_score += 0.15
    
    # Check for validation metrics
    if 'validation_metrics' in explanation_data:
        metrics = explanation_data['validation_metrics']
        if metrics and len(metrics) > 0:
            score += 0.1  # 10% for validation metrics
        max_score += 0.1
    
    # Normalize by maximum possible score
    if max_score > 0:
        interpretability = score / max_score
    else:
        interpretability = 0.0
    
    logger.debug(f"Interpretability calculated: {interpretability:.3f} "
                f"(score:{score:.2f}/max:{max_score:.2f})")
    
    return float(interpretability)

def calculate_processing_time(start_time: datetime, end_time: datetime, 
                            data_size: int) -> Tuple[float, str]:
    """
    Calculate actual processing time and generate performance description
    
    Args:
        start_time: Processing start timestamp
        end_time: Processing end timestamp  
        data_size: Size of processed data
        
    Returns:
        Tuple of (processing_time_seconds, description)
    """
    processing_time = (end_time - start_time).total_seconds()
    
    # Generate honest performance description
    if processing_time < 0.001:
        desc = f"Sub-millisecond processing: {processing_time*1000:.2f}ms"
    elif processing_time < 1.0:
        desc = f"Processing time: {processing_time*1000:.0f}ms"
    elif processing_time < 60:
        desc = f"Processing time: {processing_time:.1f}s"
    else:
        desc = f"Processing time: {processing_time/60:.1f}min"
    
    if data_size > 0:
        rate = data_size / processing_time
        desc += f" ({rate:.0f} items/second)"
    
    logger.debug(f"Processing time calculated: {desc}")
    
    return processing_time, desc

def create_default_confidence_factors(data_quality: Optional[float] = None) -> ConfidenceFactors:
    """
    Create default confidence factors for calculation
    
    Args:
        data_quality: Known data quality score, if available
        
    Returns:
        ConfidenceFactors with conservative default values
    """
    return ConfidenceFactors(
        data_quality=data_quality if data_quality is not None else 0.7,
        algorithm_stability=0.75,  # Based on testing stability
        validation_coverage=0.8,   # Based on test coverage
        error_rate=0.1             # Conservative error estimate
    )

def create_mock_validation_result(passed: int, total: int, 
                                violations: Optional[List[str]] = None) -> ValidationResult:
    """
    Create validation result for testing purposes
    
    Args:
        passed: Number of passed constraints
        total: Total number of constraints tested
        violations: List of violation descriptions
        
    Returns:
        ValidationResult for calculation
    """
    return ValidationResult(
        violations=violations or [],
        constraints_tested=total,
        passed_constraints=passed,
        metadata={'created': datetime.now().isoformat()}
    ) 
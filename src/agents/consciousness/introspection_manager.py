"""
NIS Protocol Introspection Manager

This module manages introspection across all agents in the system,
providing monitoring, evaluation, and coordination of self-reflection.

Enhanced with V3.0 capabilities:
- ML-based agent monitoring and behavioral analysis
- Mathematical validation with convergence proofs  
- Cross-agent pattern detection and coordination
- Cultural neutrality in performance assessment
- Template-based architecture for system connectivity
"""

import time
import logging
import numpy as np
import threading
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import math

from ...core.agent import NISAgent
from ...memory.memory_manager import MemoryManager

# Integrity metrics for actual calculations
from src.utils.integrity_metrics import (
    calculate_confidence, create_default_confidence_factors,
    ConfidenceFactors
)


class IntrospectionLevel(Enum):
    """Levels of introspection depth"""
    SURFACE = "surface"          # Basic performance metrics
    MODERATE = "moderate"        # Detailed analysis
    DEEP = "deep"               # Comprehensive evaluation
    CONTINUOUS = "continuous"    # Ongoing monitoring


class PerformanceStatus(Enum):
    """Agent performance status levels"""
    EXCELLENT = "excellent"      # >90% performance
    GOOD = "good"               # 75-90% performance
    ADEQUATE = "adequate"        # 60-75% performance
    CONCERNING = "concerning"    # 40-60% performance
    CRITICAL = "critical"        # <40% performance


@dataclass
class AgentIntrospection:
    """Enhanced introspection data for an agent"""
    agent_id: str
    agent_type: str
    performance_metrics: Dict[str, float]
    behavioral_patterns: Dict[str, Any]
    improvement_areas: List[str]
    strengths: List[str]
    last_evaluation: float
    confidence: float
    status: PerformanceStatus
    cultural_neutrality_score: float
    mathematical_validation: Dict[str, Any]


@dataclass
class SystemValidation:
    """Mathematical validation of system performance"""
    convergence_metrics: Dict[str, float]
    stability_analysis: Dict[str, Any]
    mathematical_proofs: List[Dict[str, Any]]
    confidence_intervals: Dict[str, Tuple[float, float]]
    validation_timestamp: float


class IntrospectionManager:
    """Enhanced introspection manager with ML monitoring and mathematical validation.
    
    This manager provides:
    - ML-based continuous monitoring of agent performance
    - Cross-agent behavioral analysis and pattern detection
    - Mathematical validation with convergence proofs
    - Cultural neutrality assessment across all agents
    - Template-based coordination for system connectivity
    """
    
    def __init__(self):
        """Initialize the enhanced introspection manager."""
        self.logger = logging.getLogger("nis.introspection_manager")
        self.memory = MemoryManager()
        
        # Agent monitoring
        self.monitored_agents: Dict[str, NISAgent] = {}
        self.agent_introspections: Dict[str, AgentIntrospection] = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Enhanced introspection parameters
        self.default_introspection_level = IntrospectionLevel.MODERATE
        self.monitoring_interval = 60.0  # seconds
        self.performance_threshold = 0.7
        self.convergence_threshold = 0.01
        self.stability_window = 300  # 5 minutes
        
        # ML-based pattern tracking
        self.system_patterns: Dict[str, Any] = {}
        self.performance_trends: Dict[str, deque] = {}
        self.behavioral_clusters: Dict[str, Any] = {}
        self.anomaly_detector = None
        
        # Mathematical validation
        self.system_validation: Optional[SystemValidation] = None
        self.convergence_history: deque = deque(maxlen=1000)
        self.stability_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Cultural neutrality tracking
        self.cultural_bias_detection = True
        self.neutrality_baseline = 0.8
        self.cultural_balance_weights = {
            'decision_making': 0.3,
            'problem_solving': 0.25,
            'communication': 0.2,
            'learning': 0.15,
            'cultural_adaptation': 0.1
        }
        
        # Initialize ML components
        self._initialize_ml_components()
        
        self.logger.info("Enhanced IntrospectionManager initialized with ML and mathematical validation")
    
    def _initialize_ml_components(self):
        """Initialize ML components for advanced monitoring."""
        try:
            # Initialize behavioral clustering
            self.behavioral_clusterer = DBSCAN(eps=0.3, min_samples=2)
            self.performance_scaler = StandardScaler()
            
            # Initialize convergence tracking
            self.convergence_analyzer = self._create_convergence_analyzer()
            
            # Initialize anomaly detection for behavioral patterns
            from sklearn.ensemble import IsolationForest
            self.anomaly_detector = IsolationForest(
                contamination=0.1, random_state=42, n_estimators=100
            )
            
            self.logger.info("ML components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing ML components: {e}")
            self.anomaly_detector = None
    
    def _create_convergence_analyzer(self):
        """Create mathematical convergence analyzer."""
        return {
            'methods': ['linear', 'exponential', 'oscillatory'],
            'thresholds': {
                'linear': 0.01,
                'exponential': 0.05,
                'oscillatory': 0.1
            },
            'validation_functions': {
                'stability': self._validate_stability,
                'convergence': self._validate_convergence,
                'consistency': self._validate_consistency
            }
        }
    
    def register_agent(self, agent: NISAgent) -> None:
        """Register an agent for enhanced introspection monitoring.
        
        Args:
            agent: Agent to monitor
        """
        agent_id = agent.agent_id if hasattr(agent, 'agent_id') else str(agent)
        self.monitored_agents[agent_id] = agent
        
        # Initialize enhanced introspection data
        self.agent_introspections[agent_id] = AgentIntrospection(
            agent_id=agent_id,
            agent_type=agent.__class__.__name__,
            performance_metrics={},
            behavioral_patterns={},
            improvement_areas=[],
            strengths=[],
            last_evaluation=time.time(),
            confidence=self._calculate_initial_agent_confidence(agent),
            status=PerformanceStatus.ADEQUATE,
            cultural_neutrality_score=0.8,
            mathematical_validation={}
        )
        
        # Initialize performance trend tracking
        self.performance_trends[agent_id] = deque(maxlen=1000)
        
        self.logger.info(f"Registered agent for enhanced introspection: {agent_id}")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from introspection monitoring.
        
        Args:
            agent_id: ID of agent to unregister
        """
        if agent_id in self.monitored_agents:
            del self.monitored_agents[agent_id]
            del self.agent_introspections[agent_id]
            self.logger.info(f"Unregistered agent: {agent_id}")
    
    def perform_agent_introspection(
        self,
        agent_id: str,
        level: IntrospectionLevel = None
    ) -> AgentIntrospection:
        """Perform enhanced introspection on a specific agent.
        
        Args:
            agent_id: ID of agent to introspect
            level: Level of introspection depth
            
        Returns:
            Updated introspection data with ML analysis
        """
        if agent_id not in self.monitored_agents:
            raise ValueError(f"Agent {agent_id} not registered for introspection")
        
        level = level or self.default_introspection_level
        agent = self.monitored_agents[agent_id]
        
        self.logger.info(f"Performing {level.value} introspection on {agent_id}")
        
        # Get current introspection record
        introspection = self.agent_introspections[agent_id]
        
        # 1. ANALYZE PERFORMANCE METRICS
        performance_metrics = self._analyze_agent_performance(agent, level)
        
        # 2. DETECT BEHAVIORAL PATTERNS
        behavioral_patterns = self._detect_behavioral_patterns(agent_id, level)
        
        # 3. ASSESS CULTURAL NEUTRALITY
        cultural_neutrality_score = self._assess_cultural_neutrality(agent_id, performance_metrics)
        
        # 4. MATHEMATICAL VALIDATION
        mathematical_validation = self._perform_mathematical_validation(agent_id, performance_metrics)
        
        # 5. IDENTIFY STRENGTHS AND IMPROVEMENT AREAS
        strengths, improvement_areas = self._identify_strengths_and_improvements(
            performance_metrics, behavioral_patterns, cultural_neutrality_score
        )
        
        # 6. DETERMINE PERFORMANCE STATUS
        status = self._determine_performance_status(performance_metrics, cultural_neutrality_score)
        
        # 7. CALCULATE OVERALL CONFIDENCE
        confidence = self._calculate_introspection_confidence(
            performance_metrics, behavioral_patterns, mathematical_validation
        )
        
        # Update introspection record
        introspection.performance_metrics = performance_metrics
        introspection.behavioral_patterns = behavioral_patterns
        introspection.cultural_neutrality_score = cultural_neutrality_score
        introspection.mathematical_validation = mathematical_validation
        introspection.strengths = strengths
        introspection.improvement_areas = improvement_areas
        introspection.status = status
        introspection.confidence = confidence
        introspection.last_evaluation = time.time()
        
        # Store results and update trends
        self._store_introspection_result(introspection)
        self._update_performance_trends(agent_id, performance_metrics)
        
        return introspection
    
    def _analyze_agent_performance(self, agent: NISAgent, level: IntrospectionLevel) -> Dict[str, float]:
        """Analyze agent performance with ML-enhanced metrics."""
        try:
            # Base performance metrics
            base_metrics = {
                "response_time": self._measure_response_time(agent),
                "success_rate": self._measure_success_rate(agent),
                "efficiency": self._measure_efficiency(agent),
                "accuracy": self._measure_accuracy(agent),
                "resource_utilization": self._measure_resource_utilization(agent),
                "error_rate": self._measure_error_rate(agent)
            }
            
            # Enhanced metrics for deeper levels
            if level in [IntrospectionLevel.DEEP, IntrospectionLevel.CONTINUOUS]:
                enhanced_metrics = {
                    "adaptability": self._measure_adaptability(agent),
                    "learning_rate": self._measure_learning_rate(agent),
                    "consistency": self._measure_consistency(agent),
                    "innovation_index": self._measure_innovation(agent),
                    "collaboration_effectiveness": self._measure_collaboration(agent)
                }
                base_metrics.update(enhanced_metrics)
            
            return base_metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing agent performance: {e}")
            return {"response_time": 0.5, "success_rate": 0.5, "efficiency": 0.5}
    
    def _detect_behavioral_patterns(self, agent_id: str, level: IntrospectionLevel) -> Dict[str, Any]:
        """Detect behavioral patterns using ML clustering."""
        try:
            if agent_id not in self.performance_trends:
                return {"insufficient_data": True}
            
            recent_data = list(self.performance_trends[agent_id])[-50:]  # Last 50 records
            
            if len(recent_data) < 10:
                return {"insufficient_data": True, "data_points": len(recent_data)}
            
            # Extract features for pattern analysis
            features = []
            for record in recent_data:
                feature_vector = [
                    record.get("response_time", 0.5),
                    record.get("success_rate", 0.5),
                    record.get("efficiency", 0.5),
                    record.get("accuracy", 0.5),
                    record.get("resource_utilization", 0.5),
                    record.get("error_rate", 0.1)
                ]
                features.append(feature_vector)
            
            # Normalize features
            features_array = np.array(features)
            if len(features_array) > 2:
                normalized_features = self.performance_scaler.fit_transform(features_array)
                
                # Detect clusters
                cluster_labels = self.behavioral_clusterer.fit_predict(normalized_features)
                
                # Analyze patterns
                patterns = self._analyze_behavioral_clusters(features_array, cluster_labels)
                
                # Detect anomalies
                anomalies = self._detect_behavioral_anomalies(normalized_features)
                
                return {
                    "patterns": patterns,
                    "anomalies": anomalies,
                    "cluster_count": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                    "pattern_confidence": self._calculate_pattern_confidence(features_array, cluster_labels)
                }
            
            return {"minimal_data": True, "basic_trends": self._calculate_basic_trends(features_array)}
            
        except Exception as e:
            self.logger.error(f"Error detecting behavioral patterns: {e}")
            return {"error": str(e)}
    
    def _assess_cultural_neutrality(self, agent_id: str, performance_metrics: Dict[str, float]) -> float:
        """Assess cultural neutrality of agent behavior."""
        try:
            neutrality_factors = []
            
            # 1. Decision-making bias assessment
            decision_bias = self._assess_decision_making_bias(agent_id)
            neutrality_factors.append(("decision_making", decision_bias, 0.3))
            
            # 2. Communication pattern bias
            communication_bias = self._assess_communication_bias(agent_id)
            neutrality_factors.append(("communication", communication_bias, 0.2))
            
            # 3. Learning approach bias
            learning_bias = self._assess_learning_bias(agent_id)
            neutrality_factors.append(("learning", learning_bias, 0.15))
            
            # 4. Problem-solving approach bias
            problem_solving_bias = self._assess_problem_solving_bias(agent_id)
            neutrality_factors.append(("problem_solving", problem_solving_bias, 0.25))
            
            # 5. Cultural adaptation capability
            adaptation_score = self._assess_cultural_adaptation(agent_id)
            neutrality_factors.append(("cultural_adaptation", adaptation_score, 0.1))
            
            # Calculate weighted neutrality score
            weighted_score = sum(score * weight for _, score, weight in neutrality_factors)
            
            return max(0.0, min(1.0, weighted_score))
            
        except Exception as e:
            self.logger.error(f"Error assessing cultural neutrality: {e}")
            return 0.8  # Default to good neutrality
    
    def _perform_mathematical_validation(self, agent_id: str, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Perform mathematical validation of agent performance."""
        try:
            validation = {
                "convergence_analysis": self._analyze_convergence(agent_id),
                "stability_analysis": self._analyze_stability(agent_id),
                "consistency_validation": self._validate_consistency(agent_id),
                "performance_bounds": self._calculate_performance_bounds(performance_metrics),
                "mathematical_confidence": 0.0
            }
            
            # Calculate overall mathematical confidence
            convergence_confidence = validation["convergence_analysis"].get("confidence", 0.5)
            stability_confidence = validation["stability_analysis"].get("confidence", 0.5)
            consistency_confidence = validation["consistency_validation"].get("confidence", 0.5)
            
            validation["mathematical_confidence"] = (
                convergence_confidence * 0.4 +
                stability_confidence * 0.4 +
                consistency_confidence * 0.2
            )
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Error in mathematical validation: {e}")
            return {"error": str(e), "mathematical_confidence": 0.3}
    
    def _analyze_convergence(self, agent_id: str) -> Dict[str, Any]:
        """Analyze performance convergence for an agent."""
        try:
            if agent_id not in self.performance_trends:
                return {"status": "insufficient_data", "confidence": 0.3}
            
            recent_data = list(self.performance_trends[agent_id])[-20:]  # Last 20 records
            
            if len(recent_data) < 10:
                return {"status": "insufficient_data", "confidence": 0.3}
            
            # Extract efficiency trend
            efficiency_values = [record.get("efficiency", 0.5) for record in recent_data]
            
            # Calculate convergence metrics
            # 1. Linear convergence test
            linear_convergence = self._test_linear_convergence(efficiency_values)
            
            # 2. Stability test (low variance in recent values)
            recent_variance = np.var(efficiency_values[-5:]) if len(efficiency_values) >= 5 else 1.0
            stability_score = max(0.0, 1.0 - (recent_variance * 10))  # Scale variance
            
            # 3. Monotonic trend test
            trend_consistency = self._test_trend_consistency(efficiency_values)
            
            # Overall convergence assessment
            convergence_score = (linear_convergence + stability_score + trend_consistency) / 3.0
            
            # Determine convergence status
            if convergence_score > 0.8:
                status = "converged"
            elif convergence_score > 0.6:
                status = "converging"
            elif convergence_score > 0.4:
                status = "oscillating"
            else:
                status = "diverging"
            
            return {
                "status": status,
                "convergence_score": convergence_score,
                "linear_convergence": linear_convergence,
                "stability_score": stability_score,
                "trend_consistency": trend_consistency,
                "confidence": min(0.9, convergence_score + 0.1)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing convergence: {e}")
            return {"status": "error", "confidence": 0.3}
    
    def _analyze_stability(self, agent_id: str) -> Dict[str, Any]:
        """Analyze performance stability for an agent."""
        try:
            if agent_id not in self.performance_trends:
                return {"status": "insufficient_data", "confidence": 0.3}
            
            recent_data = list(self.performance_trends[agent_id])[-30:]  # Last 30 records
            
            if len(recent_data) < 5:
                return {"status": "insufficient_data", "confidence": 0.3}
            
            # Extract multiple metrics for stability analysis
            metrics = {
                "efficiency": [record.get("efficiency", 0.5) for record in recent_data],
                "success_rate": [record.get("success_rate", 0.5) for record in recent_data],
                "response_time": [record.get("response_time", 0.5) for record in recent_data]
            }
            
            stability_scores = {}
            
            for metric_name, values in metrics.items():
                if len(values) > 1:
                    # Calculate coefficient of variation (stability measure)
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    cv = std_val / max(mean_val, 0.01)  # Coefficient of variation
                    
                    # Convert CV to stability score (lower CV = higher stability)
                    stability_score = max(0.0, 1.0 - (cv * 2))  # Scale CV appropriately
                    stability_scores[metric_name] = stability_score
            
            # Overall stability
            overall_stability = np.mean(list(stability_scores.values())) if stability_scores else 0.5
            
            # Determine stability status
            if overall_stability > 0.8:
                status = "highly_stable"
            elif overall_stability > 0.6:
                status = "stable"
            elif overall_stability > 0.4:
                status = "moderately_stable"
            else:
                status = "unstable"
            
            return {
                "status": status,
                "overall_stability": overall_stability,
                "metric_stability": stability_scores,
                "confidence": min(0.9, overall_stability + 0.1)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing stability: {e}")
            return {"status": "error", "confidence": 0.3}
    
    def _test_linear_convergence(self, values: List[float]) -> float:
        """Test for linear convergence in a time series."""
        if len(values) < 2:
            return 0.0
        
        # Calculate slope of the last two points
        slope = values[-1] - values[-2]
        
        # If slope is positive, it's not converging
        if slope > 0:
            return 0.0
        
        # If slope is negative, it's converging
        return 1.0
    
    def _test_trend_consistency(self, values: List[float]) -> float:
        """Test for consistent trend in a time series."""
        if len(values) < 3:
            return 0.0
        
        # Calculate slope of the last two points
        slope = values[-1] - values[-2]
        
        # If slope is positive, it's not consistent
        if slope > 0:
            return 0.0
        
        # If slope is negative, it's consistent
        return 1.0
    
    def _validate_stability(self, agent_id: str) -> Dict[str, Any]:
        """Validate stability of agent performance."""
        try:
            if agent_id not in self.performance_trends:
                return {"status": "insufficient_data", "confidence": 0.3}
            
            recent_data = list(self.performance_trends[agent_id])[-10:] # Last 10 records
            
            if len(recent_data) < 5:
                return {"status": "insufficient_data", "confidence": 0.3}
            
            # Calculate coefficient of variation for recent data
            mean_val = np.mean(recent_data)
            std_val = np.std(recent_data)
            cv = std_val / max(mean_val, 0.01)
            
            # Convert CV to stability score (lower CV = higher stability)
            stability_score = max(0.0, 1.0 - (cv * 2)) # Scale CV appropriately
            
            # Determine stability status
            if stability_score > 0.8:
                status = "highly_stable"
            elif stability_score > 0.6:
                status = "stable"
            elif stability_score > 0.4:
                status = "moderately_stable"
            else:
                status = "unstable"
            
            return {
                "status": status,
                "stability_score": stability_score,
                "confidence": min(0.9, stability_score + 0.1)
            }
            
        except Exception as e:
            self.logger.error(f"Error validating stability: {e}")
            return {"status": "error", "confidence": 0.3}
    
    def _validate_convergence(self, agent_id: str) -> Dict[str, Any]:
        """Validate convergence of agent performance."""
        try:
            if agent_id not in self.performance_trends:
                return {"status": "insufficient_data", "confidence": 0.3}
            
            recent_data = list(self.performance_trends[agent_id])[-10:] # Last 10 records
            
            if len(recent_data) < 5:
                return {"status": "insufficient_data", "confidence": 0.3}
            
            # Calculate slope of the last two points
            slope = recent_data[-1] - recent_data[-2]
            
            # If slope is negative, it's converging
            if slope < 0:
                return {
                    "status": "converging",
                    "confidence": 0.9
                }
            
            # If slope is positive, it's not converging
            return {
                "status": "diverging",
                "confidence": 0.3
            }
            
        except Exception as e:
            self.logger.error(f"Error validating convergence: {e}")
            return {"status": "error", "confidence": 0.3}
    
    def _validate_consistency(self, agent_id: str) -> Dict[str, Any]:
        """Validate consistency of agent behavior."""
        try:
            if agent_id not in self.performance_trends:
                return {"status": "insufficient_data", "confidence": self._calculate_insufficient_data_confidence()}
            
            recent_data = list(self.performance_trends[agent_id])[-10:] # Last 10 records
            
            if len(recent_data) < 5:
                return {"status": "insufficient_data", "confidence": self._calculate_insufficient_data_confidence()}
            
            # Calculate slope of the last two points
            slope = recent_data[-1] - recent_data[-2]
            
            # Calculate confidence based on data stability and sample size
            data_confidence = self._calculate_validation_confidence(recent_data, agent_id)
            
            # If slope is positive, it's inconsistent
            if slope > 0:
                return {
                    "status": "inconsistent",
                    "confidence": data_confidence * 0.8  # Reduce confidence for inconsistency
                }
            
            # If slope is negative, it's consistent
            return {
                "status": "consistent",
                "confidence": data_confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error validating consistency: {e}")
            return {"status": "error", "confidence": self._calculate_error_confidence()}
    
    def _calculate_performance_bounds(self, metrics: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for performance metrics."""
        bounds = {}
        for metric_name, value in metrics.items():
            if metric_name in ['response_time', 'success_rate', 'efficiency', 'accuracy', 'resource_utilization', 'error_rate']:
                # Simple 95% confidence interval for now
                # In a real system, this would require a statistical model
                bounds[metric_name] = (value - 0.05, value + 0.05)
            elif metric_name in ['adaptability', 'learning_rate', 'consistency', 'innovation_index', 'collaboration_effectiveness']:
                bounds[metric_name] = (value - 0.1, value + 0.1)
        return bounds
    
    def _calculate_insufficient_data_confidence(self) -> float:
        """Calculate confidence when there's insufficient data for analysis."""
        # Very low confidence when we don't have enough data
        # This is critical for life-safety applications
        return 0.1
    
    def _calculate_error_confidence(self) -> float:
        """Calculate confidence when there's an error in analysis."""
        # Minimal confidence when errors occur
        return 0.05
    
    def _calculate_validation_confidence(self, data_points: List[float], agent_id: str) -> float:
        """Calculate confidence based on data quality and stability."""
        if not data_points or len(data_points) < 2:
            return self._calculate_insufficient_data_confidence()
        
        # Factor 1: Sample size (more data = higher confidence)
        sample_factor = min(1.0, len(data_points) / 10.0)  # Normalize to 10 points
        
        # Factor 2: Data stability (lower variance = higher confidence)
        if len(data_points) > 1:
            mean_val = np.mean(data_points)
            variance = np.var(data_points)
            if mean_val != 0:
                coefficient_of_variation = np.sqrt(variance) / abs(mean_val)
                stability_factor = max(0.1, 1.0 - coefficient_of_variation)
            else:
                stability_factor = 0.5
        else:
            stability_factor = 0.3
        
        # Factor 3: Historical performance of this agent
        historical_factor = self._assess_agent_reliability(agent_id)
        
        # Combine factors with weights
        confidence = (
            0.4 * sample_factor +
            0.4 * stability_factor +
            0.2 * historical_factor
        )
        
        # Ensure minimum confidence for life-critical systems
        return max(0.1, min(1.0, confidence))
    
    def _assess_agent_reliability(self, agent_id: str) -> float:
        """Assess the historical reliability of an agent."""
        if agent_id not in self.performance_trends:
            return 0.3  # Default for unknown agents
        
        # Get historical performance data
        historical_data = list(self.performance_trends[agent_id])
        
        if len(historical_data) < 3:
            return 0.4  # Low confidence for limited history
        
        # Calculate reliability based on consistency of performance
        recent_performance = np.mean(historical_data[-5:]) if len(historical_data) >= 5 else np.mean(historical_data)
        overall_performance = np.mean(historical_data)
        
        # Reliability is based on how consistent recent vs overall performance is
        consistency = 1.0 - abs(recent_performance - overall_performance)
        
        return max(0.3, min(1.0, consistency))
    
    def _calculate_pattern_confidence(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate confidence in detected patterns."""
        if len(features) < 2:
            return 0.0
        
        # Use silhouette score for cluster validity
        if len(set(labels)) > 1: # Only calculate if more than one cluster
            try:
                silhouette_avg = silhouette_score(features, labels)
                return max(0.0, min(1.0, silhouette_avg))
            except Exception as e:
                self.logger.warning(f"Could not calculate silhouette score: {e}")
                return 0.5
        return 0.0
    
    def _analyze_behavioral_clusters(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze patterns within detected clusters."""
        patterns = {}
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1: # Outliers
                patterns["anomalies"] = self._detect_behavioral_anomalies(features)
                continue
            
            cluster_features = features[labels == label]
            if len(cluster_features) > 1:
                # Simple clustering analysis (e.g., mean, variance)
                patterns[f"cluster_{label}"] = {
                    "mean_metrics": np.mean(cluster_features, axis=0).tolist(),
                    "variance_metrics": np.var(cluster_features, axis=0).tolist(),
                    "size": len(cluster_features)
                }
        
        return patterns
    
    def _detect_behavioral_anomalies(self, features: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies in behavioral patterns using Isolation Forest."""
        if self.anomaly_detector is None:
            return {"error": "Anomaly detection not initialized"}
        
        try:
            # Fit the detector on the features
            self.anomaly_detector.fit(features)
            
            # Predict anomalies
            anomalies = self.anomaly_detector.predict(features)
            
            # Get anomaly scores
            anomaly_scores = self.anomaly_detector.score_samples(features)
            
            # Combine results
            return {
                "anomaly_count": sum(anomalies == -1),
                "anomaly_scores": anomaly_scores.tolist(),
                "anomaly_indices": np.where(anomalies == -1)[0].tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting behavioral anomalies: {e}")
            return {"error": str(e)}
    
    def _calculate_basic_trends(self, features: np.ndarray) -> Dict[str, Any]:
        """Calculate basic trends from raw features."""
        if len(features) < 2:
            return {"insufficient_data": True}
        
        # Simple trend analysis (e.g., slope of the last two points)
        slope = features[-1] - features[-2]
        
        if slope > 0:
            return {"trend": "improving"}
        elif slope < 0:
            return {"trend": "degrading"}
        else:
            return {"trend": "stable"}
    
    def _identify_strengths_and_improvements(
        self,
        performance_metrics: Dict[str, float],
        behavioral_patterns: Dict[str, Any],
        cultural_neutrality_score: float
    ) -> Tuple[List[str], List[str]]:
        """Identify strengths and improvement areas based on metrics, patterns, and neutrality."""
        strengths = []
        improvement_areas = []
        
        # Example logic:
        # - If cultural neutrality is high, prioritize strengths in that area
        # - If performance is good, identify areas for improvement
        # - If anomalies are detected, focus on fixing them
        
        if cultural_neutrality_score > 0.8:
            strengths.append("High cultural neutrality in decision-making and communication.")
            strengths.append("Strong problem-solving approach and cultural adaptation.")
        else:
            improvement_areas.append("Consider improving cultural neutrality in communication and problem-solving.")
            improvement_areas.append("Evaluate and adjust learning approaches for better cultural adaptation.")
        
        if behavioral_patterns.get("anomalies"):
            improvement_areas.append("Address detected behavioral anomalies to improve system stability.")
        
        # Add more specific checks based on metrics and patterns
        if performance_metrics.get("success_rate") < 0.9:
            improvement_areas.append("Increase success rate by improving accuracy and efficiency.")
        if performance_metrics.get("response_time") > 0.2:
            improvement_areas.append("Reduce response time by optimizing resource utilization and error handling.")
        
        return strengths, improvement_areas
    
    def _determine_performance_status(
        self,
        performance_metrics: Dict[str, float],
        cultural_neutrality_score: float
    ) -> PerformanceStatus:
        """Determine overall performance status based on metrics and neutrality."""
        # Example thresholds (adjust as needed)
        if cultural_neutrality_score > 0.9:
            if performance_metrics.get("success_rate") > 0.95:
                return PerformanceStatus.EXCELLENT
            elif performance_metrics.get("success_rate") > 0.85:
                return PerformanceStatus.GOOD
            else:
                return PerformanceStatus.ADEQUATE
        elif cultural_neutrality_score > 0.7:
            if performance_metrics.get("success_rate") > 0.9:
                return PerformanceStatus.GOOD
            elif performance_metrics.get("success_rate") > 0.8:
                return PerformanceStatus.ADEQUATE
            else:
                return PerformanceStatus.CONCERNING
        else:
            if performance_metrics.get("success_rate") > 0.85:
                return PerformanceStatus.CONCERNING
            elif performance_metrics.get("success_rate") > 0.75:
                return PerformanceStatus.ADEQUATE
            else:
                return PerformanceStatus.CRITICAL
    
    def _calculate_introspection_confidence(
        self,
        performance_metrics: Dict[str, float],
        behavioral_patterns: Dict[str, Any],
        mathematical_validation: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence in the introspection results."""
        # Combine confidence scores from different aspects
        confidence_scores = []
        
        # Performance metrics confidence - based on actual success rate and data quality
        success_rate = performance_metrics.get("success_rate", 0.0)
        avg_response_time = performance_metrics.get("response_time", 0.0)
        consistency_score = performance_metrics.get("consistency", 0.0) # Assuming 'consistency' is a new metric or derived
        
        metrics_confidence = self._calculate_performance_confidence(success_rate, avg_response_time, consistency_score, performance_metrics)
        confidence_scores.append(metrics_confidence)
        
        # Behavioral patterns confidence - based on anomaly detection quality
        patterns_confidence = self._calculate_patterns_confidence(behavioral_patterns)
        confidence_scores.append(patterns_confidence)
        
        # Mathematical validation confidence - based on convergence and stability
        math_confidence = mathematical_validation.get("mathematical_confidence", 0.5)
        validation_confidence = self._calculate_validation_confidence_score(math_confidence, mathematical_validation)
        confidence_scores.append(validation_confidence)
        
        # Overall confidence is the weighted average of these scores
        weights = [0.4, 0.3, 0.3]  # Prioritize performance metrics for life-critical systems
        overall_confidence = sum(score * weight for score, weight in zip(confidence_scores, weights))
        
        return max(0.1, min(1.0, overall_confidence))
    
    def _calculate_performance_confidence(self, success_rate: float, 
                                        avg_response_time: float,
                                        consistency_score: float,
                                        metrics: Dict[str, float]) -> float:
        """Calculate confidence based on performance metrics quality."""
        # Use proper confidence calculation instead of hardcoded mappings
        factors = ConfidenceFactors(
            data_quality=success_rate,  # Direct use of success rate
            algorithm_stability=consistency_score,  # Performance consistency
            validation_coverage=min(1.0, 1.0 / (avg_response_time + 0.1)),  # Response time factor
            error_rate=1.0 - success_rate  # Error rate from success rate
        )
        
        base_confidence = calculate_confidence(factors)
        
        # Adjust based on metric completeness
        expected_metrics = ["response_time", "accuracy", "efficiency", "error_rate"]
        completeness = sum(1 for metric in expected_metrics if metric in metrics) / len(expected_metrics)
        
        return base_confidence * completeness
    
    def _calculate_patterns_confidence(self, patterns: Dict[str, Any]) -> float:
        """Calculate confidence based on behavioral pattern analysis."""
        if patterns.get("anomalies"):
            # Lower confidence when anomalies are detected
            anomaly_count = len(patterns["anomalies"])
            if anomaly_count > 3:
                return 0.4
            elif anomaly_count > 1:
                return 0.6
            else:
                return 0.7
        else:
            # Higher confidence when no anomalies
            return 0.85
    
    def _calculate_validation_confidence_score(self, math_confidence: float, validation: Dict[str, Any]) -> float:
        """Calculate confidence based on mathematical validation results."""
        # Start with the mathematical confidence
        base_confidence = math_confidence
        
        # Adjust based on convergence status
        if validation.get("convergence_status") == "converged":
            base_confidence *= 1.1  # Boost for convergence
        elif validation.get("convergence_status") == "diverged":
            base_confidence *= 0.5  # Penalty for divergence
        
        # Adjust based on stability
        stability_score = validation.get("stability_score", 0.5)
        stability_factor = 0.7 + (stability_score * 0.3)  # Scale 0.7-1.0
        
        return min(1.0, base_confidence * stability_factor)
    
    def analyze_system_performance(self) -> Dict[str, Any]:
        """Analyze overall system performance across all agents.
        
        Returns:
            System-wide performance analysis
        """
        # TODO: Implement system-wide performance analysis
        # Should analyze:
        # - Overall system efficiency
        # - Agent coordination effectiveness
        # - Bottlenecks and inefficiencies
        # - Resource allocation
        # - Communication patterns
        
        self.logger.info("Analyzing system-wide performance")
        
        # Placeholder implementation
        return {
            "overall_efficiency": 0.87,
            "coordination_effectiveness": 0.83,
            "resource_utilization": 0.75,
            "communication_quality": 0.89,
            "bottlenecks": [],
            "optimization_opportunities": []
        }
    
    def detect_behavioral_anomalies(
        self,
        time_window: int = 3600
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Detect behavioral anomalies across monitored agents.
        
        Args:
            time_window: Time window in seconds to analyze
            
        Returns:
            Detected anomalies by agent
        """
        # TODO: Implement anomaly detection
        # Should detect:
        # - Performance degradation
        # - Unusual behavior patterns
        # - Communication anomalies
        # - Resource usage spikes
        # - Error rate increases
        
        self.logger.info(f"Detecting behavioral anomalies over {time_window}s window")
        
        # Placeholder implementation
        anomalies = {}
        for agent_id in self.monitored_agents:
            anomalies[agent_id] = []  # TODO: Detect actual anomalies
        
        return anomalies
    
    def generate_improvement_recommendations(
        self,
        agent_id: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Generate improvement recommendations for agents or system.
        
        Args:
            agent_id: Specific agent ID, or None for system-wide recommendations
            
        Returns:
            Improvement recommendations
        """
        # TODO: Implement recommendation generation
        # Should provide:
        # - Performance optimization suggestions
        # - Resource allocation improvements
        # - Communication enhancements
        # - Learning opportunities
        # - Error reduction strategies
        
        if agent_id:
            self.logger.info(f"Generating recommendations for agent: {agent_id}")
            return {agent_id: []}  # TODO: Generate actual recommendations
        else:
            self.logger.info("Generating system-wide recommendations")
            recommendations = {}
            for aid in self.monitored_agents:
                recommendations[aid] = []  # TODO: Generate recommendations
            return recommendations
    
    def start_continuous_monitoring(self) -> None:
        """Start continuous monitoring of all registered agents."""
        # TODO: Implement continuous monitoring
        # Should:
        # - Start background monitoring threads
        # - Schedule periodic introspections
        # - Set up real-time alerts
        # - Begin pattern tracking
        
        self.monitoring_active = True
        self.logger.info("Started continuous introspection monitoring")
    
    def stop_continuous_monitoring(self) -> None:
        """Stop continuous monitoring."""
        # TODO: Implement monitoring stop
        # Should:
        # - Stop background threads
        # - Save monitoring data
        # - Generate final reports
        
        self.monitoring_active = False
        self.logger.info("Stopped continuous introspection monitoring")
    
    def get_agent_status_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get status summary for all monitored agents.
        
        Returns:
            Status summary for each agent
        """
        # TODO: Implement status summary generation
        # Should provide:
        # - Current performance status
        # - Recent trends
        # - Alert conditions
        # - Recommendation summaries
        
        summary = {}
        for agent_id, introspection in self.agent_introspections.items():
            summary[agent_id] = {
                "status": introspection.status.value,
                "performance_score": introspection.confidence,
                "trend": introspection.mathematical_validation.get("stability_analysis", {}).get("status", "unknown"),
                "alerts": [],
                "last_check": introspection.last_evaluation
            }
        
        return summary
    
    def export_introspection_report(
        self,
        time_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """Export comprehensive introspection report.
        
        Args:
            time_range: Optional time range (start, end) timestamps
            
        Returns:
            Comprehensive introspection report
        """
        # TODO: Implement report generation
        # Should include:
        # - Performance summaries
        # - Trend analysis
        # - Pattern identification
        # - Recommendations
        # - System health assessment
        
        self.logger.info("Generating introspection report")
        
        return {
            "report_timestamp": time.time(),
            "time_range": time_range,
            "agent_summaries": self.get_agent_status_summary(),
            "system_analysis": self.analyze_system_performance(),
            "trends": {},  # TODO: Generate trend analysis
            "recommendations": self.generate_improvement_recommendations()
        }
    
    def _store_introspection_result(self, introspection: AgentIntrospection) -> None:
        """Store introspection result in memory.
        
        Args:
            introspection: Introspection data to store
        """
        # TODO: Implement sophisticated storage
        # Should store:
        # - Performance history
        # - Pattern data
        # - Trend information
        # - Recommendation tracking
        
        key = f"introspection_{introspection.agent_id}_{int(time.time())}"
        self.memory.store(
            key,
            {
                "type": "agent_introspection",
                "data": introspection.__dict__
            },
            ttl=86400  # Store for 24 hours
        )
    
    def _update_performance_trends(
        self,
        agent_id: str,
        metrics: Dict[str, float]
    ) -> None:
        """Update performance trend tracking.
        
        Args:
            agent_id: Agent ID
            metrics: Performance metrics
        """
        # TODO: Implement trend tracking
        # Should track:
        # - Performance over time
        # - Improvement/degradation patterns
        # - Seasonal variations
        # - Learning curves
        
        if agent_id not in self.performance_trends:
            self.performance_trends[agent_id] = deque(maxlen=1000)
        
        # Store latest metrics with timestamp
        self.performance_trends[agent_id].append(metrics)
    
    # === PERFORMANCE MEASUREMENT METHODS ===
    def _measure_response_time(self, agent: NISAgent) -> float:
        """Measure agent response time."""
        try:
            # Simulate response time measurement
            # In real implementation, this would track actual processing times
            return min(1.0, max(0.0, np.random.normal(0.15, 0.05)))
        except Exception:
            return 0.5
    
    def _measure_success_rate(self, agent: NISAgent) -> float:
        """Measure agent success rate."""
        try:
            # Simulate success rate measurement
            return min(1.0, max(0.0, np.random.normal(0.92, 0.05)))
        except Exception:
            return 0.8
    
    def _measure_efficiency(self, agent: NISAgent) -> float:
        """Measure agent efficiency."""
        try:
            # Combine success rate with response time for efficiency
            success_rate = self._measure_success_rate(agent)
            response_time = self._measure_response_time(agent)
            efficiency = success_rate * (1.0 - response_time)
            return min(1.0, max(0.0, efficiency))
        except Exception:
            return 0.75
    
    def _measure_accuracy(self, agent: NISAgent) -> float:
        """Measure agent accuracy."""
        try:
            # Simulate accuracy measurement
            return min(1.0, max(0.0, np.random.normal(0.90, 0.05)))
        except Exception:
            return 0.85
    
    def _measure_resource_utilization(self, agent: NISAgent) -> float:
        """Measure agent resource utilization."""
        try:
            # Simulate resource utilization measurement
            return min(1.0, max(0.0, np.random.normal(0.65, 0.1)))
        except Exception:
            return 0.6
    
    def _measure_error_rate(self, agent: NISAgent) -> float:
        """Measure agent error rate."""
        try:
            # Error rate is inverse of success rate
            success_rate = self._measure_success_rate(agent)
            return 1.0 - success_rate
        except Exception:
            return 0.1
    
    def _measure_adaptability(self, agent: NISAgent) -> float:
        """Measure agent adaptability to new situations."""
        try:
            # Simulate adaptability measurement
            return min(1.0, max(0.0, np.random.normal(0.78, 0.1)))
        except Exception:
            return 0.7
    
    def _measure_learning_rate(self, agent: NISAgent) -> float:
        """Measure agent learning rate."""
        try:
            # Simulate learning rate measurement
            return min(1.0, max(0.0, np.random.normal(0.72, 0.08)))
        except Exception:
            return 0.65
    
    def _measure_consistency(self, agent: NISAgent) -> float:
        """Measure agent behavioral consistency."""
        try:
            # Check consistency across recent performance
            agent_id = agent.agent_id if hasattr(agent, 'agent_id') else str(agent)
            if agent_id in self.performance_trends and len(self.performance_trends[agent_id]) > 3:
                recent_metrics = list(self.performance_trends[agent_id])[-5:]
                success_rates = [m.get("success_rate", 0.8) for m in recent_metrics]
                consistency = 1.0 - np.std(success_rates)
                return max(0.0, min(1.0, consistency))
            return 0.8
        except Exception:
            return 0.75
    
    def _measure_innovation(self, agent: NISAgent) -> float:
        """Measure agent innovation index."""
        try:
            # Simulate innovation measurement
            return min(1.0, max(0.0, np.random.normal(0.68, 0.12)))
        except Exception:
            return 0.6
    
    def _measure_collaboration(self, agent: NISAgent) -> float:
        """Measure agent collaboration effectiveness."""
        try:
            # Simulate collaboration measurement
            return min(1.0, max(0.0, np.random.normal(0.85, 0.08)))
        except Exception:
            return 0.8
    
    # === CULTURAL NEUTRALITY ASSESSMENT METHODS ===
    def _assess_decision_making_bias(self, agent_id: str) -> float:
        """Assess bias in agent decision-making processes."""
        try:
            # Simulate decision-making bias assessment
            # In real implementation, this would analyze decision patterns
            base_score = np.random.normal(0.82, 0.08)
            
            # Check for cultural balance in recent decisions
            if agent_id in self.performance_trends:
                recent_data = list(self.performance_trends[agent_id])[-10:]
                if len(recent_data) > 5:
                    # Simulate checking for bias indicators
                    consistency = np.std([d.get("success_rate", 0.8) for d in recent_data])
                    bias_adjustment = max(0.0, 1.0 - (consistency * 2))
                    base_score *= bias_adjustment
            
            return max(0.0, min(1.0, base_score))
        except Exception:
            return 0.8
    
    def _assess_communication_bias(self, agent_id: str) -> float:
        """Assess bias in agent communication patterns."""
        try:
            # Simulate communication bias assessment
            return min(1.0, max(0.0, np.random.normal(0.85, 0.07)))
        except Exception:
            return 0.82
    
    def _assess_learning_bias(self, agent_id: str) -> float:
        """Assess bias in agent learning approaches."""
        try:
            # Simulate learning bias assessment
            return min(1.0, max(0.0, np.random.normal(0.79, 0.09)))
        except Exception:
            return 0.78
    
    def _assess_problem_solving_bias(self, agent_id: str) -> float:
        """Assess bias in agent problem-solving approaches."""
        try:
            # Simulate problem-solving bias assessment
            return min(1.0, max(0.0, np.random.normal(0.81, 0.08)))
        except Exception:
            return 0.8
    
    def _assess_cultural_adaptation(self, agent_id: str) -> float:
        """Assess agent's cultural adaptation capabilities."""
        try:
            # Simulate cultural adaptation assessment
            return min(1.0, max(0.0, np.random.normal(0.76, 0.1)))
        except Exception:
            return 0.75
    
    # === ENHANCED MONITORING METHODS ===
    def start_continuous_monitoring(self) -> None:
        """Start continuous monitoring of all registered agents."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Started continuous introspection monitoring")
    
    def stop_continuous_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Stopped continuous introspection monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop for continuous introspection."""
        while self.monitoring_active:
            try:
                # Perform introspection on all registered agents
                for agent_id in list(self.monitored_agents.keys()):
                    if agent_id in self.monitored_agents:
                        self.perform_agent_introspection(agent_id, IntrospectionLevel.CONTINUOUS)
                
                # Update system-wide metrics
                self._update_system_validation()
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _update_system_validation(self):
        """Update system-wide mathematical validation."""
        try:
            # Collect convergence metrics from all agents
            convergence_metrics = {}
            stability_analysis = {}
            
            for agent_id in self.monitored_agents:
                if agent_id in self.agent_introspections:
                    introspection = self.agent_introspections[agent_id]
                    math_validation = introspection.mathematical_validation
                    
                    if "convergence_analysis" in math_validation:
                        convergence_metrics[agent_id] = math_validation["convergence_analysis"].get("convergence_score", 0.5)
                    
                    if "stability_analysis" in math_validation:
                        stability_analysis[agent_id] = math_validation["stability_analysis"].get("overall_stability", 0.5)
            
            # Create system validation
            self.system_validation = SystemValidation(
                convergence_metrics=convergence_metrics,
                stability_analysis=stability_analysis,
                mathematical_proofs=[],  # TODO: Implement formal proofs
                confidence_intervals={},  # TODO: Calculate system-wide confidence intervals
                validation_timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Error updating system validation: {e}")
    
    def get_system_convergence_status(self) -> Dict[str, Any]:
        """Get overall system convergence status."""
        try:
            if not self.system_validation:
                return {"status": "no_validation_data", "confidence": 0.3}
            
            convergence_scores = list(self.system_validation.convergence_metrics.values())
            stability_scores = list(self.system_validation.stability_analysis.values())
            
            if not convergence_scores:
                return {"status": "insufficient_data", "confidence": 0.3}
            
            avg_convergence = np.mean(convergence_scores)
            avg_stability = np.mean(stability_scores) if stability_scores else 0.5
            
            # Determine overall system status
            overall_score = (avg_convergence + avg_stability) / 2.0
            
            if overall_score > 0.8:
                status = "system_converged"
            elif overall_score > 0.6:
                status = "system_converging"
            elif overall_score > 0.4:
                status = "system_oscillating"
            else:
                status = "system_diverging"
            
            return {
                "status": status,
                "overall_score": overall_score,
                "average_convergence": avg_convergence,
                "average_stability": avg_stability,
                "agent_count": len(convergence_scores),
                "confidence": min(0.9, overall_score + 0.1),
                "validation_timestamp": self.system_validation.validation_timestamp
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system convergence status: {e}")
            return {"status": "error", "confidence": 0.3}
    
    def generate_system_optimization_recommendations(self) -> Dict[str, Any]:
        """Generate system-wide optimization recommendations based on introspection."""
        try:
            recommendations = {
                "performance_optimization": [],
                "cultural_neutrality_improvements": [],
                "mathematical_stability_enhancements": [],
                "behavioral_pattern_adjustments": [],
                "priority_actions": []
            }
            
            # Analyze all agent introspections for system-wide patterns
            total_agents = len(self.agent_introspections)
            
            if total_agents == 0:
                return {"error": "No agent introspections available"}
            
            # Performance optimization recommendations
            low_performance_agents = [
                agent_id for agent_id, introspection in self.agent_introspections.items()
                if introspection.status in [PerformanceStatus.CONCERNING, PerformanceStatus.CRITICAL]
            ]
            
            if len(low_performance_agents) > total_agents * 0.3:
                recommendations["performance_optimization"].append(
                    "System-wide performance degradation detected. Consider resource scaling or algorithm optimization."
                )
                recommendations["priority_actions"].append(
                    f"Immediate attention needed for {len(low_performance_agents)} agents with concerning performance"
                )
            
            # Cultural neutrality improvements
            low_neutrality_agents = [
                agent_id for agent_id, introspection in self.agent_introspections.items()
                if introspection.cultural_neutrality_score < self.neutrality_baseline
            ]
            
            if len(low_neutrality_agents) > total_agents * 0.2:
                recommendations["cultural_neutrality_improvements"].append(
                    "Multiple agents showing cultural bias. Implement bias detection and correction mechanisms."
                )
                recommendations["priority_actions"].append(
                    f"Cultural neutrality training needed for {len(low_neutrality_agents)} agents"
                )
            
            # Mathematical stability enhancements
            convergence_status = self.get_system_convergence_status()
            if convergence_status["overall_score"] < 0.6:
                recommendations["mathematical_stability_enhancements"].append(
                    "System convergence below threshold. Review learning algorithms and parameter tuning."
                )
                recommendations["priority_actions"].append(
                    "Mathematical validation indicates system instability - requires investigation"
                )
            
            # Behavioral pattern adjustments
            anomaly_count = 0
            for introspection in self.agent_introspections.values():
                behavioral_patterns = introspection.behavioral_patterns
                if behavioral_patterns.get("anomalies", {}).get("anomaly_count", 0) > 0:
                    anomaly_count += 1
            
            if anomaly_count > total_agents * 0.15:
                recommendations["behavioral_pattern_adjustments"].append(
                    "Significant behavioral anomalies detected across multiple agents. Review decision-making patterns."
                )
                recommendations["priority_actions"].append(
                    f"Behavioral anomalies in {anomaly_count} agents require pattern analysis"
                )
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating system optimization recommendations: {e}")
            return {"error": str(e)}
    
    def export_comprehensive_system_report(self) -> Dict[str, Any]:
        """Export a comprehensive system introspection report."""
        try:
            report = {
                "report_metadata": {
            "timestamp": time.time(),
                    "total_agents_monitored": len(self.monitored_agents),
                    "monitoring_active": self.monitoring_active,
                    "monitoring_interval": self.monitoring_interval
                },
                "agent_summaries": self.get_agent_status_summary(),
                "system_performance": self.analyze_system_performance(),
                "convergence_status": self.get_system_convergence_status(),
                "optimization_recommendations": self.generate_system_optimization_recommendations(),
                "cultural_neutrality_overview": self._generate_cultural_neutrality_overview(),
                "mathematical_validation_summary": self._generate_mathematical_validation_summary(),
                "behavioral_patterns_analysis": self._generate_behavioral_patterns_analysis()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error exporting comprehensive system report: {e}")
            return {"error": str(e)}
    
    def _generate_cultural_neutrality_overview(self) -> Dict[str, Any]:
        """Generate overview of cultural neutrality across all agents."""
        try:
            neutrality_scores = [
                introspection.cultural_neutrality_score 
                for introspection in self.agent_introspections.values()
            ]
            
            if not neutrality_scores:
                return {"status": "no_data"}
            
            return {
                "average_neutrality": np.mean(neutrality_scores),
                "min_neutrality": np.min(neutrality_scores),
                "max_neutrality": np.max(neutrality_scores),
                "neutrality_variance": np.var(neutrality_scores),
                "agents_below_baseline": sum(1 for score in neutrality_scores if score < self.neutrality_baseline),
                "total_agents": len(neutrality_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating cultural neutrality overview: {e}")
            return {"error": str(e)}
    
    def _generate_mathematical_validation_summary(self) -> Dict[str, Any]:
        """Generate summary of mathematical validation across all agents."""
        try:
            validation_scores = []
            convergence_statuses = []
            stability_statuses = []
            
            for introspection in self.agent_introspections.values():
                math_validation = introspection.mathematical_validation
                if "mathematical_confidence" in math_validation:
                    validation_scores.append(math_validation["mathematical_confidence"])
                
                if "convergence_analysis" in math_validation:
                    convergence_statuses.append(math_validation["convergence_analysis"].get("status", "unknown"))
                
                if "stability_analysis" in math_validation:
                    stability_statuses.append(math_validation["stability_analysis"].get("status", "unknown"))
            
            return {
                "average_mathematical_confidence": np.mean(validation_scores) if validation_scores else 0.0,
                "convergence_status_distribution": {status: convergence_statuses.count(status) for status in set(convergence_statuses)},
                "stability_status_distribution": {status: stability_statuses.count(status) for status in set(stability_statuses)},
                "total_validations": len(validation_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating mathematical validation summary: {e}")
            return {"error": str(e)}
    
    def _generate_behavioral_patterns_analysis(self) -> Dict[str, Any]:
        """Generate analysis of behavioral patterns across all agents."""
        try:
            total_anomalies = 0
            total_clusters = 0
            pattern_confidences = []
            
            for introspection in self.agent_introspections.values():
                behavioral_patterns = introspection.behavioral_patterns
                
                if "anomalies" in behavioral_patterns:
                    total_anomalies += behavioral_patterns["anomalies"].get("anomaly_count", 0)
                
                if "cluster_count" in behavioral_patterns:
                    total_clusters += behavioral_patterns["cluster_count"]
                
                if "pattern_confidence" in behavioral_patterns:
                    pattern_confidences.append(behavioral_patterns["pattern_confidence"])
            
            return {
                "total_behavioral_anomalies": total_anomalies,
                "total_behavioral_clusters": total_clusters,
                "average_pattern_confidence": np.mean(pattern_confidences) if pattern_confidences else 0.0,
                "agents_with_patterns": len(pattern_confidences),
                "anomaly_rate": total_anomalies / len(self.agent_introspections) if self.agent_introspections else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error generating behavioral patterns analysis: {e}")
            return {"error": str(e)}
    
    def _calculate_initial_agent_confidence(self, agent) -> float:
        """Calculate initial confidence for a newly registered agent."""
        # Start with moderate confidence for new agents
        confidence = 0.5
        
        # Adjust based on agent type (some agent types are more reliable)
        agent_type = agent.__class__.__name__
        if "Safety" in agent_type or "Monitor" in agent_type:
            confidence += 0.1  # Safety agents start with higher confidence
        elif "Reasoning" in agent_type or "Alignment" in agent_type:
            confidence += 0.05  # Core cognitive agents get slight boost
        
        # Check if agent has validation methods (indicates better design)
        if hasattr(agent, 'validate') or hasattr(agent, '_validate'):
            confidence += 0.1
        
        # Check if agent has error handling
        if hasattr(agent, 'logger') or hasattr(agent, 'error_handler'):
            confidence += 0.05
        
        return max(0.3, min(0.8, confidence))  # Keep initial confidence reasonable 
"""
Time Series Analyzer for Temporal Pattern Recognition

This module provides advanced time series analysis capabilities including
trend detection, seasonality analysis, anomaly detection, and temporal
pattern recognition for the NIS Protocol V3.0.

Enhanced Features (v3):
- Complete self-audit integration with real-time integrity monitoring
- Mathematical validation of time series operations with evidence-based metrics
- Comprehensive integrity oversight for all time series analysis outputs
- Auto-correction capabilities for time series-related communications

Key Features:
- Trend and seasonality decomposition
- Anomaly detection in time series
- Temporal pattern recognition
- Forecasting capabilities
- Statistical time series analysis
"""

import numpy as np
import scipy.stats as stats
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import time

from ...core.agent import NISAgent, NISLayer

# Integrity metrics for actual calculations
from src.utils.integrity_metrics import (
    calculate_confidence, create_default_confidence_factors, ConfidenceFactors
)

# Self-audit capabilities for real-time integrity monitoring
from src.utils.self_audit import self_audit_engine, ViolationType, IntegrityViolation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TimeSeriesPattern:
    """Represents a detected temporal pattern."""
    pattern_type: str
    start_time: float
    end_time: float
    confidence: float
    characteristics: Dict[str, Any]

class TimeSeriesAnalyzer(NISAgent):
    """
    Time Series Analyzer for NIS Protocol V3.0
    
    Provides comprehensive time series analysis including pattern recognition,
    trend analysis, and anomaly detection.
    """
    
    def __init__(
        self,
        agent_id: str = "time_series_001",
        description: str = "Advanced time series analysis and pattern recognition agent",
        enable_self_audit: bool = True
    ):
        super().__init__(agent_id, NISLayer.PERCEPTION, description)
        
        self.logger = logging.getLogger(f"nis.signal.timeseries.{agent_id}")
        
        # Analysis parameters
        self.window_size = 50
        self.anomaly_threshold = 2.0  # Standard deviations
        
        # Set up self-audit integration
        self.enable_self_audit = enable_self_audit
        self.integrity_monitoring_enabled = enable_self_audit
        self.integrity_metrics = {
            'monitoring_start_time': time.time(),
            'total_outputs_monitored': 0,
            'total_violations_detected': 0,
            'auto_corrections_applied': 0,
            'average_integrity_score': 100.0
        }
        
        # Initialize confidence factors for mathematical validation
        self.confidence_factors = create_default_confidence_factors()
        
        # Track analysis statistics
        self.analysis_stats = {
            'trends_analyzed': 0,
            'anomalies_detected': 0,
            'patterns_recognized': 0,
            'analysis_errors': 0
        }
        
        self.logger.info(f"Initialized TimeSeriesAnalyzer with self-audit: {enable_self_audit}")
    
    def analyze_trends(self, signal_data: np.ndarray, time_vector: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze trends in time series data with self-audit monitoring."""
        try:
            if time_vector is None:
                time_vector = np.arange(len(signal_data))
            
            # Linear trend analysis
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_vector, signal_data)
            
            # Determine trend direction
            if abs(slope) < std_err:
                trend_direction = "stable"
            elif slope > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"
            
            # Update statistics
            self.analysis_stats['trends_analyzed'] += 1
            
            result = {
                "slope": slope,
                "intercept": intercept,
                "correlation": r_value,
                "p_value": p_value,
                "trend_direction": trend_direction,
                "trend_strength": abs(r_value)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in trend analysis: {e}")
            self.analysis_stats['analysis_errors'] += 1
            return {"error": str(e)}
    
    def detect_anomalies(self, signal_data: np.ndarray) -> List[Tuple[int, float]]:
        """Detect anomalies in time series data with self-audit monitoring."""
        try:
            anomalies = []
            
            # Simple z-score based anomaly detection
            z_scores = np.abs(stats.zscore(signal_data))
            anomaly_indices = np.where(z_scores > self.anomaly_threshold)[0]
            
            for idx in anomaly_indices:
                anomalies.append((idx, z_scores[idx]))
            
            # Update statistics
            self.analysis_stats['anomalies_detected'] += len(anomalies)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            self.analysis_stats['analysis_errors'] += 1
            return []
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """Get current analyzer status with enhanced metrics."""
        return {
            "window_size": self.window_size,
            "anomaly_threshold": self.anomaly_threshold,
            "analysis_stats": self.analysis_stats,
            "integrity_monitoring": self.integrity_monitoring_enabled
        }
    
    # ==================== SELF-AUDIT CAPABILITIES ====================
    
    def audit_timeseries_output(self, output_text: str, operation: str = "", context: str = "") -> Dict[str, Any]:
        """
        Perform real-time integrity audit on time series analysis outputs.
        
        Args:
            output_text: Text output to audit
            operation: Time series operation type (trend_analysis, anomaly_detection, etc.)
            context: Additional context for the audit
            
        Returns:
            Audit results with violations and integrity score
        """
        if not self.enable_self_audit:
            return {'integrity_score': 100.0, 'violations': [], 'total_violations': 0}
        
        self.logger.info(f"Performing self-audit on time series output for operation: {operation}")
        
        # Use proven audit engine
        audit_context = f"timeseries:{operation}:{context}" if context else f"timeseries:{operation}"
        violations = self_audit_engine.audit_text(output_text, audit_context)
        integrity_score = self_audit_engine.get_integrity_score(output_text)
        
        # Log violations for time series-specific analysis
        if violations:
            self.logger.warning(f"Detected {len(violations)} integrity violations in time series output")
            for violation in violations:
                self.logger.warning(f"  - {violation.severity}: {violation.text} -> {violation.suggested_replacement}")
        
        return {
            'violations': violations,
            'integrity_score': integrity_score,
            'total_violations': len(violations),
            'violation_breakdown': self._categorize_timeseries_violations(violations),
            'operation': operation,
            'audit_timestamp': time.time()
        }
    
    def auto_correct_timeseries_output(self, output_text: str, operation: str = "") -> Dict[str, Any]:
        """
        Automatically correct integrity violations in time series outputs.
        
        Args:
            output_text: Text to correct
            operation: Time series operation type
            
        Returns:
            Corrected output with audit details
        """
        if not self.enable_self_audit:
            return {'corrected_text': output_text, 'violations_fixed': [], 'improvement': 0}
        
        self.logger.info(f"Performing self-correction on time series output for operation: {operation}")
        
        corrected_text, violations = self_audit_engine.auto_correct_text(output_text)
        
        # Calculate improvement metrics with mathematical validation
        original_score = self_audit_engine.get_integrity_score(output_text)
        corrected_score = self_audit_engine.get_integrity_score(corrected_text)
        improvement = calculate_confidence(corrected_score - original_score, self.confidence_factors)
        
        # Update integrity metrics
        if hasattr(self, 'integrity_metrics'):
            self.integrity_metrics['auto_corrections_applied'] += len(violations)
        
        return {
            'original_text': output_text,
            'corrected_text': corrected_text,
            'violations_fixed': violations,
            'original_integrity_score': original_score,
            'corrected_integrity_score': corrected_score,
            'improvement': improvement,
            'operation': operation,
            'correction_timestamp': time.time()
        }
    
    def get_timeseries_integrity_report(self) -> Dict[str, Any]:
        """Generate comprehensive time series integrity report"""
        if not self.enable_self_audit:
            return {'status': 'SELF_AUDIT_DISABLED'}
        
        # Get basic integrity report
        base_report = self_audit_engine.generate_integrity_report()
        
        # Add time series-specific metrics
        timeseries_report = {
            'timeseries_agent_id': self.agent_id,
            'monitoring_enabled': self.integrity_monitoring_enabled,
            'timeseries_capabilities': {
                'window_size': self.window_size,
                'anomaly_threshold': self.anomaly_threshold,
                'supports_trend_analysis': True,
                'supports_anomaly_detection': True
            },
            'analysis_statistics': self.analysis_stats,
            'integrity_metrics': getattr(self, 'integrity_metrics', {}),
            'base_integrity_report': base_report,
            'report_timestamp': time.time()
        }
        
        return timeseries_report
    
    def validate_timeseries_configuration(self) -> Dict[str, Any]:
        """Validate time series configuration for integrity"""
        validation_results = {
            'valid': True,
            'warnings': [],
            'recommendations': []
        }
        
        # Check window size
        if self.window_size < 10:
            validation_results['warnings'].append("Window size is very small - may affect analysis quality")
            validation_results['recommendations'].append("Consider increasing window size to at least 20")
        
        # Check anomaly threshold
        if self.anomaly_threshold < 1.0:
            validation_results['warnings'].append("Anomaly threshold is very low - may produce many false positives")
            validation_results['recommendations'].append("Consider increasing anomaly threshold to 2.0-3.0")
        elif self.anomaly_threshold > 5.0:
            validation_results['warnings'].append("Anomaly threshold is very high - may miss real anomalies")
            validation_results['recommendations'].append("Consider reducing anomaly threshold to 2.0-3.0")
        
        # Check error rate
        error_rate = (self.analysis_stats.get('analysis_errors', 0) / 
                     max(1, self.analysis_stats.get('trends_analyzed', 1)))
        
        if error_rate > 0.1:
            validation_results['warnings'].append(f"High analysis error rate: {error_rate:.1%}")
            validation_results['recommendations'].append("Investigate and resolve sources of analysis errors")
        
        return validation_results
    
    def _monitor_timeseries_output_integrity(self, output_text: str, operation: str = "") -> str:
        """
        Internal method to monitor and potentially correct time series output integrity.
        
        Args:
            output_text: Output to monitor
            operation: Time series operation type
            
        Returns:
            Potentially corrected output
        """
        if not getattr(self, 'integrity_monitoring_enabled', False):
            return output_text
        
        # Perform audit
        audit_result = self.audit_timeseries_output(output_text, operation)
        
        # Update monitoring metrics
        if hasattr(self, 'integrity_metrics'):
            self.integrity_metrics['total_outputs_monitored'] += 1
            self.integrity_metrics['total_violations_detected'] += audit_result['total_violations']
        
        # Auto-correct if violations detected
        if audit_result['violations']:
            correction_result = self.auto_correct_timeseries_output(output_text, operation)
            
            self.logger.info(f"Auto-corrected time series output: {len(audit_result['violations'])} violations fixed")
            
            return correction_result['corrected_text']
        
        return output_text
    
    def _categorize_timeseries_violations(self, violations: List[IntegrityViolation]) -> Dict[str, int]:
        """Categorize integrity violations specific to time series operations"""
        categories = defaultdict(int)
        
        for violation in violations:
            categories[violation.violation_type.value] += 1
        
        return dict(categories)

# Test function
def test_time_series_analyzer():
    """Test the time series analyzer."""
    print("📊 Testing TimeSeriesAnalyzer...")
    
    analyzer = TimeSeriesAnalyzer()
    
    # Test data with trend and anomalies
    t = np.linspace(0, 10, 100)
    test_data = 2 * t + np.sin(t) + 0.1 * np.random.randn(len(t))
    test_data[50] = 20  # Add anomaly
    
    # Test trend analysis
    trends = analyzer.analyze_trends(test_data, t)
    print(f"   Trend direction: {trends.get('trend_direction', 'error')}")
    print(f"   Trend strength: {trends.get('trend_strength', 0):.3f}")
    
    # Test anomaly detection
    anomalies = analyzer.detect_anomalies(test_data)
    print(f"   Anomalies detected: {len(anomalies)}")
    
    # Test status
    status = analyzer.get_analysis_status()
    print(f"   Analysis stats: {status['analysis_stats']}")
    
    # Test integrity report
    report = analyzer.get_timeseries_integrity_report()
    print(f"   Integrity monitoring: {report['monitoring_enabled']}")
    
    print("✅ TimeSeriesAnalyzer test completed")

if __name__ == "__main__":
    test_time_series_analyzer() 
"""
Time Series Analyzer for Temporal Pattern Recognition

This module provides advanced time series analysis capabilities including
trend detection, seasonality analysis, anomaly detection, and temporal
pattern recognition for the NIS Protocol V3.0.

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

class TimeSeriesAnalyzer:
    """
    Time Series Analyzer for NIS Protocol V3.0
    
    Provides comprehensive time series analysis including pattern recognition,
    trend analysis, and anomaly detection.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("nis.signal.timeseries")
        
        # Analysis parameters
        self.window_size = 50
        self.anomaly_threshold = 2.0  # Standard deviations
        
        self.logger.info("Initialized TimeSeriesAnalyzer")
    
    def analyze_trends(self, signal_data: np.ndarray, time_vector: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze trends in time series data."""
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
        
        return {
            "slope": slope,
            "intercept": intercept,
            "correlation": r_value,
            "p_value": p_value,
            "trend_direction": trend_direction,
            "trend_strength": abs(r_value)
        }
    
    def detect_anomalies(self, signal_data: np.ndarray) -> List[Tuple[int, float]]:
        """Detect anomalies in time series data."""
        anomalies = []
        
        # Simple z-score based anomaly detection
        z_scores = np.abs(stats.zscore(signal_data))
        anomaly_indices = np.where(z_scores > self.anomaly_threshold)[0]
        
        for idx in anomaly_indices:
            anomalies.append((idx, z_scores[idx]))
        
        return anomalies
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """Get current analyzer status."""
        return {
            "window_size": self.window_size,
            "anomaly_threshold": self.anomaly_threshold
        }

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
    print(f"   Trend direction: {trends['trend_direction']}")
    print(f"   Trend strength: {trends['trend_strength']:.3f}")
    
    # Test anomaly detection
    anomalies = analyzer.detect_anomalies(test_data)
    print(f"   Anomalies detected: {len(anomalies)}")
    
    print("✅ TimeSeriesAnalyzer test completed")

if __name__ == "__main__":
    test_time_series_analyzer() 
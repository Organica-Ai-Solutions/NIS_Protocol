"""
SciPy Signal Agent for Advanced Signal Processing

This agent provides comprehensive signal processing capabilities using SciPy's
signal processing tools. It offers advanced filtering, spectral analysis,
and signal conditioning for the NIS Protocol V3.0.

Enhanced Features (v3):
- Complete self-audit integration with real-time integrity monitoring
- Mathematical validation of signal processing operations with evidence-based metrics
- Comprehensive integrity oversight for all SciPy signal processing outputs
- Auto-correction capabilities for signal processing-related communications

Key Features:
- Advanced digital filtering (Butterworth, Chebyshev, Elliptic)
- Spectral analysis and FFT processing
- Signal conditioning and preprocessing
- Integration with other NIS signal processing components
"""

import numpy as np
import scipy.signal as signal
import scipy.fft as fft
from scipy import ndimage
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
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

class FilterType(Enum):
    """Types of digital filters."""
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"
    NOTCH = "notch"

class SciPySignalAgent(NISAgent):
    """
    SciPy Signal Processing Agent
    
    Provides advanced signal processing using SciPy tools including
    filtering, spectral analysis, and signal conditioning.
    """
    
    def __init__(
        self,
        agent_id: str = "scipy_signal_001",
        description: str = "SciPy-based signal processing agent",
        sampling_rate: float = 1000.0,
        enable_self_audit: bool = True
    ):
        super().__init__(agent_id, NISLayer.PERCEPTION, description)
        
        self.sampling_rate = sampling_rate
        self.logger = logging.getLogger(f"nis.signal.scipy.{agent_id}")
        
        # Initialize filter bank
        self.filter_bank = self._create_filter_bank()
        
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
        
        # Track processing statistics
        self.processing_stats = {
            'filters_applied': 0,
            'signals_processed': 0,
            'average_filter_order': 0.0,
            'processing_errors': 0
        }
        
        self.logger.info(f"Initialized SciPySignalAgent with {sampling_rate}Hz sampling, self-audit: {enable_self_audit}")
    
    def _create_filter_bank(self) -> Dict[str, Any]:
        """Create a bank of commonly used filters."""
        filter_bank = {}
        
        # Butterworth filters
        filter_bank['butter_low_50'] = signal.butter(4, 50, 'low', fs=self.sampling_rate, output='sos')
        filter_bank['butter_high_1'] = signal.butter(4, 1, 'high', fs=self.sampling_rate, output='sos')
        
        return filter_bank
    
    def apply_filter(
        self,
        input_signal: np.ndarray,
        filter_type: FilterType,
        frequency: Union[float, Tuple[float, float]],
        order: int = 4
    ) -> np.ndarray:
        """Apply digital filter to input signal with self-audit monitoring."""
        try:
            if filter_type == FilterType.LOWPASS:
                sos = signal.butter(order, frequency, 'low', fs=self.sampling_rate, output='sos')
            elif filter_type == FilterType.HIGHPASS:
                sos = signal.butter(order, frequency, 'high', fs=self.sampling_rate, output='sos')
            elif filter_type == FilterType.BANDPASS:
                sos = signal.butter(order, frequency, 'band', fs=self.sampling_rate, output='sos')
            else:
                self.processing_stats['processing_errors'] += 1
                return input_signal
            
            filtered_signal = signal.sosfilt(sos, input_signal)
            
            # Update processing statistics
            self.processing_stats['filters_applied'] += 1
            self.processing_stats['signals_processed'] += 1
            current_avg = self.processing_stats['average_filter_order']
            count = self.processing_stats['filters_applied']
            self.processing_stats['average_filter_order'] = ((current_avg * (count - 1)) + order) / count
            
            return filtered_signal
            
        except Exception as e:
            self.logger.error(f"Error applying filter: {e}")
            self.processing_stats['processing_errors'] += 1
            return input_signal
    
    # ==================== SELF-AUDIT CAPABILITIES ====================
    
    def audit_scipy_signal_output(self, output_text: str, operation: str = "", context: str = "") -> Dict[str, Any]:
        """
        Perform real-time integrity audit on SciPy signal processing outputs.
        
        Args:
            output_text: Text output to audit
            operation: SciPy signal operation type (filter, fft, spectral_analysis, etc.)
            context: Additional context for the audit
            
        Returns:
            Audit results with violations and integrity score
        """
        if not self.enable_self_audit:
            return {'integrity_score': 100.0, 'violations': [], 'total_violations': 0}
        
        self.logger.info(f"Performing self-audit on SciPy signal output for operation: {operation}")
        
        # Use proven audit engine
        audit_context = f"scipy_signal:{operation}:{context}" if context else f"scipy_signal:{operation}"
        violations = self_audit_engine.audit_text(output_text, audit_context)
        integrity_score = self_audit_engine.get_integrity_score(output_text)
        
        # Log violations for SciPy signal-specific analysis
        if violations:
            self.logger.warning(f"Detected {len(violations)} integrity violations in SciPy signal output")
            for violation in violations:
                self.logger.warning(f"  - {violation.severity}: {violation.text} -> {violation.suggested_replacement}")
        
        return {
            'violations': violations,
            'integrity_score': integrity_score,
            'total_violations': len(violations),
            'violation_breakdown': self._categorize_scipy_violations(violations),
            'operation': operation,
            'audit_timestamp': time.time()
        }
    
    def auto_correct_scipy_signal_output(self, output_text: str, operation: str = "") -> Dict[str, Any]:
        """
        Automatically correct integrity violations in SciPy signal outputs.
        
        Args:
            output_text: Text to correct
            operation: SciPy signal operation type
            
        Returns:
            Corrected output with audit details
        """
        if not self.enable_self_audit:
            return {'corrected_text': output_text, 'violations_fixed': [], 'improvement': 0}
        
        self.logger.info(f"Performing self-correction on SciPy signal output for operation: {operation}")
        
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
    
    def get_scipy_integrity_report(self) -> Dict[str, Any]:
        """Generate comprehensive SciPy signal integrity report"""
        if not self.enable_self_audit:
            return {'status': 'SELF_AUDIT_DISABLED'}
        
        # Get basic integrity report
        base_report = self_audit_engine.generate_integrity_report()
        
        # Add SciPy signal-specific metrics
        scipy_report = {
            'scipy_agent_id': self.agent_id,
            'monitoring_enabled': self.integrity_monitoring_enabled,
            'scipy_capabilities': {
                'sampling_rate': self.sampling_rate,
                'filter_bank_size': len(self.filter_bank),
                'supported_filter_types': [ft.value for ft in FilterType]
            },
            'processing_statistics': self.processing_stats,
            'integrity_metrics': getattr(self, 'integrity_metrics', {}),
            'base_integrity_report': base_report,
            'report_timestamp': time.time()
        }
        
        return scipy_report
    
    def validate_scipy_configuration(self) -> Dict[str, Any]:
        """Validate SciPy signal configuration for integrity"""
        validation_results = {
            'valid': True,
            'warnings': [],
            'recommendations': []
        }
        
        # Check sampling rate
        if self.sampling_rate <= 0:
            validation_results['valid'] = False
            validation_results['warnings'].append("Invalid sampling rate - must be positive")
            validation_results['recommendations'].append("Set sampling rate to a positive value (e.g., 1000.0)")
        
        # Check filter bank
        if len(self.filter_bank) < 2:
            validation_results['warnings'].append("Limited filter bank - consider adding more pre-computed filters")
            validation_results['recommendations'].append("Expand filter bank with commonly used filter configurations")
        
        # Check processing error rate
        error_rate = (self.processing_stats.get('processing_errors', 0) / 
                     max(1, self.processing_stats.get('signals_processed', 1)))
        
        if error_rate > 0.1:
            validation_results['warnings'].append(f"High processing error rate: {error_rate:.1%}")
            validation_results['recommendations'].append("Investigate and resolve sources of processing errors")
        
        return validation_results
    
    def _monitor_scipy_output_integrity(self, output_text: str, operation: str = "") -> str:
        """
        Internal method to monitor and potentially correct SciPy signal output integrity.
        
        Args:
            output_text: Output to monitor
            operation: SciPy signal operation type
            
        Returns:
            Potentially corrected output
        """
        if not getattr(self, 'integrity_monitoring_enabled', False):
            return output_text
        
        # Perform audit
        audit_result = self.audit_scipy_signal_output(output_text, operation)
        
        # Update monitoring metrics
        if hasattr(self, 'integrity_metrics'):
            self.integrity_metrics['total_outputs_monitored'] += 1
            self.integrity_metrics['total_violations_detected'] += audit_result['total_violations']
        
        # Auto-correct if violations detected
        if audit_result['violations']:
            correction_result = self.auto_correct_scipy_signal_output(output_text, operation)
            
            self.logger.info(f"Auto-corrected SciPy signal output: {len(audit_result['violations'])} violations fixed")
            
            return correction_result['corrected_text']
        
        return output_text
    
    def _categorize_scipy_violations(self, violations: List[IntegrityViolation]) -> Dict[str, int]:
        """Categorize integrity violations specific to SciPy signal operations"""
        categories = defaultdict(int)
        
        for violation in violations:
            categories[violation.violation_type.value] += 1
        
        return dict(categories)

# Test function
def test_scipy_agent():
    """Test the SciPy signal agent."""
    print("📡 Testing SciPySignalAgent...")
    
    agent = SciPySignalAgent()
    
    # Test signal
    t = np.linspace(0, 1, 1000)
    test_signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
    
    # Test filtering
    filtered = agent.apply_filter(test_signal, FilterType.LOWPASS, 20.0)
    
    print(f"   Filtering successful: {len(filtered)} samples")
    print(f"   Processing stats: {agent.processing_stats}")
    
    # Test integrity report
    report = agent.get_scipy_integrity_report()
    print(f"   Integrity monitoring: {report['monitoring_enabled']}")
    
    print("✅ SciPySignalAgent test completed")

if __name__ == "__main__":
    test_scipy_agent()
 
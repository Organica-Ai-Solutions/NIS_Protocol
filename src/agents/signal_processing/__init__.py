"""
Signal Agents Module - V3.0 Signal Processing Intelligence

This module implements signal processing agents that provide comprehensive signal analysis,
time-series compression, frequency domain processing, and Laplace transform capabilities
for the NIS Protocol V3.0 template framework.

Key Components:
- SignalProcessingAgent: comprehensive signal analysis and processing
- LaplaceSignalProcessor: Laplace transform integration for signal compression
- SciPySignalAgent: SciPy-based signal filtering and analysis
- TimeSeriesAnalyzer: Temporal pattern recognition and analysis
- FrequencyDomainProcessor: Fourier and spectral analysis
"""

from .unified_signal_agent import SignalProcessingAgent, LaplaceSignalProcessor, SciPySignalAgent, TimeSeriesAnalyzer

__all__ = [
    'SignalProcessingAgent',
    'LaplaceSignalProcessor', 
    'SciPySignalAgent',
    'TimeSeriesAnalyzer'
] 
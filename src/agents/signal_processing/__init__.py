"""
Signal Agents Module - V3.0 Signal Processing Intelligence

This module implements signal processing agents that provide advanced signal analysis,
time-series compression, frequency domain processing, and Laplace transform capabilities
for the NIS Protocol V3.0 template framework.

Key Components:
- SignalProcessingAgent: Advanced signal analysis and processing
- LaplaceSignalProcessor: Laplace transform integration for signal compression
- SciPySignalAgent: SciPy-based signal filtering and analysis
- TimeSeriesAnalyzer: Temporal pattern recognition and analysis
- FrequencyDomainProcessor: Fourier and spectral analysis
"""

from .signal_agent import SignalProcessingAgent
from .laplace_processor import LaplaceSignalProcessor
from .scipy_signal_agent import SciPySignalAgent
from .time_series_analyzer import TimeSeriesAnalyzer

__all__ = [
    'SignalProcessingAgent',
    'LaplaceSignalProcessor', 
    'SciPySignalAgent',
    'TimeSeriesAnalyzer'
] 
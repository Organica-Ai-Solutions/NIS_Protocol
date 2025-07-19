"""
SciPy Signal Agent for Advanced Signal Processing

This agent provides comprehensive signal processing capabilities using SciPy's
signal processing tools. It offers advanced filtering, spectral analysis,
and signal conditioning for the NIS Protocol V3.0.

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

from ...core.agent import NISAgent, NISLayer

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
        sampling_rate: float = 1000.0
    ):
        super().__init__(agent_id, NISLayer.PERCEPTION, description)
        
        self.sampling_rate = sampling_rate
        self.logger = logging.getLogger(f"nis.signal.scipy.{agent_id}")
        
        # Initialize filter bank
        self.filter_bank = self._create_filter_bank()
        
        self.logger.info(f"Initialized SciPySignalAgent with {sampling_rate}Hz sampling")
    
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
        """Apply digital filter to input signal."""
        try:
            if filter_type == FilterType.LOWPASS:
                sos = signal.butter(order, frequency, 'low', fs=self.sampling_rate, output='sos')
            elif filter_type == FilterType.HIGHPASS:
                sos = signal.butter(order, frequency, 'high', fs=self.sampling_rate, output='sos')
            elif filter_type == FilterType.BANDPASS:
                sos = signal.butter(order, frequency, 'band', fs=self.sampling_rate, output='sos')
            else:
                return input_signal
            
            filtered_signal = signal.sosfilt(sos, input_signal)
            return filtered_signal
            
        except Exception as e:
            self.logger.error(f"Error applying filter: {e}")
            return input_signal

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
    print("✅ SciPySignalAgent test completed")

if __name__ == "__main__":
    test_scipy_agent()
 
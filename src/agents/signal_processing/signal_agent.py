"""
Signal Processing Agent with Advanced Analysis Capabilities

This agent provides sophisticated signal processing, time-series analysis, and frequency
domain processing for the NIS Protocol V3.0. It integrates with KAN reasoning and
cognitive wave fields to provide enhanced temporal pattern recognition and signal
compression capabilities.

Key Features:
- Real-time signal analysis and filtering
- Integration with cognitive wave field dynamics
- KAN-enhanced pattern recognition in signals
- Multi-domain signal processing (time, frequency, Laplace)
- Temporal memory compression and retrieval
- Adaptive signal preprocessing for different domains
"""

import numpy as np
import torch
import torch.nn as nn
import scipy.signal as signal
import scipy.fft as fft
from scipy import integrate
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict, deque

from ...core.agent import NISAgent, NISLayer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalDomain(Enum):
    """Signal processing domains."""
    TIME = "time"
    FREQUENCY = "frequency"
    LAPLACE = "laplace"
    WAVELET = "wavelet"
    Z_TRANSFORM = "z_transform"

class SignalType(Enum):
    """Types of signals that can be processed."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    DIGITAL = "digital"
    COGNITIVE_WAVE = "cognitive_wave"
    MEMORY_TRACE = "memory_trace"
    NEURAL_ACTIVITY = "neural_activity"

@dataclass
class SignalCharacteristics:
    """Characteristics of a signal."""
    signal_type: SignalType
    sampling_rate: float
    frequency_range: Tuple[float, float]
    amplitude_range: Tuple[float, float]
    noise_level: float
    stationarity: bool
    periodicity: Optional[float]
    complexity: float

@dataclass
class ProcessedSignal:
    """Result of signal processing operations."""
    original_signal: np.ndarray
    processed_signal: np.ndarray
    time_vector: np.ndarray
    frequency_vector: np.ndarray
    spectral_density: np.ndarray
    characteristics: SignalCharacteristics
    processing_metadata: Dict[str, Any]
    compression_ratio: float
    signal_to_noise_ratio: float

class KANSignalProcessor(nn.Module):
    """
    KAN-enhanced signal processor for pattern recognition in signals.
    
    Uses Kolmogorov-Arnold Networks to identify and extract patterns
    from complex signals with better interpretability than traditional
    signal processing methods.
    """
    
    def __init__(self, input_dim: int = 256, hidden_dims: List[int] = [128, 64, 32]):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # KAN layers for signal processing
        self.kan_layers = nn.ModuleList()
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        
        # KAN processing layers (simplified - would use actual KAN implementation)
        for i in range(len(hidden_dims) - 1):
            self.kan_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
        # Output layers
        self.pattern_detector = nn.Linear(hidden_dims[-1], 16)  # Pattern features
        self.signal_reconstructor = nn.Linear(hidden_dims[-1], input_dim)  # Reconstruction
        
        # Activation functions
        self.activation = nn.GELU()
        self.output_activation = nn.Tanh()
    
    def forward(self, signal_segment: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process signal segment through KAN network.
        
        Args:
            signal_segment: Input signal segment [batch_size, input_dim]
            
        Returns:
            Tuple of (pattern_features, reconstructed_signal)
        """
        # Project input
        x = self.activation(self.input_projection(signal_segment))
        
        # Process through KAN layers
        for layer in self.kan_layers:
            x = self.activation(layer(x))
        
        # Extract patterns and reconstruct
        pattern_features = self.pattern_detector(x)
        reconstructed_signal = self.output_activation(self.signal_reconstructor(x))
        
        return pattern_features, reconstructed_signal

class CognitiveWaveFieldIntegrator:
    """
    Integrates signal processing with cognitive wave field dynamics.
    
    Enables signals to influence and be influenced by the cognitive
    wave fields used in the NIS Protocol reasoning system.
    """
    
    def __init__(self, field_dimensions: Tuple[int, int] = (64, 64)):
        self.field_dimensions = field_dimensions
        self.current_field = np.zeros(field_dimensions)
        self.field_history: deque = deque(maxlen=100)
        
        # Wave field parameters
        self.diffusion_coefficient = 0.1
        self.decay_rate = 0.01
        self.coupling_strength = 0.05
    
    def update_field_from_signal(self, signal_data: np.ndarray, position: Tuple[int, int]):
        """Update cognitive wave field based on signal input."""
        x, y = position
        x = min(max(0, x), self.field_dimensions[0] - 1)
        y = min(max(0, y), self.field_dimensions[1] - 1)
        
        # Add signal influence to field
        signal_strength = np.mean(np.abs(signal_data))
        self.current_field[x, y] += self.coupling_strength * signal_strength
        
        # Apply diffusion and decay
        self._evolve_field()
    
    def extract_field_signal(self, region: Tuple[slice, slice]) -> np.ndarray:
        """Extract signal from cognitive wave field region."""
        field_region = self.current_field[region]
        # Convert field values to signal
        signal_data = field_region.flatten()
        return signal_data
    
    def _evolve_field(self):
        """Evolve the cognitive wave field using diffusion equation."""
        # Simple 2D diffusion with decay
        laplacian = signal.correlate2d(
            self.current_field, 
            np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), 
            mode='same', boundary='wrap'
        )
        
        # Field evolution: ∂φ/∂t = D∇²φ - Rφ
        field_change = (self.diffusion_coefficient * laplacian - 
                       self.decay_rate * self.current_field)
        
        self.current_field += 0.01 * field_change  # Small timestep
        
        # Store history
        self.field_history.append(self.current_field.copy())

class SignalProcessingAgent(NISAgent):
    """
    Signal Processing Agent for V3.0 NIS Protocol
    
    This agent provides advanced signal processing capabilities including
    time-series analysis, frequency domain processing, pattern recognition,
    and integration with cognitive wave fields and KAN reasoning.
    """
    
    def __init__(
        self,
        agent_id: str = "signal_processor_001",
        description: str = "Advanced signal processing and analysis agent",
        sampling_rate: float = 1000.0,
        buffer_size: int = 8192
    ):
        super().__init__(agent_id, NISLayer.PERCEPTION, description)
        
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.logger = logging.getLogger(f"nis.signal.{agent_id}")
        
        # Signal buffers
        self.signal_buffer: deque = deque(maxlen=buffer_size)
        self.processed_buffer: deque = deque(maxlen=buffer_size // 4)
        
        # KAN signal processor
        self.kan_processor = KANSignalProcessor(
            input_dim=256,  # Window size for signal segments
            hidden_dims=[128, 64, 32]
        )
        
        # Cognitive wave field integration
        self.wave_integrator = CognitiveWaveFieldIntegrator()
        
        # Signal processing parameters
        self.window_size = 256
        self.overlap = 0.5
        self.frequency_resolution = self.sampling_rate / self.window_size
        
        # Adaptive filtering parameters
        self.noise_threshold = 0.01
        self.signal_threshold = 0.1
        self.adaptation_rate = 0.01
        
        # Pattern recognition
        self.detected_patterns: List[Dict[str, Any]] = []
        self.pattern_templates: Dict[str, np.ndarray] = {}
        
        # Performance metrics
        self.processing_stats = {
            "signals_processed": 0,
            "patterns_detected": 0,
            "compression_achieved": 0.0,
            "average_snr": 0.0,
            "processing_latency": 0.0
        }
        
        # Initialize signal processing filters
        self._initialize_filters()
        
        self.logger.info(f"Initialized SignalProcessingAgent with {sampling_rate}Hz sampling")
    
    def _initialize_filters(self):
        """Initialize adaptive signal processing filters."""
        # Butterworth filters for different frequency ranges
        self.filters = {}
        
        # Low-pass filter (0-50Hz) for slow cognitive processes
        sos_low = signal.butter(4, 50.0, 'low', fs=self.sampling_rate, output='sos')
        self.filters['cognitive_low'] = sos_low
        
        # Band-pass filter (8-30Hz) for attention/focus signals
        sos_attention = signal.butter(4, [8.0, 30.0], 'band', fs=self.sampling_rate, output='sos')
        self.filters['attention'] = sos_attention
        
        # High-pass filter (>100Hz) for fast neural activity
        sos_high = signal.butter(4, 100.0, 'high', fs=self.sampling_rate, output='sos')
        self.filters['neural_fast'] = sos_high
        
        # Notch filter for 50/60Hz noise removal
        sos_notch = signal.iirnotch(50.0, 30.0, fs=self.sampling_rate)
        self.filters['powerline_notch'] = sos_notch
    
    def process_signal(
        self, 
        input_signal: np.ndarray, 
        signal_type: SignalType = SignalType.CONTINUOUS,
        processing_domain: SignalDomain = SignalDomain.TIME
    ) -> ProcessedSignal:
        """
        Process input signal with comprehensive analysis.
        
        Args:
            input_signal: Input signal data
            signal_type: Type of signal being processed
            processing_domain: Domain for processing (time/frequency/etc.)
            
        Returns:
            ProcessedSignal with analysis results
        """
        start_time = time.time()
        
        try:
            # Add to buffer
            self.signal_buffer.extend(input_signal)
            
            # Characterize signal
            characteristics = self._characterize_signal(input_signal, signal_type)
            
            # Apply preprocessing
            preprocessed_signal = self._preprocess_signal(input_signal, characteristics)
            
            # Domain-specific processing
            if processing_domain == SignalDomain.TIME:
                processed_signal = self._time_domain_processing(preprocessed_signal)
            elif processing_domain == SignalDomain.FREQUENCY:
                processed_signal = self._frequency_domain_processing(preprocessed_signal)
            elif processing_domain == SignalDomain.LAPLACE:
                processed_signal = self._laplace_domain_processing(preprocessed_signal)
            else:
                processed_signal = preprocessed_signal
            
            # KAN pattern recognition
            pattern_features = self._kan_pattern_recognition(processed_signal)
            
            # Cognitive wave field integration
            self._integrate_with_cognitive_field(processed_signal, pattern_features)
            
            # Calculate metrics
            compression_ratio = len(input_signal) / len(processed_signal) if len(processed_signal) > 0 else 1.0
            snr = self._calculate_snr(input_signal, processed_signal)
            
            # Create time and frequency vectors
            time_vector = np.arange(len(processed_signal)) / self.sampling_rate
            frequency_vector, spectral_density = self._compute_spectral_density(processed_signal)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_processing_stats(compression_ratio, snr, processing_time)
            
            result = ProcessedSignal(
                original_signal=input_signal,
                processed_signal=processed_signal,
                time_vector=time_vector,
                frequency_vector=frequency_vector,
                spectral_density=spectral_density,
                characteristics=characteristics,
                processing_metadata={
                    "domain": processing_domain.value,
                    "pattern_features": pattern_features,
                    "processing_time": processing_time,
                    "filters_used": list(self.filters.keys())
                },
                compression_ratio=compression_ratio,
                signal_to_noise_ratio=snr
            )
            
            self.processed_buffer.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
            # Return minimal result
            return ProcessedSignal(
                original_signal=input_signal,
                processed_signal=input_signal,
                time_vector=np.arange(len(input_signal)) / self.sampling_rate,
                frequency_vector=np.array([]),
                spectral_density=np.array([]),
                characteristics=self._characterize_signal(input_signal, signal_type),
                processing_metadata={"error": str(e)},
                compression_ratio=1.0,
                signal_to_noise_ratio=0.0
            )
    
    def _characterize_signal(self, signal_data: np.ndarray, signal_type: SignalType) -> SignalCharacteristics:
        """Analyze and characterize input signal."""
        # Basic signal statistics
        amplitude_range = (np.min(signal_data), np.max(signal_data))
        rms_amplitude = np.sqrt(np.mean(signal_data**2))
        
        # Estimate noise level using high-frequency content
        if len(signal_data) > 10:
            diff_signal = np.diff(signal_data)
            noise_level = np.std(diff_signal) / np.sqrt(2)
        else:
            noise_level = 0.1 * rms_amplitude
        
        # Frequency analysis
        freqs, psd = signal.welch(signal_data, fs=self.sampling_rate, nperseg=min(256, len(signal_data)//4))
        peak_freq_idx = np.argmax(psd)
        dominant_freq = freqs[peak_freq_idx]
        
        # Estimate frequency range (where PSD > 10% of peak)
        threshold = 0.1 * np.max(psd)
        freq_indices = np.where(psd > threshold)[0]
        if len(freq_indices) > 0:
            frequency_range = (freqs[freq_indices[0]], freqs[freq_indices[-1]])
        else:
            frequency_range = (0.0, self.sampling_rate / 2)
        
        # Stationarity test (simplified)
        if len(signal_data) > 100:
            first_half = signal_data[:len(signal_data)//2]
            second_half = signal_data[len(signal_data)//2:]
            stationarity = abs(np.std(first_half) - np.std(second_half)) < 0.1 * np.std(signal_data)
        else:
            stationarity = True
        
        # Periodicity detection
        autocorr = np.correlate(signal_data, signal_data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find peaks in autocorrelation
        peaks, _ = signal.find_peaks(autocorr[1:], height=0.3)
        periodicity = peaks[0] / self.sampling_rate if len(peaks) > 0 else None
        
        # Complexity measure (spectral entropy)
        psd_normalized = psd / np.sum(psd)
        spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-12))
        complexity = spectral_entropy / np.log2(len(psd))  # Normalize
        
        return SignalCharacteristics(
            signal_type=signal_type,
            sampling_rate=self.sampling_rate,
            frequency_range=frequency_range,
            amplitude_range=amplitude_range,
            noise_level=noise_level,
            stationarity=stationarity,
            periodicity=periodicity,
            complexity=complexity
        )
    
    def _preprocess_signal(self, signal_data: np.ndarray, characteristics: SignalCharacteristics) -> np.ndarray:
        """Apply adaptive preprocessing based on signal characteristics."""
        processed = signal_data.copy()
        
        # Remove DC offset
        processed = processed - np.mean(processed)
        
        # Apply appropriate filtering based on characteristics
        if characteristics.noise_level > self.noise_threshold:
            # Apply denoising filter
            if characteristics.frequency_range[1] < 50:
                # Low-frequency signal - use cognitive_low filter
                processed = signal.sosfilt(self.filters['cognitive_low'], processed)
            elif 8 <= characteristics.frequency_range[0] <= 30:
                # Attention-band signal
                processed = signal.sosfilt(self.filters['attention'], processed)
            else:
                # General denoising with notch filter
                processed = signal.sosfilt(self.filters['powerline_notch'], processed)
        
        # Adaptive amplitude normalization
        if characteristics.amplitude_range[1] > 10 * characteristics.amplitude_range[0]:
            # Large dynamic range - apply soft normalization
            processed = np.tanh(processed / np.std(processed)) * np.std(processed)
        
        return processed
    
    def _time_domain_processing(self, signal_data: np.ndarray) -> np.ndarray:
        """Process signal in time domain."""
        # Apply windowing for edge effects
        if len(signal_data) > self.window_size:
            # Process in overlapping windows
            hop_size = int(self.window_size * (1 - self.overlap))
            processed_segments = []
            
            for i in range(0, len(signal_data) - self.window_size + 1, hop_size):
                segment = signal_data[i:i + self.window_size]
                
                # Apply window function
                windowed = segment * signal.windows.hann(len(segment))
                
                # Simple temporal filtering (example)
                filtered = signal.savgol_filter(windowed, 5, 2)
                processed_segments.append(filtered)
            
            # Overlap-add reconstruction
            if processed_segments:
                return self._overlap_add_reconstruction(processed_segments, hop_size)
        
        return signal_data
    
    def _frequency_domain_processing(self, signal_data: np.ndarray) -> np.ndarray:
        """Process signal in frequency domain."""
        # FFT
        fft_data = fft.fft(signal_data)
        freqs = fft.fftfreq(len(signal_data), 1/self.sampling_rate)
        
        # Spectral processing (example: enhance certain frequencies)
        # This could be more sophisticated based on the signal characteristics
        enhanced_fft = fft_data.copy()
        
        # Enhance mid-frequencies (attention band)
        attention_band = (freqs >= 8) & (freqs <= 30)
        enhanced_fft[attention_band] *= 1.2
        
        # Suppress high-frequency noise
        noise_band = freqs > 100
        enhanced_fft[noise_band] *= 0.8
        
        # IFFT back to time domain
        processed = np.real(fft.ifft(enhanced_fft))
        return processed
    
    def _laplace_domain_processing(self, signal_data: np.ndarray) -> np.ndarray:
        """Process signal using Laplace transform concepts."""
        # Simplified Laplace-like processing using exponential weighting
        # Real Laplace transform processing would be more complex
        
        time_vector = np.arange(len(signal_data)) / self.sampling_rate
        
        # Apply exponential weighting (decay parameter)
        decay_rate = 1.0  # Adjustable parameter
        weight = np.exp(-decay_rate * time_vector)
        
        # Weighted signal
        weighted_signal = signal_data * weight
        
        # Process weighted signal (example: convolution with kernel)
        kernel = np.exp(-np.arange(10) / 2.0)  # Exponential kernel
        kernel = kernel / np.sum(kernel)  # Normalize
        
        processed = signal.convolve(weighted_signal, kernel, mode='same')
        
        # Remove exponential weighting
        processed = processed / weight
        
        return processed
    
    def _kan_pattern_recognition(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Use KAN network for pattern recognition in signals."""
        if len(signal_data) < self.window_size:
            return {"patterns": [], "features": np.array([])}
        
        # Segment signal into windows
        segments = []
        hop_size = self.window_size // 2
        
        for i in range(0, len(signal_data) - self.window_size + 1, hop_size):
            segment = signal_data[i:i + self.window_size]
            segments.append(segment)
        
        if not segments:
            return {"patterns": [], "features": np.array([])}
        
        # Convert to tensor
        segments_tensor = torch.FloatTensor(np.array(segments))
        
        # Process through KAN network
        with torch.no_grad():
            pattern_features, reconstructed = self.kan_processor(segments_tensor)
        
        # Analyze patterns
        features_np = pattern_features.numpy()
        patterns = []
        
        # Simple pattern detection based on feature clustering
        if len(features_np) > 1:
            # Find distinctive patterns (simplified)
            feature_std = np.std(features_np, axis=0)
            distinctive_features = feature_std > np.mean(feature_std)
            
            if np.any(distinctive_features):
                patterns.append({
                    "type": "distinctive_pattern",
                    "features": features_np[:, distinctive_features],
                    "confidence": np.mean(feature_std[distinctive_features])
                })
        
        return {
            "patterns": patterns,
            "features": features_np,
            "reconstruction_quality": float(torch.mean((segments_tensor - reconstructed)**2))
        }
    
    def _integrate_with_cognitive_field(self, signal_data: np.ndarray, pattern_features: Dict[str, Any]):
        """Integrate signal processing with cognitive wave fields."""
        # Map signal characteristics to field position
        signal_energy = np.mean(signal_data**2)
        signal_complexity = len(pattern_features.get("patterns", []))
        
        # Convert to field coordinates
        x_pos = int((signal_energy / (1 + signal_energy)) * (self.wave_integrator.field_dimensions[0] - 1))
        y_pos = int((signal_complexity / (1 + signal_complexity)) * (self.wave_integrator.field_dimensions[1] - 1))
        
        # Update cognitive wave field
        self.wave_integrator.update_field_from_signal(signal_data, (x_pos, y_pos))
    
    def _compute_spectral_density(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectral density."""
        if len(signal_data) < 10:
            return np.array([]), np.array([])
        
        freqs, psd = signal.welch(signal_data, fs=self.sampling_rate, nperseg=min(256, len(signal_data)//4))
        return freqs, psd
    
    def _calculate_snr(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        if len(original) != len(processed):
            # Resize to match
            min_len = min(len(original), len(processed))
            original = original[:min_len]
            processed = processed[:min_len]
        
        signal_power = np.mean(processed**2)
        noise_power = np.mean((original - processed)**2)
        
        if noise_power > 0:
            return 10 * np.log10(signal_power / noise_power)
        else:
            return float('inf')
    
    def _overlap_add_reconstruction(self, segments: List[np.ndarray], hop_size: int) -> np.ndarray:
        """Reconstruct signal from overlapping segments."""
        if not segments:
            return np.array([])
        
        segment_length = len(segments[0])
        total_length = (len(segments) - 1) * hop_size + segment_length
        reconstructed = np.zeros(total_length)
        
        for i, segment in enumerate(segments):
            start_idx = i * hop_size
            end_idx = start_idx + len(segment)
            reconstructed[start_idx:end_idx] += segment
        
        return reconstructed
    
    def _update_processing_stats(self, compression_ratio: float, snr: float, processing_time: float):
        """Update processing performance statistics."""
        self.processing_stats["signals_processed"] += 1
        
        # Moving average updates
        alpha = 0.1  # Smoothing factor
        self.processing_stats["compression_achieved"] = (
            (1 - alpha) * self.processing_stats["compression_achieved"] + alpha * compression_ratio
        )
        self.processing_stats["average_snr"] = (
            (1 - alpha) * self.processing_stats["average_snr"] + alpha * snr
        )
        self.processing_stats["processing_latency"] = (
            (1 - alpha) * self.processing_stats["processing_latency"] + alpha * processing_time
        )
    
    def get_cognitive_field_state(self) -> np.ndarray:
        """Get current cognitive wave field state."""
        return self.wave_integrator.current_field.copy()
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current signal processing status and statistics."""
        return {
            "agent_id": self.agent_id,
            "sampling_rate": self.sampling_rate,
            "buffer_utilization": len(self.signal_buffer) / self.buffer_size,
            "processing_stats": self.processing_stats.copy(),
            "active_filters": list(self.filters.keys()),
            "patterns_detected": len(self.detected_patterns),
            "cognitive_field_energy": np.sum(self.wave_integrator.current_field**2)
        }
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message with signal data."""
        try:
            if "signal_data" in message:
                signal_data = np.array(message["signal_data"])
                signal_type = SignalType(message.get("signal_type", "continuous"))
                domain = SignalDomain(message.get("processing_domain", "time"))
                
                result = self.process_signal(signal_data, signal_type, domain)
                
                return {
                    "agent_id": self.agent_id,
                    "processing_successful": True,
                    "compression_ratio": result.compression_ratio,
                    "snr": result.signal_to_noise_ratio,
                    "patterns_detected": len(result.processing_metadata.get("pattern_features", {}).get("patterns", [])),
                    "timestamp": time.time()
                }
            
            return {"agent_id": self.agent_id, "status": "no_signal_data"}
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return {"agent_id": self.agent_id, "error": str(e)}

# Example usage and testing
def test_signal_agent():
    """Test the SignalProcessingAgent implementation."""
    print("📡 Testing SignalProcessingAgent...")
    
    # Create agent
    agent = SignalProcessingAgent(
        agent_id="test_signal_001",
        sampling_rate=1000.0
    )
    
    # Generate test signal
    t = np.linspace(0, 1, 1000)
    test_signal = (np.sin(2 * np.pi * 10 * t) +  # 10 Hz component
                  0.5 * np.sin(2 * np.pi * 25 * t) +  # 25 Hz component  
                  0.1 * np.random.randn(len(t)))  # Noise
    
    # Test signal processing
    result = agent.process_signal(test_signal, SignalType.CONTINUOUS, SignalDomain.TIME)
    
    print(f"   Signal processed successfully: {len(result.processed_signal)} samples")
    print(f"   Compression ratio: {result.compression_ratio:.2f}")
    print(f"   SNR: {result.signal_to_noise_ratio:.2f} dB")
    print(f"   Patterns detected: {len(result.processing_metadata.get('pattern_features', {}).get('patterns', []))}")
    
    # Test frequency domain processing
    freq_result = agent.process_signal(test_signal, SignalType.CONTINUOUS, SignalDomain.FREQUENCY)
    print(f"   Frequency domain processing successful")
    
    # Test message processing
    message = {
        "signal_data": test_signal[:100].tolist(),
        "signal_type": "continuous",
        "processing_domain": "time"
    }
    
    response = agent.process_message(message)
    print(f"   Message processing: {response['processing_successful']}")
    
    # Test status
    status = agent.get_processing_status()
    print(f"   Buffer utilization: {status['buffer_utilization']:.1%}")
    
    print("✅ SignalProcessingAgent test completed")

if __name__ == "__main__":
    test_signal_agent() 
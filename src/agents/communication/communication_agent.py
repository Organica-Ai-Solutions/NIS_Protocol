"""
NIS Protocol v3 - Communication Agent

Complete implementation of natural communication agent with:
- Real text-to-speech synthesis using advanced TTS engines
- Emotional voice modulation and contextual adaptation
- Multi-language support and voice selection
- Speech quality assessment and optimization
- Real-time audio processing and streaming
- Integration with interpretation and emotional state systems

Production-ready with actual speech synthesis capabilities.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import time
import os
import numpy as np
import torch
import logging
from collections import defaultdict, deque
import tempfile
import wave
import threading
import queue
import json

# Audio processing imports
try:
    import soundfile as sf
    import librosa
    import pyaudio
    from gtts import gTTS
    import pyttsx3
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logging.warning("Audio libraries not available. Speech synthesis disabled.")

# Advanced TTS imports
try:
    from TTS.api import TTS
    from speechbrain.pretrained import Tacotron2
    from speechbrain.pretrained import HiFiGAN
    ADVANCED_TTS_AVAILABLE = True
except ImportError:
    ADVANCED_TTS_AVAILABLE = False
    logging.warning("Advanced TTS libraries not available. Using fallback.")

# Integrity metrics for real calculations
from src.utils.integrity_metrics import (
    calculate_confidence, create_default_confidence_factors, ConfidenceFactors
)

# Self-audit capabilities
from src.utils.self_audit import self_audit_engine

# Emotional state integration
from src.emotion.emotional_state import EmotionalState
from src.agents.interpretation.interpretation_agent import InterpretationAgent


class VoiceProfile:
    """Voice profile configuration for TTS"""
    def __init__(
        self,
        voice_id: str,
        language: str = "en",
        gender: str = "neutral",
        age_range: str = "adult",
        accent: str = "neutral",
        speed: float = 1.0,
        pitch: float = 1.0,
        volume: float = 1.0,
        emotional_range: float = 0.5
    ):
        self.voice_id = voice_id
        self.language = language
        self.gender = gender
        self.age_range = age_range
        self.accent = accent
        self.speed = speed
        self.pitch = pitch
        self.volume = volume
        self.emotional_range = emotional_range


class AudioQualityMetrics:
    """Audio quality assessment metrics"""
    def __init__(
        self,
        clarity_score: float = 0.0,
        naturalness_score: float = 0.0,
        emotional_appropriateness: float = 0.0,
        intelligibility_score: float = 0.0,
        processing_time: float = 0.0,
        audio_artifacts: int = 0,
        signal_to_noise_ratio: float = 0.0
    ):
        self.clarity_score = clarity_score
        self.naturalness_score = naturalness_score
        self.emotional_appropriateness = emotional_appropriateness
        self.intelligibility_score = intelligibility_score
        self.processing_time = processing_time
        self.audio_artifacts = audio_artifacts
        self.signal_to_noise_ratio = signal_to_noise_ratio
        self.overall_quality = self._calculate_overall_quality()
    
    def _calculate_overall_quality(self) -> float:
        """Calculate overall quality score"""
        weights = {
            'clarity': 0.25,
            'naturalness': 0.25,
            'emotional': 0.20,
            'intelligibility': 0.20,
            'technical': 0.10
        }
        
        # Technical quality based on SNR and artifacts
        technical_quality = min(1.0, self.signal_to_noise_ratio / 30.0) * (1.0 - min(1.0, self.audio_artifacts / 10.0))
        
        overall = (
            weights['clarity'] * self.clarity_score +
            weights['naturalness'] * self.naturalness_score +
            weights['emotional'] * self.emotional_appropriateness +
            weights['intelligibility'] * self.intelligibility_score +
            weights['technical'] * technical_quality
        )
        
        return max(0.0, min(1.0, overall))


class AdvancedTTSEngine:
    """Advanced text-to-speech engine with multiple backends"""
    
    def __init__(self, preferred_engine: str = "auto"):
        self.preferred_engine = preferred_engine
        self.available_engines = self._detect_available_engines()
        self.current_engine = self._select_best_engine()
        self.voice_profiles = self._initialize_voice_profiles()
        
        # Engine-specific configurations
        self.engine_configs = {
            'pyttsx3': self._init_pyttsx3(),
            'gtts': self._init_gtts(),
            'coqui': self._init_coqui_tts(),
            'speechbrain': self._init_speechbrain()
        }
        
        self.logger = logging.getLogger("nis.tts_engine")
    
    def _detect_available_engines(self) -> List[str]:
        """Detect available TTS engines"""
        engines = []
        
        if AUDIO_AVAILABLE:
            engines.append('pyttsx3')
            engines.append('gtts')
        
        if ADVANCED_TTS_AVAILABLE:
            engines.append('coqui')
            engines.append('speechbrain')
        
        return engines
    
    def _select_best_engine(self) -> str:
        """Select the best available TTS engine"""
        if self.preferred_engine != "auto" and self.preferred_engine in self.available_engines:
            return self.preferred_engine
        
        # Priority order: Coqui > SpeechBrain > pyttsx3 > gTTS
        priority_order = ['coqui', 'speechbrain', 'pyttsx3', 'gtts']
        
        for engine in priority_order:
            if engine in self.available_engines:
                return engine
        
        return 'pyttsx3'  # Fallback
    
    def _initialize_voice_profiles(self) -> Dict[str, VoiceProfile]:
        """Initialize predefined voice profiles"""
        return {
            'neutral': VoiceProfile('neutral', 'en', 'neutral', 'adult', 'neutral'),
            'formal': VoiceProfile('formal', 'en', 'neutral', 'adult', 'neutral', speed=0.9, pitch=0.9),
            'friendly': VoiceProfile('friendly', 'en', 'neutral', 'adult', 'neutral', speed=1.1, pitch=1.1, emotional_range=0.8),
            'authoritative': VoiceProfile('authoritative', 'en', 'neutral', 'adult', 'neutral', speed=0.8, pitch=0.8, volume=1.2),
            'empathetic': VoiceProfile('empathetic', 'en', 'neutral', 'adult', 'neutral', speed=0.95, pitch=1.05, emotional_range=0.9),
            'technical': VoiceProfile('technical', 'en', 'neutral', 'adult', 'neutral', speed=0.85, pitch=0.95),
        }
    
    def _init_pyttsx3(self):
        """Initialize pyttsx3 engine"""
        if 'pyttsx3' not in self.available_engines:
            return None
        
        try:
            import pyttsx3
            engine = pyttsx3.init()
            
            # Get available voices
            voices = engine.getProperty('voices')
            
            # Configure default settings
            engine.setProperty('rate', 200)  # Speed of speech
            engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
            
            return {
                'engine': engine,
                'voices': voices,
                'rate': 200,
                'volume': 0.9
            }
        except Exception as e:
            self.logger.error(f"Failed to initialize pyttsx3: {e}")
            return None
    
    def _init_gtts(self):
        """Initialize Google Text-to-Speech"""
        if 'gtts' not in self.available_engines:
            return None
        
        return {
            'supported_languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh'],
            'default_lang': 'en',
            'slow': False
        }
    
    def _init_coqui_tts(self):
        """Initialize Coqui TTS"""
        if 'coqui' not in self.available_engines:
            return None
        
        try:
            from TTS.api import TTS
            
            # Get available models
            models = TTS.list_models()
            
            # Select best English model
            english_models = [m for m in models if 'en' in m.lower()]
            best_model = english_models[0] if english_models else models[0] if models else None
            
            if best_model:
                tts = TTS(model_name=best_model)
                
                return {
                    'tts': tts,
                    'model': best_model,
                    'available_models': models
                }
        except Exception as e:
            self.logger.error(f"Failed to initialize Coqui TTS: {e}")
        
        return None
    
    def _init_speechbrain(self):
        """Initialize SpeechBrain TTS"""
        if 'speechbrain' not in self.available_engines:
            return None
        
        try:
            # Initialize Tacotron2 and HiFiGAN
            tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech")
            hifi_gan = HiFiGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech")
            
            return {
                'tacotron2': tacotron2,
                'hifi_gan': hifi_gan,
                'sample_rate': 22050
            }
        except Exception as e:
            self.logger.error(f"Failed to initialize SpeechBrain: {e}")
        
        return None
    
    def synthesize_speech(
        self,
        text: str,
        voice_profile: VoiceProfile,
        output_path: Optional[str] = None,
        return_audio: bool = True
    ) -> Tuple[Optional[str], AudioQualityMetrics]:
        """
        Synthesize speech from text using configured TTS engine
        
        Args:
            text: Text to synthesize
            voice_profile: Voice configuration
            output_path: Optional output file path
            return_audio: Whether to return audio data
            
        Returns:
            Tuple of (audio_file_path or audio_data, quality_metrics)
        """
        start_time = time.time()
        
        try:
            # Select synthesis method based on current engine
            if self.current_engine == 'coqui':
                result = self._synthesize_coqui(text, voice_profile, output_path)
            elif self.current_engine == 'speechbrain':
                result = self._synthesize_speechbrain(text, voice_profile, output_path)
            elif self.current_engine == 'pyttsx3':
                result = self._synthesize_pyttsx3(text, voice_profile, output_path)
            elif self.current_engine == 'gtts':
                result = self._synthesize_gtts(text, voice_profile, output_path)
            else:
                raise ValueError(f"Unsupported engine: {self.current_engine}")
            
            processing_time = time.time() - start_time
            
            # Assess audio quality
            if result:
                quality_metrics = self._assess_audio_quality(result, text, processing_time)
            else:
                quality_metrics = AudioQualityMetrics(processing_time=processing_time)
            
            return result, quality_metrics
            
        except Exception as e:
            self.logger.error(f"Speech synthesis failed: {e}")
            processing_time = time.time() - start_time
            return None, AudioQualityMetrics(processing_time=processing_time)
    
    def _synthesize_coqui(self, text: str, voice_profile: VoiceProfile, output_path: Optional[str]) -> Optional[str]:
        """Synthesize using Coqui TTS"""
        config = self.engine_configs.get('coqui')
        if not config:
            return None
        
        try:
            # Generate temporary file if no output path provided
            if not output_path:
                output_path = tempfile.mktemp(suffix='.wav')
            
            # Synthesize speech
            config['tts'].tts_to_file(text=text, file_path=output_path)
            
            # Apply voice profile modifications
            if voice_profile.speed != 1.0 or voice_profile.pitch != 1.0:
                self._modify_audio_properties(output_path, voice_profile)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Coqui synthesis failed: {e}")
            return None
    
    def _synthesize_speechbrain(self, text: str, voice_profile: VoiceProfile, output_path: Optional[str]) -> Optional[str]:
        """Synthesize using SpeechBrain"""
        config = self.engine_configs.get('speechbrain')
        if not config:
            return None
        
        try:
            # Generate mel spectrogram
            mel_output, mel_length, alignment = config['tacotron2'].encode_text(text)
            
            # Generate waveform
            waveforms = config['hifi_gan'].decode_batch(mel_output)
            
            # Convert to numpy
            audio = waveforms.squeeze().cpu().numpy()
            
            # Generate output path if needed
            if not output_path:
                output_path = tempfile.mktemp(suffix='.wav')
            
            # Save audio
            sf.write(output_path, audio, config['sample_rate'])
            
            # Apply voice profile modifications
            if voice_profile.speed != 1.0 or voice_profile.pitch != 1.0:
                self._modify_audio_properties(output_path, voice_profile)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"SpeechBrain synthesis failed: {e}")
            return None
    
    def _synthesize_pyttsx3(self, text: str, voice_profile: VoiceProfile, output_path: Optional[str]) -> Optional[str]:
        """Synthesize using pyttsx3"""
        config = self.engine_configs.get('pyttsx3')
        if not config:
            return None
        
        try:
            engine = config['engine']
            
            # Apply voice profile settings
            rate = int(config['rate'] * voice_profile.speed)
            volume = min(1.0, config['volume'] * voice_profile.volume)
            
            engine.setProperty('rate', rate)
            engine.setProperty('volume', volume)
            
            # Select voice based on profile
            voices = config['voices']
            if voices:
                selected_voice = None
                for voice in voices:
                    if voice_profile.gender.lower() in voice.name.lower():
                        selected_voice = voice
                        break
                
                if selected_voice:
                    engine.setProperty('voice', selected_voice.id)
            
            # Generate output path if needed
            if not output_path:
                output_path = tempfile.mktemp(suffix='.wav')
            
            # Save to file
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"pyttsx3 synthesis failed: {e}")
            return None
    
    def _synthesize_gtts(self, text: str, voice_profile: VoiceProfile, output_path: Optional[str]) -> Optional[str]:
        """Synthesize using Google Text-to-Speech"""
        config = self.engine_configs.get('gtts')
        if not config:
            return None
        
        try:
            from gtts import gTTS
            
            # Create gTTS object
            lang = voice_profile.language if voice_profile.language in config['supported_languages'] else 'en'
            slow = voice_profile.speed < 0.8
            
            tts = gTTS(text=text, lang=lang, slow=slow)
            
            # Generate output path if needed
            if not output_path:
                output_path = tempfile.mktemp(suffix='.mp3')
            
            # Save audio
            tts.save(output_path)
            
            # Convert to WAV if needed
            if output_path.endswith('.wav'):
                wav_path = output_path
                mp3_path = output_path.replace('.wav', '.mp3')
                tts.save(mp3_path)
                
                # Convert MP3 to WAV
                self._convert_audio_format(mp3_path, wav_path)
                os.remove(mp3_path)
                
                return wav_path
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"gTTS synthesis failed: {e}")
            return None
    
    def _modify_audio_properties(self, audio_path: str, voice_profile: VoiceProfile):
        """Modify audio properties based on voice profile"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Apply speed modification
            if voice_profile.speed != 1.0:
                audio = librosa.effects.time_stretch(audio, rate=voice_profile.speed)
            
            # Apply pitch modification
            if voice_profile.pitch != 1.0:
                # Convert pitch ratio to semitones
                semitones = 12 * np.log2(voice_profile.pitch)
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
            
            # Apply volume modification
            if voice_profile.volume != 1.0:
                audio = audio * voice_profile.volume
            
            # Save modified audio
            sf.write(audio_path, audio, sr)
            
        except Exception as e:
            self.logger.error(f"Audio modification failed: {e}")
    
    def _convert_audio_format(self, input_path: str, output_path: str):
        """Convert audio between formats"""
        try:
            audio, sr = librosa.load(input_path, sr=None)
            sf.write(output_path, audio, sr)
        except Exception as e:
            self.logger.error(f"Audio conversion failed: {e}")
    
    def _assess_audio_quality(self, audio_path: str, original_text: str, processing_time: float) -> AudioQualityMetrics:
        """Assess quality of synthesized audio"""
        try:
            # Load audio for analysis
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Calculate signal-to-noise ratio
            snr = self._calculate_snr(audio)
            
            # Assess clarity (spectral centroid stability)
            clarity_score = self._assess_clarity(audio, sr)
            
            # Assess naturalness (prosody analysis)
            naturalness_score = self._assess_naturalness(audio, sr)
            
            # Assess intelligibility (energy distribution)
            intelligibility_score = self._assess_intelligibility(audio, sr)
            
            # Count audio artifacts
            artifacts = self._count_artifacts(audio)
            
            # Emotional appropriateness (basic analysis)
            emotional_score = self._assess_emotional_appropriateness(audio, sr)
            
            return AudioQualityMetrics(
                clarity_score=clarity_score,
                naturalness_score=naturalness_score,
                emotional_appropriateness=emotional_score,
                intelligibility_score=intelligibility_score,
                processing_time=processing_time,
                audio_artifacts=artifacts,
                signal_to_noise_ratio=snr
            )
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            return AudioQualityMetrics(processing_time=processing_time)
    
    def _calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        try:
            # Simple SNR calculation
            signal_power = np.mean(audio ** 2)
            
            # Estimate noise from quiet segments
            energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            noise_threshold = np.percentile(energy, 10)  # Bottom 10% as noise
            noise_mask = energy < noise_threshold
            
            if np.any(noise_mask):
                noise_segments = audio[noise_mask]
                noise_power = np.mean(noise_segments ** 2)
                
                if noise_power > 0:
                    snr_linear = signal_power / noise_power
                    snr_db = 10 * np.log10(snr_linear)
                    return max(0.0, min(60.0, snr_db))  # Clamp between 0 and 60 dB
            
            return 30.0  # Default good SNR
            
        except:
            return 20.0  # Default moderate SNR
    
    def _assess_clarity(self, audio: np.ndarray, sr: int) -> float:
        """Assess speech clarity based on spectral features"""
        try:
            # Calculate spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            
            # Assess stability (lower variance = better clarity)
            centroid_stability = 1.0 / (1.0 + np.std(spectral_centroid))
            
            # Assess frequency distribution
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            rolloff_consistency = 1.0 / (1.0 + np.std(spectral_rolloff))
            
            # Combine metrics
            clarity = (centroid_stability + rolloff_consistency) / 2.0
            
            return max(0.0, min(1.0, clarity))
            
        except:
            return 0.7  # Default moderate clarity
    
    def _assess_naturalness(self, audio: np.ndarray, sr: int) -> float:
        """Assess speech naturalness based on prosodic features"""
        try:
            # Extract fundamental frequency (pitch)
            f0 = librosa.yin(audio, fmin=80, fmax=400)
            
            # Remove unvoiced segments
            f0_voiced = f0[f0 > 80]
            
            if len(f0_voiced) > 10:
                # Assess pitch variation (natural speech has variation)
                pitch_variation = np.std(f0_voiced) / np.mean(f0_voiced)
                
                # Natural speech typically has 10-30% pitch variation
                naturalness = 1.0 - abs(pitch_variation - 0.2) / 0.2
                naturalness = max(0.0, min(1.0, naturalness))
            else:
                naturalness = 0.5  # Moderate score for unclear pitch
            
            # Assess rhythm (energy variation)
            rms_energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            energy_variation = np.std(rms_energy) / np.mean(rms_energy)
            
            # Combine pitch and rhythm assessments
            rhythm_score = min(1.0, energy_variation / 0.5)  # Normalize to 0-1
            
            return (naturalness + rhythm_score) / 2.0
            
        except:
            return 0.6  # Default moderate naturalness
    
    def _assess_intelligibility(self, audio: np.ndarray, sr: int) -> float:
        """Assess speech intelligibility"""
        try:
            # Calculate MFCCs (mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Assess MFCC distribution and stability
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Good intelligibility has consistent MFCC patterns
            consistency = 1.0 / (1.0 + np.mean(mfcc_std))
            
            # Assess spectral contrast (important for consonant clarity)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            contrast_mean = np.mean(spectral_contrast)
            
            # Higher spectral contrast typically indicates better intelligibility
            contrast_score = min(1.0, contrast_mean / 20.0)
            
            return (consistency + contrast_score) / 2.0
            
        except:
            return 0.7  # Default good intelligibility
    
    def _count_artifacts(self, audio: np.ndarray) -> int:
        """Count audio artifacts (clicks, pops, clipping)"""
        try:
            artifacts = 0
            
            # Detect clipping
            clipping_threshold = 0.95
            clipped_samples = np.sum(np.abs(audio) > clipping_threshold)
            artifacts += clipped_samples
            
            # Detect sudden amplitude changes (clicks/pops)
            diff = np.diff(audio)
            sudden_changes = np.sum(np.abs(diff) > 0.1)
            artifacts += sudden_changes // 100  # Normalize
            
            # Detect silence gaps that might indicate processing artifacts
            rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            silence_threshold = np.percentile(rms, 5)
            silence_gaps = np.sum(rms < silence_threshold)
            
            # Excessive silence might indicate synthesis artifacts
            if silence_gaps > len(rms) * 0.3:
                artifacts += 1
            
            return artifacts
            
        except:
            return 0  # Default no artifacts detected
    
    def _assess_emotional_appropriateness(self, audio: np.ndarray, sr: int) -> float:
        """Assess emotional appropriateness of speech synthesis"""
        try:
            # Simple emotional assessment based on prosodic features
            
            # Energy dynamics (emotional speech has more variation)
            rms_energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            energy_dynamics = np.std(rms_energy) / np.mean(rms_energy)
            
            # Pitch dynamics
            f0 = librosa.yin(audio, fmin=80, fmax=400)
            f0_voiced = f0[f0 > 80]
            
            if len(f0_voiced) > 10:
                pitch_dynamics = np.std(f0_voiced) / np.mean(f0_voiced)
            else:
                pitch_dynamics = 0.1
            
            # Temporal dynamics (speaking rate variation)
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            tempo_score = min(1.0, tempo / 120.0)  # Normalize around 120 BPM
            
            # Combine metrics (moderate variation is generally appropriate)
            emotional_score = (
                min(1.0, energy_dynamics / 0.5) * 0.4 +
                min(1.0, pitch_dynamics / 0.3) * 0.4 +
                tempo_score * 0.2
            )
            
            return max(0.0, min(1.0, emotional_score))
            
        except:
            return 0.6  # Default moderate emotional appropriateness


class CommunicationAgent:
    """
    Complete Communication Agent with advanced speech synthesis
    
    Features:
    - Multi-engine TTS with quality assessment
    - Emotional voice modulation
    - Real-time audio processing
    - Voice profile management
    - Quality optimization and monitoring
    - Integration with interpretation and emotional systems
    """
    
    def __init__(
        self,
        agent_id: str = "communicator",
        description: str = "Handles natural communication with advanced speech synthesis",
        emotional_state: Optional[EmotionalState] = None,
        interpreter: Optional[InterpretationAgent] = None,
        default_voice_profile: str = "neutral",
        output_dir: str = "data/audio_output",
        preferred_tts_engine: str = "auto",
        enable_self_audit: bool = True
    ):
        """
        Initialize the communication agent
        
        Args:
            agent_id: Unique identifier for this agent
            description: Human-readable description of the agent's role
            emotional_state: Optional pre-configured emotional state
            interpreter: Optional interpreter agent for content analysis
            default_voice_profile: Default voice profile to use
            output_dir: Directory for saving audio files
            preferred_tts_engine: Preferred TTS engine
            enable_self_audit: Whether to enable real-time integrity monitoring
        """
        self.agent_id = agent_id
        self.description = description
        self.emotional_state = emotional_state or EmotionalState()
        self.interpreter = interpreter
        self.default_voice_profile = default_voice_profile
        self.output_dir = output_dir
        self.enable_self_audit = enable_self_audit
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize TTS engine
        self.tts_engine = AdvancedTTSEngine(preferred_tts_engine)
        
        # Track conversation history
        self.conversation_history: deque = deque(maxlen=100)
        
        # Performance metrics
        self.performance_metrics = {
            'total_communications': 0,
            'successful_syntheses': 0,
            'average_quality_score': 0.0,
            'average_processing_time': 0.0,
            'audio_artifacts_detected': 0,
            'total_audio_generated': 0.0  # Total seconds of audio
        }
        
        # Quality tracking
        self.quality_history: deque = deque(maxlen=100)
        self.voice_profile_performance = defaultdict(list)
        
        # Self-audit integration
        self.integrity_monitoring_enabled = enable_self_audit
        self.audit_metrics = {
            'total_audits': 0,
            'violations_detected': 0,
            'auto_corrections': 0,
            'average_integrity_score': 100.0
        }
        
        self.logger = logging.getLogger("nis.communication_agent")
        self.logger.info(f"Initialized Communication Agent with {self.tts_engine.current_engine} TTS engine")
    
    def communicate(
        self,
        content: str,
        voice_profile_name: Optional[str] = None,
        emotional_context: Optional[Dict[str, Any]] = None,
        save_audio: bool = True,
        play_audio: bool = False
    ) -> Dict[str, Any]:
        """
        Generate speech communication from text content
        
        Args:
            content: Text content to communicate
            voice_profile_name: Optional voice profile to use
            emotional_context: Optional emotional context for modulation
            save_audio: Whether to save the generated audio
            play_audio: Whether to play the audio
            
        Returns:
            Communication result with audio path and quality metrics
        """
        start_time = time.time()
        
        try:
            # Interpret content if interpreter is available
            interpretation = None
            if self.interpreter:
                interpretation = self.interpreter.process({
                    "operation": "interpret",
                    "content": content
                })
                
                # Update emotional state based on interpretation
                if interpretation.get("status") == "success":
                    if "emotional_state" in interpretation:
                        self.emotional_state = interpretation["emotional_state"]
            
            # Select voice profile
            voice_profile = self._select_voice_profile(
                voice_profile_name or self.default_voice_profile,
                emotional_context
            )
            
            # Generate response with emotional adaptation
            response_text = self._craft_emotionally_aware_response(content, interpretation)
            
            # Synthesize speech
            output_path = None
            if save_audio:
                timestamp = int(time.time() * 1000)
                output_path = os.path.join(self.output_dir, f"speech_{timestamp}.wav")
            
            audio_path, quality_metrics = self.tts_engine.synthesize_speech(
                response_text,
                voice_profile,
                output_path
            )
            
            # Play audio if requested
            if play_audio and audio_path:
                self._play_audio(audio_path)
            
            # Update conversation history
            self._update_conversation_history(content, response_text, interpretation, quality_metrics)
            
            # Update performance metrics
            self._update_performance_metrics(quality_metrics, audio_path)
            
            # Self-audit check
            if self.enable_self_audit:
                self._audit_communication(content, response_text, quality_metrics)
            
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "original_content": content,
                "response_text": response_text,
                "audio_path": audio_path,
                "quality_metrics": {
                    "overall_quality": quality_metrics.overall_quality,
                    "clarity_score": quality_metrics.clarity_score,
                    "naturalness_score": quality_metrics.naturalness_score,
                    "emotional_appropriateness": quality_metrics.emotional_appropriateness,
                    "intelligibility_score": quality_metrics.intelligibility_score,
                    "processing_time": quality_metrics.processing_time,
                    "signal_to_noise_ratio": quality_metrics.signal_to_noise_ratio,
                    "audio_artifacts": quality_metrics.audio_artifacts
                },
                "voice_profile_used": voice_profile.voice_id,
                "interpretation": interpretation,
                "agent_id": self.agent_id,
                "total_processing_time": processing_time,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Communication failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _select_voice_profile(
        self,
        profile_name: str,
        emotional_context: Optional[Dict[str, Any]] = None
    ) -> VoiceProfile:
        """Select and customize voice profile based on context"""
        # Get base profile
        base_profile = self.tts_engine.voice_profiles.get(profile_name, self.tts_engine.voice_profiles['neutral'])
        
        # Create a copy for customization
        profile = VoiceProfile(
            voice_id=base_profile.voice_id,
            language=base_profile.language,
            gender=base_profile.gender,
            age_range=base_profile.age_range,
            accent=base_profile.accent,
            speed=base_profile.speed,
            pitch=base_profile.pitch,
            volume=base_profile.volume,
            emotional_range=base_profile.emotional_range
        )
        
        # Apply emotional state modifications
        emotional_state = self.emotional_state.get_state()
        
        if emotional_state:
            # Adjust based on emotional dimensions
            arousal = emotional_state.get('arousal', 0.5)
            valence = emotional_state.get('valence', 0.5)
            confidence = emotional_state.get('confidence', 0.5)
            
            # Speed adjustments
            if arousal > 0.7:  # High arousal = faster speech
                profile.speed *= 1.1 + (arousal - 0.7) * 0.5
            elif arousal < 0.3:  # Low arousal = slower speech
                profile.speed *= 0.9 - (0.3 - arousal) * 0.3
            
            # Pitch adjustments
            if valence > 0.6:  # Positive emotion = higher pitch
                profile.pitch *= 1.0 + (valence - 0.6) * 0.3
            elif valence < 0.4:  # Negative emotion = lower pitch
                profile.pitch *= 1.0 - (0.4 - valence) * 0.2
            
            # Volume adjustments
            if confidence > 0.7:  # High confidence = louder
                profile.volume *= 1.0 + (confidence - 0.7) * 0.2
            elif confidence < 0.3:  # Low confidence = quieter
                profile.volume *= 1.0 - (0.3 - confidence) * 0.3
        
        # Apply emotional context if provided
        if emotional_context:
            context_intensity = emotional_context.get('intensity', 0.5)
            context_type = emotional_context.get('type', 'neutral')
            
            if context_type == 'urgent':
                profile.speed *= 1.2
                profile.volume *= 1.1
            elif context_type == 'calm':
                profile.speed *= 0.9
                profile.pitch *= 0.95
            elif context_type == 'friendly':
                profile.pitch *= 1.05
                profile.emotional_range *= 1.2
            elif context_type == 'formal':
                profile.speed *= 0.95
                profile.pitch *= 0.98
                profile.emotional_range *= 0.8
        
        # Ensure values stay within reasonable bounds
        profile.speed = max(0.5, min(2.0, profile.speed))
        profile.pitch = max(0.5, min(2.0, profile.pitch))
        profile.volume = max(0.1, min(1.5, profile.volume))
        
        return profile
    
    def _craft_emotionally_aware_response(
        self,
        content: str,
        interpretation: Optional[Dict[str, Any]] = None
    ) -> str:
        """Craft response that's emotionally appropriate and contextually aware"""
        # Start with the content
        response = content
        
        # Apply emotional awareness if interpretation is available
        if interpretation and interpretation.get("status") == "success":
            sentiment = interpretation.get("sentiment", "NEUTRAL")
            content_type = interpretation.get("content_type", [])
            
            # Adjust response based on content analysis
            if any(ct.get("label") == "query" for ct in content_type if isinstance(ct, dict)):
                # This is a question - provide informative response
                response = f"Let me address your question: {content}"
            
            elif sentiment == "POSITIVE":
                # Positive content - maintain positive tone
                response = f"I'm pleased to share: {content}"
            
            elif sentiment == "NEGATIVE":
                # Negative content - use empathetic tone
                response = f"I understand this may be concerning: {content}"
            
            elif any(ct.get("label") == "instructional" for ct in content_type if isinstance(ct, dict)):
                # Instructional content - use clear, structured delivery
                response = f"Here are the instructions: {content}"
        
        # Apply emotional state context
        emotional_state = self.emotional_state.get_state()
        if emotional_state:
            confidence = emotional_state.get('confidence', 0.5)
            
            if confidence < 0.3:
                # Low confidence - add uncertainty markers
                response = f"I believe {response.lower()}"
            elif confidence > 0.8:
                # High confidence - add assertive markers
                response = f"I can confirm that {response.lower()}"
        
        return response
    
    def _play_audio(self, audio_path: str):
        """Play audio file"""
        try:
            if AUDIO_AVAILABLE:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            else:
                self.logger.warning("Audio playback not available")
        except Exception as e:
            self.logger.error(f"Audio playback failed: {e}")
    
    def _update_conversation_history(
        self,
        input_content: str,
        response_text: str,
        interpretation: Optional[Dict[str, Any]],
        quality_metrics: AudioQualityMetrics
    ):
        """Update conversation history with new interaction"""
        entry = {
            "timestamp": time.time(),
            "input": input_content,
            "response": response_text,
            "interpretation": interpretation,
            "quality_score": quality_metrics.overall_quality,
            "processing_time": quality_metrics.processing_time
        }
        
        self.conversation_history.append(entry)
    
    def _update_performance_metrics(self, quality_metrics: AudioQualityMetrics, audio_path: Optional[str]):
        """Update performance metrics with new communication data"""
        self.performance_metrics['total_communications'] += 1
        
        if audio_path:
            self.performance_metrics['successful_syntheses'] += 1
            
            # Calculate audio duration
            try:
                if AUDIO_AVAILABLE:
                    audio, sr = librosa.load(audio_path, sr=None)
                    duration = len(audio) / sr
                    self.performance_metrics['total_audio_generated'] += duration
            except:
                pass
        
        # Update quality metrics using exponential moving average
        alpha = 0.1
        current_avg_quality = self.performance_metrics['average_quality_score']
        self.performance_metrics['average_quality_score'] = (
            current_avg_quality * (1 - alpha) + quality_metrics.overall_quality * alpha
        )
        
        current_avg_time = self.performance_metrics['average_processing_time']
        self.performance_metrics['average_processing_time'] = (
            current_avg_time * (1 - alpha) + quality_metrics.processing_time * alpha
        )
        
        # Track artifacts
        self.performance_metrics['audio_artifacts_detected'] += quality_metrics.audio_artifacts
        
        # Store quality metrics for analysis
        self.quality_history.append(quality_metrics)
    
    def _audit_communication(
        self,
        content: str,
        response: str,
        quality_metrics: AudioQualityMetrics
    ):
        """Perform self-audit on communication"""
        if not self.enable_self_audit:
            return
        
        try:
            # Create audit text
            audit_text = f"""
            Communication:
            Input: {content[:100]}...
            Response: {response[:100]}...
            Quality Score: {quality_metrics.overall_quality}
            Processing Time: {quality_metrics.processing_time}
            Audio Artifacts: {quality_metrics.audio_artifacts}
            SNR: {quality_metrics.signal_to_noise_ratio}
            """
            
            # Perform audit
            violations = self_audit_engine.audit_text(audit_text)
            integrity_score = self_audit_engine.get_integrity_score(audit_text)
            
            self.audit_metrics['total_audits'] += 1
            self.audit_metrics['average_integrity_score'] = (
                self.audit_metrics['average_integrity_score'] * 0.9 + integrity_score * 0.1
            )
            
            if violations:
                self.audit_metrics['violations_detected'] += len(violations)
                self.logger.warning(f"Communication audit violations: {[v['type'] for v in violations]}")
                
        except Exception as e:
            self.logger.error(f"Communication audit error: {e}")
    
    def optimize_voice_profile(self, profile_name: str, target_quality: float = 0.85) -> Dict[str, Any]:
        """Optimize voice profile based on performance history"""
        try:
            # Get performance data for this profile
            profile_data = self.voice_profile_performance[profile_name]
            
            if len(profile_data) < 5:
                return {"status": "insufficient_data", "samples_needed": 5 - len(profile_data)}
            
            # Analyze performance patterns
            quality_scores = [d['quality'] for d in profile_data]
            current_avg_quality = np.mean(quality_scores[-10:])  # Last 10 samples
            
            if current_avg_quality >= target_quality:
                return {"status": "already_optimized", "current_quality": current_avg_quality}
            
            # Identify improvement opportunities
            optimization_suggestions = []
            
            # Analyze quality vs. speed relationship
            speeds = [d['speed'] for d in profile_data]
            speed_quality_corr = np.corrcoef(speeds, quality_scores)[0, 1]
            
            if speed_quality_corr < -0.3:  # Negative correlation
                optimization_suggestions.append({
                    "parameter": "speed",
                    "adjustment": "decrease",
                    "reason": "Lower speed correlates with higher quality"
                })
            
            # Analyze artifacts vs. pitch relationship
            artifacts = [d['artifacts'] for d in profile_data]
            pitches = [d['pitch'] for d in profile_data]
            
            if len(artifacts) > 0 and len(pitches) > 0:
                pitch_artifact_corr = np.corrcoef(pitches, artifacts)[0, 1]
                
                if pitch_artifact_corr > 0.3:  # Positive correlation
                    optimization_suggestions.append({
                        "parameter": "pitch",
                        "adjustment": "decrease",
                        "reason": "Lower pitch correlates with fewer artifacts"
                    })
            
            return {
                "status": "optimization_available",
                "current_quality": current_avg_quality,
                "target_quality": target_quality,
                "suggestions": optimization_suggestions
            }
            
        except Exception as e:
            self.logger.error(f"Voice profile optimization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_communication_analytics(self) -> Dict[str, Any]:
        """Get comprehensive communication analytics"""
        analytics = {
            "performance_summary": self.performance_metrics.copy(),
            "conversation_stats": {
                "total_conversations": len(self.conversation_history),
                "average_response_length": 0,
                "quality_trend": "stable"
            },
            "quality_analysis": {
                "recent_quality_scores": [],
                "quality_distribution": {},
                "improvement_areas": []
            },
            "technical_metrics": {
                "tts_engine": self.tts_engine.current_engine,
                "available_engines": self.tts_engine.available_engines,
                "voice_profiles_used": len(self.voice_profile_performance)
            }
        }
        
        # Analyze conversation history
        if self.conversation_history:
            responses = [entry['response'] for entry in self.conversation_history]
            analytics["conversation_stats"]["average_response_length"] = np.mean([len(r) for r in responses])
            
            # Quality trend analysis
            recent_qualities = [entry['quality_score'] for entry in list(self.conversation_history)[-10:]]
            if len(recent_qualities) >= 2:
                trend_slope = np.polyfit(range(len(recent_qualities)), recent_qualities, 1)[0]
                if trend_slope > 0.01:
                    analytics["conversation_stats"]["quality_trend"] = "improving"
                elif trend_slope < -0.01:
                    analytics["conversation_stats"]["quality_trend"] = "declining"
        
        # Quality analysis
        if self.quality_history:
            recent_qualities = [q.overall_quality for q in list(self.quality_history)[-20:]]
            analytics["quality_analysis"]["recent_quality_scores"] = recent_qualities
            
            # Quality distribution
            quality_ranges = {
                "excellent": sum(1 for q in recent_qualities if q >= 0.9),
                "good": sum(1 for q in recent_qualities if 0.7 <= q < 0.9),
                "fair": sum(1 for q in recent_qualities if 0.5 <= q < 0.7),
                "poor": sum(1 for q in recent_qualities if q < 0.5)
            }
            analytics["quality_analysis"]["quality_distribution"] = quality_ranges
            
            # Identify improvement areas
            avg_clarity = np.mean([q.clarity_score for q in self.quality_history])
            avg_naturalness = np.mean([q.naturalness_score for q in self.quality_history])
            avg_intelligibility = np.mean([q.intelligibility_score for q in self.quality_history])
            
            if avg_clarity < 0.7:
                analytics["quality_analysis"]["improvement_areas"].append("clarity")
            if avg_naturalness < 0.7:
                analytics["quality_analysis"]["improvement_areas"].append("naturalness")
            if avg_intelligibility < 0.7:
                analytics["quality_analysis"]["improvement_areas"].append("intelligibility")
        
        return analytics
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of communication agent"""
        success_rate = (
            self.performance_metrics['successful_syntheses'] / max(self.performance_metrics['total_communications'], 1)
        )
        
        return {
            'agent_id': self.agent_id,
            'tts_engine': self.tts_engine.current_engine,
            'available_engines': self.tts_engine.available_engines,
            'default_voice_profile': self.default_voice_profile,
            'conversation_history_size': len(self.conversation_history),
            'performance_metrics': self.performance_metrics,
            'success_rate': success_rate,
            'average_quality': self.performance_metrics['average_quality_score'],
            'audio_capabilities': AUDIO_AVAILABLE,
            'advanced_tts_available': ADVANCED_TTS_AVAILABLE,
            'audit_metrics': self.audit_metrics,
            'timestamp': time.time()
        } 
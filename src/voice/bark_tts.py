"""
Bark TTS - Natural, Expressive Text-to-Speech
Like ChatGPT Voice Mode / VibeVoice quality
"""

import io
import logging
import numpy as np
from typing import Dict, Any, Optional
import warnings

# Suppress Bark warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class BarkTTS:
    """
    Bark TTS for natural, expressive speech
    
    Features:
    - Natural, human-like voice
    - Multiple speakers/voices
    - Emotional expression
    - Background sounds (optional)
    - Much better than gTTS!
    """
    
    # Voice presets - different personalities
    VOICES = {
        "default": "v2/en_speaker_6",      # Friendly, neutral
        "professional": "v2/en_speaker_9",  # Clear, professional
        "friendly": "v2/en_speaker_3",      # Warm, conversational
        "energetic": "v2/en_speaker_1",     # Upbeat, enthusiastic
    }
    
    def __init__(self, voice: str = "friendly", device: str = "cpu"):
        """
        Initialize Bark TTS
        
        Args:
            voice: Voice preset (default, professional, friendly, energetic)
            device: cpu or cuda
        """
        self.voice_preset = self.VOICES.get(voice, self.VOICES["friendly"])
        self.device = device
        self.model = None
        self.processor = None
        self.initialized = False
        
    def initialize(self):
        """Load Bark model (lazy loading)"""
        if self.initialized:
            return True
            
        try:
            from bark import SAMPLE_RATE, generate_audio, preload_models
            from transformers import AutoProcessor, BarkModel
            
            logger.info(f"🎙️ Loading Bark TTS model ({self.voice_preset})...")
            
            # Preload models
            preload_models(
                text_use_gpu=(self.device == "cuda"),
                coarse_use_gpu=(self.device == "cuda"),
                fine_use_gpu=(self.device == "cuda"),
            )
            
            # Load model and processor
            self.processor = AutoProcessor.from_pretrained("suno/bark")
            self.model = BarkModel.from_pretrained("suno/bark")
            
            if self.device == "cuda":
                self.model = self.model.to("cuda")
            
            self.sample_rate = SAMPLE_RATE
            self.initialized = True
            
            logger.info(f"✅ Bark TTS loaded successfully!")
            return True
            
        except ImportError:
            logger.error("❌ Bark not installed. Run: pip install git+https://github.com/suno-ai/bark.git")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to load Bark: {e}")
            return False
    
    def synthesize(self, text: str, voice: Optional[str] = None) -> Dict[str, Any]:
        """
        Synthesize natural speech from text
        
        Args:
            text: Text to speak
            voice: Override default voice (optional)
            
        Returns:
            {
                "success": True,
                "audio_data": base64 encoded WAV,
                "sample_rate": 24000,
                "format": "wav"
            }
        """
        try:
            # Initialize if needed
            if not self.initialized:
                if not self.initialize():
                    return {
                        "success": False,
                        "error": "Bark not available",
                        "audio_data": ""
                    }
            
            # Use custom voice or default
            voice_preset = voice if voice else self.voice_preset
            
            logger.info(f"🎤 Synthesizing: '{text[:50]}...' with voice {voice_preset}")
            
            # Prepare inputs
            inputs = self.processor(
                text=text,
                voice_preset=voice_preset,
                return_tensors="pt"
            )
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate audio
            import torch
            with torch.no_grad():
                audio_array = self.model.generate(**inputs)
            
            # Convert to numpy
            audio_array = audio_array.cpu().numpy().squeeze()
            
            # Normalize audio to prevent clipping
            audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Convert to WAV bytes
            import scipy.io.wavfile as wavfile
            wav_io = io.BytesIO()
            wavfile.write(wav_io, self.sample_rate, (audio_array * 32767).astype(np.int16))
            wav_bytes = wav_io.getvalue()
            
            # Encode to base64
            import base64
            audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
            
            logger.info(f"✅ Synthesized {len(audio_array) / self.sample_rate:.2f}s of audio")
            
            return {
                "success": True,
                "audio_data": audio_base64,
                "sample_rate": self.sample_rate,
                "format": "wav",
                "duration": len(audio_array) / self.sample_rate,
                "voice": voice_preset
            }
            
        except Exception as e:
            logger.error(f"❌ Bark synthesis error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "audio_data": ""
            }
    
    def long_form_synthesize(self, text: str, voice: Optional[str] = None) -> Dict[str, Any]:
        """
        Synthesize long text by splitting into sentences
        
        Args:
            text: Long text to speak
            voice: Voice preset
            
        Returns:
            Combined audio result
        """
        try:
            # Split into sentences
            import nltk
            try:
                sentences = nltk.sent_tokenize(text)
            except:
                # Download punkt if not available
                nltk.download('punkt')
                sentences = nltk.sent_tokenize(text)
            
            logger.info(f"📝 Synthesizing {len(sentences)} sentences...")
            
            # Synthesize each sentence
            audio_segments = []
            total_duration = 0
            
            for i, sentence in enumerate(sentences):
                result = self.synthesize(sentence, voice)
                
                if result["success"]:
                    import base64
                    audio_bytes = base64.b64decode(result["audio_data"])
                    
                    # Read WAV data
                    import scipy.io.wavfile as wavfile
                    wav_io = io.BytesIO(audio_bytes)
                    sample_rate, audio_data = wavfile.read(wav_io)
                    
                    audio_segments.append(audio_data)
                    total_duration += result["duration"]
                    
                    # Add silence between sentences (0.25s)
                    silence = np.zeros(int(0.25 * sample_rate))
                    audio_segments.append(silence.astype(np.int16))
                else:
                    logger.warning(f"Failed to synthesize sentence {i+1}")
            
            if not audio_segments:
                return {
                    "success": False,
                    "error": "No audio generated",
                    "audio_data": ""
                }
            
            # Concatenate all segments
            combined_audio = np.concatenate(audio_segments)
            
            # Convert to WAV
            import scipy.io.wavfile as wavfile
            wav_io = io.BytesIO()
            wavfile.write(wav_io, sample_rate, combined_audio)
            wav_bytes = wav_io.getvalue()
            
            # Encode to base64
            import base64
            audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
            
            logger.info(f"✅ Combined audio: {total_duration:.2f}s")
            
            return {
                "success": True,
                "audio_data": audio_base64,
                "sample_rate": sample_rate,
                "format": "wav",
                "duration": total_duration,
                "sentences": len(sentences),
                "voice": voice or self.voice_preset
            }
            
        except Exception as e:
            logger.error(f"❌ Long-form synthesis error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "audio_data": ""
            }


# Global instance
_bark_instance: Optional[BarkTTS] = None

def get_bark_tts(voice: str = "friendly", device: str = "cpu") -> BarkTTS:
    """Get global Bark TTS instance"""
    global _bark_instance
    if _bark_instance is None:
        _bark_instance = BarkTTS(voice=voice, device=device)
    return _bark_instance


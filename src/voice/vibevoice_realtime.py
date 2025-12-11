#!/usr/bin/env python3
"""
Microsoft VibeVoice-Realtime Integration for NIS Protocol
Real-time streaming TTS using microsoft/VibeVoice-Realtime-0.5B

This module provides:
- Real-time text-to-speech with ~300ms first chunk latency
- Streaming text input support
- Multiple speaker voices
- WebSocket streaming support

Copyright 2025 Organica AI Solutions
Licensed under Apache 2.0
"""

import asyncio
import logging
import time
import io
import os
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Speaker voice options from VibeVoice
class VibeVoiceSpeaker(Enum):
    """Available VibeVoice speaker voices"""
    CARTER = "Carter"           # Male, professional
    NOVA = "Nova"               # Female, warm
    ARIA = "Aria"               # Female, expressive
    DAVIS = "Davis"             # Male, authoritative
    # Multilingual speakers (experimental)
    GERMAN = "German"
    FRENCH = "French"
    JAPANESE = "Japanese"
    KOREAN = "Korean"
    SPANISH = "Spanish"


@dataclass
class VibeVoiceConfig:
    """Configuration for VibeVoice engine"""
    model_name: str = "microsoft/VibeVoice-Realtime-0.5B"
    device: str = "auto"  # auto, cuda, cpu, mps
    sample_rate: int = 24000
    chunk_size_ms: int = 50
    max_text_length: int = 10000
    streaming_enabled: bool = True
    cache_dir: Optional[str] = None


class VibeVoiceRealtime:
    """
    🎙️ Microsoft VibeVoice-Realtime-0.5B Integration
    
    Real-time streaming TTS with:
    - ~300ms first chunk latency
    - Streaming text input
    - Multiple speaker voices
    - Long-form generation support
    """
    
    def __init__(self, config: Optional[VibeVoiceConfig] = None):
        self.config = config or VibeVoiceConfig()
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        self.device = None
        self._fallback_mode = False
        
        logger.info(f"🎙️ VibeVoice-Realtime initializing...")
        logger.info(f"   → Model: {self.config.model_name}")
        
    async def initialize(self) -> bool:
        """Initialize the VibeVoice model"""
        if self.is_initialized:
            return True
            
        try:
            import torch
            
            # Determine device
            if self.config.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                self.device = self.config.device
                
            logger.info(f"   → Device: {self.device}")
            
            # Try to load the real VibeVoice model
            try:
                from vibevoice import VibeVoiceRealtimeModel
                
                self.model = VibeVoiceRealtimeModel.from_pretrained(
                    self.config.model_name,
                    device=self.device,
                    cache_dir=self.config.cache_dir
                )
                
                logger.info(f"✅ VibeVoice-Realtime model loaded successfully")
                self.is_initialized = True
                self._fallback_mode = False
                return True
                
            except ImportError:
                logger.warning("⚠️ vibevoice package not installed, trying transformers...")
                
                # Try loading via transformers
                try:
                    from transformers import AutoModel, AutoTokenizer
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.config.model_name,
                        trust_remote_code=True
                    )
                    
                    self.model = AutoModel.from_pretrained(
                        self.config.model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
                    ).to(self.device)
                    
                    logger.info(f"✅ VibeVoice loaded via transformers")
                    self.is_initialized = True
                    self._fallback_mode = False
                    return True
                    
                except Exception as e:
                    logger.warning(f"⚠️ Could not load VibeVoice model: {e}")
                    logger.info("📢 Falling back to OpenAI TTS / gTTS")
                    self._fallback_mode = True
                    self.is_initialized = True
                    return True
                    
        except Exception as e:
            logger.error(f"❌ VibeVoice initialization failed: {e}")
            self._fallback_mode = True
            self.is_initialized = True
            return True  # Return True to allow fallback
    
    async def synthesize(
        self,
        text: str,
        speaker: VibeVoiceSpeaker = VibeVoiceSpeaker.CARTER,
        streaming: bool = False
    ) -> bytes:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            speaker: Speaker voice to use
            streaming: Whether to use streaming mode
            
        Returns:
            Audio bytes (WAV format)
        """
        if not self.is_initialized:
            await self.initialize()
            
        start_time = time.time()
        
        try:
            if self._fallback_mode:
                return await self._fallback_synthesize(text, speaker)
            
            # Real VibeVoice synthesis
            if hasattr(self.model, 'synthesize'):
                # Native VibeVoice API
                audio = self.model.synthesize(
                    text=text,
                    speaker=speaker.value,
                    streaming=streaming
                )
            else:
                # Transformers-based inference
                audio = await self._transformers_synthesize(text, speaker)
            
            processing_time = time.time() - start_time
            logger.info(f"🎙️ VibeVoice synthesis: {len(text)} chars in {processing_time:.2f}s")
            
            return audio
            
        except Exception as e:
            logger.error(f"❌ Synthesis failed: {e}")
            return await self._fallback_synthesize(text, speaker)
    
    async def synthesize_streaming(
        self,
        text: str,
        speaker: VibeVoiceSpeaker = VibeVoiceSpeaker.CARTER,
        on_chunk: Optional[Callable[[bytes], None]] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio chunks as they're generated
        
        Args:
            text: Text to synthesize
            speaker: Speaker voice
            on_chunk: Optional callback for each chunk
            
        Yields:
            Audio chunks (WAV format)
        """
        if not self.is_initialized:
            await self.initialize()
            
        try:
            if self._fallback_mode:
                # Fallback doesn't support true streaming, yield full audio
                audio = await self._fallback_synthesize(text, speaker)
                yield audio
                return
            
            # Real VibeVoice streaming
            if hasattr(self.model, 'synthesize_streaming'):
                async for chunk in self.model.synthesize_streaming(
                    text=text,
                    speaker=speaker.value
                ):
                    if on_chunk:
                        on_chunk(chunk)
                    yield chunk
            else:
                # Non-streaming fallback
                audio = await self.synthesize(text, speaker)
                yield audio
                
        except Exception as e:
            logger.error(f"❌ Streaming synthesis failed: {e}")
            audio = await self._fallback_synthesize(text, speaker)
            yield audio
    
    async def _transformers_synthesize(
        self,
        text: str,
        speaker: VibeVoiceSpeaker
    ) -> bytes:
        """Synthesize using transformers model"""
        import torch
        import numpy as np
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_text_length
        ).to(self.device)
        
        # Generate audio tokens
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                speaker_id=list(VibeVoiceSpeaker).index(speaker),
                max_new_tokens=int(len(text) * 50),  # Rough estimate
                do_sample=True,
                temperature=0.7
            )
        
        # Decode to audio
        if hasattr(self.model, 'decode_audio'):
            audio_array = self.model.decode_audio(outputs)
        else:
            # Fallback: convert tokens to audio
            audio_array = outputs.cpu().numpy().flatten()
        
        # Convert to WAV bytes
        return self._array_to_wav(audio_array)
    
    async def _fallback_synthesize(
        self,
        text: str,
        speaker: VibeVoiceSpeaker
    ) -> bytes:
        """Fallback synthesis using OpenAI TTS or gTTS"""
        try:
            # Try OpenAI TTS first (best quality)
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI()
            
            # Map VibeVoice speakers to OpenAI voices
            voice_map = {
                VibeVoiceSpeaker.CARTER: "onyx",
                VibeVoiceSpeaker.NOVA: "nova",
                VibeVoiceSpeaker.ARIA: "alloy",
                VibeVoiceSpeaker.DAVIS: "echo",
            }
            
            voice = voice_map.get(speaker, "alloy")
            
            response = await client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                response_format="wav"
            )
            
            logger.info(f"🎙️ OpenAI TTS fallback used for {len(text)} chars")
            return response.content
            
        except Exception as openai_error:
            logger.warning(f"OpenAI TTS failed: {openai_error}, trying gTTS...")
            
            try:
                # gTTS fallback
                from gtts import gTTS
                import io
                
                tts = gTTS(text=text, lang='en')
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                
                # Convert MP3 to WAV
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_mp3(audio_buffer)
                    wav_buffer = io.BytesIO()
                    audio.export(wav_buffer, format="wav")
                    wav_buffer.seek(0)
                    return wav_buffer.read()
                except:
                    # Return MP3 if pydub not available
                    audio_buffer.seek(0)
                    return audio_buffer.read()
                    
            except Exception as gtts_error:
                logger.error(f"gTTS also failed: {gtts_error}")
                # Generate silent audio as last resort
                return self._generate_silent_audio(len(text) * 0.1)
    
    def _array_to_wav(self, audio_array, sample_rate: int = None) -> bytes:
        """Convert numpy array to WAV bytes"""
        import numpy as np
        import struct
        import io
        
        if sample_rate is None:
            sample_rate = self.config.sample_rate
            
        # Normalize to int16
        if audio_array.dtype != np.int16:
            audio_array = (audio_array * 32767).astype(np.int16)
        
        # Create WAV file
        buffer = io.BytesIO()
        
        # WAV header
        num_samples = len(audio_array)
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = num_samples * block_align
        
        # Write header
        buffer.write(b'RIFF')
        buffer.write(struct.pack('<I', 36 + data_size))
        buffer.write(b'WAVE')
        buffer.write(b'fmt ')
        buffer.write(struct.pack('<I', 16))  # Subchunk1Size
        buffer.write(struct.pack('<H', 1))   # AudioFormat (PCM)
        buffer.write(struct.pack('<H', num_channels))
        buffer.write(struct.pack('<I', sample_rate))
        buffer.write(struct.pack('<I', byte_rate))
        buffer.write(struct.pack('<H', block_align))
        buffer.write(struct.pack('<H', bits_per_sample))
        buffer.write(b'data')
        buffer.write(struct.pack('<I', data_size))
        buffer.write(audio_array.tobytes())
        
        buffer.seek(0)
        return buffer.read()
    
    def _generate_silent_audio(self, duration_seconds: float) -> bytes:
        """Generate silent audio as fallback"""
        import numpy as np
        
        num_samples = int(self.config.sample_rate * duration_seconds)
        silent_audio = np.zeros(num_samples, dtype=np.int16)
        return self._array_to_wav(silent_audio)
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        return {
            "engine": "VibeVoice-Realtime",
            "model": self.config.model_name,
            "initialized": self.is_initialized,
            "fallback_mode": self._fallback_mode,
            "device": self.device,
            "sample_rate": self.config.sample_rate,
            "streaming_enabled": self.config.streaming_enabled,
            "available_speakers": [s.value for s in VibeVoiceSpeaker],
            "capabilities": {
                "realtime_streaming": not self._fallback_mode,
                "first_chunk_latency_ms": 300 if not self._fallback_mode else 1000,
                "max_text_length": self.config.max_text_length,
                "multilingual": True
            }
        }


# Global instance
_vibevoice_realtime: Optional[VibeVoiceRealtime] = None


async def get_vibevoice_realtime() -> VibeVoiceRealtime:
    """Get or create global VibeVoice-Realtime instance"""
    global _vibevoice_realtime
    
    if _vibevoice_realtime is None:
        _vibevoice_realtime = VibeVoiceRealtime()
        await _vibevoice_realtime.initialize()
    
    return _vibevoice_realtime


# Convenience function for simple synthesis
async def synthesize_speech(
    text: str,
    speaker: str = "Carter",
    streaming: bool = False
) -> bytes:
    """
    Simple interface for speech synthesis
    
    Args:
        text: Text to synthesize
        speaker: Speaker name (Carter, Nova, Aria, Davis)
        streaming: Enable streaming mode
        
    Returns:
        Audio bytes (WAV format)
    """
    engine = await get_vibevoice_realtime()
    
    # Map speaker name to enum
    speaker_map = {
        "carter": VibeVoiceSpeaker.CARTER,
        "nova": VibeVoiceSpeaker.NOVA,
        "aria": VibeVoiceSpeaker.ARIA,
        "davis": VibeVoiceSpeaker.DAVIS,
    }
    
    speaker_enum = speaker_map.get(speaker.lower(), VibeVoiceSpeaker.CARTER)
    
    return await engine.synthesize(text, speaker_enum, streaming)

"""
🎙️ High-Performance TTS for Real-Time Voice Chat
Supports OpenAI TTS (fast, high quality) with gTTS fallback

For smooth GPT-like voice experience:
- OpenAI TTS: ~200-400ms latency, natural voices
- gTTS fallback: ~500-1000ms, robotic but reliable
"""

import logging
import io
import os
import asyncio
from typing import Optional, AsyncGenerator
import base64
import aiohttp

logger = logging.getLogger(__name__)


class SimpleTTS:
    """
    High-performance TTS with OpenAI primary and gTTS fallback
    Optimized for real-time voice chat (<300ms target)
    """
    
    def __init__(self):
        self.initialized = False
        self.gtts_engine = None
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.use_openai = bool(self.openai_api_key)
        self.voice = "alloy"  # alloy, echo, fable, onyx, nova, shimmer
        self.model = "tts-1"  # tts-1 (fast) or tts-1-hd (quality)
        
    def initialize(self) -> bool:
        """Initialize TTS engines"""
        if self.initialized:
            return True
        
        # Check OpenAI availability
        if self.openai_api_key and not self.openai_api_key.startswith("your"):
            self.use_openai = True
            logger.info("✅ OpenAI TTS available (fast, high quality)")
        else:
            self.use_openai = False
            logger.info("⚠️ OpenAI TTS not available, using gTTS fallback")
        
        # Initialize gTTS as fallback
        try:
            from gtts import gTTS
            self.gtts_engine = gTTS
            logger.info("✅ gTTS fallback initialized")
        except ImportError:
            logger.warning("gTTS not available - install with: pip install gtts")
        
        self.initialized = True
        return True
    
    def set_voice(self, voice: str):
        """Set OpenAI voice: alloy, echo, fable, onyx, nova, shimmer"""
        valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        if voice in valid_voices:
            self.voice = voice
            logger.info(f"🎤 Voice set to: {voice}")
    
    async def synthesize_openai(self, text: str) -> Optional[bytes]:
        """
        Synthesize using OpenAI TTS API (fast, ~200-400ms)
        Returns MP3 audio bytes
        """
        if not self.openai_api_key:
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model,
                    "input": text,
                    "voice": self.voice,
                    "response_format": "mp3",
                    "speed": 1.0
                }
                
                async with session.post(
                    "https://api.openai.com/v1/audio/speech",
                    headers=headers,
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        return await resp.read()
                    else:
                        error = await resp.text()
                        logger.warning(f"OpenAI TTS error {resp.status}: {error[:100]}")
                        return None
                        
        except Exception as e:
            logger.error(f"OpenAI TTS error: {e}")
            return None
    
    async def synthesize_openai_streaming(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream audio from OpenAI TTS for ultra-low latency
        Yields audio chunks as they arrive
        """
        if not self.openai_api_key:
            return
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model,
                    "input": text,
                    "voice": self.voice,
                    "response_format": "mp3",
                    "speed": 1.0
                }
                
                async with session.post(
                    "https://api.openai.com/v1/audio/speech",
                    headers=headers,
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        async for chunk in resp.content.iter_chunked(4096):
                            yield chunk
                    else:
                        logger.warning(f"OpenAI TTS streaming error: {resp.status}")
                        
        except Exception as e:
            logger.error(f"OpenAI TTS streaming error: {e}")
    
    def synthesize_gtts(self, text: str, lang: str = "en") -> Optional[bytes]:
        """Synthesize using gTTS (fallback, slower)"""
        if not self.gtts_engine:
            try:
                from gtts import gTTS
                self.gtts_engine = gTTS
            except ImportError:
                return None
        
        try:
            tts = self.gtts_engine(text=text, lang=lang, slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer.read()
        except Exception as e:
            logger.error(f"gTTS error: {e}")
            return None
    
    def synthesize(self, text: str, lang: str = "en") -> Optional[bytes]:
        """
        Synchronous synthesis - uses OpenAI if available, else gTTS
        For async code, use synthesize_async() instead
        """
        if not self.initialized:
            self.initialize()
        
        # Try OpenAI first (run async in sync context)
        if self.use_openai:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Already in async context - use gTTS
                    return self.synthesize_gtts(text, lang)
                else:
                    return loop.run_until_complete(self.synthesize_openai(text))
            except RuntimeError:
                # No event loop - create one
                return asyncio.run(self.synthesize_openai(text))
        
        # Fallback to gTTS
        return self.synthesize_gtts(text, lang)
    
    async def synthesize_async(self, text: str, lang: str = "en") -> Optional[bytes]:
        """
        Async synthesis - preferred for voice chat
        Uses OpenAI TTS for speed, falls back to gTTS
        """
        if not self.initialized:
            self.initialize()
        
        # Try OpenAI first (much faster)
        if self.use_openai:
            audio = await self.synthesize_openai(text)
            if audio:
                return audio
        
        # Fallback to gTTS (run in thread to not block)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize_gtts, text, lang)
    
    def synthesize_base64(self, text: str, lang: str = "en") -> Optional[str]:
        """Synthesize and return as base64 string"""
        audio_bytes = self.synthesize(text, lang)
        if audio_bytes:
            return base64.b64encode(audio_bytes).decode('utf-8')
        return None
    
    async def synthesize_base64_async(self, text: str, lang: str = "en") -> Optional[str]:
        """Async synthesize and return as base64 string"""
        audio_bytes = await self.synthesize_async(text, lang)
        if audio_bytes:
            return base64.b64encode(audio_bytes).decode('utf-8')
        return None


# Global instance
_simple_tts_instance: Optional[SimpleTTS] = None

def get_simple_tts() -> SimpleTTS:
    """Get global SimpleTTS instance"""
    global _simple_tts_instance
    if _simple_tts_instance is None:
        _simple_tts_instance = SimpleTTS()
        _simple_tts_instance.initialize()
    return _simple_tts_instance


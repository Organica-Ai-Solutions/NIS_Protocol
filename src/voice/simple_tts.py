"""
Simple TTS for immediate voice output
Uses gTTS (Google Text-to-Speech) as reliable fallback
"""

import logging
import io
from typing import Optional
import base64

logger = logging.getLogger(__name__)

class SimpleTTS:
    """Simple, reliable TTS using gTTS"""
    
    def __init__(self):
        self.initialized = False
        self.tts_engine = None
        
    def initialize(self) -> bool:
        """Initialize gTTS"""
        if self.initialized:
            return True
            
        try:
            from gtts import gTTS
            self.tts_engine = gTTS
            self.initialized = True
            logger.info("✅ Simple TTS (gTTS) initialized")
            return True
        except ImportError:
            logger.warning("gTTS not available - install with: pip install gtts")
            return False
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}")
            return False
    
    def synthesize(self, text: str, lang: str = "en") -> Optional[bytes]:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            lang: Language code (default: en)
            
        Returns:
            Audio bytes in MP3 format, or None if failed
        """
        if not self.initialized:
            if not self.initialize():
                return None
        
        try:
            # Create TTS object
            tts = self.tts_engine(text=text, lang=lang, slow=False)
            
            # Save to bytes buffer
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            return audio_buffer.read()
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return None
    
    def synthesize_base64(self, text: str, lang: str = "en") -> Optional[str]:
        """Synthesize and return as base64 string"""
        audio_bytes = self.synthesize(text, lang)
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


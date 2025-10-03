"""
Whisper Speech-to-Text Integration
GPT-like voice conversation experience
"""

import io
import base64
import logging
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class WhisperSTT:
    """
    Whisper Speech-to-Text for real-time voice conversation
    Like ChatGPT's voice mode
    """
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper STT
        
        Args:
            model_size: tiny, base, small, medium, large
                       base = good balance (74MB, fast)
                       small = better accuracy (244MB)
        """
        self.model_size = model_size
        self.model = None
        self.initialized = False
        
    def initialize(self):
        """Load Whisper model (lazy loading)"""
        if self.initialized:
            return True
            
        try:
            import whisper
            logger.info(f"🎤 Loading Whisper {self.model_size} model...")
            self.model = whisper.load_model(self.model_size)
            self.initialized = True
            logger.info(f"✅ Whisper {self.model_size} loaded successfully")
            return True
        except ImportError:
            logger.error("❌ Whisper not installed. Run: pip install openai-whisper")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper: {e}")
            return False
    
    async def transcribe_base64(self, audio_base64: str) -> Dict[str, Any]:
        """
        Transcribe base64 audio (like GPT voice mode)
        
        Args:
            audio_base64: Base64 encoded audio
            
        Returns:
            {
                "text": "transcribed text",
                "confidence": 0.95,
                "language": "en",
                "success": True
            }
        """
        try:
            # Initialize if needed
            if not self.initialized:
                if not self.initialize():
                    return {
                        "success": False,
                        "error": "Whisper not available",
                        "text": ""
                    }
            
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_base64)
            
            # Save to temporary file (handles WebM, MP3, WAV, etc.)
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            try:
                # Whisper can handle the file directly
                result = self.model.transcribe(
                    tmp_path,
                    language="en",  # Auto-detect or specify
                    task="transcribe",
                    fp16=False  # CPU compatible
                )
                
                return {
                    "success": True,
                    "text": result["text"].strip(),
                    "language": result.get("language", "en"),
                    "confidence": self._estimate_confidence(result),
                    "segments": len(result.get("segments", []))
                }
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": ""
            }
    
    def _estimate_confidence(self, result: Dict) -> float:
        """Estimate confidence from Whisper result"""
        try:
            segments = result.get("segments", [])
            if not segments:
                return 0.5
            
            # Average probability across segments
            probs = [seg.get("avg_logprob", -1.0) for seg in segments]
            avg_prob = np.mean(probs)
            
            # Convert log prob to confidence (0-1)
            # Whisper logprobs are typically -0.5 to -2.0
            confidence = max(0.0, min(1.0, 1.0 + avg_prob / 2.0))
            return round(confidence, 2)
        except:
            return 0.5


# Global instance
_whisper_instance: Optional[WhisperSTT] = None

def get_whisper_stt(model_size: str = "base") -> WhisperSTT:
    """Get global Whisper STT instance"""
    global _whisper_instance
    if _whisper_instance is None:
        _whisper_instance = WhisperSTT(model_size)
    return _whisper_instance


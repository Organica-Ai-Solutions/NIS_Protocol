"""
Voice Processing Module
Speech-to-Text and Text-to-Speech for GPT-like voice chat

Includes:
- WhisperSTT: OpenAI Whisper for speech-to-text
- VibeVoiceRealtime: Microsoft VibeVoice-Realtime-0.5B for TTS
- SimpleTTS: OpenAI TTS / gTTS fallback
"""

from .whisper_stt import WhisperSTT, get_whisper_stt

# VibeVoice imports (lazy loaded to avoid startup delays)
def get_vibevoice_realtime():
    """Get VibeVoice-Realtime engine (async)"""
    from .vibevoice_realtime import get_vibevoice_realtime as _get
    return _get()

def synthesize_speech(text: str, speaker: str = "Carter", streaming: bool = False):
    """Synthesize speech using VibeVoice (async)"""
    from .vibevoice_realtime import synthesize_speech as _synth
    return _synth(text, speaker, streaming)

__all__ = [
    'WhisperSTT', 
    'get_whisper_stt',
    'get_vibevoice_realtime',
    'synthesize_speech'
]


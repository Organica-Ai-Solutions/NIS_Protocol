"""
Voice Processing Module
Speech-to-Text and Text-to-Speech for GPT-like voice chat
"""

from .whisper_stt import WhisperSTT, get_whisper_stt

__all__ = ['WhisperSTT', 'get_whisper_stt']


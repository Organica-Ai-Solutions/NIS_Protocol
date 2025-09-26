"""
VibeVoice Communication Agent Module

Copyright 2025 Organica AI Solutions
Licensed under Apache License 2.0
"""

from .vibevoice_communication_agent import (
    VibeVoiceCommunicationAgent,
    create_vibevoice_communication_agent,
    TTSRequest,
    TTSResponse,
    SpeakerVoice
)

__all__ = [
    "VibeVoiceCommunicationAgent",
    "create_vibevoice_communication_agent", 
    "TTSRequest",
    "TTSResponse",
    "SpeakerVoice"
]

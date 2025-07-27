from typing import Dict, List, Optional, Tuple
import numpy as np

# Optional audio dependencies
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    sd = None
    SOUNDDEVICE_AVAILABLE = False

try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    BARK_AVAILABLE = True
except ImportError:
    SAMPLE_RATE = 22050  # Default sample rate
    generate_audio = None
    preload_models = None
    BARK_AVAILABLE = False
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import json

@dataclass
class ConversationEntry:
    text: str
    speaker: str
    timestamp: datetime = field(default_factory=datetime.now)
    sentiment: Optional[Dict[str, float]] = None
    audio: Optional[np.ndarray] = None

class CommunicationAgent:
    def __init__(
        self,
        history_size: int = 100,
        default_voice_preset: str = "v2/en_speaker_6",
        sample_rate: int = SAMPLE_RATE
    ):
        # Initialize Bark if available
        if BARK_AVAILABLE and preload_models:
            preload_models()
        
        self.history_size = history_size
        self.conversation_history = deque(maxlen=history_size)
        self.default_voice_preset = default_voice_preset
        self.sample_rate = sample_rate
        
        # Voice presets for different emotional states
        self.voice_presets = {
            "neutral": "v2/en_speaker_6",
            "happy": "v2/en_speaker_7",
            "sad": "v2/en_speaker_8",
            "excited": "v2/en_speaker_9"
        }
        
        # Integration with InterpretationAgent
        self.interpretation_agent = None
        
        # Audio parameters
        self.audio_params = {
            "waveform_temp": 0.7,
            "impulse_temp": 0.7,
            "sample_rate": sample_rate
        }
    
    def set_interpretation_agent(self, agent):
        """Connect to the InterpretationAgent for content analysis."""
        self.interpretation_agent = agent
    
    def generate_speech(
        self,
        text: str,
        voice_preset: Optional[str] = None,
        emotional_state: Optional[str] = None
    ) -> np.ndarray:
        """Generate speech audio from text using Bark."""
        # Select voice preset based on emotional state or use default
        selected_preset = voice_preset
        if emotional_state and emotional_state in self.voice_presets:
            selected_preset = self.voice_presets[emotional_state]
        elif not selected_preset:
            selected_preset = self.default_voice_preset
            
        # Generate audio if available
        if BARK_AVAILABLE and generate_audio:
            audio = generate_audio(
                text,
                history_prompt=selected_preset,
                text_temp=self.audio_params["waveform_temp"],
                waveform_temp=self.audio_params["waveform_temp"]
            )
        else:
            # Return empty array if audio generation not available
            audio = np.array([])
        
        return audio
    
    def speak(
        self,
        text: str,
        voice_preset: Optional[str] = None,
        emotional_state: Optional[str] = None,
        blocking: bool = True
    ):
        """Generate and play speech."""
        audio = self.generate_speech(text, voice_preset, emotional_state)
        
        # Play audio if available
        if SOUNDDEVICE_AVAILABLE and sd and len(audio) > 0:
            sd.play(audio, samplerate=self.sample_rate, blocking=blocking)
        
        # Add to conversation history
        sentiment = None
        if self.interpretation_agent:
            sentiment = self.interpretation_agent.analyze_sentiment(text)
            
        entry = ConversationEntry(
            text=text,
            speaker="system",
            sentiment=sentiment,
            audio=audio
        )
        self.conversation_history.append(entry)
        
        return audio
    
    def add_user_message(self, text: str):
        """Add a user message to the conversation history."""
        sentiment = None
        if self.interpretation_agent:
            sentiment = self.interpretation_agent.analyze_sentiment(text)
            
        entry = ConversationEntry(
            text=text,
            speaker="user",
            sentiment=sentiment
        )
        self.conversation_history.append(entry)
    
    def get_conversation_history(
        self,
        last_n: Optional[int] = None,
        include_audio: bool = False
    ) -> List[Dict]:
        """Get the conversation history as a list of dictionaries."""
        history = list(self.conversation_history)
        if last_n:
            history = history[-last_n:]
            
        result = []
        for entry in history:
            entry_dict = {
                "text": entry.text,
                "speaker": entry.speaker,
                "timestamp": entry.timestamp.isoformat(),
                "sentiment": entry.sentiment
            }
            if include_audio and entry.audio is not None:
                entry_dict["audio"] = entry.audio.tolist()
            result.append(entry_dict)
            
        return result
    
    def update_audio_params(self, params: Dict[str, float]):
        """Update audio generation parameters."""
        self.audio_params.update(params)
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history.clear() 
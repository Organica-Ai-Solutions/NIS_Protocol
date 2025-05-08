"""
Communication Agent

Handles natural communication including text-to-speech synthesis in the NIS Protocol.
"""

from typing import Dict, Any, Optional, List
import time
import os
import numpy as np
import torch
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import sounddevice as sd

from src.core.registry import NISAgent, NISLayer
from src.emotion.emotional_state import EmotionalState
from src.agents.interpretation.interpretation_agent import InterpretationAgent

class CommunicationAgent(NISAgent):
    """
    Agent responsible for natural communication including speech synthesis.
    
    This agent handles text-to-speech conversion and natural communication
    with emotional awareness and voice modulation.
    """
    
    def __init__(
        self,
        agent_id: str = "communicator",
        description: str = "Handles natural communication",
        emotional_state: Optional[EmotionalState] = None,
        interpreter: Optional[InterpretationAgent] = None,
        voice_preset: str = "v2/en_speaker_6",
        output_dir: str = "data/audio_output",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the communication agent.
        
        Args:
            agent_id: Unique identifier for this agent
            description: Human-readable description of the agent's role
            emotional_state: Optional pre-configured emotional state
            interpreter: Optional interpreter agent for content analysis
            voice_preset: Bark voice preset to use
            output_dir: Directory for saving audio files
            device: Device to use for inference ('cuda' or 'cpu')
        """
        super().__init__(agent_id, NISLayer.COMMUNICATION, description)
        self.emotional_state = emotional_state or EmotionalState()
        self.interpreter = interpreter
        self.voice_preset = voice_preset
        self.output_dir = output_dir
        self.device = device
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Bark
        print("Loading Bark models...")
        preload_models()
        
        # Track conversation history
        self.conversation_history = []
        self.max_history = 10
        
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process communication requests.
        
        Args:
            message: Message containing communication operation
                'operation': Operation to perform
                    ('speak', 'synthesize', 'respond', 'adjust_voice')
                'content': Text content to communicate
                + Additional parameters based on operation
                
        Returns:
            Result of the communication operation
        """
        operation = message.get("operation", "").lower()
        content = message.get("content", "")
        
        if not content and operation not in ["adjust_voice"]:
            return {
                "status": "error",
                "error": "No content provided for communication",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        # Process the requested operation
        if operation == "speak":
            return self._speak_content(content, message.get("save_audio", True))
        elif operation == "synthesize":
            return self._synthesize_speech(content)
        elif operation == "respond":
            return self._generate_response(content)
        elif operation == "adjust_voice":
            voice_params = message.get("voice_params", {})
            return self._adjust_voice_parameters(voice_params)
        else:
            return {
                "status": "error",
                "error": f"Unknown operation: {operation}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _speak_content(self, content: str, save_audio: bool = True) -> Dict[str, Any]:
        """
        Convert text to speech and play it.
        
        Args:
            content: Text content to speak
            save_audio: Whether to save the audio file
            
        Returns:
            Speech operation result
        """
        try:
            # Generate audio
            audio_array = generate_audio(
                content,
                history_prompt=self.voice_preset
            )
            
            # Play audio
            sd.play(audio_array, SAMPLE_RATE)
            sd.wait()
            
            # Save audio if requested
            audio_path = None
            if save_audio:
                timestamp = int(time.time())
                audio_path = os.path.join(
                    self.output_dir,
                    f"speech_{timestamp}.wav"
                )
                write_wav(audio_path, SAMPLE_RATE, audio_array)
            
            # Update conversation history
            self._update_history({
                "role": "system",
                "content": content,
                "timestamp": time.time()
            })
            
            return {
                "status": "success",
                "message": "Speech generated and played",
                "audio_path": audio_path,
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Speech generation failed: {str(e)}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _synthesize_speech(self, content: str) -> Dict[str, Any]:
        """
        Generate speech audio without playing it.
        
        Args:
            content: Text content to synthesize
            
        Returns:
            Speech synthesis result
        """
        try:
            # Generate audio
            audio_array = generate_audio(
                content,
                history_prompt=self.voice_preset
            )
            
            # Save audio
            timestamp = int(time.time())
            audio_path = os.path.join(
                self.output_dir,
                f"synthesis_{timestamp}.wav"
            )
            write_wav(audio_path, SAMPLE_RATE, audio_array)
            
            return {
                "status": "success",
                "message": "Speech synthesized",
                "audio_path": audio_path,
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Speech synthesis failed: {str(e)}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _generate_response(self, content: str) -> Dict[str, Any]:
        """
        Generate and speak a response to the given content.
        
        This method uses the interpreter for content analysis and
        emotional awareness in generating responses.
        
        Args:
            content: Input content to respond to
            
        Returns:
            Response operation result
        """
        try:
            # Interpret content if interpreter is available
            interpretation = None
            if self.interpreter:
                interpretation = self.interpreter.process({
                    "operation": "interpret",
                    "content": content
                })
                
                # Update emotional state based on interpretation
                if interpretation["status"] == "success":
                    self.emotional_state = interpretation["emotional_state"]
            
            # Update conversation history
            self._update_history({
                "role": "user",
                "content": content,
                "timestamp": time.time()
            })
            
            # Generate response based on interpretation and emotional state
            response = self._craft_response(content, interpretation)
            
            # Speak the response
            speak_result = self._speak_content(response)
            
            return {
                "status": "success",
                "response": response,
                "interpretation": interpretation,
                "speech_result": speak_result,
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Response generation failed: {str(e)}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _craft_response(
        self,
        content: str,
        interpretation: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Craft an appropriate response based on content and interpretation.
        
        Args:
            content: Input content to respond to
            interpretation: Optional content interpretation
            
        Returns:
            Crafted response text
        """
        # This is a simple response generation implementation
        # In a full system, this would use more sophisticated NLG
        
        if not interpretation:
            return f"I hear you saying: {content}"
            
        sentiment = interpretation.get("sentiment", "NEUTRAL")
        content_type = interpretation.get("content_type", [])
        
        # Adjust response based on content type and sentiment
        if any(c["label"] == "query" for c in content_type):
            return f"Let me help you with your question about {content}"
        elif sentiment == "POSITIVE":
            return f"I'm glad to hear that! {content}"
        elif sentiment == "NEGATIVE":
            return f"I understand your concern about {content}"
        else:
            return f"I understand you're saying: {content}"
    
    def _adjust_voice_parameters(self, voice_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust voice synthesis parameters.
        
        Args:
            voice_params: Dictionary of voice parameters to adjust
                'preset': Voice preset name
                'rate': Speech rate multiplier
                'pitch': Pitch adjustment
                
        Returns:
            Parameter adjustment result
        """
        if "preset" in voice_params:
            self.voice_preset = voice_params["preset"]
        
        return {
            "status": "success",
            "message": "Voice parameters adjusted",
            "current_params": {
                "preset": self.voice_preset
            },
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _update_history(self, entry: Dict[str, Any]) -> None:
        """
        Update conversation history.
        
        Args:
            entry: Conversation entry to add
        """
        self.conversation_history.append(entry)
        
        # Maintain maximum history length
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the conversation history.
        
        Returns:
            List of conversation entries
        """
        return self.conversation_history
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = [] 
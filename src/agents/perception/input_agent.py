"""
Input Agent

Processes non-visual inputs such as text commands, speech, or sensor readings.
Serves as the initial point of contact for data entering the system.
"""

from typing import Dict, Any, List, Optional
import time
import re

from src.core.registry import NISAgent, NISLayer, NISRegistry
from src.emotion.emotional_state import EmotionalState, EmotionalDimension


class InputAgent(NISAgent):
    """
    Agent that processes non-visual inputs (text, speech, sensors, etc.)
    
    The Input Agent is responsible for:
    - Processing incoming text, speech, or data streams
    - Converting raw input into structured formats
    - Initial filtering and prioritization of inputs
    - Detecting potential threats or anomalies in input data
    """
    
    def __init__(
        self,
        agent_id: str = "input",
        description: str = "Processes non-visual inputs and translates to structured data",
        emotional_state: Optional[EmotionalState] = None
    ):
        """
        Initialize a new Input Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            description: Human-readable description of the agent's role
            emotional_state: Optional pre-configured emotional state
        """
        super().__init__(agent_id, NISLayer.PERCEPTION, description)
        self.emotional_state = emotional_state or EmotionalState()
        self.input_history = []
        
        # Keywords that may affect emotional state
        self.urgency_keywords = {
            "urgent", "immediately", "asap", "emergency", "critical", 
            "time-sensitive", "deadline", "now", "quickly"
        }
        
        self.suspicion_keywords = {
            "unusual", "suspicious", "strange", "unexpected", "abnormal",
            "unexpected", "security", "threat", "warning", "alert"
        }
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a non-visual input.
        
        Args:
            message: Message containing input data
                'text': Text input (optional)
                'speech': Speech input (optional)
                'sensor_data': Sensor readings (optional)
        
        Returns:
            Processed input with metadata
        """
        if not self._validate_message(message):
            return {
                "status": "error",
                "error": "Invalid message format or missing required fields",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        # Process based on input type
        structured_data = {}
        metadata = {}
        
        # Process text if present
        if "text" in message:
            text_data, text_metadata = self._process_text(message["text"])
            structured_data.update(text_data)
            metadata.update(text_metadata)
            
            # Update emotional state based on text content
            self._update_emotion_from_text(message["text"])
        
        # Process speech if present
        if "speech" in message:
            speech_data, speech_metadata = self._process_speech(message["speech"])
            structured_data.update(speech_data)
            metadata.update(speech_metadata)
        
        # Process sensor data if present
        if "sensor_data" in message:
            sensor_data, sensor_metadata = self._process_sensor_data(message["sensor_data"])
            structured_data.update(sensor_data)
            metadata.update(sensor_metadata)
        
        # Track input history (limited to most recent 100)
        self.input_history.append({
            "timestamp": time.time(),
            "structured_data": structured_data
        })
        if len(self.input_history) > 100:
            self.input_history.pop(0)
        
        return {
            "status": "success",
            "structured_data": structured_data,
            "metadata": metadata,
            "emotional_state": self.emotional_state.get_state(),
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _validate_message(self, message: Dict[str, Any]) -> bool:
        """
        Validate incoming message format.
        
        Args:
            message: The message to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(message, dict):
            return False
            
        # Must contain at least one of these data fields
        required_fields = ["text", "speech", "sensor_data"]
        return any(field in message for field in required_fields)
    
    def _process_text(self, text: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process text input and extract structured data.
        
        Args:
            text: Raw text input
            
        Returns:
            Tuple of (structured_data, metadata)
        """
        # Parse commands, entities, intent
        commands = self._extract_commands(text)
        entities = self._extract_entities(text)
        intent = self._determine_intent(text)
        
        structured_data = {
            "commands": commands,
            "entities": entities,
            "intent": intent,
            "original_text": text
        }
        
        metadata = {
            "text_length": len(text),
            "word_count": len(text.split()),
            "language": "en",  # Would detect actual language
            "processing_time": 0.001  # Simulated processing time
        }
        
        return structured_data, metadata
    
    def _process_speech(self, speech: Any) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process speech input and convert to structured data.
        
        Args:
            speech: Raw speech input
            
        Returns:
            Tuple of (structured_data, metadata)
        """
        # In a real implementation, this would use speech recognition
        # Here we just simulate the result
        
        structured_data = {
            "transcribed_text": "Simulated speech transcription",
            "commands": [],
            "entities": [],
            "intent": "unknown"
        }
        
        metadata = {
            "audio_length": 0.0,  # Would be actual audio length in seconds
            "confidence": 0.9,  # Would be actual transcription confidence
            "language": "en"  # Would be detected language
        }
        
        return structured_data, metadata
    
    def _process_sensor_data(self, sensor_data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process sensor data input.
        
        Args:
            sensor_data: Raw sensor readings
            
        Returns:
            Tuple of (structured_data, metadata)
        """
        # In a real implementation, this would process and normalize sensor readings
        
        structured_data = {
            "normalized_readings": {},
            "anomalies": []
        }
        
        for sensor_id, reading in sensor_data.items():
            # Apply some simulated normalization
            structured_data["normalized_readings"][sensor_id] = reading
            
            # Detect anomalies (in this simulation, just check if value > 100)
            if isinstance(reading, (int, float)) and reading > 100:
                structured_data["anomalies"].append({
                    "sensor_id": sensor_id,
                    "reading": reading,
                    "severity": "high"
                })
        
        metadata = {
            "sensor_count": len(sensor_data),
            "anomaly_count": len(structured_data["anomalies"]),
            "timestamp": time.time()
        }
        
        return structured_data, metadata
    
    def _extract_commands(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract command structures from text.
        
        Args:
            text: Text to extract commands from
            
        Returns:
            List of command objects
        """
        # Simple command extraction logic
        # In a real implementation, this would use NLP techniques
        
        commands = []
        
        # Look for common command patterns
        if re.search(r"^(start|begin|run)\s+(\w+)", text, re.IGNORECASE):
            match = re.search(r"^(start|begin|run)\s+(\w+)", text, re.IGNORECASE)
            if match:
                commands.append({
                    "action": match.group(1).lower(),
                    "target": match.group(2).lower(),
                    "args": []
                })
        
        return commands
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of entity objects
        """
        # Simple entity extraction
        # In a real implementation, would use NER models
        
        entities = []
        
        # Simple date pattern
        date_pattern = r"\d{1,2}/\d{1,2}/\d{2,4}"
        for match in re.finditer(date_pattern, text):
            entities.append({
                "type": "date",
                "value": match.group(0),
                "position": (match.start(), match.end())
            })
        
        # Simple email pattern
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        for match in re.finditer(email_pattern, text):
            entities.append({
                "type": "email",
                "value": match.group(0),
                "position": (match.start(), match.end())
            })
        
        return entities
    
    def _determine_intent(self, text: str) -> str:
        """
        Determine the intent of the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Intent classification
        """
        # Basic intent detection based on keywords
        # Real implementation would use intent classification models
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["help", "assist", "support"]):
            return "help_request"
        
        if any(word in text_lower for word in ["create", "make", "build", "start"]):
            return "creation"
        
        if any(word in text_lower for word in ["delete", "remove", "destroy"]):
            return "deletion"
        
        if any(word in text_lower for word in ["update", "change", "modify"]):
            return "modification"
        
        if any(word in text_lower for word in ["find", "search", "locate", "where"]):
            return "query"
        
        return "unknown"
    
    def _update_emotion_from_text(self, text: str) -> None:
        """
        Update emotional state based on text content.
        
        Args:
            text: Text to analyze for emotional cues
        """
        text_lower = text.lower()
        
        # Check for urgency keywords
        urgency_score = sum(1 for word in self.urgency_keywords if word in text_lower)
        if urgency_score > 0:
            # Scale to 0.6-0.9 range based on number of matches
            normalized_urgency = 0.6 + min(0.3, urgency_score * 0.1)
            self.emotional_state.update(EmotionalDimension.URGENCY.value, normalized_urgency)
        
        # Check for suspicion keywords
        suspicion_score = sum(1 for word in self.suspicion_keywords if word in text_lower)
        if suspicion_score > 0:
            # Scale to 0.6-0.9 range based on number of matches
            normalized_suspicion = 0.6 + min(0.3, suspicion_score * 0.1)
            self.emotional_state.update(EmotionalDimension.SUSPICION.value, normalized_suspicion) 
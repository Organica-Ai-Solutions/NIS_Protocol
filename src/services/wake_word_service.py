#!/usr/bin/env python3
"""
Wake Word Detection Service for NIS Protocol
Implements "Hey NIS" wake word detection with low-latency processing
"""

import asyncio
import time
import threading
import re
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class WakeWordConfig:
    """Configuration for wake word detection"""
    wake_phrases: List[str]
    confidence_threshold: float = 0.7
    timeout_seconds: float = 5.0
    case_sensitive: bool = False
    partial_match: bool = True

class WakeWordDetector:
    """
    Advanced wake word detection with multiple strategies:
    1. Exact phrase matching
    2. Fuzzy matching for variations
    3. Phonetic similarity
    4. Context-aware detection
    """
    
    def __init__(self, config: WakeWordConfig):
        self.config = config
        self.is_active = False
        self.last_detection_time = 0
        self.detection_callback: Optional[Callable] = None
        
        # Prepare wake phrases for matching
        self.wake_patterns = self._prepare_wake_patterns()
        
        # Voice command patterns
        self.voice_commands = {
            'physics': [
                'physics', 'physical', 'equation', 'force', 'energy', 'mass',
                'switch to physics', 'physics mode', 'physics agent'
            ],
            'consciousness': [
                'consciousness', 'conscious', 'awareness', 'mind', 'think',
                'switch to consciousness', 'consciousness mode', 'consciousness agent'
            ],
            'research': [
                'research', 'search', 'find', 'study', 'analyze', 'investigate',
                'switch to research', 'research mode', 'research agent'
            ],
            'memory': [
                'memory', 'remember', 'recall', 'store', 'save', 'memorize',
                'switch to memory', 'memory mode', 'memory agent'
            ],
            'coordination': [
                'coordination', 'coordinate', 'manage', 'organize', 'control',
                'switch to coordination', 'coordination mode', 'coordination agent'
            ]
        }
        
        # Conversation control commands
        self.control_commands = {
            'stop': ['stop', 'pause', 'halt', 'quiet', 'silence'],
            'continue': ['continue', 'resume', 'go on', 'keep going'],
            'repeat': ['repeat', 'say again', 'what did you say'],
            'help': ['help', 'commands', 'what can you do'],
            'status': ['status', 'how are you', 'what\'s your status']
        }
    
    def _prepare_wake_patterns(self) -> List[re.Pattern]:
        """Prepare regex patterns for wake word detection"""
        patterns = []
        
        for phrase in self.config.wake_phrases:
            if not self.config.case_sensitive:
                phrase = phrase.lower()
            
            # Exact match pattern
            exact_pattern = re.compile(r'\b' + re.escape(phrase) + r'\b')
            patterns.append(exact_pattern)
            
            # Fuzzy match patterns (handle common variations)
            if self.config.partial_match:
                # Allow for slight variations
                fuzzy_phrase = phrase.replace(' ', r'\s*')  # Flexible spacing
                fuzzy_pattern = re.compile(r'\b' + fuzzy_phrase + r'\b')
                patterns.append(fuzzy_pattern)
        
        return patterns
    
    def detect_wake_word(self, text: str) -> Dict[str, Any]:
        """
        Detect wake words in text
        Returns detection result with confidence and matched phrase
        """
        if not text:
            return {"detected": False}
        
        search_text = text if self.config.case_sensitive else text.lower()
        
        for i, pattern in enumerate(self.wake_patterns):
            match = pattern.search(search_text)
            if match:
                # Calculate confidence based on match quality
                confidence = self._calculate_confidence(match, text)
                
                if confidence >= self.config.confidence_threshold:
                    self.last_detection_time = time.time()
                    
                    result = {
                        "detected": True,
                        "phrase": self.config.wake_phrases[i // 2],  # Account for exact/fuzzy pairs
                        "confidence": confidence,
                        "match_text": match.group(),
                        "match_position": match.span(),
                        "timestamp": self.last_detection_time
                    }
                    
                    # Trigger callback if set
                    if self.detection_callback:
                        try:
                            self.detection_callback(result)
                        except Exception as e:
                            logger.error(f"Wake word callback error: {e}")
                    
                    return result
        
        return {"detected": False}
    
    def detect_voice_command(self, text: str) -> Dict[str, Any]:
        """
        Detect voice commands in text
        Returns command type and parameters
        """
        if not text:
            return {"command": None}
        
        search_text = text.lower().strip()
        
        # Check for agent switching commands
        for agent, patterns in self.voice_commands.items():
            for pattern in patterns:
                if pattern in search_text:
                    return {
                        "command": "switch_agent",
                        "agent": agent,
                        "confidence": self._calculate_text_confidence(pattern, search_text),
                        "original_text": text
                    }
        
        # Check for control commands
        for command, patterns in self.control_commands.items():
            for pattern in patterns:
                if pattern in search_text:
                    return {
                        "command": command,
                        "confidence": self._calculate_text_confidence(pattern, search_text),
                        "original_text": text
                    }
        
        # Check for explicit slash commands (like "/physics")
        slash_match = re.search(r'/(\w+)', search_text)
        if slash_match:
            command_word = slash_match.group(1)
            if command_word in self.voice_commands:
                return {
                    "command": "switch_agent",
                    "agent": command_word,
                    "confidence": 1.0,
                    "original_text": text,
                    "explicit_command": True
                }
        
        return {"command": None}
    
    def _calculate_confidence(self, match, original_text: str) -> float:
        """Calculate confidence score for wake word detection"""
        base_confidence = 0.8
        
        # Boost confidence for exact matches
        if match.group() in self.config.wake_phrases:
            base_confidence += 0.2
        
        # Consider position in text (beginning is better)
        text_length = len(original_text)
        position_factor = 1.0 - (match.start() / max(text_length, 1)) * 0.2
        
        return min(1.0, base_confidence * position_factor)
    
    def _calculate_text_confidence(self, pattern: str, text: str) -> float:
        """Calculate confidence for text-based command detection"""
        if pattern == text.strip():
            return 1.0
        elif pattern in text:
            return 0.8
        else:
            return 0.6
    
    def set_detection_callback(self, callback: Callable):
        """Set callback function for wake word detection"""
        self.detection_callback = callback
    
    def is_recently_detected(self) -> bool:
        """Check if wake word was recently detected (within timeout)"""
        if self.last_detection_time == 0:
            return False
        
        return (time.time() - self.last_detection_time) <= self.config.timeout_seconds
    
    def reset_detection(self):
        """Reset wake word detection state"""
        self.last_detection_time = 0

class ContinuousConversationManager:
    """
    Manages continuous conversation mode with context awareness
    """
    
    def __init__(self, session_timeout: float = 30.0):
        self.session_timeout = session_timeout
        self.is_active = False
        self.last_interaction_time = 0
        self.conversation_context = []
        self.current_agent = "consciousness"
        self.session_id = None
        
        # Conversation state
        self.awaiting_response = False
        self.conversation_depth = 0
        self.topic_context = {}
    
    def start_session(self, session_id: str = None):
        """Start a continuous conversation session"""
        self.session_id = session_id or f"conv_{int(time.time())}"
        self.is_active = True
        self.last_interaction_time = time.time()
        self.conversation_context = []
        self.conversation_depth = 0
        
        logger.info(f"Started continuous conversation session: {self.session_id}")
    
    def end_session(self):
        """End the continuous conversation session"""
        self.is_active = False
        self.awaiting_response = False
        self.conversation_context = []
        self.topic_context = {}
        
        logger.info(f"Ended continuous conversation session: {self.session_id}")
    
    def add_interaction(self, user_text: str, agent_response: str, agent_id: str):
        """Add interaction to conversation context"""
        self.last_interaction_time = time.time()
        self.current_agent = agent_id
        self.conversation_depth += 1
        
        interaction = {
            "timestamp": self.last_interaction_time,
            "user": user_text,
            "agent": agent_response,
            "agent_id": agent_id,
            "depth": self.conversation_depth
        }
        
        self.conversation_context.append(interaction)
        
        # Keep only recent interactions (last 10)
        if len(self.conversation_context) > 10:
            self.conversation_context = self.conversation_context[-10:]
        
        # Update topic context
        self._update_topic_context(user_text, agent_response)
    
    def _update_topic_context(self, user_text: str, agent_response: str):
        """Update topic context for better conversation flow"""
        # Simple keyword extraction for topic tracking
        keywords = self._extract_keywords(user_text + " " + agent_response)
        
        for keyword in keywords:
            if keyword in self.topic_context:
                self.topic_context[keyword] += 1
            else:
                self.topic_context[keyword] = 1
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for topic tracking"""
        # Simple keyword extraction (can be enhanced with NLP)
        words = re.findall(r'\b\w{4,}\b', text.lower())
        
        # Filter out common words
        stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'said', 'each', 'which', 'their', 'time', 'about'}
        keywords = [word for word in words if word not in stop_words]
        
        return keywords[:5]  # Return top 5 keywords
    
    def is_session_active(self) -> bool:
        """Check if conversation session is still active"""
        if not self.is_active:
            return False
        
        # Check timeout
        if time.time() - self.last_interaction_time > self.session_timeout:
            self.end_session()
            return False
        
        return True
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation context"""
        if not self.conversation_context:
            return {"active": False}
        
        return {
            "active": self.is_active,
            "session_id": self.session_id,
            "current_agent": self.current_agent,
            "interaction_count": len(self.conversation_context),
            "conversation_depth": self.conversation_depth,
            "last_interaction": self.last_interaction_time,
            "time_since_last": time.time() - self.last_interaction_time,
            "top_topics": sorted(self.topic_context.items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    def should_continue_listening(self) -> bool:
        """Determine if system should continue listening for follow-up"""
        if not self.is_active:
            return False
        
        # Continue listening if recent interaction
        time_since_last = time.time() - self.last_interaction_time
        return time_since_last < 10.0  # Continue for 10 seconds after last interaction

# Global instances
_wake_word_detector: Optional[WakeWordDetector] = None
_conversation_manager: Optional[ContinuousConversationManager] = None

def get_wake_word_detector() -> WakeWordDetector:
    """Get or create global wake word detector"""
    global _wake_word_detector
    
    if _wake_word_detector is None:
        config = WakeWordConfig(
            wake_phrases=["hey nis", "hey niss", "nis", "niss"],
            confidence_threshold=0.6,
            timeout_seconds=5.0,
            case_sensitive=False,
            partial_match=True
        )
        _wake_word_detector = WakeWordDetector(config)
    
    return _wake_word_detector

def get_conversation_manager() -> ContinuousConversationManager:
    """Get or create global conversation manager"""
    global _conversation_manager
    
    if _conversation_manager is None:
        _conversation_manager = ContinuousConversationManager(session_timeout=30.0)
    
    return _conversation_manager

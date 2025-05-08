from typing import Optional, Dict, Any, List
from ..base_neural_agent import NeuralAgent, NeuralLayer, NeuralSignal
from transformers import pipeline
import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class EmotionalState:
    valence: float  # -1 (negative) to 1 (positive)
    arousal: float  # 0 (calm) to 1 (excited)
    dominance: float  # 0 (submissive) to 1 (dominant)
    primary_emotion: str
    confidence: float
    timestamp: datetime = datetime.now()

class EmotionalProcessingAgent(NeuralAgent):
    """Agent for processing emotional content and maintaining emotional state"""
    
    def __init__(
        self,
        agent_id: str = "emotional_processor",
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        emotion_decay: float = 0.95,
        memory_influence: float = 0.3
    ):
        super().__init__(
            agent_id=agent_id,
            layer=NeuralLayer.EMOTIONAL,
            description="Processes emotional content and maintains emotional state"
        )
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=-1  # CPU
        )
        
        # Emotional state parameters
        self.emotion_decay = emotion_decay
        self.memory_influence = memory_influence
        
        # Current emotional state
        self.current_state = EmotionalState(
            valence=0.0,
            arousal=0.0,
            dominance=0.5,
            primary_emotion="neutral",
            confidence=1.0
        )
        
        # Emotion mapping
        self.emotion_map = {
            "joy": {"valence": 0.8, "arousal": 0.7, "dominance": 0.7},
            "sadness": {"valence": -0.7, "arousal": 0.3, "dominance": 0.3},
            "anger": {"valence": -0.8, "arousal": 0.9, "dominance": 0.8},
            "fear": {"valence": -0.7, "arousal": 0.8, "dominance": 0.2},
            "surprise": {"valence": 0.5, "arousal": 0.8, "dominance": 0.5},
            "neutral": {"valence": 0.0, "arousal": 0.3, "dominance": 0.5}
        }
        
        # Emotional history
        self.emotional_history: List[EmotionalState] = []
    
    def process_signal(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Process incoming signal and update emotional state"""
        if not isinstance(signal.content, dict):
            return None
            
        # Process active memories for emotional content
        memories = signal.content.get('active_memories', [])
        memory_emotions = []
        
        for memory in memories:
            if isinstance(memory['content'], dict):
                text = memory['content'].get('original_text', '')
                if text:
                    sentiment = self.sentiment_analyzer(text)[0]
                    memory_emotions.append({
                        'sentiment': sentiment,
                        'importance': memory['importance']
                    })
        
        # Update emotional state
        self._update_emotional_state(memory_emotions)
        
        # Add to history
        self.emotional_history.append(self.current_state)
        if len(self.emotional_history) > 100:
            self.emotional_history = self.emotional_history[-100:]
        
        # Generate signal for executive layer
        return NeuralSignal(
            source_layer=self.layer,
            target_layer=NeuralLayer.EXECUTIVE,
            content={
                'emotional_state': self._get_state_dict(),
                'emotional_context': self._analyze_emotional_context(memory_emotions)
            },
            priority=max(self.current_state.arousal, 0.5)  # Higher priority for high arousal
        )
    
    def _update_emotional_state(self, memory_emotions: List[Dict]):
        """Update emotional state based on memory emotions"""
        if not memory_emotions:
            # Decay current state
            self.current_state.valence *= self.emotion_decay
            self.current_state.arousal *= self.emotion_decay
            return
        
        # Calculate weighted average of memory emotions
        total_importance = sum(mem['importance'] for mem in memory_emotions)
        weighted_valence = 0.0
        weighted_arousal = 0.0
        
        for mem in memory_emotions:
            weight = mem['importance'] / total_importance
            sentiment = mem['sentiment']
            
            # Convert sentiment to valence/arousal
            if sentiment['label'] == 'POSITIVE':
                valence = sentiment['score']
                arousal = sentiment['score'] * 0.7  # Positive emotions are moderately arousing
            else:
                valence = -sentiment['score']
                arousal = sentiment['score'] * 0.8  # Negative emotions are more arousing
            
            weighted_valence += valence * weight
            weighted_arousal += arousal * weight
        
        # Update state with memory influence
        self.current_state.valence = (
            self.current_state.valence * (1 - self.memory_influence) +
            weighted_valence * self.memory_influence
        )
        self.current_state.arousal = (
            self.current_state.arousal * (1 - self.memory_influence) +
            weighted_arousal * self.memory_influence
        )
        
        # Update primary emotion based on valence/arousal
        self.current_state.primary_emotion = self._determine_primary_emotion()
        self.current_state.confidence = np.mean([mem['sentiment']['score'] for mem in memory_emotions])
    
    def _determine_primary_emotion(self) -> str:
        """Determine primary emotion based on current valence and arousal"""
        min_distance = float('inf')
        primary_emotion = "neutral"
        
        for emotion, coords in self.emotion_map.items():
            distance = np.sqrt(
                (self.current_state.valence - coords['valence'])**2 +
                (self.current_state.arousal - coords['arousal'])**2
            )
            if distance < min_distance:
                min_distance = distance
                primary_emotion = emotion
        
        return primary_emotion
    
    def _analyze_emotional_context(self, memory_emotions: List[Dict]) -> Dict[str, Any]:
        """Analyze emotional context from memory emotions"""
        if not memory_emotions:
            return {
                'emotional_stability': 1.0,
                'emotional_trend': 'stable',
                'context_confidence': 1.0
            }
        
        # Calculate emotional stability
        valence_std = np.std([
            mem['sentiment']['score'] * (1 if mem['sentiment']['label'] == 'POSITIVE' else -1)
            for mem in memory_emotions
        ])
        
        # Determine emotional trend
        recent_valences = [
            state.valence
            for state in self.emotional_history[-5:]  # Last 5 states
        ]
        if len(recent_valences) >= 2:
            trend = np.mean(np.diff(recent_valences))
            if abs(trend) < 0.1:
                trend_label = 'stable'
            elif trend > 0:
                trend_label = 'improving'
            else:
                trend_label = 'deteriorating'
        else:
            trend_label = 'stable'
        
        return {
            'emotional_stability': 1.0 - min(valence_std, 1.0),
            'emotional_trend': trend_label,
            'context_confidence': np.mean([mem['sentiment']['score'] for mem in memory_emotions])
        }
    
    def _get_state_dict(self) -> Dict[str, Any]:
        """Get current emotional state as dictionary"""
        return {
            'valence': self.current_state.valence,
            'arousal': self.current_state.arousal,
            'dominance': self.current_state.dominance,
            'primary_emotion': self.current_state.primary_emotion,
            'confidence': self.current_state.confidence,
            'timestamp': self.current_state.timestamp.isoformat()
        }
    
    def get_emotional_history(self, last_n: int = None) -> List[Dict[str, Any]]:
        """Get emotional history"""
        history = self.emotional_history
        if last_n:
            history = history[-last_n:]
            
        return [
            {
                'valence': state.valence,
                'arousal': state.arousal,
                'dominance': state.dominance,
                'primary_emotion': state.primary_emotion,
                'confidence': state.confidence,
                'timestamp': state.timestamp.isoformat()
            }
            for state in history
        ]
    
    def reset(self):
        """Reset emotional state"""
        super().reset()
        self.current_state = EmotionalState(
            valence=0.0,
            arousal=0.0,
            dominance=0.5,
            primary_emotion="neutral",
            confidence=1.0
        )
        # Keep emotional history for learning 
from typing import Optional, Dict, Any, List
from ..base_neural_agent import NeuralAgent, NeuralLayer, NeuralSignal
from transformers import AutoModel
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from datetime import datetime

@dataclass
class RecognizedPattern:
    pattern_type: str
    confidence: float
    features: Dict[str, Any]
    timestamp: datetime = datetime.now()

class PatternRecognitionAgent(NeuralAgent):
    """Agent for recognizing patterns in input data"""
    
    def __init__(
        self,
        agent_id: str = "pattern_recognizer",
        model_name: str = "distilbert-base-uncased",
        pattern_threshold: float = 0.7
    ):
        super().__init__(
            agent_id=agent_id,
            layer=NeuralLayer.PERCEPTION,
            description="Recognizes patterns in input data"
        )
        
        # Initialize model
        self.model = AutoModel.from_pretrained(model_name)
        self.pattern_threshold = pattern_threshold
        
        # Pattern memory
        self.known_patterns: Dict[str, torch.Tensor] = {}
        self.pattern_history: List[RecognizedPattern] = []
    
    def process_signal(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Process incoming signal to recognize patterns"""
        if not isinstance(signal.content, dict) or 'tokens' not in signal.content:
            return None
            
        # Extract features
        with torch.no_grad():
            outputs = self.model(**signal.content['tokens'])
            features = outputs.last_hidden_state.mean(dim=1)  # Pool features
            
        # Compare with known patterns
        recognized_patterns = []
        for pattern_name, pattern_features in self.known_patterns.items():
            similarity = F.cosine_similarity(features, pattern_features)
            if similarity > self.pattern_threshold:
                recognized_patterns.append(
                    RecognizedPattern(
                        pattern_type=pattern_name,
                        confidence=float(similarity),
                        features={'similarity': float(similarity)}
                    )
                )
        
        # Extract new patterns if none recognized
        if not recognized_patterns:
            pattern_name = f"pattern_{len(self.known_patterns)}"
            self.known_patterns[pattern_name] = features
            recognized_patterns.append(
                RecognizedPattern(
                    pattern_type="new_pattern",
                    confidence=1.0,
                    features={'pattern_id': pattern_name}
                )
            )
        
        # Update pattern history
        self.pattern_history.extend(recognized_patterns)
        if len(self.pattern_history) > 100:  # Keep last 100 patterns
            self.pattern_history = self.pattern_history[-100:]
        
        # Create signal for memory layer
        return NeuralSignal(
            source_layer=self.layer,
            target_layer=NeuralLayer.MEMORY,
            content={
                'original_text': signal.content['text'],
                'recognized_patterns': recognized_patterns,
                'features': features.tolist()
            },
            priority=max(p.confidence for p in recognized_patterns)
        )
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about recognized patterns"""
        return {
            'known_patterns': len(self.known_patterns),
            'pattern_history': len(self.pattern_history),
            'recent_patterns': [
                {
                    'type': p.pattern_type,
                    'confidence': p.confidence,
                    'timestamp': p.timestamp.isoformat()
                }
                for p in self.pattern_history[-5:]  # Last 5 patterns
            ]
        }
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.pattern_history = []  # Keep known patterns, reset history 
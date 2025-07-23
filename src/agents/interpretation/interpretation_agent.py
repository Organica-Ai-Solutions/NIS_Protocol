"""
Interpretation Agent

Handles semantic interpretation and understanding of information in the NIS Protocol.
"""

from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
from transformers import pipeline

from src.core.registry import NISAgent, NISLayer
from src.emotion.emotional_state import EmotionalState

class InterpretationAgent(NISAgent):
    """
    Agent responsible for semantic interpretation and understanding.
    
    This agent processes and interprets information using natural language
    understanding and semantic analysis capabilities.
    """
    
    def __init__(
        self,
        agent_id: str = "interpreter",
        description: str = "Handles semantic interpretation",
        emotional_state: Optional[EmotionalState] = None,
        model_name: str = "bert-base-uncased",
        confidence_threshold: float = 0.7
    ):
        """
        Initialize the interpretation agent.
        
        Args:
            agent_id: Unique identifier for this agent
            description: Human-readable description of the agent's role
            emotional_state: Optional pre-configured emotional state
            model_name: Name of the transformer model to use
            confidence_threshold: Minimum confidence for interpretations
        """
        super().__init__(agent_id, NISLayer.INTERPRETATION, description)
        self.emotional_state = emotional_state or EmotionalState()
        self.confidence_threshold = confidence_threshold
        
        # Initialize NLP pipelines
        self.sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)
        self.zero_shot_classifier = pipeline("zero-shot-classification", model=model_name)
        self.question_answerer = pipeline("question-answering", model=model_name)
        
        # Cache for recent interpretations
        self.interpretation_cache = {}
        self.cache_size = 100
        
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process interpretation requests.
        
        Args:
            message: Message containing interpretation operation
                'operation': Operation to perform
                    ('analyze_sentiment', 'classify', 'answer_question', 'interpret')
                'content': Text content to interpret
                + Additional parameters based on operation
                
        Returns:
            Result of the interpretation
        """
        operation = message.get("operation", "").lower()
        content = message.get("content", "")
        
        if not content:
            return {
                "status": "error",
                "error": "No content provided for interpretation",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        # Check cache first
        cache_key = f"{operation}:{content}"
        if cache_key in self.interpretation_cache:
            return self.interpretation_cache[cache_key]
        
        # Process the requested operation
        if operation == "analyze_sentiment":
            result = self._analyze_sentiment(content)
        elif operation == "classify":
            labels = message.get("labels", [])
            result = self._classify_content(content, labels)
        elif operation == "answer_question":
            question = message.get("question", "")
            context = message.get("context", "")
            result = self._answer_question(question, context)
        elif operation == "interpret":
            result = self._interpret_content(content)
        else:
            return {
                "status": "error",
                "error": f"Unknown operation: {operation}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
            
        # Update cache
        self._update_cache(cache_key, result)
        
        return result
    
    def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of the given content.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Sentiment analysis result
        """
        try:
            sentiment = self.sentiment_analyzer(content)[0]
            
            # Update emotional state based on sentiment
            if sentiment["label"] == "POSITIVE":
                self.emotional_state.adjust_valence(0.1)
            elif sentiment["label"] == "NEGATIVE":
                self.emotional_state.adjust_valence(-0.1)
            
            return {
                "status": "success",
                "sentiment": sentiment["label"],
                "confidence": sentiment["score"],
                "emotional_state": self.emotional_state.get_state(),
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Sentiment analysis failed: {str(e)}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _classify_content(self, content: str, labels: List[str]) -> Dict[str, Any]:
        """
        Classify content into given categories.
        
        Args:
            content: Text content to classify
            labels: List of possible classification labels
            
        Returns:
            Classification result
        """
        if not labels:
            return {
                "status": "error",
                "error": "No classification labels provided",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
            
        try:
            result = self.zero_shot_classifier(
                content,
                labels,
                multi_label=True
            )
            
            # Filter results by confidence threshold
            confident_labels = [
                {"label": label, "score": score}
                for label, score in zip(result["labels"], result["scores"])
                if score >= self.confidence_threshold
            ]
            
            return {
                "status": "success",
                "classifications": confident_labels,
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Classification failed: {str(e)}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _answer_question(self, question: str, context: str) -> Dict[str, Any]:
        """
        Answer a question based on given context.
        
        Args:
            question: Question to answer
            context: Context to find answer in
            
        Returns:
            Question answering result
        """
        if not question or not context:
            return {
                "status": "error",
                "error": "Both question and context must be provided",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
            
        try:
            result = self.question_answerer(
                question=question,
                context=context
            )
            
            return {
                "status": "success",
                "answer": result["answer"],
                "confidence": result["score"],
                "start": result["start"],
                "end": result["end"],
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Question answering failed: {str(e)}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _interpret_content(self, content: str) -> Dict[str, Any]:
        """
        Perform comprehensive interpretation of content.
        
        This combines multiple interpretation methods for deeper understanding.
        
        Args:
            content: Text content to interpret
            
        Returns:
            Comprehensive interpretation result
        """
        # Get sentiment
        sentiment_result = self._analyze_sentiment(content)
        
        # Classify content type
        content_types = ["factual", "emotional", "instructional", "query"]
        classification_result = self._classify_content(content, content_types)
        
        # Extract key information
        key_info = self._extract_key_information(content)
        
        return {
            "status": "success",
            "sentiment": sentiment_result.get("sentiment"),
            "content_type": classification_result.get("classifications", []),
            "key_information": key_info,
            "emotional_state": self.emotional_state.get_state(),
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _extract_key_information(self, content: str) -> Dict[str, Any]:
        """
        Extract key information from content.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Dictionary of extracted information
        """
        # Information extraction using pattern recognition
        # In a full implementation, this would use NER, relation extraction, etc.
        return {
            "length": len(content),
            "word_count": len(content.split()),
            "has_numbers": any(c.isdigit() for c in content),
            "has_special_chars": any(not c.isalnum() and not c.isspace() for c in content)
        }
    
    def _update_cache(self, key: str, value: Dict[str, Any]) -> None:
        """
        Update the interpretation cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self.interpretation_cache[key] = value
        
        # Remove oldest entries if cache is too large
        if len(self.interpretation_cache) > self.cache_size:
            oldest_key = min(
                self.interpretation_cache.keys(),
                key=lambda k: self.interpretation_cache[k]["timestamp"]
            )
            del self.interpretation_cache[oldest_key]
    
    def adjust_confidence_threshold(self, new_threshold: float) -> None:
        """
        Adjust the confidence threshold for interpretations.
        
        Args:
            new_threshold: New confidence threshold (0-1)
        """
        self.confidence_threshold = max(0.0, min(1.0, new_threshold))
        
    def clear_cache(self) -> None:
        """Clear the interpretation cache."""
        self.interpretation_cache = {} 
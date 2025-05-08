from typing import Dict, List, Optional, Union
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    pipeline
)
from cachetools import TTLCache
from dataclasses import dataclass
from datetime import datetime

@dataclass
class InterpretationResult:
    text: str
    sentiment: Dict[str, float]
    classification: Optional[Dict[str, float]] = None
    answer: Optional[str] = None
    timestamp: datetime = datetime.now()

class InterpretationAgent:
    def __init__(
        self,
        sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        qa_model: str = "distilbert-base-cased-distilled-squad",
        cache_ttl: int = 3600,  # 1 hour cache
        cache_maxsize: int = 1000
    ):
        # Initialize models
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=sentiment_model,
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.qa_pipeline = pipeline(
            "question-answering",
            model=qa_model,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Initialize cache
        self.interpretation_cache = TTLCache(
            maxsize=cache_maxsize,
            ttl=cache_ttl
        )
        
        # Emotional state integration
        self.emotional_state = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze the sentiment of the given text."""
        cache_key = f"sentiment_{text}"
        if cache_key in self.interpretation_cache:
            return self.interpretation_cache[cache_key]
        
        result = self.sentiment_analyzer(text)[0]
        sentiment = {
            "label": result["label"],
            "score": float(result["score"])
        }
        self.interpretation_cache[cache_key] = sentiment
        return sentiment
    
    def answer_question(self, context: str, question: str) -> str:
        """Answer a question based on the given context."""
        cache_key = f"qa_{question}_{context[:100]}"  # Cache using first 100 chars of context
        if cache_key in self.interpretation_cache:
            return self.interpretation_cache[cache_key]
        
        result = self.qa_pipeline(
            question=question,
            context=context
        )
        self.interpretation_cache[cache_key] = result["answer"]
        return result["answer"]
    
    def interpret_text(
        self,
        text: str,
        question: Optional[str] = None,
        context: Optional[str] = None
    ) -> InterpretationResult:
        """Comprehensive text interpretation including sentiment and optional QA."""
        sentiment = self.analyze_sentiment(text)
        
        answer = None
        if question and context:
            answer = self.answer_question(context, question)
        
        return InterpretationResult(
            text=text,
            sentiment=sentiment,
            answer=answer
        )
    
    def update_emotional_state(self, emotional_state: Dict):
        """Update the agent's awareness of the system's emotional state."""
        self.emotional_state = emotional_state
    
    def clear_cache(self):
        """Clear the interpretation cache."""
        self.interpretation_cache.clear() 
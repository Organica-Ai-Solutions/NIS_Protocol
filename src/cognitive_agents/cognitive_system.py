from typing import Dict, List, Optional, Union
from .interpretation_agent import InterpretationAgent
from .communication_agent import CommunicationAgent
from .reasoning_agent import ReasoningAgent, ReasoningStrategy
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CognitiveResponse:
    input_text: str
    interpretation: Dict
    reasoning: Dict
    response_text: str
    timestamp: datetime = datetime.now()

class CognitiveSystem:
    def __init__(
        self,
        interpretation_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        reasoning_model: str = "google/flan-t5-large",
        voice_preset: str = "v2/en_speaker_6"
    ):
        # Initialize agents
        self.interpretation_agent = InterpretationAgent(
            sentiment_model=interpretation_model
        )
        
        self.communication_agent = CommunicationAgent(
            default_voice_preset=voice_preset
        )
        
        self.reasoning_agent = ReasoningAgent(
            model_name=reasoning_model
        )
        
        # Connect agents
        self.communication_agent.set_interpretation_agent(self.interpretation_agent)
        
        # System state
        self.emotional_state = "neutral"
        self.context = {}
    
    def process_input(
        self,
        text: str,
        generate_speech: bool = False,
        reasoning_strategy: Optional[ReasoningStrategy] = None
    ) -> CognitiveResponse:
        """Process input text through all cognitive agents."""
        # Step 1: Interpret the input
        interpretation = self.interpretation_agent.interpret_text(text)
        
        # Update emotional state based on interpretation
        if interpretation.sentiment:
            if interpretation.sentiment["label"] == "POSITIVE":
                self.emotional_state = "happy" if interpretation.sentiment["score"] > 0.8 else "neutral"
            else:
                self.emotional_state = "sad" if interpretation.sentiment["score"] > 0.8 else "neutral"
        
        # Step 2: Apply reasoning
        reasoning_result = self.reasoning_agent.reason(
            text,
            strategy=reasoning_strategy,
            context=self.context
        )
        
        # Step 3: Generate response
        response_text = f"Based on my analysis and reasoning:\n\n"
        response_text += f"Sentiment: {interpretation.sentiment['label']} "
        response_text += f"(confidence: {interpretation.sentiment['score']:.2f})\n\n"
        response_text += f"Reasoning ({reasoning_result.strategy.value}):\n"
        response_text += f"Conclusion: {reasoning_result.conclusion}\n"
        response_text += f"Explanation: {reasoning_result.explanation}\n"
        response_text += f"Confidence: {reasoning_result.confidence:.2f}"
        
        # Step 4: Generate speech if requested
        if generate_speech:
            self.communication_agent.speak(
                response_text,
                emotional_state=self.emotional_state,
                blocking=False
            )
        
        # Add to conversation history
        self.communication_agent.add_user_message(text)
        
        # Update context
        self.context.update({
            "last_input": text,
            "last_sentiment": interpretation.sentiment,
            "last_reasoning": {
                "strategy": reasoning_result.strategy.value,
                "conclusion": reasoning_result.conclusion
            }
        })
        
        return CognitiveResponse(
            input_text=text,
            interpretation={
                "sentiment": interpretation.sentiment,
                "emotional_state": self.emotional_state
            },
            reasoning={
                "strategy": reasoning_result.strategy.value,
                "conclusion": reasoning_result.conclusion,
                "explanation": reasoning_result.explanation,
                "confidence": reasoning_result.confidence,
                "chain": reasoning_result.reasoning_chain
            },
            response_text=response_text
        )
    
    def get_conversation_history(
        self,
        last_n: Optional[int] = None
    ) -> List[Dict]:
        """Get the conversation history."""
        return self.communication_agent.get_conversation_history(last_n)
    
    def clear_state(self):
        """Clear all agent states and caches."""
        self.interpretation_agent.clear_cache()
        self.reasoning_agent.clear_cache()
        self.communication_agent.clear_history()
        self.emotional_state = "neutral"
        self.context = {} 
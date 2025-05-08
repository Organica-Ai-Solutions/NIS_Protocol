from typing import Dict, List, Optional, Union
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from cachetools import TTLCache

class ReasoningStrategy(Enum):
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    CAUSAL = "causal"
    ANALOGICAL = "analogical"

@dataclass
class ReasoningResult:
    strategy: ReasoningStrategy
    input_text: str
    conclusion: str
    explanation: str
    confidence: float
    reasoning_chain: List[str]
    timestamp: datetime = datetime.now()

class ReasoningAgent:
    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        cache_ttl: int = 3600,  # 1 hour cache
        cache_maxsize: int = 1000,
        device: Optional[str] = None
    ):
        # Initialize model and tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Initialize cache
        self.reasoning_cache = TTLCache(
            maxsize=cache_maxsize,
            ttl=cache_ttl
        )
        
        # Strategy prompts
        self.strategy_prompts = {
            ReasoningStrategy.DEDUCTIVE: "Using deductive reasoning, if {premises}, then what follows?",
            ReasoningStrategy.INDUCTIVE: "Based on these observations: {observations}, what general conclusion can we draw?",
            ReasoningStrategy.ABDUCTIVE: "Given this observation: {observation}, what is the best explanation?",
            ReasoningStrategy.CAUSAL: "What are the likely causes and effects of: {event}?",
            ReasoningStrategy.ANALOGICAL: "How is {source} similar to {target}, and what can we learn from this comparison?"
        }
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 512,
        num_beams: int = 4,
        temperature: float = 0.7
    ) -> str:
        """Generate text using FLAN-T5."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def apply_reasoning_strategy(
        self,
        strategy: ReasoningStrategy,
        input_text: str,
        context: Optional[Dict] = None
    ) -> ReasoningResult:
        """Apply a specific reasoning strategy to the input text."""
        cache_key = f"{strategy.value}_{input_text}"
        if cache_key in self.reasoning_cache:
            return self.reasoning_cache[cache_key]
        
        # Prepare prompt based on strategy
        if strategy == ReasoningStrategy.DEDUCTIVE:
            prompt = self.strategy_prompts[strategy].format(premises=input_text)
        elif strategy == ReasoningStrategy.INDUCTIVE:
            prompt = self.strategy_prompts[strategy].format(observations=input_text)
        elif strategy == ReasoningStrategy.ABDUCTIVE:
            prompt = self.strategy_prompts[strategy].format(observation=input_text)
        elif strategy == ReasoningStrategy.CAUSAL:
            prompt = self.strategy_prompts[strategy].format(event=input_text)
        elif strategy == ReasoningStrategy.ANALOGICAL:
            source, target = input_text.split("|")
            prompt = self.strategy_prompts[strategy].format(source=source.strip(), target=target.strip())
        
        # Generate reasoning steps
        reasoning_chain = []
        
        # Step 1: Initial analysis
        analysis = self.generate_text(f"Analyze this: {prompt}")
        reasoning_chain.append(f"Analysis: {analysis}")
        
        # Step 2: Generate potential conclusions
        conclusions = self.generate_text(f"Based on this analysis: {analysis}, what conclusions can we draw?")
        reasoning_chain.append(f"Potential conclusions: {conclusions}")
        
        # Step 3: Evaluate constraints and context
        if context:
            context_evaluation = self.generate_text(
                f"Given this context: {context}, how does it affect our conclusions: {conclusions}?"
            )
            reasoning_chain.append(f"Context evaluation: {context_evaluation}")
        
        # Step 4: Final conclusion and explanation
        final_prompt = f"Given this reasoning chain: {' '.join(reasoning_chain)}, provide a final conclusion and explanation."
        final_output = self.generate_text(final_prompt)
        
        # Split final output into conclusion and explanation
        parts = final_output.split("Explanation:", 1)
        conclusion = parts[0].strip()
        explanation = parts[1].strip() if len(parts) > 1 else "No detailed explanation available."
        
        # Calculate confidence based on consistency and clarity
        confidence = min(
            len(reasoning_chain) / 4.0,  # More steps = higher confidence
            0.95  # Cap at 95% confidence
        )
        
        result = ReasoningResult(
            strategy=strategy,
            input_text=input_text,
            conclusion=conclusion,
            explanation=explanation,
            confidence=confidence,
            reasoning_chain=reasoning_chain
        )
        
        self.reasoning_cache[cache_key] = result
        return result
    
    def reason(
        self,
        input_text: str,
        strategy: Optional[ReasoningStrategy] = None,
        context: Optional[Dict] = None
    ) -> ReasoningResult:
        """Apply reasoning to the input text, automatically selecting strategy if none provided."""
        if not strategy:
            # Determine the best strategy based on input structure and content
            strategy_prompt = f"What type of reasoning (deductive, inductive, abductive, causal, or analogical) would be most appropriate for this input: {input_text}?"
            strategy_suggestion = self.generate_text(strategy_prompt).lower()
            
            for s in ReasoningStrategy:
                if s.value in strategy_suggestion:
                    strategy = s
                    break
            
            # Default to deductive if no clear match
            strategy = strategy or ReasoningStrategy.DEDUCTIVE
        
        return self.apply_reasoning_strategy(strategy, input_text, context)
    
    def clear_cache(self):
        """Clear the reasoning cache."""
        self.reasoning_cache.clear() 
from typing import Optional, Dict, Any
from ..base_neural_agent import NeuralAgent, NeuralLayer, NeuralSignal
from transformers import AutoTokenizer

class TextInputAgent(NeuralAgent):
    """Agent for processing text input in the sensory layer"""
    
    def __init__(
        self,
        agent_id: str = "text_input",
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 512
    ):
        super().__init__(
            agent_id=agent_id,
            layer=NeuralLayer.SENSORY,
            description="Processes text input and tokenization"
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Track input statistics
        self.processed_inputs = 0
        self.avg_input_length = 0
    
    def process_signal(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Process incoming text signal"""
        if not isinstance(signal.content, str):
            return None
            
        # Tokenize input
        tokens = self.tokenizer(
            signal.content,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Update statistics
        self.processed_inputs += 1
        self.avg_input_length = (
            (self.avg_input_length * (self.processed_inputs - 1) + len(signal.content))
            / self.processed_inputs
        )
        
        # Create processed signal for perception layer
        return NeuralSignal(
            source_layer=self.layer,
            target_layer=NeuralLayer.PERCEPTION,
            content={
                'text': signal.content,
                'tokens': tokens,
                'length': len(signal.content),
                'token_count': len(tokens['input_ids'][0])
            },
            priority=signal.priority
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'processed_inputs': self.processed_inputs,
            'avg_input_length': self.avg_input_length,
            'tokenizer_name': self.tokenizer.name_or_path,
            'max_length': self.max_length
        } 
"""
NIS Protocol LLM Provider Interface

This module defines the base interface for LLM providers that can be used
by agents in the NIS Protocol for cognitive processing.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

@dataclass
class LLMResponse:
    """Response from an LLM provider"""
    content: str
    metadata: Dict[str, Any]
    usage: Dict[str, int]
    model: str
    finish_reason: str
    
class LLMRole(Enum):
    """Roles for LLM interactions"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"

@dataclass
class LLMMessage:
    """Message for LLM interaction"""
    role: LLMRole
    content: str
    name: Optional[str] = None
    
class BaseLLMProvider(ABC):
    """Base interface for LLM providers.
    
    This abstract class defines the interface that all LLM providers
    must implement to be compatible with the NIS Protocol.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the LLM provider.
        
        Args:
            config: Configuration for the provider
        """
        self.config = config
        self.model = config.get("model", "default")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
    
    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM.
        
        Args:
            messages: List of messages for the conversation
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            stop: Optional list of stop sequences
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse containing the generated text and metadata
        """
        pass
    
    @abstractmethod
    async def embed(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text.
        
        Args:
            text: Text or list of texts to embed
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of embeddings or list of lists for batch embedding
        """
        pass
    
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in a text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        pass
    
    def prepare_messages(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: Optional[List[LLMMessage]] = None
    ) -> List[LLMMessage]:
        """Prepare messages for LLM interaction.
        
        Args:
            system_prompt: System prompt defining behavior
            user_message: Current user message
            conversation_history: Optional previous messages
            
        Returns:
            List of messages ready for LLM
        """
        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content=system_prompt)
        ]
        
        if conversation_history:
            messages.extend(conversation_history)
            
        messages.append(LLMMessage(role=LLMRole.USER, content=user_message))
        
        return messages 
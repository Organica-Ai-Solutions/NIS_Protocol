"""
NIS Protocol Deepseek LLM Provider

This module implements the Deepseek API integration for the NIS Protocol.
"""

import aiohttp
import json
from typing import Dict, Any, List, Optional, Union
import tiktoken
import logging

from ..base_llm_provider import BaseLLMProvider, LLMResponse, LLMMessage, LLMRole

class DeepseekProvider(BaseLLMProvider):
    """Deepseek API provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Deepseek provider.
        
        Args:
            config: Configuration including API key and model settings
        """
        super().__init__(config)
        
        self.api_key = config["api_key"]
        self.api_base = config.get("api_base", "https://api.deepseek.com/v1")
        self.encoding = tiktoken.encoding_for_model(self.model)
        self.logger = logging.getLogger("deepseek_provider")
        
        # Initialize session
        self.session = None
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
    
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response using Deepseek API.
        
        Args:
            messages: List of conversation messages
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            stop: Optional stop sequences
            **kwargs: Additional API parameters
            
        Returns:
            LLMResponse with generated content
        """
        await self._ensure_session()
        
        # Prepare request
        request_messages = [
            {
                "role": msg.role.value,
                "content": msg.content,
                **({"name": msg.name} if msg.name else {})
            }
            for msg in messages
        ]
        
        request_data = {
            "model": self.model,
            "messages": request_messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            **({"stop": stop} if stop else {}),
            **kwargs
        }
        
        try:
            async with self.session.post(
                f"{self.api_base}/chat/completions",
                json=request_data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Deepseek API error: {error_text}")
                
                result = await response.json()
                
                return LLMResponse(
                    content=result["choices"][0]["message"]["content"],
                    metadata={
                        "id": result["id"],
                        "created": result["created"],
                        "model": result["model"]
                    },
                    usage=result["usage"],
                    model=result["model"],
                    finish_reason=result["choices"][0]["finish_reason"]
                )
                
        except Exception as e:
            self.logger.error(f"Error calling Deepseek API: {str(e)}")
            raise
    
    async def embed(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using Deepseek API.
        
        Args:
            text: Text or texts to embed
            **kwargs: Additional API parameters
            
        Returns:
            List of embeddings
        """
        await self._ensure_session()
        
        texts = [text] if isinstance(text, str) else text
        
        try:
            async with self.session.post(
                f"{self.api_base}/embeddings",
                json={
                    "model": self.model,
                    "input": texts,
                    **kwargs
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Deepseek API error: {error_text}")
                
                result = await response.json()
                embeddings = [data["embedding"] for data in result["data"]]
                
                return embeddings[0] if isinstance(text, str) else embeddings
                
        except Exception as e:
            self.logger.error(f"Error getting embeddings: {str(e)}")
            raise
    
    def get_token_count(self, text: str) -> int:
        """Get token count using tiktoken.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None 
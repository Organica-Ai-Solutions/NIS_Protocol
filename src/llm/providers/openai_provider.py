"""
NIS Protocol OpenAI LLM Provider

This module implements the OpenAI API integration for the NIS Protocol.
Enhanced to support environment variables for configuration.
"""

import aiohttp
import json
import os
from typing import Dict, Any, List, Optional, Union
import tiktoken
import logging
import asyncio

from ..base_llm_provider import BaseLLMProvider, LLMResponse, LLMMessage, LLMRole

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the OpenAI provider.
        
        Args:
            config: Configuration including API key and model settings
        """
        super().__init__(config)
        
        # Support both direct config and environment variables
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.api_base = config.get("api_base") or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.organization = config.get("organization") or os.getenv("OPENAI_ORGANIZATION")
        
        if not self.api_key or self.api_key in ["YOUR_OPENAI_API_KEY", "your_openai_api_key_here"]:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or provide in config.")
        
        # Initialize tokenizer based on model
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Fallback for custom or newer models
            self.encoding = tiktoken.get_encoding("cl100k_base")
            
        self.logger = logging.getLogger("openai_provider")
        
        # Initialize session
        self.session = None
        
    def __del__(self):
        """Cleanup aiohttp session."""
        if self.session and not self.session.closed:
            try:
                asyncio.get_event_loop().create_task(self.session.close())
            except:
                pass
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self.session is None:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            if self.organization:
                headers["OpenAI-Organization"] = self.organization
                
            self.session = aiohttp.ClientSession(headers=headers)
    
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response using OpenAI API.
        
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
                    raise Exception(f"OpenAI API error: {error_text}")
                
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
            self.logger.error(f"Error calling OpenAI API: {str(e)}")
            raise
    
    async def embed(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using OpenAI API.
        
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
                    "model": kwargs.get("model", "text-embedding-3-small"),
                    "input": texts,
                    **{k: v for k, v in kwargs.items() if k != "model"}
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error: {error_text}")
                
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
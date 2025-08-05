"""
NIS Protocol Anthropic Claude LLM Provider

This module implements the Anthropic Claude API integration for the NIS Protocol.
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

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Anthropic provider.
        
        Args:
            config: Configuration including API key and model settings
        """
        super().__init__(config)
        
        # Support both direct config and environment variables
        self.api_key = config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        self.api_base = config.get("api_base") or os.getenv("ANTHROPIC_API_BASE", "https://api.anthropic.com/v1")
        self.anthropic_version = config.get("version") or os.getenv("ANTHROPIC_VERSION", "2023-06-01")
        
        if not self.api_key or self.api_key in ["YOUR_ANTHROPIC_API_KEY", "your_anthropic_api_key_here"]:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or provide in config.")
        
        # Use tiktoken for approximate token counting (Claude uses similar tokenization)
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            # Fallback if tiktoken not available
            self.encoding = None
            
        self.logger = logging.getLogger("anthropic_provider")
        
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
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "anthropic-version": self.anthropic_version
                }
            )
    
    def _convert_messages_to_anthropic(self, messages: List[LLMMessage]) -> Dict[str, Any]:
        """Convert NIS messages to Anthropic format."""
        system_content = ""
        conversation_messages = []
        
        for msg in messages:
            if msg.role == LLMRole.SYSTEM:
                system_content += msg.content + "\n"
            else:
                conversation_messages.append({
                    "role": "user" if msg.role == LLMRole.USER else "assistant",
                    "content": msg.content
                })
        
        result = {"messages": conversation_messages}
        if system_content.strip():
            result["system"] = system_content.strip()
            
        return result
    
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response using Anthropic Claude API.
        
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
        
        # Convert messages to Anthropic format
        anthropic_messages = self._convert_messages_to_anthropic(messages)
        
        request_data = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            **anthropic_messages,
            **({"stop_sequences": stop} if stop else {}),
            **kwargs
        }
        
        try:
            async with self.session.post(
                f"{self.api_base}/messages",
                json=request_data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Anthropic API error: {error_text}")
                
                result = await response.json()
                
                # Extract content from Claude's response format
                content = ""
                if "content" in result and result["content"]:
                    if isinstance(result["content"], list):
                        content = result["content"][0].get("text", "")
                    else:
                        content = str(result["content"])
                
                return LLMResponse(
                    content=content,
                    metadata={
                        "id": result.get("id", ""),
                        "model": result.get("model", self.model),
                        "type": result.get("type", "message")
                    },
                    usage=result.get("usage", {}),
                    model=result.get("model", self.model),
                    finish_reason=result.get("stop_reason", "end_turn")
                )
                
        except Exception as e:
            self.logger.error(f"Error calling Anthropic API: {str(e)}")
            raise
    
    async def embed(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using Anthropic API.
        
        Note: Anthropic doesn't provide embedding endpoints.
        This method raises NotImplementedError.
        
        Args:
            text: Text or texts to embed
            **kwargs: Additional API parameters
            
        Returns:
            List of embeddings
            
        Raises:
            NotImplementedError: Anthropic doesn't provide embeddings
        """
        raise NotImplementedError(
            "Anthropic Claude doesn't provide embedding endpoints. "
            "Use OpenAI or another provider for embeddings."
        )
    
    def get_token_count(self, text: str) -> int:
        """Get approximate token count using tiktoken.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Approximate number of tokens
        """
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Rough approximation: 1 token per 4 characters
            return len(text) // 4
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None 
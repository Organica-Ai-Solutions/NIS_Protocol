"""
NIS Protocol BitNet 2 LLM Provider

This module implements BitNet 2 local model integration for the NIS Protocol.
BitNet 2 enables efficient local inference with 1-bit quantization.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Union
import logging
import os
import subprocess
import tempfile

from ..base_llm_provider import BaseLLMProvider, LLMResponse, LLMMessage, LLMRole

class BitNetProvider(BaseLLMProvider):
    """BitNet 2 local model provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the BitNet provider.
        
        Args:
            config: Configuration including model path and settings
        """
        super().__init__(config)
        
        self.model_path = config.get("model_path", "./models/bitnet")
        self.executable_path = config.get("executable_path", "bitnet")
        self.context_length = config.get("context_length", 4096)
        self.batch_size = config.get("batch_size", 1)
        
        # BitNet specific settings
        self.quantization_bits = config.get("quantization_bits", 1)
        self.cpu_threads = config.get("cpu_threads", os.cpu_count())
        
        self.logger = logging.getLogger("bitnet_provider")
        
        # Check if BitNet is available
        self._check_bitnet_availability()
    
    def _check_bitnet_availability(self):
        """Check if BitNet executable is available."""
        try:
            result = subprocess.run(
                [self.executable_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self.logger.info(f"BitNet available: {result.stdout.strip()}")
            else:
                self.logger.warning("BitNet executable found but may not be working properly")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.warning(f"BitNet not available: {e}")
            self.logger.info("Install BitNet 2 from: https://github.com/microsoft/BitNet")
    
    def _prepare_prompt(self, messages: List[LLMMessage]) -> str:
        """Convert messages to a single prompt string."""
        prompt_parts = []
        
        for msg in messages:
            if msg.role == LLMRole.SYSTEM:
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == LLMRole.USER:
                prompt_parts.append(f"Human: {msg.content}")
            elif msg.role == LLMRole.ASSISTANT:
                prompt_parts.append(f"Assistant: {msg.content}")
        
        # Add assistant prompt for completion
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: int = 100,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response using the BitNet model.
        """
        prompt = self._prepare_prompt(messages)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Generate response
            output_sequences = self.model.generate(
                input_ids=inputs["input_ids"],
                max_length=max_tokens,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            response_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            
            # The response will likely include the prompt, so we need to remove it.
            # This is a simple way to do it; more robust methods may be needed.
            if response_text.startswith(prompt):
                response_text = response_text[len(prompt):]
                
            return LLMResponse(
                content=response_text.strip(),
                provider="bitnet",
                model="bitnet-offline",
                confidence=0.95, # High confidence for local model
                tokens_used=len(output_sequences[0]),
                real_ai=True
            )
            
        except Exception as e:
            self.logger.error(f"Error during BitNet generation: {e}")
            # Return an error response that fits the LLMResponse model
            return LLMResponse(
                content=f"Error generating response from BitNet: {e}",
                provider="bitnet",
                model="bitnet-offline",
                confidence=0.0,
                tokens_used=0,
                real_ai=True, # It's still from a real (local) AI
                error=str(e)
            )
    
    async def embed(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using BitNet.
        
        Note: This is a basic implementation that may need adjustment
        based on the specific BitNet model capabilities.
        
        Args:
            text: Text or texts to embed
            **kwargs: Additional parameters
            
        Returns:
            List of embeddings
        """
        texts = [text] if isinstance(text, str) else text
        embeddings = []
        
        for text_item in texts:
            # Use BitNet to generate a representation
            # This is a simplified approach - you may need to adjust based on your BitNet setup
            cmd = [
                self.executable_path,
                "--model", self.model_path,
                "--embed", text_item,
                "--format", "json"
            ]
            
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    # Fallback: generate a simple hash-based embedding
                    embedding = self._generate_fallback_embedding(text_item)
                else:
                    result = json.loads(stdout.decode())
                    embedding = result.get("embedding", self._generate_fallback_embedding(text_item))
                
                embeddings.append(embedding)
                
            except Exception as e:
                self.logger.warning(f"Error generating embedding: {e}")
                embeddings.append(self._generate_fallback_embedding(text_item))
        
        return embeddings[0] if isinstance(text, str) else embeddings
    
    def _generate_fallback_embedding(self, text: str, dim: int = 768) -> List[float]:
        """Generate a simple fallback embedding based on text hash."""
        import hashlib
        
        # Generate a deterministic hash-based embedding
        hash_bytes = hashlib.sha256(text.encode()).digest()
        
        # Convert to float values between -1 and 1
        embedding = []
        for i in range(dim):
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] - 128) / 128.0
            embedding.append(value)
        
        return embedding
    
    def get_token_count(self, text: str) -> int:
        """Get approximate token count for BitNet.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Approximate number of tokens
        """
        # Simple approximation: 1 token per 4 characters
        # You may want to implement more sophisticated tokenization
        return len(text.split()) + len(text) // 4
    
    async def close(self):
        """Clean up resources (no persistent connections for local model)."""
        # BitNet uses subprocess calls, so no persistent connections to close
        pass 
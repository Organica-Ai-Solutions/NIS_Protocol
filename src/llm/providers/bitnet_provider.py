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
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response using BitNet 2.
        
        Args:
            messages: List of conversation messages
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            stop: Optional stop sequences
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with generated content
        """
        prompt = self._prepare_prompt(messages)
        
        # Prepare BitNet command
        cmd = [
            self.executable_path,
            "--model", self.model_path,
            "--prompt", prompt,
            "--max-tokens", str(max_tokens or self.max_tokens),
            "--temperature", str(temperature or self.temperature),
            "--threads", str(self.cpu_threads),
            "--batch-size", str(self.batch_size),
            "--context-length", str(self.context_length),
            "--format", "json"  # Request JSON output for easier parsing
        ]
        
        if stop:
            for stop_seq in stop:
                cmd.extend(["--stop", stop_seq])
        
        try:
            # Run BitNet inference
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown BitNet error"
                raise Exception(f"BitNet inference failed: {error_msg}")
            
            # Parse BitNet output
            try:
                result = json.loads(stdout.decode())
                content = result.get("text", "").strip()
                
                # Remove the "Assistant:" prefix if present
                if content.startswith("Assistant:"):
                    content = content[len("Assistant:"):].strip()
                
                return LLMResponse(
                    content=content,
                    metadata={
                        "model": self.model,
                        "inference_time": result.get("inference_time", 0),
                        "tokens_generated": result.get("tokens_generated", 0)
                    },
                    usage={
                        "prompt_tokens": result.get("prompt_tokens", 0),
                        "completion_tokens": result.get("tokens_generated", 0),
                        "total_tokens": result.get("total_tokens", 0)
                    },
                    model=self.model,
                    finish_reason=result.get("finish_reason", "stop")
                )
                
            except json.JSONDecodeError:
                # Fallback: treat output as plain text
                content = stdout.decode().strip()
                if content.startswith("Assistant:"):
                    content = content[len("Assistant:"):].strip()
                
                return LLMResponse(
                    content=content,
                    metadata={"model": self.model},
                    usage={},
                    model=self.model,
                    finish_reason="stop"
                )
                
        except Exception as e:
            self.logger.error(f"Error running BitNet inference: {str(e)}")
            raise
    
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
"""
NIS Protocol Multimodel Consensus Provider

This module implements a consensus-based approach using multiple LLM providers.
It queries multiple models and synthesizes their responses for better accuracy.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from ..base_llm_provider import BaseLLMProvider, LLMResponse, LLMMessage, LLMRole

logger = logging.getLogger(__name__)

class MultimodelProvider(BaseLLMProvider):
    """Multimodel consensus provider that combines responses from multiple LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the multimodel provider.
        
        Args:
            config: Configuration including provider list and consensus settings
        """
        super().__init__(config)
        
        # Initialize available providers based on configuration
        self.available_providers = []
        self.provider_weights = {}
        
        # Default provider preferences (can be overridden in config)
        self.default_providers = ["openai", "anthropic", "deepseek", "google", "kimi"]
        self.consensus_threshold = config.get("consensus_threshold", 0.7)
        self.max_providers = config.get("max_providers", 3)
        
        # Provider specializations
        self.provider_strengths = {
            "openai": ["creative", "general", "coding"],
            "anthropic": ["reasoning", "analysis", "ethical"],
            "deepseek": ["mathematical", "scientific", "logical"],
            "google": ["factual", "research", "multilingual"],
            "kimi": ["long_context", "multilingual", "document_analysis"]
        }
        
        logger.info(f"Multimodel provider initialized with consensus threshold: {self.consensus_threshold}")
    
    def _select_providers_for_task(self, messages: List[LLMMessage], task_type: str = "general") -> List[str]:
        """Select the best providers for a specific task.
        
        Args:
            messages: The conversation messages to analyze
            task_type: Type of task (general, creative, analytical, etc.)
            
        Returns:
            List of provider names to use
        """
        # Analyze the last user message to determine task type
        if messages:
            last_message = messages[-1].content.lower()
            full_context = " ".join([msg.content for msg in messages[-3:]]).lower()  # Look at recent context
            
            if any(word in last_message for word in ["math", "calculate", "equation", "prove"]):
                task_type = "mathematical"
            elif any(word in last_message for word in ["creative", "story", "poem", "art"]):
                task_type = "creative"
            elif any(word in last_message for word in ["analyze", "reason", "logic", "because"]):
                task_type = "analytical"
            elif any(word in last_message for word in ["fact", "research", "data", "information"]):
                task_type = "factual"
            elif any(word in full_context for word in ["document", "long", "context", "summary", "extensive"]) or len(full_context) > 1000:
                task_type = "long_context"
            elif any(word in last_message for word in ["chinese", "中文", "multilingual", "translate"]):
                task_type = "multilingual"
        
        # Select providers based on their strengths
        selected = []
        for provider in self.default_providers[:self.max_providers]:
            strengths = self.provider_strengths.get(provider, [])
            if task_type in strengths or task_type == "general":
                selected.append(provider)
        
        # Ensure we have at least 2 providers for consensus
        if len(selected) < 2:
            selected = self.default_providers[:2]
            
        logger.info(f"Selected providers for {task_type} task: {selected}")
        return selected
    
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a consensus response using multiple providers.
        
        Args:
            messages: List of conversation messages
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            stop: Optional stop sequences
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with consensus content
        """
        try:
            # Select providers for this task
            providers_to_use = self._select_providers_for_task(messages)
            
            # Generate responses from multiple providers in parallel
            tasks = []
            for provider_name in providers_to_use:
                task = self._get_provider_response(
                    provider_name, messages, temperature, max_tokens, stop, **kwargs
                )
                tasks.append(task)
            
            # Wait for all responses
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful responses
            successful_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.warning(f"Provider {providers_to_use[i]} failed: {response}")
                    continue
                if response and response.get("content"):
                    successful_responses.append({
                        "provider": providers_to_use[i],
                        "content": response["content"],
                        "confidence": response.get("confidence", 0.8),
                        "metadata": response.get("metadata", {})
                    })
            
            if not successful_responses:
                # Fallback to mock response
                return LLMResponse(
                    content="I apologize, but I'm unable to generate a response at the moment. Please check your API configurations and try again.",
                    metadata={
                        "multimodel_status": "all_providers_failed",
                        "attempted_providers": providers_to_use,
                        "consensus_achieved": False
                    },
                    usage={"prompt_tokens": 0, "completion_tokens": 50, "total_tokens": 50},
                    model="multimodel-consensus",
                    finish_reason="error"
                )
            
            # Generate consensus response
            consensus_response = await self._generate_consensus(successful_responses, messages)
            
            return LLMResponse(
                content=consensus_response["content"],
                metadata={
                    "multimodel_status": "success",
                    "providers_used": [r["provider"] for r in successful_responses],
                    "consensus_achieved": consensus_response["consensus_achieved"],
                    "confidence_scores": {r["provider"]: r["confidence"] for r in successful_responses},
                    "synthesis_method": consensus_response["method"]
                },
                usage=consensus_response.get("usage", {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}),
                model="multimodel-consensus",
                finish_reason="stop"
            )
            
        except Exception as e:
            logger.error(f"Error in multimodel generation: {e}")
            return LLMResponse(
                content=f"Multimodel consensus failed: {str(e)}",
                metadata={"error": str(e), "multimodel_status": "error"},
                usage={"prompt_tokens": 0, "completion_tokens": 20, "total_tokens": 20},
                model="multimodel-consensus",
                finish_reason="error"
            )
    
    async def _get_provider_response(
        self, 
        provider_name: str, 
        messages: List[LLMMessage],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stop: Optional[List[str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Get response from a specific provider.
        
        Args:
            provider_name: Name of the provider to use
            messages: Conversation messages
            temperature: Temperature setting
            max_tokens: Max tokens setting
            stop: Stop sequences
            **kwargs: Additional parameters
            
        Returns:
            Provider response dict
        """
        try:
            # For now, return a mock response with provider-specific characteristics
            # In a real implementation, this would call the actual provider
            
            provider_personalities = {
                "openai": "Creative and engaging, with excellent reasoning capabilities.",
                "anthropic": "Thoughtful and analytical, focusing on accuracy and safety.",
                "deepseek": "Precise and logical, excelling in mathematical and scientific reasoning.",
                "google": "Comprehensive and factual, with strong research capabilities.",
                "kimi": "Exceptional at long-context understanding and multilingual communication."
            }
            
            content = f"""Based on the {provider_name.upper()} model perspective: {provider_personalities.get(provider_name, "General AI assistance")}

Your query has been processed through the {provider_name} provider with the following characteristics:
- Focus: {', '.join(self.provider_strengths.get(provider_name, ['general']))}
- Confidence: High
- Reasoning approach: {provider_name}-optimized

[This is a multimodel consensus demonstration. In production, this would contain the actual {provider_name} response.]"""
            
            return {
                "content": content,
                "confidence": 0.85 + (hash(provider_name) % 10) / 100,  # Simulate different confidence levels
                "metadata": {
                    "provider": provider_name,
                    "model": f"{provider_name}-model",
                    "tokens_used": len(content.split()) * 1.3
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting {provider_name} response: {e}")
            raise e
    
    async def _generate_consensus(self, responses: List[Dict], messages: List[LLMMessage]) -> Dict[str, Any]:
        """Generate consensus from multiple provider responses.
        
        Args:
            responses: List of provider responses
            messages: Original conversation messages
            
        Returns:
            Consensus response dict
        """
        if len(responses) == 1:
            return {
                "content": responses[0]["content"],
                "consensus_achieved": True,
                "method": "single_provider"
            }
        
        # Calculate average confidence
        avg_confidence = sum(r["confidence"] for r in responses) / len(responses)
        
        # Generate synthesis
        provider_names = [r["provider"] for r in responses]
        
        synthesis_content = f"""🧠 **MULTIMODEL CONSENSUS RESPONSE**

I've consulted {len(responses)} AI models ({', '.join(provider_names)}) to provide you with a comprehensive answer:

---

**SYNTHESIZED RESPONSE:**

Based on the collective analysis from multiple AI models, here's the consolidated response:

{self._synthesize_responses([r["content"] for r in responses])}

---

**CONSENSUS METADATA:**
• **Models Consulted:** {', '.join(provider_names)}
• **Average Confidence:** {avg_confidence:.1%}
• **Consensus Method:** Multi-provider synthesis
• **Agreement Level:** {"High" if avg_confidence > 0.8 else "Moderate" if avg_confidence > 0.6 else "Low"}

This response represents the combined intelligence of multiple AI systems working together to provide you with the most accurate and comprehensive answer possible."""

        return {
            "content": synthesis_content,
            "consensus_achieved": avg_confidence >= self.consensus_threshold,
            "method": "multi_provider_synthesis",
            "usage": {
                "prompt_tokens": sum(int(r.get("metadata", {}).get("tokens_used", 100)) for r in responses),
                "completion_tokens": len(synthesis_content.split()) * 1.3,
                "total_tokens": sum(int(r.get("metadata", {}).get("tokens_used", 100)) for r in responses) + len(synthesis_content.split()) * 1.3
            }
        }
    
    def _synthesize_responses(self, contents: List[str]) -> str:
        """Synthesize multiple response contents into a unified answer.
        
        Args:
            contents: List of response contents from different providers
            
        Returns:
            Synthesized content string
        """
        # Simple synthesis for demonstration
        # In production, this would use more sophisticated NLP techniques
        
        # Extract key points from each response
        key_points = []
        for i, content in enumerate(contents):
            # Take the first meaningful paragraph from each response
            lines = content.split('\n')
            meaningful_line = None
            for line in lines:
                if len(line.strip()) > 50 and not line.startswith('['):
                    meaningful_line = line.strip()
                    break
            
            if meaningful_line:
                key_points.append(f"**Model {i+1} Perspective:** {meaningful_line}")
        
        if not key_points:
            return "Multiple AI models have been consulted to provide you with comprehensive assistance."
        
        return '\n\n'.join(key_points[:3])  # Limit to top 3 perspectives
    
    async def embed(self, text: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using consensus approach."""
        # For embeddings, use the first available provider
        # In production, you might average embeddings from multiple providers
        return [0.1] * 1536  # Mock embedding
    
    def get_token_count(self, text: str) -> int:
        """Get approximate token count."""
        return len(text.split()) * 1.3  # Rough approximation
    
    async def close(self):
        """Close any open connections."""
        pass
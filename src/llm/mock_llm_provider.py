"""
Mock LLM Provider for Testing and Fallback
==========================================

This provides a simple mock LLM provider that can be used when the real
LLM providers are not available or properly configured.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional

class MockLLMProvider:
    """Mock LLM provider for testing and fallback scenarios."""
    
    def __init__(self):
        self.provider_name = "mock_provider"
        
    async def generate_response(self, messages: List[Dict[str, str]], 
                              temperature: float = 0.7, 
                              agent_type: str = 'default', 
                              requested_provider: Optional[str] = None) -> Dict[str, Any]:
        """Generate a mock response that mimics real LLM output."""
        
        # Extract the user message
        user_message = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        # Generate appropriate mock responses based on message content
        if not user_message:
            response_content = "I'm a mock LLM provider. Please provide a message for me to respond to."
        elif "quantum" in user_message.lower():
            response_content = f"Based on your question about quantum concepts, I can explain that quantum computing involves superposition, entanglement, and quantum gates. The question '{user_message}' touches on fundamental quantum mechanics principles that govern quantum information processing."
        elif "machine learning" in user_message.lower() or "ai" in user_message.lower():
            response_content = f"Regarding your machine learning question: '{user_message}' - Machine learning involves training algorithms on data to recognize patterns and make predictions. This includes supervised learning, unsupervised learning, and reinforcement learning approaches."
        elif "neural network" in user_message.lower():
            response_content = f"Neural networks are computational models inspired by biological neural networks. For your question '{user_message}', neural networks use layers of interconnected nodes (neurons) that process information through weighted connections and activation functions."
        elif "explain" in user_message.lower():
            response_content = f"I'll explain the concept you asked about: '{user_message}'. This involves understanding the fundamental principles, applications, and implications of the topic you're interested in learning about."
        else:
            response_content = f"Thank you for your message: '{user_message}'. I'm currently running in mock mode. This response demonstrates that the chat endpoints are working, but you'll need to configure proper LLM providers (OpenAI, Anthropic, or Google) for full functionality."
        
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        return {
            "content": response_content,
            "confidence": 0.85,
            "provider": self.provider_name,
            "real_ai": False,  # Important: mark as mock response
            "model": "mock-model-v1",
            "usage": {
                "prompt_tokens": len(str(messages)),
                "completion_tokens": len(response_content),
                "total_tokens": len(str(messages)) + len(response_content)
            },
            "mock_response": True
        }
    
    async def get_available_models(self) -> List[str]:
        """Return list of available mock models."""
        return ["mock-model-v1", "mock-test-model"]
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Return provider information."""
        return {
            "name": self.provider_name,
            "type": "mock",
            "models": ["mock-model-v1", "mock-test-model"],
            "status": "active",
            "description": "Mock LLM provider for testing and fallback scenarios"
        }
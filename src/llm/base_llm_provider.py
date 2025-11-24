#!/usr/bin/env python3
"""
Base LLM Provider for NIS Protocol
Provides basic message and role classes for LLM operations

Copyright 2025 Organica AI Solutions

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass


class LLMRole(Enum):
    """Enumeration of LLM message roles"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class LLMMessage:
    """Represents a message in LLM conversation"""
    role: LLMRole
    content: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format"""
        result = {
            "role": self.role.value,
            "content": self.content
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMMessage':
        """Create message from dictionary"""
        return cls(
            role=LLMRole(data["role"]),
            content=data["content"],
            metadata=data.get("metadata")
        )


@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseLLMProvider:
    """
    Base class for LLM providers in NIS Protocol
    
    This provides a common interface for different LLM providers
    while maintaining compatibility with existing code.
    """
    
    def __init__(self, provider_name: str = "base"):
        self.provider_name = provider_name
    
    async def generate_response(self, messages: list[LLMMessage]) -> str:
        """Generate response from messages (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement generate_response")
    
    def format_messages(self, messages: list[LLMMessage]) -> list[Dict[str, Any]]:
        """Format messages for API calls"""
        return [msg.to_dict() for msg in messages]

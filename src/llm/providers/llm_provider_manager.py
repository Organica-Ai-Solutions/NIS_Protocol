"""
NIS Protocol — LLM Provider Manager Stub
Provides LLMProviderManager, TaskType, LLMProvider, ResponseConfidence
used by DRL multi-LLM coordination.
"""
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nis.llm.providers")


class TaskType(str, Enum):
    REASONING = "reasoning"
    CODE = "code"
    CHAT = "chat"
    SUMMARIZATION = "summarization"
    PLANNING = "planning"
    VISION = "vision"
    MATH = "math"
    GENERAL = "general"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    LOCAL = "local"
    NIS = "nis"
    MOCK = "mock"


class ResponseConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class LLMProviderManager:
    """
    Manages multiple LLM providers and routes tasks to the best available one.
    Stub implementation — extend with real provider integrations.
    """

    def __init__(self):
        self.logger = logging.getLogger("nis.llm.provider_manager")
        self._providers: Dict[LLMProvider, Any] = {}
        self._metrics: Dict[LLMProvider, Dict[str, float]] = {}
        self.initialized = False

    async def initialize(self) -> bool:
        self.initialized = True
        self.logger.info("LLMProviderManager initialized")
        return True

    def register_provider(self, provider: LLMProvider, client: Any) -> None:
        self._providers[provider] = client
        self._metrics[provider] = {"calls": 0, "errors": 0, "avg_latency_ms": 0.0}

    def get_available_providers(self) -> List[LLMProvider]:
        return list(self._providers.keys())

    def select_provider(
        self,
        task_type: TaskType = TaskType.GENERAL,
        prefer_fast: bool = False,
        prefer_cheap: bool = False,
    ) -> Optional[LLMProvider]:
        if not self._providers:
            return None
        # Simple priority: prefer LOCAL > NIS > GROQ > OPENAI > ANTHROPIC
        priority = [LLMProvider.LOCAL, LLMProvider.NIS, LLMProvider.GROQ,
                    LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.GOOGLE]
        for p in priority:
            if p in self._providers:
                return p
        return next(iter(self._providers))

    async def generate(
        self,
        messages: List[Dict[str, str]],
        task_type: TaskType = TaskType.GENERAL,
        provider: Optional[LLMProvider] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        t0 = time.time()
        selected = provider or self.select_provider(task_type)
        if selected and selected in self._providers:
            try:
                client = self._providers[selected]
                result = await client.generate(messages=messages, max_tokens=max_tokens)
                self._metrics[selected]["calls"] += 1
                return {
                    "content": result.get("content", ""),
                    "provider": selected,
                    "task_type": task_type,
                    "confidence": ResponseConfidence.MEDIUM,
                    "latency_ms": round((time.time() - t0) * 1000),
                }
            except Exception as e:
                self.logger.warning("Provider %s failed: %s", selected, e)
                if selected in self._metrics:
                    self._metrics[selected]["errors"] += 1

        # Mock fallback
        prompt = messages[-1].get("content", "") if messages else ""
        return {
            "content": f"[mock] Processed: {prompt[:100]}",
            "provider": LLMProvider.MOCK,
            "task_type": task_type,
            "confidence": ResponseConfidence.LOW,
            "latency_ms": round((time.time() - t0) * 1000),
        }

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "providers": {p.value: m for p, m in self._metrics.items()},
            "available": [p.value for p in self.get_available_providers()],
            "initialized": self.initialized,
        }

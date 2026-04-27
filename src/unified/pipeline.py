"""
NIS Protocol — Unified Pipeline Stub
Provides get_unified_pipeline and PipelineMode used by routes/unified.py.
"""
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nis.unified.pipeline")


class PipelineMode(str, Enum):
    STANDARD = "standard"
    FAST = "fast"
    DEEP = "deep"
    AUTONOMOUS = "autonomous"


class UnifiedPipeline:
    """Unified NIS processing pipeline — coordinates all agents."""

    def __init__(self):
        self.logger = logging.getLogger("nis.unified.pipeline")
        self.initialized = False
        self.mode = PipelineMode.STANDARD

    async def initialize(self) -> bool:
        self.initialized = True
        self.logger.info("UnifiedPipeline initialized")
        return True

    async def process(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        mode: PipelineMode = PipelineMode.STANDARD,
        image_data: Optional[str] = None,
    ) -> Dict[str, Any]:
        t0 = time.time()
        context = context or {}
        return {
            "success": True,
            "response": f"Processed: {message[:200]}",
            "mode": mode,
            "agents_used": ["reasoning", "memory"],
            "context": context,
            "latency_ms": round((time.time() - t0) * 1000),
            "timestamp": time.time(),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "initialized": self.initialized,
            "mode": self.mode,
            "available": True,
        }


_pipeline_instance: Optional[UnifiedPipeline] = None


def get_unified_pipeline() -> UnifiedPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = UnifiedPipeline()
    return _pipeline_instance

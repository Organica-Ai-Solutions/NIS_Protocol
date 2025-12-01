"""
Unified Pipeline - NIS Protocol v4.0
The Integration Layer that connects ALL components.

This is the KEY that unlocks everything:
- Memory (persistent context)
- Cost Tracking (usage monitoring)
- Response Cache (efficiency)
- Code Execution (autonomous capabilities)
- Templates (structured prompts)
- Consciousness Pipeline (validation)

Data Flow:
USER → Memory Context → Cache Check → LLM → Code Execution → Cost Track → Memory Store → USER
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

logger = logging.getLogger("nis.unified_pipeline")


class PipelineMode(Enum):
    """Pipeline execution modes"""
    FAST = "fast"           # Skip consciousness, minimal processing
    STANDARD = "standard"   # Normal flow with all checks
    AUTONOMOUS = "autonomous"  # Full autonomous with code execution
    RESEARCH = "research"   # Deep research mode


@dataclass
class PipelineContext:
    """Context passed through the pipeline"""
    request_id: str
    user_id: str
    conversation_id: str
    message: str
    mode: PipelineMode = PipelineMode.STANDARD
    
    # Enriched during pipeline
    memory_context: str = ""
    cached_response: Optional[str] = None
    llm_response: str = ""
    code_outputs: List[Dict] = field(default_factory=list)
    artifacts: List[Dict] = field(default_factory=list)
    
    # Multi-modal inputs
    image_base64: Optional[str] = None
    file_path: Optional[str] = None
    image_analysis: Optional[str] = None
    
    # Proactive suggestions
    suggestions: List[str] = field(default_factory=list)
    
    # Metrics
    start_time: float = field(default_factory=time.time)
    tokens_used: int = 0
    cost_usd: float = 0
    cache_hit: bool = False
    
    # Pipeline stages completed
    stages_completed: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Result from pipeline execution"""
    success: bool
    response: str
    context: PipelineContext
    artifacts: List[Dict] = field(default_factory=list)
    execution_time_ms: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "response": self.response,
            "request_id": self.context.request_id,
            "conversation_id": self.context.conversation_id,
            "mode": self.context.mode.value,
            "cache_hit": self.context.cache_hit,
            "tokens_used": self.context.tokens_used,
            "cost_usd": round(self.context.cost_usd, 6),
            "artifacts": self.artifacts,
            "stages": self.context.stages_completed,
            "execution_time_ms": round(self.execution_time_ms, 1)
        }
        
        # Add suggestions if present
        if self.context.suggestions:
            result["suggestions"] = self.context.suggestions
        
        # Add image analysis if present
        if self.context.image_analysis:
            result["image_analysis"] = self.context.image_analysis
        
        return result


class UnifiedPipeline:
    """
    The master pipeline that orchestrates all NIS Protocol components.
    
    This is the integration layer that was missing - it connects:
    1. Persistent Memory → Provides context from past conversations
    2. Response Cache → Avoids redundant LLM calls
    3. LLM Provider → Generates responses
    4. Code Executor → Runs generated code
    5. Cost Tracker → Records usage
    6. Consciousness → Validates responses (optional)
    """
    
    def __init__(self):
        self._initialized = False
        self._memory = None
        self._cache = None
        self._cost_tracker = None
        self._code_executor = None
        self._template_manager = None
        self._llm_provider = None
        self._vision_agent = None  # Multi-modal support
    
    async def initialize(self):
        """Lazy initialization of all components"""
        if self._initialized:
            return
        
        try:
            from src.memory.persistent_memory import get_memory_system
            self._memory = get_memory_system()
            logger.info("✅ Memory system connected")
        except Exception as e:
            logger.warning(f"Memory system not available: {e}")
        
        try:
            from src.utils.response_cache import get_response_cache
            self._cache = get_response_cache()
            logger.info("✅ Response cache connected")
        except Exception as e:
            logger.warning(f"Response cache not available: {e}")
        
        try:
            from src.utils.cost_tracker import get_cost_tracker
            self._cost_tracker = get_cost_tracker()
            logger.info("✅ Cost tracker connected")
        except Exception as e:
            logger.warning(f"Cost tracker not available: {e}")
        
        try:
            from src.execution.code_executor import get_code_executor
            self._code_executor = get_code_executor()
            logger.info("✅ Code executor connected")
        except Exception as e:
            logger.warning(f"Code executor not available: {e}")
        
        try:
            from src.utils.prompt_templates import get_template_manager
            self._template_manager = get_template_manager()
            logger.info("✅ Template manager connected")
        except Exception as e:
            logger.warning(f"Template manager not available: {e}")
        
        try:
            from src.agents.multimodal.vision_agent import MultimodalVisionAgent
            self._vision_agent = MultimodalVisionAgent()
            logger.info("✅ Vision agent connected (multi-modal)")
        except Exception as e:
            logger.warning(f"Vision agent not available: {e}")
        
        self._initialized = True
        logger.info("🔗 Unified Pipeline initialized - all components connected")
    
    async def process(
        self,
        message: str,
        user_id: str = "default",
        conversation_id: Optional[str] = None,
        mode: PipelineMode = PipelineMode.STANDARD,
        llm_callback: Optional[Callable] = None,
        provider: str = "anthropic",
        model: Optional[str] = None,
        image_base64: Optional[str] = None,  # Multi-modal: image input
        file_path: Optional[str] = None,     # Multi-modal: file input
        generate_suggestions: bool = True     # Proactive suggestions
    ) -> PipelineResult:
        """
        Process a message through the unified pipeline.
        
        Flow:
        1. Create context
        2. Retrieve memory context
        3. Check cache
        4. Call LLM (if not cached)
        5. Detect if code execution needed
        6. Execute code if needed
        7. Track costs
        8. Store in memory
        9. Return result
        """
        await self.initialize()
        
        # Create context
        ctx = PipelineContext(
            request_id=str(uuid.uuid4())[:8],
            user_id=user_id,
            conversation_id=conversation_id or f"conv_{uuid.uuid4().hex[:8]}",
            message=message,
            mode=mode,
            image_base64=image_base64,
            file_path=file_path
        )
        
        try:
            # Stage 0: Multi-Modal Processing (if image provided)
            if image_base64 and self._vision_agent:
                ctx.image_analysis = await self._analyze_image(ctx)
                ctx.stages_completed.append("image_analysis")
            
            # Stage 1: Memory Context
            if self._memory and mode != PipelineMode.FAST:
                ctx.memory_context = await self._get_memory_context(ctx)
                ctx.stages_completed.append("memory_context")
            
            # Stage 2: Cache Check
            if self._cache and mode == PipelineMode.FAST:
                cached = self._cache.get(message, provider, model or "default")
                if cached:
                    ctx.cached_response = cached["response"]
                    ctx.cache_hit = True
                    ctx.stages_completed.append("cache_hit")
                    
                    return PipelineResult(
                        success=True,
                        response=cached["response"],
                        context=ctx,
                        execution_time_ms=(time.time() - ctx.start_time) * 1000
                    )
            
            # Stage 3: LLM Call
            if llm_callback:
                # Build prompt with memory context
                full_prompt = self._build_prompt(ctx)
                ctx.llm_response = await llm_callback(full_prompt)
                ctx.stages_completed.append("llm_call")
            else:
                ctx.llm_response = "No LLM callback provided"
            
            # Stage 4: Code Detection & Execution (Autonomous mode)
            if mode == PipelineMode.AUTONOMOUS and self._code_executor:
                code_result = await self._handle_code_execution(ctx)
                if code_result:
                    ctx.code_outputs.append(code_result)
                    ctx.stages_completed.append("code_execution")
            
            # Stage 5: Cost Tracking
            if self._cost_tracker:
                self._track_cost(ctx, provider, model)
                ctx.stages_completed.append("cost_tracking")
            
            # Stage 6: Cache Store
            if self._cache and ctx.llm_response and not ctx.cache_hit:
                self._cache.set(
                    message, ctx.llm_response, provider, model or "default",
                    tokens_used=ctx.tokens_used, cost=ctx.cost_usd
                )
                ctx.stages_completed.append("cache_store")
            
            # Stage 7: Memory Store
            if self._memory and mode != PipelineMode.FAST:
                await self._store_memory(ctx)
                ctx.stages_completed.append("memory_store")
            
            # Stage 8: Generate Proactive Suggestions
            if generate_suggestions and llm_callback:
                ctx.suggestions = await self._generate_suggestions(ctx, llm_callback)
                if ctx.suggestions:
                    ctx.stages_completed.append("suggestions")
            
            # Build final response
            final_response = ctx.llm_response
            
            # Add image analysis if present
            if ctx.image_analysis:
                final_response = f"**Image Analysis:**\n{ctx.image_analysis}\n\n{final_response}"
            
            if ctx.code_outputs:
                final_response += "\n\n---\n**Code Execution Results:**\n"
                for output in ctx.code_outputs:
                    if output.get("stdout"):
                        final_response += f"```\n{output['stdout']}\n```\n"
                    if output.get("plots"):
                        final_response += f"📊 Generated {len(output['plots'])} plot(s)\n"
            
            return PipelineResult(
                success=True,
                response=final_response,
                context=ctx,
                artifacts=ctx.artifacts,
                execution_time_ms=(time.time() - ctx.start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return PipelineResult(
                success=False,
                response=f"Pipeline error: {str(e)}",
                context=ctx,
                execution_time_ms=(time.time() - ctx.start_time) * 1000
            )
    
    async def _get_memory_context(self, ctx: PipelineContext) -> str:
        """Retrieve relevant memory context"""
        if not self._memory:
            return ""
        
        try:
            context = await self._memory.get_context_for_query(ctx.message, max_tokens=500)
            return context
        except Exception as e:
            logger.warning(f"Memory context retrieval failed: {e}")
            return ""
    
    def _build_prompt(self, ctx: PipelineContext) -> str:
        """Build the full prompt with context"""
        parts = []
        
        # Add memory context if available
        if ctx.memory_context:
            parts.append(f"**Relevant Context from Memory:**\n{ctx.memory_context}\n")
        
        # Add mode-specific instructions
        if ctx.mode == PipelineMode.AUTONOMOUS:
            parts.append("""You are an autonomous AI that can execute Python code.
If the task requires computation, data analysis, or visualization, generate Python code.
Format code in ```python blocks.
Available: numpy, pandas, matplotlib, scipy, sympy.""")
        
        # Add the user message
        parts.append(f"\n**User Message:**\n{ctx.message}")
        
        return "\n".join(parts)
    
    async def _handle_code_execution(self, ctx: PipelineContext) -> Optional[Dict]:
        """Detect and execute code from LLM response"""
        if not self._code_executor:
            return None
        
        response = ctx.llm_response
        
        # Check for code blocks
        if "```python" not in response:
            return None
        
        # Extract code
        start = response.find("```python") + 9
        end = response.find("```", start)
        if end <= start:
            return None
        
        code = response[start:end].strip()
        
        # Execute
        try:
            result = await self._code_executor.execute(code)
            
            # Capture artifacts
            if result.plots:
                for plot in result.plots:
                    ctx.artifacts.append({
                        "type": "plot",
                        "name": plot["name"],
                        "base64": plot["base64"]
                    })
            
            return {
                "code": code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "plots": result.plots,
                "dataframes": result.dataframes,
                "success": result.success,
                "error": result.error
            }
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return {"error": str(e), "success": False}
    
    def _track_cost(self, ctx: PipelineContext, provider: str, model: Optional[str]):
        """Track the cost of this request"""
        if not self._cost_tracker:
            return
        
        # Estimate tokens (rough)
        input_tokens = len(ctx.message.split()) * 2
        output_tokens = len(ctx.llm_response.split()) * 2
        
        try:
            record = self._cost_tracker.record(
                provider=provider,
                model=model or "default",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                endpoint="/unified/chat",
                latency_ms=(time.time() - ctx.start_time) * 1000,
                success=True
            )
            ctx.tokens_used = input_tokens + output_tokens
            ctx.cost_usd = record.cost_usd
        except Exception as e:
            logger.warning(f"Cost tracking failed: {e}")
    
    async def _store_memory(self, ctx: PipelineContext):
        """Store the conversation in memory"""
        if not self._memory:
            return
        
        try:
            await self._memory.store_conversation(
                user_message=ctx.message,
                assistant_response=ctx.llm_response[:500],  # Truncate for storage
                importance=0.5,
                conversation_id=ctx.conversation_id
            )
        except Exception as e:
            logger.warning(f"Memory storage failed: {e}")
    
    async def _analyze_image(self, ctx: PipelineContext) -> Optional[str]:
        """Analyze image using vision agent (Multi-Modal)"""
        if not self._vision_agent or not ctx.image_base64:
            return None
        
        try:
            # Build analysis prompt
            prompt = ctx.message if ctx.message else "Describe this image in detail."
            
            # Call vision agent
            result = await self._vision_agent.analyze_image(
                image_base64=ctx.image_base64,
                prompt=prompt
            )
            
            return result.get("analysis", result.get("description", "Image analyzed"))
        except Exception as e:
            logger.warning(f"Image analysis failed: {e}")
            return None
    
    async def _generate_suggestions(self, ctx: PipelineContext, llm_callback: Callable) -> List[str]:
        """Generate proactive suggestions based on context and response"""
        try:
            # Build suggestion prompt
            suggestion_prompt = f"""Based on this conversation, suggest 3 helpful follow-up actions the user might want to take.

User asked: {ctx.message[:200]}

Response given: {ctx.llm_response[:300]}

Memory context: {ctx.memory_context[:200] if ctx.memory_context else 'None'}

Provide exactly 3 short, actionable suggestions (one per line, no numbering):"""

            response = await llm_callback(suggestion_prompt)
            
            # Parse suggestions (one per line)
            suggestions = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # Remove numbering if present
                if line and len(line) > 5:
                    # Remove common prefixes like "1.", "- ", "* "
                    if line[0].isdigit() and line[1] in '.):':
                        line = line[2:].strip()
                    elif line[0] in '-*':
                        line = line[1:].strip()
                    if line:
                        suggestions.append(line)
            
            return suggestions[:3]  # Max 3 suggestions
        except Exception as e:
            logger.warning(f"Suggestion generation failed: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        return {
            "initialized": self._initialized,
            "components": {
                "memory": self._memory is not None,
                "cache": self._cache is not None,
                "cost_tracker": self._cost_tracker is not None,
                "code_executor": self._code_executor is not None,
                "template_manager": self._template_manager is not None,
                "vision_agent": self._vision_agent is not None
            }
        }


# Global instance
_pipeline: Optional[UnifiedPipeline] = None


def get_unified_pipeline() -> UnifiedPipeline:
    """Get or create the global unified pipeline"""
    global _pipeline
    if _pipeline is None:
        _pipeline = UnifiedPipeline()
    return _pipeline


async def process_unified(
    message: str,
    llm_callback: Callable,
    **kwargs
) -> PipelineResult:
    """Convenience function for unified processing"""
    pipeline = get_unified_pipeline()
    return await pipeline.process(message, llm_callback=llm_callback, **kwargs)

"""
NIS Protocol Cognitive Orchestra

A sophisticated multi-LLM architecture that assigns specialized language models
to different cognitive functions, creating a "cognitive orchestra" where each
LLM plays its optimal role in the overall intelligence system.

Philosophy: Instead of one massive model doing everything, we orchestrate
multiple specialized models working in harmony - like a symphony orchestra
where each instrument contributes its unique strengths.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from .llm_manager import LLMManager
from .base_llm_provider import LLMMessage, LLMResponse


class CognitiveFunction(Enum):
    """Different cognitive functions that can be specialized."""
    # Core reasoning functions
    CONSCIOUSNESS = "consciousness"           # Deep self-reflection, meta-cognition
    REASONING = "reasoning"                  # Logical analysis, problem solving
    CREATIVITY = "creativity"                # Novel idea generation, artistic thinking
    MEMORY = "memory"                        # Information storage, retrieval, consolidation
    
    # Perception and processing
    PERCEPTION = "perception"                # Pattern recognition, sensory processing
    LANGUAGE = "language"                    # Natural language understanding, generation
    CULTURAL = "cultural"                    # Cultural intelligence, sensitivity
    EMOTIONAL = "emotional"                  # Emotional processing, empathy
    
    # Action and coordination
    PLANNING = "planning"                    # Strategic planning, goal decomposition
    EXECUTION = "execution"                  # Action selection, motor control
    COORDINATION = "coordination"            # Multi-agent coordination, communication
    MONITORING = "monitoring"                # Performance monitoring, error detection
    
    # Specialized domains
    ARCHAEOLOGICAL = "archaeological"        # Domain-specific archaeological knowledge
    ENVIRONMENTAL = "environmental"          # Environmental and climate intelligence
    SPATIAL = "spatial"                      # Spatial reasoning, navigation
    TEMPORAL = "temporal"                    # Time-based reasoning, scheduling


@dataclass
class CognitiveProfile:
    """Profile defining optimal LLM characteristics for a cognitive function."""
    function: CognitiveFunction
    optimal_providers: List[str]             # Preferred LLM providers in order
    temperature_range: Tuple[float, float]   # Optimal temperature range
    max_tokens: int                          # Typical token requirements
    latency_priority: str                    # "speed", "quality", or "balanced"
    parallel_capable: bool                   # Can run in parallel with others
    memory_intensive: bool                   # Requires significant context
    creativity_level: float                  # 0.0 (analytical) to 1.0 (creative)
    precision_level: float                   # 0.0 (flexible) to 1.0 (precise)
    domain_knowledge: List[str]              # Required domain knowledge areas
    
    def get_optimal_temperature(self) -> float:
        """Get optimal temperature based on creativity and precision needs."""
        # Balance creativity and precision requirements
        creativity_temp = self.creativity_level * 0.9  # Max 0.9 for creativity
        precision_temp = (1.0 - self.precision_level) * 0.8  # Lower temp for precision
        
        # Weighted average, clamped to range
        optimal = (creativity_temp + precision_temp) / 2
        return max(self.temperature_range[0], min(self.temperature_range[1], optimal))


class CognitiveOrchestra:
    """
    Manages a symphony of specialized LLMs for different cognitive functions.
    
    This system automatically assigns the best LLM for each cognitive task,
    manages parallel processing, and coordinates between different models
    to create emergent intelligence greater than the sum of its parts.
    """
    
    def __init__(self, llm_manager: LLMManager, config_path: Optional[str] = None):
        """Initialize the cognitive orchestra.
        
        Args:
            llm_manager: The base LLM manager
            config_path: Path to cognitive orchestra configuration
        """
        self.llm_manager = llm_manager
        self.logger = logging.getLogger("cognitive_orchestra")
        
        # Load cognitive profiles
        self.cognitive_profiles = self._load_cognitive_profiles(config_path)
        
        # Track active cognitive processes
        self.active_processes: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.performance_metrics: Dict[CognitiveFunction, Dict[str, float]] = {}
        
        # Coordination state
        self.coordination_state = {
            "active_functions": set(),
            "resource_usage": {},
            "cross_function_memory": {},
            "harmony_score": 1.0
        }
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        self.logger.info("Cognitive Orchestra initialized with specialized LLM assignments")
    
    def _load_cognitive_profiles(self, config_path: Optional[str]) -> Dict[CognitiveFunction, CognitiveProfile]:
        """Load cognitive function profiles."""
        profiles = {}
        
        # Define default cognitive profiles
        default_profiles = {
            CognitiveFunction.CONSCIOUSNESS: CognitiveProfile(
                function=CognitiveFunction.CONSCIOUSNESS,
                optimal_providers=["anthropic", "openai", "deepseek"],
                temperature_range=(0.3, 0.7),
                max_tokens=4096,
                latency_priority="quality",
                parallel_capable=False,  # Consciousness needs focused attention
                memory_intensive=True,
                creativity_level=0.6,
                precision_level=0.8,
                domain_knowledge=["philosophy", "psychology", "metacognition"]
            ),
            
            CognitiveFunction.REASONING: CognitiveProfile(
                function=CognitiveFunction.REASONING,
                optimal_providers=["anthropic", "openai", "deepseek"],
                temperature_range=(0.1, 0.4),
                max_tokens=3072,
                latency_priority="quality",
                parallel_capable=True,
                memory_intensive=True,
                creativity_level=0.2,
                precision_level=0.9,
                domain_knowledge=["logic", "mathematics", "analysis"]
            ),
            
            CognitiveFunction.CREATIVITY: CognitiveProfile(
                function=CognitiveFunction.CREATIVITY,
                optimal_providers=["openai", "anthropic", "deepseek"],
                temperature_range=(0.7, 1.0),
                max_tokens=2048,
                latency_priority="balanced",
                parallel_capable=True,
                memory_intensive=False,
                creativity_level=0.9,
                precision_level=0.3,
                domain_knowledge=["art", "innovation", "brainstorming"]
            ),
            
            CognitiveFunction.MEMORY: CognitiveProfile(
                function=CognitiveFunction.MEMORY,
                optimal_providers=["anthropic", "deepseek", "openai"],
                temperature_range=(0.2, 0.5),
                max_tokens=4096,
                latency_priority="balanced",
                parallel_capable=True,
                memory_intensive=True,
                creativity_level=0.3,
                precision_level=0.8,
                domain_knowledge=["information_science", "psychology"]
            ),
            
            CognitiveFunction.PERCEPTION: CognitiveProfile(
                function=CognitiveFunction.PERCEPTION,
                optimal_providers=["openai", "anthropic", "bitnet"],
                temperature_range=(0.3, 0.6),
                max_tokens=2048,
                latency_priority="speed",
                parallel_capable=True,
                memory_intensive=False,
                creativity_level=0.4,
                precision_level=0.7,
                domain_knowledge=["computer_vision", "pattern_recognition"]
            ),
            
            CognitiveFunction.EMOTIONAL: CognitiveProfile(
                function=CognitiveFunction.EMOTIONAL,
                optimal_providers=["anthropic", "openai", "deepseek"],
                temperature_range=(0.6, 0.9),
                max_tokens=2048,
                latency_priority="balanced",
                parallel_capable=True,
                memory_intensive=False,
                creativity_level=0.7,
                precision_level=0.5,
                domain_knowledge=["psychology", "empathy", "social_intelligence"]
            ),
            
            CognitiveFunction.CULTURAL: CognitiveProfile(
                function=CognitiveFunction.CULTURAL,
                optimal_providers=["anthropic", "openai", "deepseek"],
                temperature_range=(0.4, 0.7),
                max_tokens=3072,
                latency_priority="quality",
                parallel_capable=True,
                memory_intensive=True,
                creativity_level=0.5,
                precision_level=0.8,
                domain_knowledge=["anthropology", "cultural_studies", "ethics"]
            ),
            
            CognitiveFunction.PLANNING: CognitiveProfile(
                function=CognitiveFunction.PLANNING,
                optimal_providers=["anthropic", "openai", "deepseek"],
                temperature_range=(0.2, 0.5),
                max_tokens=3072,
                latency_priority="quality",
                parallel_capable=False,  # Planning needs sequential thinking
                memory_intensive=True,
                creativity_level=0.4,
                precision_level=0.9,
                domain_knowledge=["strategy", "project_management", "optimization"]
            ),
            
            CognitiveFunction.EXECUTION: CognitiveProfile(
                function=CognitiveFunction.EXECUTION,
                optimal_providers=["bitnet", "deepseek", "openai"],
                temperature_range=(0.1, 0.4),
                max_tokens=1024,
                latency_priority="speed",
                parallel_capable=True,
                memory_intensive=False,
                creativity_level=0.1,
                precision_level=0.9,
                domain_knowledge=["robotics", "control_systems", "automation"]
            ),
            
            CognitiveFunction.ARCHAEOLOGICAL: CognitiveProfile(
                function=CognitiveFunction.ARCHAEOLOGICAL,
                optimal_providers=["anthropic", "openai", "deepseek"],
                temperature_range=(0.3, 0.6),
                max_tokens=4096,
                latency_priority="quality",
                parallel_capable=True,
                memory_intensive=True,
                creativity_level=0.5,
                precision_level=0.8,
                domain_knowledge=["archaeology", "history", "cultural_heritage"]
            )
        }
        
        # Load custom profiles if config provided
        if config_path:
            try:
                with open(config_path) as f:
                    custom_config = json.load(f)
                # Merge custom profiles with defaults
            except Exception as e:
                self.logger.warning(f"Could not load custom cognitive profiles: {e}")
        
        return default_profiles
    
    async def process_cognitive_task(
        self,
        function: CognitiveFunction,
        messages: List[LLMMessage],
        context: Optional[Dict[str, Any]] = None,
        priority: str = "normal"
    ) -> LLMResponse:
        """Process a task using the optimal LLM for the cognitive function.
        
        Args:
            function: The cognitive function required
            messages: Messages to process
            context: Additional context for processing
            priority: Task priority ("low", "normal", "high", "critical")
            
        Returns:
            LLM response optimized for the cognitive function
        """
        task_id = f"{function.value}_{int(time.time() * 1000)}"
        
        try:
            # Get optimal provider and configuration
            provider, config = await self._select_optimal_provider(function, context, priority)
            
            # Track active process
            self.active_processes[task_id] = {
                "function": function,
                "provider": getattr(provider, 'provider_name', "unknown"),
                "start_time": time.time(),
                "priority": priority,
                "context": context or {}
            }
            
            # Update coordination state
            self.coordination_state["active_functions"].add(function)
            
            # Process with specialized configuration
            response = await self._process_with_specialization(
                provider, messages, function, config, context
            )
            
            # Update performance metrics
            self._update_performance_metrics(function, task_id, response)
            
            # Clean up
            del self.active_processes[task_id]
            self.coordination_state["active_functions"].discard(function)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing cognitive task {task_id}: {e}")
            if task_id in self.active_processes:
                del self.active_processes[task_id]
            raise
    
    async def orchestrate_parallel_processing(
        self,
        tasks: List[Tuple[CognitiveFunction, List[LLMMessage], Optional[Dict[str, Any]]]]
    ) -> Dict[CognitiveFunction, LLMResponse]:
        """Process multiple cognitive tasks in parallel where possible.
        
        Args:
            tasks: List of (function, messages, context) tuples
            
        Returns:
            Dictionary mapping functions to their responses
        """
        # Separate parallel-capable from sequential tasks
        parallel_tasks = []
        sequential_tasks = []
        
        for function, messages, context in tasks:
            profile = self.cognitive_profiles.get(function)
            if profile and profile.parallel_capable:
                parallel_tasks.append((function, messages, context))
            else:
                sequential_tasks.append((function, messages, context))
        
        results = {}
        
        # Process parallel tasks concurrently
        if parallel_tasks:
            parallel_futures = [
                self.process_cognitive_task(func, msgs, ctx)
                for func, msgs, ctx in parallel_tasks
            ]
            
            parallel_results = await asyncio.gather(*parallel_futures, return_exceptions=True)
            
            for i, (function, _, _) in enumerate(parallel_tasks):
                result = parallel_results[i]
                if isinstance(result, Exception):
                    self.logger.error(f"Parallel task {function.value} failed: {result}")
                else:
                    results[function] = result
        
        # Process sequential tasks one by one
        for function, messages, context in sequential_tasks:
            try:
                result = await self.process_cognitive_task(function, messages, context)
                results[function] = result
            except Exception as e:
                self.logger.error(f"Sequential task {function.value} failed: {e}")
        
        # Update harmony score based on coordination
        self._update_harmony_score(results)
        
        return results
    
    async def _select_optimal_provider(
        self,
        function: CognitiveFunction,
        context: Optional[Dict[str, Any]],
        priority: str
    ) -> Tuple[Any, Dict[str, Any]]:
        """Select the optimal LLM provider for a cognitive function."""
        profile = self.cognitive_profiles.get(function)
        if not profile:
            # Fallback to default provider
            provider = self.llm_manager.get_provider()
            config = {"temperature": 0.7, "max_tokens": 2048}
            return provider, config
        
        # Try providers in order of preference
        for provider_name in profile.optimal_providers:
            if self.llm_manager.is_provider_configured(provider_name):
                try:
                    provider = self.llm_manager.get_provider(provider_name)
                    
                    # Configure based on cognitive profile
                    config = {
                        "temperature": profile.get_optimal_temperature(),
                        "max_tokens": profile.max_tokens
                    }
                    
                    # Adjust for priority
                    if priority == "critical":
                        config["temperature"] *= 0.8  # More focused for critical tasks
                    elif priority == "low":
                        config["max_tokens"] = min(config["max_tokens"], 1024)  # Limit tokens for low priority
                    
                    return provider, config
                    
                except Exception as e:
                    self.logger.warning(f"Could not use provider {provider_name} for {function.value}: {e}")
                    continue
        
        # Fallback to any available provider
        provider = self.llm_manager.get_provider()
        config = {"temperature": 0.7, "max_tokens": 2048}
        return provider, config
    
    async def _process_with_specialization(
        self,
        provider: Any,
        messages: List[LLMMessage],
        function: CognitiveFunction,
        config: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> LLMResponse:
        """Process messages with cognitive function specialization."""
        # Add function-specific system prompt enhancement
        enhanced_messages = self._enhance_messages_for_function(messages, function, context)
        
        # Process with provider
        response = await provider.generate(enhanced_messages, **config)
        
        # Post-process response based on cognitive function
        enhanced_response = self._enhance_response_for_function(response, function)
        
        return enhanced_response
    
    def _enhance_messages_for_function(
        self,
        messages: List[LLMMessage],
        function: CognitiveFunction,
        context: Optional[Dict[str, Any]]
    ) -> List[LLMMessage]:
        """Enhance messages with function-specific context."""
        profile = self.cognitive_profiles.get(function)
        if not profile:
            return messages
        
        # Create function-specific system enhancement
        enhancement = self._get_function_enhancement(function, context)
        
        # Add enhancement to system message or create new one
        enhanced_messages = messages.copy()
        if enhanced_messages and enhanced_messages[0].role == "system":
            enhanced_messages[0].content += f"\n\n{enhancement}"
        else:
            enhanced_messages.insert(0, LLMMessage(role="system", content=enhancement))
        
        return enhanced_messages
    
    def _get_function_enhancement(self, function: CognitiveFunction, context: Optional[Dict[str, Any]]) -> str:
        """Get function-specific enhancement text."""
        enhancements = {
            CognitiveFunction.CONSCIOUSNESS: """
You are operating in CONSCIOUSNESS mode. Focus on:
- Meta-cognitive analysis and self-reflection
- Understanding your own reasoning processes
- Identifying potential biases or limitations
- Considering multiple perspectives and their validity
- Questioning assumptions and exploring deeper meanings
""",
            CognitiveFunction.REASONING: """
You are operating in REASONING mode. Focus on:
- Logical analysis and structured thinking
- Breaking down complex problems systematically
- Identifying cause-and-effect relationships
- Using evidence-based reasoning
- Maintaining precision and accuracy in conclusions
""",
            CognitiveFunction.CREATIVITY: """
You are operating in CREATIVITY mode. Focus on:
- Generating novel ideas and unique perspectives
- Making unexpected connections between concepts
- Exploring unconventional solutions
- Embracing ambiguity and possibility
- Thinking outside established patterns
""",
            CognitiveFunction.CULTURAL: """
You are operating in CULTURAL INTELLIGENCE mode. Focus on:
- Understanding cultural context and sensitivity
- Recognizing diverse perspectives and values
- Avoiding cultural appropriation or insensitivity
- Respecting indigenous knowledge and rights
- Considering historical and social implications
""",
            CognitiveFunction.ARCHAEOLOGICAL: """
You are operating in ARCHAEOLOGICAL EXPERTISE mode. Focus on:
- Archaeological methodology and best practices
- Cultural heritage preservation principles
- Historical context and significance
- Interdisciplinary collaboration approaches
- Ethical considerations in archaeological work
"""
        }
        
        base_enhancement = enhancements.get(function, "")
        
        # Add context-specific enhancements
        if context:
            if "domain" in context:
                base_enhancement += f"\nDomain context: {context['domain']}"
            if "cultural_context" in context:
                base_enhancement += f"\nCultural context: {context['cultural_context']}"
        
        return base_enhancement
    
    def _enhance_response_for_function(self, response: LLMResponse, function: CognitiveFunction) -> LLMResponse:
        """Enhance response based on cognitive function."""
        # Add function-specific metadata
        if not hasattr(response, 'metadata'):
            response.metadata = {}
        
        response.metadata['cognitive_function'] = function.value
        response.metadata['specialization_applied'] = True
        
        return response
    
    def _update_performance_metrics(self, function: CognitiveFunction, task_id: str, response: LLMResponse):
        """Update performance metrics for cognitive function."""
        if function not in self.performance_metrics:
            self.performance_metrics[function] = {
                "total_tasks": 0,
                "avg_response_time": 0.0,
                "success_rate": 0.0,
                "avg_quality_score": 0.0
            }
        
        metrics = self.performance_metrics[function]
        task_info = self.active_processes.get(task_id, {})
        
        # Update metrics
        metrics["total_tasks"] += 1
        
        if "start_time" in task_info:
            response_time = time.time() - task_info["start_time"]
            metrics["avg_response_time"] = (
                (metrics["avg_response_time"] * (metrics["total_tasks"] - 1) + response_time) /
                metrics["total_tasks"]
            )
        
        # Simple success rate (could be enhanced with quality assessment)
        if response and response.content:
            success = 1.0
        else:
            success = 0.0
        
        metrics["success_rate"] = (
            (metrics["success_rate"] * (metrics["total_tasks"] - 1) + success) /
            metrics["total_tasks"]
        )
    
    def _update_harmony_score(self, results: Dict[CognitiveFunction, LLMResponse]):
        """Update the harmony score based on coordination between functions."""
        # Simple harmony metric based on successful coordination
        successful_functions = len([r for r in results.values() if r and r.content])
        total_functions = len(results)
        
        if total_functions > 0:
            current_harmony = successful_functions / total_functions
            # Exponential moving average
            self.coordination_state["harmony_score"] = (
                0.7 * self.coordination_state["harmony_score"] + 0.3 * current_harmony
            )
    
    def get_orchestra_status(self) -> Dict[str, Any]:
        """Get current status of the cognitive orchestra."""
        return {
            "active_processes": len(self.active_processes),
            "active_functions": list(self.coordination_state["active_functions"]),
            "harmony_score": self.coordination_state["harmony_score"],
            "performance_metrics": self.performance_metrics,
            "available_providers": self.llm_manager.get_configured_providers(),
            "cognitive_profiles": {
                func.value: {
                    "optimal_providers": profile.optimal_providers,
                    "latency_priority": profile.latency_priority,
                    "parallel_capable": profile.parallel_capable
                }
                for func, profile in self.cognitive_profiles.items()
            }
        }
    
    async def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
        await self.llm_manager.close() 
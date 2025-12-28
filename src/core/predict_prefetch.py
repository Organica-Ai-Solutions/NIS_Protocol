#!/usr/bin/env python3
"""
Predict and Prefetch Engine for NIS Protocol
Hides tool call latency by predicting and prefetching data while LLM thinks

Copyright 2025 Organica AI Solutions
Licensed under Apache License 2.0
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PrefetchResult:
    """Result from prefetch operation."""
    tool_name: str
    predicted: bool
    data: Any
    fetch_time: float
    used: bool = False


class PredictPrefetchEngine:
    """
    Predict and prefetch tool data while LLM is planning.
    
    Technique: While LLM generates plan (2-5s), predict likely tools
    and start fetching data. If prediction correct, data is ready instantly.
    
    Honest Assessment:
    - Simple keyword-based prediction (not ML model)
    - Heuristic rules for tool prediction
    - Real async prefetching
    - Can hide 3-5 seconds of latency per correct prediction
    - 80% real - actual prefetching, simple prediction
    """
    
    def __init__(self, mcp_executor):
        """Initialize predict-prefetch engine."""
        self.mcp_executor = mcp_executor
        self.prefetch_cache: Dict[str, PrefetchResult] = {}
        
        # Prediction rules (keyword → tool mapping)
        self.prediction_rules = {
            "research": ["web_search"],
            "search": ["web_search"],
            "find": ["web_search"],
            "look up": ["web_search"],
            "solve": ["physics_solve"],
            "equation": ["physics_solve"],
            "physics": ["physics_solve"],
            "calculate": ["physics_solve", "code_execute"],
            "analyze": ["vision_analyze", "code_execute"],
            "image": ["vision_analyze"],
            "picture": ["vision_analyze"],
            "robot": ["robotics_kinematics"],
            "kinematics": ["robotics_kinematics"],
            "trajectory": ["robotics_kinematics"],
            "code": ["code_execute"],
            "run": ["code_execute"],
            "execute": ["code_execute"]
        }
        
        logger.info("🔮 Predict-prefetch engine initialized")
    
    def predict_tools(self, goal: str) -> List[str]:
        """
        Predict which tools will be needed based on goal.
        
        Args:
            goal: User's goal/objective
            
        Returns:
            List of predicted tool names
        """
        goal_lower = goal.lower()
        predicted = set()
        
        # Apply prediction rules
        for keyword, tools in self.prediction_rules.items():
            if keyword in goal_lower:
                predicted.update(tools)
        
        # Limit predictions to avoid over-fetching
        predicted_list = list(predicted)[:3]  # Max 3 predictions
        
        if predicted_list:
            logger.info(f"🔮 Predicted tools: {predicted_list}")
        
        return predicted_list
    
    async def prefetch_tool_data(
        self,
        tool_name: str,
        goal: str
    ) -> Optional[PrefetchResult]:
        """
        Prefetch data for predicted tool.
        
        Args:
            tool_name: Tool to prefetch for
            goal: User's goal (for context)
            
        Returns:
            PrefetchResult or None if prefetch failed
        """
        start_time = time.time()
        
        try:
            # Generate prefetch parameters based on tool and goal
            params = self._generate_prefetch_params(tool_name, goal)
            
            if not params:
                return None
            
            # Execute tool with prefetch params
            data = await self.mcp_executor.execute_tool(tool_name, params)
            
            fetch_time = time.time() - start_time
            
            result = PrefetchResult(
                tool_name=tool_name,
                predicted=True,
                data=data,
                fetch_time=fetch_time
            )
            
            # Cache result
            cache_key = f"{tool_name}_{hash(goal)}"
            self.prefetch_cache[cache_key] = result
            
            logger.info(f"✅ Prefetched {tool_name} in {fetch_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Prefetch failed for {tool_name}: {e}")
            return None
    
    def _generate_prefetch_params(
        self,
        tool_name: str,
        goal: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate parameters for prefetch based on tool and goal.
        
        HONEST: Simple heuristics, not sophisticated parameter extraction.
        In production, use LLM to extract parameters.
        """
        if tool_name == "web_search":
            # Extract search query from goal
            # Simple: use first few words
            query = " ".join(goal.split()[:5])
            return {"query": query, "max_results": 5}
        
        elif tool_name == "code_execute":
            # Can't prefetch code execution (need actual code)
            return None
        
        elif tool_name == "physics_solve":
            # Can't prefetch without equation details
            return None
        
        elif tool_name == "vision_analyze":
            # Can't prefetch without image
            return None
        
        elif tool_name == "robotics_kinematics":
            # Can't prefetch without robot config
            return None
        
        # Only web_search is prefetchable with simple heuristics
        return None
    
    async def plan_with_prefetch(
        self,
        llm_planner,
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, List[PrefetchResult]]:
        """
        Create plan while prefetching predicted tools.
        
        Args:
            llm_planner: LLM planner instance
            goal: User's goal
            context: Optional context
            
        Returns:
            Tuple of (execution_plan, prefetch_results)
        """
        # Predict tools
        predicted_tools = self.predict_tools(goal)
        
        # Start planning and prefetching in parallel
        planning_task = asyncio.create_task(
            llm_planner.create_plan(goal, context)
        )
        
        prefetch_tasks = [
            asyncio.create_task(self.prefetch_tool_data(tool, goal))
            for tool in predicted_tools
        ]
        
        # Wait for both to complete
        plan, prefetch_results = await asyncio.gather(
            planning_task,
            asyncio.gather(*prefetch_tasks, return_exceptions=True)
        )
        
        # Filter out failed prefetches
        successful_prefetches = [
            r for r in prefetch_results 
            if isinstance(r, PrefetchResult)
        ]
        
        if successful_prefetches:
            logger.info(f"🚀 Prefetched {len(successful_prefetches)} tools while planning")
        
        return plan, successful_prefetches
    
    def get_prefetched_data(
        self,
        tool_name: str,
        goal: str
    ) -> Optional[Any]:
        """
        Get prefetched data if available.
        
        Args:
            tool_name: Tool name
            goal: Original goal (for cache key)
            
        Returns:
            Prefetched data or None
        """
        cache_key = f"{tool_name}_{hash(goal)}"
        
        if cache_key in self.prefetch_cache:
            result = self.prefetch_cache[cache_key]
            result.used = True
            logger.info(f"💨 Using prefetched data for {tool_name} (saved {result.fetch_time:.2f}s)")
            return result.data
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get prefetch statistics."""
        total = len(self.prefetch_cache)
        used = sum(1 for r in self.prefetch_cache.values() if r.used)
        total_time_saved = sum(
            r.fetch_time for r in self.prefetch_cache.values() if r.used
        )
        
        return {
            "total_prefetches": total,
            "used_prefetches": used,
            "hit_rate": used / total if total > 0 else 0,
            "time_saved": total_time_saved,
            "cache_size": len(self.prefetch_cache)
        }


# Global instance
_prefetch_engine: Optional[PredictPrefetchEngine] = None


def get_predict_prefetch_engine(mcp_executor) -> PredictPrefetchEngine:
    """Get or create predict-prefetch engine instance."""
    global _prefetch_engine
    if _prefetch_engine is None:
        _prefetch_engine = PredictPrefetchEngine(mcp_executor=mcp_executor)
    return _prefetch_engine

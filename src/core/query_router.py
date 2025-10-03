"""
Intelligent Query Router for Chat Optimization

Pattern-based classifier that routes queries to optimal processing pipelines based on:
- Query complexity (low, medium, high)
- Query type (simple chat, technical, physics, research, creative)
- Performance requirements
- Context needs

ARCHITECTURE: Pattern-matching router (NOT a true MoE neural network)
- Uses regex patterns and heuristics for classification
- Routes to different processing paths (FAST, STANDARD, FULL)
- Inspired by Mixture of Experts concept but implemented as rule-based routing

Similar to consensus system for LLM selection, but for query processing paths.
"""

import re
from typing import Dict, Any, List, Optional
from enum import Enum
import logging

logger = logging.getLogger("nis.query_router")


class QueryType(Enum):
    """Types of queries that require different processing"""
    SIMPLE_CHAT = "simple_chat"          # Greetings, simple questions
    TECHNICAL = "technical"               # Technical NIS questions
    PHYSICS = "physics"                   # Physics/scientific queries
    RESEARCH = "research"                 # Deep research queries
    CREATIVE = "creative"                 # Creative/generative tasks


class ProcessingPath(Enum):
    """Available processing paths"""
    FAST = "fast"                        # Skip pipeline, minimal context (2-3s)
    STANDARD = "standard"                # Light pipeline, normal context (5-7s)
    FULL = "full"                        # Full pipeline, deep context (10-15s)


class QueryRouter:
    """
    Intelligent pattern-based router that selects optimal processing path
    
    Uses regex patterns and heuristics (NOT neural networks) to classify queries
    and route them to appropriate processing pipelines.
    
    Similar to how consensus controller picks LLM providers,
    this picks the right processing path (FAST/STANDARD/FULL) for each query.
    
    Inspired by MoE concept but implemented as rule-based classifier.
    """
    
    def __init__(self):
        self.logger = logger
        
        # Pattern recognition for query types
        self.patterns = {
            QueryType.SIMPLE_CHAT: [
                r'\b(hi|hello|hey|thanks|thank you|bye|goodbye)\b',
                r'\bhow are you\b',
                r'\bwhat is your name\b',
                r'\bcan you help\b'
            ],
            QueryType.PHYSICS: [
                r'\b(physics|force|energy|momentum|velocity|acceleration)\b',
                r'\b(laplace|fourier|transform|frequency)\b',
                r'\b(conservation|thermodynamics|quantum)\b',
                r'\b(PINN|physics-informed|PDE|differential equation)\b'
            ],
            QueryType.TECHNICAL: [
                r'\b(NIS|protocol|architecture|agent|pipeline)\b',
                r'\b(API|endpoint|integration|configuration)\b',
                r'\b(KAN|neural|network|model)\b',
                r'\b(MCP|A2A|ACP|protocol adapter)\b'
            ],
            QueryType.RESEARCH: [
                r'\b(research|analyze|investigate|explore deeply)\b',
                r'\b(compare|contrast|evaluate|assess)\b',
                r'\b(comprehensive|detailed analysis)\b',
                r'\bwhat are the differences between\b'
            ],
            QueryType.CREATIVE: [
                r'\b(generate|create|write|compose|design)\b',
                r'\b(story|poem|code|script|plan)\b',
                r'\b(imagine|brainstorm|suggest)\b'
            ]
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            'high': [
                r'\bexplain in detail\b',
                r'\bcomprehensive\b',
                r'\bstep by step\b',
                r'\bhow does.*work\b',
                r'\bwhat are all\b'
            ],
            'low': [
                r'^(yes|no|ok|sure|maybe)\b',
                r'^\w{1,5}$',  # Single word
                r'^[^.?!]{1,20}[.?!]?$'  # Very short sentences
            ]
        }
    
    def route_query(
        self,
        query: str,
        context_size: int = 0,
        user_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Route query to optimal processing path (MoE decision)
        
        Args:
            query: User query
            context_size: Number of messages in conversation
            user_preference: User's speed preference ('fast', 'balanced', 'quality')
            
        Returns:
            Routing decision with processing path and configuration
        """
        # Classify query
        query_type = self._classify_query_type(query)
        complexity = self._assess_complexity(query)
        
        # Determine processing path (MoE routing logic)
        path = self._select_processing_path(
            query_type=query_type,
            complexity=complexity,
            context_size=context_size,
            user_preference=user_preference
        )
        
        # Generate configuration for selected path
        config = self._generate_path_config(path, query_type, complexity)
        
        self.logger.info(
            f"🎯 Query Router: type={query_type.value}, "
            f"complexity={complexity}, path={path.value}"
        )
        
        return {
            "query_type": query_type.value,
            "complexity": complexity,
            "processing_path": path.value,
            "config": config,
            "estimated_time": config["estimated_time"],
            "reasoning": self._explain_routing(query_type, complexity, path)
        }
    
    def _classify_query_type(self, query: str) -> QueryType:
        """Classify query into one of the expert types"""
        query_lower = query.lower()
        
        # Check patterns for each type
        type_scores = {}
        
        for qtype, patterns in self.patterns.items():
            score = sum(
                1 for pattern in patterns
                if re.search(pattern, query_lower, re.IGNORECASE)
            )
            type_scores[qtype] = score
        
        # Return type with highest score, default to TECHNICAL
        if max(type_scores.values()) > 0:
            return max(type_scores, key=type_scores.get)
        
        # If no patterns match, use heuristics
        word_count = len(query.split())
        if word_count < 10:
            return QueryType.SIMPLE_CHAT
        elif any(term in query_lower for term in ['nis', 'protocol', 'agent', 'system']):
            return QueryType.TECHNICAL
        else:
            return QueryType.TECHNICAL  # Default
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity"""
        query_lower = query.lower()
        
        # Check high complexity indicators
        high_score = sum(
            1 for pattern in self.complexity_indicators['high']
            if re.search(pattern, query_lower, re.IGNORECASE)
        )
        
        # Check low complexity indicators
        low_score = sum(
            1 for pattern in self.complexity_indicators['low']
            if re.search(pattern, query_lower, re.IGNORECASE)
        )
        
        # Word count and structure
        word_count = len(query.split())
        has_multiple_questions = query.count('?') > 1
        
        # Determine complexity
        if low_score > 0 or word_count < 5:
            return "low"
        elif high_score > 0 or word_count > 50 or has_multiple_questions:
            return "high"
        else:
            return "medium"
    
    def _select_processing_path(
        self,
        query_type: QueryType,
        complexity: str,
        context_size: int,
        user_preference: Optional[str]
    ) -> ProcessingPath:
        """
        Smart routing decision - select optimal processing path
        
        Pattern-based selection using query characteristics.
        Similar to consensus system's LLM provider selection,
        but for choosing processing pipelines (FAST/STANDARD/FULL).
        """
        # User preference override
        if user_preference == "fast":
            return ProcessingPath.FAST
        elif user_preference == "quality":
            return ProcessingPath.FULL
        
        # Smart routing based on query characteristics
        if query_type == QueryType.SIMPLE_CHAT and complexity == "low":
            # Fast path: no pipeline, minimal context
            return ProcessingPath.FAST
        
        elif query_type == QueryType.PHYSICS or (query_type == QueryType.TECHNICAL and complexity == "high"):
            # Full path: complete pipeline needed
            return ProcessingPath.FULL
        
        elif query_type == QueryType.RESEARCH or complexity == "high":
            # Full path: deep analysis needed
            return ProcessingPath.FULL
        
        else:
            # Standard path: balanced approach
            return ProcessingPath.STANDARD
    
    def _generate_path_config(
        self,
        path: ProcessingPath,
        query_type: QueryType,
        complexity: str
    ) -> Dict[str, Any]:
        """Generate configuration for selected processing path"""
        
        if path == ProcessingPath.FAST:
            return {
                "skip_pipeline": True,
                "max_context_messages": 5,
                "enable_semantic_search": False,
                "temperature": 0.7,
                "max_tokens": 500,
                "estimated_time": "2-3s",
                "description": "Fast response - minimal processing"
            }
        
        elif path == ProcessingPath.STANDARD:
            return {
                "skip_pipeline": False,
                "pipeline_mode": "light",  # Light pipeline processing
                "max_context_messages": 10,
                "enable_semantic_search": True,
                "temperature": 0.7,
                "max_tokens": 1000,
                "estimated_time": "5-7s",
                "description": "Balanced response - standard processing"
            }
        
        else:  # ProcessingPath.FULL
            return {
                "skip_pipeline": False,
                "pipeline_mode": "full",  # Complete pipeline
                "max_context_messages": 20,
                "enable_semantic_search": True,
                "temperature": 0.8,
                "max_tokens": 2000,
                "estimated_time": "10-15s",
                "description": "Deep response - full pipeline analysis"
            }
    
    def _explain_routing(
        self,
        query_type: QueryType,
        complexity: str,
        path: ProcessingPath
    ) -> str:
        """Explain why this routing was chosen"""
        return (
            f"Query classified as {query_type.value} with {complexity} complexity. "
            f"Routing to {path.value} path for optimal speed/quality balance."
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get routing performance statistics"""
        # TODO: Implement stats tracking
        return {
            "total_queries": 0,
            "fast_path_usage": 0,
            "standard_path_usage": 0,
            "full_path_usage": 0
        }


# Global router instance
query_router = QueryRouter()


def route_chat_query(
    query: str,
    context_size: int = 0,
    user_preference: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to route a query
    
    Usage:
        routing = route_chat_query("Hello, how are you?")
        if routing["config"]["skip_pipeline"]:
            # Use fast path
        else:
            # Use full pipeline
    """
    return query_router.route_query(query, context_size, user_preference)


"""
NIS Protocol Long-Term Memory Consolidator

This module consolidates short-term memories into structured long-term storage
with sophisticated importance scoring, pattern recognition, and memory optimization.
"""

import logging
import time
import math
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
from datetime import datetime, timedelta

from ...core.agent import NISAgent, NISLayer
from ..memory_manager import MemoryManager


@dataclass
class MemoryImportanceScore:
    """Importance scoring for memory consolidation."""
    recency_score: float
    frequency_score: float
    emotional_score: float
    relevance_score: float
    uniqueness_score: float
    overall_score: float
    consolidation_priority: str


@dataclass
class ConsolidatedMemory:
    """Consolidated memory structure."""
    memory_id: str
    content: Dict[str, Any]
    importance_score: MemoryImportanceScore
    consolidation_timestamp: float
    source_memories: List[str]
    memory_type: str
    tags: List[str]
    connections: List[str]
    access_count: int
    last_accessed: float


class LTMConsolidator(NISAgent):
    """Consolidates short-term memories into structured long-term storage."""
    
    def __init__(
        self,
        agent_id: str = "ltm_consolidator",
        description: str = "Long-term memory consolidation and optimization agent"
    ):
        super().__init__(agent_id, NISLayer.MEMORY, description)
        self.logger = logging.getLogger(f"nis.{agent_id}")
        
        # Initialize memory manager
        self.memory = MemoryManager()
        
        # Consolidation parameters
        self.consolidation_threshold = 0.7  # Minimum importance score for LTM
        self.max_stm_age_hours = 24  # Maximum age for STM before forced consolidation
        self.consolidation_batch_size = 50  # Process memories in batches
        self.similarity_threshold = 0.8  # Threshold for memory similarity
        
        # Memory type weights for importance scoring
        self.memory_type_weights = {
            "episodic": 1.0,      # Personal experiences
            "semantic": 0.9,      # Facts and knowledge
            "procedural": 0.8,    # Skills and procedures
            "emotional": 1.2,     # Emotionally significant events
            "goal_related": 1.1,  # Goal-relevant information
            "social": 0.9,        # Social interactions
            "learning": 1.0,      # Learning experiences
            "error": 1.1,         # Mistakes and corrections
            "success": 0.9,       # Successful outcomes
            "routine": 0.6        # Routine activities
        }
        
        # Emotional impact weights
        self.emotional_weights = {
            "joy": 1.2, "surprise": 1.1, "fear": 1.3, "anger": 1.2,
            "sadness": 1.1, "disgust": 1.0, "anticipation": 1.0, "trust": 0.9
        }
        
        # Consolidation statistics
        self.consolidation_stats = {
            "total_processed": 0,
            "consolidated_to_ltm": 0,
            "discarded": 0,
            "merged": 0,
            "last_consolidation": 0
        }
        
        # Memory connections graph
        self.memory_connections = defaultdict(set)
        
        self.logger.info(f"Initialized {self.__class__.__name__} with threshold {self.consolidation_threshold}")
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process memory consolidation requests."""
        start_time = self._start_processing_timer()
        
        try:
            operation = message.get("operation", "consolidate_memories")
            
            if operation == "consolidate_memories":
                result = self._consolidate_memories(message)
            elif operation == "evaluate_importance":
                result = self._evaluate_memory_importance(message)
            elif operation == "optimize_ltm":
                result = self._optimize_long_term_memory(message)
            elif operation == "get_consolidation_stats":
                result = self._get_consolidation_statistics(message)
            elif operation == "force_consolidation":
                result = self._force_consolidation(message)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            response = self._create_response(
                "success",
                result,
                {"operation": operation, "consolidation_threshold": self.consolidation_threshold}
            )
            
        except Exception as e:
            self.logger.error(f"Error in memory consolidation: {str(e)}")
            response = self._create_response(
                "error",
                {"error": str(e)},
                {"operation": operation}
            )
        
        finally:
            self._end_processing_timer(start_time)
        
        return response
    
    def _consolidate_memories(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Main memory consolidation process."""
        short_term_memories = message.get("short_term_memories", [])
        force_consolidation = message.get("force_consolidation", False)
        
        if not short_term_memories:
            # Retrieve STM from memory manager
            short_term_memories = self._retrieve_short_term_memories()
        
        self.logger.info(f"Starting consolidation of {len(short_term_memories)} memories")
        
        # Process memories in batches
        consolidated_memories = []
        discarded_memories = []
        merged_memories = []
        
        for i in range(0, len(short_term_memories), self.consolidation_batch_size):
            batch = short_term_memories[i:i + self.consolidation_batch_size]
            batch_result = self._process_memory_batch(batch, force_consolidation)
            
            consolidated_memories.extend(batch_result["consolidated"])
            discarded_memories.extend(batch_result["discarded"])
            merged_memories.extend(batch_result["merged"])
        
        # Update statistics
        self.consolidation_stats["total_processed"] += len(short_term_memories)
        self.consolidation_stats["consolidated_to_ltm"] += len(consolidated_memories)
        self.consolidation_stats["discarded"] += len(discarded_memories)
        self.consolidation_stats["merged"] += len(merged_memories)
        self.consolidation_stats["last_consolidation"] = time.time()
        
        # Store consolidated memories in LTM
        for memory in consolidated_memories:
            self._store_in_long_term_memory(memory)
        
        # Update memory connections
        self._update_memory_connections(consolidated_memories)
        
        return {
            "consolidation_summary": {
                "total_processed": len(short_term_memories),
                "consolidated_to_ltm": len(consolidated_memories),
                "discarded": len(discarded_memories),
                "merged": len(merged_memories),
                "consolidation_rate": len(consolidated_memories) / len(short_term_memories) if short_term_memories else 0
            },
            "consolidated_memories": [mem.memory_id for mem in consolidated_memories],
            "memory_connections_updated": len(self.memory_connections),
            "next_consolidation_recommended": self._calculate_next_consolidation_time()
        }
    
    def _process_memory_batch(
        self,
        memory_batch: List[Dict[str, Any]],
        force_consolidation: bool = False
    ) -> Dict[str, List]:
        """Process a batch of memories for consolidation."""
        consolidated = []
        discarded = []
        merged = []
        
        # Group similar memories for potential merging
        memory_groups = self._group_similar_memories(memory_batch)
        
        for group in memory_groups:
            if len(group) == 1:
                # Single memory - evaluate for consolidation
                memory = group[0]
                importance = self._calculate_importance_score(memory)
                
                if importance.overall_score >= self.consolidation_threshold or force_consolidation:
                    consolidated_memory = self._create_consolidated_memory(memory, importance)
                    consolidated.append(consolidated_memory)
                else:
                    discarded.append(memory.get("memory_id", "unknown"))
            else:
                # Multiple similar memories - merge them
                merged_memory = self._merge_similar_memories(group)
                importance = self._calculate_importance_score(merged_memory)
                
                if importance.overall_score >= self.consolidation_threshold or force_consolidation:
                    consolidated_memory = self._create_consolidated_memory(merged_memory, importance)
                    consolidated.append(consolidated_memory)
                    merged.extend([mem.get("memory_id", "unknown") for mem in group])
                else:
                    discarded.extend([mem.get("memory_id", "unknown") for mem in group])
        
        return {
            "consolidated": consolidated,
            "discarded": discarded,
            "merged": merged
        }
    
    def _calculate_importance_score(self, memory: Dict[str, Any]) -> MemoryImportanceScore:
        """Calculate comprehensive importance score for a memory."""
        # Recency score (more recent = higher score)
        timestamp = memory.get("timestamp", time.time())
        age_hours = (time.time() - timestamp) / 3600
        recency_score = math.exp(-age_hours / 24)  # Exponential decay over 24 hours
        
        # Frequency score (how often this type of memory occurs)
        memory_type = memory.get("type", "general")
        frequency_score = self._calculate_frequency_score(memory)
        
        # Emotional score (emotional significance)
        emotional_score = self._calculate_emotional_score(memory)
        
        # Relevance score (relevance to current goals and context)
        relevance_score = self._calculate_relevance_score(memory)
        
        # Uniqueness score (how unique/novel this memory is)
        uniqueness_score = self._calculate_uniqueness_score(memory)
        
        # Calculate weighted overall score
        type_weight = self.memory_type_weights.get(memory_type, 1.0)
        
        overall_score = (
            recency_score * 0.2 +
            frequency_score * 0.15 +
            emotional_score * 0.25 +
            relevance_score * 0.25 +
            uniqueness_score * 0.15
        ) * type_weight
        
        # Determine consolidation priority
        if overall_score >= 0.9:
            priority = "critical"
        elif overall_score >= 0.7:
            priority = "high"
        elif overall_score >= 0.5:
            priority = "medium"
        elif overall_score >= 0.3:
            priority = "low"
        else:
            priority = "discard"
        
        return MemoryImportanceScore(
            recency_score=recency_score,
            frequency_score=frequency_score,
            emotional_score=emotional_score,
            relevance_score=relevance_score,
            uniqueness_score=uniqueness_score,
            overall_score=overall_score,
            consolidation_priority=priority
        )
    
    def _calculate_frequency_score(self, memory: Dict[str, Any]) -> float:
        """Calculate frequency-based importance score."""
        # Count similar memories in recent history
        memory_content = str(memory.get("content", ""))
        memory_type = memory.get("type", "general")
        
        # Simple frequency calculation based on content similarity
        # In a full implementation, this would use more sophisticated similarity measures
        similar_count = 0
        recent_memories = self._get_recent_memories(hours=168)  # Last week
        
        for recent_memory in recent_memories:
            if recent_memory.get("type") == memory_type:
                content_similarity = self._calculate_content_similarity(
                    memory_content, 
                    str(recent_memory.get("content", ""))
                )
                if content_similarity > 0.7:
                    similar_count += 1
        
        # Inverse frequency - rare events are more important
        if similar_count == 0:
            return 1.0
        else:
            return 1.0 / (1.0 + math.log(similar_count))
    
    def _calculate_emotional_score(self, memory: Dict[str, Any]) -> float:
        """Calculate emotional significance score."""
        emotional_state = memory.get("emotional_state", {})
        
        if not emotional_state:
            return 0.5  # Neutral emotional score
        
        # Calculate weighted emotional intensity
        total_weight = 0
        weighted_intensity = 0
        
        for emotion, intensity in emotional_state.items():
            if emotion in self.emotional_weights:
                weight = self.emotional_weights[emotion]
                weighted_intensity += intensity * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        # Normalize to 0-1 range
        emotional_score = weighted_intensity / total_weight
        return min(1.0, max(0.0, emotional_score))
    
    def _calculate_relevance_score(self, memory: Dict[str, Any]) -> float:
        """Calculate relevance to current goals and context."""
        # Get current active goals and context
        current_goals = self._get_current_goals()
        current_context = self._get_current_context()
        
        memory_content = str(memory.get("content", "")).lower()
        memory_tags = memory.get("tags", [])
        
        relevance_score = 0.0
        
        # Check relevance to current goals
        for goal in current_goals:
            goal_keywords = self._extract_keywords(str(goal))
            for keyword in goal_keywords:
                if keyword.lower() in memory_content:
                    relevance_score += 0.2
        
        # Check relevance to current context
        context_keywords = self._extract_keywords(str(current_context))
        for keyword in context_keywords:
            if keyword.lower() in memory_content:
                relevance_score += 0.1
        
        # Check tag relevance
        relevant_tags = ["goal", "important", "learning", "error", "success"]
        for tag in memory_tags:
            if tag.lower() in relevant_tags:
                relevance_score += 0.15
        
        return min(1.0, relevance_score)
    
    def _calculate_uniqueness_score(self, memory: Dict[str, Any]) -> float:
        """Calculate how unique/novel this memory is."""
        memory_content = str(memory.get("content", ""))
        
        # Compare with existing LTM
        existing_memories = self._get_long_term_memories(limit=1000)
        
        max_similarity = 0.0
        for existing_memory in existing_memories:
            existing_content = str(existing_memory.get("content", ""))
            similarity = self._calculate_content_similarity(memory_content, existing_content)
            max_similarity = max(max_similarity, similarity)
        
        # Uniqueness is inverse of maximum similarity
        uniqueness_score = 1.0 - max_similarity
        return max(0.0, uniqueness_score)
    
    def _group_similar_memories(self, memories: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group similar memories for potential merging."""
        groups = []
        processed = set()
        
        for i, memory in enumerate(memories):
            if i in processed:
                continue
            
            group = [memory]
            processed.add(i)
            
            for j, other_memory in enumerate(memories[i+1:], i+1):
                if j in processed:
                    continue
                
                similarity = self._calculate_memory_similarity(memory, other_memory)
                if similarity >= self.similarity_threshold:
                    group.append(other_memory)
                    processed.add(j)
            
            groups.append(group)
        
        return groups
    
    def _calculate_memory_similarity(self, memory1: Dict[str, Any], memory2: Dict[str, Any]) -> float:
        """Calculate similarity between two memories."""
        # Content similarity
        content1 = str(memory1.get("content", ""))
        content2 = str(memory2.get("content", ""))
        content_similarity = self._calculate_content_similarity(content1, content2)
        
        # Type similarity
        type1 = memory1.get("type", "")
        type2 = memory2.get("type", "")
        type_similarity = 1.0 if type1 == type2 else 0.0
        
        # Temporal similarity
        timestamp1 = memory1.get("timestamp", 0)
        timestamp2 = memory2.get("timestamp", 0)
        time_diff = abs(timestamp1 - timestamp2)
        temporal_similarity = math.exp(-time_diff / 3600)  # Decay over hours
        
        # Tag similarity
        tags1 = set(memory1.get("tags", []))
        tags2 = set(memory2.get("tags", []))
        if tags1 or tags2:
            tag_similarity = len(tags1.intersection(tags2)) / len(tags1.union(tags2))
        else:
            tag_similarity = 0.0
        
        # Weighted overall similarity
        overall_similarity = (
            content_similarity * 0.4 +
            type_similarity * 0.2 +
            temporal_similarity * 0.2 +
            tag_similarity * 0.2
        )
        
        return overall_similarity
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings."""
        # Simple word-based similarity (in practice, would use more sophisticated NLP)
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _merge_similar_memories(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge similar memories into a single consolidated memory."""
        if len(memories) == 1:
            return memories[0]
        
        # Create merged memory
        merged_memory = {
            "memory_id": f"merged_{int(time.time())}_{len(memories)}",
            "type": "merged",
            "timestamp": max(mem.get("timestamp", 0) for mem in memories),
            "content": self._merge_memory_contents(memories),
            "tags": list(set().union(*[mem.get("tags", []) for mem in memories])),
            "source_memories": [mem.get("memory_id", "unknown") for mem in memories],
            "merge_count": len(memories),
            "emotional_state": self._merge_emotional_states(memories)
        }
        
        return merged_memory
    
    def _merge_memory_contents(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge content from multiple memories."""
        merged_content = {
            "summary": "Consolidated memory from multiple similar experiences",
            "individual_memories": [],
            "common_elements": [],
            "frequency_data": {}
        }
        
        # Extract individual memory contents
        for memory in memories:
            content = memory.get("content", {})
            merged_content["individual_memories"].append({
                "timestamp": memory.get("timestamp"),
                "content": content,
                "memory_id": memory.get("memory_id")
            })
        
        # Identify common elements
        all_words = []
        for memory in memories:
            content_str = str(memory.get("content", ""))
            all_words.extend(content_str.lower().split())
        
        word_counts = Counter(all_words)
        common_words = [word for word, count in word_counts.items() if count > 1]
        merged_content["common_elements"] = common_words[:10]  # Top 10 common elements
        merged_content["frequency_data"] = dict(word_counts.most_common(20))
        
        return merged_content
    
    def _merge_emotional_states(self, memories: List[Dict[str, Any]]) -> Dict[str, float]:
        """Merge emotional states from multiple memories."""
        all_emotions = defaultdict(list)
        
        for memory in memories:
            emotional_state = memory.get("emotional_state", {})
            for emotion, intensity in emotional_state.items():
                all_emotions[emotion].append(intensity)
        
        # Calculate average emotional intensity for each emotion
        merged_emotions = {}
        for emotion, intensities in all_emotions.items():
            merged_emotions[emotion] = sum(intensities) / len(intensities)
        
        return merged_emotions
    
    def _create_consolidated_memory(
        self,
        memory: Dict[str, Any],
        importance: MemoryImportanceScore
    ) -> ConsolidatedMemory:
        """Create a consolidated memory object."""
        memory_id = memory.get("memory_id", f"consolidated_{int(time.time())}")
        
        return ConsolidatedMemory(
            memory_id=memory_id,
            content=memory.get("content", {}),
            importance_score=importance,
            consolidation_timestamp=time.time(),
            source_memories=memory.get("source_memories", [memory_id]),
            memory_type=memory.get("type", "general"),
            tags=memory.get("tags", []),
            connections=[],  # Will be populated later
            access_count=0,
            last_accessed=time.time()
        )
    
    def _store_in_long_term_memory(self, consolidated_memory: ConsolidatedMemory) -> None:
        """Store consolidated memory in long-term memory."""
        memory_data = {
            "memory_id": consolidated_memory.memory_id,
            "content": consolidated_memory.content,
            "importance_score": consolidated_memory.importance_score.__dict__,
            "consolidation_timestamp": consolidated_memory.consolidation_timestamp,
            "source_memories": consolidated_memory.source_memories,
            "memory_type": consolidated_memory.memory_type,
            "tags": consolidated_memory.tags,
            "connections": consolidated_memory.connections,
            "access_count": consolidated_memory.access_count,
            "last_accessed": consolidated_memory.last_accessed
        }
        
        # Store in memory manager with long TTL
        self.memory.store(
            f"ltm_{consolidated_memory.memory_id}",
            memory_data,
            ttl=86400 * 365  # Keep for 1 year
        )
        
        self.logger.debug(f"Stored consolidated memory: {consolidated_memory.memory_id}")
    
    def _update_memory_connections(self, consolidated_memories: List[ConsolidatedMemory]) -> None:
        """Update memory connection graph."""
        for memory in consolidated_memories:
            # Find connections based on content similarity, temporal proximity, etc.
            connections = self._find_memory_connections(memory)
            memory.connections = connections
            self.memory_connections[memory.memory_id].update(connections)
    
    def _find_memory_connections(self, memory: ConsolidatedMemory) -> List[str]:
        """Find connections to other memories."""
        connections = []
        
        # Get existing LTM for comparison
        existing_memories = self._get_long_term_memories(limit=500)
        
        for existing_memory in existing_memories:
            if existing_memory.get("memory_id") == memory.memory_id:
                continue
            
            # Calculate connection strength
            connection_strength = self._calculate_connection_strength(memory, existing_memory)
            
            if connection_strength > 0.6:  # Threshold for meaningful connection
                connections.append(existing_memory.get("memory_id"))
        
        return connections[:10]  # Limit to top 10 connections
    
    def _calculate_connection_strength(
        self,
        memory1: ConsolidatedMemory,
        memory2: Dict[str, Any]
    ) -> float:
        """Calculate connection strength between two memories."""
        # Content similarity
        content1 = str(memory1.content)
        content2 = str(memory2.get("content", ""))
        content_similarity = self._calculate_content_similarity(content1, content2)
        
        # Tag overlap
        tags1 = set(memory1.tags)
        tags2 = set(memory2.get("tags", []))
        if tags1 or tags2:
            tag_similarity = len(tags1.intersection(tags2)) / len(tags1.union(tags2))
        else:
            tag_similarity = 0.0
        
        # Type similarity
        type_similarity = 1.0 if memory1.memory_type == memory2.get("memory_type") else 0.0
        
        # Calculate overall connection strength
        connection_strength = (
            content_similarity * 0.5 +
            tag_similarity * 0.3 +
            type_similarity * 0.2
        )
        
        return connection_strength
    
    # Helper methods
    def _retrieve_short_term_memories(self) -> List[Dict[str, Any]]:
        """Retrieve short-term memories from memory manager."""
        # This would interface with the memory manager to get STM
        # For now, return empty list
        return []
    
    def _get_recent_memories(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent memories within specified time window."""
        cutoff_time = time.time() - (hours * 3600)
        # This would query the memory manager
        return []
    
    def _get_long_term_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get existing long-term memories."""
        # This would query the memory manager for LTM
        return []
    
    def _get_current_goals(self) -> List[Dict[str, Any]]:
        """Get current active goals."""
        # This would interface with the goal system
        return []
    
    def _get_current_context(self) -> Dict[str, Any]:
        """Get current context information."""
        # This would get current system context
        return {}
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction (in practice, would use NLP)
        words = text.lower().split()
        # Filter out common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:10]  # Return top 10 keywords
    
    def _calculate_next_consolidation_time(self) -> float:
        """Calculate when next consolidation should occur."""
        # Base interval of 6 hours, adjusted based on memory load
        base_interval = 6 * 3600  # 6 hours in seconds
        return time.time() + base_interval
    
    def _evaluate_memory_importance(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate importance of a specific memory."""
        memory = message.get("memory", {})
        importance = self._calculate_importance_score(memory)
        
        return {
            "memory_id": memory.get("memory_id", "unknown"),
            "importance_score": importance.__dict__,
            "consolidation_recommendation": importance.consolidation_priority
        }
    
    def _optimize_long_term_memory(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize long-term memory storage."""
        # This would implement memory optimization strategies
        # such as removing rarely accessed memories, compressing similar memories, etc.
        
        optimization_results = {
            "memories_optimized": 0,
            "storage_saved": 0,
            "connections_updated": 0,
            "optimization_timestamp": time.time()
        }
        
        return optimization_results
    
    def _get_consolidation_statistics(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Get consolidation statistics."""
        return {
            "consolidation_stats": self.consolidation_stats,
            "memory_connections_count": len(self.memory_connections),
            "consolidation_threshold": self.consolidation_threshold,
            "memory_type_weights": self.memory_type_weights
        }
    
    def _force_consolidation(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Force consolidation of all pending memories."""
        memories_to_consolidate = message.get("memories", [])
        
        if not memories_to_consolidate:
            memories_to_consolidate = self._retrieve_short_term_memories()
        
        # Force consolidation regardless of importance scores
        result = self._consolidate_memories({
            "short_term_memories": memories_to_consolidate,
            "force_consolidation": True
        })
        
        return result 
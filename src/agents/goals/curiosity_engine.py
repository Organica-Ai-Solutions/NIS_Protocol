"""
NIS Protocol Curiosity Engine

This module implements the curiosity drive mechanism that motivates exploration,
learning, and knowledge acquisition in the AGI system.
"""

import time
import random
import logging
import math
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from ...memory.memory_manager import MemoryManager


class CuriosityType(Enum):
    """Types of curiosity that can drive exploration"""
    EPISTEMIC = "epistemic"           # Knowledge-seeking curiosity
    DIVERSIVE = "diversive"           # Novelty-seeking curiosity
    SPECIFIC = "specific"             # Specific question-driven curiosity
    PERCEPTUAL = "perceptual"         # Sensory exploration curiosity
    EMPATHIC = "empathic"            # Understanding others' perspectives
    CREATIVE = "creative"             # Creative exploration curiosity


@dataclass
class CuriositySignal:
    """Represents a curiosity signal driving exploration"""
    signal_id: str
    curiosity_type: CuriosityType
    intensity: float
    focus_area: str
    specific_questions: List[str]
    context: Dict[str, Any]
    timestamp: float
    decay_rate: float
    satisfaction_threshold: float


@dataclass
class ExplorationTarget:
    """Represents a target for curiosity-driven exploration"""
    target_id: str
    domain: str
    description: str
    novelty_score: float
    complexity_score: float
    potential_value: float
    exploration_cost: float
    prerequisites: List[str]


class CuriosityEngine:
    """Engine that generates and manages curiosity-driven exploration.
    
    This engine provides:
    - Curiosity signal generation based on knowledge gaps
    - Novelty detection and assessment
    - Exploration target identification
    - Curiosity satisfaction tracking
    """
    
    def __init__(self):
        """Initialize the curiosity engine."""
        self.logger = logging.getLogger("nis.curiosity_engine")
        self.memory = MemoryManager()
        
        # Curiosity state
        self.active_curiosity_signals: Dict[str, CuriositySignal] = {}
        self.exploration_targets: Dict[str, ExplorationTarget] = {}
        self.knowledge_map: Dict[str, Any] = {}
        
        # Curiosity parameters
        self.base_curiosity_level = 0.6
        self.novelty_threshold = 0.7
        self.complexity_preference = 0.8
        self.exploration_budget = 1.0  # Resource allocation for exploration
        
        # Learning and satisfaction tracking
        self.curiosity_satisfaction_history: List[Dict[str, Any]] = []
        self.knowledge_growth_rate = 0.0
        
        self.logger.info("CuriosityEngine initialized")
    
    def detect_knowledge_gaps(
        self,
        current_context: Dict[str, Any],
        knowledge_base: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect gaps in knowledge that could drive curiosity.
        
        Args:
            current_context: Current situational context
            knowledge_base: Available knowledge
            
        Returns:
            List of detected knowledge gaps
        """
        self.logger.info("Detecting knowledge gaps")
        
        gaps = []
        
        # 1. Detect missing knowledge in current domain
        domain_gaps = self._detect_domain_knowledge_gaps(current_context, knowledge_base)
        gaps.extend(domain_gaps)
        
        # 2. Identify inconsistencies in knowledge
        inconsistency_gaps = self._detect_knowledge_inconsistencies(knowledge_base)
        gaps.extend(inconsistency_gaps)
        
        # 3. Find unexplored related areas
        exploration_gaps = self._detect_unexplored_areas(current_context, knowledge_base)
        gaps.extend(exploration_gaps)
        
        # 4. Identify questions without answers
        unanswered_gaps = self._detect_unanswered_questions(current_context, knowledge_base)
        gaps.extend(unanswered_gaps)
        
        # 5. Find incomplete understanding chains
        chain_gaps = self._detect_incomplete_chains(knowledge_base)
        gaps.extend(chain_gaps)
        
        # Score and rank gaps
        scored_gaps = self._score_knowledge_gaps(gaps, current_context)
        
        # Filter and return top gaps
        return sorted(scored_gaps, key=lambda x: x["total_score"], reverse=True)[:10]
    
    def _detect_domain_knowledge_gaps(
        self,
        context: Dict[str, Any],
        knowledge_base: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect missing knowledge within the current domain."""
        gaps = []
        current_domain = context.get("domain", "general")
        
        # Expected knowledge areas for the domain
        domain_expectations = {
            "archaeology": [
                "cultural_context", "historical_period", "geographical_location",
                "artifact_types", "preservation_methods", "cultural_significance"
            ],
            "heritage_preservation": [
                "conservation_techniques", "documentation_methods", "risk_assessment",
                "community_involvement", "legal_frameworks", "funding_sources"
            ],
            "general": [
                "basic_concepts", "relationships", "applications", "limitations"
            ]
        }
        
        expected_areas = domain_expectations.get(current_domain, domain_expectations["general"])
        domain_knowledge = knowledge_base.get(current_domain, {})
        
        for area in expected_areas:
            if area not in domain_knowledge or not domain_knowledge[area]:
                gaps.append({
                    "gap_type": "missing_domain_knowledge",
                    "domain": current_domain,
                    "area": area,
                    "description": f"Missing knowledge in {area} for {current_domain}",
                    "importance": self._calculate_domain_importance(area, current_domain),
                    "explorability": self._calculate_explorability(area, context)
                })
        
        return gaps
    
    def _detect_knowledge_inconsistencies(self, knowledge_base: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect inconsistencies or contradictions in the knowledge base."""
        gaps = []
        
        # Look for contradictory facts
        for domain, domain_data in knowledge_base.items():
            if not isinstance(domain_data, dict):
                continue
                
            facts = domain_data.get("facts", [])
            for i, fact1 in enumerate(facts):
                for j, fact2 in enumerate(facts[i+1:], i+1):
                    if self._are_contradictory(fact1, fact2):
                        gaps.append({
                            "gap_type": "knowledge_inconsistency",
                            "domain": domain,
                            "description": f"Contradictory facts: {fact1} vs {fact2}",
                            "facts": [fact1, fact2],
                            "importance": 0.9,  # High importance for contradictions
                            "explorability": 0.8
                        })
        
        return gaps
    
    def _detect_unexplored_areas(
        self,
        context: Dict[str, Any],
        knowledge_base: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect related areas that haven't been explored."""
        gaps = []
        current_topics = context.get("topics", [])
        
        # Find related topics that could be explored
        for topic in current_topics:
            related_areas = self._find_related_areas(topic, knowledge_base)
            
            for area in related_areas:
                if not self._is_area_explored(area, knowledge_base):
                    gaps.append({
                        "gap_type": "unexplored_related_area",
                        "domain": context.get("domain", "general"),
                        "area": area,
                        "related_to": topic,
                        "description": f"Unexplored area: {area} (related to {topic})",
                        "importance": self._calculate_relatedness_importance(topic, area),
                        "explorability": 0.7
                    })
        
        return gaps
    
    def _detect_unanswered_questions(
        self,
        context: Dict[str, Any],
        knowledge_base: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect questions that remain unanswered."""
        gaps = []
        
        # Look for explicit questions in context
        questions = context.get("pending_questions", [])
        for question in questions:
            if not self._has_answer(question, knowledge_base):
                gaps.append({
                    "gap_type": "unanswered_question",
                    "domain": context.get("domain", "general"),
                    "question": question,
                    "description": f"Unanswered question: {question}",
                    "importance": self._calculate_question_importance(question),
                    "explorability": self._calculate_question_explorability(question)
                })
        
        # Generate implicit questions based on partial knowledge
        implicit_questions = self._generate_implicit_questions(knowledge_base, context)
        for question in implicit_questions:
            gaps.append({
                "gap_type": "implicit_question",
                "domain": context.get("domain", "general"),
                "question": question,
                "description": f"Implicit question: {question}",
                "importance": 0.6,
                "explorability": 0.7
            })
        
        return gaps
    
    def _detect_incomplete_chains(self, knowledge_base: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect incomplete understanding chains or causal relationships."""
        gaps = []
        
        # Look for incomplete causal chains
        for domain, domain_data in knowledge_base.items():
            if not isinstance(domain_data, dict):
                continue
            
            relationships = domain_data.get("relationships", [])
            for relationship in relationships:
                if self._is_chain_incomplete(relationship):
                    gaps.append({
                        "gap_type": "incomplete_chain",
                        "domain": domain,
                        "chain": relationship,
                        "description": f"Incomplete understanding chain: {relationship.get('description', 'unknown')}",
                        "importance": 0.7,
                        "explorability": 0.6
                    })
        
        return gaps
    
    def _score_knowledge_gaps(
        self,
        gaps: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Score knowledge gaps based on importance, explorability, and context."""
        for gap in gaps:
            importance = gap.get("importance", 0.5)
            explorability = gap.get("explorability", 0.5)
            
            # Context relevance factor
            relevance = self._calculate_context_relevance(gap, context)
            
            # Novelty factor
            novelty = self._calculate_gap_novelty(gap)
            
            # Resource availability factor
            resources = self._calculate_resource_availability(gap)
            
            # Calculate total score
            total_score = (
                0.3 * importance +
                0.25 * explorability +
                0.2 * relevance +
                0.15 * novelty +
                0.1 * resources
            )
            
            gap["total_score"] = total_score
            gap["relevance"] = relevance
            gap["novelty"] = novelty
            gap["resource_availability"] = resources
        
        return gaps
    
    # Helper methods for gap detection
    def _calculate_domain_importance(self, area: str, domain: str) -> float:
        """Calculate importance of a knowledge area within a domain."""
        # Core areas are more important
        core_areas = {
            "archaeology": ["cultural_context", "artifact_types", "cultural_significance"],
            "heritage_preservation": ["conservation_techniques", "documentation_methods"]
        }
        
        if domain in core_areas and area in core_areas[domain]:
            return 0.9
        return 0.6
    
    def _calculate_explorability(self, area: str, context: Dict[str, Any]) -> float:
        """Calculate how easily an area can be explored given current context."""
        available_resources = context.get("available_resources", {})
        required_resources = self._get_required_resources(area)
        
        satisfaction = 0.0
        if required_resources:
            satisfied = sum(1 for res in required_resources if res in available_resources)
            satisfaction = satisfied / len(required_resources)
        else:
            satisfaction = 0.8  # Default if no specific requirements
        
        return min(1.0, satisfaction + 0.2)  # Add baseline explorability
    
    def _are_contradictory(self, fact1: Dict[str, Any], fact2: Dict[str, Any]) -> bool:
        """Check if two facts are contradictory."""
        if not isinstance(fact1, dict) or not isinstance(fact2, dict):
            return False
        
        # Simple contradiction check based on opposing values for same property
        for key in fact1.keys():
            if key in fact2:
                val1, val2 = fact1[key], fact2[key]
                if isinstance(val1, bool) and isinstance(val2, bool) and val1 != val2:
                    return True
                if isinstance(val1, str) and isinstance(val2, str):
                    opposite_pairs = [("positive", "negative"), ("true", "false"), ("yes", "no")]
                    for pos, neg in opposite_pairs:
                        if (pos in val1.lower() and neg in val2.lower()) or \
                           (neg in val1.lower() and pos in val2.lower()):
                            return True
        
        return False
    
    def _find_related_areas(self, topic: str, knowledge_base: Dict[str, Any]) -> List[str]:
        """Find areas related to a given topic."""
        related = []
        
        # Simple semantic similarity based on common words
        topic_words = set(topic.lower().split())
        
        for domain, domain_data in knowledge_base.items():
            if isinstance(domain_data, dict):
                for area in domain_data.keys():
                    area_words = set(area.lower().split())
                    # If areas share words, they're considered related
                    if topic_words & area_words:
                        related.append(area)
        
        return related
    
    def _is_area_explored(self, area: str, knowledge_base: Dict[str, Any]) -> bool:
        """Check if an area has been sufficiently explored."""
        for domain_data in knowledge_base.values():
            if isinstance(domain_data, dict) and area in domain_data:
                area_data = domain_data[area]
                # Consider explored if it has substantial content
                if isinstance(area_data, dict) and len(area_data) > 2:
                    return True
                elif isinstance(area_data, list) and len(area_data) > 3:
                    return True
        return False
    
    def _calculate_relatedness_importance(self, topic: str, area: str) -> float:
        """Calculate importance based on how related two topics are."""
        topic_words = set(topic.lower().split())
        area_words = set(area.lower().split())
        
        if not topic_words or not area_words:
            return 0.3
        
        # Jaccard similarity
        intersection = len(topic_words & area_words)
        union = len(topic_words | area_words)
        similarity = intersection / union if union > 0 else 0
        
        return 0.4 + similarity * 0.5  # Scale to 0.4-0.9 range
    
    def _has_answer(self, question: str, knowledge_base: Dict[str, Any]) -> bool:
        """Check if a question has an answer in the knowledge base."""
        question_words = set(question.lower().split())
        
        for domain_data in knowledge_base.values():
            if isinstance(domain_data, dict):
                answers = domain_data.get("answers", [])
                for answer in answers:
                    if isinstance(answer, dict):
                        answer_question = answer.get("question", "")
                        answer_words = set(answer_question.lower().split())
                        # Simple overlap check
                        if len(question_words & answer_words) > len(question_words) * 0.5:
                            return True
        return False
    
    def _calculate_question_importance(self, question: str) -> float:
        """Calculate importance of a question."""
        # Questions with certain keywords are more important
        high_importance_words = ["why", "how", "what causes", "significance", "meaning"]
        question_lower = question.lower()
        
        importance = 0.5  # Base importance
        for word in high_importance_words:
            if word in question_lower:
                importance += 0.2
        
        return min(1.0, importance)
    
    def _calculate_question_explorability(self, question: str) -> float:
        """Calculate how explorable a question is."""
        # Questions about observable phenomena are more explorable
        explorable_indicators = ["what", "where", "when", "which", "describe"]
        less_explorable = ["why", "should", "ought", "value"]
        
        question_lower = question.lower()
        score = 0.5
        
        for indicator in explorable_indicators:
            if indicator in question_lower:
                score += 0.2
        
        for indicator in less_explorable:
            if indicator in question_lower:
                score -= 0.1
        
        return max(0.1, min(1.0, score))
    
    def _generate_implicit_questions(
        self,
        knowledge_base: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate implicit questions based on partial knowledge."""
        questions = []
        
        # Generate questions about missing causal links
        for domain_data in knowledge_base.values():
            if isinstance(domain_data, dict):
                facts = domain_data.get("facts", [])
                for fact in facts:
                    if isinstance(fact, dict):
                        # Generate "why" questions for facts without explanations
                        if "explanation" not in fact:
                            description = fact.get("description", "this phenomenon")
                            questions.append(f"Why does {description} occur?")
                        
                        # Generate "how" questions for processes without mechanisms
                        if "mechanism" not in fact and "process" in str(fact):
                            description = fact.get("description", "this process")
                            questions.append(f"How does {description} work?")
        
        return questions[:5]  # Limit to 5 implicit questions
    
    def _is_chain_incomplete(self, relationship: Dict[str, Any]) -> bool:
        """Check if a causal or logical chain is incomplete."""
        if not isinstance(relationship, dict):
            return False
        
        # Check for missing links in causal chains
        chain = relationship.get("chain", [])
        if len(chain) < 2:
            return True
        
        # Check for gaps in the chain (missing intermediate steps)
        for i in range(len(chain) - 1):
            current = chain[i]
            next_item = chain[i + 1]
            
            # If there's a large conceptual gap, consider it incomplete
            if self._has_conceptual_gap(current, next_item):
                return True
        
        return False
    
    def _has_conceptual_gap(self, item1: Any, item2: Any) -> bool:
        """Check if there's a conceptual gap between two items in a chain."""
        # Simple heuristic: if items are very different, there might be a gap
        if isinstance(item1, str) and isinstance(item2, str):
            words1 = set(item1.lower().split())
            words2 = set(item2.lower().split())
            
            # If no common words, might indicate a gap
            return len(words1 & words2) == 0
        
        return False
    
    def _calculate_context_relevance(self, gap: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate how relevant a gap is to the current context."""
        gap_domain = gap.get("domain", "")
        context_domain = context.get("domain", "")
        
        # Domain match increases relevance
        if gap_domain == context_domain:
            return 0.9
        
        # Related domains have moderate relevance
        related_domains = {
            "archaeology": ["heritage_preservation", "cultural_studies"],
            "heritage_preservation": ["archaeology", "conservation"]
        }
        
        if context_domain in related_domains.get(gap_domain, []):
            return 0.7
        
        return 0.5  # Default relevance
    
    def _calculate_gap_novelty(self, gap: Dict[str, Any]) -> float:
        """Calculate novelty of a knowledge gap."""
        gap_type = gap.get("gap_type", "")
        
        # Some gap types are more novel/interesting
        novelty_scores = {
            "knowledge_inconsistency": 0.9,
            "implicit_question": 0.8,
            "incomplete_chain": 0.7,
            "unexplored_related_area": 0.6,
            "missing_domain_knowledge": 0.5,
            "unanswered_question": 0.6
        }
        
        return novelty_scores.get(gap_type, 0.5)
    
    def _calculate_resource_availability(self, gap: Dict[str, Any]) -> float:
        """Calculate availability of resources to address this gap."""
        # Simple heuristic based on gap type
        explorable_types = {
            "missing_domain_knowledge": 0.8,
            "unanswered_question": 0.7,
            "unexplored_related_area": 0.6,
            "implicit_question": 0.5,
            "incomplete_chain": 0.4,
            "knowledge_inconsistency": 0.3
        }
        
        gap_type = gap.get("gap_type", "")
        return explorable_types.get(gap_type, 0.5)
    
    def _get_required_resources(self, area: str) -> List[str]:
        """Get required resources for exploring an area."""
        # Domain-specific resource requirements
        resource_map = {
            "cultural_context": ["historical_databases", "expert_consultation"],
            "artifact_types": ["archaeological_databases", "image_analysis"],
            "preservation_methods": ["technical_literature", "conservation_experts"],
            "documentation_methods": ["methodology_guides", "software_tools"]
        }
        
        return resource_map.get(area, ["general_research_tools"])
    
    def assess_novelty(
        self,
        item: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Assess the novelty of an item or concept.
        
        Args:
            item: Item to assess for novelty
            context: Contextual information
            
        Returns:
            Novelty score (0.0 to 1.0)
        """
        self.logger.debug(f"Assessing novelty for: {item.get('name', 'unknown')}")
        
        # 1. Similarity to known items
        similarity_score = self._calculate_similarity_to_known(item, context)
        
        # 2. Frequency of occurrence
        frequency_score = self._calculate_frequency_novelty(item, context)
        
        # 3. Unexpectedness based on context
        unexpectedness_score = self._calculate_unexpectedness(item, context)
        
        # 4. Surprise factor based on predictions
        surprise_score = self._calculate_surprise_factor(item, context)
        
        # 5. Semantic distance from known concepts
        semantic_distance = self._calculate_semantic_distance(item, context)
        
        # Combine scores with weights
        novelty_score = (
            0.25 * (1.0 - similarity_score) +    # Less similar = more novel
            0.2 * frequency_score +              # Less frequent = more novel
            0.2 * unexpectedness_score +         # More unexpected = more novel
            0.2 * surprise_score +               # More surprising = more novel
            0.15 * semantic_distance             # More distant = more novel
        )
        
        return max(0.0, min(1.0, novelty_score))
    
    def _calculate_similarity_to_known(self, item: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate similarity to known items in the knowledge base."""
        known_items = context.get("known_items", [])
        if not known_items:
            return 0.0  # No known items means high novelty
        
        item_features = self._extract_features(item)
        max_similarity = 0.0
        
        for known_item in known_items:
            known_features = self._extract_features(known_item)
            similarity = self._calculate_feature_similarity(item_features, known_features)
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _calculate_frequency_novelty(self, item: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate novelty based on frequency of occurrence."""
        item_type = item.get("type", "unknown")
        occurrence_history = context.get("occurrence_history", {})
        
        frequency = occurrence_history.get(item_type, 0)
        total_occurrences = sum(occurrence_history.values()) if occurrence_history else 1
        
        if total_occurrences == 0:
            return 1.0  # Never seen before
        
        relative_frequency = frequency / total_occurrences
        
        # Higher frequency = lower novelty
        return max(0.0, 1.0 - relative_frequency * 2)  # Scale so 0.5 frequency = 0 novelty
    
    def _calculate_unexpectedness(self, item: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate unexpectedness based on context expectations."""
        expected_items = context.get("expected_items", [])
        current_domain = context.get("domain", "general")
        
        # Check if item matches expectations
        item_type = item.get("type", "unknown")
        item_category = item.get("category", "unknown")
        
        expectedness = 0.0
        for expected in expected_items:
            if expected.get("type") == item_type:
                expectedness += 0.5
            if expected.get("category") == item_category:
                expectedness += 0.3
            if expected.get("domain") == current_domain:
                expectedness += 0.2
        
        return max(0.0, 1.0 - min(1.0, expectedness))
    
    def _calculate_surprise_factor(self, item: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate surprise factor based on violated predictions."""
        predictions = context.get("predictions", [])
        
        surprise = 0.0
        for prediction in predictions:
            if self._violates_prediction(item, prediction):
                confidence = prediction.get("confidence", 0.5)
                surprise += confidence  # Higher confidence predictions = more surprise when violated
        
        return min(1.0, surprise)
    
    def _calculate_semantic_distance(self, item: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate semantic distance from known concepts."""
        item_keywords = self._extract_keywords(item)
        known_keywords = set()
        
        # Collect keywords from known concepts
        for known_item in context.get("known_items", []):
            known_keywords.update(self._extract_keywords(known_item))
        
        if not known_keywords:
            return 1.0  # Maximum distance if no known concepts
        
        # Calculate semantic overlap
        overlap = len(item_keywords & known_keywords)
        total_unique = len(item_keywords | known_keywords)
        
        if total_unique == 0:
            return 0.0
        
        # Distance = 1 - similarity (Jaccard index)
        similarity = overlap / total_unique
        return 1.0 - similarity
    
    def _extract_features(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant features from an item for comparison."""
        features = {}
        
        # Basic properties
        for key in ["type", "category", "size", "color", "material", "age", "origin"]:
            if key in item:
                features[key] = item[key]
        
        # Textual features (keywords from description)
        description = item.get("description", "")
        if description:
            features["keywords"] = set(description.lower().split())
        
        # Numerical features
        for key in ["value", "weight", "complexity", "importance"]:
            if key in item and isinstance(item[key], (int, float)):
                features[key] = item[key]
        
        return features
    
    def _calculate_feature_similarity(
        self,
        features1: Dict[str, Any],
        features2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two feature sets."""
        if not features1 or not features2:
            return 0.0
        
        total_similarity = 0.0
        comparison_count = 0
        
        # Compare categorical features
        for key in ["type", "category", "material", "origin"]:
            if key in features1 and key in features2:
                if features1[key] == features2[key]:
                    total_similarity += 1.0
                comparison_count += 1
        
        # Compare numerical features
        for key in ["value", "weight", "complexity", "importance"]:
            if key in features1 and key in features2:
                val1, val2 = features1[key], features2[key]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Normalized difference (0 = identical, 1 = very different)
                    max_val = max(abs(val1), abs(val2))
                    if max_val > 0:
                        diff = abs(val1 - val2) / max_val
                        total_similarity += 1.0 - min(1.0, diff)
                    else:
                        total_similarity += 1.0
                    comparison_count += 1
        
        # Compare keyword sets
        if "keywords" in features1 and "keywords" in features2:
            kw1, kw2 = features1["keywords"], features2["keywords"]
            if kw1 or kw2:
                intersection = len(kw1 & kw2)
                union = len(kw1 | kw2)
                keyword_similarity = intersection / union if union > 0 else 0
                total_similarity += keyword_similarity
                comparison_count += 1
        
        return total_similarity / comparison_count if comparison_count > 0 else 0.0
    
    def _extract_keywords(self, item: Dict[str, Any]) -> Set[str]:
        """Extract keywords from an item."""
        keywords = set()
        
        # From description
        description = item.get("description", "")
        if description:
            # Simple tokenization and filtering
            words = description.lower().split()
            # Filter out common words
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
            keywords.update(word for word in words if word not in stop_words and len(word) > 2)
        
        # From categorical fields
        for key in ["type", "category", "material", "origin", "purpose"]:
            value = item.get(key)
            if isinstance(value, str):
                keywords.update(value.lower().split())
        
        return keywords
    
    def _violates_prediction(self, item: Dict[str, Any], prediction: Dict[str, Any]) -> bool:
        """Check if an item violates a prediction."""
        predicted_type = prediction.get("predicted_type")
        predicted_category = prediction.get("predicted_category")
        predicted_properties = prediction.get("predicted_properties", {})
        
        # Check type prediction
        if predicted_type and item.get("type") != predicted_type:
            return True
        
        # Check category prediction
        if predicted_category and item.get("category") != predicted_category:
            return True
        
        # Check property predictions
        for prop, predicted_value in predicted_properties.items():
            if prop in item and item[prop] != predicted_value:
                return True
        
        return False
    
    def generate_curiosity_signal(
        self,
        trigger: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[CuriositySignal]:
        """Generate a curiosity signal based on a trigger.
        
        Args:
            trigger: Event or condition that triggers curiosity
            context: Current context
            
        Returns:
            Generated curiosity signal or None
        """
        trigger_type = trigger.get("type", "unknown")
        self.logger.info(f"Generating curiosity signal for trigger: {trigger_type}")
        
        # Determine curiosity type based on trigger
        curiosity_type = self._determine_curiosity_type(trigger, context)
        
        # Calculate intensity based on various factors
        intensity = self._calculate_curiosity_intensity(trigger, context)
        
        if intensity < 0.3:  # Below threshold
            self.logger.debug(f"Curiosity intensity {intensity:.2f} below threshold, no signal generated")
            return None
        
        # Generate focus area and specific questions
        focus_area = self._determine_focus_area(trigger, context)
        specific_questions = self._generate_specific_questions(trigger, context, curiosity_type)
        
        # Calculate decay rate based on curiosity type and context
        decay_rate = self._calculate_decay_rate(curiosity_type, context)
        
        # Calculate satisfaction threshold
        satisfaction_threshold = self._calculate_satisfaction_threshold(curiosity_type, intensity)
        
        # Create unique signal ID
        signal_id = f"curiosity_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Create the curiosity signal
        signal = CuriositySignal(
            signal_id=signal_id,
            curiosity_type=curiosity_type,
            intensity=intensity,
            focus_area=focus_area,
            specific_questions=specific_questions,
            context=context.copy(),
            timestamp=time.time(),
            decay_rate=decay_rate,
            satisfaction_threshold=satisfaction_threshold
        )
        
        # Store the signal
        self.active_curiosity_signals[signal_id] = signal
        
        self.logger.info(f"Generated curiosity signal: {curiosity_type.value} (intensity: {intensity:.2f})")
        
        return signal
    
    def _determine_curiosity_type(self, trigger: Dict[str, Any], context: Dict[str, Any]) -> CuriosityType:
        """Determine the type of curiosity based on trigger and context."""
        trigger_type = trigger.get("type", "unknown")
        
        # Map trigger types to curiosity types
        type_mapping = {
            "knowledge_gap": CuriosityType.EPISTEMIC,
            "novel_item": CuriosityType.DIVERSIVE,
            "specific_question": CuriosityType.SPECIFIC,
            "inconsistency": CuriosityType.EPISTEMIC,
            "sensory_input": CuriosityType.PERCEPTUAL,
            "other_perspective": CuriosityType.EMPATHIC,
            "creative_opportunity": CuriosityType.CREATIVE,
            "unexplored_area": CuriosityType.DIVERSIVE,
            "incomplete_understanding": CuriosityType.EPISTEMIC
        }
        
        return type_mapping.get(trigger_type, CuriosityType.EPISTEMIC)
    
    def _determine_focus_area(self, trigger: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Determine the focus area for curiosity exploration."""
        # Use trigger information to determine focus
        if "area" in trigger:
            return trigger["area"]
        elif "domain" in trigger:
            return trigger["domain"]
        elif "topic" in trigger:
            return trigger["topic"]
        else:
            # Derive from context
            return context.get("domain", "general_exploration")
    
    def _generate_specific_questions(
        self,
        trigger: Dict[str, Any],
        context: Dict[str, Any],
        curiosity_type: CuriosityType
    ) -> List[str]:
        """Generate specific questions based on trigger and curiosity type."""
        questions = []
        
        if curiosity_type == CuriosityType.EPISTEMIC:
            questions.extend(self._generate_epistemic_questions(trigger, context))
        elif curiosity_type == CuriosityType.DIVERSIVE:
            questions.extend(self._generate_diversive_questions(trigger, context))
        elif curiosity_type == CuriosityType.SPECIFIC:
            questions.extend(self._generate_specific_type_questions(trigger, context))
        elif curiosity_type == CuriosityType.PERCEPTUAL:
            questions.extend(self._generate_perceptual_questions(trigger, context))
        elif curiosity_type == CuriosityType.EMPATHIC:
            questions.extend(self._generate_empathic_questions(trigger, context))
        elif curiosity_type == CuriosityType.CREATIVE:
            questions.extend(self._generate_creative_questions(trigger, context))
        
        return questions[:5]  # Limit to 5 questions
    
    def _generate_epistemic_questions(self, trigger: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate knowledge-seeking questions."""
        questions = []
        
        if trigger.get("type") == "knowledge_gap":
            area = trigger.get("area", "this area")
            questions.extend([
                f"What is the fundamental nature of {area}?",
                f"How does {area} relate to other known concepts?",
                f"What are the underlying principles governing {area}?",
                f"What evidence supports current understanding of {area}?"
            ])
        elif trigger.get("type") == "inconsistency":
            facts = trigger.get("facts", ["these facts"])
            questions.extend([
                f"Why do these facts appear contradictory?",
                f"Which of these facts is more reliable?",
                f"What additional information could resolve this inconsistency?",
                f"Are there hidden assumptions causing this contradiction?"
            ])
        
        return questions
    
    def _generate_diversive_questions(self, trigger: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate novelty-seeking questions."""
        questions = []
        
        if trigger.get("type") == "novel_item":
            item_name = trigger.get("item", {}).get("name", "this item")
            questions.extend([
                f"What other similar items might exist?",
                f"Where else might we find {item_name}?",
                f"What variations of {item_name} are possible?",
                f"What new applications could {item_name} have?"
            ])
        elif trigger.get("type") == "unexplored_area":
            area = trigger.get("area", "this area")
            questions.extend([
                f"What interesting phenomena might exist in {area}?",
                f"What surprises might {area} hold?",
                f"How might {area} differ from what we expect?",
                f"What connections might {area} have to other domains?"
            ])
        
        return questions
    
    def _generate_specific_type_questions(self, trigger: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate specific targeted questions."""
        questions = []
        
        if "question" in trigger:
            original_question = trigger["question"]
            questions.extend([
                original_question,
                f"What are the implications of answering: {original_question}?",
                f"What would need to be true for: {original_question}?",
                f"How could we test: {original_question}?"
            ])
        
        return questions
    
    def _generate_perceptual_questions(self, trigger: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate sensory exploration questions."""
        questions = []
        
        sensory_input = trigger.get("sensory_input", "this phenomenon")
        questions.extend([
            f"What visual details can we observe about {sensory_input}?",
            f"How does {sensory_input} compare to similar phenomena?",
            f"What patterns are visible in {sensory_input}?",
            f"What changes can we detect in {sensory_input} over time?"
        ])
        
        return questions
    
    def _generate_empathic_questions(self, trigger: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate perspective-taking questions."""
        questions = []
        
        perspective = trigger.get("perspective", "others")
        questions.extend([
            f"How might {perspective} view this situation?",
            f"What motivations might {perspective} have?",
            f"What concerns might {perspective} express?",
            f"How might {perspective} approach this problem differently?"
        ])
        
        return questions
    
    def _generate_creative_questions(self, trigger: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate creative exploration questions."""
        questions = []
        
        opportunity = trigger.get("opportunity", "this situation")
        questions.extend([
            f"What creative possibilities does {opportunity} offer?",
            f"How might we reimagine {opportunity}?",
            f"What unconventional approaches could we try with {opportunity}?",
            f"What would happen if we combined {opportunity} with other ideas?"
        ])
        
        return questions
    
    def _calculate_decay_rate(self, curiosity_type: CuriosityType, context: Dict[str, Any]) -> float:
        """Calculate how quickly curiosity should decay over time."""
        # Different types of curiosity have different decay rates
        base_decay_rates = {
            CuriosityType.EPISTEMIC: 0.05,      # Slow decay - knowledge seeking persists
            CuriosityType.DIVERSIVE: 0.15,      # Faster decay - novelty seeking is ephemeral
            CuriosityType.SPECIFIC: 0.08,       # Medium decay - specific questions persist moderately
            CuriosityType.PERCEPTUAL: 0.20,     # Fast decay - sensory curiosity is immediate
            CuriosityType.EMPATHIC: 0.10,       # Medium decay - understanding others persists
            CuriosityType.CREATIVE: 0.12        # Medium-fast decay - creative impulses are dynamic
        }
        
        base_rate = base_decay_rates.get(curiosity_type, 0.10)
        
        # Adjust based on context
        emotional_state = context.get("emotional_state", {})
        urgency = emotional_state.get("urgency", 0.5)
        
        # Higher urgency increases decay rate (less time for extended curiosity)
        adjusted_rate = base_rate * (1.0 + urgency * 0.5)
        
        return min(0.5, adjusted_rate)  # Cap at 0.5 per time unit
    
    def _calculate_satisfaction_threshold(self, curiosity_type: CuriosityType, intensity: float) -> float:
        """Calculate the threshold for curiosity satisfaction."""
        # Base thresholds for different curiosity types
        base_thresholds = {
            CuriosityType.EPISTEMIC: 0.8,       # High threshold - needs substantial knowledge
            CuriosityType.DIVERSIVE: 0.6,       # Medium threshold - satisfied by exploration
            CuriosityType.SPECIFIC: 0.9,        # Very high threshold - needs specific answer
            CuriosityType.PERCEPTUAL: 0.5,      # Lower threshold - satisfied by observation
            CuriosityType.EMPATHIC: 0.7,        # High threshold - needs deep understanding
            CuriosityType.CREATIVE: 0.6         # Medium threshold - satisfied by creation
        }
        
        base_threshold = base_thresholds.get(curiosity_type, 0.7)
        
        # Adjust based on intensity - higher intensity curiosity is harder to satisfy
        adjusted_threshold = base_threshold + (intensity - 0.5) * 0.2
        
        return max(0.3, min(1.0, adjusted_threshold))
    
    def identify_exploration_targets(
        self,
        curiosity_signal: CuriositySignal,
        available_resources: Dict[str, Any]
    ) -> List[ExplorationTarget]:
        """Identify potential targets for exploration based on curiosity.
        
        Args:
            curiosity_signal: Active curiosity signal
            available_resources: Available resources for exploration
            
        Returns:
            List of potential exploration targets
        """
        # TODO: Implement target identification
        # Should identify:
        # - Relevant domains to explore
        # - Specific concepts or items to investigate
        # - Learning pathways and sequences
        # - Resource-efficient exploration options
        # - High-value exploration opportunities
        
        self.logger.info(f"Identifying exploration targets for {curiosity_signal.focus_area}")
        
        # Placeholder implementation
        targets = [
            ExplorationTarget(
                target_id=f"target_{int(time.time())}_1",
                domain=curiosity_signal.focus_area,
                description=f"Explore {curiosity_signal.focus_area} fundamentals",
                novelty_score=0.8,
                complexity_score=0.6,
                potential_value=0.9,
                exploration_cost=0.5,
                prerequisites=[]
            ),
            ExplorationTarget(
                target_id=f"target_{int(time.time())}_2",
                domain=curiosity_signal.focus_area,
                description=f"Deep dive into {curiosity_signal.focus_area} advanced concepts",
                novelty_score=0.9,
                complexity_score=0.8,
                potential_value=0.8,
                exploration_cost=0.7,
                prerequisites=["fundamentals"]
            )
        ]
        
        return targets
    
    def prioritize_exploration(
        self,
        targets: List[ExplorationTarget],
        current_goals: List[Dict[str, Any]],
        resources: Dict[str, Any]
    ) -> List[ExplorationTarget]:
        """Prioritize exploration targets based on multiple factors.
        
        Args:
            targets: Available exploration targets
            current_goals: Current system goals
            resources: Available resources
            
        Returns:
            Prioritized list of exploration targets
        """
        # TODO: Implement sophisticated prioritization
        # Should consider:
        # - Curiosity intensity and type
        # - Resource availability
        # - Goal alignment
        # - Potential learning value
        # - Prerequisites and dependencies
        # - Risk vs. reward
        
        self.logger.info(f"Prioritizing {len(targets)} exploration targets")
        
        # Placeholder implementation
        # In practice, this would use multi-criteria decision making
        def priority_score(target: ExplorationTarget) -> float:
            return (
                target.novelty_score * 0.3 +
                target.potential_value * 0.4 +
                (1.0 - target.exploration_cost) * 0.2 +
                target.complexity_score * self.complexity_preference * 0.1
            )
        
        return sorted(targets, key=priority_score, reverse=True)
    
    def track_exploration_outcome(
        self,
        target_id: str,
        outcome: Dict[str, Any],
        satisfaction_level: float
    ) -> None:
        """Track the outcome of an exploration activity.
        
        Args:
            target_id: ID of the exploration target
            outcome: Results of the exploration
            satisfaction_level: How well the exploration satisfied curiosity
        """
        # TODO: Implement outcome tracking
        # Should track:
        # - Knowledge gained
        # - Curiosity satisfaction
        # - Resource usage
        # - Unexpected discoveries
        # - Learning efficiency
        
        self.logger.info(f"Tracking exploration outcome for target: {target_id}")
        
        # Update satisfaction history
        self.curiosity_satisfaction_history.append({
            "target_id": target_id,
            "satisfaction": satisfaction_level,
            "outcome": outcome,
            "timestamp": time.time()
        })
        
        # Update knowledge map
        if "knowledge_gained" in outcome:
            self._update_knowledge_map(outcome["knowledge_gained"])
        
        # Update curiosity signals based on satisfaction
        self._update_curiosity_satisfaction(satisfaction_level)
    
    def decay_curiosity_signals(self) -> None:
        """Apply decay to active curiosity signals over time."""
        # TODO: Implement sophisticated decay
        # Should consider:
        # - Time-based decay
        # - Satisfaction-based decay
        # - Context changes
        # - Priority shifts
        
        current_time = time.time()
        signals_to_remove = []
        
        for signal_id, signal in self.active_curiosity_signals.items():
            # Apply time-based decay
            time_elapsed = current_time - signal.timestamp
            decay_factor = math.exp(-signal.decay_rate * time_elapsed)
            signal.intensity *= decay_factor
            
            # Remove signals below threshold
            if signal.intensity < 0.1:
                signals_to_remove.append(signal_id)
        
        # Remove decayed signals
        for signal_id in signals_to_remove:
            del self.active_curiosity_signals[signal_id]
    
    def get_current_curiosity_state(self) -> Dict[str, Any]:
        """Get current state of curiosity and exploration.
        
        Returns:
            Current curiosity state summary
        """
        # TODO: Implement comprehensive state summary
        # Should include:
        # - Active curiosity signals
        # - Knowledge growth metrics
        # - Exploration effectiveness
        # - Resource utilization
        # - Satisfaction trends
        
        self.decay_curiosity_signals()  # Update state first
        
        return {
            "active_signals": len(self.active_curiosity_signals),
            "average_intensity": self._calculate_average_intensity(),
            "dominant_curiosity_type": self._get_dominant_curiosity_type(),
            "knowledge_growth_rate": self.knowledge_growth_rate,
            "exploration_effectiveness": self._calculate_exploration_effectiveness(),
            "resource_utilization": self._calculate_resource_utilization()
        }
    
    def suggest_curiosity_driven_goals(
        self,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Suggest goals driven by current curiosity state.
        
        Args:
            context: Current context
            
        Returns:
            List of curiosity-driven goal suggestions
        """
        # TODO: Implement goal suggestion
        # Should suggest:
        # - Exploration goals based on active curiosity
        # - Learning goals for knowledge gaps
        # - Investigation goals for novel items
        # - Creative goals for self-expression
        
        self.logger.info("Generating curiosity-driven goal suggestions")
        
        suggestions = []
        
        for signal in self.active_curiosity_signals.values():
            if signal.intensity > 0.5:  # Only strong curiosity signals
                goal_suggestion = {
                    "goal_type": "exploration",
                    "curiosity_type": signal.curiosity_type.value,
                    "description": f"Explore {signal.focus_area} to satisfy curiosity",
                    "priority": signal.intensity,
                    "focus_area": signal.focus_area,
                    "specific_questions": signal.specific_questions,
                    "estimated_satisfaction": 0.8
                }
                suggestions.append(goal_suggestion)
        
        return suggestions
    
    def _calculate_curiosity_intensity(
        self,
        trigger: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Calculate curiosity intensity based on trigger and context.
        
        Args:
            trigger: Curiosity trigger
            context: Current context
            
        Returns:
            Calculated intensity (0.0 to 1.0)
        """
        # TODO: Implement sophisticated intensity calculation
        # Should consider:
        # - Novelty of trigger
        # - Personal relevance
        # - Knowledge gap size
        # - Emotional state
        # - Available resources
        
        base_intensity = self.base_curiosity_level
        novelty_factor = trigger.get("novelty", 0.5)
        importance_factor = trigger.get("importance", 0.5)
        
        intensity = base_intensity * (0.5 + 0.3 * novelty_factor + 0.2 * importance_factor)
        return min(intensity, 1.0)
    
    def _calculate_average_intensity(self) -> float:
        """Calculate average intensity of active curiosity signals."""
        if not self.active_curiosity_signals:
            return 0.0
        
        total_intensity = sum(
            signal.intensity for signal in self.active_curiosity_signals.values()
        )
        return total_intensity / len(self.active_curiosity_signals)
    
    def _get_dominant_curiosity_type(self) -> str:
        """Get the dominant curiosity type among active signals."""
        if not self.active_curiosity_signals:
            return "none"
        
        type_counts = {}
        for signal in self.active_curiosity_signals.values():
            curiosity_type = signal.curiosity_type.value
            type_counts[curiosity_type] = type_counts.get(curiosity_type, 0) + signal.intensity
        
        return max(type_counts, key=type_counts.get) if type_counts else "none"
    
    def _calculate_exploration_effectiveness(self) -> float:
        """Calculate effectiveness of recent explorations."""
        if not self.curiosity_satisfaction_history:
            return 0.5  # Default neutral value
        
        recent_history = self.curiosity_satisfaction_history[-10:]  # Last 10 explorations
        if not recent_history:
            return 0.5
        
        average_satisfaction = sum(
            entry["satisfaction"] for entry in recent_history
        ) / len(recent_history)
        
        return average_satisfaction
    
    def _calculate_resource_utilization(self) -> float:
        """Calculate current resource utilization for curiosity-driven activities."""
        # TODO: Implement actual resource tracking
        # For now, return a placeholder value
        return 0.6
    
    def _update_knowledge_map(self, new_knowledge: Dict[str, Any]) -> None:
        """Update the internal knowledge map with new knowledge.
        
        Args:
            new_knowledge: New knowledge to integrate
        """
        # TODO: Implement sophisticated knowledge integration
        # Should:
        # - Update knowledge graph
        # - Identify new connections
        # - Calculate knowledge growth
        # - Update understanding metrics
        
        domain = new_knowledge.get("domain", "general")
        if domain not in self.knowledge_map:
            self.knowledge_map[domain] = {}
        
        # Simple integration (placeholder)
        concepts = new_knowledge.get("concepts", [])
        for concept in concepts:
            self.knowledge_map[domain][concept] = new_knowledge.get("details", {})
    
    def _update_curiosity_satisfaction(self, satisfaction_level: float) -> None:
        """Update curiosity satisfaction metrics.
        
        Args:
            satisfaction_level: Level of satisfaction achieved
        """
        # TODO: Implement satisfaction-based curiosity adjustment
        # Should adjust:
        # - Future curiosity thresholds
        # - Exploration strategies
        # - Resource allocation
        # - Signal generation sensitivity
        
        # Simple adjustment (placeholder)
        if satisfaction_level > 0.8:
            self.base_curiosity_level = min(self.base_curiosity_level * 1.05, 1.0)
        elif satisfaction_level < 0.4:
            self.base_curiosity_level = max(self.base_curiosity_level * 0.95, 0.1) 
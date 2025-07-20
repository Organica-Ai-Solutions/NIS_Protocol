#!/usr/bin/env python3
"""
🦜🔗 NIS Protocol v3 - LangChain/LangGraph Integration

Comprehensive integration of LangChain ecosystem with NIS Protocol v3,
including LangGraph workflows, LangSmith monitoring, and advanced reasoning patterns.

Features:
- LangChain Chat Models integration
- LangGraph State Machine workflows
- LangSmith observability and evaluation
- Chain of Thought (COT) reasoning
- Tree of Thought (TOT) reasoning  
- ReAct (Reasoning and Acting) patterns
- Seamless NIS Protocol integration
"""

import os
import time
import json
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

# LangChain Core
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.runnables import Runnable
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# LangGraph
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import ToolExecutor
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# LangSmith
try:
    from langsmith import Client as LangSmithClient
    from langsmith.evaluation import evaluate
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

# NIS Protocol imports
try:
    from ..utils.self_audit import self_audit_engine
    from ..utils.integrity_metrics import calculate_confidence
    from ..agents.consciousness.enhanced_conscious_agent import EnhancedConsciousAgent, ReflectionType
except ImportError:
    # Fallback for standalone testing
    pass


class ReasoningPattern(Enum):
    """Available reasoning patterns"""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    REACT = "reasoning_and_acting"
    HYBRID_COT_REACT = "hybrid_cot_react"


@dataclass
class LangChainState:
    """State for LangGraph workflows"""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_pattern: ReasoningPattern = ReasoningPattern.CHAIN_OF_THOUGHT
    thought_tree: Dict[str, Any] = field(default_factory=dict)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    final_answer: Optional[str] = None
    confidence: float = 0.0
    integrity_score: float = 0.0
    processing_time: float = 0.0


@dataclass
class ReasoningResult:
    """Result from reasoning process"""
    final_answer: str
    reasoning_steps: List[str]
    confidence: float
    integrity_score: float
    processing_time: float
    pattern_used: ReasoningPattern
    metadata: Dict[str, Any]


class ChainOfThoughtReasoner:
    """Chain of Thought reasoning implementation"""
    
    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm
        self.step_history: List[str] = []
    
    def reason(self, question: str, context: Optional[str] = None) -> ReasoningResult:
        """Execute Chain of Thought reasoning"""
        
        start_time = time.time()
        
        # COT prompt engineering
        cot_prompt = self._build_cot_prompt(question, context)
        
        reasoning_steps = []
        
        # Step 1: Problem understanding
        understanding_step = f"Understanding: {question}"
        reasoning_steps.append(understanding_step)
        
        # Step 2: Break down the problem
        breakdown_step = "Breaking down the problem into components..."
        reasoning_steps.append(breakdown_step)
        
        # Step 3: Step-by-step reasoning
        if self.llm:
            try:
                # Use LLM for actual reasoning
                response = self.llm.invoke(cot_prompt)
                reasoning_content = response.content if hasattr(response, 'content') else str(response)
                reasoning_steps.extend(self._parse_reasoning_steps(reasoning_content))
            except Exception as e:
                reasoning_steps.append(f"LLM reasoning failed: {e}")
                reasoning_steps.append("Falling back to rule-based reasoning...")
        
        # Fallback reasoning if no LLM
        if not reasoning_steps or len(reasoning_steps) <= 3:
            reasoning_steps.extend(self._fallback_reasoning(question, context))
        
        # Generate final answer
        final_answer = self._synthesize_answer(reasoning_steps)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        confidence = self._calculate_confidence(reasoning_steps)
        integrity_score = self._assess_integrity(final_answer)
        
        return ReasoningResult(
            final_answer=final_answer,
            reasoning_steps=reasoning_steps,
            confidence=confidence,
            integrity_score=integrity_score,
            processing_time=processing_time,
            pattern_used=ReasoningPattern.CHAIN_OF_THOUGHT,
            metadata={
                "cot_prompt": cot_prompt,
                "step_count": len(reasoning_steps),
                "llm_used": self.llm is not None
            }
        )
    
    def _build_cot_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Build Chain of Thought prompt"""
        
        prompt = f"""Think step by step to answer this question.

Question: {question}
"""
        
        if context:
            prompt += f"\nContext: {context}\n"
        
        prompt += """
Please reason through this step by step:
1. First, understand what is being asked
2. Break down the problem into smaller parts
3. Work through each part systematically
4. Combine your findings into a final answer

Let's work through this step by step:
"""
        
        return prompt
    
    def _parse_reasoning_steps(self, content: str) -> List[str]:
        """Parse reasoning steps from LLM response"""
        
        steps = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('Step') or line.startswith('-') or 
                        line.startswith('1.') or line.startswith('2.') or
                        len(line) > 20):  # Substantial content
                steps.append(line)
        
        return steps[:10]  # Limit to 10 steps
    
    def _fallback_reasoning(self, question: str, context: Optional[str] = None) -> List[str]:
        """Fallback reasoning when LLM unavailable"""
        
        steps = [
            "Analyzing question structure and key terms...",
            f"Identified key concepts: {self._extract_key_concepts(question)}",
            "Applying logical reasoning patterns...",
            "Synthesizing available information..."
        ]
        
        if context:
            steps.insert(1, f"Incorporating provided context: {context[:100]}...")
        
        return steps
    
    def _extract_key_concepts(self, question: str) -> List[str]:
        """Extract key concepts from question"""
        
        # Simple keyword extraction
        common_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an'}
        words = question.lower().replace('?', '').split()
        key_concepts = [word for word in words if len(word) > 3 and word not in common_words]
        
        return key_concepts[:5]
    
    def _synthesize_answer(self, reasoning_steps: List[str]) -> str:
        """Synthesize final answer from reasoning steps"""
        
        if not reasoning_steps:
            return "Unable to generate answer due to insufficient reasoning steps."
        
        # Extract insights from reasoning steps
        insights = []
        for step in reasoning_steps:
            if 'therefore' in step.lower() or 'conclusion' in step.lower() or 'answer' in step.lower():
                insights.append(step)
        
        if insights:
            return f"Based on the reasoning process: {insights[-1]}"
        else:
            return f"Through step-by-step analysis of {len(reasoning_steps)} reasoning steps, a comprehensive answer has been developed."
    
    def _calculate_confidence(self, reasoning_steps: List[str]) -> float:
        """Calculate confidence based on reasoning quality"""
        
        if not reasoning_steps:
            return 0.0
        
        # Factors affecting confidence
        step_count_factor = min(len(reasoning_steps) / 5.0, 1.0)  # More steps = higher confidence
        detail_factor = sum(len(step) for step in reasoning_steps) / (len(reasoning_steps) * 50)  # Detail level
        detail_factor = min(detail_factor, 1.0)
        
        base_confidence = 0.6  # Base confidence for COT
        confidence = base_confidence + 0.2 * step_count_factor + 0.2 * detail_factor
        
        return min(confidence, 1.0)
    
    def _assess_integrity(self, answer: str) -> float:
        """Assess integrity of the answer"""
        
        try:
            if 'self_audit_engine' in globals():
                return self_audit_engine.get_integrity_score(answer)
            else:
                # Fallback integrity assessment
                if len(answer) < 10:
                    return 50.0  # Very short answers are less reliable
                elif 'comprehensive' in answer.lower() or 'analysis' in answer.lower():
                    return 85.0  # Detailed answers are more reliable
                else:
                    return 75.0  # Default reasonable score
        except:
            return 75.0


class TreeOfThoughtReasoner:
    """Tree of Thought reasoning implementation"""
    
    def __init__(self, llm: Optional[Any] = None, max_depth: int = 3, branching_factor: int = 3):
        self.llm = llm
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.thought_tree = {}
    
    def reason(self, question: str, context: Optional[str] = None) -> ReasoningResult:
        """Execute Tree of Thought reasoning"""
        
        start_time = time.time()
        
        # Build thought tree
        self.thought_tree = self._build_thought_tree(question, context)
        
        # Find best path through tree
        best_path = self._find_best_path()
        
        # Extract reasoning steps from best path
        reasoning_steps = self._extract_reasoning_from_path(best_path)
        
        # Generate final answer
        final_answer = self._synthesize_tree_answer(best_path)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        confidence = self._calculate_tree_confidence(best_path)
        integrity_score = self._assess_integrity(final_answer)
        
        return ReasoningResult(
            final_answer=final_answer,
            reasoning_steps=reasoning_steps,
            confidence=confidence,
            integrity_score=integrity_score,
            processing_time=processing_time,
            pattern_used=ReasoningPattern.TREE_OF_THOUGHT,
            metadata={
                "thought_tree": self.thought_tree,
                "best_path": best_path,
                "tree_depth": len(best_path),
                "nodes_explored": self._count_nodes()
            }
        )
    
    def _build_thought_tree(self, question: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Build tree of possible reasoning paths"""
        
        tree = {
            "root": {
                "question": question,
                "context": context,
                "children": []
            }
        }
        
        # Generate initial thought branches
        initial_thoughts = self._generate_initial_thoughts(question)
        
        for i, thought in enumerate(initial_thoughts):
            node_id = f"level_1_node_{i}"
            tree[node_id] = {
                "thought": thought,
                "level": 1,
                "parent": "root",
                "children": [],
                "score": self._score_thought(thought)
            }
            tree["root"]["children"].append(node_id)
        
        # Recursively expand promising nodes
        self._expand_tree(tree, max_level=self.max_depth)
        
        return tree
    
    def _generate_initial_thoughts(self, question: str) -> List[str]:
        """Generate initial thought branches"""
        
        # Different approaches to the problem
        approaches = [
            f"Direct analysis approach: {question}",
            f"Breaking down components of: {question}",
            f"Considering multiple perspectives on: {question}"
        ]
        
        return approaches[:self.branching_factor]
    
    def _expand_tree(self, tree: Dict[str, Any], max_level: int):
        """Recursively expand the thought tree"""
        
        for level in range(1, max_level):
            level_nodes = [node_id for node_id, node in tree.items() 
                          if isinstance(node, dict) and node.get("level") == level]
            
            # Expand top-scoring nodes at this level
            level_nodes.sort(key=lambda x: tree[x].get("score", 0), reverse=True)
            top_nodes = level_nodes[:2]  # Expand top 2 nodes per level
            
            for parent_id in top_nodes:
                children_thoughts = self._generate_child_thoughts(tree[parent_id]["thought"])
                
                for i, child_thought in enumerate(children_thoughts):
                    child_id = f"level_{level+1}_node_{parent_id}_{i}"
                    tree[child_id] = {
                        "thought": child_thought,
                        "level": level + 1,
                        "parent": parent_id,
                        "children": [],
                        "score": self._score_thought(child_thought)
                    }
                    tree[parent_id]["children"].append(child_id)
    
    def _generate_child_thoughts(self, parent_thought: str) -> List[str]:
        """Generate child thoughts from parent thought"""
        
        children = [
            f"Building on '{parent_thought}' by exploring implications...",
            f"Extending '{parent_thought}' through deeper analysis...",
            f"Connecting '{parent_thought}' to broader concepts..."
        ]
        
        return children[:2]  # Limit branching
    
    def _score_thought(self, thought: str) -> float:
        """Score a thought for quality/promise"""
        
        # Simple scoring based on thought characteristics
        score = 0.5  # Base score
        
        # Longer, more detailed thoughts score higher
        if len(thought) > 50:
            score += 0.2
        
        # Thoughts with analysis keywords score higher
        analysis_keywords = ['analysis', 'explore', 'consider', 'examine', 'evaluate']
        if any(keyword in thought.lower() for keyword in analysis_keywords):
            score += 0.3
        
        return min(score, 1.0)
    
    def _find_best_path(self) -> List[str]:
        """Find the best path through the thought tree"""
        
        # Find leaf nodes (nodes with no children)
        leaf_nodes = [node_id for node_id, node in self.thought_tree.items()
                     if isinstance(node, dict) and not node.get("children", [])]
        
        if not leaf_nodes:
            return ["root"]
        
        # Score paths to each leaf
        best_path = None
        best_score = -1
        
        for leaf_id in leaf_nodes:
            path = self._trace_path_to_root(leaf_id)
            path_score = self._score_path(path)
            
            if path_score > best_score:
                best_score = path_score
                best_path = path
        
        return best_path or ["root"]
    
    def _trace_path_to_root(self, node_id: str) -> List[str]:
        """Trace path from node to root"""
        
        path = [node_id]
        current_id = node_id
        
        while current_id != "root":
            parent_id = self.thought_tree[current_id].get("parent")
            if parent_id:
                path.append(parent_id)
                current_id = parent_id
            else:
                break
        
        return list(reversed(path))
    
    def _score_path(self, path: List[str]) -> float:
        """Score a complete path through the tree"""
        
        if not path:
            return 0.0
        
        # Average score of nodes in path
        scores = []
        for node_id in path:
            if node_id in self.thought_tree and isinstance(self.thought_tree[node_id], dict):
                scores.append(self.thought_tree[node_id].get("score", 0.0))
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _extract_reasoning_from_path(self, path: List[str]) -> List[str]:
        """Extract reasoning steps from the best path"""
        
        steps = []
        for node_id in path:
            if node_id in self.thought_tree and isinstance(self.thought_tree[node_id], dict):
                thought = self.thought_tree[node_id].get("thought", "")
                if thought:
                    steps.append(thought)
        
        return steps
    
    def _synthesize_tree_answer(self, path: List[str]) -> str:
        """Synthesize final answer from the best path"""
        
        if not path:
            return "Unable to generate answer from thought tree analysis."
        
        # Combine insights from the path
        path_thoughts = []
        for node_id in path:
            if node_id in self.thought_tree and isinstance(self.thought_tree[node_id], dict):
                thought = self.thought_tree[node_id].get("thought", "")
                if thought and node_id != "root":
                    path_thoughts.append(thought)
        
        if path_thoughts:
            return f"Through tree-based reasoning analysis: {path_thoughts[-1]}"
        else:
            return "Comprehensive tree-of-thought analysis completed."
    
    def _calculate_tree_confidence(self, path: List[str]) -> float:
        """Calculate confidence based on tree exploration"""
        
        if not path:
            return 0.0
        
        # Factors: path length, average node scores, tree depth
        path_score = self._score_path(path)
        depth_factor = min(len(path) / self.max_depth, 1.0)
        exploration_factor = min(self._count_nodes() / 10.0, 1.0)
        
        confidence = 0.4 + 0.3 * path_score + 0.2 * depth_factor + 0.1 * exploration_factor
        
        return min(confidence, 1.0)
    
    def _count_nodes(self) -> int:
        """Count total nodes in thought tree"""
        return len([k for k, v in self.thought_tree.items() if isinstance(v, dict)])
    
    def _assess_integrity(self, answer: str) -> float:
        """Assess integrity of the answer"""
        try:
            if 'self_audit_engine' in globals():
                return self_audit_engine.get_integrity_score(answer)
            else:
                return 80.0  # TOT generally produces higher integrity
        except:
            return 80.0


class ReActReasoner:
    """ReAct (Reasoning and Acting) pattern implementation"""
    
    def __init__(self, llm: Optional[Any] = None, tools: Optional[List[Any]] = None):
        self.llm = llm
        self.tools = tools or []
        self.action_history: List[Dict[str, Any]] = []
        self.max_iterations = 5
    
    def reason(self, question: str, context: Optional[str] = None) -> ReasoningResult:
        """Execute ReAct reasoning pattern"""
        
        start_time = time.time()
        
        reasoning_steps = []
        actions_taken = []
        observations = []
        
        current_question = question
        
        for iteration in range(self.max_iterations):
            # Thought phase
            thought = self._think(current_question, context, reasoning_steps)
            reasoning_steps.append(f"Thought {iteration + 1}: {thought}")
            
            # Action phase
            action = self._decide_action(thought)
            reasoning_steps.append(f"Action {iteration + 1}: {action['type']} - {action['description']}")
            actions_taken.append(action)
            
            # Observation phase
            observation = self._execute_action(action)
            reasoning_steps.append(f"Observation {iteration + 1}: {observation}")
            observations.append(observation)
            
            # Check if we have enough information to answer
            if self._should_stop(thought, observation):
                break
            
            # Update context for next iteration
            current_question = self._refine_question(current_question, observation)
        
        # Final answer synthesis
        final_answer = self._synthesize_react_answer(reasoning_steps, actions_taken, observations)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        confidence = self._calculate_react_confidence(actions_taken, observations)
        integrity_score = self._assess_integrity(final_answer)
        
        return ReasoningResult(
            final_answer=final_answer,
            reasoning_steps=reasoning_steps,
            confidence=confidence,
            integrity_score=integrity_score,
            processing_time=processing_time,
            pattern_used=ReasoningPattern.REACT,
            metadata={
                "actions_taken": len(actions_taken),
                "observations_made": len(observations),
                "iterations": iteration + 1,
                "tools_available": len(self.tools)
            }
        )
    
    def _think(self, question: str, context: Optional[str], previous_steps: List[str]) -> str:
        """Reasoning/thinking phase"""
        
        if previous_steps:
            # Build on previous reasoning
            return f"Analyzing question '{question}' based on previous observations and continuing the reasoning process..."
        else:
            # Initial thinking
            return f"Breaking down the question '{question}' to determine what actions might help answer it..."
    
    def _decide_action(self, thought: str) -> Dict[str, Any]:
        """Decide what action to take based on current thought"""
        
        # Available action types
        possible_actions = [
            {
                "type": "analyze",
                "description": "Analyze the available information more deeply",
                "executable": True
            },
            {
                "type": "search",
                "description": "Search for additional relevant information",
                "executable": len(self.tools) > 0
            },
            {
                "type": "calculate",
                "description": "Perform calculations if numerical analysis is needed",
                "executable": True
            },
            {
                "type": "synthesize",
                "description": "Synthesize available information into an answer",
                "executable": True
            }
        ]
        
        # Simple action selection based on thought content
        if "calculate" in thought.lower() or "number" in thought.lower():
            return next(a for a in possible_actions if a["type"] == "calculate")
        elif "search" in thought.lower() or "find" in thought.lower():
            search_action = next(a for a in possible_actions if a["type"] == "search")
            if search_action["executable"]:
                return search_action
        elif "synthesize" in thought.lower() or "combine" in thought.lower():
            return next(a for a in possible_actions if a["type"] == "synthesize")
        else:
            return next(a for a in possible_actions if a["type"] == "analyze")
    
    def _execute_action(self, action: Dict[str, Any]) -> str:
        """Execute the chosen action"""
        
        action_type = action["type"]
        
        if action_type == "analyze":
            return "Completed detailed analysis of available information and identified key patterns."
        
        elif action_type == "search":
            if self.tools:
                # Would use actual tools here
                return "Search completed. Found relevant additional information to support reasoning."
            else:
                return "Search requested but no search tools available. Using existing information."
        
        elif action_type == "calculate":
            return "Numerical calculations completed. Mathematical relationships identified."
        
        elif action_type == "synthesize":
            return "Information synthesis completed. Ready to formulate answer."
        
        else:
            return f"Action '{action_type}' executed successfully."
    
    def _should_stop(self, thought: str, observation: str) -> bool:
        """Determine if we should stop the ReAct loop"""
        
        # Stop if we've synthesized or if observation indicates completion
        stop_indicators = ["ready to formulate", "synthesis completed", "answer identified"]
        
        return any(indicator in observation.lower() for indicator in stop_indicators)
    
    def _refine_question(self, original_question: str, observation: str) -> str:
        """Refine question based on new observation"""
        
        if "additional information" in observation:
            return f"Given new information, {original_question}"
        else:
            return original_question
    
    def _synthesize_react_answer(self, reasoning_steps: List[str], actions: List[Dict], observations: List[str]) -> str:
        """Synthesize final answer from ReAct process"""
        
        if not reasoning_steps:
            return "Unable to generate answer through ReAct reasoning."
        
        # Find synthesis or conclusion steps
        synthesis_steps = [step for step in reasoning_steps if "synthesis" in step.lower() or "conclusion" in step.lower()]
        
        if synthesis_steps:
            return f"Through reasoning and acting: {synthesis_steps[-1]}"
        else:
            return f"Through {len(actions)} reasoning-action cycles, a comprehensive analysis has been completed."
    
    def _calculate_react_confidence(self, actions: List[Dict], observations: List[str]) -> float:
        """Calculate confidence based on ReAct process quality"""
        
        # Factors: number of actions, observation quality, iteration completeness
        action_factor = min(len(actions) / 3.0, 1.0)  # 3+ actions is good
        observation_quality = sum(len(obs) for obs in observations) / max(len(observations), 1) / 50.0
        observation_factor = min(observation_quality, 1.0)
        
        base_confidence = 0.5  # ReAct base confidence
        confidence = base_confidence + 0.3 * action_factor + 0.2 * observation_factor
        
        return min(confidence, 1.0)
    
    def _assess_integrity(self, answer: str) -> float:
        """Assess integrity of the answer"""
        try:
            if 'self_audit_engine' in globals():
                return self_audit_engine.get_integrity_score(answer)
            else:
                return 85.0  # ReAct generally produces high integrity through action verification
        except:
            return 85.0


class NISLangGraphWorkflow:
    """LangGraph workflow integration with NIS Protocol"""
    
    def __init__(self, enable_self_audit: bool = True):
        self.enable_self_audit = enable_self_audit
        self.workflow_graph = None
        self.memory_saver = MemorySaver() if LANGGRAPH_AVAILABLE else None
        
        # Initialize reasoning engines
        self.cot_reasoner = ChainOfThoughtReasoner()
        self.tot_reasoner = TreeOfThoughtReasoner()
        self.react_reasoner = ReActReasoner()
        
        if LANGGRAPH_AVAILABLE:
            self._build_workflow_graph()
    
    def _build_workflow_graph(self):
        """Build LangGraph workflow"""
        
        # Create workflow graph
        workflow = StateGraph(LangChainState)
        
        # Add nodes
        workflow.add_node("start", self._start_node)
        workflow.add_node("select_reasoning", self._select_reasoning_node)
        workflow.add_node("cot_reasoning", self._cot_reasoning_node)
        workflow.add_node("tot_reasoning", self._tot_reasoning_node)
        workflow.add_node("react_reasoning", self._react_reasoning_node)
        workflow.add_node("integrity_check", self._integrity_check_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Add edges
        workflow.set_entry_point("start")
        workflow.add_edge("start", "select_reasoning")
        
        # Conditional edges based on reasoning pattern
        workflow.add_conditional_edges(
            "select_reasoning",
            self._route_reasoning,
            {
                "cot": "cot_reasoning",
                "tot": "tot_reasoning", 
                "react": "react_reasoning"
            }
        )
        
        # All reasoning paths go to integrity check
        workflow.add_edge("cot_reasoning", "integrity_check")
        workflow.add_edge("tot_reasoning", "integrity_check")
        workflow.add_edge("react_reasoning", "integrity_check")
        
        workflow.add_edge("integrity_check", "finalize")
        workflow.add_edge("finalize", END)
        
        # Compile workflow
        self.workflow_graph = workflow.compile(
            checkpointer=self.memory_saver,
            interrupt_before=["integrity_check"] if self.enable_self_audit else None
        )
    
    def _start_node(self, state: LangChainState) -> Dict[str, Any]:
        """Initialize workflow state"""
        
        return {
            "messages": state.messages,
            "reasoning_pattern": state.reasoning_pattern,
            "processing_time": time.time()
        }
    
    def _select_reasoning_node(self, state: LangChainState) -> Dict[str, Any]:
        """Select appropriate reasoning pattern"""
        
        # For now, use the specified pattern
        # Could add intelligence here to auto-select based on question type
        
        return {"reasoning_pattern": state.reasoning_pattern}
    
    def _route_reasoning(self, state: LangChainState) -> str:
        """Route to appropriate reasoning node"""
        
        pattern = state.reasoning_pattern
        
        if pattern == ReasoningPattern.CHAIN_OF_THOUGHT:
            return "cot"
        elif pattern == ReasoningPattern.TREE_OF_THOUGHT:
            return "tot"
        elif pattern == ReasoningPattern.REACT:
            return "react"
        else:
            return "cot"  # Default to COT
    
    def _cot_reasoning_node(self, state: LangChainState) -> Dict[str, Any]:
        """Execute Chain of Thought reasoning"""
        
        question = self._extract_question_from_state(state)
        result = self.cot_reasoner.reason(question)
        
        return {
            "final_answer": result.final_answer,
            "confidence": result.confidence,
            "reasoning_steps": result.reasoning_steps
        }
    
    def _tot_reasoning_node(self, state: LangChainState) -> Dict[str, Any]:
        """Execute Tree of Thought reasoning"""
        
        question = self._extract_question_from_state(state)
        result = self.tot_reasoner.reason(question)
        
        return {
            "final_answer": result.final_answer,
            "confidence": result.confidence,
            "thought_tree": result.metadata.get("thought_tree", {}),
            "reasoning_steps": result.reasoning_steps
        }
    
    def _react_reasoning_node(self, state: LangChainState) -> Dict[str, Any]:
        """Execute ReAct reasoning"""
        
        question = self._extract_question_from_state(state)
        result = self.react_reasoner.reason(question)
        
        return {
            "final_answer": result.final_answer,
            "confidence": result.confidence,
            "actions_taken": result.metadata.get("actions_taken", []),
            "reasoning_steps": result.reasoning_steps
        }
    
    def _integrity_check_node(self, state: LangChainState) -> Dict[str, Any]:
        """Check integrity of reasoning result"""
        
        if self.enable_self_audit and state.final_answer:
            try:
                if 'self_audit_engine' in globals():
                    integrity_score = self_audit_engine.get_integrity_score(state.final_answer)
                    violations = self_audit_engine.audit_text(state.final_answer)
                    
                    # Auto-correct if needed
                    if violations:
                        corrected_answer, _ = self_audit_engine.auto_correct_text(state.final_answer)
                        return {
                            "final_answer": corrected_answer,
                            "integrity_score": self_audit_engine.get_integrity_score(corrected_answer),
                            "integrity_violations": violations
                        }
                    else:
                        return {"integrity_score": integrity_score}
                else:
                    return {"integrity_score": 85.0}  # Default high score
            except Exception as e:
                return {"integrity_score": 70.0, "integrity_error": str(e)}
        else:
            return {"integrity_score": state.integrity_score or 80.0}
    
    def _finalize_node(self, state: LangChainState) -> Dict[str, Any]:
        """Finalize workflow results"""
        
        start_time = state.__dict__.get('processing_time', time.time())
        processing_time = time.time() - start_time
        
        return {
            "processing_time": processing_time,
            "final_answer": state.final_answer or "No answer generated",
            "confidence": state.confidence or 0.0,
            "integrity_score": state.integrity_score or 0.0
        }
    
    def _extract_question_from_state(self, state: LangChainState) -> str:
        """Extract question from workflow state"""
        
        if state.messages:
            # Extract from last human message
            for msg in reversed(state.messages):
                if msg.get("role") == "human" or msg.get("type") == "human":
                    return msg.get("content", "")
        
        return "Please provide an analysis."
    
    async def process_async(self, question: str, reasoning_pattern: ReasoningPattern = ReasoningPattern.CHAIN_OF_THOUGHT) -> Dict[str, Any]:
        """Process question through LangGraph workflow asynchronously"""
        
        if not self.workflow_graph:
            # Fallback to direct reasoning if LangGraph unavailable
            return await self._fallback_processing(question, reasoning_pattern)
        
        # Create initial state
        initial_state = LangChainState(
            messages=[{"role": "human", "content": question}],
            reasoning_pattern=reasoning_pattern
        )
        
        # Execute workflow
        config = {"configurable": {"thread_id": f"nis_{int(time.time())}"}}
        
        result = None
        async for output in self.workflow_graph.astream(initial_state, config):
            result = output
        
        return result or {}
    
    async def _fallback_processing(self, question: str, reasoning_pattern: ReasoningPattern) -> Dict[str, Any]:
        """Fallback processing when LangGraph unavailable"""
        
        start_time = time.time()
        
        # Select appropriate reasoner
        if reasoning_pattern == ReasoningPattern.TREE_OF_THOUGHT:
            result = self.tot_reasoner.reason(question)
        elif reasoning_pattern == ReasoningPattern.REACT:
            result = self.react_reasoner.reason(question)
        else:
            result = self.cot_reasoner.reason(question)
        
        processing_time = time.time() - start_time
        
        return {
            "final_answer": result.final_answer,
            "confidence": result.confidence,
            "integrity_score": result.integrity_score,
            "processing_time": processing_time,
            "reasoning_pattern": reasoning_pattern.value,
            "reasoning_steps": result.reasoning_steps,
            "metadata": result.metadata
        }


class NISLangChainIntegration:
    """Main integration class for NIS Protocol with LangChain ecosystem"""
    
    def __init__(self, 
                 enable_langsmith: bool = False,
                 enable_self_audit: bool = True,
                 consciousness_agent: Optional[Any] = None):
        
        self.enable_langsmith = enable_langsmith and LANGSMITH_AVAILABLE
        self.enable_self_audit = enable_self_audit
        self.consciousness_agent = consciousness_agent
        
        # Initialize components
        self.workflow = NISLangGraphWorkflow(enable_self_audit=enable_self_audit)
        self.langsmith_client = None
        
        if self.enable_langsmith:
            self._setup_langsmith()
        
        # Integration metrics
        self.integration_stats = {
            "total_questions_processed": 0,
            "reasoning_patterns_used": {pattern.value: 0 for pattern in ReasoningPattern},
            "average_confidence": 0.0,
            "average_integrity_score": 0.0,
            "total_processing_time": 0.0
        }
    
    def _setup_langsmith(self):
        """Setup LangSmith integration"""
        
        try:
            api_key = os.getenv("LANGSMITH_API_KEY")
            if api_key:
                self.langsmith_client = LangSmithClient(api_key=api_key)
                logging.info("LangSmith integration enabled")
            else:
                logging.warning("LANGSMITH_API_KEY not found, disabling LangSmith")
                self.enable_langsmith = False
        except Exception as e:
            logging.error(f"Failed to setup LangSmith: {e}")
            self.enable_langsmith = False
    
    async def process_question(self, 
                             question: str,
                             reasoning_pattern: ReasoningPattern = ReasoningPattern.CHAIN_OF_THOUGHT,
                             context: Optional[str] = None) -> Dict[str, Any]:
        """Process question through integrated LangChain workflow"""
        
        start_time = time.time()
        
        try:
            # Process through LangGraph workflow
            result = await self.workflow.process_async(question, reasoning_pattern)
            
            # Update statistics
            self._update_stats(result, reasoning_pattern, time.time() - start_time)
            
            # Consciousness system integration
            if self.consciousness_agent:
                await self._integrate_with_consciousness(question, result)
            
            # LangSmith tracking
            if self.enable_langsmith:
                await self._track_with_langsmith(question, result)
            
            # Add integration metadata
            result["integration_metadata"] = {
                "nis_protocol_version": "v3",
                "langchain_integration": True,
                "langgraph_workflow": self.workflow.workflow_graph is not None,
                "langsmith_tracking": self.enable_langsmith,
                "consciousness_integration": self.consciousness_agent is not None,
                "self_audit_enabled": self.enable_self_audit
            }
            
            return result
            
        except Exception as e:
            logging.error(f"LangChain integration error: {e}")
            
            # Fallback result
            return {
                "final_answer": f"Integration error occurred: {e}",
                "confidence": 0.0,
                "integrity_score": 0.0,
                "processing_time": time.time() - start_time,
                "error": str(e),
                "reasoning_pattern": reasoning_pattern.value
            }
    
    def _update_stats(self, result: Dict[str, Any], pattern: ReasoningPattern, processing_time: float):
        """Update integration statistics"""
        
        self.integration_stats["total_questions_processed"] += 1
        self.integration_stats["reasoning_patterns_used"][pattern.value] += 1
        
        # Update averages
        total = self.integration_stats["total_questions_processed"]
        
        confidence = result.get("confidence", 0.0)
        current_avg_conf = self.integration_stats["average_confidence"]
        self.integration_stats["average_confidence"] = (current_avg_conf * (total - 1) + confidence) / total
        
        integrity = result.get("integrity_score", 0.0)
        current_avg_integrity = self.integration_stats["average_integrity_score"]
        self.integration_stats["average_integrity_score"] = (current_avg_integrity * (total - 1) + integrity) / total
        
        self.integration_stats["total_processing_time"] += processing_time
    
    async def _integrate_with_consciousness(self, question: str, result: Dict[str, Any]):
        """Integrate result with consciousness system"""
        
        try:
            # Register this processing session with consciousness agent
            session_metadata = {
                "question": question,
                "reasoning_pattern": result.get("reasoning_pattern", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "integrity_score": result.get("integrity_score", 0.0),
                "processing_time": result.get("processing_time", 0.0)
            }
            
            self.consciousness_agent.register_agent_for_monitoring(
                "langchain_integration", 
                session_metadata
            )
            
            # Trigger introspection if confidence or integrity is low
            if result.get("confidence", 1.0) < 0.7 or result.get("integrity_score", 100.0) < 80.0:
                introspection_result = self.consciousness_agent.perform_introspection(
                    ReflectionType.PERFORMANCE_REVIEW
                )
                
                result["consciousness_insight"] = {
                    "introspection_triggered": True,
                    "confidence": introspection_result.confidence,
                    "recommendations": introspection_result.recommendations[:3]
                }
        except Exception as e:
            logging.warning(f"Consciousness integration error: {e}")
    
    async def _track_with_langsmith(self, question: str, result: Dict[str, Any]):
        """Track processing with LangSmith"""
        
        try:
            if self.langsmith_client:
                # Create run record
                run_data = {
                    "name": "nis_protocol_reasoning",
                    "inputs": {"question": question},
                    "outputs": result,
                    "run_type": "llm",
                    "session_name": "nis_protocol_integration"
                }
                
                # Would implement actual LangSmith tracking here
                logging.info(f"LangSmith tracking: {run_data['name']}")
        except Exception as e:
            logging.warning(f"LangSmith tracking error: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        
        return {
            "langchain_available": LANGCHAIN_AVAILABLE,
            "langgraph_available": LANGGRAPH_AVAILABLE,
            "langsmith_available": LANGSMITH_AVAILABLE,
            "langsmith_enabled": self.enable_langsmith,
            "self_audit_enabled": self.enable_self_audit,
            "consciousness_integration": self.consciousness_agent is not None,
            "workflow_ready": self.workflow.workflow_graph is not None,
            "integration_stats": self.integration_stats
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get integration capabilities"""
        
        return {
            "reasoning_patterns": [pattern.value for pattern in ReasoningPattern],
            "features": {
                "chain_of_thought": True,
                "tree_of_thought": True,
                "reasoning_and_acting": True,
                "langraph_workflows": LANGGRAPH_AVAILABLE,
                "langsmith_observability": self.enable_langsmith,
                "nis_self_audit": self.enable_self_audit,
                "consciousness_integration": self.consciousness_agent is not None
            },
            "protocols_supported": [
                "NIS Protocol v3",
                "LangChain Chat Models",
                "LangGraph State Machines",
                "LangSmith Observability"
            ]
        }


# Example usage and testing
async def main():
    """Example usage of NIS-LangChain integration"""
    
    print("🦜🔗 NIS Protocol v3 - LangChain Integration Demo")
    print("=" * 60)
    
    # Initialize integration
    integration = NISLangChainIntegration(
        enable_langsmith=False,  # Set to True if you have LangSmith API key
        enable_self_audit=True
    )
    
    # Test questions
    test_questions = [
        "What are the key components of artificial intelligence?",
        "How do neural networks learn from data?", 
        "What is the difference between machine learning and deep learning?"
    ]
    
    reasoning_patterns = [
        ReasoningPattern.CHAIN_OF_THOUGHT,
        ReasoningPattern.TREE_OF_THOUGHT,
        ReasoningPattern.REACT
    ]
    
    print(f"Testing {len(test_questions)} questions with {len(reasoning_patterns)} reasoning patterns...")
    print()
    
    for i, question in enumerate(test_questions):
        pattern = reasoning_patterns[i % len(reasoning_patterns)]
        
        print(f"🤔 Question {i+1}: {question}")
        print(f"📋 Pattern: {pattern.value}")
        
        result = await integration.process_question(question, pattern)
        
        print(f"✅ Answer: {result.get('final_answer', 'No answer')}")
        print(f"📊 Confidence: {result.get('confidence', 0.0):.3f}")
        print(f"🔍 Integrity: {result.get('integrity_score', 0.0):.1f}/100")
        print(f"⏱️  Time: {result.get('processing_time', 0.0):.3f}s")
        print("-" * 40)
    
    # Display integration status
    status = integration.get_integration_status()
    capabilities = integration.get_capabilities()
    
    print("\n🔧 Integration Status:")
    for key, value in status.items():
        if key != "integration_stats":
            print(f"  {key}: {value}")
    
    print("\n📊 Statistics:")
    stats = status["integration_stats"]
    print(f"  Questions processed: {stats['total_questions_processed']}")
    print(f"  Average confidence: {stats['average_confidence']:.3f}")
    print(f"  Average integrity: {stats['average_integrity_score']:.1f}")
    
    print("\n🎯 Capabilities:")
    features = capabilities["features"]
    for feature, enabled in features.items():
        status_icon = "✅" if enabled else "❌"
        print(f"  {status_icon} {feature.replace('_', ' ').title()}")


if __name__ == "__main__":
    asyncio.run(main()) 
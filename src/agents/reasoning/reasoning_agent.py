"""
Reasoning Agent

Handles logical reasoning, decision making, and problem solving in the NIS Protocol.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import time
import numpy as np
from transformers import pipeline

from src.core.registry import NISAgent, NISLayer
from src.emotion.emotional_state import EmotionalState
from src.agents.interpretation.interpretation_agent import InterpretationAgent

class ReasoningAgent(NISAgent):
    """
    Agent responsible for logical reasoning and decision making.
    
    This agent implements various reasoning strategies including:
    - Deductive reasoning
    - Inductive reasoning
    - Abductive reasoning
    - Causal reasoning
    - Analogical reasoning
    """
    
    def __init__(
        self,
        agent_id: str = "reasoner",
        description: str = "Handles logical reasoning",
        emotional_state: Optional[EmotionalState] = None,
        interpreter: Optional[InterpretationAgent] = None,
        model_name: str = "google/flan-t5-large",
        confidence_threshold: float = 0.7,
        max_reasoning_steps: int = 5
    ):
        """
        Initialize the reasoning agent.
        
        Args:
            agent_id: Unique identifier for this agent
            description: Human-readable description of the agent's role
            emotional_state: Optional pre-configured emotional state
            interpreter: Optional interpreter for understanding inputs
            model_name: Name of the transformer model to use
            confidence_threshold: Minimum confidence for conclusions
            max_reasoning_steps: Maximum steps in reasoning chain
        """
        super().__init__(agent_id, NISLayer.REASONING, description)
        self.emotional_state = emotional_state or EmotionalState()
        self.interpreter = interpreter
        self.confidence_threshold = confidence_threshold
        self.max_reasoning_steps = max_reasoning_steps
        
        # Initialize reasoning pipelines
        self.text_generator = pipeline(
            "text2text-generation",
            model=model_name,
            device=-1  # CPU
        )
        
        # Cache for reasoning chains
        self.reasoning_cache = {}
        self.cache_size = 50
        
        # Track active reasoning chains
        self.active_chains = {}
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process reasoning requests.
        
        Args:
            message: Message containing reasoning operation
                'operation': Operation to perform
                    ('analyze', 'deduce', 'induce', 'solve', 'explain')
                'content': Content to reason about
                + Additional parameters based on operation
                
        Returns:
            Result of the reasoning operation
        """
        operation = message.get("operation", "").lower()
        content = message.get("content", "")
        
        if not content:
            return {
                "status": "error",
                "error": "No content provided for reasoning",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        # Check cache first
        cache_key = f"{operation}:{content}"
        if cache_key in self.reasoning_cache:
            return self.reasoning_cache[cache_key]
        
        # Process the requested operation
        if operation == "analyze":
            result = self._analyze_problem(content)
        elif operation == "deduce":
            premises = message.get("premises", [])
            result = self._deductive_reasoning(content, premises)
        elif operation == "induce":
            examples = message.get("examples", [])
            result = self._inductive_reasoning(content, examples)
        elif operation == "solve":
            constraints = message.get("constraints", {})
            result = self._problem_solving(content, constraints)
        elif operation == "explain":
            context = message.get("context", "")
            result = self._generate_explanation(content, context)
        else:
            return {
                "status": "error",
                "error": f"Unknown operation: {operation}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
            
        # Update cache
        self._update_cache(cache_key, result)
        
        return result
    
    def _analyze_problem(self, content: str) -> Dict[str, Any]:
        """
        Analyze a problem to identify key components and relationships.
        
        Args:
            content: Problem description to analyze
            
        Returns:
            Analysis result
        """
        try:
            # First interpret the content if interpreter is available
            interpretation = None
            if self.interpreter:
                interpretation = self.interpreter.process({
                    "operation": "interpret",
                    "content": content
                })
            
            # Generate problem analysis prompt
            prompt = f"Analyze this problem: {content}\n\nIdentify:\n1. Key elements\n2. Relationships\n3. Constraints\n4. Goals"
            
            # Generate analysis
            analysis = self.text_generator(
                prompt,
                max_length=200,
                num_return_sequences=1
            )[0]["generated_text"]
            
            return {
                "status": "success",
                "analysis": analysis,
                "interpretation": interpretation,
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Problem analysis failed: {str(e)}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _deductive_reasoning(
        self,
        conclusion: str,
        premises: List[str]
    ) -> Dict[str, Any]:
        """
        Apply deductive reasoning to reach a conclusion.
        
        Args:
            conclusion: Proposed conclusion
            premises: List of premises to reason from
            
        Returns:
            Deductive reasoning result
        """
        try:
            # Format premises for the model
            premises_text = "\n".join(f"- {p}" for p in premises)
            prompt = f"Given these premises:\n{premises_text}\n\nIs this conclusion valid? {conclusion}\n\nExplain the logical steps:"
            
            # Generate reasoning chain
            reasoning = self.text_generator(
                prompt,
                max_length=300,
                num_return_sequences=1
            )[0]["generated_text"]
            
            # Extract validity assessment (simple heuristic)
            is_valid = any(
                marker in reasoning.lower()
                for marker in ["valid", "correct", "follows", "therefore"]
            )
            
            return {
                "status": "success",
                "is_valid": is_valid,
                "reasoning_chain": reasoning,
                "confidence": 0.8 if is_valid else 0.4,
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Deductive reasoning failed: {str(e)}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _inductive_reasoning(
        self,
        hypothesis: str,
        examples: List[str]
    ) -> Dict[str, Any]:
        """
        Apply inductive reasoning to evaluate a hypothesis.
        
        Args:
            hypothesis: Proposed hypothesis
            examples: List of supporting examples
            
        Returns:
            Inductive reasoning result
        """
        try:
            # Format examples for the model
            examples_text = "\n".join(f"- {e}" for e in examples)
            prompt = f"Consider these examples:\n{examples_text}\n\nEvaluate this hypothesis: {hypothesis}\n\nExplain the pattern and confidence:"
            
            # Generate reasoning
            reasoning = self.text_generator(
                prompt,
                max_length=300,
                num_return_sequences=1
            )[0]["generated_text"]
            
            # Extract confidence assessment (simple heuristic)
            confidence = 0.0
            if "strong evidence" in reasoning.lower():
                confidence = 0.9
            elif "moderate evidence" in reasoning.lower():
                confidence = 0.7
            elif "weak evidence" in reasoning.lower():
                confidence = 0.4
            else:
                confidence = 0.5
            
            return {
                "status": "success",
                "hypothesis_evaluation": reasoning,
                "confidence": confidence,
                "supporting_examples": examples,
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Inductive reasoning failed: {str(e)}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _problem_solving(
        self,
        problem: str,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply problem-solving strategies to find a solution.
        
        Args:
            problem: Problem description
            constraints: Dictionary of problem constraints
            
        Returns:
            Problem-solving result
        """
        try:
            # Format constraints for the model
            constraints_text = "\n".join(
                f"- {k}: {v}"
                for k, v in constraints.items()
            )
            
            prompt = f"""
            Solve this problem: {problem}
            
            Constraints:
            {constraints_text}
            
            Follow these steps:
            1. Analyze the problem
            2. Break it into sub-problems
            3. Generate potential solutions
            4. Evaluate solutions against constraints
            5. Recommend the best solution
            """
            
            # Generate solution
            solution = self.text_generator(
                prompt,
                max_length=500,
                num_return_sequences=1
            )[0]["generated_text"]
            
            return {
                "status": "success",
                "solution": solution,
                "problem_analysis": self._analyze_problem(problem),
                "constraints_met": self._evaluate_constraints(solution, constraints),
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Problem solving failed: {str(e)}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _generate_explanation(
        self,
        phenomenon: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Generate an explanation for a phenomenon.
        
        Args:
            phenomenon: The phenomenon to explain
            context: Additional context information
            
        Returns:
            Explanation result
        """
        try:
            prompt = f"""
            Explain this phenomenon: {phenomenon}
            
            Additional context: {context}
            
            Provide:
            1. Causal factors
            2. Underlying mechanisms
            3. Supporting evidence
            4. Potential implications
            """
            
            # Generate explanation
            explanation = self.text_generator(
                prompt,
                max_length=400,
                num_return_sequences=1
            )[0]["generated_text"]
            
            return {
                "status": "success",
                "explanation": explanation,
                "context_used": bool(context),
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Explanation generation failed: {str(e)}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _evaluate_constraints(
        self,
        solution: str,
        constraints: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Evaluate if a solution meets given constraints.
        
        Args:
            solution: Proposed solution
            constraints: Dictionary of constraints
            
        Returns:
            Dictionary of constraint satisfaction results
        """
        results = {}
        for constraint_name, constraint_value in constraints.items():
            # This is a simple check - in practice, you'd want more sophisticated
            # constraint evaluation based on the type of constraint
            is_satisfied = str(constraint_value).lower() in solution.lower()
            results[constraint_name] = is_satisfied
        
        return results
    
    def _update_cache(self, key: str, value: Dict[str, Any]) -> None:
        """
        Update the reasoning cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self.reasoning_cache[key] = value
        
        # Remove oldest entries if cache is too large
        if len(self.reasoning_cache) > self.cache_size:
            oldest_key = min(
                self.reasoning_cache.keys(),
                key=lambda k: self.reasoning_cache[k]["timestamp"]
            )
            del self.reasoning_cache[oldest_key]
    
    def start_reasoning_chain(self, chain_id: str) -> None:
        """
        Start a new reasoning chain.
        
        Args:
            chain_id: Unique identifier for the chain
        """
        self.active_chains[chain_id] = {
            "steps": [],
            "start_time": time.time(),
            "status": "active"
        }
    
    def add_reasoning_step(
        self,
        chain_id: str,
        step_type: str,
        content: Dict[str, Any]
    ) -> None:
        """
        Add a step to an active reasoning chain.
        
        Args:
            chain_id: Chain identifier
            step_type: Type of reasoning step
            content: Step content and results
        """
        if chain_id not in self.active_chains:
            return
            
        self.active_chains[chain_id]["steps"].append({
            "type": step_type,
            "content": content,
            "timestamp": time.time()
        })
    
    def end_reasoning_chain(self, chain_id: str) -> Dict[str, Any]:
        """
        End a reasoning chain and return results.
        
        Args:
            chain_id: Chain identifier
            
        Returns:
            Chain results
        """
        if chain_id not in self.active_chains:
            return {
                "status": "error",
                "error": "Chain not found",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
            
        chain = self.active_chains[chain_id]
        chain["status"] = "completed"
        chain["end_time"] = time.time()
        chain["duration"] = chain["end_time"] - chain["start_time"]
        
        return {
            "status": "success",
            "chain_id": chain_id,
            "steps": chain["steps"],
            "duration": chain["duration"],
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def get_active_chains(self) -> List[str]:
        """
        Get list of active reasoning chains.
        
        Returns:
            List of active chain IDs
        """
        return [
            chain_id
            for chain_id, chain in self.active_chains.items()
            if chain["status"] == "active"
        ]
    
    def clear_cache(self) -> None:
        """Clear the reasoning cache."""
        self.reasoning_cache = {} 
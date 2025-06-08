"""
NIS Protocol Conscious Agent

This agent provides meta-cognitive capabilities, self-reflection, and monitoring
of other agents in the system. It represents the "consciousness" layer that
observes and evaluates the system's own thinking processes.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from ...core.agent import NISAgent, NISLayer
from ...memory.memory_manager import MemoryManager


class ReflectionType(Enum):
    """Types of reflection the conscious agent can perform"""
    PERFORMANCE_REVIEW = "performance_review"
    ERROR_ANALYSIS = "error_analysis"
    GOAL_EVALUATION = "goal_evaluation"
    EMOTIONAL_STATE_REVIEW = "emotional_state_review"
    MEMORY_CONSOLIDATION = "memory_consolidation"


@dataclass
class IntrospectionResult:
    """Result of introspective analysis"""
    reflection_type: ReflectionType
    agent_id: str
    findings: Dict[str, Any]
    recommendations: List[str]
    confidence: float
    timestamp: float


class ConsciousAgent(NISAgent):
    """Meta-cognitive agent that monitors and reflects on system behavior.
    
    This agent provides:
    - Self-reflection on decision quality
    - Performance monitoring of other agents
    - Error detection and analysis
    - Meta-cognitive reasoning about thinking processes
    """
    
    def __init__(
        self,
        agent_id: str = "conscious_agent",
        description: str = "Meta-cognitive agent for self-reflection and monitoring"
    ):
        super().__init__(agent_id, NISLayer.REASONING, description)
        self.logger = logging.getLogger(f"nis.{agent_id}")
        
        # Initialize memory for introspection
        self.memory = MemoryManager()
        
        # Track agent performance over time
        self.agent_performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Meta-cognitive state
        self.current_focus: Optional[str] = None
        self.reflection_queue: List[Dict[str, Any]] = []
        self.confidence_threshold = 0.7
        
        self.logger.info(f"Initialized {self.__class__.__name__}")
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process introspection requests and meta-cognitive tasks.
        
        Args:
            message: Input message with introspection request
            
        Returns:
            Introspection results and recommendations
        """
        start_time = self._start_processing_timer()
        
        try:
            operation = message.get("operation", "introspect")
            
            if operation == "introspect":
                result = self._perform_introspection(message)
            elif operation == "evaluate_decision":
                result = self._evaluate_decision(message)
            elif operation == "monitor_agent":
                result = self._monitor_agent(message)
            elif operation == "reflect_on_error":
                result = self._reflect_on_error(message)
            elif operation == "consolidate_insights":
                result = self._consolidate_insights(message)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            # Update emotional state based on findings
            emotional_state = self._assess_emotional_impact(result)
            
            response = self._create_response(
                "success",
                result,
                {"operation": operation, "focus": self.current_focus},
                emotional_state
            )
            
        except Exception as e:
            self.logger.error(f"Error in conscious processing: {str(e)}")
            response = self._create_response(
                "error",
                {"error": str(e)},
                {"operation": operation}
            )
        
        finally:
            self._end_processing_timer(start_time)
        
        return response
    
    def _perform_introspection(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Perform introspective analysis on system state.
        
        Args:
            message: Message containing introspection parameters
            
        Returns:
            Introspection results
        """
        target_agent = message.get("target_agent")
        reflection_type = ReflectionType(message.get("reflection_type", "performance_review"))
        
        # Gather data about the target agent or system
        agent_data = self._gather_agent_data(target_agent)
        
        # Perform reflection based on type
        if reflection_type == ReflectionType.PERFORMANCE_REVIEW:
            findings = self._analyze_performance(agent_data)
        elif reflection_type == ReflectionType.ERROR_ANALYSIS:
            findings = self._analyze_errors(agent_data)
        elif reflection_type == ReflectionType.GOAL_EVALUATION:
            findings = self._evaluate_goals(agent_data)
        elif reflection_type == ReflectionType.EMOTIONAL_STATE_REVIEW:
            findings = self._review_emotional_state(agent_data)
        else:
            findings = self._general_reflection(agent_data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(findings, reflection_type)
        
        # Create introspection result
        result = IntrospectionResult(
            reflection_type=reflection_type,
            agent_id=target_agent or "system",
            findings=findings,
            recommendations=recommendations,
            confidence=self._calculate_confidence(findings),
            timestamp=time.time()
        )
        
        # Store for future reference
        self._store_introspection_result(result)
        
        return {
            "introspection_result": result.__dict__,
            "meta_insights": self._extract_meta_insights(result)
        }
    
    def _evaluate_decision(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the quality of a decision made by the system.
        
        Args:
            message: Message containing decision information
            
        Returns:
            Decision evaluation results
        """
        decision_data = message.get("decision_data", {})
        
        # Analyze decision quality
        quality_metrics = {
            "logical_consistency": self._check_logical_consistency(decision_data),
            "emotional_appropriateness": self._check_emotional_appropriateness(decision_data),
            "goal_alignment": self._check_goal_alignment(decision_data),
            "risk_assessment": self._assess_decision_risk(decision_data)
        }
        
        # Calculate overall decision quality
        overall_quality = sum(quality_metrics.values()) / len(quality_metrics)
        
        # Generate improvement suggestions
        improvements = self._suggest_improvements(quality_metrics, decision_data)
        
        return {
            "decision_quality": overall_quality,
            "quality_breakdown": quality_metrics,
            "improvement_suggestions": improvements,
            "meta_commentary": self._generate_meta_commentary(decision_data, quality_metrics)
        }
    
    def _monitor_agent(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor another agent's performance and behavior.
        
        Args:
            message: Message containing agent monitoring request
            
        Returns:
            Agent monitoring results
        """
        agent_id = message.get("agent_id")
        monitoring_duration = message.get("duration", 60)  # seconds
        
        if not agent_id:
            raise ValueError("Agent ID required for monitoring")
        
        # Collect performance data
        performance_data = self._collect_agent_performance(agent_id, monitoring_duration)
        
        # Analyze patterns
        patterns = self._detect_performance_patterns(performance_data)
        
        # Identify issues
        issues = self._identify_performance_issues(performance_data)
        
        # Track in history
        self._update_performance_history(agent_id, performance_data)
        
        return {
            "agent_id": agent_id,
            "monitoring_duration": monitoring_duration,
            "performance_summary": performance_data,
            "detected_patterns": patterns,
            "identified_issues": issues,
            "recommendations": self._generate_agent_recommendations(agent_id, performance_data)
        }
    
    def _reflect_on_error(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on an error that occurred in the system.
        
        Args:
            message: Message containing error information
            
        Returns:
            Error reflection results
        """
        error_data = message.get("error_data", {})
        
        # Analyze error context
        error_context = self._analyze_error_context(error_data)
        
        # Identify root causes
        root_causes = self._identify_root_causes(error_data, error_context)
        
        # Generate learning insights
        learning_insights = self._extract_learning_insights(error_data, root_causes)
        
        # Suggest preventive measures
        preventive_measures = self._suggest_preventive_measures(root_causes)
        
        return {
            "error_analysis": {
                "context": error_context,
                "root_causes": root_causes,
                "learning_insights": learning_insights,
                "preventive_measures": preventive_measures
            },
            "meta_learning": self._generate_meta_learning(error_data, root_causes)
        }
    
    def _consolidate_insights(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate insights from multiple introspection sessions.
        
        Args:
            message: Message containing consolidation parameters
            
        Returns:
            Consolidated insights
        """
        # Retrieve recent introspection results
        recent_results = self._get_recent_introspection_results()
        
        # Find patterns across sessions
        patterns = self._find_cross_session_patterns(recent_results)
        
        # Generate system-level insights
        system_insights = self._generate_system_insights(patterns)
        
        # Create learning recommendations
        learning_recommendations = self._create_learning_recommendations(system_insights)
        
        return {
            "consolidated_patterns": patterns,
            "system_level_insights": system_insights,
            "learning_recommendations": learning_recommendations,
            "meta_evolution": self._track_meta_evolution(recent_results)
        }
    
    # Helper methods for analysis
    def _gather_agent_data(self, agent_id: Optional[str]) -> Dict[str, Any]:
        """Gather performance and state data for an agent."""
        # Implementation would collect real agent data
        return {
            "processing_times": [0.1, 0.15, 0.08, 0.12],
            "success_rate": 0.95,
            "error_count": 2,
            "last_activities": ["perception", "reasoning", "action"]
        }
    
    def _analyze_performance(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze agent performance metrics."""
        return {
            "efficiency": agent_data.get("success_rate", 0.0),
            "speed": 1.0 / (sum(agent_data.get("processing_times", [1.0])) / len(agent_data.get("processing_times", [1.0]))),
            "reliability": 1.0 - (agent_data.get("error_count", 0) / 100.0)
        }
    
    def _generate_recommendations(self, findings: Dict[str, Any], reflection_type: ReflectionType) -> List[str]:
        """Generate actionable recommendations based on findings."""
        recommendations = []
        
        if reflection_type == ReflectionType.PERFORMANCE_REVIEW:
            if findings.get("efficiency", 0) < 0.8:
                recommendations.append("Consider optimizing processing algorithms")
            if findings.get("speed", 0) < 0.5:
                recommendations.append("Review processing pipeline for bottlenecks")
        
        return recommendations
    
    def _calculate_confidence(self, findings: Dict[str, Any]) -> float:
        """Calculate confidence in the introspection results."""
        # Simple confidence calculation based on data completeness
        data_completeness = len([v for v in findings.values() if v is not None]) / len(findings)
        return min(data_completeness * 1.2, 1.0)
    
    def _store_introspection_result(self, result: IntrospectionResult) -> None:
        """Store introspection result in memory."""
        self.memory.store(
            f"introspection_{result.timestamp}",
            {
                "type": "introspection_result",
                "data": result.__dict__
            },
            ttl=86400  # Store for 24 hours
        )
    
    def _assess_emotional_impact(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Assess emotional impact of introspection findings."""
        confidence = result.get("introspection_result", {}).get("confidence", 0.5)
        
        # Higher confidence in findings increases satisfaction
        satisfaction = confidence
        
        # Low performance findings increase concern
        concern = 1.0 - confidence
        
        return {
            "satisfaction": satisfaction,
            "concern": concern,
            "curiosity": 0.7  # Always curious for meta-cognitive agent
        }
    
    # Placeholder methods for complex analysis (to be implemented)
    def _analyze_errors(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"error_patterns": [], "severity": "low"}
    
    def _evaluate_goals(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"goal_progress": 0.8, "alignment": "good"}
    
    def _review_emotional_state(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"emotional_stability": 0.9, "appropriateness": "high"}
    
    def _general_reflection(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"overall_status": "healthy", "improvement_areas": []}
    
    def _extract_meta_insights(self, result: IntrospectionResult) -> Dict[str, Any]:
        return {"learning_potential": "high", "adaptation_needed": "minimal"}
    
    def _check_logical_consistency(self, decision_data: Dict[str, Any]) -> float:
        return 0.85  # Placeholder
    
    def _check_emotional_appropriateness(self, decision_data: Dict[str, Any]) -> float:
        return 0.90  # Placeholder
    
    def _check_goal_alignment(self, decision_data: Dict[str, Any]) -> float:
        return 0.88  # Placeholder
    
    def _assess_decision_risk(self, decision_data: Dict[str, Any]) -> float:
        return 0.75  # Placeholder (higher is lower risk)
    
    def _suggest_improvements(self, quality_metrics: Dict[str, float], decision_data: Dict[str, Any]) -> List[str]:
        return ["Consider additional emotional context", "Verify goal alignment"]
    
    def _generate_meta_commentary(self, decision_data: Dict[str, Any], quality_metrics: Dict[str, float]) -> str:
        return "Decision shows good logical consistency but could benefit from enhanced emotional awareness."
    
    def _collect_agent_performance(self, agent_id: str, duration: int) -> Dict[str, Any]:
        return {"avg_response_time": 0.12, "success_rate": 0.95, "errors": 1}
    
    def _detect_performance_patterns(self, performance_data: Dict[str, Any]) -> List[str]:
        return ["Consistent response times", "High success rate maintained"]
    
    def _identify_performance_issues(self, performance_data: Dict[str, Any]) -> List[str]:
        return [] if performance_data.get("success_rate", 0) > 0.9 else ["Low success rate detected"]
    
    def _update_performance_history(self, agent_id: str, performance_data: Dict[str, Any]) -> None:
        if agent_id not in self.agent_performance_history:
            self.agent_performance_history[agent_id] = []
        self.agent_performance_history[agent_id].append({
            "timestamp": time.time(),
            "data": performance_data
        })
    
    def _generate_agent_recommendations(self, agent_id: str, performance_data: Dict[str, Any]) -> List[str]:
        return ["Continue current performance level", "Monitor for any degradation"]
    
    def _analyze_error_context(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"context_type": "processing", "severity": "medium"}
    
    def _identify_root_causes(self, error_data: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        return ["Insufficient input validation", "Resource constraints"]
    
    def _extract_learning_insights(self, error_data: Dict[str, Any], root_causes: List[str]) -> List[str]:
        return ["Implement stricter input validation", "Consider resource scaling"]
    
    def _suggest_preventive_measures(self, root_causes: List[str]) -> List[str]:
        return ["Add input validation layer", "Implement resource monitoring"]
    
    def _generate_meta_learning(self, error_data: Dict[str, Any], root_causes: List[str]) -> str:
        return "System shows capacity for error recovery and learning from mistakes."
    
    def _get_recent_introspection_results(self) -> List[Dict[str, Any]]:
        # Search memory for recent introspection results
        pattern = {"type": "introspection_result"}
        return self.memory.search(pattern)
    
    def _find_cross_session_patterns(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"recurring_themes": ["performance optimization", "error reduction"]}
    
    def _generate_system_insights(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        return {"overall_trend": "improving", "focus_areas": ["efficiency", "reliability"]}
    
    def _create_learning_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        return ["Continue performance monitoring", "Enhance error prevention"]
    
    def _track_meta_evolution(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"evolution_stage": "early", "growth_potential": "high"} 
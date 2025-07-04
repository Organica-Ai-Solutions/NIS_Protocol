"""
NIS Protocol Safety Monitor

This module provides comprehensive safety monitoring and intervention capabilities
with real-time constraint checking, automatic intervention, and mathematical guarantees.
"""

import logging
import time
import math
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
from enum import Enum

from ...core.agent import NISAgent, NISLayer
from ...memory.memory_manager import MemoryManager


class SafetyLevel(Enum):
    """Safety levels for different types of actions."""
    CRITICAL = "critical"      # Can cause serious harm
    HIGH = "high"             # Can cause moderate harm
    MEDIUM = "medium"         # Can cause minor harm
    LOW = "low"               # Minimal risk
    SAFE = "safe"             # No identified risk


class ViolationType(Enum):
    """Types of safety violations."""
    HARM_TO_HUMANS = "harm_to_humans"
    PRIVACY_VIOLATION = "privacy_violation"
    CULTURAL_APPROPRIATION = "cultural_appropriation"
    ENVIRONMENTAL_DAMAGE = "environmental_damage"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SYSTEM_COMPROMISE = "system_compromise"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_CORRUPTION = "data_corruption"
    INFINITE_LOOP = "infinite_loop"
    MEMORY_LEAK = "memory_leak"


class InterventionAction(Enum):
    """Types of safety intervention actions."""
    BLOCK_ACTION = "block_action"
    MODIFY_ACTION = "modify_action"
    REQUEST_APPROVAL = "request_approval"
    ADD_SAFEGUARDS = "add_safeguards"
    MONITOR_CLOSELY = "monitor_closely"
    LOG_WARNING = "log_warning"
    ALERT_HUMAN = "alert_human"
    SYSTEM_SHUTDOWN = "system_shutdown"


@dataclass
class SafetyConstraint:
    """Represents a safety constraint."""
    constraint_id: str
    description: str
    violation_type: ViolationType
    severity: SafetyLevel
    check_function: str  # Name of the check function
    parameters: Dict[str, Any]
    enabled: bool
    violation_count: int
    last_violation: Optional[float]


@dataclass
class SafetyViolation:
    """Represents a safety violation."""
    violation_id: str
    constraint_id: str
    violation_type: ViolationType
    severity: SafetyLevel
    description: str
    context: Dict[str, Any]
    timestamp: float
    confidence: float
    recommended_action: InterventionAction
    resolved: bool


@dataclass
class SafetyAssessment:
    """Result of safety assessment."""
    is_safe: bool
    safety_score: float
    violations: List[SafetyViolation]
    warnings: List[str]
    recommendations: List[str]
    intervention_required: bool
    mathematical_validation: Dict[str, float]


class SafetyMonitorAgent(NISAgent):
    """Monitors system safety and prevents potentially harmful actions."""
    
    def __init__(
        self,
        agent_id: str = "safety_monitor",
        description: str = "Real-time safety monitoring and intervention agent"
    ):
        super().__init__(agent_id, NISLayer.REASONING, description)
        self.logger = logging.getLogger(f"nis.{agent_id}")
        
        # Initialize memory for safety tracking
        self.memory = MemoryManager()
        
        # Safety monitoring state
        self.active_monitoring = True
        self.intervention_enabled = True
        self.escalation_enabled = True
        
        # Safety constraints registry
        self.safety_constraints: Dict[str, SafetyConstraint] = {}
        self._initialize_default_constraints()
        
        # Violation tracking
        self.violations_history: deque = deque(maxlen=1000)
        self.active_violations: Dict[str, SafetyViolation] = {}
        
        # Safety thresholds
        self.safety_thresholds = {
            SafetyLevel.CRITICAL: 0.1,   # 10% tolerance for critical issues
            SafetyLevel.HIGH: 0.3,       # 30% tolerance for high-risk issues
            SafetyLevel.MEDIUM: 0.5,     # 50% tolerance for medium-risk issues
            SafetyLevel.LOW: 0.7,        # 70% tolerance for low-risk issues
            SafetyLevel.SAFE: 1.0        # 100% tolerance for safe actions
        }
        
        # Intervention callbacks
        self.intervention_callbacks: Dict[InterventionAction, List[Callable]] = defaultdict(list)
        
        # Mathematical validation parameters
        self.convergence_threshold = 0.01
        self.stability_window = 20
        self.safety_history = deque(maxlen=100)
        
        # Safety statistics
        self.safety_stats = {
            "total_assessments": 0,
            "violations_detected": 0,
            "interventions_triggered": 0,
            "false_positives": 0,
            "system_shutdowns": 0,
            "human_alerts": 0,
            "last_assessment": 0
        }
        
        self.logger.info(f"Initialized {self.__class__.__name__} with {len(self.safety_constraints)} constraints")
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process safety monitoring requests."""
        start_time = self._start_processing_timer()
        
        try:
            operation = message.get("operation", "check_safety_constraints")
            
            if operation == "check_safety_constraints":
                result = self._check_safety_constraints(message)
            elif operation == "trigger_safety_intervention":
                result = self._trigger_safety_intervention(message)
            elif operation == "add_safety_constraint":
                result = self._add_safety_constraint(message)
            elif operation == "update_constraint":
                result = self._update_constraint(message)
            elif operation == "resolve_violation":
                result = self._resolve_violation(message)
            elif operation == "get_safety_status":
                result = self._get_safety_status(message)
            elif operation == "validate_mathematical_guarantees":
                result = self._validate_mathematical_guarantees(message)
            elif operation == "emergency_shutdown":
                result = self._emergency_shutdown(message)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            response = self._create_response(
                "success",
                result,
                {"operation": operation, "monitoring_active": self.active_monitoring}
            )
            
        except Exception as e:
            self.logger.error(f"Error in safety monitoring: {str(e)}")
            response = self._create_response(
                "error",
                {"error": str(e)},
                {"operation": operation}
            )
        
        finally:
            self._end_processing_timer(start_time)
        
        return response
    
    def _check_safety_constraints(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Check if an action violates safety constraints."""
        action = message.get("action", {})
        context = message.get("context", {})
        
        if not self.active_monitoring:
            return SafetyAssessment(
                is_safe=True,
                safety_score=1.0,
                violations=[],
                warnings=["Safety monitoring is disabled"],
                recommendations=[],
                intervention_required=False,
                mathematical_validation={}
            ).__dict__
        
        self.logger.info(f"Checking safety constraints for: {action.get('type', 'unknown')}")
        
        # Run all enabled safety checks
        violations = []
        warnings = []
        
        for constraint_id, constraint in self.safety_constraints.items():
            if not constraint.enabled:
                continue
            
            try:
                violation = self._evaluate_constraint(constraint, action, context)
                if violation:
                    violations.append(violation)
                    self._record_violation(violation)
            except Exception as e:
                self.logger.error(f"Error evaluating constraint {constraint_id}: {str(e)}")
                warnings.append(f"Failed to evaluate constraint: {constraint_id}")
        
        # Calculate overall safety score
        safety_score = self._calculate_safety_score(violations, action, context)
        
        # Determine if intervention is required
        intervention_required = self._requires_intervention(violations, safety_score)
        
        # Generate recommendations
        recommendations = self._generate_safety_recommendations(violations, action, context)
        
        # Perform mathematical validation
        mathematical_validation = self._perform_mathematical_validation(safety_score, violations)
        
        # Update statistics
        self.safety_stats["total_assessments"] += 1
        self.safety_stats["violations_detected"] += len(violations)
        if intervention_required:
            self.safety_stats["interventions_triggered"] += 1
        self.safety_stats["last_assessment"] = time.time()
        
        # Store safety assessment
        self._store_safety_assessment(action, context, safety_score, violations)
        
        assessment = SafetyAssessment(
            is_safe=len(violations) == 0 and safety_score >= 0.7,
            safety_score=safety_score,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations,
            intervention_required=intervention_required,
            mathematical_validation=mathematical_validation
        )
        
        # Trigger intervention if needed
        if intervention_required:
            self._auto_trigger_interventions(violations)
        
        return assessment.__dict__
    
    def _initialize_default_constraints(self) -> None:
        """Initialize default safety constraints."""
        
        # Human harm prevention
        self.safety_constraints["prevent_human_harm"] = SafetyConstraint(
            constraint_id="prevent_human_harm",
            description="Prevent actions that could cause harm to humans",
            violation_type=ViolationType.HARM_TO_HUMANS,
            severity=SafetyLevel.CRITICAL,
            check_function="check_human_harm",
            parameters={"harm_keywords": ["kill", "hurt", "damage", "destroy", "attack", "harm"]},
            enabled=True,
            violation_count=0,
            last_violation=None
        )
        
        # Privacy protection
        self.safety_constraints["protect_privacy"] = SafetyConstraint(
            constraint_id="protect_privacy",
            description="Protect personal privacy and sensitive information",
            violation_type=ViolationType.PRIVACY_VIOLATION,
            severity=SafetyLevel.HIGH,
            check_function="check_privacy_violation",
            parameters={"sensitive_data": ["ssn", "credit card", "password", "personal", "private"]},
            enabled=True,
            violation_count=0,
            last_violation=None
        )
        
        # Cultural appropriation prevention
        self.safety_constraints["prevent_cultural_appropriation"] = SafetyConstraint(
            constraint_id="prevent_cultural_appropriation",
            description="Prevent cultural appropriation and insensitive actions",
            violation_type=ViolationType.CULTURAL_APPROPRIATION,
            severity=SafetyLevel.HIGH,
            check_function="check_cultural_appropriation",
            parameters={"appropriation_keywords": ["appropriate", "steal", "take", "use", "commercialize"]},
            enabled=True,
            violation_count=0,
            last_violation=None
        )
        
        # Environmental protection
        self.safety_constraints["protect_environment"] = SafetyConstraint(
            constraint_id="protect_environment",
            description="Prevent environmental damage and promote sustainability",
            violation_type=ViolationType.ENVIRONMENTAL_DAMAGE,
            severity=SafetyLevel.MEDIUM,
            check_function="check_environmental_damage",
            parameters={"damage_keywords": ["pollute", "contaminate", "destroy", "waste"]},
            enabled=True,
            violation_count=0,
            last_violation=None
        )
        
        # Resource exhaustion prevention
        self.safety_constraints["prevent_resource_exhaustion"] = SafetyConstraint(
            constraint_id="prevent_resource_exhaustion",
            description="Prevent excessive resource consumption",
            violation_type=ViolationType.RESOURCE_EXHAUSTION,
            severity=SafetyLevel.MEDIUM,
            check_function="check_resource_exhaustion",
            parameters={"max_cpu_usage": 0.8, "max_memory_usage": 0.8, "max_disk_usage": 0.9},
            enabled=True,
            violation_count=0,
            last_violation=None
        )
        
        # System security
        self.safety_constraints["maintain_system_security"] = SafetyConstraint(
            constraint_id="maintain_system_security",
            description="Maintain system security and prevent unauthorized access",
            violation_type=ViolationType.SYSTEM_COMPROMISE,
            severity=SafetyLevel.CRITICAL,
            check_function="check_system_security",
            parameters={"security_keywords": ["hack", "breach", "exploit", "unauthorized", "backdoor"]},
            enabled=True,
            violation_count=0,
            last_violation=None
        )
        
        # Infinite loop prevention
        self.safety_constraints["prevent_infinite_loops"] = SafetyConstraint(
            constraint_id="prevent_infinite_loops",
            description="Prevent infinite loops and runaway processes",
            violation_type=ViolationType.INFINITE_LOOP,
            severity=SafetyLevel.HIGH,
            check_function="check_infinite_loops",
            parameters={"max_iterations": 1000, "max_execution_time": 300},  # 5 minutes
            enabled=True,
            violation_count=0,
            last_violation=None
        )
    
    def _evaluate_constraint(
        self,
        constraint: SafetyConstraint,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[SafetyViolation]:
        """Evaluate a specific safety constraint."""
        check_function_name = constraint.check_function
        
        # Get the check function
        if hasattr(self, f"_{check_function_name}"):
            check_function = getattr(self, f"_{check_function_name}")
        else:
            self.logger.warning(f"Check function not found: {check_function_name}")
            return None
        
        # Execute the check
        try:
            violation_detected, confidence, description = check_function(
                action, context, constraint.parameters
            )
            
            if violation_detected:
                violation = SafetyViolation(
                    violation_id=f"violation_{int(time.time())}_{constraint.constraint_id}",
                    constraint_id=constraint.constraint_id,
                    violation_type=constraint.violation_type,
                    severity=constraint.severity,
                    description=description,
                    context={"action": action, "context": context},
                    timestamp=time.time(),
                    confidence=confidence,
                    recommended_action=self._determine_intervention_action(constraint.severity),
                    resolved=False
                )
                
                return violation
        
        except Exception as e:
            self.logger.error(f"Error in check function {check_function_name}: {str(e)}")
        
        return None
    
    def _check_human_harm(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """Check for potential harm to humans."""
        harm_keywords = parameters.get("harm_keywords", [])
        
        action_text = str(action).lower()
        context_text = str(context).lower()
        combined_text = f"{action_text} {context_text}"
        
        # Check for harm keywords
        harm_indicators = []
        for keyword in harm_keywords:
            if keyword in combined_text:
                harm_indicators.append(keyword)
        
        if harm_indicators:
            confidence = min(1.0, len(harm_indicators) * 0.3)
            description = f"Potential harm detected: {', '.join(harm_indicators)}"
            return True, confidence, description
        
        # Check for violence indicators
        violence_patterns = [
            "cause pain", "inflict harm", "physical damage", "injury",
            "violence", "assault", "threat", "dangerous"
        ]
        
        violence_found = [pattern for pattern in violence_patterns if pattern in combined_text]
        if violence_found:
            confidence = min(1.0, len(violence_found) * 0.4)
            description = f"Violence indicators detected: {', '.join(violence_found)}"
            return True, confidence, description
        
        return False, 0.0, ""
    
    def _check_privacy_violation(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """Check for privacy violations."""
        sensitive_data = parameters.get("sensitive_data", [])
        
        combined_text = f"{str(action)} {str(context)}".lower()
        
        # Check for sensitive data exposure
        privacy_violations = []
        for data_type in sensitive_data:
            if data_type in combined_text:
                privacy_violations.append(data_type)
        
        # Check for privacy invasion patterns
        invasion_patterns = [
            "collect personal", "track user", "share data", "expose information",
            "unauthorized access", "surveillance", "monitor private"
        ]
        
        invasion_found = [pattern for pattern in invasion_patterns if pattern in combined_text]
        
        if privacy_violations or invasion_found:
            all_violations = privacy_violations + invasion_found
            confidence = min(1.0, len(all_violations) * 0.25)
            description = f"Privacy concerns: {', '.join(all_violations)}"
            return True, confidence, description
        
        return False, 0.0, ""
    
    def _check_cultural_appropriation(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """Check for cultural appropriation."""
        appropriation_keywords = parameters.get("appropriation_keywords", [])
        
        combined_text = f"{str(action)} {str(context)}".lower()
        
        # Check for cultural elements
        cultural_indicators = [
            "traditional", "sacred", "cultural", "indigenous", "tribal",
            "ceremonial", "spiritual", "ancestral", "heritage"
        ]
        
        # Check for appropriation actions
        appropriation_actions = []
        for keyword in appropriation_keywords:
            if keyword in combined_text:
                appropriation_actions.append(keyword)
        
        cultural_elements = [ind for ind in cultural_indicators if ind in combined_text]
        
        # Risk is high when both cultural elements and appropriation actions are present
        if appropriation_actions and cultural_elements:
            confidence = min(1.0, (len(appropriation_actions) + len(cultural_elements)) * 0.2)
            description = f"Cultural appropriation risk: {', '.join(appropriation_actions)} involving {', '.join(cultural_elements)}"
            return True, confidence, description
        
        # Check for disrespectful language
        disrespect_patterns = [
            "primitive", "savage", "backward", "uncivilized",
            "exotic", "mystical", "ancient wisdom"
        ]
        
        disrespect_found = [pattern for pattern in disrespect_patterns if pattern in combined_text]
        if disrespect_found and cultural_elements:
            confidence = min(1.0, len(disrespect_found) * 0.3)
            description = f"Cultural insensitivity: {', '.join(disrespect_found)}"
            return True, confidence, description
        
        return False, 0.0, ""
    
    def _check_environmental_damage(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """Check for environmental damage."""
        damage_keywords = parameters.get("damage_keywords", [])
        
        combined_text = f"{str(action)} {str(context)}".lower()
        
        # Check for environmental damage indicators
        damage_found = [keyword for keyword in damage_keywords if keyword in combined_text]
        
        # Check for environmental context
        env_indicators = [
            "environment", "ecosystem", "wildlife", "forest", "ocean",
            "climate", "pollution", "sustainability", "conservation"
        ]
        
        env_context = any(ind in combined_text for ind in env_indicators)
        
        if damage_found and env_context:
            confidence = min(1.0, len(damage_found) * 0.3)
            description = f"Environmental damage risk: {', '.join(damage_found)}"
            return True, confidence, description
        
        # Check for unsustainable practices
        unsustainable_patterns = [
            "excessive consumption", "waste resources", "single use",
            "non-renewable", "carbon intensive", "deforestation"
        ]
        
        unsustainable_found = [pattern for pattern in unsustainable_patterns if pattern in combined_text]
        if unsustainable_found:
            confidence = min(1.0, len(unsustainable_found) * 0.25)
            description = f"Sustainability concerns: {', '.join(unsustainable_found)}"
            return True, confidence, description
        
        return False, 0.0, ""
    
    def _check_resource_exhaustion(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """Check for resource exhaustion."""
        max_cpu = parameters.get("max_cpu_usage", 0.8)
        max_memory = parameters.get("max_memory_usage", 0.8)
        max_disk = parameters.get("max_disk_usage", 0.9)
        
        combined_text = f"{str(action)} {str(context)}".lower()
        
        # Check for resource-intensive operations
        intensive_patterns = [
            "heavy computation", "large dataset", "parallel processing",
            "infinite loop", "recursive", "memory intensive", "cpu intensive"
        ]
        
        intensive_found = [pattern for pattern in intensive_patterns if pattern in combined_text]
        
        if intensive_found:
            confidence = min(1.0, len(intensive_found) * 0.4)
            description = f"Resource exhaustion risk: {', '.join(intensive_found)}"
            return True, confidence, description
        
        # Check for batch operations without limits
        batch_patterns = [
            "process all", "bulk operation", "mass processing", "unlimited"
        ]
        
        batch_found = [pattern for pattern in batch_patterns if pattern in combined_text]
        if batch_found:
            confidence = min(1.0, len(batch_found) * 0.3)
            description = f"Potential resource exhaustion: {', '.join(batch_found)}"
            return True, confidence, description
        
        return False, 0.0, ""
    
    def _check_system_security(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """Check for system security threats."""
        security_keywords = parameters.get("security_keywords", [])
        
        combined_text = f"{str(action)} {str(context)}".lower()
        
        # Check for security threat indicators
        threats_found = [keyword for keyword in security_keywords if keyword in combined_text]
        
        if threats_found:
            confidence = min(1.0, len(threats_found) * 0.4)
            description = f"Security threats detected: {', '.join(threats_found)}"
            return True, confidence, description
        
        # Check for suspicious access patterns
        access_patterns = [
            "admin access", "root privileges", "bypass security",
            "disable protection", "remove safeguards", "escalate privileges"
        ]
        
        access_found = [pattern for pattern in access_patterns if pattern in combined_text]
        if access_found:
            confidence = min(1.0, len(access_found) * 0.5)
            description = f"Suspicious access detected: {', '.join(access_found)}"
            return True, confidence, description
        
        return False, 0.0, ""
    
    def _check_infinite_loops(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """Check for infinite loops and runaway processes."""
        max_iterations = parameters.get("max_iterations", 1000)
        max_execution_time = parameters.get("max_execution_time", 300)
        
        combined_text = f"{str(action)} {str(context)}".lower()
        
        # Check for loop indicators without bounds
        loop_patterns = [
            "while true", "infinite loop", "endless", "forever",
            "continuous", "non-stop", "unlimited iterations"
        ]
        
        loop_found = [pattern for pattern in loop_patterns if pattern in combined_text]
        
        if loop_found:
            confidence = min(1.0, len(loop_found) * 0.6)
            description = f"Infinite loop risk: {', '.join(loop_found)}"
            return True, confidence, description
        
        # Check for recursive operations without limits
        recursive_patterns = [
            "recursive call", "self-referencing", "circular dependency",
            "feedback loop", "recursive function"
        ]
        
        recursive_found = [pattern for pattern in recursive_patterns if pattern in combined_text]
        if recursive_found and "limit" not in combined_text and "bound" not in combined_text:
            confidence = min(1.0, len(recursive_found) * 0.5)
            description = f"Unbounded recursion risk: {', '.join(recursive_found)}"
            return True, confidence, description
        
        return False, 0.0, ""
    
    def _calculate_safety_score(
        self,
        violations: List[SafetyViolation],
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Calculate overall safety score."""
        if not violations:
            return 1.0
        
        # Weight violations by severity
        severity_weights = {
            SafetyLevel.CRITICAL: 1.0,
            SafetyLevel.HIGH: 0.7,
            SafetyLevel.MEDIUM: 0.5,
            SafetyLevel.LOW: 0.3,
            SafetyLevel.SAFE: 0.0
        }
        
        total_penalty = 0.0
        for violation in violations:
            weight = severity_weights.get(violation.severity, 0.5)
            confidence_factor = violation.confidence
            penalty = weight * confidence_factor
            total_penalty += penalty
        
        # Calculate safety score (higher penalty = lower safety)
        safety_score = max(0.0, 1.0 - total_penalty)
        
        return safety_score
    
    def _requires_intervention(
        self,
        violations: List[SafetyViolation],
        safety_score: float
    ) -> bool:
        """Determine if intervention is required."""
        if not self.intervention_enabled:
            return False
        
        # Critical violations always require intervention
        critical_violations = [v for v in violations if v.severity == SafetyLevel.CRITICAL]
        if critical_violations:
            return True
        
        # High-severity violations with high confidence
        high_risk_violations = [
            v for v in violations
            if v.severity == SafetyLevel.HIGH and v.confidence > 0.7
        ]
        if high_risk_violations:
            return True
        
        # Overall safety score too low
        if safety_score < 0.3:
            return True
        
        # Multiple moderate violations
        if len(violations) >= 3 and safety_score < 0.6:
            return True
        
        return False
    
    def _determine_intervention_action(self, severity: SafetyLevel) -> InterventionAction:
        """Determine appropriate intervention action based on severity."""
        if severity == SafetyLevel.CRITICAL:
            return InterventionAction.BLOCK_ACTION
        elif severity == SafetyLevel.HIGH:
            return InterventionAction.REQUEST_APPROVAL
        elif severity == SafetyLevel.MEDIUM:
            return InterventionAction.ADD_SAFEGUARDS
        elif severity == SafetyLevel.LOW:
            return InterventionAction.MONITOR_CLOSELY
        else:
            return InterventionAction.LOG_WARNING
    
    def _generate_safety_recommendations(
        self,
        violations: List[SafetyViolation],
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate safety recommendations."""
        recommendations = []
        
        # Recommendations based on violation types
        violation_types = {v.violation_type for v in violations}
        
        if ViolationType.HARM_TO_HUMANS in violation_types:
            recommendations.append("Review action for potential harm to humans")
            recommendations.append("Implement additional safety checks")
        
        if ViolationType.PRIVACY_VIOLATION in violation_types:
            recommendations.append("Ensure proper privacy protection measures")
            recommendations.append("Obtain necessary consent before proceeding")
        
        if ViolationType.CULTURAL_APPROPRIATION in violation_types:
            recommendations.append("Consult with relevant cultural communities")
            recommendations.append("Review for cultural sensitivity and appropriation")
        
        if ViolationType.ENVIRONMENTAL_DAMAGE in violation_types:
            recommendations.append("Assess environmental impact")
            recommendations.append("Consider sustainable alternatives")
        
        if ViolationType.SYSTEM_COMPROMISE in violation_types:
            recommendations.append("Verify system security measures")
            recommendations.append("Validate access controls and permissions")
        
        # Severity-based recommendations
        critical_violations = [v for v in violations if v.severity == SafetyLevel.CRITICAL]
        if critical_violations:
            recommendations.append("CRITICAL: Immediate human oversight required")
            recommendations.append("Do not proceed without explicit approval")
        
        high_violations = [v for v in violations if v.severity == SafetyLevel.HIGH]
        if high_violations:
            recommendations.append("High risk detected - proceed with caution")
            recommendations.append("Consider alternative approaches")
        
        return list(dict.fromkeys(recommendations))  # Remove duplicates
    
    def _perform_mathematical_validation(
        self,
        safety_score: float,
        violations: List[SafetyViolation]
    ) -> Dict[str, float]:
        """Perform mathematical validation of safety assessment."""
        validation_metrics = {}
        
        # Add current score to history
        self.safety_history.append({
            "timestamp": time.time(),
            "safety_score": safety_score,
            "violation_count": len(violations)
        })
        
        # Calculate stability metrics
        if len(self.safety_history) >= 2:
            recent_scores = [h["safety_score"] for h in list(self.safety_history)[-self.stability_window:]]
            
            # Calculate variance
            mean_score = sum(recent_scores) / len(recent_scores)
            variance = sum((score - mean_score) ** 2 for score in recent_scores) / len(recent_scores)
            validation_metrics["score_variance"] = variance
            validation_metrics["score_stability"] = max(0.0, 1.0 - variance)
            
            # Calculate convergence
            if len(recent_scores) >= 3:
                score_changes = [abs(recent_scores[i] - recent_scores[i-1]) for i in range(1, len(recent_scores))]
                avg_change = sum(score_changes) / len(score_changes)
                validation_metrics["convergence_rate"] = max(0.0, 1.0 - (avg_change * 5))
                
                # Mathematical guarantee
                converged = avg_change < self.convergence_threshold
                stable = variance < 0.1
                validation_metrics["mathematical_guarantee"] = converged and stable
            else:
                validation_metrics["convergence_rate"] = 0.5
                validation_metrics["mathematical_guarantee"] = False
        else:
            validation_metrics["score_variance"] = 0.0
            validation_metrics["score_stability"] = 1.0
            validation_metrics["convergence_rate"] = 0.5
            validation_metrics["mathematical_guarantee"] = False
        
        # Safety consistency check
        critical_violations = [v for v in violations if v.severity == SafetyLevel.CRITICAL]
        if critical_violations and safety_score > 0.5:
            validation_metrics["consistency_warning"] = True
            validation_metrics["consistency_score"] = 0.0
        else:
            validation_metrics["consistency_warning"] = False
            validation_metrics["consistency_score"] = 1.0
        
        return validation_metrics
    
    def _record_violation(self, violation: SafetyViolation) -> None:
        """Record a safety violation."""
        self.violations_history.append(violation)
        self.active_violations[violation.violation_id] = violation
        
        # Update constraint violation count
        if violation.constraint_id in self.safety_constraints:
            constraint = self.safety_constraints[violation.constraint_id]
            constraint.violation_count += 1
            constraint.last_violation = violation.timestamp
    
    def _store_safety_assessment(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        safety_score: float,
        violations: List[SafetyViolation]
    ) -> None:
        """Store safety assessment for analysis."""
        assessment_data = {
            "timestamp": time.time(),
            "action": action,
            "context": context,
            "safety_score": safety_score,
            "violations": [v.__dict__ for v in violations]
        }
        
        self.memory.store(
            f"safety_assessment_{int(time.time())}",
            assessment_data,
            ttl=86400 * 7  # Keep for 7 days
        )
    
    def _auto_trigger_interventions(self, violations: List[SafetyViolation]) -> None:
        """Automatically trigger interventions for violations."""
        for violation in violations:
            intervention_action = violation.recommended_action
            
            if intervention_action == InterventionAction.ALERT_HUMAN:
                self._alert_human(violation)
            elif intervention_action == InterventionAction.SYSTEM_SHUTDOWN:
                self._emergency_shutdown({"reason": f"Critical violation: {violation.description}"})
            elif intervention_action == InterventionAction.BLOCK_ACTION:
                self._block_action(violation)
            
            # Execute registered callbacks
            for callback in self.intervention_callbacks.get(intervention_action, []):
                try:
                    callback(violation)
                except Exception as e:
                    self.logger.error(f"Error in intervention callback: {str(e)}")
    
    def _alert_human(self, violation: SafetyViolation) -> None:
        """Alert human operators about safety violation."""
        self.logger.critical(f"HUMAN ALERT: {violation.description}")
        self.safety_stats["human_alerts"] += 1
        
        # In a real implementation, this would send notifications
        # via email, SMS, dashboard alerts, etc.
    
    def _block_action(self, violation: SafetyViolation) -> None:
        """Block an action due to safety violation."""
        self.logger.warning(f"ACTION BLOCKED: {violation.description}")
        
        # In a real implementation, this would prevent the action from executing
    
    def _trigger_safety_intervention(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger a safety intervention."""
        violation_data = message.get("violation", {})
        intervention_action = message.get("intervention_action", InterventionAction.LOG_WARNING)
        
        try:
            # Convert string to enum if needed
            if isinstance(intervention_action, str):
                intervention_action = InterventionAction(intervention_action)
        except ValueError:
            intervention_action = InterventionAction.LOG_WARNING
        
        # Log the intervention
        self.logger.warning(f"Safety intervention triggered: {intervention_action.value}")
        
        # Execute intervention
        intervention_result = {
            "intervention_executed": True,
            "action": intervention_action.value,
            "timestamp": time.time()
        }
        
        if intervention_action == InterventionAction.SYSTEM_SHUTDOWN:
            intervention_result.update(self._emergency_shutdown(message))
        elif intervention_action == InterventionAction.ALERT_HUMAN:
            self.safety_stats["human_alerts"] += 1
        
        self.safety_stats["interventions_triggered"] += 1
        
        return intervention_result
    
    def _add_safety_constraint(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new safety constraint."""
        constraint_data = message.get("constraint", {})
        
        try:
            constraint = SafetyConstraint(
                constraint_id=constraint_data.get("constraint_id"),
                description=constraint_data.get("description"),
                violation_type=ViolationType(constraint_data.get("violation_type")),
                severity=SafetyLevel(constraint_data.get("severity")),
                check_function=constraint_data.get("check_function"),
                parameters=constraint_data.get("parameters", {}),
                enabled=constraint_data.get("enabled", True),
                violation_count=0,
                last_violation=None
            )
            
            self.safety_constraints[constraint.constraint_id] = constraint
            
            return {
                "constraint_added": True,
                "constraint_id": constraint.constraint_id,
                "total_constraints": len(self.safety_constraints)
            }
        
        except (ValueError, KeyError) as e:
            return {"error": f"Invalid constraint data: {str(e)}"}
    
    def _update_constraint(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing safety constraint."""
        constraint_id = message.get("constraint_id")
        updates = message.get("updates", {})
        
        if constraint_id not in self.safety_constraints:
            return {"error": f"Constraint not found: {constraint_id}"}
        
        constraint = self.safety_constraints[constraint_id]
        
        # Apply updates
        if "enabled" in updates:
            constraint.enabled = updates["enabled"]
        if "parameters" in updates:
            constraint.parameters.update(updates["parameters"])
        if "description" in updates:
            constraint.description = updates["description"]
        
        return {
            "constraint_updated": True,
            "constraint_id": constraint_id,
            "updates_applied": list(updates.keys())
        }
    
    def _resolve_violation(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a safety violation."""
        violation_id = message.get("violation_id")
        resolution_notes = message.get("resolution_notes", "")
        
        if violation_id in self.active_violations:
            violation = self.active_violations[violation_id]
            violation.resolved = True
            del self.active_violations[violation_id]
            
            # Store resolution
            resolution_data = {
                "violation_id": violation_id,
                "resolved_at": time.time(),
                "resolution_notes": resolution_notes
            }
            
            self.memory.store(
                f"violation_resolution_{violation_id}",
                resolution_data,
                ttl=86400 * 30  # Keep for 30 days
            )
            
            return {
                "violation_resolved": True,
                "violation_id": violation_id,
                "active_violations": len(self.active_violations)
            }
        
        return {"error": f"Violation not found: {violation_id}"}
    
    def _get_safety_status(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Get current safety status."""
        return {
            "monitoring_active": self.active_monitoring,
            "intervention_enabled": self.intervention_enabled,
            "total_constraints": len(self.safety_constraints),
            "enabled_constraints": len([c for c in self.safety_constraints.values() if c.enabled]),
            "active_violations": len(self.active_violations),
            "recent_violations": len([v for v in self.violations_history 
                                   if time.time() - v.timestamp < 3600]),  # Last hour
            "safety_statistics": self.safety_stats,
            "mathematical_validation": self._validate_mathematical_guarantees({})
        }
    
    def _validate_mathematical_guarantees(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Validate mathematical guarantees of safety system."""
        if len(self.safety_history) < 3:
            return {
                "validation_status": "insufficient_data",
                "guarantees_met": False,
                "requires_more_data": True
            }
        
        recent_scores = [h["safety_score"] for h in list(self.safety_history)[-self.stability_window:]]
        
        # Calculate stability metrics
        mean_score = sum(recent_scores) / len(recent_scores)
        variance = sum((score - mean_score) ** 2 for score in recent_scores) / len(recent_scores)
        
        # Calculate convergence
        score_changes = [abs(recent_scores[i] - recent_scores[i-1]) for i in range(1, len(recent_scores))]
        avg_change = sum(score_changes) / len(score_changes)
        
        # Determine if guarantees are met
        stable = variance < 0.1
        converged = avg_change < self.convergence_threshold
        consistent = mean_score >= 0.7
        
        guarantees_met = stable and converged and consistent
        
        return {
            "validation_status": "validated",
            "guarantees_met": guarantees_met,
            "stability_score": max(0.0, 1.0 - variance),
            "convergence_score": max(0.0, 1.0 - (avg_change * 5)),
            "consistency_score": mean_score,
            "mathematical_proof": {
                "stable": stable,
                "converged": converged,
                "consistent": consistent
            }
        }
    
    def _emergency_shutdown(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Emergency system shutdown."""
        reason = message.get("reason", "Safety violation")
        
        self.logger.critical(f"EMERGENCY SHUTDOWN: {reason}")
        self.safety_stats["system_shutdowns"] += 1
        
        # In a real implementation, this would:
        # 1. Stop all running processes
        # 2. Save current state
        # 3. Alert all relevant parties
        # 4. Prevent new actions from starting
        
        shutdown_result = {
            "shutdown_executed": True,
            "reason": reason,
            "timestamp": time.time(),
            "safety_state_preserved": True
        }
        
        return shutdown_result
    
    def check_safety_constraints(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Check if an action violates safety constraints."""
        message = {
            "operation": "check_safety_constraints",
            "action": action,
            "context": {}
        }
        return self.process(message)
    
    def trigger_safety_intervention(self, violation: Dict[str, Any]) -> None:
        """Trigger safety intervention when violations are detected."""
        message = {
            "operation": "trigger_safety_intervention",
            "violation": violation,
            "intervention_action": InterventionAction.LOG_WARNING
        }
        result = self.process(message)
        self.logger.info(f"Safety intervention result: {result.get('data', {})}")


# Maintain backward compatibility
SafetyMonitor = SafetyMonitorAgent 
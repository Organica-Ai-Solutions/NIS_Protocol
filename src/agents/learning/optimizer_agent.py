"""
Optimizer Agent

Handles parameter optimization for learning in the NIS Protocol.

Enhanced Features (v3):
- Complete self-audit integration with real-time integrity monitoring
- Mathematical validation of optimization operations with evidence-based metrics
- Comprehensive integrity oversight for all optimization outputs
- Auto-correction capabilities for optimization-related communications
"""

from typing import Dict, Any, Optional, List
import time
import numpy as np
import logging
from collections import defaultdict

from src.core.registry import NISAgent, NISLayer
from src.emotion.emotional_state import EmotionalState
from .learning_agent import LearningAgent

# Self-audit capabilities for real-time integrity monitoring
from src.utils.self_audit import self_audit_engine, ViolationType, IntegrityViolation

class OptimizerAgent(LearningAgent):
    """
    Agent responsible for optimizing learning parameters.
    
    This agent implements various optimization strategies to improve
    learning performance across the system.
    """
    
    def __init__(
        self,
        agent_id: str = "optimizer",
        description: str = "Optimizes learning parameters",
        emotional_state: Optional[EmotionalState] = None,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        decay: float = 0.0001,
        enable_self_audit: bool = True
    ):
        """
        Initialize the optimizer agent.
        
        Args:
            agent_id: Unique identifier for this agent
            description: Human-readable description of the agent's role
            emotional_state: Optional pre-configured emotional state
            learning_rate: Base learning rate
            momentum: Momentum factor for optimization
            decay: Learning rate decay factor
            enable_self_audit: Whether to enable real-time integrity monitoring
        """
        super().__init__(agent_id, description, emotional_state, learning_rate, enable_self_audit)
        self.momentum = momentum
        self.decay = decay
        self.velocity = {}  # Stores momentum updates
        self.iteration = 0
        
        # Update logging for optimizer-specific context
        self.logger = logging.getLogger(f"nis_optimizer_agent_{agent_id}")
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Optimizer Agent initialized with self-audit: {enable_self_audit}")

    def _update_parameters(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update parameters using optimization strategy.
        
        Args:
            message: Message with update parameters
                'params': Dictionary of parameters to update
                'gradients': Dictionary of parameter gradients
                
        Returns:
            Update operation result
        """
        params = message.get("params", {})
        gradients = message.get("gradients", {})
        
        if not params or not gradients:
            return {
                "status": "error",
                "error": "Missing params or gradients",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        # Apply momentum optimization
        updated_params = {}
        for param_name, param_value in params.items():
            if param_name not in gradients:
                continue
                
            # Initialize velocity if not exists
            if param_name not in self.velocity:
                self.velocity[param_name] = np.zeros_like(param_value)
            
            # Update velocity
            self.velocity[param_name] = (
                self.momentum * self.velocity[param_name] +
                self.learning_rate * gradients[param_name]
            )
            
            # Update parameter
            updated_params[param_name] = param_value - self.velocity[param_name]
        
        # Update iteration count and decay learning rate
        self.iteration += 1
        self.learning_rate *= (1.0 / (1.0 + self.decay * self.iteration))
        
        return {
            "status": "success",
            "updated_params": updated_params,
            "current_learning_rate": self.learning_rate,
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _get_parameters(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get current optimization parameters.
        
        Args:
            message: Message with parameter query
            
        Returns:
            Current parameter values
        """
        return {
            "status": "success",
            "parameters": {
                "learning_rate": self.learning_rate,
                "momentum": self.momentum,
                "decay": self.decay,
                "iteration": self.iteration
            },
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _reset_parameters(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reset optimization parameters.
        
        Args:
            message: Message with reset parameters
            
        Returns:
            Reset operation result
        """
        # Reset to initial values
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.decay = 0.0001
        self.velocity = {}
        self.iteration = 0
        
        return {
            "status": "success",
            "message": "Parameters reset to initial values",
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def suggest_learning_rate(self, loss_history: List[float]) -> float:
        """
        Suggest an optimal learning rate based on loss history.
        
        Args:
            loss_history: List of recent loss values
            
        Returns:
            Suggested learning rate
        """
        if len(loss_history) < 2:
            return self.learning_rate
            
        # Calculate loss trend
        loss_diff = np.diff(loss_history)
        avg_diff = np.mean(loss_diff)
        
        if avg_diff > 0:  # Loss increasing
            return self.learning_rate * 0.5
        elif avg_diff < 0:  # Loss decreasing
            return self.learning_rate * 1.1
        else:
            return self.learning_rate 
    
    # ==================== OPTIMIZER-SPECIFIC SELF-AUDIT CAPABILITIES ====================
    
    def audit_optimization_output(self, output_text: str, operation: str = "", context: str = "") -> Dict[str, Any]:
        """
        Perform real-time integrity audit on optimization operation outputs.
        
        Args:
            output_text: Text output to audit
            operation: Optimization operation type (optimize, suggest_lr, etc.)
            context: Additional context for the audit
            
        Returns:
            Audit results with violations and integrity score
        """
        if not self.enable_self_audit:
            return {'integrity_score': 100.0, 'violations': [], 'total_violations': 0}
        
        self.logger.info(f"Performing self-audit on optimization output for operation: {operation}")
        
        # Use proven audit engine
        audit_context = f"optimizer:{operation}:{context}" if context else f"optimizer:{operation}"
        violations = self_audit_engine.audit_text(output_text, audit_context)
        integrity_score = self_audit_engine.get_integrity_score(output_text)
        
        # Log violations for optimizer-specific analysis
        if violations:
            self.logger.warning(f"Detected {len(violations)} integrity violations in optimization output")
            for violation in violations:
                self.logger.warning(f"  - {violation.severity}: {violation.text} -> {violation.suggested_replacement}")
        
        return {
            'violations': violations,
            'integrity_score': integrity_score,
            'total_violations': len(violations),
            'violation_breakdown': self._categorize_optimizer_violations(violations),
            'operation': operation,
            'audit_timestamp': time.time()
        }
    
    def get_optimizer_integrity_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimizer integrity report"""
        if not self.enable_self_audit:
            return {'status': 'SELF_AUDIT_DISABLED'}
        
        # Get base learning integrity report
        base_report = super().get_learning_integrity_report()
        
        # Add optimizer-specific metrics
        optimizer_metrics = {
            'optimization_parameters': {
                'learning_rate': self.learning_rate,
                'momentum': self.momentum,
                'decay': self.decay,
                'iteration': self.iteration
            },
            'velocity_states': len(self.velocity),
            'optimization_active': self.iteration > 0
        }
        
        # Combine reports
        optimizer_report = {
            'optimizer_agent_id': self.agent_id,
            'optimizer_metrics': optimizer_metrics,
            'base_learning_report': base_report,
            'report_timestamp': time.time()
        }
        
        return optimizer_report
    
    def validate_optimization_parameters(self) -> Dict[str, Any]:
        """Validate optimization parameters for mathematical consistency"""
        validation_results = {
            'valid': True,
            'warnings': [],
            'recommendations': []
        }
        
        # Validate learning rate
        if self.learning_rate <= 0 or self.learning_rate > 1:
            validation_results['valid'] = False
            validation_results['warnings'].append(f"Learning rate {self.learning_rate} is outside typical range (0, 1]")
            validation_results['recommendations'].append("Consider adjusting learning rate to a value between 0.001 and 0.1")
        
        # Validate momentum
        if self.momentum < 0 or self.momentum >= 1:
            validation_results['valid'] = False
            validation_results['warnings'].append(f"Momentum {self.momentum} is outside typical range [0, 1)")
            validation_results['recommendations'].append("Consider adjusting momentum to a value between 0.8 and 0.99")
        
        # Validate decay
        if self.decay < 0:
            validation_results['valid'] = False
            validation_results['warnings'].append(f"Decay {self.decay} cannot be negative")
            validation_results['recommendations'].append("Set decay to a small positive value like 0.0001")
        
        # Add mathematical consistency checks
        if self.learning_rate > 0.1 and self.momentum > 0.9:
            validation_results['warnings'].append("High learning rate with high momentum may cause instability")
            validation_results['recommendations'].append("Consider reducing either learning rate or momentum")
        
        return validation_results
    
    def _categorize_optimizer_violations(self, violations: List[IntegrityViolation]) -> Dict[str, int]:
        """Categorize integrity violations specific to optimizer operations"""
        categories = defaultdict(int)
        
        for violation in violations:
            categories[violation.violation_type.value] += 1
        
        return dict(categories)
    
    def _monitor_optimization_output_integrity(self, output_text: str, operation: str = "") -> str:
        """
        Internal method to monitor and potentially correct optimization output integrity.
        
        Args:
            output_text: Output to monitor
            operation: Optimization operation type
            
        Returns:
            Potentially corrected output
        """
        if not getattr(self, 'integrity_monitoring_enabled', False):
            return output_text
        
        # Perform audit
        audit_result = self.audit_optimization_output(output_text, operation)
        
        # Update monitoring metrics
        if hasattr(self, 'integrity_metrics'):
            self.integrity_metrics['total_outputs_monitored'] += 1
            self.integrity_metrics['total_violations_detected'] += audit_result['total_violations']
        
        # Auto-correct if violations detected
        if audit_result['violations']:
            correction_result = self.auto_correct_learning_output(output_text, operation)
            
            self.logger.info(f"Auto-corrected optimization output: {len(audit_result['violations'])} violations fixed")
            
            return correction_result['corrected_text']
        
        return output_text 
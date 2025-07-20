"""
Memory Agent

Stores and retrieves information for use by other agents in the system.
Analogous to the hippocampus in the brain, responsible for memory formation and recall.

Enhanced Features (v3):
- Complete self-audit integration with real-time integrity monitoring
- Mathematical validation of memory operations with evidence-based metrics
- Comprehensive integrity oversight for all memory outputs
- Auto-correction capabilities for memory-related communications
"""

from typing import Dict, Any, List, Optional, Union
import time
import json
import os
import datetime
import logging
from collections import deque, defaultdict

from src.core.registry import NISAgent, NISLayer, NISRegistry
from src.emotion.emotional_state import EmotionalState, EmotionalDimension

# Integrity metrics for actual calculations
from src.utils.integrity_metrics import (
    calculate_confidence, create_default_confidence_factors, ConfidenceFactors
)

# Self-audit capabilities for real-time integrity monitoring
from src.utils.self_audit import self_audit_engine, ViolationType, IntegrityViolation


class MemoryAgent(NISAgent):
    """
    Agent that manages storage and retrieval of information.
    
    The Memory Agent is responsible for:
    - Storing information from other agents
    - Retrieving relevant information based on queries
    - Maintaining both short-term and long-term memory
    - Providing context for decision-making
    """
    
    def __init__(
        self,
        agent_id: str = "memory",
        description: str = "Stores and retrieves information for the system",
        emotional_state: Optional[EmotionalState] = None,
        storage_path: Optional[str] = None,
        short_term_capacity: int = 1000,
        enable_self_audit: bool = True
    ):
        """
        Initialize a new Memory Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            description: Human-readable description of the agent's role
            emotional_state: Optional pre-configured emotional state
            storage_path: Path to store persistent memory data
            short_term_capacity: Maximum number of items in short-term memory
            enable_self_audit: Whether to enable real-time integrity monitoring
        """
        super().__init__(agent_id, NISLayer.MEMORY, description)
        self.emotional_state = emotional_state or EmotionalState()
        
        # Short-term memory (in-memory cache)
        self.short_term = deque(maxlen=short_term_capacity)
        
        # Long-term memory (persistent storage)
        self.storage_path = storage_path
        if storage_path and not os.path.exists(storage_path):
            os.makedirs(storage_path, exist_ok=True)
        
        # Set up self-audit integration
        self.enable_self_audit = enable_self_audit
        self.integrity_monitoring_enabled = enable_self_audit
        self.integrity_metrics = {
            'monitoring_start_time': time.time(),
            'total_outputs_monitored': 0,
            'total_violations_detected': 0,
            'auto_corrections_applied': 0,
            'average_integrity_score': 100.0
        }
        
        # Initialize confidence factors for mathematical validation
        self.confidence_factors = create_default_confidence_factors()
        
        # Set up logging
        self.logger = logging.getLogger(f"nis_memory_agent_{agent_id}")
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Memory Agent initialized with self-audit: {enable_self_audit}")
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a memory-related request with integrated self-audit monitoring.
        
        Args:
            message: Message containing memory operation
                'operation': 'store', 'retrieve', 'query', 'forget'
                'data': Data to store (for 'store' operation)
                'query': Query parameters (for 'retrieve' or 'query' operations)
                'memory_id': ID of memory to forget (for 'forget' operation)
        
        Returns:
            Result of the memory operation with integrity monitoring
        """
        if not self._validate_message(message):
            error_response = {
                "status": "error",
                "error": "Invalid message format or missing required fields",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
            
            # Apply self-audit monitoring to error response
            if self.enable_self_audit:
                error_text = error_response.get("error", "")
                error_response["error"] = self._monitor_memory_output_integrity(error_text, "validation_error")
            
            return error_response

        operation = message.get("operation", "").lower()
        
        # Route to appropriate handler with self-audit monitoring
        try:
            if operation == "store":
                result = self._store_memory(message)
            elif operation == "retrieve":
                result = self._retrieve_memory(message)
            elif operation == "query":
                result = self._query_memory(message)
            elif operation == "forget":
                result = self._forget_memory(message)
            else:
                result = {
                    "status": "error",
                    "error": f"Unknown operation: {operation}",
                    "agent_id": self.agent_id,
                    "timestamp": time.time()
                }
            
            # Apply self-audit monitoring to all responses
            if self.enable_self_audit and result:
                result = self._apply_memory_integrity_monitoring(result, operation)
            
            return result
            
        except Exception as e:
            error_response = {
                "status": "error", 
                "error": f"Memory operation failed: {str(e)}",
                "operation": operation,
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
            
            # Apply self-audit monitoring to exception response
            if self.enable_self_audit:
                error_text = error_response.get("error", "")
                error_response["error"] = self._monitor_memory_output_integrity(error_text, f"{operation}_error")
            
            self.logger.error(f"Memory operation {operation} failed: {str(e)}")
            
            return error_response
    
    def _validate_message(self, message: Dict[str, Any]) -> bool:
        """
        Validate incoming message format.
        
        Args:
            message: The message to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(message, dict):
            return False
        
        if "operation" not in message:
            return False
        
        operation = message.get("operation", "").lower()
        
        if operation == "store" and "data" not in message:
            return False
        
        if operation in ["retrieve", "query"] and "query" not in message:
            return False
        
        if operation == "forget" and "memory_id" not in message:
            return False
        
        return True
    
    def _store_memory(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a new memory.
        
        Args:
            message: Message with data to store
            
        Returns:
            Storage result
        """
        data = message.get("data", {})
        
        # Generate memory ID if not provided
        memory_id = message.get("memory_id", f"mem_{time.time()}")
        
        # Prepare memory object
        memory = {
            "memory_id": memory_id,
            "timestamp": time.time(),
            "created": datetime.datetime.now().isoformat(),
            "data": data,
            "tags": message.get("tags", []),
            "importance": message.get("importance", 0.5),
            "source_agent": message.get("source_agent", "unknown")
        }
        
        # Store in short-term memory
        self.short_term.append(memory)
        
        # Store in long-term memory if path is configured
        if self.storage_path:
            self._persist_memory(memory)
        
        # Update emotional state based on importance
        if memory.get("importance", 0.5) > 0.7:
            self.emotional_state.update(EmotionalDimension.INTEREST.value, 0.7)
        
        return {
            "status": "success",
            "memory_id": memory_id,
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _retrieve_memory(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve a specific memory by ID.
        
        Args:
            message: Message with memory ID to retrieve
            
        Returns:
            Retrieved memory or error
        """
        memory_id = message.get("query", {}).get("memory_id")
        
        if not memory_id:
            return {
                "status": "error",
                "error": "No memory_id provided",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        # First check short-term memory
        for memory in self.short_term:
            if memory.get("memory_id") == memory_id:
                return {
                    "status": "success",
                    "memory": memory,
                    "source": "short_term",
                    "agent_id": self.agent_id,
                    "timestamp": time.time()
                }
        
        # Then check long-term memory
        if self.storage_path:
            memory = self._load_memory(memory_id)
            if memory:
                # Add to short-term memory for faster future access
                self.short_term.append(memory)
                
                return {
                    "status": "success",
                    "memory": memory,
                    "source": "long_term",
                    "agent_id": self.agent_id,
                    "timestamp": time.time()
                }
        
        return {
            "status": "error",
            "error": f"Memory not found: {memory_id}",
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _query_memory(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query memories based on criteria.
        
        Args:
            message: Message with query parameters
            
        Returns:
            List of matching memories
        """
        query = message.get("query", {})
        
        # Extract query parameters with defaults
        max_results = query.get("max_results", 10)
        start_time = query.get("start_time", 0)
        end_time = query.get("end_time", time.time())
        tags = query.get("tags", [])
        min_importance = query.get("min_importance", 0.0)
        source_agent = query.get("source_agent", None)
        
        results = []
        
        # Search short-term memory
        for memory in self.short_term:
            if self._memory_matches_query(memory, start_time, end_time, tags, min_importance, source_agent):
                results.append(memory)
                
                if len(results) >= max_results:
                    break
        
        # If not enough results and we have long-term storage, search there
        if len(results) < max_results and self.storage_path:
            # This is a simplified version; a real implementation would use 
            # a database or vector store for efficient querying
            long_term_results = self._query_long_term(
                start_time, end_time, tags, min_importance, source_agent, max_results - len(results)
            )
            results.extend(long_term_results)
        
        return {
            "status": "success",
            "results": results,
            "result_count": len(results),
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _forget_memory(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove a memory from storage.
        
        Args:
            message: Message with memory ID to forget
            
        Returns:
            Result of the operation
        """
        memory_id = message.get("memory_id")
        
        # Remove from short-term memory
        self.short_term = deque([m for m in self.short_term if m.get("memory_id") != memory_id], 
                               maxlen=self.short_term.maxlen)
        
        # Remove from long-term memory
        if self.storage_path:
            memory_path = os.path.join(self.storage_path, f"{memory_id}.json")
            if os.path.exists(memory_path):
                os.remove(memory_path)
        
        return {
            "status": "success",
            "memory_id": memory_id,
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _persist_memory(self, memory: Dict[str, Any]) -> None:
        """
        Save memory to persistent storage.
        
        Args:
            memory: Memory object to persist
        """
        if not self.storage_path:
            return
        
        memory_id = memory.get("memory_id")
        memory_path = os.path.join(self.storage_path, f"{memory_id}.json")
        
        with open(memory_path, 'w') as f:
            json.dump(memory, f)
    
    def _load_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Load memory from persistent storage.
        
        Args:
            memory_id: ID of memory to load
            
        Returns:
            Memory object or None if not found
        """
        if not self.storage_path:
            return None
        
        memory_path = os.path.join(self.storage_path, f"{memory_id}.json")
        
        if not os.path.exists(memory_path):
            return None
        
        try:
            with open(memory_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def _memory_matches_query(self, 
                             memory: Dict[str, Any], 
                             start_time: float, 
                             end_time: float, 
                             tags: List[str], 
                             min_importance: float, 
                             source_agent: Optional[str]) -> bool:
        """
        Check if a memory matches query criteria.
        
        Args:
            memory: Memory to check
            start_time: Minimum timestamp
            end_time: Maximum timestamp
            tags: Required tags (any match)
            min_importance: Minimum importance value
            source_agent: Source agent filter
            
        Returns:
            True if memory matches criteria
        """
        # Check timestamp
        timestamp = memory.get("timestamp", 0)
        if timestamp < start_time or timestamp > end_time:
            return False
        
        # Check importance
        importance = memory.get("importance", 0.0)
        if importance < min_importance:
            return False
        
        # Check source agent
        if source_agent and memory.get("source_agent") != source_agent:
            return False
        
        # Check tags (any match)
        if tags:
            memory_tags = memory.get("tags", [])
            if not any(tag in memory_tags for tag in tags):
                return False
        
        return True
    
    def _query_long_term(self, 
                        start_time: float, 
                        end_time: float, 
                        tags: List[str], 
                        min_importance: float, 
                        source_agent: Optional[str], 
                        max_results: int) -> List[Dict[str, Any]]:
        """
        Query long-term memory storage.
        
        Args:
            start_time: Minimum timestamp
            end_time: Maximum timestamp
            tags: Required tags (any match)
            min_importance: Minimum importance value
            source_agent: Source agent filter
            max_results: Maximum number of results
            
        Returns:
            List of matching memories
        """
        if not self.storage_path:
            return []
        
        results = []
        
        # This is a simple implementation that scans all files
        # A real implementation would use a database or index
        try:
            memory_files = os.listdir(self.storage_path)
            for filename in memory_files:
                if not filename.endswith('.json'):
                    continue
                
                memory_path = os.path.join(self.storage_path, filename)
                
                try:
                    with open(memory_path, 'r') as f:
                        memory = json.load(f)
                        
                        if self._memory_matches_query(
                            memory, start_time, end_time, tags, min_importance, source_agent
                        ):
                            results.append(memory)
                            
                            if len(results) >= max_results:
                                break
                except (json.JSONDecodeError, IOError):
                    continue
        except OSError:
            pass
        
        return results 
    
    # ==================== SELF-AUDIT CAPABILITIES ====================
    
    def audit_memory_output(self, output_text: str, operation: str = "", context: str = "") -> Dict[str, Any]:
        """
        Perform real-time integrity audit on memory operation outputs.
        
        Args:
            output_text: Text output to audit
            operation: Memory operation type (store, retrieve, query, etc.)
            context: Additional context for the audit
            
        Returns:
            Audit results with violations and integrity score
        """
        if not self.enable_self_audit:
            return {'integrity_score': 100.0, 'violations': [], 'total_violations': 0}
        
        self.logger.info(f"Performing self-audit on memory output for operation: {operation}")
        
        # Use proven audit engine
        audit_context = f"memory:{operation}:{context}" if context else f"memory:{operation}"
        violations = self_audit_engine.audit_text(output_text, audit_context)
        integrity_score = self_audit_engine.get_integrity_score(output_text)
        
        # Log violations for memory-specific analysis
        if violations:
            self.logger.warning(f"Detected {len(violations)} integrity violations in memory output")
            for violation in violations:
                self.logger.warning(f"  - {violation.severity}: {violation.text} -> {violation.suggested_replacement}")
        
        return {
            'violations': violations,
            'integrity_score': integrity_score,
            'total_violations': len(violations),
            'violation_breakdown': self._categorize_memory_violations(violations),
            'operation': operation,
            'audit_timestamp': time.time()
        }
    
    def auto_correct_memory_output(self, output_text: str, operation: str = "") -> Dict[str, Any]:
        """
        Automatically correct integrity violations in memory outputs.
        
        Args:
            output_text: Text to correct
            operation: Memory operation type
            
        Returns:
            Corrected output with audit details
        """
        if not self.enable_self_audit:
            return {'corrected_text': output_text, 'violations_fixed': [], 'improvement': 0}
        
        self.logger.info(f"Performing self-correction on memory output for operation: {operation}")
        
        corrected_text, violations = self_audit_engine.auto_correct_text(output_text)
        
        # Calculate improvement metrics with mathematical validation
        original_score = self_audit_engine.get_integrity_score(output_text)
        corrected_score = self_audit_engine.get_integrity_score(corrected_text)
        improvement = calculate_confidence(corrected_score - original_score, self.confidence_factors)
        
        # Update integrity metrics
        if hasattr(self, 'integrity_metrics'):
            self.integrity_metrics['auto_corrections_applied'] += len(violations)
        
        return {
            'original_text': output_text,
            'corrected_text': corrected_text,
            'violations_fixed': violations,
            'original_integrity_score': original_score,
            'corrected_integrity_score': corrected_score,
            'improvement': improvement,
            'operation': operation,
            'correction_timestamp': time.time()
        }
    
    def analyze_memory_integrity_trends(self, time_window: int = 3600) -> Dict[str, Any]:
        """
        Analyze memory operation integrity trends for self-improvement.
        
        Args:
            time_window: Time window in seconds to analyze
            
        Returns:
            Memory integrity trend analysis with mathematical validation
        """
        if not self.enable_self_audit:
            return {'integrity_status': 'MONITORING_DISABLED'}
        
        self.logger.info(f"Analyzing memory integrity trends over {time_window} seconds")
        
        # Get integrity report from audit engine
        integrity_report = self_audit_engine.generate_integrity_report()
        
        # Calculate memory-specific metrics
        memory_metrics = {
            'short_term_utilization': len(self.short_term) / self.short_term.maxlen if self.short_term.maxlen else 0,
            'storage_path_configured': bool(self.storage_path),
            'total_stored_memories': len(self.short_term)
        }
        
        # Generate memory-specific recommendations
        recommendations = self._generate_memory_integrity_recommendations(
            integrity_report, memory_metrics
        )
        
        return {
            'integrity_status': integrity_report['integrity_status'],
            'total_violations': integrity_report['total_violations'],
            'memory_metrics': memory_metrics,
            'integrity_trend': self._calculate_memory_integrity_trend(),
            'recommendations': recommendations,
            'analysis_timestamp': time.time()
        }
    
    def get_memory_integrity_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory integrity report"""
        if not self.enable_self_audit:
            return {'status': 'SELF_AUDIT_DISABLED'}
        
        # Get basic integrity report
        base_report = self_audit_engine.generate_integrity_report()
        
        # Add memory-specific metrics
        memory_report = {
            'memory_agent_id': self.agent_id,
            'monitoring_enabled': self.integrity_monitoring_enabled,
            'memory_capacity_status': {
                'short_term': f"{len(self.short_term)}/{self.short_term.maxlen}" if self.short_term.maxlen else "unlimited"
            },
            'storage_configuration': {
                'path_configured': bool(self.storage_path),
                'path': self.storage_path or "in_memory_only"
            },
            'integrity_metrics': getattr(self, 'integrity_metrics', {}),
            'base_integrity_report': base_report,
            'report_timestamp': time.time()
        }
        
        return memory_report
    
    def _monitor_memory_output_integrity(self, output_text: str, operation: str = "") -> str:
        """
        Internal method to monitor and potentially correct memory output integrity.
        
        Args:
            output_text: Output to monitor
            operation: Memory operation type
            
        Returns:
            Potentially corrected output
        """
        if not getattr(self, 'integrity_monitoring_enabled', False):
            return output_text
        
        # Perform audit
        audit_result = self.audit_memory_output(output_text, operation)
        
        # Update monitoring metrics
        if hasattr(self, 'integrity_metrics'):
            self.integrity_metrics['total_outputs_monitored'] += 1
            self.integrity_metrics['total_violations_detected'] += audit_result['total_violations']
        
        # Auto-correct if violations detected
        if audit_result['violations']:
            correction_result = self.auto_correct_memory_output(output_text, operation)
            
            self.logger.info(f"Auto-corrected memory output: {len(audit_result['violations'])} violations fixed")
            
            return correction_result['corrected_text']
        
        return output_text
    
    def _apply_memory_integrity_monitoring(self, result: Dict[str, Any], operation: str) -> Dict[str, Any]:
        """Apply integrity monitoring to response data"""
        if not self.enable_self_audit or not result:
            return result
        
        # Monitor text fields in the response
        for key, value in result.items():
            if isinstance(value, str) and len(value) > 10:  # Only monitor substantial text
                monitored_text = self._monitor_memory_output_integrity(value, f"{operation}_{key}")
                if monitored_text != value:
                    result[key] = monitored_text
        
        return result
    
    def _categorize_memory_violations(self, violations: List[IntegrityViolation]) -> Dict[str, int]:
        """Categorize integrity violations specific to memory operations"""
        categories = defaultdict(int)
        
        for violation in violations:
            categories[violation.violation_type.value] += 1
        
        return dict(categories)
    
    def _generate_memory_integrity_recommendations(self, integrity_report: Dict[str, Any], memory_metrics: Dict[str, Any]) -> List[str]:
        """Generate memory-specific integrity improvement recommendations"""
        recommendations = []
        
        if integrity_report.get('total_violations', 0) > 5:
            recommendations.append("Consider implementing more rigorous input validation for memory operations")
        
        if memory_metrics.get('short_term_utilization', 0) > 0.9:
            recommendations.append("Short-term memory approaching capacity - consider increasing capacity or implementing cleanup")
        
        if not memory_metrics.get('storage_path_configured', False):
            recommendations.append("Configure persistent storage path for improved memory reliability")
        
        if len(recommendations) == 0:
            recommendations.append("Memory integrity status is excellent - maintain current practices")
        
        return recommendations
    
    def _calculate_memory_integrity_trend(self) -> Dict[str, Any]:
        """Calculate memory integrity trends with mathematical validation"""
        if not hasattr(self, 'integrity_metrics'):
            return {'trend': 'INSUFFICIENT_DATA'}
        
        monitoring_time = time.time() - self.integrity_metrics.get('monitoring_start_time', time.time())
        total_outputs = self.integrity_metrics.get('total_outputs_monitored', 0)
        total_violations = self.integrity_metrics.get('total_violations_detected', 0)
        
        if total_outputs == 0:
            return {'trend': 'NO_OUTPUTS_MONITORED'}
        
        violation_rate = total_violations / total_outputs
        violations_per_hour = (total_violations / monitoring_time) * 3600 if monitoring_time > 0 else 0
        
        # Calculate trend with mathematical validation
        trend_score = calculate_confidence(1.0 - violation_rate, self.confidence_factors)
        
        return {
            'trend': 'IMPROVING' if trend_score > 0.8 else 'STABLE' if trend_score > 0.6 else 'NEEDS_ATTENTION',
            'violation_rate': violation_rate,
            'violations_per_hour': violations_per_hour,
            'trend_score': trend_score,
            'monitoring_duration_hours': monitoring_time / 3600
        } 
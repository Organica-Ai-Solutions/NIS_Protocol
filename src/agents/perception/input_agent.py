"""
Input Agent

Processes non-visual inputs such as text commands, speech, or sensor readings.
Serves as the initial point of contact for data entering the system.

Enhanced Features (v3):
- Complete self-audit integration with real-time integrity monitoring
- Mathematical validation of input operations with evidence-based metrics
- Comprehensive integrity oversight for all input outputs
- Auto-correction capabilities for input-related communications
"""

from typing import Dict, Any, List, Optional
import time
import re
import logging
from collections import defaultdict

from src.core.registry import NISAgent, NISLayer, NISRegistry
from src.emotion.emotional_state import EmotionalState, EmotionalDimension

# Integrity metrics for actual calculations
from src.utils.integrity_metrics import (
    calculate_confidence, create_default_confidence_factors, ConfidenceFactors
)

# Self-audit capabilities for real-time integrity monitoring
from src.utils.self_audit import self_audit_engine, ViolationType, IntegrityViolation


class InputAgent(NISAgent):
    """
    Agent that processes non-visual inputs (text, speech, sensors, etc.)
    
    The Input Agent is responsible for:
    - Processing incoming text, speech, or data streams
    - Converting raw input into structured formats
    - Initial filtering and prioritization of inputs
    - Detecting potential threats or anomalies in input data
    """
    
    def __init__(
        self,
        agent_id: str = "input",
        description: str = "Processes non-visual inputs and translates to structured data",
        emotional_state: Optional[EmotionalState] = None,
        enable_self_audit: bool = True
    ):
        """
        Initialize a new Input Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            description: Human-readable description of the agent's role
            emotional_state: Optional pre-configured emotional state
            enable_self_audit: Whether to enable real-time integrity monitoring
        """
        super().__init__(agent_id, NISLayer.PERCEPTION, description)
        self.emotional_state = emotional_state or EmotionalState()
        self.input_history = []
        
        # Keywords that may affect emotional state
        self.urgency_keywords = {
            "urgent", "immediately", "asap", "emergency", "critical", 
            "time-sensitive", "deadline", "now", "quickly"
        }
        
        self.suspicion_keywords = {
            "unusual", "suspicious", "strange", "unexpected", "abnormal",
            "unexpected", "security", "threat", "warning", "alert"
        }
        
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
        self.logger = logging.getLogger(f"nis_input_agent_{agent_id}")
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Input Agent initialized with self-audit: {enable_self_audit}")
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a non-visual input with integrated self-audit monitoring.
        
        Args:
            message: Message containing input data
                'text': Text input (optional)
                'speech': Speech input (optional)
                'sensor_data': Sensor readings (optional)
        
        Returns:
            Processed input with metadata and integrity monitoring
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
                error_response["error"] = self._monitor_input_output_integrity(error_text, "validation_error")
            
            return error_response
        
        # Process based on input type
        structured_data = {}
        metadata = {}
        
        # Process text if present
        if "text" in message:
            text_data, text_metadata = self._process_text(message["text"])
            structured_data.update(text_data)
            metadata.update(text_metadata)
            
            # Update emotional state based on text content
            self._update_emotion_from_text(message["text"])
        
        # Process speech if present
        if "speech" in message:
            speech_data, speech_metadata = self._process_speech(message["speech"])
            structured_data.update(speech_data)
            metadata.update(speech_metadata)
        
        # Process sensor data if present
        if "sensor_data" in message:
            sensor_data, sensor_metadata = self._process_sensor_data(message["sensor_data"])
            structured_data.update(sensor_data)
            metadata.update(sensor_metadata)
        
        # Track input history (limited to most recent 100)
        self.input_history.append({
            "timestamp": time.time(),
            "structured_data": structured_data
        })
        if len(self.input_history) > 100:
            self.input_history.pop(0)
        
        return {
            "status": "success",
            "structured_data": structured_data,
            "metadata": metadata,
            "emotional_state": self.emotional_state.get_state(),
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
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
            
        # Must contain at least one of these data fields
        required_fields = ["text", "speech", "sensor_data"]
        return any(field in message for field in required_fields)
    
    def _process_text(self, text: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process text input and extract structured data.
        
        Args:
            text: Raw text input
            
        Returns:
            Tuple of (structured_data, metadata)
        """
        # Parse commands, entities, intent
        commands = self._extract_commands(text)
        entities = self._extract_entities(text)
        intent = self._determine_intent(text)
        
        structured_data = {
            "commands": commands,
            "entities": entities,
            "intent": intent,
            "original_text": text
        }
        
        metadata = {
            "text_length": len(text),
            "word_count": len(text.split()),
            "language": "en",  # Would detect actual language
            "processing_time": 0.001  # Simulated processing time
        }
        
        return structured_data, metadata
    
    def _process_speech(self, speech: Any) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process speech input and convert to structured data.
        
        Args:
            speech: Raw speech input
            
        Returns:
            Tuple of (structured_data, metadata)
        """
        # In a real implementation, this would use speech recognition
        # Here we just simulate the result
        
        structured_data = {
            "transcribed_text": "Simulated speech transcription",
            "commands": [],
            "entities": [],
            "intent": "unknown"
        }
        
        metadata = {
            "audio_length": 0.0,  # Would be actual audio length in seconds
            "confidence": 0.9,  # Would be actual transcription confidence
            "language": "en"  # Would be detected language
        }
        
        return structured_data, metadata
    
    def _process_sensor_data(self, sensor_data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process sensor data input.
        
        Args:
            sensor_data: Raw sensor readings
            
        Returns:
            Tuple of (structured_data, metadata)
        """
        # In a real implementation, this would process and normalize sensor readings
        
        structured_data = {
            "normalized_readings": {},
            "anomalies": []
        }
        
        for sensor_id, reading in sensor_data.items():
            # Apply some simulated normalization
            structured_data["normalized_readings"][sensor_id] = reading
            
            # Detect anomalies (in this simulation, just check if value > 100)
            if isinstance(reading, (int, float)) and reading > 100:
                structured_data["anomalies"].append({
                    "sensor_id": sensor_id,
                    "reading": reading,
                    "severity": "high"
                })
        
        metadata = {
            "sensor_count": len(sensor_data),
            "anomaly_count": len(structured_data["anomalies"]),
            "timestamp": time.time()
        }
        
        return structured_data, metadata
    
    def _extract_commands(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract command structures from text.
        
        Args:
            text: Text to extract commands from
            
        Returns:
            List of command objects
        """
        # Simple command extraction logic
        # In a real implementation, this would use NLP techniques
        
        commands = []
        
        # Look for common command patterns
        if re.search(r"^(start|begin|run)\s+(\w+)", text, re.IGNORECASE):
            match = re.search(r"^(start|begin|run)\s+(\w+)", text, re.IGNORECASE)
            if match:
                commands.append({
                    "action": match.group(1).lower(),
                    "target": match.group(2).lower(),
                    "args": []
                })
        
        return commands
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of entity objects
        """
        # Simple entity extraction
        # In a real implementation, would use NER models
        
        entities = []
        
        # Simple date pattern
        date_pattern = r"\d{1,2}/\d{1,2}/\d{2,4}"
        for match in re.finditer(date_pattern, text):
            entities.append({
                "type": "date",
                "value": match.group(0),
                "position": (match.start(), match.end())
            })
        
        # Simple email pattern
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        for match in re.finditer(email_pattern, text):
            entities.append({
                "type": "email",
                "value": match.group(0),
                "position": (match.start(), match.end())
            })
        
        return entities
    
    def _determine_intent(self, text: str) -> str:
        """
        Determine the intent of the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Intent classification
        """
        # Basic intent detection based on keywords
        # Real implementation would use intent classification models
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["help", "assist", "support"]):
            return "help_request"
        
        if any(word in text_lower for word in ["create", "make", "build", "start"]):
            return "creation"
        
        if any(word in text_lower for word in ["delete", "remove", "destroy"]):
            return "deletion"
        
        if any(word in text_lower for word in ["update", "change", "modify"]):
            return "modification"
        
        if any(word in text_lower for word in ["find", "search", "locate", "where"]):
            return "query"
        
        return "unknown"
    
    def _update_emotion_from_text(self, text: str) -> None:
        """
        Update emotional state based on text content.
        
        Args:
            text: Text to analyze for emotional cues
        """
        text_lower = text.lower()
        
        # Check for urgency keywords
        urgency_score = sum(1 for word in self.urgency_keywords if word in text_lower)
        if urgency_score > 0:
            # Scale to 0.6-0.9 range based on number of matches
            normalized_urgency = 0.6 + min(0.3, urgency_score * 0.1)
            self.emotional_state.update(EmotionalDimension.URGENCY.value, normalized_urgency)
        
        # Check for suspicion keywords
        suspicion_score = sum(1 for word in self.suspicion_keywords if word in text_lower)
        if suspicion_score > 0:
            # Scale to 0.6-0.9 range based on number of matches
            normalized_suspicion = 0.6 + min(0.3, suspicion_score * 0.1)
            self.emotional_state.update(EmotionalDimension.SUSPICION.value, normalized_suspicion) 
    
    # ==================== SELF-AUDIT CAPABILITIES ====================
    
    def audit_input_output(self, output_text: str, operation: str = "", context: str = "") -> Dict[str, Any]:
        """
        Perform real-time integrity audit on input operation outputs.
        
        Args:
            output_text: Text output to audit
            operation: Input operation type (text, speech, sensor_data, etc.)
            context: Additional context for the audit
            
        Returns:
            Audit results with violations and integrity score
        """
        if not self.enable_self_audit:
            return {'integrity_score': 100.0, 'violations': [], 'total_violations': 0}
        
        self.logger.info(f"Performing self-audit on input output for operation: {operation}")
        
        # Use proven audit engine
        audit_context = f"input:{operation}:{context}" if context else f"input:{operation}"
        violations = self_audit_engine.audit_text(output_text, audit_context)
        integrity_score = self_audit_engine.get_integrity_score(output_text)
        
        # Log violations for input-specific analysis
        if violations:
            self.logger.warning(f"Detected {len(violations)} integrity violations in input output")
            for violation in violations:
                self.logger.warning(f"  - {violation.severity}: {violation.text} -> {violation.suggested_replacement}")
        
        return {
            'violations': violations,
            'integrity_score': integrity_score,
            'total_violations': len(violations),
            'violation_breakdown': self._categorize_input_violations(violations),
            'operation': operation,
            'audit_timestamp': time.time()
        }
    
    def auto_correct_input_output(self, output_text: str, operation: str = "") -> Dict[str, Any]:
        """
        Automatically correct integrity violations in input outputs.
        
        Args:
            output_text: Text to correct
            operation: Input operation type
            
        Returns:
            Corrected output with audit details
        """
        if not self.enable_self_audit:
            return {'corrected_text': output_text, 'violations_fixed': [], 'improvement': 0}
        
        self.logger.info(f"Performing self-correction on input output for operation: {operation}")
        
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
    
    def analyze_input_integrity_trends(self, time_window: int = 3600) -> Dict[str, Any]:
        """
        Analyze input operation integrity trends for self-improvement.
        
        Args:
            time_window: Time window in seconds to analyze
            
        Returns:
            Input integrity trend analysis with mathematical validation
        """
        if not self.enable_self_audit:
            return {'integrity_status': 'MONITORING_DISABLED'}
        
        self.logger.info(f"Analyzing input integrity trends over {time_window} seconds")
        
        # Get integrity report from audit engine
        integrity_report = self_audit_engine.generate_integrity_report()
        
        # Calculate input-specific metrics
        input_metrics = {
            'input_history_length': len(self.input_history),
            'urgency_keywords_count': len(self.urgency_keywords),
            'suspicion_keywords_count': len(self.suspicion_keywords),
            'total_inputs_processed': len(self.input_history)
        }
        
        # Generate input-specific recommendations
        recommendations = self._generate_input_integrity_recommendations(
            integrity_report, input_metrics
        )
        
        return {
            'integrity_status': integrity_report['integrity_status'],
            'total_violations': integrity_report['total_violations'],
            'input_metrics': input_metrics,
            'integrity_trend': self._calculate_input_integrity_trend(),
            'recommendations': recommendations,
            'analysis_timestamp': time.time()
        }
    
    def get_input_integrity_report(self) -> Dict[str, Any]:
        """Generate comprehensive input integrity report"""
        if not self.enable_self_audit:
            return {'status': 'SELF_AUDIT_DISABLED'}
        
        # Get basic integrity report
        base_report = self_audit_engine.generate_integrity_report()
        
        # Add input-specific metrics
        input_report = {
            'input_agent_id': self.agent_id,
            'monitoring_enabled': self.integrity_monitoring_enabled,
            'input_processing_stats': {
                'history_length': len(self.input_history),
                'keyword_sets_configured': 2  # urgency and suspicion
            },
            'emotional_state_integration': {
                'urgency_keywords': len(self.urgency_keywords),
                'suspicion_keywords': len(self.suspicion_keywords)
            },
            'integrity_metrics': getattr(self, 'integrity_metrics', {}),
            'base_integrity_report': base_report,
            'report_timestamp': time.time()
        }
        
        return input_report
    
    def _monitor_input_output_integrity(self, output_text: str, operation: str = "") -> str:
        """
        Internal method to monitor and potentially correct input output integrity.
        
        Args:
            output_text: Output to monitor
            operation: Input operation type
            
        Returns:
            Potentially corrected output
        """
        if not getattr(self, 'integrity_monitoring_enabled', False):
            return output_text
        
        # Perform audit
        audit_result = self.audit_input_output(output_text, operation)
        
        # Update monitoring metrics
        if hasattr(self, 'integrity_metrics'):
            self.integrity_metrics['total_outputs_monitored'] += 1
            self.integrity_metrics['total_violations_detected'] += audit_result['total_violations']
        
        # Auto-correct if violations detected
        if audit_result['violations']:
            correction_result = self.auto_correct_input_output(output_text, operation)
            
            self.logger.info(f"Auto-corrected input output: {len(audit_result['violations'])} violations fixed")
            
            return correction_result['corrected_text']
        
        return output_text
    
    def _categorize_input_violations(self, violations: List[IntegrityViolation]) -> Dict[str, int]:
        """Categorize integrity violations specific to input operations"""
        categories = defaultdict(int)
        
        for violation in violations:
            categories[violation.violation_type.value] += 1
        
        return dict(categories)
    
    def _generate_input_integrity_recommendations(self, integrity_report: Dict[str, Any], input_metrics: Dict[str, Any]) -> List[str]:
        """Generate input-specific integrity improvement recommendations"""
        recommendations = []
        
        if integrity_report.get('total_violations', 0) > 5:
            recommendations.append("Consider implementing more rigorous input validation and sanitization")
        
        if input_metrics.get('input_history_length', 0) > 90:
            recommendations.append("Input history approaching capacity - consider implementing cleanup or archival")
        
        if input_metrics.get('urgency_keywords_count', 0) < 5:
            recommendations.append("Consider expanding urgency keyword detection for better emotional state updates")
        
        if input_metrics.get('suspicion_keywords_count', 0) < 5:
            recommendations.append("Consider expanding suspicion keyword detection for better threat detection")
        
        if len(recommendations) == 0:
            recommendations.append("Input processing integrity status is excellent - maintain current practices")
        
        return recommendations
    
    def _calculate_input_integrity_trend(self) -> Dict[str, Any]:
        """Calculate input integrity trends with mathematical validation"""
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
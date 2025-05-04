"""
Log Agent

Records system events and performance metrics for analysis and debugging.
Provides a historical record of system activity.
"""

from typing import Dict, Any, List, Optional, Union
import time
import json
import os
import datetime
import logging
from collections import deque

from src.core.registry import NISAgent, NISLayer, NISRegistry
from src.emotion.emotional_state import EmotionalState, EmotionalDimension


class LogAgent(NISAgent):
    """
    Agent that maintains a log of system events.
    
    The Log Agent is responsible for:
    - Recording system events and agent activities
    - Tracking performance metrics
    - Storing error and warning information
    - Providing data for system diagnostics
    """
    
    def __init__(
        self,
        agent_id: str = "log",
        description: str = "Records system events and performance metrics",
        emotional_state: Optional[EmotionalState] = None,
        log_path: Optional[str] = None,
        memory_size: int = 1000,
        log_level: int = logging.INFO
    ):
        """
        Initialize a new Log Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            description: Human-readable description of the agent's role
            emotional_state: Optional pre-configured emotional state
            log_path: Path to store log files
            memory_size: Maximum number of log entries to keep in memory
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        super().__init__(agent_id, NISLayer.MEMORY, description)
        self.emotional_state = emotional_state or EmotionalState()
        
        # In-memory log storage
        self.log_memory = deque(maxlen=memory_size)
        
        # Set up file-based logging if path provided
        self.log_path = log_path
        self.logger = None
        
        if log_path:
            if not os.path.exists(log_path):
                os.makedirs(log_path, exist_ok=True)
            
            # Configure logger
            logger_name = f"nis_log_agent_{agent_id}"
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(log_level)
            
            # Add file handler
            log_file = os.path.join(log_path, f"nis_log_{time.strftime('%Y%m%d')}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(file_handler)
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a log request.
        
        Args:
            message: Message containing log data
                'level': Log level (debug, info, warning, error, critical)
                'content': Log message content
                'source_agent': Agent that generated the log
                'metadata': Additional data to log
        
        Returns:
            Result of the log operation
        """
        if not self._validate_message(message):
            return {
                "status": "error",
                "error": "Invalid message format or missing required fields",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        # Extract log data
        level = message.get("level", "info").lower()
        content = message.get("content", "")
        source_agent = message.get("source_agent", "unknown")
        metadata = message.get("metadata", {})
        
        # Create log entry
        log_entry = {
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().isoformat(),
            "level": level,
            "content": content,
            "source_agent": source_agent,
            "metadata": metadata
        }
        
        # Store in memory
        self.log_memory.append(log_entry)
        
        # Write to file logger if configured
        if self.logger:
            self._write_to_log(level, f"[{source_agent}] {content}", metadata)
            
        # Update emotional state based on log level
        self._update_emotional_state(level)
        
        return {
            "status": "success",
            "log_id": len(self.log_memory),
            "level": level,
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
        
        # Must have content to log
        if "content" not in message:
            return False
        
        return True
    
    def _write_to_log(self, level: str, content: str, metadata: Dict[str, Any]) -> None:
        """
        Write entry to file logger.
        
        Args:
            level: Log level
            content: Log message
            metadata: Additional data
        """
        if not self.logger:
            return
        
        # Convert level string to logging level
        log_method = getattr(self.logger, level, None)
        if not log_method:
            log_method = self.logger.info
        
        # Add metadata as JSON if present
        if metadata:
            metadata_str = json.dumps(metadata)
            content = f"{content} - Metadata: {metadata_str}"
        
        # Write to log
        log_method(content)
    
    def _update_emotional_state(self, level: str) -> None:
        """
        Update emotional state based on log level.
        
        Args:
            level: Log level
        """
        # Error and critical logs increase urgency and suspicion
        if level in ["error", "critical"]:
            self.emotional_state.update(EmotionalDimension.URGENCY.value, 0.8)
            self.emotional_state.update(EmotionalDimension.SUSPICION.value, 0.7)
            # Decrease confidence when errors occur
            self.emotional_state.update(EmotionalDimension.CONFIDENCE.value, 0.3)
        
        # Warning logs increase suspicion
        elif level == "warning":
            self.emotional_state.update(EmotionalDimension.SUSPICION.value, 0.6)
    
    def get_recent_logs(self, count: int = 10, level: Optional[str] = None, 
                        source_agent: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent log entries with optional filtering.
        
        Args:
            count: Maximum number of logs to return
            level: Filter by log level
            source_agent: Filter by source agent
            
        Returns:
            List of matching log entries
        """
        results = []
        
        # Filter logs
        for entry in reversed(self.log_memory):
            if level and entry.get("level") != level:
                continue
                
            if source_agent and entry.get("source_agent") != source_agent:
                continue
                
            results.append(entry)
            
            if len(results) >= count:
                break
        
        return results
    
    def clear_logs(self) -> Dict[str, Any]:
        """
        Clear the in-memory logs.
        
        Returns:
            Operation result
        """
        count = len(self.log_memory)
        self.log_memory.clear()
        
        return {
            "status": "success",
            "cleared_count": count,
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about logged events.
        
        Returns:
            Statistics about logs
        """
        if not self.log_memory:
            return {
                "status": "success",
                "log_count": 0,
                "level_counts": {},
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        # Count by level
        level_counts = {}
        source_counts = {}
        
        for entry in self.log_memory:
            level = entry.get("level", "unknown")
            source = entry.get("source_agent", "unknown")
            
            level_counts[level] = level_counts.get(level, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Calculate time range
        oldest = min(entry.get("timestamp", time.time()) for entry in self.log_memory)
        newest = max(entry.get("timestamp", time.time()) for entry in self.log_memory)
        
        return {
            "status": "success",
            "log_count": len(self.log_memory),
            "level_counts": level_counts,
            "source_counts": source_counts,
            "time_range": {
                "oldest": oldest,
                "newest": newest,
                "span_seconds": newest - oldest
            },
            "agent_id": self.agent_id,
            "timestamp": time.time()
        } 
"""
Unit Tests for A2A Adapter

Tests the Agent2Agent Protocol adapter implementation including:
- Task creation and lifecycle
- UX negotiation and message parts
- Error handling and retry logic
- Circuit breaker behavior
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from src.adapters.a2a_adapter import A2AAdapter
from src.adapters.protocol_errors import (
    ProtocolConnectionError,
    ProtocolTimeoutError,
    ProtocolValidationError
)


class TestA2AAdapterInitialization:
    """Test A2A adapter initialization"""
    
    def test_adapter_creation(self):
        """Test basic adapter instantiation"""
        config = {
            "base_url": "https://api.google.com/a2a",
            "api_key": "test-key",
            "timeout": 30
        }
        adapter = A2AAdapter(config)
        
        assert adapter.protocol_name == "a2a"
        assert adapter.config == config
        assert adapter.circuit_breaker is not None
        assert adapter.metrics is not None
        assert len(adapter.active_tasks) == 0
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        valid_config = {"base_url": "https://api.google.com/a2a"}
        adapter = A2AAdapter(valid_config)
        assert adapter.validate_config() is True
        
        # Invalid config (missing base_url)
        invalid_config = {"api_key": "test-key"}
        adapter = A2AAdapter(invalid_config)
        assert adapter.validate_config() is False


class TestA2ATaskOperations:
    """Test A2A task creation and lifecycle"""
    
    @pytest.mark.asyncio
    async def test_task_creation_success(self):
        """Test successful task creation"""
        config = {"base_url": "https://api.google.com/a2a"}
        adapter = A2AAdapter(config)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "task_id": "task-123",
            "status": "pending",
            "agent_id": "agent-456"
        }
        
        with patch('requests.post', return_value=mock_response):
            result = await adapter.create_task(
                description="Test task",
                agent_id="agent-456",
                parameters={"input": "test"}
            )
            
            assert result["task_id"] == "task-123"
            assert result["status"] == "pending"
            assert "task-123" in adapter.active_tasks
    
    @pytest.mark.asyncio
    async def test_task_status_retrieval(self):
        """Test getting task status"""
        config = {"base_url": "https://api.google.com/a2a"}
        adapter = A2AAdapter(config)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "task_id": "task-123",
            "status": "in_progress",
            "progress": 50
        }
        
        with patch('requests.get', return_value=mock_response):
            result = await adapter.get_task_status("task-123")
            
            assert result["status"] == "in_progress"
            assert result["progress"] == 50
    
    @pytest.mark.asyncio
    async def test_task_cancellation(self):
        """Test cancelling a task"""
        config = {"base_url": "https://api.google.com/a2a"}
        adapter = A2AAdapter(config)
        adapter.active_tasks["task-123"] = {
            "task_id": "task-123",
            "status": "in_progress"
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "task_id": "task-123",
            "status": "cancelled"
        }
        
        with patch('requests.post', return_value=mock_response):
            result = await adapter.cancel_task("task-123")
            
            assert result["status"] == "cancelled"
            assert "task-123" not in adapter.active_tasks
    
    @pytest.mark.asyncio
    async def test_task_completion_polling(self):
        """Test waiting for task completion"""
        config = {"base_url": "https://api.google.com/a2a"}
        adapter = A2AAdapter(config)
        
        # Mock responses: first pending, then completed
        responses = [
            {"task_id": "task-123", "status": "pending"},
            {"task_id": "task-123", "status": "in_progress"},
            {"task_id": "task-123", "status": "completed", "result": "success"}
        ]
        response_iter = iter(responses)
        
        def mock_get(*args, **kwargs):
            mock = Mock()
            mock.status_code = 200
            mock.json.return_value = next(response_iter)
            return mock
        
        with patch('requests.get', side_effect=mock_get):
            result = await adapter.wait_for_task_completion(
                "task-123",
                poll_interval=0.1,
                timeout=5.0
            )
            
            assert result["status"] == "completed"
            assert result["result"] == "success"
    
    @pytest.mark.asyncio
    async def test_task_timeout(self):
        """Test task timeout during polling"""
        config = {"base_url": "https://api.google.com/a2a"}
        adapter = A2AAdapter(config)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "task_id": "task-123",
            "status": "in_progress"
        }
        
        with patch('requests.get', return_value=mock_response):
            with pytest.raises(ProtocolTimeoutError):
                await adapter.wait_for_task_completion(
                    "task-123",
                    poll_interval=0.1,
                    timeout=0.3
                )


class TestA2AUXNegotiation:
    """Test A2A UX negotiation features"""
    
    def test_message_with_text_parts(self):
        """Test creating message with text parts"""
        config = {"base_url": "https://api.google.com/a2a"}
        adapter = A2AAdapter(config)
        
        message = adapter.create_message_with_parts([
            {"type": "text", "content": "Hello world"}
        ])
        
        assert "parts" in message
        assert len(message["parts"]) == 1
        assert message["parts"][0]["type"] == "text"
    
    def test_message_with_multiple_parts(self):
        """Test creating message with multiple content types"""
        config = {"base_url": "https://api.google.com/a2a"}
        adapter = A2AAdapter(config)
        
        message = adapter.create_message_with_parts([
            {"type": "text", "content": "Check this image:"},
            {"type": "image", "url": "https://example.com/image.png"},
            {"type": "iframe", "url": "https://example.com/embed"}
        ])
        
        assert len(message["parts"]) == 3
        assert message["parts"][0]["type"] == "text"
        assert message["parts"][1]["type"] == "image"
        assert message["parts"][2]["type"] == "iframe"
    
    def test_message_with_data_part(self):
        """Test creating message with structured data"""
        config = {"base_url": "https://api.google.com/a2a"}
        adapter = A2AAdapter(config)
        
        message = adapter.create_message_with_parts([
            {
                "type": "data",
                "content": {"key": "value", "count": 42}
            }
        ])
        
        assert message["parts"][0]["type"] == "data"
        assert message["parts"][0]["content"]["count"] == 42


class TestA2AErrorHandling:
    """Test error handling and retry logic"""
    
    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        """Test retry logic on connection error"""
        import requests
        
        config = {"base_url": "https://api.google.com/a2a"}
        adapter = A2AAdapter(config)
        
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise requests.exceptions.ConnectionError()
            
            mock = Mock()
            mock.status_code = 200
            mock.json.return_value = {
                "task_id": "task-123",
                "status": "pending"
            }
            return mock
        
        with patch('requests.post', side_effect=side_effect):
            result = await adapter.create_task(
                description="Test",
                agent_id="agent-1",
                parameters={}
            )
            
            assert call_count == 2  # Failed once, succeeded on retry
            assert result["task_id"] == "task-123"
    
    @pytest.mark.asyncio
    async def test_timeout_error(self):
        """Test timeout error handling"""
        import requests
        
        config = {"base_url": "https://api.google.com/a2a", "timeout": 1}
        adapter = A2AAdapter(config)
        
        with patch('requests.post', side_effect=requests.exceptions.Timeout):
            with pytest.raises(ProtocolTimeoutError) as exc_info:
                await adapter.create_task(
                    description="Test",
                    agent_id="agent-1",
                    parameters={}
                )
            
            assert exc_info.value.timeout == 1
    
    @pytest.mark.asyncio
    async def test_http_error_handling(self):
        """Test HTTP error handling"""
        import requests
        
        config = {"base_url": "https://api.google.com/a2a"}
        adapter = A2AAdapter(config)
        
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": {"code": "UNAUTHORIZED", "message": "Invalid API key"}
        }
        
        http_error = requests.exceptions.HTTPError()
        http_error.response = mock_response
        
        with patch('requests.post', side_effect=http_error):
            with pytest.raises(Exception):  # Should raise specific protocol error
                await adapter.create_task(
                    description="Test",
                    agent_id="agent-1",
                    parameters={}
                )
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_behavior(self):
        """Test circuit breaker opens after failures"""
        import requests
        
        config = {
            "base_url": "https://api.google.com/a2a",
            "failure_threshold": 3
        }
        adapter = A2AAdapter(config)
        
        with patch('requests.post', side_effect=requests.exceptions.ConnectionError):
            # Trigger failures
            for _ in range(3):
                try:
                    await adapter.create_task(
                        description="Test",
                        agent_id="agent-1",
                        parameters={}
                    )
                except (ProtocolConnectionError, Exception):
                    pass
            
            # Circuit should be open or half-open
            state = adapter.circuit_breaker.get_state()
            assert state["state"] in ["open", "half_open"]


class TestA2AHealthMonitoring:
    """Test health monitoring and metrics"""
    
    def test_health_status(self):
        """Test getting health status"""
        config = {"base_url": "https://api.google.com/a2a"}
        adapter = A2AAdapter(config)
        
        health = adapter.get_health_status()
        
        assert "protocol" in health
        assert health["protocol"] == "a2a"
        assert "healthy" in health
        assert "circuit_breaker" in health
        assert "metrics" in health
        assert "active_tasks_count" in health
    
    def test_metrics_tracking(self):
        """Test metrics are tracked correctly"""
        config = {"base_url": "https://api.google.com/a2a"}
        adapter = A2AAdapter(config)
        
        # Record some requests
        adapter.metrics.record_request(True, 0.5)
        adapter.metrics.record_request(True, 0.3)
        adapter.metrics.record_request(False, 1.0, "timeout")
        
        metrics = adapter.metrics.to_dict()
        
        assert metrics["total_requests"] == 3
        assert metrics["successful_requests"] == 2
        assert metrics["failed_requests"] == 1
        assert 0.5 < metrics["success_rate"] < 1.0
    
    def test_reset_functionality(self):
        """Test reset methods"""
        config = {"base_url": "https://api.google.com/a2a"}
        adapter = A2AAdapter(config)
        
        # Add some state
        adapter.metrics.record_request(True, 0.5)
        adapter.active_tasks["task-1"] = {}
        
        # Reset metrics
        adapter.reset_metrics()
        assert adapter.metrics.total_requests == 0
        
        # Reset circuit breaker
        for _ in range(10):
            adapter.circuit_breaker._record_failure()
        
        adapter.reset_circuit_breaker()
        state = adapter.circuit_breaker.get_state()
        assert state["state"] == "closed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""
Unit Tests for ACP Adapter

Tests the Agent Communication Protocol adapter implementation including:
- Agent Card export for offline discovery
- Async/sync execution modes
- Task polling for long-running operations
- Error handling and retry logic
- Circuit breaker behavior
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from src.adapters.acp_adapter import ACPAdapter
from src.adapters.protocol_errors import (
    ProtocolConnectionError,
    ProtocolTimeoutError,
    ProtocolValidationError
)


class TestACPAdapterInitialization:
    """Test ACP adapter initialization"""
    
    def test_adapter_creation(self):
        """Test basic adapter instantiation"""
        config = {
            "base_url": "http://localhost:8080",
            "api_key": "test-key",
            "timeout": 30
        }
        adapter = ACPAdapter(config)
        
        assert adapter.protocol_name == "acp"
        assert adapter.config == config
        assert adapter.circuit_breaker is not None
        assert adapter.metrics is not None
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        valid_config = {"base_url": "http://localhost:8080"}
        adapter = ACPAdapter(valid_config)
        assert adapter.validate_config() is True
        
        # Invalid config (missing base_url)
        invalid_config = {"api_key": "test-key"}
        adapter = ACPAdapter(invalid_config)
        assert adapter.validate_config() is False


class TestACPAgentCard:
    """Test ACP Agent Card export for offline discovery"""
    
    def test_agent_card_structure(self):
        """Test Agent Card has correct structure"""
        config = {"base_url": "http://localhost:5000"}
        adapter = ACPAdapter(config)
        
        card = adapter.export_agent_card()
        
        assert "acp" in card
        assert "version" in card["acp"]
        assert "agent" in card["acp"]
    
    def test_agent_card_capabilities(self):
        """Test Agent Card includes NIS capabilities"""
        config = {"base_url": "http://localhost:5000"}
        adapter = ACPAdapter(config)
        
        card = adapter.export_agent_card()
        agent = card["acp"]["agent"]
        
        assert agent["id"] == "nis_protocol_v3.2"
        assert agent["name"] == "NIS Protocol"
        assert "capabilities" in agent
        
        capabilities = agent["capabilities"]
        assert "physics_validation" in capabilities
        assert "symbolic_reasoning" in capabilities
        assert "consciousness_assessment" in capabilities
        assert "laplace_signal_processing" in capabilities
        assert "kan_interpretability" in capabilities
        assert "pinn_physics_constraints" in capabilities
    
    def test_agent_card_endpoints(self):
        """Test Agent Card includes endpoint information"""
        config = {"base_url": "http://localhost:5000"}
        adapter = ACPAdapter(config)
        
        card = adapter.export_agent_card()
        endpoints = card["acp"]["agent"]["endpoints"]
        
        assert "base" in endpoints
        assert "execute" in endpoints
        assert "status" in endpoints
        assert "capabilities" in endpoints
        assert endpoints["base"] == "http://localhost:5000"
    
    def test_agent_card_authentication(self):
        """Test Agent Card includes auth information"""
        # With API key
        config_with_key = {
            "base_url": "http://localhost:5000",
            "api_key": "test-key"
        }
        adapter_with_key = ACPAdapter(config_with_key)
        card_with_key = adapter_with_key.export_agent_card()
        
        assert card_with_key["acp"]["agent"]["authentication"]["required"] is True
        
        # Without API key
        config_no_key = {"base_url": "http://localhost:5000"}
        adapter_no_key = ACPAdapter(config_no_key)
        card_no_key = adapter_no_key.export_agent_card()
        
        assert card_no_key["acp"]["agent"]["authentication"]["required"] is False
    
    def test_agent_card_metadata(self):
        """Test Agent Card includes metadata"""
        config = {"base_url": "http://localhost:5000"}
        adapter = ACPAdapter(config)
        
        card = adapter.export_agent_card()
        metadata = card["acp"]["agent"]["metadata"]
        
        assert metadata["framework"] == "nis-protocol"
        assert metadata["supports_async"] is True
        assert metadata["supports_streaming"] is True
        assert "pipeline_stages" in metadata
        assert len(metadata["pipeline_stages"]) == 4  # laplace, kan, pinn, llm


class TestACPExecution:
    """Test ACP agent execution (async and sync modes)"""
    
    @pytest.mark.asyncio
    async def test_sync_execution(self):
        """Test synchronous execution mode"""
        config = {"base_url": "http://localhost:8080"}
        adapter = ACPAdapter(config)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": "success",
            "output": "Processing complete"
        }
        
        with patch('requests.post', return_value=mock_response) as mock_post:
            result = await adapter.execute_agent(
                agent_url="http://external-agent:8080",
                message={"query": "test"},
                async_mode=False
            )
            
            assert result["result"] == "success"
            assert result["output"] == "Processing complete"
            
            # Verify sync mode was used
            call_args = mock_post.call_args
            assert call_args[1]["json"]["mode"] == "sync"
    
    @pytest.mark.asyncio
    async def test_async_execution_immediate(self):
        """Test async execution with immediate result"""
        config = {"base_url": "http://localhost:8080"}
        adapter = ACPAdapter(config)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": "success",
            "output": "Processing complete"
        }
        
        with patch('requests.post', return_value=mock_response) as mock_post:
            result = await adapter.execute_agent(
                agent_url="http://external-agent:8080",
                message={"query": "test"},
                async_mode=True
            )
            
            assert result["result"] == "success"
            
            # Verify async mode was used
            call_args = mock_post.call_args
            assert call_args[1]["json"]["mode"] == "async"
    
    @pytest.mark.asyncio
    async def test_async_execution_with_polling(self):
        """Test async execution with task polling"""
        config = {"base_url": "http://localhost:8080"}
        adapter = ACPAdapter(config)
        
        # Mock initial POST response with task_id
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {
            "task_id": "task-789",
            "status": "pending"
        }
        
        # Mock GET responses for polling
        poll_responses = [
            {"task_id": "task-789", "status": "in_progress"},
            {"task_id": "task-789", "status": "in_progress"},
            {"task_id": "task-789", "status": "completed", "result": "done"}
        ]
        response_iter = iter(poll_responses)
        
        def mock_get(*args, **kwargs):
            mock = Mock()
            mock.status_code = 200
            mock.json.return_value = next(response_iter)
            return mock
        
        with patch('requests.post', return_value=mock_post_response):
            with patch('requests.get', side_effect=mock_get):
                result = await adapter.execute_agent(
                    agent_url="http://external-agent:8080",
                    message={"query": "test"},
                    async_mode=True
                )
                
                assert result["status"] == "completed"
                assert result["result"] == "done"
    
    @pytest.mark.asyncio
    async def test_polling_timeout(self):
        """Test task polling times out"""
        config = {"base_url": "http://localhost:8080"}
        adapter = ACPAdapter(config)
        
        # Mock always returns in_progress
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {"task_id": "task-999"}
        
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {"status": "in_progress"}
        
        with patch('requests.post', return_value=mock_post_response):
            with patch('requests.get', return_value=mock_get_response):
                # Override polling timeout for test
                with patch.object(adapter, '_poll_task_result') as mock_poll:
                    mock_poll.side_effect = ProtocolTimeoutError(
                        "Task timed out",
                        timeout=0.5
                    )
                    
                    with pytest.raises(ProtocolTimeoutError):
                        await adapter.execute_agent(
                            agent_url="http://external-agent:8080",
                            message={"query": "test"},
                            async_mode=True
                        )


class TestACPMessageTranslation:
    """Test message translation between NIS and ACP formats"""
    
    def test_translate_to_nis(self):
        """Test translating ACP message to NIS format"""
        config = {"base_url": "http://localhost:8080"}
        adapter = ACPAdapter(config)
        
        acp_message = {
            "headers": {
                "message_id": "msg-123",
                "sender_id": "agent-1",
                "action": "query",
                "conversation_id": "conv-456"
            },
            "body": {
                "text": "Hello",
                "emotional_state": {"valence": 0.8, "arousal": 0.5}
            }
        }
        
        nis_message = adapter.translate_to_nis(acp_message)
        
        assert nis_message["protocol"] == "nis"
        assert nis_message["source_protocol"] == "acp"
        assert nis_message["payload"]["action"] == "query"
        assert nis_message["payload"]["data"]["text"] == "Hello"
        assert "emotional_state" in nis_message
        assert nis_message["metadata"]["acp_message_id"] == "msg-123"
    
    def test_translate_from_nis(self):
        """Test translating NIS message to ACP format"""
        config = {"base_url": "http://localhost:8080"}
        adapter = ACPAdapter(config)
        
        nis_message = {
            "protocol": "nis",
            "payload": {
                "action": "response",
                "data": {"result": "success"}
            },
            "metadata": {
                "acp_message_id": "msg-123",
                "acp_sender_id": "agent-1",
                "acp_conversation_id": "conv-456"
            },
            "emotional_state": {"valence": 0.9, "arousal": 0.3}
        }
        
        acp_message = adapter.translate_from_nis(nis_message)
        
        assert "headers" in acp_message
        assert "body" in acp_message
        assert acp_message["headers"]["action"] == "response"
        assert acp_message["body"]["result"] == "success"
        assert "emotional_state" in acp_message["body"]


class TestACPErrorHandling:
    """Test error handling and retry logic"""
    
    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """Test retry logic on timeout"""
        import requests
        
        config = {"base_url": "http://localhost:8080"}
        adapter = ACPAdapter(config)
        
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise requests.exceptions.Timeout()
            
            mock = Mock()
            mock.status_code = 200
            mock.json.return_value = {"result": "success"}
            return mock
        
        with patch('requests.post', side_effect=side_effect):
            result = await adapter.execute_agent(
                agent_url="http://external-agent:8080",
                message={"query": "test"},
                async_mode=False
            )
            
            assert call_count == 2
            assert result["result"] == "success"
    
    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Test connection error handling"""
        import requests
        
        config = {"base_url": "http://localhost:8080"}
        adapter = ACPAdapter(config)
        
        with patch('requests.post', side_effect=requests.exceptions.ConnectionError):
            with pytest.raises(ProtocolConnectionError):
                await adapter.execute_agent(
                    agent_url="http://external-agent:8080",
                    message={"query": "test"},
                    async_mode=False
                )
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self):
        """Test circuit breaker opens after failures"""
        import requests
        
        config = {
            "base_url": "http://localhost:8080",
            "failure_threshold": 3
        }
        adapter = ACPAdapter(config)
        
        with patch('requests.post', side_effect=requests.exceptions.ConnectionError):
            for _ in range(3):
                try:
                    await adapter.execute_agent(
                        agent_url="http://external-agent:8080",
                        message={"query": "test"},
                        async_mode=False
                    )
                except (ProtocolConnectionError, Exception):
                    pass
            
            state = adapter.circuit_breaker.get_state()
            assert state["state"] in ["open", "half_open"]


class TestACPHealthMonitoring:
    """Test health monitoring and metrics"""
    
    def test_health_status(self):
        """Test getting health status"""
        config = {"base_url": "http://localhost:8080"}
        adapter = ACPAdapter(config)
        
        health = adapter.get_health_status()
        
        assert "protocol" in health
        assert health["protocol"] == "acp"
        assert "healthy" in health
        assert "circuit_breaker" in health
        assert "metrics" in health
        assert "agent_card" in health
        assert health["agent_card"]["id"] == "nis_protocol_v3.2"
    
    def test_metrics_reset(self):
        """Test resetting metrics"""
        config = {"base_url": "http://localhost:8080"}
        adapter = ACPAdapter(config)
        
        adapter.metrics.record_request(True, 0.5)
        assert adapter.metrics.total_requests > 0
        
        adapter.reset_metrics()
        assert adapter.metrics.total_requests == 0
    
    def test_circuit_breaker_reset(self):
        """Test resetting circuit breaker"""
        config = {"base_url": "http://localhost:8080"}
        adapter = ACPAdapter(config)
        
        for _ in range(10):
            adapter.circuit_breaker._record_failure()
        
        state_before = adapter.circuit_breaker.get_state()
        assert state_before["state"] == "open"
        
        adapter.reset_circuit_breaker()
        
        state_after = adapter.circuit_breaker.get_state()
        assert state_after["state"] == "closed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


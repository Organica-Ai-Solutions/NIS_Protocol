"""
Unit Tests for MCP Adapter

Tests the Model Context Protocol adapter implementation including:
- Initialization and lifecycle
- Tool discovery and execution
- Resource and prompt discovery
- Error handling and retry logic
- Circuit breaker behavior
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.adapters.mcp_adapter import MCPAdapter
from src.adapters.protocol_errors import (
    ProtocolConnectionError,
    ProtocolTimeoutError,
    ProtocolValidationError
)


class TestMCPAdapterInitialization:
    """Test MCP adapter initialization and lifecycle"""
    
    def test_adapter_creation(self):
        """Test basic adapter instantiation"""
        config = {
            "server_url": "http://localhost:3000",
            "timeout": 30
        }
        adapter = MCPAdapter(config)
        
        assert adapter.protocol_name == "mcp"
        assert adapter.config == config
        assert adapter.initialized is False
        assert adapter.circuit_breaker is not None
        assert adapter.metrics is not None
    
    @pytest.mark.asyncio
    async def test_successful_initialization(self):
        """Test successful MCP initialization handshake"""
        config = {"server_url": "http://localhost:3000"}
        adapter = MCPAdapter(config)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {}
                },
                "serverInfo": {
                    "name": "test-server",
                    "version": "1.0"
                }
            }
        }
        
        with patch('requests.post', return_value=mock_response):
            result = await adapter.initialize()
            
            assert adapter.initialized is True
            assert "capabilities" in result
            assert result["capabilities"]["tools"]["listChanged"] is True
    
    @pytest.mark.asyncio
    async def test_initialization_timeout(self):
        """Test initialization timeout handling"""
        import requests
        
        config = {"server_url": "http://localhost:3000", "timeout": 1}
        adapter = MCPAdapter(config)
        
        with patch('requests.post', side_effect=requests.exceptions.Timeout):
            with pytest.raises(ProtocolTimeoutError) as exc_info:
                await adapter.initialize()
            
            assert "timed out" in str(exc_info.value).lower()
            assert adapter.initialized is False
    
    @pytest.mark.asyncio
    async def test_initialization_connection_error(self):
        """Test initialization connection error handling"""
        import requests
        
        config = {"server_url": "http://localhost:3000"}
        adapter = MCPAdapter(config)
        
        with patch('requests.post', side_effect=requests.exceptions.ConnectionError):
            with pytest.raises(ProtocolConnectionError):
                await adapter.initialize()
            
            assert adapter.initialized is False


class TestMCPToolOperations:
    """Test MCP tool discovery and execution"""
    
    @pytest.mark.asyncio
    async def test_tool_discovery(self):
        """Test discovering available tools"""
        config = {"server_url": "http://localhost:3000"}
        adapter = MCPAdapter(config)
        adapter.initialized = True
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "tools": [
                    {
                        "name": "calculator_arithmetic",
                        "description": "Perform arithmetic calculations",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "expression": {"type": "string"}
                            }
                        }
                    }
                ]
            }
        }
        
        with patch('requests.post', return_value=mock_response):
            await adapter.discover_tools()
            
            assert len(adapter.tools_registry) == 1
            assert "calculator_arithmetic" in adapter.tools_registry
    
    @pytest.mark.asyncio
    async def test_tool_execution_success(self):
        """Test successful tool execution"""
        config = {"server_url": "http://localhost:3000"}
        adapter = MCPAdapter(config)
        adapter.initialized = True
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 3,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": "Result: 42"
                    }
                ]
            }
        }
        
        with patch('requests.post', return_value=mock_response):
            result = await adapter.call_tool(
                "calculator_arithmetic",
                {"expression": "6 * 7"}
            )
            
            assert "content" in result
            assert result["content"][0]["text"] == "Result: 42"
    
    @pytest.mark.asyncio
    async def test_tool_execution_validation_error(self):
        """Test tool execution with invalid response"""
        config = {"server_url": "http://localhost:3000"}
        adapter = MCPAdapter(config)
        adapter.initialized = True
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 3,
            "result": {}  # Missing 'content' field
        }
        
        with patch('requests.post', return_value=mock_response):
            with pytest.raises(ProtocolValidationError):
                await adapter.call_tool("test_tool", {})


class TestMCPResourceOperations:
    """Test MCP resource discovery and retrieval"""
    
    @pytest.mark.asyncio
    async def test_resource_discovery(self):
        """Test discovering available resources"""
        config = {"server_url": "http://localhost:3000"}
        adapter = MCPAdapter(config)
        adapter.initialized = True
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 4,
            "result": {
                "resources": [
                    {
                        "uri": "file://docs/readme.md",
                        "name": "README",
                        "mimeType": "text/markdown"
                    }
                ]
            }
        }
        
        with patch('requests.post', return_value=mock_response):
            await adapter.discover_resources()
            
            assert len(adapter.resources_registry) == 1
            assert "file://docs/readme.md" in adapter.resources_registry
    
    @pytest.mark.asyncio
    async def test_resource_reading(self):
        """Test reading a resource"""
        config = {"server_url": "http://localhost:3000"}
        adapter = MCPAdapter(config)
        adapter.initialized = True
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 5,
            "result": {
                "contents": [
                    {
                        "uri": "file://docs/readme.md",
                        "mimeType": "text/markdown",
                        "text": "# README\nThis is a test"
                    }
                ]
            }
        }
        
        with patch('requests.post', return_value=mock_response):
            result = await adapter.read_resource("file://docs/readme.md")
            
            assert "contents" in result
            assert result["contents"][0]["text"] == "# README\nThis is a test"


class TestMCPErrorHandling:
    """Test error handling and retry logic"""
    
    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """Test retry logic on timeout"""
        import requests
        
        config = {"server_url": "http://localhost:3000"}
        adapter = MCPAdapter(config)
        adapter.initialized = True
        
        # Mock to fail twice, then succeed
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise requests.exceptions.Timeout()
            
            mock = Mock()
            mock.status_code = 200
            mock.json.return_value = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"content": [{"type": "text", "text": "success"}]}
            }
            return mock
        
        with patch('requests.post', side_effect=side_effect):
            result = await adapter.call_tool("test_tool", {})
            assert call_count == 3  # Failed twice, succeeded on third
            assert result["content"][0]["text"] == "success"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self):
        """Test circuit breaker opens after threshold failures"""
        import requests
        
        config = {
            "server_url": "http://localhost:3000",
            "failure_threshold": 3
        }
        adapter = MCPAdapter(config)
        adapter.initialized = True
        
        with patch('requests.post', side_effect=requests.exceptions.ConnectionError):
            # Trigger failures to open circuit breaker
            for _ in range(3):
                try:
                    await adapter.call_tool("test_tool", {})
                except (ProtocolConnectionError, Exception):
                    pass
            
            # Circuit should now be open
            state = adapter.circuit_breaker.get_state()
            assert state["state"] in ["open", "half_open"]


class TestMCPHealthMonitoring:
    """Test health monitoring and metrics"""
    
    def test_health_status(self):
        """Test getting health status"""
        config = {"server_url": "http://localhost:3000"}
        adapter = MCPAdapter(config)
        
        health = adapter.get_health_status()
        
        assert "protocol" in health
        assert health["protocol"] == "mcp"
        assert "healthy" in health
        assert "circuit_breaker" in health
        assert "metrics" in health
    
    def test_metrics_reset(self):
        """Test resetting metrics"""
        config = {"server_url": "http://localhost:3000"}
        adapter = MCPAdapter(config)
        
        # Record some metrics
        adapter.metrics.record_request(True, 0.5)
        adapter.metrics.record_request(False, 1.0, "timeout")
        
        assert adapter.metrics.total_requests > 0
        
        adapter.reset_metrics()
        
        assert adapter.metrics.total_requests == 0
    
    def test_circuit_breaker_reset(self):
        """Test resetting circuit breaker"""
        config = {"server_url": "http://localhost:3000"}
        adapter = MCPAdapter(config)
        
        # Force circuit to open
        for _ in range(10):
            adapter.circuit_breaker._record_failure()
        
        state_before = adapter.circuit_breaker.get_state()
        assert state_before["state"] == "open"
        
        adapter.reset_circuit_breaker()
        
        state_after = adapter.circuit_breaker.get_state()
        assert state_after["state"] == "closed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


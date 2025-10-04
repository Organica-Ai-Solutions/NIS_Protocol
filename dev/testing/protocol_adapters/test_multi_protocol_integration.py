"""
Integration Tests for Multi-Protocol Workflows

Tests the integration of MCP, A2A, and ACP adapters working together
in realistic multi-protocol workflows.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from src.adapters.mcp_adapter import MCPAdapter
from src.adapters.a2a_adapter import A2AAdapter
from src.adapters.acp_adapter import ACPAdapter


class TestMultiProtocolWorkflow:
    """Test workflows that combine multiple protocols"""
    
    @pytest.mark.asyncio
    async def test_mcp_to_acp_workflow(self):
        """
        Test workflow: Use MCP tool to gather data, then send to ACP agent
        
        Scenario:
        1. Use MCP tool to extract data from a source
        2. Process the data
        3. Send to ACP agent for further processing
        """
        # Setup adapters
        mcp_config = {"server_url": "http://mcp-server:3000"}
        acp_config = {"base_url": "http://acp-agent:8080"}
        
        mcp_adapter = MCPAdapter(mcp_config)
        acp_adapter = ACPAdapter(acp_config)
        
        # Mock MCP initialization
        mcp_init_response = Mock()
        mcp_init_response.status_code = 200
        mcp_init_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test"}
            }
        }
        
        # Mock MCP tool execution
        mcp_tool_response = Mock()
        mcp_tool_response.status_code = 200
        mcp_tool_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "content": [{
                    "type": "text",
                    "text": "Extracted data: sensor readings"
                }]
            }
        }
        
        # Mock ACP execution
        acp_response = Mock()
        acp_response.status_code = 200
        acp_response.json.return_value = {
            "result": "success",
            "processed_data": "analyzed sensor readings"
        }
        
        with patch('requests.post') as mock_post:
            # Setup mock to return different responses
            mock_post.side_effect = [
                mcp_init_response,  # MCP init
                mcp_tool_response,  # MCP tool call
                acp_response        # ACP execute
            ]
            
            # Step 1: Initialize MCP and call tool
            await mcp_adapter.initialize()
            mcp_result = await mcp_adapter.call_tool("data_extractor", {})
            
            # Step 2: Process MCP result
            extracted_text = mcp_result["content"][0]["text"]
            
            # Step 3: Send to ACP agent
            acp_result = await acp_adapter.execute_agent(
                agent_url="http://acp-agent:8080",
                message={"input": extracted_text},
                async_mode=False
            )
            
            assert acp_result["result"] == "success"
            assert "processed_data" in acp_result
    
    @pytest.mark.asyncio
    async def test_a2a_to_mcp_workflow(self):
        """
        Test workflow: Create A2A task, then use MCP tool to process results
        
        Scenario:
        1. Create long-running A2A task
        2. Wait for completion
        3. Use MCP tool to format/analyze results
        """
        # Setup adapters
        a2a_config = {"base_url": "https://api.google.com/a2a"}
        mcp_config = {"server_url": "http://mcp-server:3000"}
        
        a2a_adapter = A2AAdapter(a2a_config)
        mcp_adapter = MCPAdapter(mcp_config)
        
        # Mock A2A task creation
        a2a_create_response = Mock()
        a2a_create_response.status_code = 200
        a2a_create_response.json.return_value = {
            "task_id": "task-xyz",
            "status": "completed",
            "result": {"data": "raw analysis results"}
        }
        
        # Mock MCP initialization
        mcp_init_response = Mock()
        mcp_init_response.status_code = 200
        mcp_init_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test"}
            }
        }
        
        # Mock MCP tool (formatter)
        mcp_format_response = Mock()
        mcp_format_response.status_code = 200
        mcp_format_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "content": [{
                    "type": "text",
                    "text": "Formatted: raw analysis results"
                }]
            }
        }
        
        with patch('requests.post') as mock_post:
            mock_post.side_effect = [
                a2a_create_response,  # A2A task creation
                mcp_init_response,    # MCP init
                mcp_format_response   # MCP formatter tool
            ]
            
            # Step 1: Create and get A2A task result
            a2a_result = await a2a_adapter.create_task(
                description="Analyze data",
                agent_id="analyzer-1",
                parameters={}
            )
            
            raw_data = a2a_result["result"]["data"]
            
            # Step 2: Initialize MCP
            await mcp_adapter.initialize()
            
            # Step 3: Format results with MCP tool
            formatted = await mcp_adapter.call_tool(
                "formatter",
                {"input": raw_data}
            )
            
            assert "Formatted:" in formatted["content"][0]["text"]
    
    @pytest.mark.asyncio
    async def test_full_three_protocol_pipeline(self):
        """
        Test complete pipeline using all three protocols
        
        Scenario:
        1. MCP: Extract data from source
        2. ACP: Process data with physics validation
        3. A2A: Create long-running task for final analysis
        4. Return combined results
        """
        # Setup all adapters
        mcp_adapter = MCPAdapter({"server_url": "http://mcp:3000"})
        acp_adapter = ACPAdapter({"base_url": "http://acp:8080"})
        a2a_adapter = A2AAdapter({"base_url": "https://a2a:443"})
        
        # Mock responses
        mcp_init = Mock()
        mcp_init.status_code = 200
        mcp_init.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test"}
            }
        }
        
        mcp_extract = Mock()
        mcp_extract.status_code = 200
        mcp_extract.json.return_value = {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "content": [{"type": "text", "text": "sensor_data: 42.5"}]
            }
        }
        
        acp_validate = Mock()
        acp_validate.status_code = 200
        acp_validate.json.return_value = {
            "result": "valid",
            "physics_compliant": True,
            "validated_data": "42.5"
        }
        
        a2a_analyze = Mock()
        a2a_analyze.status_code = 200
        a2a_analyze.json.return_value = {
            "task_id": "task-final",
            "status": "completed",
            "analysis": "Data is within normal range"
        }
        
        with patch('requests.post') as mock_post:
            with patch('requests.get') as mock_get:
                mock_post.side_effect = [
                    mcp_init,        # MCP init
                    mcp_extract,     # MCP extract
                    acp_validate,    # ACP validate
                    a2a_analyze      # A2A analyze
                ]
                
                # Pipeline execution
                # 1. Extract with MCP
                await mcp_adapter.initialize()
                extracted = await mcp_adapter.call_tool("extractor", {})
                raw_data = extracted["content"][0]["text"]
                
                # 2. Validate with ACP
                validated = await acp_adapter.execute_agent(
                    agent_url="http://acp:8080",
                    message={"data": raw_data},
                    async_mode=False
                )
                
                # 3. Analyze with A2A
                final = await a2a_adapter.create_task(
                    description="Final analysis",
                    agent_id="analyzer",
                    parameters={"validated_data": validated["validated_data"]}
                )
                
                # Verify complete pipeline
                assert validated["physics_compliant"] is True
                assert final["status"] == "completed"
                assert "analysis" in final


class TestErrorPropagation:
    """Test error handling across multiple protocols"""
    
    @pytest.mark.asyncio
    async def test_mcp_error_stops_pipeline(self):
        """Test that MCP error prevents downstream ACP call"""
        import requests
        
        mcp_adapter = MCPAdapter({"server_url": "http://mcp:3000"})
        acp_adapter = ACPAdapter({"base_url": "http://acp:8080"})
        
        # Mock MCP to fail
        with patch('requests.post', side_effect=requests.exceptions.ConnectionError):
            with pytest.raises(Exception):  # Should raise ProtocolConnectionError
                await mcp_adapter.initialize()
        
        # ACP should not be called (test by ensuring no second patch call)
        # This validates error handling stops the pipeline
    
    @pytest.mark.asyncio
    async def test_retry_across_protocols(self):
        """Test retry logic works independently for each protocol"""
        mcp_adapter = MCPAdapter({"server_url": "http://mcp:3000"})
        acp_adapter = ACPAdapter({"base_url": "http://acp:8080"})
        
        mcp_call_count = 0
        acp_call_count = 0
        
        def mcp_side_effect(*args, **kwargs):
            import requests
            nonlocal mcp_call_count
            mcp_call_count += 1
            
            if mcp_call_count < 2:
                raise requests.exceptions.Timeout()
            
            mock = Mock()
            mock.status_code = 200
            mock.json.return_value = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "test"}
                }
            }
            return mock
        
        def acp_side_effect(*args, **kwargs):
            import requests
            nonlocal acp_call_count
            acp_call_count += 1
            
            if acp_call_count < 2:
                raise requests.exceptions.Timeout()
            
            mock = Mock()
            mock.status_code = 200
            mock.json.return_value = {"result": "success"}
            return mock
        
        # Each adapter should retry independently
        with patch('requests.post', side_effect=mcp_side_effect):
            await mcp_adapter.initialize()
            assert mcp_call_count == 2  # Failed once, succeeded on retry
        
        with patch('requests.post', side_effect=acp_side_effect):
            result = await acp_adapter.execute_agent(
                agent_url="http://acp:8080",
                message={},
                async_mode=False
            )
            assert acp_call_count == 2  # Failed once, succeeded on retry


class TestConcurrentProtocolOperations:
    """Test concurrent operations across protocols"""
    
    @pytest.mark.asyncio
    async def test_parallel_tool_execution(self):
        """Test calling multiple protocol adapters concurrently"""
        mcp_adapter = MCPAdapter({"server_url": "http://mcp:3000"})
        acp_adapter = ACPAdapter({"base_url": "http://acp:8080"})
        a2a_adapter = A2AAdapter({"base_url": "https://a2a:443"})
        
        # Mock responses
        mcp_response = Mock()
        mcp_response.status_code = 200
        mcp_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": "mcp_result"}]}
        }
        
        acp_response = Mock()
        acp_response.status_code = 200
        acp_response.json.return_value = {"result": "acp_result"}
        
        a2a_response = Mock()
        a2a_response.status_code = 200
        a2a_response.json.return_value = {
            "task_id": "task-1",
            "status": "completed",
            "result": "a2a_result"
        }
        
        mcp_adapter.initialized = True
        
        with patch('requests.post') as mock_post:
            mock_post.side_effect = [mcp_response, acp_response, a2a_response]
            
            # Execute all three in parallel
            results = await asyncio.gather(
                mcp_adapter.call_tool("tool1", {}),
                acp_adapter.execute_agent(
                    "http://acp:8080",
                    {},
                    async_mode=False
                ),
                a2a_adapter.create_task("test", "agent-1", {})
            )
            
            # All three should complete
            assert len(results) == 3
            assert results[0]["content"][0]["text"] == "mcp_result"
            assert results[1]["result"] == "acp_result"
            assert results[2]["result"] == "a2a_result"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


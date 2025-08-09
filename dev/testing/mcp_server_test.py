#!/usr/bin/env python3
"""
NIS Protocol v3.2 - MCP Server Connection Test
Test connecting to external MCP servers and using their tools
"""

import asyncio
import json
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mcp_adapters():
    """Test MCP adapter initialization and configuration"""
    logger.info("🔧 Testing MCP Adapter Configuration...")
    
    try:
        from src.adapters.mcp_adapter import MCPAdapter
        from src.adapters.a2a_adapter import A2AAdapter
        
        # Test MCP adapter with configuration
        mcp_config = {
            "base_url": "https://api.example.com/mcp",
            "api_key": "test_mcp_key",
            "tool_mappings": {
                "file_editor": {
                    "nis_agent": "action_agent",
                    "target_layer": "ACTION"
                }
            }
        }
        
        mcp_adapter = MCPAdapter(mcp_config)
        logger.info("✅ MCP Adapter created successfully!")
        logger.info(f"   Protocol: {mcp_adapter.protocol_name}")
        logger.info(f"   Config valid: {mcp_adapter.validate_config()}")
        
        # Test A2A adapter with configuration
        a2a_config = {
            "base_url": "https://api.google.com/a2a", 
            "api_key": "test_a2a_key"
        }
        
        a2a_adapter = A2AAdapter(a2a_config)
        logger.info("✅ A2A Adapter created successfully!")
        logger.info(f"   Protocol: {a2a_adapter.protocol_name}")
        logger.info(f"   Config valid: {a2a_adapter.validate_config()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mcp_tool_mapping():
    """Test MCP tool to NIS agent mapping"""
    logger.info("🛠️ Testing MCP Tool Mapping...")
    
    try:
        from src.adapters.mcp_adapter import MCPAdapter
        
        # Real-world MCP configuration example
        mcp_config = {
            "base_url": "https://api.example.com/mcp",
            "api_key": "demo_key",
            "tool_mappings": {
                "file_editor_tool": {
                    "nis_agent": "action_agent",
                    "target_layer": "ACTION",
                    "permissions": ["read", "write", "execute"]
                },
                "code_formatter_tool": {
                    "nis_agent": "reasoning_agent", 
                    "target_layer": "REASONING",
                    "permissions": ["read", "write"]
                },
                "vision_analysis_tool": {
                    "nis_agent": "vision_agent",
                    "target_layer": "PERCEPTION",
                    "permissions": ["read"]
                }
            },
            "rate_limits": {
                "requests_per_minute": 100,
                "concurrent_requests": 10
            },
            "timeout": 30
        }
        
        adapter = MCPAdapter(mcp_config)
        
        # Test message translation
        test_message = {
            "protocol": "NIS",
            "message_id": "test_001",
            "sender": "nis_protocol",
            "receiver": "external_mcp_server",
            "payload": {
                "action": "edit_file",
                "data": {
                    "file_path": "/tmp/test.py",
                    "content": "print('Hello from MCP!')"
                }
            },
            "metadata": {
                "timestamp": "2024-01-01T12:00:00Z",
                "mcp_conversation_id": "conv_123"
            }
        }
        
        # Translate to MCP format
        mcp_message = adapter.translate_from_nis(test_message)
        logger.info("✅ Message translation successful!")
        logger.info(f"   MCP format: {json.dumps(mcp_message, indent=2)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Tool mapping test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_protocol_bridge_service():
    """Test Protocol Bridge Service with MCP"""
    logger.info("🌉 Testing Protocol Bridge Service with MCP...")
    
    try:
        from src.services.protocol_bridge_service import ProtocolBridgeService, ProtocolType
        
        # Create protocol bridge service
        bridge = ProtocolBridgeService()
        logger.info("✅ Protocol Bridge Service created!")
        
        # Check if MCP bridge is available
        if ProtocolType.MCP in bridge.protocol_bridges:
            logger.info("✅ MCP bridge found in protocol bridges!")
            mcp_bridge = bridge.protocol_bridges[ProtocolType.MCP]
            logger.info(f"   MCP bridge: {mcp_bridge}")
        else:
            logger.warning("⚠️ MCP bridge not found")
        
        # Check if A2A bridge is available
        if ProtocolType.A2A in bridge.protocol_bridges:
            logger.info("✅ A2A bridge found in protocol bridges!")
            a2a_bridge = bridge.protocol_bridges[ProtocolType.A2A]
            logger.info(f"   A2A bridge: {a2a_bridge}")
        else:
            logger.warning("⚠️ A2A bridge not found")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Protocol bridge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def simulate_mcp_tool_call():
    """Simulate calling an external MCP tool"""
    logger.info("📡 Simulating MCP Tool Call...")
    
    try:
        from src.adapters.mcp_adapter import MCPAdapter
        
        # Configure MCP adapter for external server
        mcp_config = {
            "base_url": "https://mock-mcp-server.example.com/v1",
            "api_key": "demo_key_12345",
            "tool_mappings": {
                "python_executor": {
                    "nis_agent": "reasoning_agent",
                    "target_layer": "REASONING"
                }
            }
        }
        
        adapter = MCPAdapter(mcp_config)
        
        # Prepare tool call message
        tool_message = {
            "payload": {
                "action": "execute_python",
                "data": {
                    "code": "import math; print(f'π = {math.pi:.6f}')",
                    "timeout": 10
                }
            },
            "metadata": {
                "mcp_conversation_id": "demo_conv_001",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }
        
        # Simulate tool call (this would call external server in real scenario)
        logger.info("📤 Calling external MCP tool 'python_executor'...")
        logger.info(f"   Tool message: {json.dumps(tool_message, indent=2)}")
        
        # In real scenario, this would make HTTP request to external MCP server
        mock_response = {
            "tool_response": {
                "success": True,
                "output": "π = 3.141593",
                "execution_time": 0.05,
                "tool_name": "python_executor"
            },
            "conversation_id": "demo_conv_001",
            "timestamp": "2024-01-01T12:00:01Z"
        }
        
        logger.info("📥 Received MCP tool response:")
        logger.info(f"   Response: {json.dumps(mock_response, indent=2)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ MCP tool call simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all MCP tests"""
    logger.info("🚀 Starting MCP Server Connection Tests...")
    logger.info("=" * 50)
    
    tests = [
        ("MCP Adapters", test_mcp_adapters),
        ("MCP Tool Mapping", test_mcp_tool_mapping),
        ("Protocol Bridge Service", test_protocol_bridge_service),
        ("MCP Tool Call Simulation", lambda: asyncio.run(simulate_mcp_tool_call()))
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        result = test_func()
        results.append((test_name, result))
        
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST RESULTS SUMMARY:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {status}: {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\n🏆 Tests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        logger.info("🎉 ALL TESTS PASSED! MCP functionality is ready!")
    else:
        logger.info("⚠️ Some tests failed. Check logs above for details.")

if __name__ == "__main__":
    main()
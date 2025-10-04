"""
MCP (Model Context Protocol) Integration Example

Demonstrates how to use the NIS Protocol MCP adapter to:
1. Connect to an MCP server
2. Discover available tools and resources
3. Execute tools
4. Read resources
5. Handle errors gracefully

This example shows real-world usage patterns with proper error handling.
"""

import asyncio
import logging
from src.adapters.mcp_adapter import MCPAdapter
from src.adapters.protocol_errors import (
    ProtocolConnectionError,
    ProtocolTimeoutError,
    ProtocolValidationError
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def mcp_workflow_example():
    """Complete MCP workflow example"""
    
    print("=" * 70)
    print("NIS Protocol - MCP Integration Example")
    print("=" * 70 + "\n")
    
    # Configure MCP adapter
    config = {
        "server_url": "http://localhost:3000",  # Your MCP server URL
        "timeout": 30,
        "failure_threshold": 5,
        "recovery_timeout": 60
    }
    
    # Create adapter
    adapter = MCPAdapter(config)
    
    try:
        # =====================================================================
        # Step 1: Initialize Connection
        # =====================================================================
        print("Step 1: Initializing MCP connection...")
        
        init_result = await adapter.initialize()
        
        print(f"✅ Connected to MCP server")
        print(f"   Server: {init_result.get('serverInfo', {}).get('name', 'Unknown')}")
        print(f"   Capabilities: {list(init_result.get('capabilities', {}).keys())}")
        print()
        
        # =====================================================================
        # Step 2: Discover Tools
        # =====================================================================
        print("Step 2: Discovering available tools...")
        
        await adapter.discover_tools()
        
        print(f"✅ Found {len(adapter.tools_registry)} tools:")
        for tool_name, tool_info in adapter.tools_registry.items():
            print(f"   - {tool_name}: {tool_info.get('description', 'No description')}")
        print()
        
        # =====================================================================
        # Step 3: Execute a Tool
        # =====================================================================
        if adapter.tools_registry:
            print("Step 3: Executing a tool...")
            
            # Get first available tool
            tool_name = list(adapter.tools_registry.keys())[0]
            
            # Example: Call a calculator tool
            result = await adapter.call_tool(
                tool_name,
                {"expression": "2 + 2"}  # Adjust parameters for your tool
            )
            
            print(f"✅ Tool '{tool_name}' executed successfully")
            print(f"   Result: {result.get('content', [{}])[0].get('text', 'No output')}")
            print()
        else:
            print("⚠️  No tools available to execute")
            print()
        
        # =====================================================================
        # Step 4: Discover Resources
        # =====================================================================
        print("Step 4: Discovering available resources...")
        
        await adapter.discover_resources()
        
        if adapter.resources_registry:
            print(f"✅ Found {len(adapter.resources_registry)} resources:")
            for uri, resource in adapter.resources_registry.items():
                print(f"   - {resource.get('name', 'Unnamed')}: {uri}")
            print()
            
            # =====================================================================
            # Step 5: Read a Resource
            # =====================================================================
            print("Step 5: Reading a resource...")
            
            # Get first available resource
            resource_uri = list(adapter.resources_registry.keys())[0]
            
            resource_content = await adapter.read_resource(resource_uri)
            
            print(f"✅ Resource '{resource_uri}' read successfully")
            content_preview = str(resource_content.get('contents', [{}])[0].get('text', ''))[:100]
            print(f"   Content preview: {content_preview}...")
            print()
        else:
            print("⚠️  No resources available")
            print()
        
        # =====================================================================
        # Step 6: Health Check
        # =====================================================================
        print("Step 6: Checking adapter health...")
        
        health = adapter.get_health_status()
        
        print(f"✅ Adapter Health Status:")
        print(f"   Healthy: {health['healthy']}")
        print(f"   Circuit Breaker: {health['circuit_breaker']['state']}")
        print(f"   Success Rate: {health['metrics']['success_rate']:.1%}")
        print(f"   Total Requests: {health['metrics']['total_requests']}")
        print()
        
    except ProtocolConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print("   Make sure your MCP server is running at the configured URL")
        
    except ProtocolTimeoutError as e:
        print(f"❌ Timeout Error: {e}")
        print(f"   Request timed out after {e.timeout}s")
        
    except ProtocolValidationError as e:
        print(f"❌ Validation Error: {e}")
        print("   The server response didn't match expected format")
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        
    finally:
        print("=" * 70)
        print("MCP Example Complete")
        print("=" * 70)


async def mcp_error_handling_example():
    """Demonstrate MCP error handling and retry logic"""
    
    print("\n" + "=" * 70)
    print("MCP Error Handling Example")
    print("=" * 70 + "\n")
    
    config = {
        "server_url": "http://localhost:3000",
        "timeout": 5  # Short timeout for demonstration
    }
    
    adapter = MCPAdapter(config)
    
    try:
        # This will demonstrate retry logic if server is slow
        print("Attempting to initialize (with retry logic)...")
        await adapter.initialize()
        print("✅ Initialization successful (possibly after retries)")
        
    except ProtocolTimeoutError as e:
        print(f"❌ Still timed out after retries: {e}")
        print(f"   Circuit breaker state: {adapter.circuit_breaker.state.value}")
        
        # Show metrics
        metrics = adapter.metrics.to_dict()
        print(f"   Failed attempts: {metrics['failed_requests']}")
        print(f"   Error types: {metrics.get('error_types', {})}")


async def mcp_circuit_breaker_example():
    """Demonstrate circuit breaker behavior"""
    
    print("\n" + "=" * 70)
    print("MCP Circuit Breaker Example")
    print("=" * 70 + "\n")
    
    config = {
        "server_url": "http://nonexistent:3000",  # Intentionally wrong
        "failure_threshold": 3,
        "timeout": 1
    }
    
    adapter = MCPAdapter(config)
    
    print("Triggering circuit breaker with repeated failures...")
    
    for attempt in range(5):
        try:
            await adapter.initialize()
        except Exception as e:
            state = adapter.circuit_breaker.get_state()
            print(f"   Attempt {attempt + 1}: {type(e).__name__}")
            print(f"   Circuit state: {state['state']}")
            
            if state['state'] == 'open':
                print("   ⚠️  Circuit breaker opened - preventing further requests")
                break
    
    # Show how to reset
    print("\nResetting circuit breaker...")
    adapter.reset_circuit_breaker()
    print(f"✅ Circuit state: {adapter.circuit_breaker.state.value}")


def main():
    """Run all examples"""
    asyncio.run(mcp_workflow_example())
    asyncio.run(mcp_error_handling_example())
    asyncio.run(mcp_circuit_breaker_example())


if __name__ == "__main__":
    main()


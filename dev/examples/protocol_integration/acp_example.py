"""
ACP (Agent Communication Protocol) Integration Example

Demonstrates how to use the NIS Protocol ACP adapter to:
1. Export Agent Card for offline discovery
2. Execute agents in sync/async modes
3. Translate between NIS and ACP message formats
4. Handle long-running async operations
5. Monitor adapter health

This example shows real-world usage patterns for IBM's ACP protocol.
"""

import asyncio
import json
import logging
from src.adapters.acp_adapter import ACPAdapter
from src.adapters.protocol_errors import (
    ProtocolConnectionError,
    ProtocolTimeoutError
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def acp_agent_card_example():
    """Demonstrate Agent Card export for offline discovery"""
    
    print("=" * 70)
    print("NIS Protocol - ACP Agent Card Example")
    print("=" * 70 + "\n")
    
    # Configure ACP adapter
    config = {
        "base_url": "http://localhost:5000",
        "api_key": "your-api-key"  # Optional
    }
    
    adapter = ACPAdapter(config)
    
    # =========================================================================
    # Export Agent Card
    # =========================================================================
    print("Exporting NIS Protocol Agent Card...")
    print("(This enables offline discovery per IBM ACP spec)\n")
    
    agent_card = adapter.export_agent_card()
    
    print("✅ Agent Card generated:")
    print(json.dumps(agent_card, indent=2))
    print()
    
    # Show key information
    agent_info = agent_card["acp"]["agent"]
    
    print("Agent Information:")
    print(f"   ID: {agent_info['id']}")
    print(f"   Name: {agent_info['name']}")
    print(f"   Version: {agent_info['version']}")
    print()
    
    print("Capabilities:")
    for capability in agent_info["capabilities"]:
        print(f"   - {capability}")
    print()
    
    print("Endpoints:")
    for name, path in agent_info["endpoints"].items():
        if name == "base":
            print(f"   {name}: {path}")
        else:
            print(f"   {name}: {agent_info['endpoints']['base']}{path}")
    print()
    
    print("Pipeline Stages:")
    for stage in agent_info["metadata"]["pipeline_stages"]:
        print(f"   {stage.upper()}")
    print()
    
    # =========================================================================
    # Save to package.json for npm distribution
    # =========================================================================
    print("💡 Tip: Embed this Agent Card in package.json:")
    print('   "acp": { ... }')
    print("   This enables scale-to-zero and offline discovery.\n")


async def acp_sync_execution_example():
    """Demonstrate synchronous agent execution"""
    
    print("\n" + "=" * 70)
    print("ACP Synchronous Execution Example")
    print("=" * 70 + "\n")
    
    config = {
        "base_url": "http://localhost:5000"
    }
    
    adapter = ACPAdapter(config)
    
    try:
        print("Executing ACP agent (synchronous mode)...")
        
        result = await adapter.execute_agent(
            agent_url="http://external-acp-agent:8080",
            message={
                "query": "Validate physics constraints for system state",
                "state": {
                    "temperature": 298.15,
                    "pressure": 101325,
                    "volume": 0.0224
                }
            },
            async_mode=False  # Synchronous execution
        )
        
        print("✅ Execution complete")
        print(f"   Result: {result.get('result', 'No result')}")
        if "output" in result:
            print(f"   Output: {result['output']}")
        print()
        
    except ProtocolConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print("   Make sure the ACP agent is running")
        
    except ProtocolTimeoutError as e:
        print(f"❌ Timeout: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def acp_async_execution_example():
    """Demonstrate asynchronous agent execution with polling"""
    
    print("\n" + "=" * 70)
    print("ACP Asynchronous Execution Example")
    print("=" * 70 + "\n")
    
    config = {
        "base_url": "http://localhost:5000"
    }
    
    adapter = ACPAdapter(config)
    
    try:
        print("Executing ACP agent (asynchronous mode)...")
        print("This is the default mode per ACP spec\n")
        
        result = await adapter.execute_agent(
            agent_url="http://external-acp-agent:8080",
            message={
                "task": "Run physics simulation",
                "parameters": {
                    "simulation_type": "molecular_dynamics",
                    "timesteps": 100000,
                    "temperature": 300
                }
            },
            async_mode=True  # Asynchronous execution (default)
        )
        
        print("✅ Async execution complete")
        
        if "task_id" in result:
            print(f"   Task ID: {result['task_id']}")
        
        print(f"   Status: {result.get('status', 'completed')}")
        
        if "result" in result:
            print(f"   Result: {result['result']}")
        
        print()
        
    except ProtocolTimeoutError as e:
        print(f"❌ Task timed out: {e}")
        print("   Increase timeout or check task status manually")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def acp_message_translation_example():
    """Demonstrate message translation between NIS and ACP"""
    
    print("\n" + "=" * 70)
    print("ACP Message Translation Example")
    print("=" * 70 + "\n")
    
    config = {"base_url": "http://localhost:5000"}
    adapter = ACPAdapter(config)
    
    # =========================================================================
    # Example 1: ACP to NIS
    # =========================================================================
    print("Example 1: Translating ACP message to NIS format\n")
    
    acp_message = {
        "headers": {
            "message_id": "msg-abc-123",
            "sender_id": "external-agent",
            "receiver_id": "nis_protocol",
            "action": "physics_validation_request",
            "conversation_id": "conv-xyz-789",
            "timestamp": 1696118400000
        },
        "body": {
            "system_state": {
                "energy": 1.5e-19,
                "momentum": [0.1, 0.2, 0.0]
            },
            "constraints": ["energy_conservation", "momentum_conservation"],
            "emotional_state": {
                "valence": 0.7,
                "arousal": 0.4
            }
        }
    }
    
    nis_message = adapter.translate_to_nis(acp_message)
    
    print("ACP Message:")
    print(json.dumps(acp_message, indent=2))
    print()
    
    print("Translated to NIS:")
    print(json.dumps(nis_message, indent=2))
    print()
    
    # =========================================================================
    # Example 2: NIS to ACP
    # =========================================================================
    print("\nExample 2: Translating NIS message to ACP format\n")
    
    nis_message_out = {
        "protocol": "nis",
        "timestamp": 1696118500.0,
        "payload": {
            "action": "validation_result",
            "data": {
                "valid": True,
                "physics_compliant": True,
                "constraints_satisfied": ["energy_conservation", "momentum_conservation"],
                "confidence": 0.95
            }
        },
        "metadata": {
            "acp_message_id": "msg-abc-123",
            "acp_sender_id": "external-agent",
            "acp_conversation_id": "conv-xyz-789"
        },
        "emotional_state": {
            "valence": 0.8,
            "arousal": 0.3
        }
    }
    
    acp_message_out = adapter.translate_from_nis(nis_message_out)
    
    print("NIS Message:")
    print(json.dumps(nis_message_out, indent=2))
    print()
    
    print("Translated to ACP:")
    print(json.dumps(acp_message_out, indent=2))
    print()


async def acp_send_to_external_agent_example():
    """Demonstrate sending messages to external ACP agents"""
    
    print("\n" + "=" * 70)
    print("ACP External Agent Communication Example")
    print("=" * 70 + "\n")
    
    config = {
        "base_url": "http://localhost:5000",
        "api_key": "your-api-key"
    }
    
    adapter = ACPAdapter(config)
    
    try:
        # Create a NIS message
        nis_message = {
            "protocol": "nis",
            "timestamp": 1696118600.0,
            "payload": {
                "action": "analyze",
                "data": {
                    "input": "sensor data",
                    "analysis_type": "anomaly_detection"
                }
            },
            "metadata": {}
        }
        
        print("Sending message to external ACP agent...")
        print(f"Message: {nis_message['payload']['action']}\n")
        
        response = adapter.send_to_external_agent(
            agent_id="quality-assurance-agent",
            message=nis_message
        )
        
        print("✅ Response received:")
        print(json.dumps(response, indent=2))
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def acp_health_monitoring_example():
    """Demonstrate health monitoring and metrics"""
    
    print("\n" + "=" * 70)
    print("ACP Health Monitoring Example")
    print("=" * 70 + "\n")
    
    config = {"base_url": "http://localhost:5000"}
    adapter = ACPAdapter(config)
    
    # Simulate some requests
    print("Simulating requests to generate metrics...\n")
    
    adapter.metrics.record_request(True, 0.5)
    adapter.metrics.record_request(True, 0.3)
    adapter.metrics.record_request(False, 1.2, "timeout")
    adapter.metrics.record_request(True, 0.4)
    
    # Get health status
    health = adapter.get_health_status()
    
    print("Adapter Health Status:")
    print(json.dumps(health, indent=2))
    print()
    
    # Show key metrics
    print("Key Metrics:")
    print(f"   Protocol: {health['protocol']}")
    print(f"   Healthy: {'✅ Yes' if health['healthy'] else '❌ No'}")
    print(f"   Circuit Breaker: {health['circuit_breaker']['state']}")
    print(f"   Success Rate: {health['metrics']['success_rate']:.1%}")
    print(f"   Avg Response Time: {health['metrics']['avg_response_time']:.3f}s")
    print(f"   Total Requests: {health['metrics']['total_requests']}")
    print()
    
    # Reset metrics
    print("Resetting metrics...")
    adapter.reset_metrics()
    
    new_health = adapter.get_health_status()
    print(f"✅ Metrics reset. Total requests now: {new_health['metrics']['total_requests']}")


def main():
    """Run all examples"""
    print("\n" + "🔧 " * 30 + "\n")
    
    asyncio.run(acp_agent_card_example())
    asyncio.run(acp_sync_execution_example())
    asyncio.run(acp_async_execution_example())
    asyncio.run(acp_message_translation_example())
    asyncio.run(acp_send_to_external_agent_example())
    asyncio.run(acp_health_monitoring_example())
    
    print("\n" + "=" * 70)
    print("All ACP Examples Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()


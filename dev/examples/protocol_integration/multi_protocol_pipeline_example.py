"""
Multi-Protocol Pipeline Example

Demonstrates a complete real-world workflow using all three protocols:
MCP, A2A, and ACP working together within the NIS Protocol.

Real-World Scenario:
===================
Industrial IoT Quality Assurance Pipeline

1. MCP: Extract sensor data from manufacturing equipment
2. ACP: Validate physics constraints and detect anomalies  
3. A2A: Generate comprehensive analysis report and recommendations

This demonstrates true protocol interoperability and the power of the
NIS Protocol's unified agent communication architecture.
"""

import asyncio
import json
import logging
from datetime import datetime
from src.adapters.mcp_adapter import MCPAdapter
from src.adapters.a2a_adapter import A2AAdapter
from src.adapters.acp_adapter import ACPAdapter
from src.adapters.protocol_errors import (
    ProtocolConnectionError,
    ProtocolTimeoutError
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def industrial_quality_assurance_pipeline():
    """
    Complete multi-protocol pipeline for industrial quality assurance
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + "MULTI-PROTOCOL PIPELINE EXAMPLE")
    print(" " * 15 + "Industrial IoT Quality Assurance System")
    print("=" * 80 + "\n")
    
    # =========================================================================
    # Setup: Initialize all three protocol adapters
    # =========================================================================
    print("🔧 SETUP: Initializing protocol adapters...")
    print("-" * 80)
    
    mcp_adapter = MCPAdapter({
        "server_url": "http://factory-mcp-server:3000",
        "timeout": 30
    })
    
    acp_adapter = ACPAdapter({
        "base_url": "http://physics-validator-acp:8080",
        "api_key": "physics-validator-key"
    })
    
    a2a_adapter = A2AAdapter({
        "base_url": "https://analytics-a2a-service:443",
        "api_key": "analytics-service-key"
    })
    
    print("✅ MCP Adapter: Ready (Manufacturing data extraction)")
    print("✅ ACP Adapter: Ready (Physics validation)")
    print("✅ A2A Adapter: Ready (Analytics & reporting)")
    print()
    
    try:
        # =====================================================================
        # STAGE 1: MCP - Extract Sensor Data
        # =====================================================================
        print("📊 STAGE 1: Extracting sensor data via MCP")
        print("-" * 80)
        
        # Initialize MCP connection
        print("   → Connecting to factory MCP server...")
        await mcp_adapter.initialize()
        print("   ✓ Connected")
        
        # Discover available data sources
        print("   → Discovering data extraction tools...")
        await mcp_adapter.discover_tools()
        print(f"   ✓ Found {len(mcp_adapter.tools_registry)} tools")
        
        # Extract sensor data
        print("   → Executing 'sensor_data_extractor' tool...")
        sensor_data_result = await mcp_adapter.call_tool(
            "sensor_data_extractor",
            {
                "equipment_id": "PRESS-MACHINE-A42",
                "time_range": "last_hour",
                "metrics": ["pressure", "temperature", "vibration", "speed"]
            }
        )
        
        # Parse extracted data
        sensor_data_text = sensor_data_result["content"][0]["text"]
        # In real scenario, this would be JSON from the MCP server
        sensor_data = {
            "equipment_id": "PRESS-MACHINE-A42",
            "timestamp": datetime.now().isoformat(),
            "readings": {
                "pressure": 145.2,  # PSI
                "temperature": 185.5,  # °F
                "vibration": 0.8,  # mm/s
                "speed": 1200  # RPM
            }
        }
        
        print(f"   ✓ Extracted data from {sensor_data['equipment_id']}")
        print(f"     • Pressure: {sensor_data['readings']['pressure']} PSI")
        print(f"     • Temperature: {sensor_data['readings']['temperature']} °F")
        print(f"     • Vibration: {sensor_data['readings']['vibration']} mm/s")
        print(f"     • Speed: {sensor_data['readings']['speed']} RPM")
        print()
        
        # =====================================================================
        # STAGE 2: ACP - Physics Validation
        # =====================================================================
        print("🔬 STAGE 2: Validating physics constraints via ACP")
        print("-" * 80)
        
        # Prepare message for ACP agent
        print("   → Sending data to physics validation agent...")
        validation_result = await acp_adapter.execute_agent(
            agent_url="http://physics-validator-acp:8080",
            message={
                "action": "validate_operating_conditions",
                "equipment_type": "hydraulic_press",
                "sensor_readings": sensor_data["readings"],
                "constraints": [
                    "thermodynamic_equilibrium",
                    "pressure_limits",
                    "vibration_safety"
                ]
            },
            async_mode=False  # Synchronous for fast validation
        )
        
        physics_valid = validation_result.get("physics_compliant", False)
        anomalies = validation_result.get("anomalies", [])
        
        print(f"   ✓ Physics validation complete")
        print(f"     • Physics compliant: {'✅ Yes' if physics_valid else '❌ No'}")
        
        if anomalies:
            print(f"     • ⚠️  Anomalies detected: {len(anomalies)}")
            for anomaly in anomalies:
                print(f"       - {anomaly}")
        else:
            print(f"     • ✅ No anomalies detected")
        
        print()
        
        # =====================================================================
        # STAGE 3: A2A - Generate Analysis Report
        # =====================================================================
        print("📈 STAGE 3: Generating comprehensive analysis via A2A")
        print("-" * 80)
        
        # Create long-running A2A task for detailed analysis
        print("   → Creating analysis task...")
        analysis_task = await a2a_adapter.create_task(
            description="Generate equipment health and quality assurance report",
            agent_id="industrial-analytics-agent",
            parameters={
                "sensor_data": sensor_data,
                "validation_result": validation_result,
                "include_recommendations": True,
                "include_predictive_maintenance": True,
                "report_format": "comprehensive"
            }
        )
        
        task_id = analysis_task["task_id"]
        print(f"   ✓ Task created: {task_id}")
        
        # Wait for analysis completion
        print("   → Waiting for analysis completion (this may take a moment)...")
        completed_task = await a2a_adapter.wait_for_task_completion(
            task_id,
            poll_interval=2.0,
            timeout=60.0
        )
        
        analysis_report = completed_task.get("result", {})
        
        print(f"   ✓ Analysis complete")
        print()
        
        # =====================================================================
        # RESULTS: Display Complete Pipeline Output
        # =====================================================================
        print("📋 PIPELINE RESULTS")
        print("=" * 80)
        
        print("\n1. Data Extraction (MCP):")
        print(f"   Equipment: {sensor_data['equipment_id']}")
        print(f"   Timestamp: {sensor_data['timestamp']}")
        print(f"   Metrics collected: {len(sensor_data['readings'])}")
        
        print("\n2. Physics Validation (ACP):")
        print(f"   Status: {'✅ PASS' if physics_valid else '❌ FAIL'}")
        print(f"   Constraints checked: {len(validation_result.get('constraints', []))}")
        print(f"   Anomalies: {len(anomalies)}")
        
        print("\n3. Comprehensive Analysis (A2A):")
        print(f"   Task ID: {task_id}")
        print(f"   Status: {completed_task['status']}")
        
        # Display analysis findings
        if "health_score" in analysis_report:
            print(f"   Equipment Health Score: {analysis_report['health_score']}/100")
        
        if "recommendations" in analysis_report:
            print(f"\n   Recommendations:")
            for i, rec in enumerate(analysis_report["recommendations"][:3], 1):
                print(f"      {i}. {rec}")
        
        if "artifacts" in completed_task:
            print(f"\n   Generated Reports: {len(completed_task['artifacts'])}")
            for artifact in completed_task["artifacts"]:
                print(f"      • {artifact.get('name', 'Unnamed report')}")
        
        print()
        
        # =====================================================================
        # HEALTH METRICS: Show adapter performance
        # =====================================================================
        print("💚 ADAPTER HEALTH METRICS")
        print("=" * 80)
        
        mcp_health = mcp_adapter.get_health_status()
        acp_health = acp_adapter.get_health_status()
        a2a_health = a2a_adapter.get_health_status()
        
        print(f"\nMCP Adapter:")
        print(f"   Success Rate: {mcp_health['metrics']['success_rate']:.1%}")
        print(f"   Avg Response Time: {mcp_health['metrics']['avg_response_time']:.3f}s")
        
        print(f"\nACP Adapter:")
        print(f"   Success Rate: {acp_health['metrics']['success_rate']:.1%}")
        print(f"   Avg Response Time: {acp_health['metrics']['avg_response_time']:.3f}s")
        
        print(f"\nA2A Adapter:")
        print(f"   Success Rate: {a2a_health['metrics']['success_rate']:.1%}")
        print(f"   Avg Response Time: {a2a_health['metrics']['avg_response_time']:.3f}s")
        print(f"   Active Tasks: {a2a_health['active_tasks_count']}")
        
        print()
        
        # =====================================================================
        # SUCCESS
        # =====================================================================
        print("=" * 80)
        print("✅ MULTI-PROTOCOL PIPELINE COMPLETE")
        print("=" * 80)
        print("\nSuccessfully demonstrated:")
        print("   • MCP for data extraction")
        print("   • ACP for physics validation")
        print("   • A2A for long-running analysis")
        print("   • Seamless protocol interoperability")
        print()
        
    except ProtocolConnectionError as e:
        print(f"\n❌ CONNECTION ERROR: {e}")
        print("   Ensure all protocol servers are running and accessible")
        
    except ProtocolTimeoutError as e:
        print(f"\n❌ TIMEOUT ERROR: {e}")
        print("   One of the operations took too long")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.exception("Pipeline error")


async def concurrent_multi_protocol_example():
    """
    Demonstrate running multiple protocols concurrently for maximum efficiency
    """
    
    print("\n" + "=" * 80)
    print(" " * 25 + "CONCURRENT PROTOCOL EXECUTION")
    print("=" * 80 + "\n")
    
    # Setup adapters
    mcp = MCPAdapter({"server_url": "http://mcp:3000"})
    acp = ACPAdapter({"base_url": "http://acp:8080"})
    a2a = A2AAdapter({"base_url": "https://a2a:443"})
    
    print("Running three independent operations in parallel...\n")
    
    # Initialize MCP first
    await mcp.initialize()
    
    try:
        # Run all three protocols concurrently
        results = await asyncio.gather(
            # MCP: Extract multiple datasets
            mcp.call_tool("data_extractor", {"source": "sensors"}),
            
            # ACP: Validate system state
            acp.execute_agent(
                "http://acp:8080",
                {"action": "validate", "data": {}},
                async_mode=False
            ),
            
            # A2A: Run background analysis
            a2a.create_task(
                "Background analysis",
                "analyzer-1",
                {"type": "trending"}
            )
        )
        
        print("✅ All three operations completed concurrently")
        print(f"   MCP result: {type(results[0])}")
        print(f"   ACP result: {type(results[1])}")
        print(f"   A2A result: {type(results[2])}")
        print(f"\n   Total execution time: reduced by parallel processing")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all multi-protocol examples"""
    
    print("\n" + "🌐 " * 35)
    print("\n" + " " * 20 + "NIS PROTOCOL - MULTI-PROTOCOL INTEGRATION")
    print(" " * 25 + "Complete Workflow Demonstration")
    print("\n" + "🌐 " * 35)
    
    # Run main pipeline
    asyncio.run(industrial_quality_assurance_pipeline())
    
    # Run concurrent example
    asyncio.run(concurrent_multi_protocol_example())
    
    print("\n" + "=" * 80)
    print(" " * 30 + "ALL EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nThe NIS Protocol successfully integrates:")
    print("   • MCP (Model Context Protocol) - Anthropic")
    print("   • A2A (Agent2Agent Protocol) - Google")
    print("   • ACP (Agent Communication Protocol) - IBM")
    print("\nEnabling true multi-vendor AI agent interoperability.")
    print()


if __name__ == "__main__":
    main()


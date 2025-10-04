"""
A2A (Agent2Agent Protocol) Integration Example

Demonstrates how to use the NIS Protocol A2A adapter to:
1. Create tasks on external A2A agents
2. Monitor task progress
3. Handle long-running operations
4. Use UX negotiation for rich content
5. Cancel tasks when needed

This example shows real-world usage patterns for Google's A2A protocol.
"""

import asyncio
import logging
from src.adapters.a2a_adapter import A2AAdapter
from src.adapters.protocol_errors import (
    ProtocolConnectionError,
    ProtocolTimeoutError
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def a2a_task_workflow_example():
    """Complete A2A task workflow example"""
    
    print("=" * 70)
    print("NIS Protocol - A2A Integration Example")
    print("=" * 70 + "\n")
    
    # Configure A2A adapter
    config = {
        "base_url": "https://api.google.com/a2a/v1",  # Example URL
        "api_key": "your-api-key-here",  # Replace with actual key
        "timeout": 30
    }
    
    # Create adapter
    adapter = A2AAdapter(config)
    
    try:
        # =====================================================================
        # Step 1: Create a Task
        # =====================================================================
        print("Step 1: Creating an A2A task...")
        
        task = await adapter.create_task(
            description="Analyze customer feedback and generate summary report",
            agent_id="analyzer-agent-123",
            parameters={
                "dataset": "customer_feedback_q4_2024",
                "analysis_type": "sentiment",
                "include_insights": True
            },
            callback_url="https://your-server.com/callbacks/task-updates"
        )
        
        task_id = task["task_id"]
        print(f"✅ Task created successfully")
        print(f"   Task ID: {task_id}")
        print(f"   Status: {task['status']}")
        print()
        
        # =====================================================================
        # Step 2: Check Task Status
        # =====================================================================
        print("Step 2: Checking task status...")
        
        status = await adapter.get_task_status(task_id)
        
        print(f"✅ Task status retrieved")
        print(f"   Status: {status['status']}")
        if "progress" in status:
            print(f"   Progress: {status['progress']}%")
        print()
        
        # =====================================================================
        # Step 3: Wait for Task Completion
        # =====================================================================
        print("Step 3: Waiting for task completion...")
        print("   (Polling every 2 seconds...)")
        
        completed_task = await adapter.wait_for_task_completion(
            task_id,
            poll_interval=2.0,
            timeout=60.0
        )
        
        print(f"✅ Task completed!")
        print(f"   Final status: {completed_task['status']}")
        if "result" in completed_task:
            print(f"   Result: {str(completed_task['result'])[:100]}...")
        if "artifacts" in completed_task:
            print(f"   Artifacts: {len(completed_task['artifacts'])} files")
        print()
        
    except ProtocolConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print("   Make sure the A2A service is accessible")
        
    except ProtocolTimeoutError as e:
        print(f"❌ Timeout Error: {e}")
        print("   Task took too long to complete")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Show health status
        health = adapter.get_health_status()
        print(f"\nAdapter Health:")
        print(f"   Active tasks: {health['active_tasks_count']}")
        print(f"   Success rate: {health['metrics']['success_rate']:.1%}")


async def a2a_ux_negotiation_example():
    """Demonstrate UX negotiation with multi-format messages"""
    
    print("\n" + "=" * 70)
    print("A2A UX Negotiation Example")
    print("=" * 70 + "\n")
    
    config = {
        "base_url": "https://api.google.com/a2a/v1",
        "api_key": "your-api-key-here"
    }
    
    adapter = A2AAdapter(config)
    
    # =========================================================================
    # Example 1: Text-only message
    # =========================================================================
    print("Example 1: Simple text message")
    
    text_message = adapter.create_message_with_parts([
        {
            "type": "text",
            "content": "Please analyze this dataset and provide insights."
        }
    ])
    
    print(f"✅ Created message with {len(text_message['parts'])} part(s)")
    print(f"   Type: {text_message['parts'][0]['type']}")
    print()
    
    # =========================================================================
    # Example 2: Rich multi-format message
    # =========================================================================
    print("Example 2: Rich multi-format message")
    
    rich_message = adapter.create_message_with_parts([
        {
            "type": "text",
            "content": "Here are the analysis results:"
        },
        {
            "type": "image",
            "url": "https://example.com/charts/sentiment-analysis.png",
            "alt_text": "Sentiment analysis chart"
        },
        {
            "type": "data",
            "content": {
                "positive": 65,
                "neutral": 25,
                "negative": 10
            }
        },
        {
            "type": "iframe",
            "url": "https://example.com/embed/interactive-dashboard",
            "width": 800,
            "height": 600
        }
    ])
    
    print(f"✅ Created rich message with {len(rich_message['parts'])} parts:")
    for i, part in enumerate(rich_message['parts'], 1):
        print(f"   {i}. {part['type']}")
    print()
    
    # =========================================================================
    # Example 3: Structured data for AI processing
    # =========================================================================
    print("Example 3: Structured data message")
    
    data_message = adapter.create_message_with_parts([
        {
            "type": "text",
            "content": "Customer feedback summary"
        },
        {
            "type": "data",
            "content": {
                "summary": {
                    "total_responses": 1250,
                    "avg_satisfaction": 4.2,
                    "top_issues": [
                        "slow response time",
                        "unclear documentation",
                        "missing features"
                    ]
                },
                "recommendations": [
                    "Improve support response SLA",
                    "Expand documentation with examples",
                    "Prioritize feature requests"
                ]
            }
        }
    ])
    
    print(f"✅ Created structured data message")
    print(f"   Data preview: {list(data_message['parts'][1]['content'].keys())}")
    print()


async def a2a_task_cancellation_example():
    """Demonstrate task cancellation"""
    
    print("\n" + "=" * 70)
    print("A2A Task Cancellation Example")
    print("=" * 70 + "\n")
    
    config = {
        "base_url": "https://api.google.com/a2a/v1",
        "api_key": "your-api-key-here"
    }
    
    adapter = A2AAdapter(config)
    
    try:
        # Create a long-running task
        print("Creating a long-running task...")
        
        task = await adapter.create_task(
            description="Process large dataset",
            agent_id="processor-agent-456",
            parameters={"dataset_size": "large"}
        )
        
        task_id = task["task_id"]
        print(f"✅ Task created: {task_id}")
        print()
        
        # Wait a moment
        await asyncio.sleep(2)
        
        # Cancel the task
        print(f"Cancelling task {task_id}...")
        
        cancelled = await adapter.cancel_task(task_id)
        
        print(f"✅ Task cancelled")
        print(f"   Status: {cancelled['status']}")
        print(f"   Task removed from active tasks: {task_id not in adapter.active_tasks}")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def a2a_multiple_tasks_example():
    """Demonstrate managing multiple concurrent tasks"""
    
    print("\n" + "=" * 70)
    print("A2A Multiple Tasks Example")
    print("=" * 70 + "\n")
    
    config = {
        "base_url": "https://api.google.com/a2a/v1",
        "api_key": "your-api-key-here"
    }
    
    adapter = A2AAdapter(config)
    
    try:
        # Create multiple tasks
        print("Creating 3 concurrent tasks...")
        
        tasks = await asyncio.gather(
            adapter.create_task(
                "Analyze Q1 data",
                "analyzer-1",
                {"quarter": "Q1"}
            ),
            adapter.create_task(
                "Analyze Q2 data",
                "analyzer-2",
                {"quarter": "Q2"}
            ),
            adapter.create_task(
                "Analyze Q3 data",
                "analyzer-3",
                {"quarter": "Q3"}
            )
        )
        
        print(f"✅ Created {len(tasks)} tasks")
        for i, task in enumerate(tasks, 1):
            print(f"   {i}. {task['task_id']} - {task['status']}")
        print()
        
        print(f"Active tasks: {len(adapter.active_tasks)}")
        
        # Get status of all tasks
        print("\nChecking status of all tasks...")
        
        statuses = await asyncio.gather(
            *[adapter.get_task_status(task["task_id"]) for task in tasks]
        )
        
        for status in statuses:
            print(f"   {status['task_id']}: {status['status']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all examples"""
    print("\n" + "🚀 " * 30 + "\n")
    
    asyncio.run(a2a_task_workflow_example())
    asyncio.run(a2a_ux_negotiation_example())
    asyncio.run(a2a_task_cancellation_example())
    asyncio.run(a2a_multiple_tasks_example())
    
    print("\n" + "=" * 70)
    print("All A2A Examples Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()


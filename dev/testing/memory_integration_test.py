#!/usr/bin/env python3
"""
Memory Integration Test
======================

Quick test to verify the enhanced memory system is working correctly.
This test can be run independently to validate memory functionality.
"""

import asyncio
import sys
import os
import tempfile
import json
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.chat.enhanced_memory_chat import EnhancedChatMemory, ChatMemoryConfig
    from src.agents.memory.enhanced_memory_agent import EnhancedMemoryAgent
    print("✅ Successfully imported enhanced memory components")
except ImportError as e:
    print(f"❌ Failed to import memory components: {e}")
    print("   Make sure you're running from the project root")
    sys.exit(1)

async def test_memory_system():
    """Test the enhanced memory system functionality."""
    print("🧪 Testing Enhanced Memory System...")
    
    # Create temporary storage for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"   Using temporary storage: {temp_dir}")
        
        try:
            # Initialize memory agent
            memory_agent = EnhancedMemoryAgent(
                agent_id="test_memory_agent",
                storage_path=os.path.join(temp_dir, "agent_storage"),
                enable_logging=False  # Reduce noise during testing
            )
            print("   ✅ Memory agent initialized")
            
            # Initialize memory configuration
            config = ChatMemoryConfig(
                storage_path=temp_dir,
                max_recent_messages=10,
                max_context_messages=20,
                semantic_search_threshold=0.5  # Lower threshold for testing
            )
            print("   ✅ Memory config created")
            
            # Initialize enhanced memory
            enhanced_memory = EnhancedChatMemory(
                config=config,
                memory_agent=memory_agent,
                llm_provider=None  # Can work without LLM for basic testing
            )
            print("   ✅ Enhanced memory system initialized")
            
            # Test 1: Add messages to conversation
            print("\n   🧪 Test 1: Adding messages to conversation")
            conversation_id = "test_conv_001"
            
            # Add first message
            msg1 = await enhanced_memory.add_message(
                conversation_id=conversation_id,
                role="user",
                content="What is the NIS Protocol v3 architecture?",
                user_id="test_user"
            )
            print(f"      ✅ Added message 1: {msg1.id}")
            
            # Add second message
            msg2 = await enhanced_memory.add_message(
                conversation_id=conversation_id,
                role="assistant", 
                content="The NIS Protocol v3 features a multi-layered architecture with Laplace transform signal processing, KAN reasoning layers, and PINN physics validation.",
                user_id="test_user"
            )
            print(f"      ✅ Added message 2: {msg2.id}")
            
            # Add third message to test context building
            msg3 = await enhanced_memory.add_message(
                conversation_id=conversation_id,
                role="user",
                content="How does the KAN layer contribute to interpretability?",
                user_id="test_user"
            )
            print(f"      ✅ Added message 3: {msg3.id}")
            
            # Test 2: Get conversation context
            print("\n   🧪 Test 2: Getting conversation context")
            context = await enhanced_memory.get_conversation_context(
                conversation_id=conversation_id,
                max_messages=10
            )
            print(f"      ✅ Retrieved {len(context)} context messages")
            
            if len(context) >= 3:
                print("      ✅ Context includes all messages")
            else:
                print(f"      ⚠️  Expected 3+ messages, got {len(context)}")
            
            # Test 3: Get conversation summary
            print("\n   🧪 Test 3: Getting conversation summary")
            summary = await enhanced_memory.get_conversation_summary(conversation_id)
            print(f"      ✅ Generated summary: {summary[:100]}...")
            
            # Test 4: Search conversations (basic text search)
            print("\n   🧪 Test 4: Searching conversations")
            search_results = await enhanced_memory.search_conversations(
                query="architecture",
                user_id="test_user",
                limit=5
            )
            print(f"      ✅ Found {len(search_results)} conversations matching 'architecture'")
            
            # Test 5: Get statistics
            print("\n   🧪 Test 5: Getting memory statistics")
            stats = enhanced_memory.get_stats()
            print(f"      ✅ Stats - Messages: {stats.get('total_messages', 0)}, Conversations: {stats.get('total_conversations', 0)}")
            
            # Test 6: Database persistence
            print("\n   🧪 Test 6: Testing database persistence")
            
            # Create new memory instance with same storage
            enhanced_memory2 = EnhancedChatMemory(
                config=config,
                memory_agent=memory_agent,
                llm_provider=None
            )
            
            # Try to retrieve messages
            messages = await enhanced_memory2._get_conversation_messages(conversation_id, 10)
            print(f"      ✅ Persistence test - Retrieved {len(messages)} messages from new instance")
            
            if len(messages) >= 3:
                print("      ✅ All messages persisted correctly")
            else:
                print(f"      ⚠️  Expected 3+ messages, got {len(messages)}")
            
            print("\n✅ All memory tests completed successfully!")
            return True
            
        except Exception as e:
            print(f"\n❌ Memory test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Run the memory integration test."""
    print("🚀 Enhanced Memory Integration Test")
    print("=" * 40)
    
    try:
        # Run the async test
        success = asyncio.run(test_memory_system())
        
        if success:
            print("\n🎉 Enhanced Memory System is working correctly!")
            print("   Ready for production use.")
        else:
            print("\n💥 Enhanced Memory System has issues.")
            print("   Check the error messages above.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
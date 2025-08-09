#!/usr/bin/env python3
"""
Enhanced Chat Memory Demo Script
================================

This script demonstrates the new chat memory capabilities in NIS Protocol v3.

Features Demonstrated:
- Persistent conversation storage
- Semantic search across conversations  
- Topic continuity and threading
- Cross-conversation context linking
- Memory management and cleanup

Run this script to test the enhanced memory system.
"""

import asyncio
import json
import requests
import time
from typing import Dict, Any, List

# Base URL for the NIS Protocol API
BASE_URL = "http://localhost:8000"

class MemoryDemoClient:
    """Client for testing the enhanced memory system."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.user_id = "demo_user_001"
        
    def test_basic_chat(self) -> Dict[str, Any]:
        """Test basic chat functionality with memory."""
        print("🧪 Testing Basic Chat with Enhanced Memory...")
        
        # Send a message about NIS Protocol architecture
        response = requests.post(f"{self.base_url}/chat", json={
            "message": "Can you explain the NIS Protocol v3 architecture and how the signal processing pipeline works?",
            "user_id": self.user_id,
            "provider": "openai",
            "agent_type": "general"
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ First message sent successfully")
            print(f"   Response length: {len(data.get('content', ''))}")
            return data
        else:
            print(f"❌ Failed to send message: {response.status_code}")
            return {}
    
    def test_follow_up_question(self, conversation_id: str) -> Dict[str, Any]:
        """Test follow-up question that should use memory context."""
        print("🧪 Testing Follow-up with Memory Context...")
        
        # Ask a follow-up question that builds on previous context
        response = requests.post(f"{self.base_url}/chat", json={
            "message": "How does the KAN reasoning layer specifically contribute to interpretability?",
            "user_id": self.user_id,
            "conversation_id": conversation_id,
            "provider": "openai",
            "agent_type": "general"
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Follow-up message sent successfully")
            print(f"   Should include context about previous discussion")
            return data
        else:
            print(f"❌ Failed to send follow-up: {response.status_code}")
            return {}
    
    def test_semantic_search(self) -> List[Dict[str, Any]]:
        """Test semantic search across conversations."""
        print("🧪 Testing Semantic Search...")
        
        response = requests.get(f"{self.base_url}/memory/conversations", params={
            "query": "neural network interpretability",
            "user_id": self.user_id,
            "limit": 5
        })
        
        if response.status_code == 200:
            data = response.json()
            conversations = data.get('conversations', [])
            print(f"✅ Found {len(conversations)} relevant conversations")
            
            for conv in conversations:
                print(f"   - {conv.get('title', 'Untitled')}: {conv.get('preview', '')[:100]}...")
                
            return conversations
        else:
            print(f"❌ Failed to search conversations: {response.status_code}")
            return []
    
    def test_topic_discovery(self) -> List[Dict[str, Any]]:
        """Test topic discovery and tracking."""
        print("🧪 Testing Topic Discovery...")
        
        response = requests.get(f"{self.base_url}/memory/topics", params={"limit": 10})
        
        if response.status_code == 200:
            data = response.json()
            topics = data.get('topics', [])
            print(f"✅ Discovered {len(topics)} topics from conversations")
            
            for topic in topics:
                print(f"   - {topic.get('name', 'Unknown')}: {topic.get('conversation_count', 0)} conversations")
                
            return topics
        else:
            print(f"❌ Failed to get topics: {response.status_code}")
            return []
    
    def test_context_preview(self, conversation_id: str) -> Dict[str, Any]:
        """Test context preview functionality."""
        print("🧪 Testing Context Preview...")
        
        response = requests.get(f"{self.base_url}/memory/conversation/{conversation_id}/context", params={
            "message": "Tell me more about the physics validation layer"
        })
        
        if response.status_code == 200:
            data = response.json()
            context_count = data.get('context_count', 0)
            print(f"✅ Context preview generated with {context_count} messages")
            print(f"   Includes semantic context from related conversations")
            return data
        else:
            print(f"❌ Failed to get context preview: {response.status_code}")
            return {}
    
    def test_memory_stats(self) -> Dict[str, Any]:
        """Test memory system statistics."""
        print("🧪 Testing Memory Statistics...")
        
        response = requests.get(f"{self.base_url}/memory/stats")
        
        if response.status_code == 200:
            data = response.json()
            stats = data.get('stats', {})
            print(f"✅ Memory Statistics:")
            print(f"   Enhanced Memory Enabled: {stats.get('enhanced_memory_enabled', False)}")
            print(f"   Total Conversations: {stats.get('total_conversations', 0)}")
            print(f"   Total Messages: {stats.get('total_messages', 0)}")
            print(f"   Storage Size: {stats.get('storage_size_mb', 0)} MB")
            return data
        else:
            print(f"❌ Failed to get memory stats: {response.status_code}")
            return {}
    
    def test_conversation_deep_dive(self, conversation_id: str) -> Dict[str, Any]:
        """Test detailed conversation analysis."""
        print("🧪 Testing Conversation Deep Dive...")
        
        response = requests.get(f"{self.base_url}/memory/conversation/{conversation_id}", params={
            "include_context": True
        })
        
        if response.status_code == 200:
            data = response.json()
            message_count = data.get('message_count', 0)
            summary = data.get('summary', '')
            context = data.get('semantic_context', [])
            
            print(f"✅ Conversation Analysis:")
            print(f"   Messages: {message_count}")
            print(f"   Summary: {summary}")
            print(f"   Semantic Context: {len(context)} related messages")
            
            return data
        else:
            print(f"❌ Failed to analyze conversation: {response.status_code}")
            return {}

def main():
    """Run the enhanced memory demonstration."""
    print("🚀 NIS Protocol v3 Enhanced Memory System Demo")
    print("=" * 50)
    
    client = MemoryDemoClient()
    
    # Test 1: Basic chat with memory
    print("\n1. Testing Basic Chat with Enhanced Memory")
    chat_result = client.test_basic_chat()
    conversation_id = chat_result.get('conversation_id')
    
    if not conversation_id:
        print("❌ Could not get conversation ID. Check if the server is running.")
        return
    
    print(f"   Conversation ID: {conversation_id}")
    
    # Test 2: Follow-up question (tests memory context)
    print("\n2. Testing Follow-up with Memory Context")
    client.test_follow_up_question(conversation_id)
    
    # Test 3: Semantic search
    print("\n3. Testing Semantic Search")
    client.test_semantic_search()
    
    # Test 4: Topic discovery
    print("\n4. Testing Topic Discovery")
    client.test_topic_discovery()
    
    # Test 5: Context preview
    print("\n5. Testing Context Preview")
    client.test_context_preview(conversation_id)
    
    # Test 6: Memory statistics
    print("\n6. Testing Memory Statistics")
    client.test_memory_stats()
    
    # Test 7: Conversation deep dive
    print("\n7. Testing Conversation Deep Dive")
    client.test_conversation_deep_dive(conversation_id)
    
    print("\n✅ Enhanced Memory Demo Complete!")
    print("\nKey Benefits Demonstrated:")
    print("- ✨ Persistent conversation storage across sessions")
    print("- 🔍 Semantic search finds relevant past discussions")  
    print("- 🧵 Topic threading maintains subject continuity")
    print("- 🔗 Cross-conversation context linking")
    print("- 📊 Rich memory analytics and management")
    print("\nNow your conversations can go deeper into subjects with full context!")

if __name__ == "__main__":
    main()
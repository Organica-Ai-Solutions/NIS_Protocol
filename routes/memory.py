"""
NIS Protocol v4.0 - Memory Routes

This module contains memory management endpoints:
- Memory stats
- Conversation search and retrieval
- Topic management
- Memory cleanup
- Persistent memory store/retrieve

MIGRATION STATUS: Ready for testing
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("nis.routes.memory")

# Create router
router = APIRouter(prefix="/memory", tags=["Memory"])


# ====== Request Models ======

class MemoryStoreRequest(BaseModel):
    namespace: str = Field(default="default", description="Memory namespace")
    key: str = Field(..., description="Memory key")
    value: Any = Field(..., description="Value to store")
    ttl: Optional[int] = Field(default=None, description="Time to live in seconds")


class MemoryRetrieveRequest(BaseModel):
    namespace: str = Field(default="default")
    key: str


class MemoryContextRequest(BaseModel):
    conversation_id: str
    message: str
    max_context: int = 10


# ====== Dependency Injection ======

def get_enhanced_chat_memory():
    """Get enhanced chat memory instance"""
    return getattr(router, '_enhanced_chat_memory', None)

def get_conversation_memory():
    """Get conversation memory dict"""
    return getattr(router, '_conversation_memory', {})

def get_persistent_memory():
    """Get persistent memory system"""
    return getattr(router, '_persistent_memory', None)


# ====== Endpoints ======

@router.get("/stats")
async def get_memory_stats():
    """Get statistics about the chat memory system."""
    try:
        enhanced_chat_memory = get_enhanced_chat_memory()
        conversation_memory = get_conversation_memory()
        
        if enhanced_chat_memory:
            stats = enhanced_chat_memory.get_stats()
            stats["enhanced_memory_enabled"] = True
        else:
            stats = {
                "enhanced_memory_enabled": False,
                "total_conversations": len(conversation_memory),
                "total_messages": sum(len(msgs) for msgs in conversation_memory.values())
            }
        
        return {
            "status": "success",
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get memory stats: {str(e)}")


@router.get("/conversations")
async def search_conversations(
    query: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 10
):
    """Search conversations by content or get recent conversations."""
    try:
        enhanced_chat_memory = get_enhanced_chat_memory()
        conversation_memory = get_conversation_memory()
        
        if enhanced_chat_memory and query:
            conversations = await enhanced_chat_memory.search_conversations(
                query=query,
                user_id=user_id,
                limit=limit
            )
        else:
            conversations = []
            for conv_id, messages in list(conversation_memory.items())[:limit]:
                if messages:
                    conversations.append({
                        "conversation_id": conv_id,
                        "title": f"Conversation {conv_id[:8]}...",
                        "message_count": len(messages),
                        "last_activity": datetime.fromtimestamp(messages[-1].get("timestamp", time.time())).isoformat(),
                        "preview": messages[-1].get("content", "")[:200],
                        "search_type": "legacy"
                    })
        
        return {
            "status": "success",
            "conversations": conversations,
            "query": query,
            "limit": limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search conversations: {str(e)}")


@router.get("/conversation/{conversation_id}")
async def get_conversation_details(conversation_id: str, include_context: bool = True):
    """Get detailed information about a specific conversation."""
    try:
        enhanced_chat_memory = get_enhanced_chat_memory()
        conversation_memory = get_conversation_memory()
        
        if enhanced_chat_memory:
            messages = await enhanced_chat_memory._get_conversation_messages(conversation_id, 100)
            summary = await enhanced_chat_memory.get_conversation_summary(conversation_id)
            
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "topic_tags": msg.topic_tags,
                    "importance_score": msg.importance_score
                })
            
            context = []
            if include_context and formatted_messages:
                last_message = formatted_messages[-1]["content"]
                context = await enhanced_chat_memory._get_semantic_context(
                    last_message, conversation_id, max_results=5
                )
            
            return {
                "status": "success",
                "conversation_id": conversation_id,
                "summary": summary,
                "message_count": len(formatted_messages),
                "messages": formatted_messages,
                "semantic_context": context
            }
        else:
            messages = conversation_memory.get(conversation_id, [])
            return {
                "status": "success",
                "conversation_id": conversation_id,
                "summary": f"Legacy conversation with {len(messages)} messages",
                "message_count": len(messages),
                "messages": messages,
                "enhanced_memory_available": False
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get conversation details: {str(e)}")


@router.get("/topics")
async def get_topics(limit: int = 20):
    """Get list of conversation topics."""
    try:
        enhanced_chat_memory = get_enhanced_chat_memory()
        
        if enhanced_chat_memory:
            topics = []
            for topic in list(enhanced_chat_memory.topic_index.values())[:limit]:
                topics.append({
                    "id": topic.id,
                    "name": topic.name,
                    "description": topic.description,
                    "conversation_count": len(topic.conversation_ids),
                    "last_discussed": topic.last_discussed.isoformat(),
                    "importance_score": topic.importance_score
                })
            
            return {
                "status": "success",
                "topics": topics,
                "total_topics": len(enhanced_chat_memory.topic_index)
            }
        else:
            return {
                "status": "success",
                "topics": [],
                "enhanced_memory_available": False,
                "message": "Enhanced memory not available"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get topics: {str(e)}")


@router.get("/topic/{topic_name}/conversations")
async def get_topic_conversations(topic_name: str, limit: int = 10):
    """Get conversations related to a specific topic."""
    try:
        enhanced_chat_memory = get_enhanced_chat_memory()
        
        if enhanced_chat_memory:
            conversations = await enhanced_chat_memory.get_topic_conversations(topic_name, limit)
            return {
                "status": "success",
                "topic": topic_name,
                "conversations": conversations
            }
        else:
            return {
                "status": "success",
                "topic": topic_name,
                "conversations": [],
                "enhanced_memory_available": False
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get topic conversations: {str(e)}")


@router.post("/cleanup")
async def cleanup_old_memory(days_to_keep: int = 90):
    """Clean up old conversation data."""
    try:
        enhanced_chat_memory = get_enhanced_chat_memory()
        
        if enhanced_chat_memory:
            await enhanced_chat_memory.cleanup_old_data(days_to_keep)
            return {
                "status": "success",
                "message": f"Cleaned up data older than {days_to_keep} days",
                "days_kept": days_to_keep
            }
        else:
            return {
                "status": "success",
                "message": "Enhanced memory not available - no cleanup needed",
                "enhanced_memory_available": False
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup memory: {str(e)}")


@router.get("/conversation/{conversation_id}/context")
async def get_conversation_context_preview(conversation_id: str, message: str):
    """Preview the context that would be used for a message in a conversation."""
    try:
        enhanced_chat_memory = get_enhanced_chat_memory()
        
        if enhanced_chat_memory:
            context = await enhanced_chat_memory._get_semantic_context(
                message, conversation_id, max_results=10
            )
            return {
                "status": "success",
                "conversation_id": conversation_id,
                "message": message,
                "context": context
            }
        else:
            return {
                "status": "success",
                "conversation_id": conversation_id,
                "context": [],
                "enhanced_memory_available": False
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get context: {str(e)}")


# ====== Persistent Memory Endpoints ======

@router.post("/store")
async def store_memory(request: MemoryStoreRequest):
    """Store a value in persistent memory."""
    try:
        persistent_memory = get_persistent_memory()
        
        if persistent_memory:
            await persistent_memory.store(
                namespace=request.namespace,
                key=request.key,
                value=request.value,
                ttl=request.ttl
            )
            return {
                "status": "success",
                "namespace": request.namespace,
                "key": request.key,
                "stored": True
            }
        else:
            return {
                "status": "error",
                "message": "Persistent memory not available"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store memory: {str(e)}")


@router.post("/retrieve")
async def retrieve_memory(request: MemoryRetrieveRequest):
    """Retrieve a value from persistent memory."""
    try:
        persistent_memory = get_persistent_memory()
        
        if persistent_memory:
            value = await persistent_memory.retrieve(
                namespace=request.namespace,
                key=request.key
            )
            return {
                "status": "success",
                "namespace": request.namespace,
                "key": request.key,
                "value": value,
                "found": value is not None
            }
        else:
            return {
                "status": "error",
                "message": "Persistent memory not available"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve memory: {str(e)}")


@router.get("/retrieve/{namespace}/{key}")
async def retrieve_memory_by_path(namespace: str, key: str):
    """Retrieve a value from persistent memory by path."""
    try:
        persistent_memory = get_persistent_memory()
        
        if persistent_memory:
            value = await persistent_memory.retrieve(namespace=namespace, key=key)
            return {
                "status": "success",
                "namespace": namespace,
                "key": key,
                "value": value,
                "found": value is not None
            }
        else:
            return {"status": "error", "message": "Persistent memory not available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list/{namespace}")
async def list_memory_keys(namespace: str):
    """List all keys in a namespace."""
    try:
        persistent_memory = get_persistent_memory()
        
        if persistent_memory:
            keys = await persistent_memory.list_keys(namespace=namespace)
            return {
                "status": "success",
                "namespace": namespace,
                "keys": keys,
                "count": len(keys)
            }
        else:
            return {"status": "error", "message": "Persistent memory not available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/{namespace}/{key}")
async def delete_memory(namespace: str, key: str):
    """Delete a value from persistent memory."""
    try:
        persistent_memory = get_persistent_memory()
        
        if persistent_memory:
            await persistent_memory.delete(namespace=namespace, key=key)
            return {
                "status": "success",
                "namespace": namespace,
                "key": key,
                "deleted": True
            }
        else:
            return {"status": "error", "message": "Persistent memory not available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/persistent/stats")
async def get_persistent_memory_stats():
    """Get persistent memory statistics."""
    try:
        persistent_memory = get_persistent_memory()
        
        if persistent_memory:
            stats = await persistent_memory.get_stats()
            return {"status": "success", "stats": stats}
        else:
            return {"status": "error", "message": "Persistent memory not available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/context")
async def get_memory_context(request: MemoryContextRequest):
    """Get memory context for a conversation."""
    try:
        enhanced_chat_memory = get_enhanced_chat_memory()
        
        if enhanced_chat_memory:
            context = await enhanced_chat_memory._get_semantic_context(
                request.message, 
                request.conversation_id, 
                max_results=request.max_context
            )
            return {
                "status": "success",
                "context": context,
                "conversation_id": request.conversation_id
            }
        else:
            return {
                "status": "success",
                "context": [],
                "enhanced_memory_available": False
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====== Dependency Injection Helper ======

def set_dependencies(enhanced_chat_memory=None, conversation_memory=None, persistent_memory=None):
    """Set dependencies for the memory router"""
    router._enhanced_chat_memory = enhanced_chat_memory
    router._conversation_memory = conversation_memory or {}
    router._persistent_memory = persistent_memory

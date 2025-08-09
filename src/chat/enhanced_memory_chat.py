"""
Enhanced Chat Memory System for NIS Protocol v3
==============================================

This module implements a sophisticated chat memory system that maintains
conversation context across sessions and enables deep topic exploration.

Features:
- Persistent conversation storage with SQLite
- Semantic search for relevant past conversations
- Topic threading and continuity tracking
- Long-term context retention
- Intelligent context summarization
- Cross-conversation topic linkage

Adheres to NIS Protocol v3 architectural principles.
"""

import asyncio
import json
import sqlite3
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

from src.core.agent import NISAgent, NISLayer
from src.agents.memory.enhanced_memory_agent import EnhancedMemoryAgent, MemoryType
from src.emotion.emotional_state import EmotionalState
from src.llm.llm_manager import GeneralLLMProvider


class ConversationMessage(BaseModel):
    """Represents a single message in a conversation."""
    id: str
    conversation_id: str
    role: str  # user, assistant, system
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    topic_tags: List[str] = []
    importance_score: float = 0.5
    embedding: Optional[List[float]] = None


class ConversationTopic(BaseModel):
    """Represents a topic that spans multiple conversations."""
    id: str
    name: str
    description: str
    conversation_ids: List[str]
    last_discussed: datetime
    importance_score: float
    related_topics: List[str] = []


class ChatMemoryConfig(BaseModel):
    """Configuration for chat memory system."""
    storage_path: str = "data/chat_memory/"
    max_recent_messages: int = 20
    max_context_messages: int = 50
    semantic_search_threshold: float = 0.7
    topic_similarity_threshold: float = 0.8
    memory_consolidation_interval: int = 3600  # 1 hour
    enable_cross_conversation_linking: bool = True
    enable_topic_evolution: bool = True


class EnhancedChatMemory:
    """
    Enhanced chat memory system with persistent storage and semantic search.
    
    This system provides:
    - Persistent conversation storage
    - Semantic search across all conversations
    - Topic threading and continuity
    - Intelligent context selection
    - Cross-conversation knowledge linking
    """
    
    def __init__(
        self,
        config: ChatMemoryConfig,
        memory_agent: Optional[EnhancedMemoryAgent] = None,
        llm_provider: Optional[GeneralLLMProvider] = None
    ):
        """Initialize the enhanced chat memory system."""
        self.config = config
        self.memory_agent = memory_agent
        self.llm_provider = llm_provider
        
        # Set up persistent storage
        self._setup_storage()
        
        # In-memory caches for performance
        self.active_conversations: Dict[str, List[ConversationMessage]] = {}
        self.topic_index: Dict[str, ConversationTopic] = {}
        self.conversation_topics: Dict[str, List[str]] = defaultdict(list)
        
        # Load recent data into memory
        self._load_recent_data()
        
        # Track conversation flow
        self.conversation_flow: Dict[str, Dict[str, Any]] = {}
        
    def _setup_storage(self):
        """Set up SQLite database for persistent storage."""
        import os
        os.makedirs(self.config.storage_path, exist_ok=True)
        
        self.db_path = os.path.join(self.config.storage_path, "chat_memory.db")
        
        with sqlite3.connect(self.db_path) as conn:
            # Messages table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata TEXT,
                    topic_tags TEXT,
                    importance_score REAL DEFAULT 0.5,
                    embedding TEXT
                )
            """)
            
            # Topics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS topics (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    conversation_ids TEXT,
                    last_discussed REAL NOT NULL,
                    importance_score REAL DEFAULT 0.5,
                    related_topics TEXT
                )
            """)
            
            # Conversation metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT,
                    created_at REAL NOT NULL,
                    last_activity REAL NOT NULL,
                    message_count INTEGER DEFAULT 0,
                    topic_summary TEXT,
                    importance_score REAL DEFAULT 0.5
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_topics_name ON topics(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)")
            
            conn.commit()
    
    def _load_recent_data(self):
        """Load recent conversations and topics into memory."""
        # Load conversations from last 24 hours
        cutoff_time = time.time() - (24 * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            # Load recent messages
            cursor = conn.execute("""
                SELECT * FROM messages 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC 
                LIMIT 1000
            """, (cutoff_time,))
            
            for row in cursor.fetchall():
                msg = ConversationMessage(
                    id=row[0],
                    conversation_id=row[1],
                    role=row[2],
                    content=row[3],
                    timestamp=datetime.fromtimestamp(row[4]),
                    metadata=json.loads(row[5]) if row[5] else None,
                    topic_tags=json.loads(row[6]) if row[6] else [],
                    importance_score=row[7],
                    embedding=json.loads(row[8]) if row[8] else None
                )
                
                if msg.conversation_id not in self.active_conversations:
                    self.active_conversations[msg.conversation_id] = []
                self.active_conversations[msg.conversation_id].append(msg)
            
            # Load active topics
            cursor = conn.execute("""
                SELECT * FROM topics 
                WHERE last_discussed > ?
                ORDER BY importance_score DESC
            """, (cutoff_time,))
            
            for row in cursor.fetchall():
                topic = ConversationTopic(
                    id=row[0],
                    name=row[1],
                    description=row[2] or "",
                    conversation_ids=json.loads(row[3]) if row[3] else [],
                    last_discussed=datetime.fromtimestamp(row[4]),
                    importance_score=row[5],
                    related_topics=json.loads(row[6]) if row[6] else []
                )
                self.topic_index[topic.id] = topic
    
    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> ConversationMessage:
        """Add a new message to the conversation memory."""
        
        # Create message object
        msg = ConversationMessage(
            id=f"msg_{uuid.uuid4().hex[:12]}",
            conversation_id=conversation_id,
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
            importance_score=await self._calculate_importance(content, role)
        )
        
        # Generate embedding if memory agent is available
        if self.memory_agent and hasattr(self.memory_agent, 'embedding_provider'):
            try:
                embedding = await self.memory_agent.embedding_provider.get_embedding(content)
                msg.embedding = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            except Exception as e:
                print(f"Warning: Could not generate embedding: {e}")
        
        # Extract and assign topics
        msg.topic_tags = await self._extract_topics(content, conversation_id)
        
        # Add to memory caches
        if conversation_id not in self.active_conversations:
            self.active_conversations[conversation_id] = []
        self.active_conversations[conversation_id].append(msg)
        
        # Update conversation topics
        for topic_tag in msg.topic_tags:
            if topic_tag not in self.conversation_topics[conversation_id]:
                self.conversation_topics[conversation_id].append(topic_tag)
        
        # Store in database
        await self._store_message(msg)
        
        # Update conversation metadata
        await self._update_conversation_metadata(conversation_id, user_id)
        
        # Update topic tracking
        await self._update_topic_tracking(msg)
        
        return msg
    
    async def get_conversation_context(
        self,
        conversation_id: str,
        max_messages: Optional[int] = None,
        include_semantic_context: bool = True,
        current_message: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get relevant context for a conversation, including semantic context from related conversations.
        """
        max_messages = max_messages or self.config.max_context_messages
        
        # Get direct conversation messages
        direct_messages = await self._get_conversation_messages(conversation_id, max_messages)
        
        context_messages = []
        
        # Add direct conversation context
        for msg in direct_messages[-max_messages:]:
            context_messages.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "source": "current_conversation"
            })
        
        # Add semantic context if enabled and we have a current message
        if include_semantic_context and current_message:
            semantic_context = await self._get_semantic_context(
                current_message, 
                conversation_id,
                max_results=5
            )
            
            # Insert relevant context from other conversations
            for ctx in semantic_context:
                context_messages.insert(-1, {
                    "role": "system",
                    "content": f"[Relevant context from previous conversation]: {ctx['content']}",
                    "timestamp": ctx['timestamp'],
                    "source": "semantic_context",
                    "conversation_id": ctx['conversation_id'],
                    "similarity": ctx['similarity']
                })
        
        return context_messages
    
    async def _get_conversation_messages(
        self, 
        conversation_id: str, 
        limit: int
    ) -> List[ConversationMessage]:
        """Get messages for a specific conversation."""
        
        # Try memory cache first
        if conversation_id in self.active_conversations:
            return self.active_conversations[conversation_id][-limit:]
        
        # Load from database
        messages = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM messages 
                WHERE conversation_id = ? 
                ORDER BY timestamp ASC 
                LIMIT ?
            """, (conversation_id, limit))
            
            for row in cursor.fetchall():
                msg = ConversationMessage(
                    id=row[0],
                    conversation_id=row[1],
                    role=row[2],
                    content=row[3],
                    timestamp=datetime.fromtimestamp(row[4]),
                    metadata=json.loads(row[5]) if row[5] else None,
                    topic_tags=json.loads(row[6]) if row[6] else [],
                    importance_score=row[7],
                    embedding=json.loads(row[8]) if row[8] else None
                )
                messages.append(msg)
        
        return messages
    
    async def _get_semantic_context(
        self,
        query: str,
        current_conversation_id: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Get semantically relevant context from other conversations."""
        
        if not self.memory_agent or not hasattr(self.memory_agent, 'embedding_provider'):
            return []
        
        try:
            # Generate embedding for the query
            query_embedding = await self.memory_agent.embedding_provider.get_embedding(query)
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            relevant_messages = []
            
            # Search through stored messages with embeddings
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, conversation_id, content, timestamp, embedding 
                    FROM messages 
                    WHERE conversation_id != ? AND embedding IS NOT NULL 
                    AND importance_score > 0.6
                    ORDER BY timestamp DESC 
                    LIMIT 100
                """, (current_conversation_id,))
                
                for row in cursor.fetchall():
                    if row[4]:  # Has embedding
                        try:
                            msg_embedding = json.loads(row[4])
                            similarity = self._calculate_cosine_similarity(query_embedding, msg_embedding)
                            
                            if similarity > self.config.semantic_search_threshold:
                                relevant_messages.append({
                                    'id': row[0],
                                    'conversation_id': row[1],
                                    'content': row[2],
                                    'timestamp': datetime.fromtimestamp(row[3]).isoformat(),
                                    'similarity': similarity
                                })
                        except (json.JSONDecodeError, Exception):
                            continue
            
            # Sort by similarity and return top results
            relevant_messages.sort(key=lambda x: x['similarity'], reverse=True)
            return relevant_messages[:max_results]
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            
            return float(dot_product / (norm_v1 * norm_v2))
        except Exception:
            return 0.0
    
    async def _calculate_importance(self, content: str, role: str) -> float:
        """Calculate importance score for a message."""
        base_score = 0.5
        
        # Boost importance for longer messages
        length_factor = min(len(content) / 1000, 0.3)
        
        # Boost for questions and technical content
        question_factor = 0.2 if "?" in content else 0.0
        technical_factor = 0.1 if any(word in content.lower() for word in [
            "algorithm", "implementation", "architecture", "system", "protocol",
            "neural", "learning", "model", "data", "analysis"
        ]) else 0.0
        
        # Role-based adjustments
        role_factor = 0.1 if role == "user" else 0.0
        
        importance = base_score + length_factor + question_factor + technical_factor + role_factor
        return min(importance, 1.0)
    
    async def _extract_topics(self, content: str, conversation_id: str) -> List[str]:
        """Extract topics from message content."""
        
        # Use LLM to extract topics if available
        if self.llm_provider:
            try:
                topic_prompt = f"""
                Extract 1-3 main topics or themes from this message. Return only topic keywords, separated by commas.
                Focus on technical concepts, subjects being discussed, or main themes.
                
                Message: {content}
                
                Topics:"""
                
                response = await self.llm_provider.generate_response([
                    {"role": "user", "content": topic_prompt}
                ], temperature=0.3)
                
                if response and response.get('content'):
                    topics = [t.strip() for t in response['content'].split(',')]
                    return [t for t in topics if len(t) > 2 and len(t) < 50]
                    
            except Exception as e:
                print(f"Warning: Could not extract topics with LLM: {e}")
        
        # Fallback to keyword extraction
        technical_keywords = [
            "nis protocol", "neural", "ai", "machine learning", "algorithm",
            "architecture", "system", "data", "analysis", "model", "training",
            "pipeline", "agent", "memory", "reasoning", "consciousness"
        ]
        
        content_lower = content.lower()
        found_topics = []
        
        for keyword in technical_keywords:
            if keyword in content_lower:
                found_topics.append(keyword)
        
        return found_topics[:3]  # Limit to 3 topics
    
    async def _store_message(self, message: ConversationMessage):
        """Store message in persistent database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO messages 
                (id, conversation_id, role, content, timestamp, metadata, topic_tags, importance_score, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message.id,
                message.conversation_id,
                message.role,
                message.content,
                message.timestamp.timestamp(),
                json.dumps(message.metadata) if message.metadata else None,
                json.dumps(message.topic_tags),
                message.importance_score,
                json.dumps(message.embedding) if message.embedding else None
            ))
            conn.commit()
    
    async def _update_conversation_metadata(self, conversation_id: str, user_id: Optional[str]):
        """Update conversation metadata."""
        now = time.time()
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if conversation exists
            cursor = conn.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
            exists = cursor.fetchone()
            
            if exists:
                # Update existing
                conn.execute("""
                    UPDATE conversations 
                    SET last_activity = ?, message_count = message_count + 1
                    WHERE id = ?
                """, (now, conversation_id))
            else:
                # Create new
                conn.execute("""
                    INSERT INTO conversations 
                    (id, user_id, created_at, last_activity, message_count)
                    VALUES (?, ?, ?, ?, 1)
                """, (conversation_id, user_id, now, now))
            
            conn.commit()
    
    async def _update_topic_tracking(self, message: ConversationMessage):
        """Update topic tracking and relationships."""
        for topic_tag in message.topic_tags:
            # Check if topic exists
            topic_id = f"topic_{topic_tag.replace(' ', '_')}"
            
            if topic_id in self.topic_index:
                # Update existing topic
                topic = self.topic_index[topic_id]
                topic.last_discussed = message.timestamp
                if message.conversation_id not in topic.conversation_ids:
                    topic.conversation_ids.append(message.conversation_id)
                # Increase importance based on frequency
                topic.importance_score = min(topic.importance_score + 0.1, 1.0)
            else:
                # Create new topic
                topic = ConversationTopic(
                    id=topic_id,
                    name=topic_tag,
                    description=f"Topic: {topic_tag}",
                    conversation_ids=[message.conversation_id],
                    last_discussed=message.timestamp,
                    importance_score=0.6
                )
                self.topic_index[topic_id] = topic
            
            # Store in database
            await self._store_topic(topic)
    
    async def _store_topic(self, topic: ConversationTopic):
        """Store topic in persistent database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO topics 
                (id, name, description, conversation_ids, last_discussed, importance_score, related_topics)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                topic.id,
                topic.name,
                topic.description,
                json.dumps(topic.conversation_ids),
                topic.last_discussed.timestamp(),
                topic.importance_score,
                json.dumps(topic.related_topics)
            ))
            conn.commit()
    
    async def get_conversation_summary(self, conversation_id: str) -> str:
        """Generate a summary of the conversation for context."""
        messages = await self._get_conversation_messages(conversation_id, 20)
        
        if not messages:
            return "No conversation history available."
        
        # Extract key topics and themes
        topics = set()
        for msg in messages:
            topics.update(msg.topic_tags)
        
        summary_parts = []
        summary_parts.append(f"Conversation with {len(messages)} messages")
        
        if topics:
            summary_parts.append(f"Topics discussed: {', '.join(list(topics)[:5])}")
        
        # Get recent direction
        recent_messages = messages[-3:]
        if recent_messages:
            recent_content = " ".join([msg.content[:100] for msg in recent_messages])
            summary_parts.append(f"Recent focus: {recent_content[:200]}...")
        
        return ". ".join(summary_parts)
    
    async def search_conversations(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search conversations by content or topics."""
        
        results = []
        
        # Semantic search if available
        if self.memory_agent and hasattr(self.memory_agent, 'embedding_provider'):
            results.extend(await self._semantic_search_conversations(query, user_id, limit))
        
        # Fallback to text search
        text_results = await self._text_search_conversations(query, user_id, limit)
        
        # Combine and deduplicate results
        seen_conversations = set()
        combined_results = []
        
        for result in results + text_results:
            conv_id = result['conversation_id']
            if conv_id not in seen_conversations:
                seen_conversations.add(conv_id)
                combined_results.append(result)
        
        return combined_results[:limit]
    
    async def _semantic_search_conversations(
        self,
        query: str,
        user_id: Optional[str],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Perform semantic search across conversations."""
        
        try:
            query_embedding = await self.memory_agent.embedding_provider.get_embedding(query)
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            results = []
            
            with sqlite3.connect(self.db_path) as conn:
                # Build query with optional user filter
                sql_query = """
                    SELECT DISTINCT c.id, c.title, c.created_at, c.last_activity, 
                           m.content, m.timestamp, m.embedding
                    FROM conversations c
                    JOIN messages m ON c.id = m.conversation_id
                    WHERE m.embedding IS NOT NULL
                """
                params = []
                
                if user_id:
                    sql_query += " AND c.user_id = ?"
                    params.append(user_id)
                
                sql_query += " ORDER BY c.last_activity DESC LIMIT 200"
                
                cursor = conn.execute(sql_query, params)
                
                for row in cursor.fetchall():
                    if row[6]:  # Has embedding
                        try:
                            msg_embedding = json.loads(row[6])
                            similarity = self._calculate_cosine_similarity(query_embedding, msg_embedding)
                            
                            if similarity > self.config.semantic_search_threshold:
                                results.append({
                                    'conversation_id': row[0],
                                    'title': row[1] or f"Conversation {row[0][:8]}...",
                                    'created_at': datetime.fromtimestamp(row[2]).isoformat(),
                                    'last_activity': datetime.fromtimestamp(row[3]).isoformat(),
                                    'preview': row[4][:200] + "..." if len(row[4]) > 200 else row[4],
                                    'similarity': similarity,
                                    'search_type': 'semantic'
                                })
                        except (json.JSONDecodeError, Exception):
                            continue
            
            # Sort by similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            print(f"Error in semantic conversation search: {e}")
            return []
    
    async def _text_search_conversations(
        self,
        query: str,
        user_id: Optional[str],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Perform text-based search across conversations."""
        
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            # Build query with optional user filter
            sql_query = """
                SELECT DISTINCT c.id, c.title, c.created_at, c.last_activity, 
                       m.content, m.timestamp
                FROM conversations c
                JOIN messages m ON c.id = m.conversation_id
                WHERE m.content LIKE ?
            """
            params = [f"%{query}%"]
            
            if user_id:
                sql_query += " AND c.user_id = ?"
                params.append(user_id)
            
            sql_query += " ORDER BY c.last_activity DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(sql_query, params)
            
            for row in cursor.fetchall():
                results.append({
                    'conversation_id': row[0],
                    'title': row[1] or f"Conversation {row[0][:8]}...",
                    'created_at': datetime.fromtimestamp(row[2]).isoformat(),
                    'last_activity': datetime.fromtimestamp(row[3]).isoformat(),
                    'preview': row[4][:200] + "..." if len(row[4]) > 200 else row[4],
                    'search_type': 'text'
                })
        
        return results
    
    async def get_topic_conversations(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversations related to a specific topic."""
        
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT DISTINCT c.id, c.title, c.created_at, c.last_activity,
                       COUNT(m.id) as message_count
                FROM conversations c
                JOIN messages m ON c.id = m.conversation_id
                WHERE m.topic_tags LIKE ?
                GROUP BY c.id
                ORDER BY c.last_activity DESC
                LIMIT ?
            """, (f"%{topic}%", limit))
            
            for row in cursor.fetchall():
                results.append({
                    'conversation_id': row[0],
                    'title': row[1] or f"Conversation {row[0][:8]}...",
                    'created_at': datetime.fromtimestamp(row[2]).isoformat(),
                    'last_activity': datetime.fromtimestamp(row[3]).isoformat(),
                    'message_count': row[4],
                    'topic': topic
                })
        
        return results
    
    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old conversation data to manage storage."""
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            # Delete old messages
            conn.execute("DELETE FROM messages WHERE timestamp < ?", (cutoff_time,))
            
            # Delete conversations with no messages
            conn.execute("""
                DELETE FROM conversations 
                WHERE id NOT IN (SELECT DISTINCT conversation_id FROM messages)
            """)
            
            # Clean up topics
            conn.execute("DELETE FROM topics WHERE last_discussed < ?", (cutoff_time,))
            
            conn.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the chat memory system."""
        
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Message count
            cursor = conn.execute("SELECT COUNT(*) FROM messages")
            stats['total_messages'] = cursor.fetchone()[0]
            
            # Conversation count
            cursor = conn.execute("SELECT COUNT(*) FROM conversations")
            stats['total_conversations'] = cursor.fetchone()[0]
            
            # Topic count
            cursor = conn.execute("SELECT COUNT(*) FROM topics")
            stats['total_topics'] = cursor.fetchone()[0]
            
            # Recent activity (last 24 hours)
            cutoff_time = time.time() - (24 * 3600)
            cursor = conn.execute("SELECT COUNT(*) FROM messages WHERE timestamp > ?", (cutoff_time,))
            stats['recent_messages'] = cursor.fetchone()[0]
            
            # Active conversations in memory
            stats['active_conversations'] = len(self.active_conversations)
            
            # Storage size
            import os
            if os.path.exists(self.db_path):
                stats['storage_size_mb'] = round(os.path.getsize(self.db_path) / (1024 * 1024), 2)
            else:
                stats['storage_size_mb'] = 0
        
        return stats
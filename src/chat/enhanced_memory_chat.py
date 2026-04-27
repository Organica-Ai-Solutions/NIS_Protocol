"""
NIS Protocol v4.0 — Enhanced Chat Memory
SQLite-backed conversation memory with semantic search.

Provides persistent memory across Discord sessions and conversations.
Uses TF-IDF embeddings via SimpleEmbeddingProvider (Pi-safe, no heavy ML deps).
Optionally upgrades to SentenceTransformer if available.
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("nis.chat.memory")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    message_id: str
    conversation_id: str
    role: str
    content: str
    timestamp: float
    importance_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    conversation_id: str
    topic: str
    summary: str
    message_count: int
    importance_score: float
    created_at: float
    updated_at: float
    messages: List[ChatMessage] = field(default_factory=list)


@dataclass
class TopicEntry:
    topic: str
    conversation_ids: List[str]
    importance_score: float


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ChatMemoryConfig:
    storage_path: str = "/opt/nis-protocol/data/memory"
    max_conversations: int = 500
    max_messages_per_conv: int = 200
    embedding_provider: str = "auto"   # "auto", "sentence_transformer", "simple"
    similarity_threshold: float = 0.3


# ── Embedding helper ──────────────────────────────────────────────────────────

def _get_provider(provider_type: str = "auto"):
    """Get best available embedding provider."""
    if provider_type in ("auto", "sentence_transformer"):
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Using SentenceTransformer embeddings (all-MiniLM-L6-v2)")

            class STProvider:
                def encode(self, text: str) -> np.ndarray:
                    return model.encode([text])[0].astype(np.float32)

            return STProvider()
        except Exception:
            pass

    # Fallback: TF-IDF hash-based (fast, no deps, Pi-safe)
    try:
        from src.memory.embedding_utils import SimpleEmbeddingProvider
        logger.info("Using SimpleEmbeddingProvider (TF-IDF hash, Pi-safe)")
        p = SimpleEmbeddingProvider()

        class SimpleWrapper:
            def encode(self, text: str) -> np.ndarray:
                return np.array(p.get_embedding(text), dtype=np.float32)

        return SimpleWrapper()
    except Exception:
        pass

    # Last resort: character-level hash
    logger.warning("No embedding provider found — using hash fallback")

    class HashProvider:
        DIM = 128

        def encode(self, text: str) -> np.ndarray:
            vec = np.zeros(self.DIM, dtype=np.float32)
            for i, ch in enumerate(text[:512]):
                vec[hash(ch) % self.DIM] += 1.0 / (i + 1)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec

    return HashProvider()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ── EnhancedChatMemory ────────────────────────────────────────────────────────

class EnhancedChatMemory:
    """
    Persistent chat memory backed by SQLite.

    - Stores conversations and messages across restarts
    - Semantic search over past conversations
    - Topic index for quick retrieval
    - Safe for Raspberry Pi (no heavy GPU requirements)
    """

    def __init__(self, config: Optional[ChatMemoryConfig] = None):
        self.config = config or ChatMemoryConfig()
        os.makedirs(self.config.storage_path, exist_ok=True)
        self.db_path = os.path.join(self.config.storage_path, "chat_memory.db")
        self._provider = None  # lazy init
        self.topic_index: Dict[str, TopicEntry] = {}
        self._init_db()
        self._rebuild_topic_index()
        logger.info("EnhancedChatMemory initialised at %s", self.db_path)

    # ── DB setup ──────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    topic           TEXT NOT NULL DEFAULT 'General',
                    summary         TEXT NOT NULL DEFAULT '',
                    importance      REAL NOT NULL DEFAULT 0.5,
                    created_at      REAL NOT NULL,
                    updated_at      REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    message_id      TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role            TEXT NOT NULL,
                    content         TEXT NOT NULL,
                    timestamp       REAL NOT NULL,
                    importance      REAL NOT NULL DEFAULT 0.5,
                    embedding       BLOB,
                    metadata_json   TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                );
                CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_msg_ts   ON messages(timestamp);
                CREATE INDEX IF NOT EXISTS idx_conv_ts  ON conversations(updated_at DESC);
            """)

    def _rebuild_topic_index(self):
        """Load topic → conversation mapping from DB."""
        self.topic_index.clear()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT conversation_id, topic, importance FROM conversations"
            ).fetchall()
        for row in rows:
            topic = row["topic"]
            if topic not in self.topic_index:
                self.topic_index[topic] = TopicEntry(
                    topic=topic, conversation_ids=[], importance_score=0.0
                )
            self.topic_index[topic].conversation_ids.append(row["conversation_id"])
            self.topic_index[topic].importance_score = max(
                self.topic_index[topic].importance_score, row["importance"]
            )

    # ── Embedding ─────────────────────────────────────────────────────────────

    @property
    def _embed(self):
        if self._provider is None:
            self._provider = _get_provider(self.config.embedding_provider)
        return self._provider

    def _encode(self, text: str) -> bytes:
        vec = self._embed.encode(text)
        return vec.astype(np.float32).tobytes()

    def _decode(self, blob: bytes) -> Optional[np.ndarray]:
        if not blob:
            return None
        return np.frombuffer(blob, dtype=np.float32)

    # ── Write ─────────────────────────────────────────────────────────────────

    async def new_conversation(self, conversation_id: str, topic: str = "General"):
        """Register a new conversation."""
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO conversations
                   (conversation_id, topic, summary, importance, created_at, updated_at)
                   VALUES (?, ?, '', 0.5, ?, ?)""",
                (conversation_id, topic, now, now),
            )
        if topic not in self.topic_index:
            self.topic_index[topic] = TopicEntry(
                topic=topic, conversation_ids=[], importance_score=0.5
            )
        if conversation_id not in self.topic_index[topic].conversation_ids:
            self.topic_index[topic].conversation_ids.append(conversation_id)

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a message to a conversation. Creates conversation if missing."""
        await self.new_conversation(conversation_id)
        msg_id = str(uuid.uuid4())
        now = time.time()
        # Run embedding in executor so we don't block event loop
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._encode, content)
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO messages
                   (message_id, conversation_id, role, content, timestamp,
                    importance, embedding, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    msg_id, conversation_id, role, content, now,
                    importance, embedding,
                    json.dumps(metadata or {}),
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at=? WHERE conversation_id=?",
                (now, conversation_id),
            )
        return msg_id

    # ── Read ──────────────────────────────────────────────────────────────────

    async def _get_conversation_messages(
        self, conversation_id: str, limit: int = 100
    ) -> List[ChatMessage]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM messages WHERE conversation_id=?
                   ORDER BY timestamp ASC LIMIT ?""",
                (conversation_id, limit),
            ).fetchall()
        return [
            ChatMessage(
                message_id=r["message_id"],
                conversation_id=r["conversation_id"],
                role=r["role"],
                content=r["content"],
                timestamp=r["timestamp"],
                importance_score=r["importance"],
                metadata=json.loads(r["metadata_json"]),
            )
            for r in rows
        ]

    async def get_conversation_summary(
        self, conversation_id: str
    ) -> Dict[str, Any]:
        with self._conn() as conn:
            conv = conn.execute(
                "SELECT * FROM conversations WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()
            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()[0]
        if not conv:
            return {"summary": "", "message_count": 0}
        return {
            "summary": conv["summary"],
            "topic": conv["topic"],
            "message_count": count,
            "importance": conv["importance"],
            "created_at": conv["created_at"],
        }

    async def search_conversations(
        self, query: str, limit: int = 10
    ) -> List[Conversation]:
        """Semantic search over past conversations."""
        if not query:
            return []

        loop = asyncio.get_event_loop()
        q_vec = await loop.run_in_executor(None, lambda: self._embed.encode(query))

        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT conversation_id FROM messages WHERE embedding IS NOT NULL"
            ).fetchall()

        scored: List[tuple] = []
        for row in rows:
            conv_id = row[0]
            with self._conn() as conn:
                msgs = conn.execute(
                    "SELECT embedding FROM messages WHERE conversation_id=? LIMIT 20",
                    (conv_id,),
                ).fetchall()
            if not msgs:
                continue
            # Average embedding for conversation
            vecs = [self._decode(m["embedding"]) for m in msgs if m["embedding"]]
            if not vecs:
                continue
            avg_vec = np.mean(vecs, axis=0)
            score = _cosine(q_vec, avg_vec)
            if score >= self.config.similarity_threshold:
                scored.append((score, conv_id))

        scored.sort(reverse=True)
        results: List[Conversation] = []
        for score, conv_id in scored[:limit]:
            msgs = await self._get_conversation_messages(conv_id, 10)
            summary = await self.get_conversation_summary(conv_id)
            with self._conn() as conn:
                conv_row = conn.execute(
                    "SELECT * FROM conversations WHERE conversation_id=?",
                    (conv_id,),
                ).fetchone()
            if conv_row:
                results.append(
                    Conversation(
                        conversation_id=conv_id,
                        topic=conv_row["topic"],
                        summary=conv_row["summary"],
                        message_count=summary["message_count"],
                        importance_score=score,
                        created_at=conv_row["created_at"],
                        updated_at=conv_row["updated_at"],
                        messages=msgs,
                    )
                )
        return results

    async def _get_semantic_context(
        self, conversation_id: str, message_id: str, limit: int = 5
    ) -> List[ChatMessage]:
        """Find messages semantically similar to a given message."""
        with self._conn() as conn:
            ref = conn.execute(
                "SELECT embedding FROM messages WHERE message_id=?", (message_id,)
            ).fetchone()
        if not ref or not ref["embedding"]:
            return []
        ref_vec = self._decode(ref["embedding"])
        if ref_vec is None:
            return []

        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id=? AND message_id!=?",
                (conversation_id, message_id),
            ).fetchall()

        scored = []
        for row in rows:
            vec = self._decode(row["embedding"])
            if vec is not None:
                scored.append((_cosine(ref_vec, vec), row))

        scored.sort(reverse=True)
        return [
            ChatMessage(
                message_id=r["message_id"],
                conversation_id=r["conversation_id"],
                role=r["role"],
                content=r["content"],
                timestamp=r["timestamp"],
                importance_score=imp,
                metadata=json.loads(r["metadata_json"]),
            )
            for imp, r in scored[:limit]
        ]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        with self._conn() as conn:
            n_conv = conn.execute(
                "SELECT COUNT(*) FROM conversations"
            ).fetchone()[0]
            n_msg = conn.execute(
                "SELECT COUNT(*) FROM messages"
            ).fetchone()[0]
            n_embedded = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE embedding IS NOT NULL"
            ).fetchone()[0]
        return {
            "total_conversations": n_conv,
            "total_messages": n_msg,
            "embedded_messages": n_embedded,
            "topics": len(self.topic_index),
            "db_path": self.db_path,
            "provider": type(self._provider).__name__ if self._provider else "not_loaded",
        }

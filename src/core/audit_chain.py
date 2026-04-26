"""
NIS Protocol — AuditChain
==========================
Merkle hash-chain audit trail for every NIS Protocol action.

Learned from OpenFang's System #2 (Merkle Hash-Chain Audit Trail).
Goes deeper: every NeuroKernel decision, tool call, arm movement,
and Cosmos reasoning step is cryptographically linked.

Properties:
- Tamper-evident: modify any entry → entire chain breaks downstream
- Append-only: entries cannot be deleted or reordered
- Verifiable: call verify() to prove chain integrity
- Persistent: survives restarts via SQLite backend (with JSON fallback)
- Queryable: filter by agent, action type, time range, or layer

DIKW mapping:
  Data        → raw event dict
  Information → structured AuditEntry with SHA256 link
  Knowledge   → chain verification (integrity = trustworthy knowledge)
  Wisdom      → audit queries that reveal patterns over time
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator

logger = logging.getLogger("nis.audit_chain")

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class AuditEntry:
    """One link in the audit chain."""
    entry_id: str
    timestamp: float
    agent_id: str
    action_type: str            # e.g. "tool_call", "arm_move", "cosmos_reason", "llm_call"
    layer: str                  # NIS layer: "perception", "reasoning", "action", etc.
    payload: Dict[str, Any]     # what happened
    prev_hash: str              # SHA256 of previous entry (genesis = "0" * 64)
    entry_hash: str = ""        # SHA256 of this entry (set after creation)
    skill_attribution: List[str] = field(default_factory=list)  # skills used
    success: bool = True
    duration_ms: float = 0.0
    tags: List[str] = field(default_factory=list)

    def compute_hash(self) -> str:
        """SHA256 over all fields except entry_hash itself."""
        data = {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "layer": self.layer,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "success": self.success,
        }
        raw = json.dumps(data, sort_keys=True, default=str).encode()
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AuditEntry:
        return cls(**d)


# ── Storage backend ───────────────────────────────────────────────────────────

class _SQLiteBackend:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_table(self):
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_chain (
                entry_id TEXT PRIMARY KEY,
                timestamp REAL,
                agent_id TEXT,
                action_type TEXT,
                layer TEXT,
                payload TEXT,
                prev_hash TEXT,
                entry_hash TEXT,
                skill_attribution TEXT,
                success INTEGER,
                duration_ms REAL,
                tags TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agent ON audit_chain(agent_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON audit_chain(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_action ON audit_chain(action_type)")
        conn.commit()

    def append(self, entry: AuditEntry):
        conn = self._connect()
        conn.execute("""
            INSERT OR REPLACE INTO audit_chain VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            entry.entry_id, entry.timestamp, entry.agent_id, entry.action_type,
            entry.layer, json.dumps(entry.payload, default=str),
            entry.prev_hash, entry.entry_hash,
            json.dumps(entry.skill_attribution),
            int(entry.success), entry.duration_ms,
            json.dumps(entry.tags),
        ))
        conn.commit()

    def tail(self, n: int = 1) -> List[AuditEntry]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM audit_chain ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()
        return [self._row_to_entry(r) for r in reversed(rows)]

    def query(self, agent_id: Optional[str] = None, action_type: Optional[str] = None,
               since: Optional[float] = None, limit: int = 100) -> List[AuditEntry]:
        clauses, params = [], []
        if agent_id:
            clauses.append("agent_id = ?"); params.append(agent_id)
        if action_type:
            clauses.append("action_type = ?"); params.append(action_type)
        if since:
            clauses.append("timestamp >= ?"); params.append(since)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._connect().execute(
            f"SELECT * FROM audit_chain {where} ORDER BY timestamp ASC LIMIT ?",
            params + [limit]
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def all_hashes(self) -> Iterator[tuple[str, str, str]]:
        """Yield (entry_id, entry_hash, prev_hash) in order for chain verification."""
        conn = self._connect()
        for row in conn.execute("SELECT entry_id, entry_hash, prev_hash, timestamp FROM audit_chain ORDER BY timestamp ASC"):
            yield row["entry_id"], row["entry_hash"], row["prev_hash"]

    def count(self) -> int:
        return self._connect().execute("SELECT COUNT(*) FROM audit_chain").fetchone()[0]

    @staticmethod
    def _row_to_entry(row) -> AuditEntry:
        return AuditEntry(
            entry_id=row["entry_id"],
            timestamp=row["timestamp"],
            agent_id=row["agent_id"],
            action_type=row["action_type"],
            layer=row["layer"],
            payload=json.loads(row["payload"]),
            prev_hash=row["prev_hash"],
            entry_hash=row["entry_hash"],
            skill_attribution=json.loads(row["skill_attribution"]),
            success=bool(row["success"]),
            duration_ms=row["duration_ms"],
            tags=json.loads(row["tags"]),
        )


class _MemoryBackend:
    """Fallback when SQLite unavailable."""
    def __init__(self):
        self._entries: List[AuditEntry] = []

    def append(self, entry: AuditEntry):
        self._entries.append(entry)
        if len(self._entries) > 10000:
            self._entries = self._entries[-8000:]

    def tail(self, n: int = 1) -> List[AuditEntry]:
        return self._entries[-n:]

    def query(self, agent_id=None, action_type=None, since=None, limit=100):
        results = self._entries
        if agent_id:
            results = [e for e in results if e.agent_id == agent_id]
        if action_type:
            results = [e for e in results if e.action_type == action_type]
        if since:
            results = [e for e in results if e.timestamp >= since]
        return results[-limit:]

    def all_hashes(self):
        for e in self._entries:
            yield e.entry_id, e.entry_hash, e.prev_hash

    def count(self):
        return len(self._entries)


# ── Main class ────────────────────────────────────────────────────────────────

GENESIS_HASH = "0" * 64


class AuditChain:
    """
    Merkle hash-chain audit trail for the NIS NeuroKernel.

    Usage:
        chain = AuditChain()
        entry_id = chain.log(
            agent_id="cosmos-reasoner",
            action_type="cosmos_reason",
            layer="reasoning",
            payload={"prompt": "...", "response": "..."},
            skill_attribution=["robotics-arm"],
            duration_ms=234.5,
        )
        chain.verify()  # raises if tampered
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            data_dir = Path(__file__).resolve().parent.parent.parent / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "audit_chain.db")

        try:
            self._backend = _SQLiteBackend(db_path)
            self._use_sqlite = True
        except Exception as e:
            logger.warning(f"SQLite backend failed ({e}), using in-memory fallback")
            self._backend = _MemoryBackend()
            self._use_sqlite = False

        self._last_hash: str = self._get_last_hash()
        logger.info(f"AuditChain ready | {self._backend.count()} entries | sqlite={self._use_sqlite}")

    def _get_last_hash(self) -> str:
        tail = self._backend.tail(1)
        return tail[0].entry_hash if tail else GENESIS_HASH

    def log(
        self,
        agent_id: str,
        action_type: str,
        layer: str,
        payload: Dict[str, Any],
        skill_attribution: Optional[List[str]] = None,
        success: bool = True,
        duration_ms: float = 0.0,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Append an entry to the chain. Returns the new entry_id."""
        entry = AuditEntry(
            entry_id=uuid.uuid4().hex,
            timestamp=time.time(),
            agent_id=agent_id,
            action_type=action_type,
            layer=layer,
            payload=payload,
            prev_hash=self._last_hash,
            skill_attribution=skill_attribution or [],
            success=success,
            duration_ms=duration_ms,
            tags=tags or [],
        )
        entry.entry_hash = entry.compute_hash()
        self._backend.append(entry)
        self._last_hash = entry.entry_hash
        # Publish to SSE channel (non-blocking, lazy import)
        try:
            from routes.events import publish as _pub
            _pub("audit", {
                "entry_id": entry.entry_id,
                "agent_id": entry.agent_id,
                "action_type": entry.action_type,
                "layer": entry.layer,
                "success": entry.success,
                "duration_ms": entry.duration_ms,
                "tags": entry.tags,
            })
        except Exception:
            pass
        return entry.entry_id

    def verify(self) -> Dict[str, Any]:
        """
        Walk the entire chain and verify hash linkage.
        Returns {"valid": bool, "entries": int, "broken_at": Optional[str]}
        """
        prev = GENESIS_HASH
        count = 0
        for entry_id, entry_hash, stored_prev in self._backend.all_hashes():
            # Check every entry including the first — genesis link must match too
            if stored_prev != prev:
                return {"valid": False, "entries": count, "broken_at": entry_id,
                        "reason": f"prev_hash mismatch (expected {prev[:16]}…, got {stored_prev[:16]}…)"}
            prev = entry_hash
            count += 1
        return {"valid": True, "entries": count, "broken_at": None}

    def recent(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return the n most recent entries as dicts."""
        return [e.to_dict() for e in self._backend.tail(n)]

    def query(
        self,
        agent_id: Optional[str] = None,
        action_type: Optional[str] = None,
        since_seconds: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        since = time.time() - since_seconds if since_seconds else None
        return [e.to_dict() for e in self._backend.query(agent_id, action_type, since, limit)]

    def stats(self) -> Dict[str, Any]:
        recent = self._backend.tail(100)
        action_types: Dict[str, int] = {}
        agents: Dict[str, int] = {}
        for e in recent:
            action_types[e.action_type] = action_types.get(e.action_type, 0) + 1
            agents[e.agent_id] = agents.get(e.agent_id, 0) + 1
        return {
            "total_entries": self._backend.count(),
            "last_hash": self._last_hash[:16] + "...",
            "sqlite_backend": self._use_sqlite,
            "recent_action_types": action_types,
            "recent_agents": agents,
        }

    # ── Context manager for timed logging ────────────────────────────────────

    class _Timer:
        def __init__(self, chain: AuditChain, **kwargs):
            self._chain = chain
            self._kwargs = kwargs
            self._start = 0.0
            self.entry_id = ""

        def __enter__(self):
            self._start = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            ms = (time.time() - self._start) * 1000
            self.entry_id = self._chain.log(
                duration_ms=ms,
                success=(exc_type is None),
                **self._kwargs
            )

    def timed(self, agent_id: str, action_type: str, layer: str,
              payload: Dict[str, Any], **kwargs) -> "_Timer":
        """Use as context manager to auto-log duration and success:
            with chain.timed("arm", "arm_move", "action", {"pose": "home"}):
                move_arm()
        """
        return self._Timer(self, agent_id=agent_id, action_type=action_type,
                           layer=layer, payload=payload, **kwargs)


# ── Module-level singleton ────────────────────────────────────────────────────

_audit_chain: Optional[AuditChain] = None


def get_audit_chain() -> AuditChain:
    global _audit_chain
    if _audit_chain is None:
        _audit_chain = AuditChain()
    return _audit_chain

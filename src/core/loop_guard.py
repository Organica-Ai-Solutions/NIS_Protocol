"""
NIS Protocol — LoopGuard
========================
SHA256-based tool-call loop detection and circuit breaker.

Learned from OpenFang's System #13 (Loop Guard).
Goes deeper: tracks ping-pong patterns, detects semantic loops
(same intent, different wording), and integrates with the AuditChain.

Problems it solves:
  - Agents calling the same tool repeatedly with identical args (exact loop)
  - Agents oscillating between two states (A→B→A→B ping-pong)
  - LLM re-generating the same plan with minor rephrasing (semantic loop)
  - Infinite reasoning chains without progress (liveness check)

DIKW mapping:
  Data        → raw tool call fingerprints (SHA256)
  Information → call pattern with frequency and recency
  Knowledge   → "this is a loop" judgment (pattern + threshold)
  Wisdom      → circuit breaker decision (break loop, escalate, or reset)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("nis.loop_guard")


# ── Fingerprinting ─────────────────────────────────────────────────────────────

def _fingerprint(tool_name: str, args: Dict[str, Any]) -> str:
    """SHA256 fingerprint of a tool call."""
    raw = json.dumps({"t": tool_name, "a": args}, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _semantic_fingerprint(text: str) -> str:
    """Coarse semantic fingerprint: strip stopwords, sort, hash."""
    STOPWORDS = {"the", "a", "an", "to", "for", "of", "is", "in", "it", "and",
                 "or", "this", "that", "with", "be", "as", "at", "by", "on"}
    words = sorted(
        w.lower() for w in text.replace("_", " ").split()
        if w.lower() not in STOPWORDS and len(w) > 2
    )
    raw = " ".join(words[:20]).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class CallRecord:
    fingerprint: str
    tool_name: str
    timestamp: float
    args_summary: str


@dataclass
class LoopReport:
    detected: bool
    loop_type: str       # "exact", "ping_pong", "semantic", "liveness"
    fingerprint: str
    count: int           # how many times this pattern appeared
    window_secs: float   # within this time window
    recommendation: str  # "break", "warn", "allow"
    details: str


# ── Core ────────────────────────────────────────────────────────────────────────

class LoopGuard:
    """
    Circuit breaker for autonomous agent tool-call loops.

    Usage:
        guard = LoopGuard()

        # Before every tool call:
        report = guard.check("xarm_control", {"command": "home"}, context_id="session-1")
        if report.recommendation == "break":
            raise CircuitBreakerTripped(report.details)

        # After the call completes:
        guard.record("xarm_control", {"command": "home"}, context_id="session-1")
    """

    def __init__(
        self,
        window_secs: float = 60.0,        # rolling window for loop detection
        exact_threshold: int = 3,          # same fingerprint N times → loop
        pingpong_threshold: int = 4,       # A→B→A→B N times → loop
        semantic_threshold: int = 5,       # similar intent N times → loop
        liveness_timeout: float = 120.0,   # no progress for N secs → loop
        max_history: int = 200,            # per-context call history size
    ):
        self.window_secs = window_secs
        self.exact_threshold = exact_threshold
        self.pingpong_threshold = pingpong_threshold
        self.semantic_threshold = semantic_threshold
        self.liveness_timeout = liveness_timeout
        self.max_history = max_history

        # context_id → deque of CallRecord
        self._history: Dict[str, Deque[CallRecord]] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )
        # context_id → last progress timestamp
        self._last_progress: Dict[str, float] = {}
        # context_id → set of tripped fingerprints (circuit open)
        self._open_circuits: Dict[str, set] = defaultdict(set)

    # ── Check + record ─────────────────────────────────────────────────────────

    def check(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context_id: str = "default",
        semantic_text: Optional[str] = None,
    ) -> LoopReport:
        """
        Check if this tool call would be part of a loop.
        Call BEFORE executing the tool.
        """
        fp = _fingerprint(tool_name, args)
        now = time.time()
        history = self._history[context_id]
        window_start = now - self.window_secs

        # Trim old records
        while history and history[0].timestamp < window_start:
            history.popleft()

        recent = list(history)

        # 1. Exact loop check
        exact_count = sum(1 for r in recent if r.fingerprint == fp)
        if exact_count >= self.exact_threshold:
            self._open_circuits[context_id].add(fp)
            return LoopReport(
                detected=True, loop_type="exact", fingerprint=fp,
                count=exact_count, window_secs=self.window_secs,
                recommendation="break",
                details=f"Tool '{tool_name}' called {exact_count}x with identical args in {self.window_secs}s",
            )

        # 2. Ping-pong check (A→B→A→B…) — uses pingpong_threshold, not hardcoded 4
        n = self.pingpong_threshold
        if len(recent) >= n:
            fps = [r.fingerprint for r in recent[-n:]]
            # Alternating: all even-indexed identical, all odd-indexed identical, and they differ
            even_fps = fps[0::2]
            odd_fps  = fps[1::2]
            if (len(set(fps)) == 2
                    and len(set(even_fps)) == 1
                    and len(set(odd_fps)) == 1):
                return LoopReport(
                    detected=True, loop_type="ping_pong", fingerprint=fp,
                    count=n, window_secs=self.window_secs,
                    recommendation="break",
                    details=f"Ping-pong detected: alternating between 2 tool calls",
                )

        # 3. Semantic loop check
        if semantic_text:
            sfp = _semantic_fingerprint(semantic_text)
            semantic_count = sum(
                1 for r in recent
                if _semantic_fingerprint(r.args_summary) == sfp
            )
            if semantic_count >= self.semantic_threshold:
                return LoopReport(
                    detected=True, loop_type="semantic", fingerprint=sfp,
                    count=semantic_count, window_secs=self.window_secs,
                    recommendation="warn",
                    details=f"Semantic loop: similar intent repeated {semantic_count}x",
                )

        # 4. Liveness check
        last_prog = self._last_progress.get(context_id, now)
        if (now - last_prog) > self.liveness_timeout and len(recent) > 3:
            return LoopReport(
                detected=True, loop_type="liveness", fingerprint=fp,
                count=len(recent), window_secs=now - last_prog,
                recommendation="warn",
                details=f"No progress detected for {now - last_prog:.0f}s with {len(recent)} tool calls",
            )

        return LoopReport(
            detected=False, loop_type="none", fingerprint=fp,
            count=exact_count + 1, window_secs=self.window_secs,
            recommendation="allow",
            details="OK",
        )

    def record(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context_id: str = "default",
        made_progress: bool = True,
    ):
        """Record a completed tool call. Call AFTER executing the tool."""
        fp = _fingerprint(tool_name, args)
        self._history[context_id].append(CallRecord(
            fingerprint=fp,
            tool_name=tool_name,
            timestamp=time.time(),
            args_summary=json.dumps(args, default=str)[:120],
        ))
        if made_progress:
            self._last_progress[context_id] = time.time()
            # Clear open circuit for this fingerprint if we made progress
            self._open_circuits[context_id].discard(fp)

    def reset(self, context_id: str = "default"):
        """Reset all loop state for a context (e.g. new conversation)."""
        self._history.pop(context_id, None)
        self._last_progress.pop(context_id, None)
        self._open_circuits.pop(context_id, None)

    def is_open(self, tool_name: str, args: Dict[str, Any], context_id: str = "default") -> bool:
        """Check if circuit is already open for this call (fast path)."""
        fp = _fingerprint(tool_name, args)
        return fp in self._open_circuits.get(context_id, set())

    # ── Introspection ──────────────────────────────────────────────────────────

    def stats(self, context_id: Optional[str] = None) -> Dict[str, Any]:
        contexts = [context_id] if context_id else list(self._history.keys())
        result = {}
        for ctx in contexts:
            hist = list(self._history.get(ctx, []))
            tool_counts: Dict[str, int] = {}
            for r in hist:
                tool_counts[r.tool_name] = tool_counts.get(r.tool_name, 0) + 1
            result[ctx] = {
                "history_size": len(hist),
                "open_circuits": len(self._open_circuits.get(ctx, set())),
                "tool_counts": tool_counts,
                "last_progress": self._last_progress.get(ctx),
            }
        return result


class CircuitBreakerTripped(Exception):
    """Raised when the LoopGuard circuit breaker opens."""
    def __init__(self, report: LoopReport):
        self.report = report
        super().__init__(f"[LoopGuard:{report.loop_type}] {report.details}")


# ── Decorator for automatic protection ────────────────────────────────────────

def loop_protected(guard: LoopGuard, context_id: str = "default", break_on_detect: bool = True):
    """
    Decorator to automatically check + record tool calls.

    @loop_protected(guard)
    async def my_tool(tool_name, args):
        ...
    """
    import functools

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(tool_name: str, args: Dict[str, Any], *a, **kw):
            report = guard.check(tool_name, args, context_id)
            if report.detected:
                logger.warning(f"LoopGuard: {report.details}")
                if break_on_detect and report.recommendation == "break":
                    raise CircuitBreakerTripped(report)
            result = await fn(tool_name, args, *a, **kw)
            guard.record(tool_name, args, context_id, made_progress=True)
            return result
        return wrapper
    return decorator


# ── Module-level singleton ─────────────────────────────────────────────────────

_loop_guard: Optional[LoopGuard] = None


def get_loop_guard() -> LoopGuard:
    global _loop_guard
    if _loop_guard is None:
        _loop_guard = LoopGuard()
    return _loop_guard

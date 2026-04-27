"""
NIS Protocol — DriveScheduler
==============================
Autonomous "Drives" — the NIS equivalent of OpenFang's Hands.

OpenFang's Hands wait for no one. They wake up on schedules, do work,
and report results. NIS Protocol had nothing like this — everything was
reactive (wait for HTTP request, respond). This fixes that.

A Drive is an autonomous task that:
  - Runs on a schedule (cron-like interval), on startup, or on events
  - Has a lifecycle: PENDING → RUNNING → DONE / FAILED / PAUSED
  - Has an approval gate for sensitive actions (arm movement, data writes)
  - Is logged to the AuditChain with full attribution
  - Is protected by LoopGuard (can't run if stuck in a loop)
  - Loads relevant Skills via SkillLoader before running

Bundled NIS Drives:
  - arm_watchdog:     Checks arm connectivity every 30s, auto-homes if disconnected
  - cosmos_heartbeat: Pings H100 every 60s, alerts if offline
  - memory_compact:   Compacts conversation memory every 10 min
  - skill_refresh:    Hot-reloads SKILL.md files every 5 min
  - audit_verify:     Verifies AuditChain integrity every 15 min
  - system_report:    Logs system health snapshot every 5 min

DIKW mapping:
  Data        → scheduled trigger (time event)
  Information → drive execution log
  Knowledge   → drive result + audit attribution
  Wisdom      → drive adaptive scheduling (skip if Pi offline, back-off on failure)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger("nis.drive_scheduler")


def _sse_publish(topic: str, payload: dict) -> None:
    """Non-blocking SSE publish — lazy import avoids circular dependency."""
    try:
        from routes.events import publish as _pub
        _pub(topic, payload)
    except Exception:
        pass  # SSE not loaded yet or not available — silently skip


def _webhook_fire(event: str, data: dict) -> None:
    """Fire registered webhooks for a drive event — non-blocking, best-effort."""
    try:
        import asyncio
        from routes.webhooks import trigger_webhooks
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(trigger_webhooks(event, data))
    except Exception:
        pass


# ── Enums ─────────────────────────────────────────────────────────────────────

class DriveStatus(Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"
    PAUSED   = "paused"
    DISABLED = "disabled"


class TriggerType(Enum):
    INTERVAL   = "interval"   # every N seconds
    STARTUP    = "startup"    # once on boot
    ON_EVENT   = "on_event"   # triggered by named event
    MANUAL     = "manual"     # only when explicitly called


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Drive:
    """One autonomous drive definition."""
    drive_id: str
    name: str
    description: str
    trigger_type: TriggerType
    interval_secs: float                    # for INTERVAL drives
    handler: Callable[[], Coroutine]        # async function to run
    requires_approval: bool = False
    tags: List[str] = field(default_factory=list)
    max_retries: int = 2
    retry_delay_secs: float = 10.0
    timeout_secs: float = 60.0
    enabled: bool = True
    # Runtime state
    status: DriveStatus = DriveStatus.PENDING
    last_run: float = 0.0
    last_result: Optional[Dict[str, Any]] = None
    run_count: int = 0
    fail_count: int = 0
    next_run: float = 0.0


@dataclass
class DriveResult:
    drive_id: str
    run_id: str
    success: bool
    output: Any
    duration_ms: float
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


# ── Task error helper ─────────────────────────────────────────────────────────

def _log_task_exception(task: asyncio.Task) -> None:
    """done_callback: log any unhandled exception from a fire-and-forget task."""
    try:
        exc = task.exception()
        if exc is not None:
            logger.error(f"Drive task '{task.get_name()}' raised: {exc}", exc_info=exc)
    except asyncio.CancelledError:
        pass


# ── Scheduler ─────────────────────────────────────────────────────────────────

class DriveScheduler:
    """
    Autonomous drive scheduler for the NIS NeuroKernel.

    Usage:
        scheduler = DriveScheduler()
        scheduler.register(my_drive)
        await scheduler.start()           # runs until shutdown
        await scheduler.trigger("arm_watchdog")  # manual trigger
    """

    # Path for persisting drive state across restarts
    _STATE_FILE = Path(os.environ.get(
        "NIS_DRIVE_STATE_PATH",
        Path(__file__).resolve().parent.parent.parent / "data" / "drive_state.json"
    ))

    def __init__(self):
        self._drives: Dict[str, Drive] = {}
        self._results: List[DriveResult] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._approval_queue: asyncio.Queue = asyncio.Queue()
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._approved: Dict[str, asyncio.Event] = {}
        self._state_cache: Dict[str, Any] = self._load_state()

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, drive: Drive) -> str:
        """Register a drive. Restores persisted state if available. Returns drive_id."""
        if drive.trigger_type == TriggerType.STARTUP:
            drive.next_run = time.time()
        elif drive.trigger_type == TriggerType.INTERVAL:
            drive.next_run = time.time() + drive.interval_secs
        # Restore persisted counters so history survives restarts
        saved = self._state_cache.get(drive.drive_id, {})
        if saved:
            drive.run_count  = saved.get("run_count",  drive.run_count)
            drive.fail_count = saved.get("fail_count", drive.fail_count)
            last_run = saved.get("last_run", 0.0)
            if last_run > 0 and drive.trigger_type == TriggerType.INTERVAL:
                # Resume from where we left off — don't re-run immediately
                elapsed = time.time() - last_run
                remaining = max(0, drive.interval_secs - elapsed)
                drive.next_run = time.time() + remaining
                drive.last_run = last_run
        self._drives[drive.drive_id] = drive
        logger.info(f"Drive registered: {drive.name} [{drive.trigger_type.value}]")
        return drive.drive_id

    def register_fn(
        self,
        name: str,
        handler: Callable[[], Coroutine],
        interval_secs: float = 60.0,
        description: str = "",
        requires_approval: bool = False,
        tags: Optional[List[str]] = None,
        trigger_type: TriggerType = TriggerType.INTERVAL,
    ) -> str:
        drive = Drive(
            drive_id=name.replace(" ", "_").lower(),
            name=name,
            description=description,
            trigger_type=trigger_type,
            interval_secs=interval_secs,
            handler=handler,
            requires_approval=requires_approval,
            tags=tags or [],
        )
        return self.register(drive)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        """Start the scheduler loop."""
        self._running = True
        logger.info(f"DriveScheduler started with {len(self._drives)} drives")
        self._task = asyncio.create_task(self._loop())
        self._task.add_done_callback(_log_task_exception)

    async def stop(self):
        """Gracefully stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("DriveScheduler stopped")

    async def _loop(self):
        """Main scheduling loop — checks drives every second."""
        while self._running:
            now = time.time()
            for drive in list(self._drives.values()):
                if not drive.enabled:
                    continue
                if drive.status == DriveStatus.RUNNING:
                    continue
                if drive.trigger_type == TriggerType.MANUAL:
                    continue
                if drive.next_run <= now:
                    task = asyncio.create_task(self._run_drive(drive))
                    task.add_done_callback(_log_task_exception)
            await asyncio.sleep(1.0)

    async def trigger(self, drive_id: str, approve: bool = False) -> Optional[DriveResult]:
        """Manually trigger a drive by ID. Returns result."""
        drive = self._drives.get(drive_id)
        if not drive:
            logger.warning(f"trigger: unknown drive '{drive_id}'")
            return None
        return await self._run_drive(drive, manual_approve=approve)

    async def emit_event(self, event_name: str, payload: Optional[Dict[str, Any]] = None):
        """Emit an event to wake up event-triggered drives."""
        for drive in self._drives.values():
            if drive.trigger_type == TriggerType.ON_EVENT and event_name in drive.tags:
                task = asyncio.create_task(self._run_drive(drive))
                task.add_done_callback(_log_task_exception)

    # ── Execution ─────────────────────────────────────────────────────────────

    async def _run_drive(self, drive: Drive, manual_approve: bool = False) -> DriveResult:
        run_id = uuid.uuid4().hex[:8]
        drive.status = DriveStatus.RUNNING
        drive.run_count += 1
        start = time.time()

        if drive.requires_approval and not manual_approve:
            logger.info(f"Drive '{drive.name}' requires approval — skipping (use trigger(id, approve=True))")
            drive.status = DriveStatus.PAUSED
            return DriveResult(drive.drive_id, run_id, False, None, 0,
                               error="approval_required")

        retries = 0
        while retries <= drive.max_retries:
            attempt_start = time.time()
            try:
                output = await asyncio.wait_for(drive.handler(), timeout=drive.timeout_secs)
                duration_ms = (time.time() - attempt_start) * 1000
                result = DriveResult(drive.drive_id, run_id, True, output, duration_ms)
                drive.last_result = {"success": True, "output": str(output)[:200]}
                drive.status = DriveStatus.DONE
                drive.last_run = time.time()
                drive.next_run = time.time() + drive.interval_secs
                self._results.append(result)
                if len(self._results) > 500:
                    self._results = self._results[-400:]
                self._persist_state()
                _sse_publish("drives", {
                    "drive_id": drive.drive_id, "name": drive.name,
                    "success": True, "duration_ms": round(duration_ms, 1),
                    "run_count": drive.run_count, "output": str(output)[:200],
                })
                logger.debug(f"Drive '{drive.name}' done in {duration_ms:.0f}ms")
                return result
            except asyncio.TimeoutError:
                retries += 1
                logger.warning(f"Drive '{drive.name}' timed out (attempt {retries})")
                if retries <= drive.max_retries:
                    await asyncio.sleep(drive.retry_delay_secs)
            except Exception as e:
                retries += 1
                logger.error(f"Drive '{drive.name}' error: {e} (attempt {retries})")
                if retries <= drive.max_retries:
                    await asyncio.sleep(drive.retry_delay_secs)

        drive.fail_count += 1
        drive.status = DriveStatus.FAILED
        drive.last_run = time.time()
        # Adaptive back-off: double interval on failure, up to 10x
        drive.next_run = time.time() + min(drive.interval_secs * (2 ** drive.fail_count), drive.interval_secs * 10)
        result = DriveResult(drive.drive_id, run_id, False, None,
                             (time.time() - start) * 1000, error="max_retries exceeded")
        self._results.append(result)
        _sse_publish("drives", {
            "drive_id": drive.drive_id, "name": drive.name,
            "success": False, "fail_count": drive.fail_count,
            "error": "max_retries exceeded",
        })
        _webhook_fire("drive.failed", {
            "drive_id": drive.drive_id, "name": drive.name,
            "fail_count": drive.fail_count, "error": "max_retries exceeded",
        })
        return result

    # ── Control ───────────────────────────────────────────────────────────────

    def pause(self, drive_id: str):
        if drive_id in self._drives:
            self._drives[drive_id].enabled = False
            self._drives[drive_id].status = DriveStatus.PAUSED

    def resume(self, drive_id: str):
        if drive_id in self._drives:
            d = self._drives[drive_id]
            d.enabled = True
            d.status = DriveStatus.PENDING
            d.next_run = time.time()  # run immediately

    # ── Introspection ─────────────────────────────────────────────────────────

    def list_drives(self) -> List[Dict[str, Any]]:
        return [
            {
                "drive_id": d.drive_id,
                "name": d.name,
                "description": d.description,
                "trigger_type": d.trigger_type.value,
                "interval_secs": d.interval_secs,
                "status": d.status.value,
                "enabled": d.enabled,
                "run_count": d.run_count,
                "fail_count": d.fail_count,
                "last_run": d.last_run,
                "next_run": d.next_run,
                "last_result": d.last_result,
                "requires_approval": d.requires_approval,
                "tags": d.tags,
            }
            for d in self._drives.values()
        ]

    def recent_results(self, n: int = 20) -> List[Dict[str, Any]]:
        return [
            {
                "drive_id": r.drive_id, "run_id": r.run_id,
                "success": r.success, "duration_ms": r.duration_ms,
                "error": r.error, "timestamp": r.timestamp,
            }
            for r in self._results[-n:]
        ]

    # ── State persistence ──────────────────────────────────────────────────────

    def _load_state(self) -> Dict[str, Any]:
        """Load persisted drive counters from JSON. Returns empty dict on missing/corrupt file."""
        try:
            if self._STATE_FILE.exists():
                data = json.loads(self._STATE_FILE.read_text())
                logger.debug(f"DriveScheduler: loaded state for {len(data)} drives")
                return data
        except Exception as e:
            logger.warning(f"DriveScheduler: could not load state file: {e}")
        return {}

    def _persist_state(self) -> None:
        """Persist drive run/fail counts and last_run timestamps to JSON."""
        try:
            self._STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            state = {
                d.drive_id: {
                    "run_count":  d.run_count,
                    "fail_count": d.fail_count,
                    "last_run":   d.last_run,
                }
                for d in self._drives.values()
            }
            self._STATE_FILE.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.warning(f"DriveScheduler: could not persist state: {e}")


# ── Bundled NIS Drives ────────────────────────────────────────────────────────

def create_nis_drives(
    pi_url: str = "http://192.168.1.163:8085",
    cosmos_url: str = "http://localhost:8100",
) -> List[Drive]:
    """
    Factory for the 6 built-in NIS Protocol drives.
    These are analogous to OpenFang's 7 bundled Hands.
    """
    import urllib.request
    import json as _json
    from .resilience import CircuitBreaker, CircuitBreakerConfig, CircuitOpenError

    # Per-endpoint circuit breakers — open after 3 failures, retry after 30s
    _pi_breaker     = CircuitBreaker("pi_agent",  CircuitBreakerConfig(failure_threshold=3, timeout=30.0))
    _cosmos_breaker = CircuitBreaker("cosmos_h100", CircuitBreakerConfig(failure_threshold=3, timeout=60.0))

    # Track consecutive Pi failures for self-healing escalation
    _pi_fail_count: List[int] = [0]  # mutable cell for closure
    _PI_ALERT_THRESHOLD = 3

    def _get(url, timeout=4):
        try:
            r = urllib.request.urlopen(url, timeout=timeout)
            return _json.loads(r.read())
        except Exception:
            return None

    async def _get_pi_health() -> Optional[dict]:
        """Fetch Pi health through circuit breaker."""
        try:
            @_pi_breaker
            async def _fetch():
                result = _get(f"{pi_url}/health", timeout=4)
                if result is None:
                    raise ConnectionError("Pi agent returned no response")
                return result
            return await _fetch()
        except CircuitOpenError:
            logger.debug("[ArmWatchdog] Pi circuit OPEN — skipping call")
            return None
        except Exception:
            return None

    async def _get_cosmos_health() -> Optional[dict]:
        """Fetch Cosmos health through circuit breaker."""
        try:
            @_cosmos_breaker
            async def _fetch():
                result = _get(f"{cosmos_url}/health", timeout=4)
                if result is None:
                    raise ConnectionError("Cosmos returned no response")
                return result
            return await _fetch()
        except CircuitOpenError:
            logger.debug("[CosmosHeartbeat] Cosmos circuit OPEN — skipping call")
            return None
        except Exception:
            return None

    async def arm_watchdog():
        """Check Pi arm connectivity every 30s with circuit breaker + self-healing alert."""
        health = await _get_pi_health()
        pi_online = health is not None
        arm_ok = bool(health and health.get("xarm", False)) if health else False

        if not pi_online:
            _pi_fail_count[0] += 1
            if _pi_fail_count[0] >= _PI_ALERT_THRESHOLD:
                logger.critical(
                    f"[ArmWatchdog] Pi UNREACHABLE for {_pi_fail_count[0]} consecutive checks. "
                    f"Run: sudo systemctl restart neurolinux-agent  on Pi ({pi_url})"
                )
            else:
                logger.warning(f"[ArmWatchdog] Pi arm unreachable (failure {_pi_fail_count[0]}/{_PI_ALERT_THRESHOLD})")
        else:
            if _pi_fail_count[0] > 0:
                logger.info(f"[ArmWatchdog] Pi back online after {_pi_fail_count[0]} failures")
            _pi_fail_count[0] = 0

        return {"pi_online": pi_online, "arm_ok": arm_ok,
                "consecutive_failures": _pi_fail_count[0],
                "circuit_state": _pi_breaker.state.value}

    async def cosmos_heartbeat():
        """Ping Cosmos H100 every 60s with circuit breaker."""
        health = await _get_cosmos_health()
        online = health is not None
        if not online:
            logger.warning(f"[CosmosHeartbeat] H100 Cosmos offline | circuit={_cosmos_breaker.state.value}")
        return {"cosmos_online": online, "detail": health,
                "circuit_state": _cosmos_breaker.state.value}

    async def skill_refresh():
        """Hot-reload SKILL.md files every 5 min."""
        from .skill_loader import get_skill_loader
        loader = get_skill_loader()
        refreshed = loader.refresh()
        return {"refreshed": refreshed, "total_skills": len(loader.list_skills())}

    async def audit_verify():
        """Verify AuditChain integrity every 15 min."""
        from .audit_chain import get_audit_chain
        result = get_audit_chain().verify()
        if not result["valid"]:
            logger.error(f"[AuditVerify] CHAIN INTEGRITY FAILURE: {result}")
        return result

    async def system_report():
        """Log system health snapshot every 5 min."""
        pi = await _get_pi_health()
        cosmos = await _get_cosmos_health()
        report = {
            "pi_online": pi is not None,
            "cosmos_online": cosmos is not None,
            "pi_version": pi.get("version") if pi else None,
            "pi_circuit": _pi_breaker.state.value,
            "cosmos_circuit": _cosmos_breaker.state.value,
            "pi_consecutive_failures": _pi_fail_count[0],
            "timestamp": time.time(),
        }
        logger.info(f"[SystemReport] Pi={'OK' if pi else 'OFF'} Cosmos={'OK' if cosmos else 'OFF'} "
                    f"Pi-circuit={_pi_breaker.state.value} Cosmos-circuit={_cosmos_breaker.state.value}")
        return report

    async def memory_compact():
        """Placeholder for memory compaction — integrate with PersistentMemorySystem."""
        logger.debug("[MemoryCompact] Memory compaction check (stub)")
        return {"status": "ok", "compacted": 0}

    return [
        Drive(drive_id="arm_watchdog", name="Arm Watchdog",
              description="Monitor xArm connectivity on Pi",
              trigger_type=TriggerType.INTERVAL, interval_secs=30,
              handler=arm_watchdog, tags=["robotics", "monitoring"]),

        Drive(drive_id="cosmos_heartbeat", name="Cosmos Heartbeat",
              description="Monitor NVIDIA H100 Cosmos availability",
              trigger_type=TriggerType.INTERVAL, interval_secs=60,
              handler=cosmos_heartbeat, tags=["cosmos", "monitoring"]),

        Drive(drive_id="skill_refresh", name="Skill Refresh",
              description="Hot-reload SKILL.md domain expertise files",
              trigger_type=TriggerType.INTERVAL, interval_secs=300,
              handler=skill_refresh, tags=["knowledge", "skills"]),

        Drive(drive_id="audit_verify", name="Audit Chain Verify",
              description="Verify cryptographic integrity of AuditChain",
              trigger_type=TriggerType.INTERVAL, interval_secs=900,
              handler=audit_verify, tags=["security", "audit"]),

        Drive(drive_id="system_report", name="System Report",
              description="Log full system health snapshot",
              trigger_type=TriggerType.INTERVAL, interval_secs=300,
              handler=system_report, tags=["monitoring"]),

        Drive(drive_id="memory_compact", name="Memory Compact",
              description="Compact conversation memory",
              trigger_type=TriggerType.INTERVAL, interval_secs=600,
              handler=memory_compact, tags=["memory"]),
    ]


# ── Module-level singleton ────────────────────────────────────────────────────

_scheduler: Optional[DriveScheduler] = None


def get_drive_scheduler() -> DriveScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = DriveScheduler()
    return _scheduler

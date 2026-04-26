"""
NIS Protocol v4.0 — System Autonomy Engine
==========================================

Inspired by Claude Code / OpenClaw / OpenFang: the system stays working
autonomously — it does not wait to be asked.

Architecture:
  AutonomyEngine (singleton)
    ├── TaskQueue       — persistent JSONL queue survives restarts
    ├── AgentLoop       — continuous asyncio loop: pop → plan → execute → learn
    ├── EventBus        — YOLO detections / voice / sensor spikes trigger tasks
    ├── WatchdogLoop    — self-healing: detect down services → restart them
    ├── GoalMemory      — remember what was running before crash / restart
    └── SSE stream      — /autonomy/stream  real-time feed to UI

Key behaviors (what was missing):
  1. Continuous loop — keeps running without user input
  2. Event-driven    — YOLO sees object → auto-queues cookoff task
  3. Self-healing    — NIS / agent / H100 down → auto-restart / alert
  4. Goal persistence — tasks survive service restart (disk-backed queue)
  5. LLM planning    — Cosmos R2 decides WHAT to do next given context

Endpoints:
  GET  /autonomy/status          — engine state, queue, recent actions
  POST /autonomy/start           — start engine
  POST /autonomy/stop            — pause engine
  POST /autonomy/task            — inject a task (text, priority)
  POST /autonomy/trigger         — fire an event (yolo_detection, voice, sensor)
  GET  /autonomy/stream          — SSE feed of live activity
  GET  /autonomy/history         — last N completed tasks
  POST /autonomy/goal            — set persistent goal (survives restart)
  DELETE /autonomy/goal          — clear goal
  GET  /autonomy/watchdog        — watchdog status for all services
"""

import asyncio
import collections
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("nis.autonomy")

router = APIRouter(prefix="/autonomy", tags=["Autonomy Engine"])

# ── Config ────────────────────────────────────────────────────────────────────
AGENT_URL   = os.getenv("AGENT_URL",       "http://localhost:8085")
REASON2_URL = os.getenv("H100_REASON_URL", "http://localhost:8100")
NIS_URL     = os.getenv("NIS_URL",         "http://localhost:8000")

QUEUE_PATH   = Path("/opt/nis-protocol/autonomy_queue.jsonl")
MEMORY_PATH  = Path("/opt/nis-protocol/autonomy_memory.json")
CONTEXT_PATH = Path("/opt/nis-protocol/nis_context.json")   # inter-session context
FAILURE_PATH = Path("/opt/nis-protocol/failure_memory.json") # failure history for R2
HISTORY_MAX  = 200

# YOLO cooldown — same label+zone only triggers once per N seconds
YOLO_COOLDOWN_S  = 60
_yolo_last_trigger: Dict[str, float] = {}

# Services the watchdog monitors
WATCHDOG_SERVICES = [
    {"name": "nis-protocol",    "url": f"{NIS_URL}/health",     "systemd": "nis-protocol"},
    {"name": "neurolinux-agent","url": f"{AGENT_URL}/health",   "systemd": "neurolinux-agent"},
    {"name": "cosmos-reason2",  "url": f"{REASON2_URL}/health", "systemd": None, "ssh_restart": True},
    {"name": "neurohub-ui",     "url": "http://localhost:3000",  "systemd": "neurohub-ui"},
]

# ── Task priority levels ──────────────────────────────────────────────────────
PRIORITY = {"critical": 0, "high": 1, "normal": 2, "low": 3}


# ── Pydantic models ───────────────────────────────────────────────────────────
class TaskRequest(BaseModel):
    description: str
    priority:    str  = Field(default="normal", pattern="critical|high|normal|low")
    source:      str  = Field(default="user")
    context:     Optional[Dict[str, Any]] = None
    execute_arm: bool = Field(default=True)

class TriggerRequest(BaseModel):
    event:   str                           # yolo_detection | voice_command | sensor_spike | manual
    payload: Optional[Dict[str, Any]] = None

class GoalRequest(BaseModel):
    goal:        str
    description: Optional[str] = None


# ── Inter-session NIS context (survives restarts) ────────────────────────────
class NISContext:
    """Writes a JSON file after every task so context survives NIS restarts."""
    _defaults: Dict = {
        "last_task": None, "last_goal": None, "last_scene": None,
        "arm_last_position": None, "tasks_lifetime": 0,
        "session_count": 0, "updated_at": None,
    }

    def __init__(self):
        self._data: Dict = dict(self._defaults)
        self._load()
        self._data["session_count"] = self._data.get("session_count", 0) + 1
        self._save()

    def _load(self):
        if CONTEXT_PATH.exists():
            try:
                self._data = {**self._defaults,
                              **json.loads(CONTEXT_PATH.read_text(encoding="utf-8"))}
            except Exception:
                pass

    def _save(self):
        try:
            CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._data["updated_at"] = time.time()
            CONTEXT_PATH.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("NISContext save error: %s", e)

    def record_task(self, description: str, ok: bool, scene: str = ""):
        self._data["last_task"]       = {"desc": description[:120], "ok": ok, "ts": time.time()}
        self._data["tasks_lifetime"]  = self._data.get("tasks_lifetime", 0) + 1
        if scene:
            self._data["last_scene"]  = scene
        self._save()

    def record_goal(self, goal: Optional[str]):
        self._data["last_goal"] = goal
        self._save()

    def snapshot(self) -> Dict:
        return dict(self._data)


# ── Failure Memory — feed back to R2 so it learns from mistakes ───────────────
class FailureMemory:
    """Tracks recent failures per task-type so R2 can avoid repeating them."""
    MAX_FAILURES = 40

    def __init__(self):
        self._failures: List[Dict] = []
        self._load()

    def _load(self):
        if FAILURE_PATH.exists():
            try:
                self._failures = json.loads(FAILURE_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass

    def _save(self):
        try:
            FAILURE_PATH.parent.mkdir(parents=True, exist_ok=True)
            FAILURE_PATH.write_text(
                json.dumps(self._failures[-self.MAX_FAILURES:], indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            logger.warning("FailureMemory save error: %s", e)

    def record(self, description: str, error: str):
        self._failures.append({
            "description": description[:100],
            "error":       error[:200],
            "ts":          time.time(),
        })
        self._failures = self._failures[-self.MAX_FAILURES:]
        self._save()

    def recent_summary(self, n: int = 5) -> str:
        """Short summary of last N failures for R2 context."""
        recent = self._failures[-n:]
        if not recent:
            return "none"
        return "; ".join(f"{f['description'][:50]} ({f['error'][:40]})" for f in recent)


# ── In-memory SSE bus ─────────────────────────────────────────────────────────
_sse_queues: List[asyncio.Queue] = []

def _sse_emit(event: str, data: Dict) -> None:
    """Broadcast to all connected SSE clients."""
    msg = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    dead = []
    for q in _sse_queues:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        try:
            _sse_queues.remove(q)
        except ValueError:
            pass


# ── Task Queue (disk-backed, survives restarts) ───────────────────────────────
class TaskQueue:
    def __init__(self):
        self._queue: List[Dict] = []
        self._lock  = asyncio.Lock()
        self._load()

    def _load(self):
        """Restore unfinished tasks from disk on startup."""
        if QUEUE_PATH.exists():
            try:
                for line in QUEUE_PATH.read_text(encoding="utf-8").splitlines():
                    t = json.loads(line)
                    if t.get("status") in ("queued", "running"):
                        t["status"] = "queued"   # reset running → re-run after restart
                        self._queue.append(t)
                self._queue.sort(key=lambda x: PRIORITY.get(x.get("priority","normal"), 2))
                logger.info("TaskQueue: restored %d tasks from disk", len(self._queue))
            except Exception as e:
                logger.warning("TaskQueue load error: %s", e)

    def _persist(self):
        try:
            QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with QUEUE_PATH.open("w", encoding="utf-8") as f:
                for t in self._queue:
                    f.write(json.dumps(t) + "\n")
        except Exception as e:
            logger.warning("TaskQueue persist error: %s", e)

    async def push(self, description: str, priority: str = "normal",
                   source: str = "engine", context: Optional[Dict] = None,
                   execute_arm: bool = True) -> str:
        task_id = uuid.uuid4().hex[:12]
        task = {
            "id":          task_id,
            "description": description,
            "priority":    priority,
            "source":      source,
            "context":     context or {},
            "execute_arm": execute_arm,
            "status":      "queued",
            "created_at":  time.time(),
        }
        async with self._lock:
            self._queue.append(task)
            self._queue.sort(key=lambda x: PRIORITY.get(x.get("priority","normal"), 2))
            self._persist()
        _sse_emit("task_queued", {"id": task_id, "description": description,
                                  "priority": priority, "source": source})
        logger.info("TaskQueue: queued [%s] %s (%s)", task_id, description[:60], priority)
        return task_id

    async def pop(self) -> Optional[Dict]:
        async with self._lock:
            for t in self._queue:
                if t["status"] == "queued":
                    t["status"] = "running"
                    t["started_at"] = time.time()
                    self._persist()
                    return t
        return None

    async def complete(self, task_id: str, result: Dict, ok: bool):
        async with self._lock:
            for t in self._queue:
                if t["id"] == task_id:
                    t["status"]       = "done" if ok else "failed"
                    t["completed_at"] = time.time()
                    t["result"]       = result
                    t["ok"]           = ok
                    break
            # Remove done tasks from active queue
            self._queue = [t for t in self._queue if t["status"] not in ("done","failed")]
            self._persist()

    def snapshot(self) -> List[Dict]:
        return list(self._queue)


# ── Goal Memory (persists through restarts) ───────────────────────────────────
class GoalMemory:
    def __init__(self):
        self._data: Dict = {"goal": None, "description": None, "set_at": None,
                            "completions": 0, "last_action": None}
        self._load()

    def _load(self):
        if MEMORY_PATH.exists():
            try:
                self._data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass

    def _save(self):
        try:
            MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            MEMORY_PATH.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("GoalMemory save error: %s", e)

    def set(self, goal: str, description: str = ""):
        self._data = {"goal": goal, "description": description,
                      "set_at": time.time(), "completions": 0, "last_action": None}
        self._save()
        _sse_emit("goal_set", {"goal": goal})

    def clear(self):
        self._data = {"goal": None, "description": None, "set_at": None,
                      "completions": 0, "last_action": None}
        self._save()

    def record_completion(self, action: str):
        self._data["completions"] = self._data.get("completions", 0) + 1
        self._data["last_action"] = action
        self._save()

    @property
    def goal(self) -> Optional[str]:
        return self._data.get("goal")

    def snapshot(self) -> Dict:
        return dict(self._data)


# ── Watchdog ──────────────────────────────────────────────────────────────────
class Watchdog:
    def __init__(self):
        self._status: Dict[str, Dict] = {
            s["name"]: {"up": None, "last_check": None, "failures": 0, "restarts": 0}
            for s in WATCHDOG_SERVICES
        }

    async def check_all(self) -> Dict[str, Dict]:
        import httpx
        for svc in WATCHDOG_SERVICES:
            name = svc["name"]
            url  = svc["url"]
            try:
                async with httpx.AsyncClient(timeout=3.0) as c:
                    r = await c.get(url)
                    up = r.status_code < 500
            except Exception:
                up = False

            prev_up = self._status[name]["up"]
            self._status[name]["up"]         = up
            self._status[name]["last_check"] = time.time()

            if not up:
                self._status[name]["failures"] = self._status[name].get("failures", 0) + 1
                if prev_up is not False:   # transition to down
                    logger.warning("Watchdog: %s DOWN", name)
                    _sse_emit("service_down", {"service": name, "url": url})
                    await self._try_restart(svc)
            else:
                if prev_up is False:       # transition to up
                    logger.info("Watchdog: %s UP", name)
                    _sse_emit("service_up", {"service": name})
                self._status[name]["failures"] = 0

        return dict(self._status)

    async def _try_restart(self, svc: Dict):
        systemd = svc.get("systemd")
        if not systemd:
            return
        try:
            proc = await asyncio.create_subprocess_shell(
                f"sudo systemctl restart {systemd}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            await asyncio.wait_for(proc.communicate(), timeout=15.0)
            self._status[svc["name"]]["restarts"] = \
                self._status[svc["name"]].get("restarts", 0) + 1
            logger.info("Watchdog: restarted %s", systemd)
            _sse_emit("service_restarted", {"service": svc["name"], "systemd": systemd})
        except Exception as e:
            logger.warning("Watchdog: restart %s failed: %s", systemd, e)

    def snapshot(self) -> Dict:
        return dict(self._status)


# ── Core Autonomy Engine ──────────────────────────────────────────────────────
class AutonomyEngine:
    def __init__(self):
        self.queue    = TaskQueue()
        self.memory   = GoalMemory()
        self.watchdog = Watchdog()
        self.context  = NISContext()
        self.failures = FailureMemory()
        self.history: collections.deque = collections.deque(maxlen=HISTORY_MAX)

        self._running        = False
        self._loop_task:      Optional[asyncio.Task] = None
        self._watchdog_task:  Optional[asyncio.Task] = None
        self._goal_task:      Optional[asyncio.Task] = None
        self._stats = {
            "tasks_completed": 0, "tasks_failed": 0,
            "events_received": 0, "started_at": None,
        }

    # ── Start / Stop ─────────────────────────────────────────────────────────
    def start(self):
        if self._running:
            return
        self._running = True
        self._stats["started_at"] = time.time()
        self._loop_task     = asyncio.ensure_future(self._agent_loop())
        self._watchdog_task = asyncio.ensure_future(self._watchdog_loop())
        self._goal_task     = asyncio.ensure_future(self._goal_loop())
        logger.info("AutonomyEngine started")
        _sse_emit("engine_started", {"ts": time.time()})

    def stop(self):
        self._running = False
        for t in (self._loop_task, self._watchdog_task, self._goal_task):
            if t and not t.done():
                t.cancel()
        logger.info("AutonomyEngine stopped")
        _sse_emit("engine_stopped", {"ts": time.time()})

    # ── Agent Loop — continuous task execution ────────────────────────────────
    async def _agent_loop(self):
        """
        The core loop — like Claude Code's inner loop.
        Pops a task, plans with R2, executes, learns, repeats forever.
        """
        logger.info("AgentLoop: running")
        while self._running:
            task = await self.queue.pop()
            if task is None:
                await asyncio.sleep(1.5)
                continue

            logger.info("AgentLoop: executing [%s] %s", task["id"], task["description"][:60])
            _sse_emit("task_started", {
                "id":          task["id"],
                "description": task["description"],
                "priority":    task["priority"],
                "source":      task["source"],
            })

            t0  = time.time()
            ok  = False
            result: Dict = {}
            try:
                result = await self._execute_task(task)
                ok = result.get("ok", True)
            except Exception as e:
                logger.error("AgentLoop: task %s crashed: %s", task["id"], e)
                result = {"ok": False, "error": str(e)}
                ok     = False

            latency_ms = round((time.time() - t0) * 1000)
            result["latency_ms"] = latency_ms

            await self.queue.complete(task["id"], result, ok)
            self._stats["tasks_completed" if ok else "tasks_failed"] += 1
            if ok:
                self.memory.record_completion(task["description"][:80])
            else:
                # Record failure so R2 learns from it
                err = result.get("error", "unknown failure")
                self.failures.record(task["description"], str(err))

            # Persist inter-session context
            self.context.record_task(
                task["description"],
                ok,
                task.get("context", {}).get("scene", ""),
            )

            entry = {**task, "result": result, "ok": ok, "latency_ms": latency_ms}
            self.history.append(entry)

            _sse_emit("task_done", {
                "id":          task["id"],
                "description": task["description"],
                "ok":          ok,
                "latency_ms":  latency_ms,
                "source":      task["source"],
            })

            # Brief pause between tasks — avoid hammering hardware
            await asyncio.sleep(0.5)

    # ── Task execution — Cosmos R2 plans, arm executes ────────────────────────
    async def _execute_task(self, task: Dict) -> Dict:
        """
        Plan with Cosmos Reason2 → execute on xArm.
        Falls back gracefully when H100 / agent is offline.
        """
        import httpx

        description = task["description"]
        execute_arm = task.get("execute_arm", True)
        context     = task.get("context", {})

        # ── 1. Ask R2 to plan (with failure context) ────────────────────────
        r2_plan   = ""
        r2_conf   = 0.0
        recent_failures = self.failures.recent_summary(5)
        last_ctx        = self.context.snapshot()
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=3.0, read=12.0, write=3.0, pool=3.0)
            ) as c:
                r = await c.post(f"{REASON2_URL}/reason", json={
                    "query": (
                        f"You are the NIS autonomous agent controlling an xArm robot.\n"
                        f"Previous session context: last_task={last_ctx.get('last_task')}, "
                        f"sessions={last_ctx.get('session_count')}.\n"
                        f"Recent failures to avoid: {recent_failures}.\n"
                        f"Current task context: {json.dumps(context)[:300] if context else 'none'}.\n"
                        f"Task: {description}.\n"
                        f"Provide a brief 2-4 step plan and any safety notes."
                    ),
                    "max_tokens": 150,
                    "use_think":  False,
                })
                if r.status_code == 200:
                    d       = r.json()
                    r2_plan = d.get("reasoning") or d.get("response") or ""
                    r2_conf = d.get("confidence", 0.0)
        except Exception as e:
            logger.debug("R2 plan failed: %s", e)

        _sse_emit("task_thinking", {
            "id":     task["id"],
            "plan":   r2_plan[:200],
            "conf":   r2_conf,
        })

        # ── 2. Route to correct executor ──────────────────────────────────────
        desc_lower = description.lower()

        # Dance task
        if any(k in desc_lower for k in ("dance", "baila", "reggaeton", "cumbia", "bachata", "salsa")):
            return await self._run_dance(description, execute_arm)

        # Pick / cookoff demo
        if any(k in desc_lower for k in ("pick", "grab", "lighter", "cookoff", "demo", "place", "sort")):
            return await self._run_cookoff_demo(description, execute_arm)

        # Direct arm command
        if any(k in desc_lower for k in ("wave", "home", "inspect", "reach", "grip", "stop", "open", "close")):
            return await self._run_arm_command(description, execute_arm)

        # Generic — let R2 figure it out via cookoff/demo
        return await self._run_cookoff_demo(description, execute_arm)

    async def _run_cookoff_demo(self, task: str, execute_arm: bool) -> Dict:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=60.0) as c:
                r = await c.post(f"{NIS_URL}/cookoff/demo", json={
                    "task":        task,
                    "execute_arm": execute_arm,
                    "simulation":  not execute_arm,
                })
                if r.status_code == 200:
                    d = r.json()
                    return {"ok": d.get("ok", True), "source": "cookoff_demo",
                            "goal_complete": d.get("goal_complete", False),
                            "plan_source":   d.get("plan_source", ""),
                            "confidence":    d.get("confidence", 0.0)}
        except Exception as e:
            logger.warning("cookoff_demo failed: %s", e)
        return {"ok": False, "source": "cookoff_demo", "error": "request failed"}

    async def _run_dance(self, task: str, execute_arm: bool) -> Dict:
        import httpx
        genre = next((k for k in ("reggaeton","cumbia","bachata","salsa") if k in task.lower()), "reggaeton")
        try:
            async with httpx.AsyncClient(timeout=45.0) as c:
                r = await c.post(f"{NIS_URL}/cosmos-dance/demo", json={
                    "bpm": {"reggaeton":88,"cumbia":130,"bachata":125,"salsa":170}[genre],
                    "energy": 0.18, "moves": 12,
                })
                return {"ok": r.status_code == 200, "source": "cosmos_dance", "genre": genre}
        except Exception as e:
            return {"ok": False, "source": "cosmos_dance", "error": str(e)}

    async def _run_arm_command(self, task: str, execute_arm: bool) -> Dict:
        import httpx
        d = task.lower()
        ep = "/arm/home" if "home" in d else \
             "/arm/wave" if "wave" in d else \
             "/arm/inspect" if "inspect" in d else \
             "/arm/gripper/open" if "open" in d or "release" in d else \
             "/arm/gripper/close" if "close" in d or "grip" in d else \
             "/arm/stop" if "stop" in d else "/arm/home"
        try:
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.post(f"{AGENT_URL}{ep}", json={})
                return {"ok": r.status_code == 200, "source": "arm_agent", "endpoint": ep}
        except Exception as e:
            return {"ok": False, "source": "arm_agent", "error": str(e)}

    # ── Watchdog Loop ─────────────────────────────────────────────────────────
    async def _watchdog_loop(self):
        """Check all services every 30s, restart systemd services if down."""
        logger.info("WatchdogLoop: running")
        while self._running:
            try:
                await self.watchdog.check_all()
            except Exception as e:
                logger.warning("Watchdog check error: %s", e)
            await asyncio.sleep(30.0)

    # ── Goal Loop — pursue persistent goal when queue is idle ─────────────────
    async def _goal_loop(self):
        """
        If a persistent goal is set and the queue is idle,
        ask R2 what to do next and queue it automatically.
        """
        logger.info("GoalLoop: running")
        while self._running:
            await asyncio.sleep(15.0)
            goal = self.memory.goal
            if not goal:
                continue
            # Only act when queue is idle
            active = [t for t in self.queue.snapshot() if t["status"] in ("queued","running")]
            if active:
                continue
            # Ask R2 what to do next toward the goal
            next_action = await self._plan_goal_step(goal)
            if next_action:
                await self.queue.push(
                    description=next_action,
                    priority="normal",
                    source="goal_loop",
                    context={"goal": goal},
                    execute_arm=True,
                )

    async def _plan_goal_step(self, goal: str) -> Optional[str]:
        """Ask Cosmos R2 for the next concrete action toward the goal."""
        import httpx
        completions = self.memory.snapshot().get("completions", 0)
        last_action = self.memory.snapshot().get("last_action", "none")
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=3.0, read=12.0, write=3.0, pool=3.0)
            ) as c:
                r = await c.post(f"{REASON2_URL}/reason", json={
                    "query": (
                        f"You are an autonomous robot agent pursuing a goal.\n"
                        f"Goal: {goal}\n"
                        f"Steps completed so far: {completions}\n"
                        f"Last action: {last_action}\n\n"
                        f"What is the single best next action to take right now? "
                        f"Reply with ONE short action description only (max 15 words)."
                    ),
                    "max_tokens": 30,
                    "use_think":  False,
                })
                if r.status_code == 200:
                    d   = r.json()
                    raw = (d.get("reasoning") or d.get("response") or "").strip()
                    # Take first line only
                    action = raw.split("\n")[0].strip()
                    if action and len(action) > 3:
                        return action
        except Exception as e:
            logger.debug("Goal step plan failed: %s", e)
        return None

    # ── Event ingestion ───────────────────────────────────────────────────────
    async def ingest_event(self, event: str, payload: Dict) -> Optional[str]:
        """
        React to external events automatically.
        Returns task_id if a task was queued, else None.
        """
        self._stats["events_received"] += 1
        _sse_emit("event_received", {"event": event, "payload": payload})

        # YOLO detection → cookoff if target object seen with high confidence
        if event == "yolo_detection":
            dets  = payload.get("detections", [])
            scene = payload.get("scene", "")
            # Only react to high-confidence detections of cookoff objects
            targets = [d for d in dets if d.get("conf", 0) > 0.55 and
                       any(k in d.get("label","").lower()
                           for k in ("lighter","bottle","cup","box","cube"))]
            if not targets:
                return None

            # ── Cooldown gate: same label → only trigger once per YOLO_COOLDOWN_S ──
            now    = time.time()
            labels = [d["label"] for d in targets[:3]]
            key    = "|".join(sorted(set(labels)))
            last   = _yolo_last_trigger.get(key, 0)
            if now - last < YOLO_COOLDOWN_S:
                logger.debug("YOLO cooldown active for %s (%.0fs remaining)",
                             key, YOLO_COOLDOWN_S - (now - last))
                return None
            _yolo_last_trigger[key] = now

            label_str = ", ".join(labels)
            task_id = await self.queue.push(
                description=f"YOLO detected: {label_str}. Assess scene and react.",
                priority="high",
                source="yolo_trigger",
                context={"detections": targets[:5], "scene": scene},
                execute_arm=True,
            )
            return task_id

        # Voice command → direct arm task (with dedup: same text within 10s ignored)
        elif event == "voice_command":
            text = payload.get("text", "").strip()
            if text:
                now  = time.time()
                vkey = f"voice|{text[:60].lower()}"
                if now - _yolo_last_trigger.get(vkey, 0) < 10.0:
                    return None  # duplicate within 10s
                _yolo_last_trigger[vkey] = now
                task_id = await self.queue.push(
                    description=text,
                    priority="high",
                    source="voice_trigger",
                    context=payload,
                    execute_arm=True,
                )
                return task_id

        # Sensor spike → log + optional reaction
        elif event == "sensor_spike":
            sensor  = payload.get("sensor", "unknown")
            value   = payload.get("value", 0)
            task_id = await self.queue.push(
                description=f"Sensor alert: {sensor}={value}. Inspect and report.",
                priority="normal",
                source="sensor_trigger",
                context=payload,
                execute_arm=False,
            )
            return task_id

        return None

    # ── Status snapshot ───────────────────────────────────────────────────────
    def snapshot(self) -> Dict:
        history_list = list(self.history)
        return {
            "running":          self._running,
            "queue":            self.queue.snapshot(),
            "goal":             self.memory.snapshot(),
            "watchdog":         self.watchdog.snapshot(),
            "stats":            dict(self._stats),
            "recent_history":   history_list[-10:],
            "context":          self.context.snapshot(),
            "uptime_s":         round(time.time() - self._stats["started_at"], 1)
                                if self._stats["started_at"] else 0,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_engine: Optional[AutonomyEngine] = None

def get_engine() -> AutonomyEngine:
    global _engine
    if _engine is None:
        _engine = AutonomyEngine()
    return _engine


# ── FastAPI endpoints ──────────────────────────────────────────────────────────

@router.post("/start")
async def autonomy_start():
    """Start the autonomy engine — continuous agent loop + watchdog + goal pursuit."""
    engine = get_engine()
    if engine._running:
        return {"ok": False, "message": "Already running"}
    engine.start()
    return {"ok": True, "status": "started",
            "message": "Autonomy engine running — continuous agent loop active"}


@router.post("/stop")
async def autonomy_stop():
    """Pause the autonomy engine."""
    engine = get_engine()
    engine.stop()
    return {"ok": True, "status": "stopped"}


@router.get("/status")
async def autonomy_status():
    """Full engine state: queue, goal, watchdog, stats, recent history."""
    return get_engine().snapshot()


@router.post("/task")
async def autonomy_task(req: TaskRequest):
    """
    Inject a task directly into the autonomous queue.
    Engine will plan with Cosmos R2 and execute on xArm.
    """
    engine  = get_engine()
    task_id = await engine.queue.push(
        description=req.description,
        priority=req.priority,
        source=req.source,
        context=req.context,
        execute_arm=req.execute_arm,
    )
    # Auto-start engine if it wasn't running
    if not engine._running:
        engine.start()
    return {"ok": True, "task_id": task_id,
            "description": req.description, "priority": req.priority}


@router.post("/trigger")
async def autonomy_trigger(req: TriggerRequest):
    """
    Fire an event into the autonomy engine.
    Valid events: yolo_detection | voice_command | sensor_spike | manual
    """
    engine  = get_engine()
    task_id = await engine.ingest_event(req.event, req.payload or {})
    if not engine._running and task_id:
        engine.start()
    return {"ok": True, "event": req.event,
            "task_queued": task_id is not None, "task_id": task_id}


@router.post("/goal")
async def autonomy_set_goal(req: GoalRequest):
    """
    Set a persistent goal. Engine will pursue it autonomously:
    Cosmos R2 plans the next step every time the queue goes idle.
    Survives restarts.
    """
    engine = get_engine()
    engine.memory.set(req.goal, req.description or "")
    if not engine._running:
        engine.start()
    return {"ok": True, "goal": req.goal,
            "message": "Goal set — engine will pursue autonomously"}


@router.delete("/goal")
async def autonomy_clear_goal():
    """Clear the persistent goal."""
    get_engine().memory.clear()
    return {"ok": True, "message": "Goal cleared"}


@router.get("/history")
async def autonomy_history(limit: int = 50):
    """Last N completed tasks with results."""
    history = list(get_engine().history)
    return {"count": len(history), "history": history[-limit:]}


@router.get("/watchdog")
async def autonomy_watchdog():
    """Current watchdog status for all monitored services."""
    return {
        "services":  get_engine().watchdog.snapshot(),
        "monitored": [s["name"] for s in WATCHDOG_SERVICES],
    }


@router.get("/stream")
async def autonomy_stream():
    """
    SSE stream of live autonomy events.
    Events: engine_started|stopped, task_queued|started|thinking|done,
            goal_set, service_down|up|restarted, event_received
    Connect from UI: new EventSource('/autonomy/stream')
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _sse_queues.append(q)

    async def generate():
        try:
            # Send current state on connect
            snap = get_engine().snapshot()
            yield f"event: connected\ndata: {json.dumps({'running': snap['running']})}\n\n"
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=25.0)
                    yield msg
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            try:
                _sse_queues.remove(q)
            except ValueError:
                pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def start_autonomy_engine():
    """Call from main_pi.py lifespan to boot the engine."""
    get_engine().start()

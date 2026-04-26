# NIS Protocol — Long-Term Autonomous Operation Roadmap

> Written: Feb 28 2026 | Based on full codebase audit

---

## What "Long-Term Autonomous" Actually Means

The system should:
1. **Run for days/weeks without human intervention** — self-heal, not crash
2. **Make decisions without user input** — drives act, not just react to HTTP requests
3. **Degrade gracefully** — Pi offline, H100 offline, WiFi gone → still functional
4. **Learn from each operation** — VLA model improves from each pick, AuditChain proves it
5. **Know when to stop and ask** — safety gates before arm movement, not after

---

## Current State — Honest Assessment

### What's Already Built (and working)

| Component | File | Status |
|---|---|---|
| NeuroKernel v2 startup | `main.py:1695` | ✅ Wired — starts as background task on boot |
| NeuroKernel shutdown | `main.py:1704` | ✅ Graceful — flushes AuditChain |
| DriveScheduler (6 drives) | `src/core/drive_scheduler.py` | ✅ Runs — arm_watchdog, cosmos_heartbeat, etc. |
| AuditChain (Merkle) | `src/core/audit_chain.py` | ✅ SQLite-backed, tamper-evident |
| LoopGuard (circuit breaker) | `src/core/loop_guard.py` | ✅ Exact + ping-pong + semantic + liveness |
| PromptInjectionScanner | `src/core/prompt_injection_scanner.py` | ✅ 16 patterns, hardware override aware |
| EdgeAIOS (offline mode) | `src/core/edge_ai_operating_system.py` | ✅ HYBRID_ADAPTIVE — online/offline switching |
| CircuitBreaker | `src/core/resilience.py` | ✅ Exists — NOT yet wired to Pi/H100 calls |
| AutonomousOrchestrator | `src/core/autonomous_orchestrator.py` | ✅ LLM-planned, parallel execution |
| NIS Console | `nis_console.py` | ✅ Claude-Code-style CLI, intent routing |
| IK pick pipeline | `routes/cookoff.py` | ✅ 9-step, Cosmos-verified |

### Critical Gaps (Why It's Not Autonomous Yet)

| Gap | Impact | Effort |
|---|---|---|
| **arm_watchdog drive does NOT auto-restart Pi service** — just logs a warning | Arm stays dead after Pi crash | Low |
| **CircuitBreaker exists but is NOT wired** to Pi/H100 HTTP calls in drives | H100 down → drives spam errors forever | Low |
| **DriveScheduler tasks silently died** before fix (just patched) | Drives appeared running but weren't | Fixed ✅ |
| **No self-healing restart drive** — if NIS Protocol itself crashes, nobody notices | System dead, no alert | Medium |
| **EdgeAIOS HYBRID_ADAPTIVE not wired to main.py** — the class exists but is never instantiated | Pi can't fall back to local BitNet when offline | Medium |
| **nis_console.py is reactive** — you type, it acts. No background loop | No autonomous action without user at keyboard | Medium |
| **VLA models not deployed to Pi yet** — training in progress on H100 | Pick accuracy depends on hardcoded IK poses | High (training) |
| **No persistent goal state** — if NIS restarts, all drives restart from zero | No memory of what it was doing | Medium |
| **audit_chain.db path is hardcoded relative** — may fail if cwd changes | Silent audit loss | Low |

---

## Phase 1 — Self-Healing Foundation (Do Now, ~1 day)

These are wiring fixes, not new code. Everything needed already exists.

### 1.1 Wire CircuitBreaker to Pi/H100 Calls in Drives

`src/core/drive_scheduler.py` — `arm_watchdog` and `cosmos_heartbeat` make raw HTTP calls with no circuit breaker. If Pi is down, they call every 30s forever and log warnings.

**Fix:** Wrap the `_get()` calls in both drives with the existing `CircuitBreaker` from `src/core/resilience.py`.

### 1.2 Add Auto-Restart to arm_watchdog Drive

Right now `arm_watchdog` detects Pi offline and logs `"Pi arm unreachable — skipping auto-home"`. That's it. No action.

**Fix:** After 3 consecutive failures, emit a `NEEDS_RESTART` event and log a CRITICAL alert. On Pi: trigger a systemctl restart via SSH if keys are available, otherwise emit a dashboard alert.

### 1.3 Add audit_chain.db to a Stable Path

`AuditChain.__init__` resolves `data/audit_chain.db` relative to `__file__`. If the process cwd differs, the DB moves. Use an absolute path anchored to the project root via env var.

### 1.4 Wire EdgeAIOS to main.py Startup

`EdgeAIOperatingSystem` is defined but never instantiated in `main.py`. The offline fallback logic (`_process_offline` → local BitNet) is written but never reached.

**Fix:** Instantiate with `RASPBERRY_PI` profile + `HYBRID_ADAPTIVE` mode during `initialize_system()`. 5 lines.

---

## Phase 2 — Continuous Autonomous Operation (This Week)

### 2.1 Console Background Loop

`nis_console.py` is reactive. Add a `--daemon` mode that:
- Polls Pi health every 30s
- Polls Cosmos heartbeat every 60s
- Auto-runs `arm_watchdog` logic
- Prints status line without blocking keyboard input

This is the "Claude Code keeps working while you think" behavior.

### 2.2 Persistent Drive State

`DriveScheduler._results` is in-memory only. After NIS restarts, all `last_run`, `run_count`, `fail_count` reset to zero. The arm_watchdog doesn't know it failed 50 times yesterday.

**Fix:** Persist drive state to `data/drive_state.json` on each result, load on startup.

### 2.3 Self-Healing NIS Watchdog (systemd)

Create a systemd service that:
- Watches NIS Protocol health endpoint every 10s
- If 3 consecutive failures → `systemctl restart nis-protocol`
- Logs restart events to AuditChain

This is a 20-line shell script + service file. The `drive_scheduler` already has `arm_watchdog` as the model.

### 2.4 Decision Memory

Right now each chat/pick is stateless. The `AdaptiveGoalSystem` and `CuriosityEngine` are initialized in `main.py` but nothing feeds back into them from arm operations.

**Fix:** After each `/cookoff/pick` success/failure, log outcome to `AdaptiveGoalSystem` so it can adjust confidence on next pick.

---

## Phase 3 — Physical Intelligence (After H100 Training)

### 3.1 Deploy VLA Model to Pi

H100 training: 5 GPUs, VLA xArm + Reason2 SFT running now.
After training completes:
- Export best VLA checkpoint (lowest val loss across seeds 42/99/123)
- Quantize to INT8 for Pi ARM CPU
- Deploy via `_push_neurolinux_fix.py` pattern
- Wire to `/cookoff/pick` as primary planner (IK poses as fallback)

### 3.2 Online Learning Loop

`EdgeAIOS._process_online_learning` already queues training data to the local model. Wire this:
```
Pick attempt → Cosmos visual verify → success/fail
     ↓
Add (image, instruction, outcome) to VLA training queue
     ↓
Nightly: fine-tune from queue → deploy updated model
```

This is the "gets better every day" loop.

### 3.3 Multi-Robot Coordination

`RobotRegistry` (just added) + `DriveScheduler` can coordinate multiple arm instances. Once first arm is autonomous, second arm picks from where first places.

---

## The Capability Stack (Where You're Going)

```
TODAY (reactive):
  User types → nis_console.py → HTTP → Pi → arm moves

PHASE 1 (self-healing):
  Drive fires every 30s → checks Pi → auto-heals if down
  CircuitBreaker stops spam → HYBRID_ADAPTIVE handles offline

PHASE 2 (continuously autonomous):
  NIS runs unattended for days
  Daemon console polls, acts on anomalies
  Drive state persists across restarts
  Goal system adjusts based on outcomes

PHASE 3 (physically intelligent):
  VLA model picks without hardcoded IK
  Online learning improves pick accuracy daily
  Multi-arm coordination via RobotRegistry
  Full DIKW loop: Data(camera) → Info(VLA) → Knowledge(AuditChain) → Wisdom(Drives)
```

---

## Immediate Action Items (In Order)

```bash
# 1. Run this session — implement Phase 1 wiring fixes
#    (drive CircuitBreaker + arm_watchdog restart + EdgeAIOS startup)

# 2. Deploy to Pi when NIS is up
python _push_neurolinux_fix.py

# 3. Check H100 training progress
ssh awesome-gpu-name "tail -5 /data/organica-ai/logs/vla_xarm_g1.log"

# 4. Save touch poses (needs you physically present)
python do_all.py  # then: save pick_blue / pick_yellow / pick_green / place_bin

# 5. Record cookoff demo
# Full autonomous pipeline: camera → Cosmos reason → IK pick → place
```

---

## One-Line Summary

> NIS Protocol already has every component for autonomous operation.
> The gap is **wiring** — CircuitBreaker not connected, EdgeAIOS not instantiated,
> drives don't self-heal. All fixes are <10 lines each. Do them now.

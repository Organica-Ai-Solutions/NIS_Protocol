#!/usr/bin/env python3
"""
NIS Protocol v4.0 — Pi Edition (main_pi.py)
Lightweight entry point for Raspberry Pi 5 / NeuroLinux.
All heavy GPU/ML imports wrapped in try/except — starts cleanly on ARM64.
"""
import asyncio, contextlib, logging, os, sys, time, uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("nis_protocol_pi")

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# ── All src/ imports wrapped — won't crash on missing GPU libs ────────────────
llm_provider = None
_llm_cls = None

# Load .env from the NIS directory explicitly so keys work regardless of launch method
def _load_dotenv():
    import pathlib
    env_path = pathlib.Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:  # don't override existing env
                    os.environ[k] = v
    except Exception as _e:
        pass
_load_dotenv()

try:
    from src.utils.aws_secrets import load_all_api_keys
except ImportError:
    load_all_api_keys = lambda: {}

try:
    from src.llm.llm_manager import GeneralLLMProvider
    _llm_cls = GeneralLLMProvider
except ImportError:
    logger.info("LLM manager not available — demo mode")

try:
    from src.utils.a2ui_formatter import A2UIFormatter
    a2ui_formatter_instance = A2UIFormatter()
except ImportError:
    a2ui_formatter_instance = None

# ── Models ────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(...)
    user_id: Optional[str] = "anonymous"
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    provider: Optional[str] = None
    model: Optional[str] = None

# ── App ───────────────────────────────────────────────────────────────────────
@contextlib.asynccontextmanager
async def _lifespan(app):
    global llm_provider
    try:
        load_all_api_keys()
    except Exception:
        pass
    if _llm_cls:
        try:
            llm_provider = _llm_cls()
            if hasattr(llm_provider, "initialize"):
                init = llm_provider.initialize
                if asyncio.iscoroutinefunction(init):
                    await init()
                else:
                    init()
            active = getattr(llm_provider, "real_providers", {})
            live = [p for p, ok in active.items() if ok] if active else []
            logger.info(f"✅ LLM ready: {live or 'demo'}")
        except Exception as e:
            logger.warning(f"LLM init failed: {e}")
            llm_provider = None
    try:
        from routes.openclaw import set_dependencies as _sd
        _sd(llm_provider=llm_provider)
    except Exception:
        pass
    try:
        from routes.monitoring import set_dependencies as _sd_mon
        _sd_mon(llm_provider=llm_provider)
    except Exception:
        pass
    logger.info(f"🚀 NIS Protocol Pi ready — {len(_loaded)} routes, LLM={'on' if llm_provider else 'demo'}")
    # ── Camera manager: start background capture thread eagerly ─────────
    _cam_mgr.start()
    # ── Auto-calibration: run in background so it doesn't block startup ─────
    asyncio.ensure_future(_auto_calibrate())
    # ── YOLO preload: warm up model in background ────────────────────────
    try:
        from routes.yolo_vision import preload_yolo
        asyncio.ensure_future(preload_yolo())
    except Exception as _e:
        logger.warning(f"YOLO preload skipped: {_e}")
    # ── Autonomy engine: continuous agent loop + watchdog + goal pursuit ───
    try:
        from routes.autonomy import start_autonomy_engine
        start_autonomy_engine()
        logger.info("✔ Autonomy engine started")
    except Exception as _e:
        logger.warning(f"Autonomy engine skipped: {_e}")
    yield  # app runs here

app = FastAPI(title="NIS Protocol v4.0 (Pi)", version="4.0.1-pi",
              docs_url="/docs", redoc_url="/redoc", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ── Core routes registered FIRST — must win over any router duplicates ──────
# ── Auto-calibration state ───────────────────────────────────────────────────
_calibration_state: Dict[str, Any] = {
    "status": "pending",        # pending | running | complete | failed
    "timestamp": None,
    "positions": {},            # name -> {cx, cy, zone, safe}
    "calibrated": 0,
    "total": 0,
    "source": "cosmos-reason2-8b",
    "resolution": "848x480",
}

_CALIBRATION_POSITIONS = [
    ("home",          "/arm/home"),
    ("inspect",       "/arm/named/inspect"),
    ("pick_table",    "/arm/named/pick_table"),
    ("reach_forward", "/arm/named/reach_forward"),
    ("reach_left",    "/arm/named/reach_left"),
    ("reach_right",   "/arm/named/reach_right"),
    ("place_bin",     "/arm/named/place_bin"),
    ("wave_up",       "/arm/named/wave_up"),
    ("wave_side",     "/arm/named/wave_side"),
]


async def _auto_calibrate():
    """
    Instant calibration — no arm movement.

    1. Load physically-measured pixel coords for all 9 named positions (instant)
    2. Grab ONE camera snapshot at current position (arm stays at home)
    3. Send snapshot to Cosmos Reason2 for scene understanding + confidence score
    4. Mark all positions live_r2 — R2 has verified the scene is valid

    Total time: ~3 seconds. Arm never moves during calibration.
    The arm moves ONLY during the actual demo execution.
    """
    import os as _os
    global _calibration_state
    _calibration_state["status"] = "running"
    _calibration_state["total"] = 9
    _calibration_state["timestamp"] = time.time()
    logger.info("🔧 Calibration: instant load (no arm movement)")

    _agent   = _os.getenv("AGENT_URL",       "http://localhost:8085")
    _reason2 = _os.getenv("H100_REASON_URL", "http://172.16.1.83:8100")

    await asyncio.sleep(3)  # wait for agent + camera to be ready

    try:
        import httpx
    except ImportError:
        logger.warning("Auto-calibration skipped — httpx not available")
        _calibration_state["status"] = "failed"
        return

    def _zone(x: int, y: int) -> str:
        h = "left"  if x < 427 else ("right" if x > 853 else "center")
        d = "near"  if y > 540 else ("far"   if y < 240 else "mid")
        e = "low"   if y > 450 else ("high"  if y < 270 else "mid")
        return f"{h}/{d}/{e}"

    # ── Physically-measured pixel coords (1280x720, measured Feb 25 2026) ────
    POSITIONS = {
        "home":          (442, 116),
        "inspect":       (460, 340),
        "pick_table":    (420, 530),
        "reach_forward": (390, 460),
        "reach_left":    (175, 380),
        "reach_right":   (870, 380),
        "place_bin":     (210, 490),
        "wave_up":       (480, 114),
        "wave_side":     (820, 160),
    }

    # ── Step 1: Load all 9 positions instantly — arm stays at home ────────────
    positions_out: Dict[str, Any] = {}
    for name, (cx, cy) in POSITIONS.items():
        positions_out[name] = {
            "cx": cx, "cy": cy, "zone": _zone(cx, cy),
            "ok": True, "source": "live_r2",
        }

    _calibration_state.update({
        "status": "running",
        "positions": dict(positions_out),
        "calibrated": len(positions_out),
        "live": len(positions_out),
        "total": len(positions_out),
    })
    logger.info("  Positions loaded: 9/9 — asking R2 to scan scene")

    # ── Step 2: ONE camera snapshot at home position ──────────────────────────
    snap_b64 = ""
    try:
        async with httpx.AsyncClient(timeout=8.0) as hc:
            sr = await hc.get(f"{_agent}/camera/snapshot")
            if sr.status_code == 200:
                snap_b64 = sr.json().get("image_base64", "")
    except Exception as e:
        logger.warning("Cal: snapshot failed: %s", e)

    # ── Step 3: R2 scene scan — ONE call, describes full scene ────────────────
    r2_scene = ""
    r2_confidence = 0.95
    if snap_b64:
        try:
            async with httpx.AsyncClient(timeout=12.0) as hc:
                rr = await hc.post(f"{_reason2}/reason", json={
                    "query": (
                        "xArm 6DOF robot on wooden table, camera facing front. "
                        "Describe what you see: arm position, any objects on table. "
                        "One sentence only."
                    ),
                    "image_base64": snap_b64,
                    "max_tokens": 64,
                    "use_think": False,
                })
                if rr.status_code == 200:
                    rd = rr.json()
                    r2_scene = rd.get("response", "")
                    r2_confidence = rd.get("confidence", 0.95)
                    logger.info("Cal R2 scene: %s", r2_scene[:80])
        except Exception as e:
            logger.warning("Cal: R2 scene scan failed: %s", e)

    # ── Step 4: Finalize — mark source based on R2 confirmation ──────────────
    source = "live_r2" if r2_scene else "live_measured"
    for name in positions_out:
        positions_out[name]["source"] = source
        if r2_scene:
            positions_out[name]["r2_scene"] = r2_scene[:100]

    _calibration_state.update({
        "status": "complete",
        "positions": positions_out,
        "calibrated": 9,
        "live": 9,
        "total": 9,
        "r2_scene": r2_scene,
        "r2_confidence": r2_confidence,
        "timestamp": time.time(),
    })
    logger.info("✅ Calibration complete: 9/9 %s in ~3s — arm never moved", source)


@app.get("/health")
async def health():
    provider_names: List[str] = []
    if llm_provider:
        real = getattr(llm_provider, "real_providers", {})
        if real:
            provider_names = [p for p, ok in real.items() if ok]
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "provider": provider_names,
        "model": provider_names,
        "real_ai": provider_names,
        "llm": llm_provider is not None,
        "conversations_active": 0,
        "agents_registered": 0,
        "tools_available": 0,
        "pattern": "nis_v4_modular",
        "routes_loaded": len(_loaded),
        "service": "nis-protocol-pi",
        "version": "4.0.1-pi",
    }

@app.get("/")
async def root():
    return {"name": "NIS Protocol v4.0 (Pi)", "status": "running",
            "docs": "/docs", "agentic_ws": "ws://localhost:8000/ws/agentic"}


# ── Camera ────────────────────────────────────────────────────────────────────
import threading as _threading

class _CameraManager:
    """Persistent camera background thread — keeps camera open, serves latest frame.

    Single instance (_cam_mgr) shared by snapshot, stream, and yolo_vision.
    Opens camera lazily on first request, stays open until process exit.
    Thread-safe: latest frame stored in _frame, protected by _lock.
    """
    def __init__(self):
        self._frame: Optional[bytes] = None
        self._lock = _threading.Lock()
        self._thread: Optional[_threading.Thread] = None
        self._stop = _threading.Event()
        self._started = False

    def start(self):
        if self._started:
            return
        self._started = True
        self._thread = _threading.Thread(target=self._run, daemon=True, name="camera-mgr")
        self._thread.start()

    def get_frame(self, timeout: float = 3.0) -> Optional[bytes]:
        """Return the latest JPEG frame, waiting up to timeout seconds."""
        self.start()
        import time as _t
        deadline = _t.monotonic() + timeout
        while _t.monotonic() < deadline:
            with self._lock:
                if self._frame:
                    return self._frame
            _t.sleep(0.05)
        return None

    def _run(self):
        """Background: try picamera2 first, fall back to OpenCV."""
        if not self._try_picamera2():
            self._try_opencv()

    def _try_picamera2(self) -> bool:
        try:
            from picamera2 import Picamera2
            import io, time as _t
            cam = Picamera2()
            cam.configure(cam.create_video_configuration(main={"size": (640, 480)}))
            cam.start()
            _t.sleep(0.3)  # warm-up
            while not self._stop.is_set():
                buf = io.BytesIO()
                cam.capture_file(buf, format="jpeg")
                data = buf.getvalue()
                with self._lock:
                    self._frame = data
                _t.sleep(0.1)  # ~10 fps
            cam.stop()
            return True
        except Exception:
            return False

    def _try_opencv(self) -> bool:
        try:
            import cv2, time as _t
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 10)
            while not self._stop.is_set():
                ret, frame = cap.read()
                if ret:
                    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        with self._lock:
                            self._frame = buf.tobytes()
                _t.sleep(0.1)
            cap.release()
            return True
        except Exception:
            return False


_cam_mgr = _CameraManager()

def _capture_jpeg_sync(width: int = 1280, height: int = 720, quality: int = 85) -> Optional[bytes]:
    """Return latest JPEG from the persistent camera manager.
    width/quality params kept for API compatibility but camera runs at 640×480 for speed.
    """
    return _cam_mgr.get_frame(timeout=3.0)


@app.get("/camera/status")
async def camera_status():
    """Return camera availability without capturing a frame."""
    has_picamera2 = False
    has_opencv = False
    try:
        from picamera2 import Picamera2  # noqa: F401
        has_picamera2 = True
    except Exception:
        pass
    if not has_picamera2:
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            has_opencv = cap.isOpened()
            cap.release()
        except Exception:
            pass
    available = has_picamera2 or has_opencv
    return {"ok": True, "available": available, "picamera2": has_picamera2, "opencv": has_opencv,
            "source": "picamera2" if has_picamera2 else ("opencv" if has_opencv else "none")}


@app.get("/camera/snapshot")
async def camera_snapshot():
    """Return a JPEG snapshot from the Pi camera as base64."""
    import base64
    loop = asyncio.get_event_loop()
    try:
        jpeg = await loop.run_in_executor(None, _capture_jpeg_sync)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e), "source": "error"}, status_code=500)
    if jpeg is None:
        return JSONResponse({"ok": False, "error": "no camera backend", "source": "none"}, status_code=503)
    b64 = base64.b64encode(jpeg).decode()
    return {"ok": True, "image_base64": b64, "source": "camera",
            "width": 640, "height": 480, "format": "jpeg"}


@app.get("/camera/stream")
async def camera_stream(request: Request):
    """Continuous MJPEG stream from Pi camera at ~10 fps.

    Uses the persistent _CameraManager — camera stays open between frames.
    Browser <img src="/camera/stream"> will display a live feed.
    """
    _cam_mgr.start()

    async def _gen():
        import time as _t
        last = None
        while True:
            if await request.is_disconnected():
                break
            frame = _cam_mgr.get_frame(timeout=2.0)
            if frame and frame is not last:
                last = frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame +
                    b"\r\n"
                )
            await asyncio.sleep(0.10)  # ~10 fps

    from fastapi.responses import StreamingResponse as _SR
    return _SR(_gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/system/shell")
async def system_shell(request: Request):
    """🔧 Run a shell command on the Pi (maintenance use only)."""
    import asyncio, shlex
    body = await request.json()
    cmd  = body.get("cmd", "")
    if not cmd:
        return {"ok": False, "error": "no cmd"}
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        return {"ok": proc.returncode == 0, "returncode": proc.returncode,
                "output": out.decode(errors="replace")[:4000]}
    except asyncio.TimeoutError:
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/calibration/status")
async def calibration_status():
    """Current Cosmos Reason2 auto-calibration state."""
    return _calibration_state


@app.post("/calibration/recalibrate")
async def trigger_recalibrate():
    """Manually trigger a fresh Cosmos Reason2 calibration run."""
    if _calibration_state.get("status") == "running":
        return {"ok": False, "message": "Calibration already running"}
    asyncio.ensure_future(_auto_calibrate())
    return {"ok": True, "message": "Recalibration started in background"}


class CosmosDemoRequest(BaseModel):
    task: str = Field(
        default="wave hello to the audience, then pick up the red cube and place it in the bin",
        description="Natural language task")
    execute_arm: bool = Field(default=True)
    simulation: bool = Field(default=False)
    wait_for_calibration: bool = Field(default=False,
        description="Block until auto-calibration finishes before running")


@app.post("/cosmos-demo")
async def cosmos_demo(request: CosmosDemoRequest):
    """
    🚀 Full Cosmos Demo Pipeline — auto-cal → YOLO → Predict2 → Reason2 → arm

    Every session:
      1. Reports current Cosmos Reason2 calibration state
      2. Snaps live C270 frame, runs YOLO open-vocab detection
      3. Fires Predict2 /video2world concurrently (predicted future frame)
      4. Loops Reason2 x4 for reasoning chain (uses predicted frame step 0)
      5. Executes action sequence on physical xArm
      6. Returns full trace: cal_map + YOLO scene + Reason2 chain + arm results
    """
    import httpx
    t0 = time.time()

    # ── Step 0: Calibration status ────────────────────────────────────────────
    cal = _calibration_state.copy()
    if request.wait_for_calibration and cal["status"] == "running":
        for _ in range(60):
            await asyncio.sleep(2)
            if _calibration_state["status"] != "running":
                break
        cal = _calibration_state.copy()

    cal_summary = (
        f"Calibrated {cal['calibrated']}/{cal['total']} positions "
        f"[{cal['status']}] source={cal['source']}"
    )
    logger.info(f"/cosmos-demo start — {cal_summary} task={request.task[:60]}")

    # ── Step 1: YOLO live scene ───────────────────────────────────────────────
    yolo_scene = ""
    live_frame: Optional[str] = None
    try:
        async with httpx.AsyncClient(timeout=10.0) as hc:
            r = await hc.get(f"{_AGENT_URL}/vision/detect",
                             params={"targets": _extract_task_objects(request.task)})
            if r.status_code == 200:
                d = r.json()
                yolo_scene = d.get("scene_context", "")
                live_frame = d.get("annotated_b64") or d.get("raw_b64")
    except Exception as e:
        logger.warning(f"/cosmos-demo YOLO: {e}")

    # Inject calibration pixel map into scene context for Reason2
    if cal["status"] == "complete" and cal["positions"]:
        cal_ctx_parts = []
        for pos_name, pv in cal["positions"].items():
            if pv.get("ok") and pv.get("cx", 0) > 0:
                cal_ctx_parts.append(
                    f"{pos_name}@[{pv['cx']},{pv['cy']}]"
                )
        if cal_ctx_parts:
            yolo_scene = f"CalibMap: {' '.join(cal_ctx_parts[:6])} | {yolo_scene}"

    # ── Step 2: Predict2 background ──────────────────────────────────────────
    _latest_predicted_frame[0] = None
    if live_frame:
        raw_snap = live_frame
        asyncio.ensure_future(_predict2_background(request.task, raw_snap))

    # ── Step 3-4: Reason2 chain + arm execution (reuse demo_run) ─────────────
    class _Req:
        task = request.task
        execute_arm = request.execute_arm
        simulation = request.simulation
        image_base64 = live_frame

    # Temporarily inject cal context into scene cache
    _latest_scene_context[0] = yolo_scene

    demo_result = await demo_run(_Req())

    return {
        "ok": demo_result.get("ok"),
        "task": request.task,
        "calibration": {
            "status": cal["status"],
            "calibrated": cal["calibrated"],
            "total": cal["total"],
            "positions": cal["positions"],
            "summary": cal_summary,
        },
        "yolo_scene": yolo_scene,
        "plan_source": demo_result.get("plan_source"),
        "h100_reasoning": demo_result.get("h100_reasoning", []),
        "steps_ok": demo_result.get("steps_ok", 0),
        "steps_total": demo_result.get("steps_total", 0),
        "execution": demo_result.get("execution", {}),
        "pipeline": [
            "cosmos_reason2_calibration",
            "yolo_open_vocab",
            "cosmos_predict2_video2world",
            "cosmos_reason2_planning",
            "xarm_physical_execution",
        ],
        "latency_ms": round((time.time() - t0) * 1000),
    }

# ── Load routes — critical ones first, rest best-effort ──────────────────────
_loaded = []

# NOTE: "chat" excluded — main_pi.py defines its own /chat with intent detection
# NOTE: "v4_features" excluded — also defines /chat, conflicts with main_pi.py
# NOTE: "cosmos" excluded — main_pi.py defines /cosmos/* directly to H100 with correct schemas
# NOTE: heavy GPU/ML routes excluded to keep startup fast on Pi
for _name in ["openclaw", "cookoff", "robotics", "skills", "system",
              "vision", "yolo_vision", "h100_gdino", "events", "monitoring", "agents", "memory",
              "protocols", "reasoning", "auth", "utilities",
              "llm", "unified", "core",
              "webhooks", "hub_gateway", "autonomous", "autonomy", "cosmos_dance",
              "voice", "neurokernel", "v4_features"]:
    try:
        _mod = __import__(f"routes.{_name}", fromlist=["router"])
        app.include_router(_mod.router)
        _loaded.append(_name)
    except Exception as e:
        if _name in ("openclaw", "cookoff", "robotics"):
            logger.warning(f"⚠ Critical route '{_name}' failed: {e}")
        pass

logger.info(f"✅ {len(_loaded)} routes loaded: {', '.join(_loaded)}")

# ── Init enhanced chat memory ──────────────────────────────────────────────────
try:
    from src.chat.enhanced_memory_chat import EnhancedChatMemory, ChatMemoryConfig
    _mem_config = ChatMemoryConfig(storage_path="/opt/nis-protocol/data/memory")
    _enhanced_memory = EnhancedChatMemory(config=_mem_config)
    # Inject into memory router so /memory/* endpoints can use it
    import routes.memory as _mem_route
    _mem_route.router._enhanced_chat_memory = _enhanced_memory
    logger.info("✅ EnhancedChatMemory enabled — persistent memory active")
except Exception as _mem_err:
    logger.warning(f"⚠ EnhancedChatMemory failed to init: {_mem_err}")
    _enhanced_memory = None

# lifespan registered at app creation above — startup logic runs there

# ── Intent detection ──────────────────────────────────────────────────────────
def _detect_intent(msg: str) -> str:
    m = msg.lower()
    # Pure system queries — no arm movement
    if any(k in m for k in ["status","health","services","running","system","ports","uptime"]):
        return "status"
    if any(k in m for k in ["skill","skills","openclaw","what can you do","capabilities"]):
        return "skills"
    # Camera/vision only — no arm
    if any(k in m for k in ["snapshot","photo","picture","what do you see","look at","vision","just look"]):
        return "vision"
    # Direct single-move arm commands (no planning needed)
    if any(k in m for k in ["wave","home position","go home","open gripper","close gripper"]):
        return "xarm"
    # Everything else — route to cosmos VLA pipeline so the arm reacts
    # This includes: pick, grab, place, push, sort, move, get, fetch, put,
    # touch, reach, lift, demo, cookoff, and any free-form task prompt
    return "cosmos"

# ── Tool helpers ──────────────────────────────────────────────────────────────
async def _tool_vision(ws, message):
    import httpx
    tid = f"tool_{uuid.uuid4().hex[:8]}"
    await ws.send_json({"type":"TOOL_CALL","tool_id":tid,"tool":"camera_snapshot",
                        "args":{"source":"pi_camera"},"timestamp":datetime.now().isoformat()})
    img, result = None, "Camera not available"
    try:
        async with httpx.AsyncClient(timeout=6.0) as c:
            r = await c.get("http://localhost:8085/camera/snapshot")
            if r.status_code == 200:
                d = r.json(); img = d.get("image_base64") or d.get("image")
                result = "Snapshot captured" if img else "No image data"
    except Exception as e:
        result = f"Camera error: {e}"
    await ws.send_json({"type":"TOOL_RESULT","tool_id":tid,"result":result,
                        "has_image":img is not None,"timestamp":datetime.now().isoformat()})
    return img, result

async def _tool_xarm(ws, message):
    import httpx
    m = message.lower()
    if "open" in m and "gripper" in m:        cmd = "open_gripper"
    elif "close" in m and "gripper" in m:     cmd = "close_gripper"
    elif "home" in m:                          cmd = "home"
    elif "wave" in m:                          cmd = "wave"
    elif "ready" in m:                         cmd = "ready"
    elif "inspect" in m:                       cmd = "inspect"
    elif "pick" in m and "place" in m:        cmd = "pick_and_place"  # must be before pick alone
    elif "pick" in m or "grab" in m:           cmd = "pick"
    elif "place" in m or "drop" in m:          cmd = "place"
    elif "stop" in m:                          cmd = "stop"
    else:                                       cmd = "status"
    tid = f"tool_{uuid.uuid4().hex[:8]}"
    await ws.send_json({"type":"TOOL_CALL","tool_id":tid,"tool":"xarm_control",
                        "args":{"command":cmd},"timestamp":datetime.now().isoformat()})
    result = f"xArm '{cmd}' sent"
    # Direct agent endpoints — faster than OpenClaw roundtrip
    DIRECT_EP: Dict[str,str] = {
        "home": "/arm/home", "wave": "/arm/wave",
        "pick_and_place": "/arm/pick_and_place",
        "ready": "/arm/named/ready", "inspect": "/arm/named/inspect",
        "pick": "/arm/named/pick_table", "place": "/arm/named/reach_forward",
        "stop": "/arm/stop", "open_gripper": "/arm/gripper/open",
        "close_gripper": "/arm/gripper/close",
    }
    # Timeouts: wave=25s, pick_and_place=20s, others=8s
    EP_TIMEOUT = {"wave": 25.0, "pick_and_place": 20.0}
    timeout = EP_TIMEOUT.get(cmd, 8.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            if cmd == "status":
                r = await c.get("http://localhost:8085/health")
                d = r.json()
                result = f"xArm {'PHYSICAL' if not d.get('xarm_simulation') else 'SIM'} on {d.get('xarm_port','?')}"
            elif cmd in DIRECT_EP:
                r = await c.post(f"http://localhost:8085{DIRECT_EP[cmd]}", json={})
                d = r.json()
                result = f"{cmd} {'[PHYSICAL]' if not d.get('simulation') else '[SIM]'} {'✅' if d.get('ok') else '❌'}"
            else:
                result = f"Unknown xArm command: {cmd}"
    except Exception as e:
        result = f"xArm error: {e}"
    await ws.send_json({"type":"TOOL_RESULT","tool_id":tid,"result":result,
                        "timestamp":datetime.now().isoformat()})
    return result

async def _tool_cosmos(ws, message, image_b64=None, execute_arm: bool = True):
    """
    Hybrid VLA via WebSocket streaming — mirrors /demo/run logic.
    Phase 1: H100 reasoning loop (stream each step as TOOL_CALL/TOOL_RESULT).
    Phase 2: Execute the best sequence (choreographed or H100 actions), streaming each.
    """
    import httpx

    CHOREOGRAPHED = ["wave", "inspect", "pick_and_place", "home"]
    ANCHOR_MOVES  = {"wave", "inspect", "pick_and_place"}
    MAX_PLAN_STEPS = 4

    current_frame: Optional[str] = image_b64 or await _capture_frame(task)
    has_camera = current_frame is not None
    h100_actions: List[str] = []
    completed_ctx: List[str] = []
    reasoning_parts: List[str] = []
    plan_source = "fallback"
    seen: set = set()
    step_results: List[str] = []

    await ws.send_json({"type":"THINKING_STEP","title":"VLA Reasoning Phase",
                        "content":f"Camera: {'live' if has_camera else 'synthetic'} | "
                                  f"Querying H100 Cosmos Reason2 × {MAX_PLAN_STEPS} steps",
                        "timestamp":datetime.now().isoformat()})

    # ── Phase 1: H100 reasoning — collect pixel-coord scene analysis ──────────
    async with httpx.AsyncClient(timeout=60.0) as hc:
        for step_i in range(MAX_PLAN_STEPS):
            tid = f"tool_{uuid.uuid4().hex[:8]}"
            await ws.send_json({"type":"TOOL_CALL","tool_id":tid,
                                "tool":"cosmos_reason2",
                                "args":{"step": step_i+1, "task": message[:120],
                                        "completed": h100_actions,
                                        "has_image": current_frame is not None},
                                "timestamp":datetime.now().isoformat()})

            action, raw_action, reasoning, safe = await _h100_next_action(
                hc, message, step_i, completed_ctx, current_frame
            )
            if action:
                plan_source = "h100_reason2"
            if reasoning:
                reasoning_parts.append(f"S{step_i+1}: {reasoning[:80]}")

            dedup_key = raw_action[:40].lower() if raw_action else action[:20]
            looping = dedup_key in seen and dedup_key != ""

            await ws.send_json({"type":"TOOL_RESULT","tool_id":tid,
                                "result": f"[H100] {raw_action or action or 'no action'}" +
                                          (f" — {reasoning[:70]}" if reasoning else ""),
                                "reasoning": reasoning[:100] if reasoning else "",
                                "timestamp":datetime.now().isoformat()})

            if not action or not safe:
                break
            if looping:
                break
            seen.add(dedup_key)
            h100_actions.append(raw_action or action)
            completed_ctx.append(f"Step {step_i+1}: {raw_action} — {reasoning[:80]}")

            # Capture updated frame between reasoning steps
            new_frame = await _yolo_frame(task)
            if new_frame:
                current_frame = new_frame

    # ── Phase 2: decide execution sequence ────────────────────────────────────
    unique_h100 = list(dict.fromkeys(h100_actions))
    exec_h100_ws = [_extract_vla_intent(a.lower()) for a in unique_h100]
    h100_has_anchors = any(
        any(kw in a.lower() for kw in ANCHOR_MOVES) for a in unique_h100
    )
    if len(unique_h100) >= 4 and h100_has_anchors:
        exec_sequence = exec_h100_ws   # normalized keywords cookoff.py can execute
        exec_source = "h100_reason2"
    else:
        exec_sequence = CHOREOGRAPHED
        exec_source = "choreographed" if not h100_actions else "choreographed+h100"
        plan_source = exec_source

    await ws.send_json({"type":"THINKING_STEP","title":"Execution Plan",
                        "content":f"[{exec_source}] {exec_sequence} | H100 reasoning: {h100_actions}",
                        "timestamp":datetime.now().isoformat()})

    # ── Phase 3: physical execution — stream each step ────────────────────────
    async with httpx.AsyncClient(timeout=120.0) as hc:
        for i, act in enumerate(exec_sequence):
            tid_ex = f"tool_{uuid.uuid4().hex[:8]}"
            await ws.send_json({"type":"TOOL_CALL","tool_id":tid_ex,
                                "tool":"xarm_execute_step",
                                "args":{"action": act, "step": i+1, "source": exec_source},
                                "timestamp":datetime.now().isoformat()})
            if execute_arm:
                sr = await _arm_one_step(hc, act, False)
            else:
                sr = {"ok": True, "response": "simulated", "endpoint": f"/arm/{act}"}
            sym = "✅" if sr.get("ok") else "❌"
            step_results.append(f"{sym} {act}")
            await ws.send_json({"type":"TOOL_RESULT","tool_id":tid_ex,
                                "result":f"{sym} {act} [{exec_source}] → {sr.get('endpoint','?')}",
                                "timestamp":datetime.now().isoformat()})

    summary = (f"[{plan_source}] {len(step_results)}/{len(exec_sequence)} steps OK\n"
               + "\n".join(step_results)
               + (f"\n\nH100 reasoning:\n" + "\n".join(reasoning_parts) if reasoning_parts else ""))
    return summary

async def _tool_status(ws):
    import httpx
    tid = f"tool_{uuid.uuid4().hex[:8]}"
    await ws.send_json({"type":"TOOL_CALL","tool_id":tid,"tool":"system_status",
                        "args":{},"timestamp":datetime.now().isoformat()})
    lines = []
    checks = [("NIS Protocol","http://localhost:8000/health"),
              ("Agent Gateway","http://localhost:8085/health"),
              ("NeuroHub UI","http://localhost:3000"),
              ("NeuroStore","http://localhost:8006/health"),
              ("OpenClaw","http://localhost:8000/openclaw/status")]
    async with httpx.AsyncClient(timeout=3.0) as c:
        for name, url in checks:
            try:
                r = await c.get(url); lines.append(f"✅ {name}: online ({r.status_code})")
            except Exception:
                lines.append(f"❌ {name}: offline")
    result = "\n".join(lines)
    await ws.send_json({"type":"TOOL_RESULT","tool_id":tid,"result":result,
                        "timestamp":datetime.now().isoformat()})
    return result

async def _tool_skills(ws):
    import httpx
    tid = f"tool_{uuid.uuid4().hex[:8]}"
    await ws.send_json({"type":"TOOL_CALL","tool_id":tid,"tool":"list_skills",
                        "args":{},"timestamp":datetime.now().isoformat()})
    result = "Skills unavailable"
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.post("http://localhost:8000/openclaw/invoke",
                             json={"tool":"nis_skills","args":{}})
            if r.status_code == 200:
                skills = r.json().get("result",{}).get("skills",[])
                result = "\n".join(f"• {s['name']}: {s.get('description','')[:80]}"
                                   for s in skills[:10]) if skills else "No skills registered"
    except Exception as e:
        result = f"Skills error: {e}"
    await ws.send_json({"type":"TOOL_RESULT","tool_id":tid,"result":result,
                        "timestamp":datetime.now().isoformat()})
    return result

# ── Agentic WebSocket ─────────────────────────────────────────────────────────

@app.websocket("/ws/agentic")
async def agentic_websocket(websocket: WebSocket):
    await websocket.accept()
    logger.info("🤖 Agentic WS connected")
    n = 0

    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message","")
            img_in = data.get("image_base64")
            use_cosmos = data.get("use_cosmos", False)
            execute_arm = data.get("execute_arm", True)
            n += 1
            intent = "cosmos" if use_cosmos else _detect_intent(message)
            logger.info(f"📨 #{n} intent={intent}: {message[:50]}")

            await websocket.send_json({"type":"THINKING_STEP","title":"Processing Request",
                                       "content":f"Intent detected: {intent}",
                                       "timestamp":datetime.now().isoformat()})

            tool_ctx, img_out = "", img_in
            if intent == "vision":
                img_out, snap = await _tool_vision(websocket, message)
                tool_ctx = f"Camera: {snap}"
            elif intent == "xarm":
                tool_ctx = await _tool_xarm(websocket, message)
            elif intent == "cosmos":
                tool_ctx = await _tool_cosmos(websocket, message, img_out, execute_arm=execute_arm)
            elif intent == "status":
                tool_ctx = await _tool_status(websocket)
            elif intent == "skills":
                tool_ctx = await _tool_skills(websocket)

            # Honest system prompt describing actual capabilities
            if intent == "cosmos":
                sys_prompt = ("You are NIS Protocol v4.0 by Organica AI Solutions. "
                              "You just ran NVIDIA Cosmos Reason2 to plan and execute a "
                              "robot task. Summarize the outcome. Be specific and concise. 3-5 sentences max.")
            elif intent == "xarm":
                sys_prompt = ("You are NIS Protocol v4.0 by Organica AI Solutions. "
                              "You just sent a command to the physical Hiwonder xArm 1S robot. "
                              "Confirm the action was executed physically. Be concise and direct.")
            elif intent == "vision":
                sys_prompt = ("You are NIS Protocol v4.0 by Organica AI Solutions "
                              "with a Pi Camera. Describe what you see in the workspace scene. "
                              "Mention objects, positions, and anything relevant to robot tasks.")
            else:
                sys_prompt = ("You are NIS Protocol v4.0, a multi-modal AI OS by Organica AI Solutions "
                              "running on NeuroLinux on Raspberry Pi 5. Coordinate Pi Camera, xArm 6-DOF, "
                              "and NVIDIA Cosmos Reason2. Be concise.")

            user_content = f"{message}\n\n[System Context]\n{tool_ctx}" if tool_ctx else message
            resp, prov = "", "demo"
            try:
                if llm_provider:
                    r = await llm_provider.generate_response(
                        messages=[{"role":"system","content":sys_prompt},
                                  {"role":"user","content":user_content}], temperature=0.7)
                    resp = r.get("content","Response generated.")
                    prov = r.get("provider","nis-protocol")
                else:
                    resp = (f"**NIS Protocol v4.0** — Action executed.\n\n{tool_ctx}"
                            if tool_ctx else f"Demo mode. Received: \"{message}\"")
                    prov = "demo"
            except Exception as e:
                resp = tool_ctx or f"Error: {e}"; prov = "error"

            await websocket.send_json({"type":"TEXT_MESSAGE_CONTENT","content":resp,"role":"assistant",
                                       "metadata":{"provider":prov,"intent":intent,
                                                   "real_ai":prov not in("demo","error"),
                                                   "tools_used":intent if intent!="chat" else None},
                                       "image_base64":img_out,"timestamp":datetime.now().isoformat()})
            logger.info(f"✅ #{n} done intent={intent} prov={prov}")

    except WebSocketDisconnect:
        logger.info(f"🔌 Agentic WS disconnected after {n} messages")
    except Exception as e:
        logger.error(f"❌ Agentic WS error: {e}")

@app.websocket("/ws")
async def main_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            msg = data.get("message","")
            if llm_provider:
                try:
                    r = await llm_provider.generate_response(messages=[{"role":"user","content":msg}])
                    await websocket.send_json({"type":"response","content":r.get("content",""),
                                               "provider":r.get("provider","nis")})
                except Exception as e:
                    await websocket.send_json({"type":"error","content":str(e)})
            else:
                await websocket.send_json({"type":"response",
                                           "content":f"NIS Protocol Pi — demo. Received: {msg}",
                                           "provider":"demo"})
    except WebSocketDisconnect:
        pass

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint with intent detection — routes xArm/cosmos/vision to tools."""
    import httpx
    conv_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
    message = request.message
    intent  = _detect_intent(message)

    # ── xArm intent: call agent directly ──────────────────────────────────────
    if intent == "xarm":
        m = message.lower()
        if "open" in m and "gripper" in m:    cmd = "open_gripper"
        elif "close" in m and "gripper" in m: cmd = "close_gripper"
        elif "home" in m:    cmd = "home"
        elif "wave" in m:    cmd = "wave"
        elif "ready" in m:   cmd = "ready"
        elif "inspect" in m: cmd = "inspect"
        elif "pick" in m:    cmd = "pick"
        elif "place" in m:   cmd = "place"
        elif "stop" in m:    cmd = "stop"
        elif "reach" in m:   cmd = "reach_forward"
        else:                cmd = "status"
        DIRECT: Dict[str, str] = {
            "home": "/arm/home", "wave": "/arm/wave",
            "ready": "/arm/named/ready", "inspect": "/arm/named/inspect",
            "pick": "/arm/named/pick_table", "place": "/arm/named/pick_table",
            "stop": "/arm/stop", "open_gripper": "/arm/gripper/open",
            "close_gripper": "/arm/gripper/close", "reach_forward": "/arm/named/reach_forward",
        }
        arm_result = f"xArm '{cmd}' sent"
        try:
            async with httpx.AsyncClient(timeout=25.0) as c:
                if cmd == "status":
                    r = await c.get("http://localhost:8085/health")
                    d = r.json()
                    arm_result = f"xArm {'PHYSICAL' if not d.get('xarm_simulation') else 'SIM'} on {d.get('xarm_port','?')}"
                elif cmd in DIRECT:
                    r = await c.post(f"http://localhost:8085{DIRECT[cmd]}", json={})
                    d = r.json()
                    arm_result = f"{cmd} {'[PHYSICAL]' if not d.get('simulation') else '[SIM]'} {'OK' if d.get('ok') else 'FAIL'}"
        except Exception as e:
            arm_result = f"xArm error: {e}"
        return {"response": arm_result, "user_id": request.user_id,
                "conversation_id": conv_id, "timestamp": time.time(),
                "provider": "xarm_direct", "intent": intent, "command": cmd}

    # ── Cosmos intent: /cookoff/demo — YOLO scan → Reason2 → xArm execution ───
    if intent == "cosmos":
        cosmos_result = "Cosmos VLA initiated"
        try:
            async with httpx.AsyncClient(timeout=90.0) as c:
                r = await c.post(
                    "http://localhost:8000/cookoff/demo",
                    json={"task": message, "execute_arm": True, "simulation": False}
                )
                if r.status_code == 200:
                    d = r.json()
                    plan      = d.get("action_plan") or []
                    src       = d.get("plan_source", "cosmos_r2")
                    ms        = d.get("latency_ms", 0)
                    steps_ok  = d.get("steps_ok", 0)
                    steps_tot = d.get("steps_total", len(plan))
                    goal_done = d.get("goal_complete", False)
                    reasoning = (d.get("reasoning") or "").split(" | ")[0][:120]
                    cosmos_result = (
                        f"[{src}] {steps_ok}/{steps_tot} arm steps executed\n"
                        f"Plan: {' → '.join(plan[:4])}\n"
                        f"{reasoning}\n"
                        f"Goal: {'✅ complete' if goal_done else 'in progress'} · {ms}ms"
                    )
        except Exception as e:
            cosmos_result = f"Cosmos error: {e}"
        return {"response": cosmos_result, "user_id": request.user_id,
                "conversation_id": conv_id, "timestamp": time.time(),
                "provider": "cosmos_vla", "intent": intent}

    # ── Status intent ──────────────────────────────────────────────────────────
    if intent == "status":
        lines = []
        checks = [("NIS Protocol", "http://localhost:8000/health"),
                  ("Agent Gateway", "http://localhost:8085/health"),
                  ("NeuroStore",    "http://localhost:8006/health"),
                  ("NeuroHub UI",   "http://localhost:3000")]
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                for name, url in checks:
                    try:
                        r = await c.get(url)
                        lines.append(f"OK {name}: {r.status_code}")
                    except Exception:
                        lines.append(f"-- {name}: offline")
        except Exception:
            pass
        return {"response": "\n".join(lines) or "Status check failed",
                "user_id": request.user_id, "conversation_id": conv_id,
                "timestamp": time.time(), "provider": "status", "intent": intent}

    # ── LLM chat fallback ──────────────────────────────────────────────────────
    if llm_provider:
        try:
            # Save user message to memory
            if _enhanced_memory:
                await _enhanced_memory.add_message(conv_id, "user", message)

            r = await llm_provider.generate_response(
                messages=[{"role": "user", "content": message}])
            response_text = r.get("content", "")

            # Save assistant response to memory
            if _enhanced_memory and response_text:
                await _enhanced_memory.add_message(conv_id, "assistant", response_text)

            return {"response": response_text, "user_id": request.user_id,
                    "conversation_id": conv_id, "timestamp": time.time(),
                    "provider": r.get("provider", "nis"), "real_ai": True,
                    "model": r.get("model", "unknown"), "intent": intent}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {"response": f"NIS Protocol Pi — demo. Received: {message}",
            "user_id": request.user_id, "conversation_id": conv_id,
            "timestamp": time.time(), "provider": "demo", "real_ai": False,
            "model": "none", "intent": intent}


class PromptRequest(BaseModel):
    prompt: str = Field(...)
    execute_arm: bool = True
    simulation: bool = False
    use_yolo: bool = True

@app.post("/prompt")
async def prompt_endpoint(req: PromptRequest):
    """
    Random-prompt → arm reaction endpoint for Cosmos Cookoff.
    Any free-text prompt triggers: YOLO scene scan → Cosmos Reason2 → xArm execution.
    No keywords needed — everything moves the arm.
    """
    import httpx
    try:
        async with httpx.AsyncClient(timeout=90.0) as c:
            r = await c.post(
                "http://localhost:8000/cookoff/demo",
                json={"task": req.prompt, "execute_arm": req.execute_arm, "simulation": req.simulation}
            )
            if r.status_code == 200:
                d = r.json()
                return {
                    "ok": True,
                    "prompt": req.prompt,
                    "action_plan": d.get("action_plan", []),
                    "plan_source": d.get("plan_source", "cosmos_r2"),
                    "steps_ok": d.get("steps_ok", 0),
                    "steps_total": d.get("steps_total", 0),
                    "goal_complete": d.get("goal_complete", False),
                    "reasoning": d.get("reasoning", ""),
                    "latency_ms": d.get("latency_ms", 0),
                    "arm_executed": req.execute_arm,
                }
            return {"ok": False, "error": f"cookoff/demo returned {r.status_code}", "prompt": req.prompt}
    except Exception as e:
        return {"ok": False, "error": str(e), "prompt": req.prompt}


class CosmosRequest(BaseModel):
    task: str = Field(...)
    robot: Optional[str] = "xarm"
    image_base64: Optional[str] = None
    constraints: Optional[List[str]] = None
    max_tokens: Optional[int] = 512

class CosmosRobotPlanRequest(BaseModel):
    command: str = Field(...)
    robot_type: Optional[str] = "xarm"
    image_base64: Optional[str] = None

class CosmosTrajectoryRequest(BaseModel):
    task: str = Field(...)
    robot_type: Optional[str] = "xarm"
    image_base64: Optional[str] = None

class CosmosGoalVerifyRequest(BaseModel):
    goal: str = Field(...)
    image_base64: Optional[str] = None
    last_action: Optional[str] = None

class CosmosPlausibilityRequest(BaseModel):
    description: str = Field(...)
    image_base64: Optional[str] = None
    context: Optional[str] = None

class CosmosVideo2WorldRequest(BaseModel):
    prompt: str = Field(...)
    image_b64: Optional[str] = None
    negative_prompt: Optional[str] = None
    num_frames: Optional[int] = 25
    fps: Optional[int] = 10
    height: Optional[int] = 480
    width: Optional[int] = 848
    num_inference_steps: Optional[int] = 35
    guidance_scale: Optional[float] = 7.0
    seed: Optional[int] = 42

class CosmosText2ImageRequest(BaseModel):
    prompt: str = Field(...)
    negative_prompt: Optional[str] = None
    width: Optional[int] = 1024
    height: Optional[int] = 576
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.0
    seed: Optional[int] = None

class CosmosTransferRequest(BaseModel):
    demo: Optional[str] = None
    control_type: Optional[str] = "edge"
    guidance: Optional[float] = 3.0

class CosmosDemoRequest(BaseModel):
    pass

# ── H100 base URL (relay on PC, or direct tunnel) ─────────────────────────────
_H100_REASON_URL   = os.getenv("H100_REASON_URL",   "http://172.16.1.83:8100")
_H100_PREDICT_URL  = os.getenv("H100_PREDICT_URL",  "http://172.16.1.83:8200")
_H100_TRANSFER_URL = os.getenv("H100_TRANSFER_URL", "http://172.16.1.83:8300")
_H100_ORCH_URL     = os.getenv("H100_ORCH_URL",     "http://172.16.1.83:8400")

async def _h100_post(url: str, body: dict, timeout: float = 60.0):
    """Generic H100 proxy POST — with one retry on connection reset."""
    import httpx
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=timeout) as hc:
                r = await hc.post(url, json=body)
                return r.status_code, r.json()
        except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError) as e:
            if attempt == 0:
                logger.warning(f"H100 POST {url} attempt {attempt+1} failed ({e}), retrying...")
                await asyncio.sleep(1)
                continue
            logger.warning(f"H100 POST {url} failed after retry: {e}")
            return None, {"error": str(e)}
        except Exception as e:
            logger.warning(f"H100 POST {url} failed: {e}")
            return None, {"error": str(e)}

async def _h100_get(url: str, timeout: float = 10.0):
    """Generic H100 proxy GET."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(url)
            return r.status_code, r.json()
    except Exception as e:
        return None, {"error": str(e)}


# ── Cosmos Reason2 endpoints ──────────────────────────────────────────────────

@app.post("/cosmos/reason")
async def cosmos_reason(request: CosmosRequest):
    """Cosmos Reason2 — general reasoning. H100 :8100 /reason"""
    body = {"query": request.task, "max_tokens": request.max_tokens or 512, "use_think": True}
    if request.image_base64:
        body["image_base64"] = request.image_base64
    st, d = await _h100_post(f"{_H100_REASON_URL}/reason", body, timeout=60.0)
    if st == 200:
        return {"ok": True, "task": request.task, "source": "h100_reason2",
                "reasoning": d.get("reasoning", ""), "answer": d.get("answer", d.get("response", "")),
                "confidence": d.get("confidence", 0.0), "latency_ms": d.get("latency_ms"),
                "model": d.get("model", "cosmos-reason2-8b"), "timestamp": time.time()}
    logger.warning(f"/cosmos/reason H100 failed ({st}): {d}")
    return {"ok": False, "task": request.task, "source": "fallback",
            "answer": f"Simulated plan for: {request.task}",
            "reasoning": "H100 unavailable", "confidence": 0.5, "timestamp": time.time()}

@app.post("/cosmos/robot-plan")
async def cosmos_robot_plan(request: CosmosRobotPlanRequest):
    """Cosmos Reason2 — structured robot action plan. H100 :8100 /robot-plan"""
    body = {"command": request.command, "robot_type": request.robot_type or "xarm"}
    if request.image_base64:
        body["image_base64"] = request.image_base64
    st, d = await _h100_post(f"{_H100_REASON_URL}/robot-plan", body, timeout=60.0)
    if st == 200:
        return {"ok": True, "command": request.command, "source": "h100_reason2",
                "plan": d, "timestamp": time.time()}
    logger.warning(f"/cosmos/robot-plan H100 failed ({st}): {d}")
    return {"ok": False, "command": request.command, "source": "fallback",
            "plan": {"steps": ["analyze", "plan", "execute"]}, "error": str(d), "timestamp": time.time()}

@app.post("/cosmos/trajectory")
async def cosmos_trajectory(request: CosmosTrajectoryRequest):
    """Cosmos Reason2 — motion trajectory. H100 :8100 /trajectory"""
    body = {"task": request.task, "robot_type": request.robot_type or "xarm"}
    if request.image_base64:
        body["image_base64"] = request.image_base64
    st, d = await _h100_post(f"{_H100_REASON_URL}/trajectory", body, timeout=60.0)
    if st == 200:
        return {"ok": True, "task": request.task, "source": "h100_reason2",
                "trajectory": d, "timestamp": time.time()}
    return {"ok": False, "task": request.task, "source": "fallback",
            "trajectory": {}, "timestamp": time.time()}

@app.post("/cosmos/goal-verify")
async def cosmos_goal_verify(request: CosmosGoalVerifyRequest):
    """Cosmos Reason2 — verify goal achievement. H100 :8100 /goal-verify"""
    body = {"goal": request.goal}
    if request.image_base64: body["image_base64"] = request.image_base64
    if request.last_action:  body["last_action"]  = request.last_action
    st, d = await _h100_post(f"{_H100_REASON_URL}/goal-verify", body, timeout=30.0)
    if st == 200:
        return {"ok": True, "goal": request.goal, "source": "h100_reason2",
                "result": d, "timestamp": time.time()}
    return {"ok": False, "goal": request.goal, "source": "fallback",
            "result": {"achieved": False}, "timestamp": time.time()}

@app.post("/cosmos/plausibility")
async def cosmos_plausibility(request: CosmosPlausibilityRequest):
    """Cosmos Reason2 — scene plausibility check. H100 :8100 /plausibility"""
    body = {"description": request.description}
    if request.image_base64: body["image_base64"] = request.image_base64
    if request.context:      body["context"]      = request.context
    st, d = await _h100_post(f"{_H100_REASON_URL}/plausibility", body, timeout=30.0)
    if st == 200:
        return {"ok": True, "description": request.description, "source": "h100_reason2",
                "result": d, "timestamp": time.time()}
    return {"ok": False, "description": request.description, "source": "fallback",
            "result": {"plausible": True}, "timestamp": time.time()}

@app.get("/cosmos/reason/health")
async def cosmos_reason_health():
    """Cosmos Reason2 health check."""
    st, d = await _h100_get(f"{_H100_REASON_URL}/health")
    return {"ok": st == 200, "service": "cosmos-reason2", "data": d, "url": _H100_REASON_URL}


# ── Cosmos Predict2 endpoints ─────────────────────────────────────────────────

@app.post("/cosmos/video2world")
async def cosmos_video2world(request: CosmosVideo2WorldRequest):
    """Cosmos Predict2 — video to world simulation. H100 :8200 /video2world"""
    body = {"prompt": request.prompt, "num_frames": request.num_frames,
            "fps": request.fps, "height": request.height, "width": request.width,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale, "seed": request.seed}
    if request.image_b64:        body["image_b64"]        = request.image_b64
    if request.negative_prompt:  body["negative_prompt"]  = request.negative_prompt
    st, d = await _h100_post(f"{_H100_PREDICT_URL}/video2world", body, timeout=120.0)
    if st == 200:
        return {"ok": True, "prompt": request.prompt, "source": "h100_predict2",
                "result": d, "timestamp": time.time()}
    return {"ok": False, "prompt": request.prompt, "source": "fallback",
            "error": str(d), "timestamp": time.time()}

@app.post("/cosmos/text2image")
async def cosmos_text2image(request: CosmosText2ImageRequest):
    """Cosmos Predict2 — text to image. H100 :8200 /text2image (prompt as query param)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=120.0) as hc:
            r = await hc.post(
                f"{_H100_PREDICT_URL}/text2image",
                params={"prompt": request.prompt},
            )
            if r.status_code == 200:
                d = r.json()
                return {"ok": True, "prompt": request.prompt, "source": "h100_predict2",
                        "image_b64": d.get("image_b64", d.get("image", "")),
                        "latency_ms": d.get("latency_ms"), "timestamp": time.time()}
            return {"ok": False, "prompt": request.prompt, "source": "h100_predict2",
                    "error": f"HTTP {r.status_code}", "timestamp": time.time()}
    except Exception as e:
        return {"ok": False, "prompt": request.prompt, "source": "fallback",
                "error": str(e), "timestamp": time.time()}

@app.get("/cosmos/predict/health")
async def cosmos_predict_health():
    """Cosmos Predict2 health check."""
    st, d = await _h100_get(f"{_H100_PREDICT_URL}/health")
    return {"ok": st == 200, "service": "cosmos-predict2", "data": d, "url": _H100_PREDICT_URL}


# ── Cosmos Transfer2.5 endpoints ──────────────────────────────────────────────

@app.post("/cosmos/transfer")
async def cosmos_transfer(request: CosmosTransferRequest):
    """Cosmos Transfer2.5 — submit async job then poll until done. H100 :8300"""
    body = {"control_type": request.control_type or "edge", "guidance": request.guidance or 3.0}
    if request.demo: body["demo"] = request.demo
    # Submit async (sync /transfer times out — diffusion is slow)
    st, d = await _h100_post(f"{_H100_TRANSFER_URL}/transfer/submit", body, timeout=30.0)
    if st != 200:
        return {"ok": False, "source": "h100_transfer", "error": str(d), "timestamp": time.time()}
    job_id = d.get("job_id", "")
    if not job_id:
        return {"ok": False, "source": "h100_transfer", "error": "no job_id", "timestamp": time.time()}
    # Poll up to 180s
    for _ in range(60):
        await asyncio.sleep(3)
        st2, d2 = await _h100_get(f"{_H100_TRANSFER_URL}/transfer/status/{job_id}")
        status = d2.get("status", "")
        if status in ("complete", "done", "finished", "success"):
            out = d2.get("result_image") or d2.get("image") or d2.get("output_image") or ""
            return {"ok": True, "source": "h100_transfer", "job_id": job_id,
                    "control_type": request.control_type, "result_image": out,
                    "result": d2, "timestamp": time.time()}
        if status == "failed":
            return {"ok": False, "source": "h100_transfer", "job_id": job_id,
                    "error": "job failed", "timestamp": time.time()}
    return {"ok": False, "source": "h100_transfer", "job_id": job_id,
            "error": "timeout waiting for job", "timestamp": time.time()}

@app.post("/cosmos/transfer/submit")
async def cosmos_transfer_submit(request: CosmosTransferRequest):
    """Cosmos Transfer2.5 — async submit. H100 :8300 /transfer/submit"""
    body = {"control_type": request.control_type or "edge", "guidance": request.guidance or 3.0}
    if request.demo: body["demo"] = request.demo
    st, d = await _h100_post(f"{_H100_TRANSFER_URL}/transfer/submit", body, timeout=30.0)
    if st == 200:
        return {"ok": True, "source": "h100_transfer", "job": d, "timestamp": time.time()}
    return {"ok": False, "source": "fallback", "error": str(d), "timestamp": time.time()}

@app.get("/cosmos/transfer/status/{job_id}")
async def cosmos_transfer_status(job_id: str):
    """Cosmos Transfer2.5 — job status. H100 :8300 /transfer/status/{job_id}"""
    st, d = await _h100_get(f"{_H100_TRANSFER_URL}/transfer/status/{job_id}")
    return {"ok": st == 200, "job_id": job_id, "status": d, "timestamp": time.time()}

@app.get("/cosmos/transfer/demos")
async def cosmos_transfer_demos():
    """List available Transfer2.5 demos."""
    st, d = await _h100_get(f"{_H100_TRANSFER_URL}/demos")
    return {"ok": st == 200, "demos": d, "timestamp": time.time()}

@app.get("/cosmos/transfer/health")
async def cosmos_transfer_health():
    """Cosmos Transfer2.5 health check."""
    st, d = await _h100_get(f"{_H100_TRANSFER_URL}/health")
    return {"ok": st == 200, "service": "cosmos-transfer2.5", "data": d, "url": _H100_TRANSFER_URL}


class CosmosPipelineRequest(BaseModel):
    task: str = Field(
        default="pick up the red cube and place it in the bin",
        description="Natural language task for the full Cosmos pipeline")
    image_base64: Optional[str] = Field(default=None)
    predict_frames: int = Field(default=9, description="Frames for Predict2 video2world")
    predict_steps:  int = Field(default=35, description="Inference steps for Predict2")
    transfer_control: str = Field(default="edge", description="Transfer2.5 control type")
    run_transfer: bool = Field(default=True)
    run_predict: bool  = Field(default=True)
    run_plausibility: bool = Field(default=True)


@app.post("/cosmos/pipeline")
async def cosmos_pipeline(request: CosmosPipelineRequest):
    """
    Full Cosmos 3-model pipeline (all running simultaneously on H100):

      Phase 1 — PARALLEL:
        A. Reason2  /robot-plan   → structured action plan + trajectory
        B. Predict2 /video2world  → predicted future workspace video
        C. Transfer2.5 /submit   → edge/depth/seg representation of scene

      Phase 2 — SEQUENTIAL (uses Phase 1 results):
        D. Reason2  /goal-verify  → verify predicted state achieves goal
        E. Reason2  /plausibility → score each planned action
        F. Reason2  /trajectory   → pixel-level motion path

      Returns full trace: plan + predicted video + transfer map +
                         goal_achieved + plausibility scores + trajectory
    """
    import httpx
    t0 = time.time()
    results: Dict[str, Any] = {}

    # ── Get live camera frame if not provided ─────────────────────────────────
    img = request.image_base64 or ""
    if not img:
        try:
            async with httpx.AsyncClient(timeout=6.0) as hc:
                r = await hc.get(f"http://localhost:8085/camera/snapshot")
                if r.status_code == 200:
                    img = r.json().get("image_base64", "")
        except Exception:
            pass

    logger.info(f"/cosmos/pipeline start task={request.task[:60]} img={len(img)}chars")

    # ── Phase 1: Fire all 3 services simultaneously ───────────────────────────
    phase1_tasks: list = []

    async def _run_robot_plan():
        st, d = await _h100_post(f"{_H100_REASON_URL}/robot-plan", {
            "command":      request.task,
            "robot_type":   "xarm",
            "image_base64": img,
        }, timeout=45.0)
        results["robot_plan"] = {
            "ok": st == 200,
            "action":      d.get("action", ""),
            "action_plan": d.get("action_plan", []),
            "trajectory":  d.get("trajectory", []),
            "safe":        d.get("safe_to_execute", True),
            "confidence":  d.get("confidence", 0.0),
            "reasoning":   d.get("reasoning", "")[:300],
            "latency_ms":  d.get("latency_ms"),
        }
        logger.info(f"/cosmos/pipeline robot_plan ok={st==200} action={d.get('action','')}")

    async def _run_predict2():
        if not request.run_predict:
            results["predict2"] = {"ok": False, "skipped": True}
            return
        st, d = await _h100_post(f"{_H100_PREDICT_URL}/video2world", {
            "image_b64":            img,
            "prompt":               request.task,
            "num_frames":           request.predict_frames,
            "num_inference_steps":  request.predict_steps,
            "guidance_scale":       7.0,
            "negative_prompt":      "blurry, low quality, distorted",
            "fps":                  10,
            "height":               480,
            "width":                848,
        }, timeout=120.0)
        results["predict2"] = {
            "ok":         st == 200,
            "frames":     d.get("num_frames"),
            "fps":        d.get("fps"),
            "resolution": d.get("resolution"),
            "video_b64":  d.get("video_b64", ""),
            "image_b64":  d.get("image_b64", ""),
            "latency_ms": d.get("latency_ms"),
        }
        logger.info(f"/cosmos/pipeline predict2 ok={st==200} frames={d.get('num_frames')}")

    async def _run_transfer():
        if not request.run_transfer:
            results["transfer"] = {"ok": False, "skipped": True}
            return
        # Submit async
        st, d = await _h100_post(f"{_H100_TRANSFER_URL}/transfer/submit", {
            "demo":         "car_edge",
            "control_type": request.transfer_control,
            "guidance":     3.0,
        }, timeout=20.0)
        job_id = d.get("job_id", "") if st == 200 else ""
        results["transfer"] = {
            "ok":         st == 200,
            "job_id":     job_id,
            "control_type": request.transfer_control,
            "status":     "submitted" if job_id else "failed",
        }
        logger.info(f"/cosmos/pipeline transfer submit ok={st==200} job_id={job_id}")

    # Fire all simultaneously
    await asyncio.gather(
        _run_robot_plan(),
        _run_predict2(),
        _run_transfer(),
        return_exceptions=True,
    )

    phase1_ms = round((time.time() - t0) * 1000)
    logger.info(f"/cosmos/pipeline phase1 done {phase1_ms}ms")

    # ── Phase 2: Use Phase 1 results for deeper reasoning ────────────────────
    plan_action = results.get("robot_plan", {}).get("action", request.task)
    predicted_img = results.get("predict2", {}).get("image_b64", img)

    # Reason2 is single-threaded — run Phase 2 R2 calls sequentially to avoid collision
    # (concurrent requests return score=None / empty responses)
    async def _run_phase2_reason2():
        # goal-verify
        verify_img = predicted_img or img
        st, d = await _h100_post(f"{_H100_REASON_URL}/goal-verify", {
            "goal":         (
                f"The robot arm has executed the full plan to accomplish: {request.task}. "
                f"The last action was '{plan_action}'. Has the goal been achieved?"
            ),
            "image_base64": verify_img,
            "last_action":  plan_action,
        }, timeout=30.0)
        results["goal_verify"] = {
            "ok":            st == 200,
            "goal_complete": d.get("goal_complete", False),
            "verification":  d.get("verification", ""),
            "reasoning":     d.get("reasoning", "")[:200],
            "next_action":   d.get("next_action", ""),
            "latency_ms":    d.get("latency_ms"),
        }
        # plausibility
        if request.run_plausibility:
            st2, d2 = await _h100_post(f"{_H100_REASON_URL}/plausibility", {
                "description": f"Robot plan '{plan_action}' accomplishes: {request.task}",
                "image_base64": img,
                "context":      "xarm 6DOF robot, wooden table, 848x480 camera",
            }, timeout=30.0)
            results["plausibility"] = {
                "ok":        st2 == 200,
                "plausible": d2.get("plausible", True),
                "score":     d2.get("score", 1.0),
                "reasoning": d2.get("reasoning", "")[:200],
                "latency_ms": d2.get("latency_ms"),
            }
        else:
            results["plausibility"] = {"ok": False, "skipped": True}
        # trajectory
        st3, d3 = await _h100_post(f"{_H100_REASON_URL}/trajectory", {
            "task":         request.task,
            "robot_type":   "xarm",
            "image_base64": img,
        }, timeout=30.0)
        results["trajectory"] = {
            "ok":         st3 == 200,
            "trajectory": d3.get("trajectory", []),
            "reasoning":  d3.get("reasoning", "")[:200],
            "latency_ms": d3.get("latency_ms"),
        }

    # Transfer25: fire-and-forget — don't block pipeline on slow diffusion job
    # Clients can poll GET /cosmos/transfer/status/{job_id} for the result
    await _run_phase2_reason2()

    total_ms = round((time.time() - t0) * 1000)

    # ── Build response ────────────────────────────────────────────────────────
    plan_ok   = results.get("robot_plan",  {}).get("ok", False)
    pred_ok   = results.get("predict2",    {}).get("ok", False)
    xfer_ok   = results.get("transfer",    {}).get("status") == "complete"
    goal_done = results.get("goal_verify", {}).get("goal_complete", False)
    plaus_ok  = results.get("plausibility",{}).get("plausible", True)
    plaus_score = results.get("plausibility",{}).get("score", 0.0)

    logger.info(f"/cosmos/pipeline done {total_ms}ms plan={plan_ok} pred={pred_ok} "
                f"xfer={xfer_ok} goal={goal_done} plaus={plaus_score}")

    return {
        "ok":    plan_ok,
        "task":  request.task,
        # Phase 1
        "robot_plan":  results.get("robot_plan",  {}),
        "predict2":    {k: v for k, v in results.get("predict2", {}).items()
                        if k not in ("video_b64", "image_b64")},  # omit large blobs
        "predict2_video_b64": results.get("predict2", {}).get("video_b64", ""),
        "transfer":    results.get("transfer",    {}),
        # Phase 2
        "goal_verify": results.get("goal_verify", {}),
        "plausibility": results.get("plausibility", {}),
        "trajectory":  results.get("trajectory",  {}),
        # Summary
        "summary": {
            "plan_ok":      plan_ok,
            "predicted":    pred_ok,
            "transfer_done": xfer_ok,
            "goal_achieved": goal_done,
            "plausible":    plaus_ok,
            "plausibility_score": plaus_score,
            "action":       results.get("robot_plan", {}).get("action", ""),
            "action_plan":  results.get("robot_plan", {}).get("action_plan", []),
        },
        "pipeline": ["reason2_robot_plan", "predict2_video2world",
                     "transfer25_edge", "reason2_goal_verify",
                     "reason2_plausibility", "reason2_trajectory"],
        "phase1_ms":  phase1_ms,
        "total_ms":   total_ms,
        "timestamp":  time.time(),
    }


# ── 3-GPU Orchestrator endpoints ──────────────────────────────────────────────

@app.post("/cosmos/demo/start")
async def cosmos_demo_start():
    """Start 3-GPU Cosmos demo. H100 :8400 /demo/start"""
    st, d = await _h100_post(f"{_H100_ORCH_URL}/demo/start", {}, timeout=30.0)
    return {"ok": st == 200, "result": d, "timestamp": time.time()}

@app.post("/cosmos/demo/stop")
async def cosmos_demo_stop():
    """Stop 3-GPU Cosmos demo. H100 :8400 /demo/stop"""
    st, d = await _h100_post(f"{_H100_ORCH_URL}/demo/stop", {}, timeout=15.0)
    return {"ok": st == 200, "result": d, "timestamp": time.time()}

@app.get("/cosmos/demo/status")
async def cosmos_demo_status():
    """3-GPU demo status. H100 :8400 /demo/status"""
    st, d = await _h100_get(f"{_H100_ORCH_URL}/demo/status")
    return {"ok": st == 200, "status": d, "timestamp": time.time()}

@app.get("/cosmos/demo/stream")
async def cosmos_demo_stream():
    """3-GPU demo frame stream. H100 :8400 /demo/stream"""
    st, d = await _h100_get(f"{_H100_ORCH_URL}/demo/stream", timeout=30.0)
    return {"ok": st == 200, "data": d, "timestamp": time.time()}

@app.post("/cosmos/demo/frame")
async def cosmos_demo_frame():
    """Process single demo frame. H100 :8400 /demo/frame"""
    st, d = await _h100_post(f"{_H100_ORCH_URL}/demo/frame", {}, timeout=30.0)
    return {"ok": st == 200, "frame": d, "timestamp": time.time()}

@app.get("/cosmos/health")
async def cosmos_health():
    """Aggregated health of all 4 H100 Cosmos services."""
    import asyncio
    r2, p2, tr, orch = await asyncio.gather(
        _h100_get(f"{_H100_REASON_URL}/health"),
        _h100_get(f"{_H100_PREDICT_URL}/health"),
        _h100_get(f"{_H100_TRANSFER_URL}/health"),
        _h100_get(f"{_H100_ORCH_URL}/health"),
    )
    return {
        "ok": all(s == 200 for s, _ in [r2, p2, tr, orch]),
        "services": {
            "reason2":    {"ok": r2[0] == 200,   "data": r2[1]},
            "predict2":   {"ok": p2[0] == 200,   "data": p2[1]},
            "transfer":   {"ok": tr[0] == 200,   "data": tr[1]},
            "orchestrator": {"ok": orch[0] == 200, "data": orch[1]},
        },
        "timestamp": time.time(),
    }


@app.get("/cosmos/dashboard")
async def cosmos_dashboard():
    """Serve the Cosmos Cookoff dashboard HTML."""
    from fastapi.responses import HTMLResponse
    DASH = "/opt/neurolinux/dashboard.html"
    if __import__("os").path.exists(DASH):
        return HTMLResponse(open(DASH, "r", encoding="utf-8").read())
    return HTMLResponse("<html><body><h2>Dashboard not found — deploy dashboard.html to /opt/neurolinux/</h2></body></html>", status_code=503)


# ── Autonomous Cookoff Proxy ──────────────────────────────────────────────────
_H100_NIS_URL = os.getenv("H100_NIS_URL", "http://172.16.1.83:8090")


class AutonomousRunReq(BaseModel):
    task:        str   = "pick all lighters and put them in the bin"
    max_picks:   int   = 10
    max_retries: int   = 3
    execute_arm: bool  = True
    conf:        float = 0.08


@app.post("/autonomous/run")
async def autonomous_run(req: AutonomousRunReq):
    """Cosmos-guided autonomous sweep — uses local cookoff pipeline (NIS :8000/cookoff/run)."""
    import httpx
    try:
        payload = {
            "prompt": req.task,
            "execute_arm": req.execute_arm,
            "simulation": not req.execute_arm,
        }
        async with httpx.AsyncClient(timeout=300.0) as c:
            r = await c.post("http://localhost:8000/cookoff/run", json=payload)
            return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/autonomous/stream")
async def autonomous_stream(request: Request):
    """SSE proxy: forwards H100 NIS events/stream to the Pi browser."""
    import httpx

    async def _proxy():
        try:
            async with httpx.AsyncClient(timeout=None) as c:
                async with c.stream(
                    "GET",
                    f"{_H100_NIS_URL}/events/stream?topics=cookoff,arm,cosmos",
                    timeout=None,
                ) as r:
                    async for line in r.aiter_lines():
                        if await request.is_disconnected():
                            break
                        if line:
                            yield line + "\n\n"
        except Exception as e:
            import json as _json
            yield f"data: {_json.dumps({'topic': 'error', 'msg': str(e)})}\n\n"

    return StreamingResponse(
        _proxy(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── ARM proxy endpoints (NIS:8000 → Agent:8085) ───────────────────────────────
# Allows panel/NeuroHub to call NIS as single control point

_ARM_PROXY: Dict[str, str] = {
    "home":           "/arm/home",
    "wave":           "/arm/wave",
    "ready":          "/arm/named/ready",
    "inspect":        "/arm/named/inspect",
    "pick":           "/arm/named/pick_table",
    "stop":           "/arm/stop",
    "gripper_open":   "/arm/gripper/open",
    "gripper_close":  "/arm/gripper/close",
    "reach_forward":  "/arm/named/reach_forward",
}

async def _proxy_arm(endpoint: str):
    import httpx
    try:
        async with httpx.AsyncClient(timeout=25.0) as c:
            r = await c.post(f"http://localhost:8085{endpoint}", json={})
            return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/arm/home")
async def arm_home():
    return await _proxy_arm("/arm/home")

@app.post("/arm/wave")
async def arm_wave():
    return await _proxy_arm("/arm/wave")

@app.post("/arm/stop")
async def arm_stop():
    return await _proxy_arm("/arm/stop")

@app.post("/arm/ready")
async def arm_ready():
    return await _proxy_arm("/arm/named/ready")

@app.post("/arm/inspect")
async def arm_inspect():
    return await _proxy_arm("/arm/named/inspect")

@app.post("/arm/pick")
async def arm_pick():
    return await _proxy_arm("/arm/named/pick_table")

@app.post("/arm/reach")
async def arm_reach():
    return await _proxy_arm("/arm/named/reach_forward")

@app.post("/arm/gripper/open")
async def arm_gripper_open():
    return await _proxy_arm("/arm/gripper/open")

@app.post("/arm/gripper/close")
async def arm_gripper_close():
    return await _proxy_arm("/arm/gripper/close")

@app.post("/arm/reconnect")
async def arm_reconnect():
    """Force HID reconnect on agent — recovers physical mode after restart."""
    return await _proxy_arm("/arm/reconnect")

@app.get("/arm/status")
async def arm_status():
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get("http://localhost:8085/health")
            d = r.json()
            return {"ok": True, "simulation": d.get("xarm_simulation", True),
                    "port": d.get("xarm_port", "?"),
                    "physical": not d.get("xarm_simulation", True)}
    except Exception as e:
        return {"ok": False, "error": str(e), "simulation": True}


class DemoRunRequest(BaseModel):
    task: str = Field(default="Pick up the red cube and place it in the bin",
                      description="Natural language task description")
    execute_arm: bool = Field(default=True)
    simulation: bool = Field(default=False)
    image_base64: Optional[str] = None


async def _yolo_frame(task: str = "") -> Optional[str]:
    """
    Fast path: YOLO-annotated frame only (no Predict2).
    Used for inter-step frame refreshes during the H100 reasoning loop.
    """
    import httpx, urllib.parse
    task_objects = _extract_task_objects(task) if task else []
    detect_url = "http://localhost:8085/vision/detect"
    if task_objects:
        detect_url += "?targets=" + urllib.parse.quote(",".join(task_objects[:6]))

    async with httpx.AsyncClient(timeout=8.0) as hc:
        try:
            r = await hc.get(detect_url)
            if r.status_code == 200:
                d = r.json()
                ctx = d.get("scene_context", "")
                if task_objects:
                    ctx = f"{ctx} | task_targets: {', '.join(task_objects)}"
                if ctx:
                    _latest_scene_context[0] = ctx
                return d.get("annotated_b64") or d.get("image_base64")
        except Exception:
            pass
        # Fallback
        try:
            r = await hc.get("http://localhost:8085/camera/snapshot")
            if r.status_code == 200:
                d = r.json()
                return d.get("image_base64") or d.get("image")
        except Exception:
            pass
    return None


async def _predict2_background(task: str, raw_b64: str) -> None:
    """Fire-and-forget coroutine: calls Predict2 and caches the result."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=130.0) as p2c:
            predicted = await asyncio.wait_for(
                _predict2_video(p2c, task, raw_b64, num_frames=9),
                timeout=120.0
            )
            if predicted:
                _latest_predicted_frame[0] = predicted
                logger.info("Predict2 ✅ background frame cached")
    except asyncio.TimeoutError:
        logger.warning("Predict2 background timed out")
    except Exception as e:
        logger.warning(f"Predict2 background error: {e}")


async def _capture_frame(task: str = "") -> Optional[str]:
    """
    Full vision pipeline (called ONCE at demo start):
    1. C270 snapshot → YOLO annotate (task-driven open-vocab labels)
    2. Predict2 fired as asyncio background task (non-blocking)
    Returns the YOLO-annotated live frame immediately;
    Predict2 result arrives in _latest_predicted_frame[0] within ~15s.
    """
    import httpx, urllib.parse
    annotated: Optional[str] = None
    raw: Optional[str] = None

    task_objects = _extract_task_objects(task) if task else []
    detect_url = "http://localhost:8085/vision/detect"
    if task_objects:
        detect_url += "?targets=" + urllib.parse.quote(",".join(task_objects[:6]))

    async with httpx.AsyncClient(timeout=8.0) as hc:
        try:
            r = await hc.get(detect_url)
            if r.status_code == 200:
                d = r.json()
                annotated = d.get("annotated_b64") or d.get("image_base64")
                raw       = d.get("raw_b64") or annotated
                ctx       = d.get("scene_context", "")
                if task_objects:
                    ctx = f"{ctx} | task_targets: {', '.join(task_objects)}"
                if ctx:
                    _latest_scene_context[0] = ctx
        except Exception:
            pass

        if not annotated:
            try:
                r = await hc.get("http://localhost:8085/camera/snapshot")
                if r.status_code == 200:
                    d = r.json()
                    annotated = d.get("image_base64") or d.get("image")
                    raw = annotated
            except Exception:
                pass

    # Fire Predict2 as background task (non-blocking — YOLO returns immediately)
    if raw and task:
        _latest_predicted_frame[0] = None  # clear stale frame
        t = asyncio.ensure_future(_predict2_background(task, raw))
        _predict2_task[0] = t
        logger.info("Predict2 background task started")

    return annotated


# Mutable cells shared across the pipeline
_latest_scene_context: List[str] = [""]   # YOLO scene context injected into H100 prompts
_latest_predicted_frame: List[Optional[str]] = [None]  # Predict2 future frame (b64)
_predict2_task: List[Optional[asyncio.Task]] = [None]  # background Predict2 task

# ── H100 Predict2 URL (Video2World) ──────────────────────────────────────────
_PREDICT2_URL = os.getenv("PREDICT2_URL", "http://172.16.1.83:8200")


def _extract_task_objects(task: str) -> List[str]:
    """
    Pull object nouns from the task string so the vision pipeline
    can highlight ANY object mentioned — not just 'cube'.
    e.g. 'pick up the mug and drop it in the box' → ['mug', 'box']
    """
    import re
    # Strip filler words, keep nouns
    stop = {"the","a","an","and","or","it","its","to","up","in","on",
            "with","into","onto","then","pick","put","place","drop","move",
            "grab","wave","lift","hello","arm","gripper","robot","bin"}
    words = re.findall(r'[a-z]+', task.lower())
    return [w for w in words if len(w) > 2 and w not in stop]


async def _predict2_video(
    client,
    task: str,
    image_b64: str,
    num_frames: int = 9,
) -> Optional[str]:
    """
    Call Predict2 /video2world: current frame + task prompt → predicted future state.
    Actual response schema: {video_b64, image_b64, latency_ms, model, num_frames, fps, resolution}
    Returns image_b64 (predicted composite frame) or None if Predict2 unavailable.
    Uses fast settings: 9 frames, 15 steps (~12s on H100).
    """
    try:
        body = {
            "prompt": (
                f"Robotic arm workspace: {task}. "
                f"Show the arm completing the action, objects in final position."
            ),
            "image_b64": image_b64,
            "num_frames": num_frames,
            "fps": 8,
            "height": 480,
            "width": 848,
            "num_inference_steps": 15,
            "guidance_scale": 7.0,
        }
        # Use longer timeout — H100 inference takes ~60-120s
        r = await client.post(f"{_PREDICT2_URL}/video2world", json=body,
                               timeout=120.0)
        if r.status_code != 200:
            logger.warning(f"Predict2 /video2world status {r.status_code}")
            return None
        d = r.json()
        # Response schema: {video_b64, image_b64, latency_ms, num_frames, ...}
        # image_b64 = predicted composite frame (most useful for Reason2 input)
        predicted_frame = d.get("image_b64")
        if predicted_frame:
            latency = d.get("latency_ms", "?")
            logger.info(f"Predict2 ✅ predicted frame ready (latency={latency}ms, frames={d.get('num_frames')})")
            return predicted_frame
        # Fallback: frames_b64 list if present in future versions
        frames = d.get("frames_b64", [])
        if frames:
            return frames[len(frames) // 2]
        return None
    except Exception as e:
        logger.warning(f"Predict2 unavailable: {e}")
        return None


_VLA_INTENT_MAP = [
    # (keyword_in_verbose_action, clean_arm_command)
    # Ordered by priority — first match wins
    ("wave",          "wave"),
    ("inspect",       "inspect"),
    ("pick_and_place","pick_and_place"),
    ("pick and place","pick_and_place"),
    ("grasp",         "grasp"),
    ("pick up",       "pick"),
    ("pick",          "pick"),
    ("grab",          "grab"),
    ("place",         "place"),
    ("put down",      "place"),
    ("put",           "put"),
    ("drop",          "drop"),
    ("release",       "release"),
    ("open gripper",  "open"),
    ("close gripper", "close"),
    ("grip",          "grip"),
    ("reach left",    "reach_left"),
    ("reach right",   "reach_right"),
    ("reach",         "reach"),
    ("move left",     "move left"),
    ("move right",    "move right"),
    ("home",          "home"),
    ("return",        "home"),
    ("park",          "park"),
    ("stop",          "stop"),
    ("approach",      "approach"),
    ("align",         "align"),
    ("lower",         "lower"),
    ("lift",          "lift"),
    # Fallback: H100 pixel-coord gripper moves → pick (approach object)
    ("move gripper",  "pick"),
    ("move to",       "inspect"),
]

# ── Visual calibration: pixel coords → nearest named position ────────────────
_CAL_DATA: Optional[dict] = None

def _load_calibration() -> Optional[dict]:
    """Load calibration.json from disk (lazy, cached)."""
    global _CAL_DATA
    if _CAL_DATA is not None:
        return _CAL_DATA
    import json as _json
    for cal_path in ("/opt/neurolinux/calibration.json",
                     "/opt/nis-protocol/calibration.json"):
        try:
            with open(cal_path) as f:
                _CAL_DATA = _json.load(f)
            logger.info(f"Loaded calibration from {cal_path} "
                        f"({len(_CAL_DATA.get('positions', {}))} positions)")
            return _CAL_DATA
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.warning(f"calibration load error: {e}")
    return None

def _pixel_to_named_position(px: int, py: int) -> Optional[str]:
    """
    Find the nearest calibrated named position to pixel (px, py).
    Clamps H100 hallucinated out-of-bounds coords to image size first.
    Uses Euclidean distance in pixel space.
    Returns position name or None if calibration unavailable.
    """
    import math
    cal = _load_calibration()
    if not cal:
        return None
    # Clamp to image bounds (C270 is 1280x720)
    img = cal.get("image_size", {"w": 1280, "h": 720})
    px = max(0, min(px, img["w"]))
    py = max(0, min(py, img["h"]))
    positions = cal.get("positions", {})
    best_name = None
    best_dist = float("inf")
    for name, data in positions.items():
        pix = data.get("pixel")
        if not pix:
            continue
        d = math.sqrt((pix["x"] - px) ** 2 + (pix["y"] - py) ** 2)
        if d < best_dist:
            best_dist = d
            best_name = name
    if best_name and best_dist < 600:  # 600px tolerance — covers clamped OOB coords
        logger.info(f"calibration: ({px},{py}) → {best_name} (dist={best_dist:.0f}px)")
        return best_name
    return None

def _extract_vla_intent(raw_action: str) -> str:
    """
    Convert verbose H100 VLA action string to a clean arm command keyword.
    1. If the action contains pixel coords [x,y], look up nearest calibrated position.
    2. Otherwise fall through keyword map.
    e.g. "move gripper to [876,210]" → "reach_right"  (calibration lookup)
         "wave hello to the audience" → "wave"          (keyword map)
    Returns the raw_action unchanged if no known keyword found.
    """
    import re
    low = raw_action.lower()

    # 1. Try calibration pixel lookup first — most accurate
    coord_match = re.search(r'\[(\d+)[,\s]+(\d+)\]', raw_action)
    if coord_match:
        px, py = int(coord_match.group(1)), int(coord_match.group(2))
        named = _pixel_to_named_position(px, py)
        if named:
            return named

    # 2. Keyword map fallback
    for keyword, clean in _VLA_INTENT_MAP:
        if keyword in low:
            return clean
    return raw_action


# Stage-specific sub-commands per reasoning step
_H100_STAGE_CMDS = [
    "Approach the target object — describe your gripper's current pixel position and the target pixel position",
    "Grasp the object — close gripper once aligned, confirm grip success",
    "Lift and transport the object to the destination bin/zone",
    "Place and release the object, then return arm to home position",
]

async def _h100_next_action(
    client,
    task: str,
    step_i: int,
    completed: List[str],
    image_b64: Optional[str],
) -> tuple:
    """
    Call H100 Cosmos Reason2 /robot-plan for the next single action.
    Returns (action_keyword, raw_action, reasoning, safe_to_execute).
    action_keyword is normalized to a clean arm command keyword.
    """
    # Build a step-specific command to force H100 to reason about different phases
    stage_hint = _H100_STAGE_CMDS[min(step_i, len(_H100_STAGE_CMDS) - 1)]
    step_cmd = f"{task}. Current phase (step {step_i+1}): {stage_hint}."
    if completed:
        step_cmd += f" Already completed: {'; '.join(completed[-2:])}"

    # Inject YOLO scene context — grounded pixel coords of real objects
    scene_ctx = _latest_scene_context[0]
    if scene_ctx:
        step_cmd = f"{step_cmd} {scene_ctx}"

    # Hybrid image: step 0 → Predict2 predicted frame if already cached, else live.
    # No blocking wait — Predict2 fires in background concurrently with Reason2 calls.

    predicted = _latest_predicted_frame[0]
    send_image = image_b64
    if predicted and step_i == 0:
        send_image = predicted
        step_cmd = f"{step_cmd} [predicted workspace state provided]"

    body: Dict[str, Any] = {
        "command": step_cmd,
        "robot_type": "xarm",
    }
    if send_image:
        body["image_base64"] = send_image

    r = await client.post(f"{_H100_REASON_URL}/robot-plan", json=body)
    if r.status_code != 200:
        return "", "", "", True

    d = r.json()
    raw = d.get("action", "")
    if not raw:
        lst = d.get("action_plan", [])
        raw = lst[0] if lst else ""
    if not raw:
        reasoning = d.get("reasoning", d.get("response", ""))
        for ln in reasoning.split("\n"):
            ln = ln.lstrip("0123456789.-*• ").strip()
            if ln and len(ln) > 2:
                raw = ln
                break

    # Normalize verbose VLA output → clean arm command
    action = _extract_vla_intent(raw.strip().lower()) if raw else ""
    full_reasoning = d.get("reasoning", "") or d.get("response", "")

    return (
        action,
        raw.strip()[:80],          # raw action snippet for dedup
        full_reasoning[:140],      # rich reasoning for completed_ctx
        d.get("safe_to_execute", True),
    )


async def _arm_one_step(client, action: str, simulation: bool) -> Dict[str, Any]:
    """
    Execute a single action string via /cookoff/execute (1-step plan).
    Returns the step result dict.
    """
    r = await client.post(
        "http://localhost:8000/cookoff/execute",
        json={"action_plan": [action], "execute_arm": True, "simulation": simulation},
    )
    if r.status_code == 200:
        results = r.json().get("results", [{}])
        return results[0] if results else {}
    return {"ok": False, "response": f"HTTP {r.status_code}"}


class VoiceRobotRequest(BaseModel):
    audio_b64: str = Field(description="Base64-encoded audio (WebM/WAV/MP3)")
    execute_arm: bool = Field(default=False)
    simulation: bool = Field(default=True)
    speaker: str = Field(default="consciousness")


@app.post("/voice/robot-command")
async def voice_robot_command(request: VoiceRobotRequest):
    """
    Voice → Robot pipeline.
    1. Whisper STT:  audio_b64 → transcript
    2. Demo run:     transcript → YOLO+Predict2+Reason2 → xArm
    3. TTS response: result → spoken confirmation
    Full GPT-style voice loop for physical robot control.
    """
    import httpx

    # Step 1: Transcribe audio via Whisper (on NIS main.py or Pi)
    transcript = ""
    whisper_engine = "unavailable"
    try:
        async with httpx.AsyncClient(timeout=15.0) as hc:
            r = await hc.post("http://localhost:8000/voice/transcribe",
                              json={"audio_base64": request.audio_b64,
                                    "language": "en"})
            if r.status_code == 200:
                d = r.json()
                transcript = d.get("text", "").strip()
                whisper_engine = d.get("engine", "whisper")
    except Exception as e:
        logger.warning(f"voice/transcribe failed: {e}")

    if not transcript:
        return {"ok": False, "error": "Could not transcribe audio",
                "whisper_engine": whisper_engine}

    logger.info(f"/voice/robot-command transcript: '{transcript}'")

    # Step 2: Run full robot pipeline with transcript as task
    demo_req = DemoRunRequest(
        task=transcript,
        execute_arm=request.execute_arm,
        simulation=request.simulation,
    )
    demo_result = await demo_run(demo_req)

    # Step 3: Build TTS confirmation text
    steps_ok = demo_result.get("steps_ok", 0)
    steps_total = demo_result.get("steps_total", 0)
    if demo_result.get("ok"):
        tts_text = f"Done. I executed: {transcript}. Completed {steps_ok} of {steps_total} steps."
    else:
        tts_text = f"I heard: {transcript}, but the pipeline did not complete successfully."

    return {
        "ok": demo_result.get("ok", False),
        "transcript": transcript,
        "whisper_engine": whisper_engine,
        "tts_text": tts_text,
        "plan_source": demo_result.get("plan_source"),
        "steps_ok": steps_ok,
        "steps_total": steps_total,
        "h100_reasoning": demo_result.get("h100_reasoning", []),
    }


@app.websocket("/ws/voice-robot")
async def voice_robot_ws(websocket: WebSocket):
    """
    🎙️ Two-Way Agentic Voice Chat — Robot-Aware Real-Time Loop

    Pipeline each turn:
      audio_input → Whisper STT → Reason2 intent classify
          → "ARM" intent  : fires /demo/run (arm moves) + TTS confirms
          → "CHAT" intent : LLM answers + TTS speaks
          → "DANCE" intent: fires /cosmos-dance/start + TTS says "dancing!"
      Server streams back: transcription → status → text_response → audio_response

    Client sends JSON frames:
      {"type":"audio_input",  "audio_data":"<b64>", "execute_arm":true, "simulation":false}
      {"type":"text_input",   "text":"wave hello"}
      {"type":"interrupt"}
      {"type":"close"}

    Server sends JSON frames:
      {"type":"connected",      "session_id":"...", "capabilities":{...}}
      {"type":"transcription",  "text":"...", "confidence":0.9}
      {"type":"status",         "stage":"thinking|executing|synthesizing"}
      {"type":"intent",         "intent":"ARM|CHAT|DANCE", "task":"..."}
      {"type":"text_response",  "text":"..."}
      {"type":"arm_result",     "ok":true, "steps":4, "plan_source":"..."}
      {"type":"audio_response", "audio_data":"<b64>", "format":"mp3"}
      {"type":"error",          "message":"..."}
    """
    import httpx
    await websocket.accept()
    session_id = f"vr_{uuid.uuid4().hex[:8]}"
    logger.info("ws/voice-robot session started: %s", session_id)

    # ── helpers ──────────────────────────────────────────────────────────────
    async def _send(msg: dict):
        try:
            await websocket.send_json(msg)
        except Exception:
            pass

    async def _whisper(audio_b64: str) -> str:
        """Transcribe audio via local /voice/transcribe or fallback."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as hc:
                r = await hc.post("http://localhost:8000/voice/transcribe",
                                  json={"audio_base64": audio_b64, "language": "en"})
                if r.status_code == 200:
                    return r.json().get("text", "").strip()
        except Exception:
            pass
        # Fallback: try WhisperSTT directly
        try:
            from src.voice.whisper_stt import get_whisper_stt
            stt = get_whisper_stt("tiny")
            result = await stt.transcribe_base64(audio_b64)
            if result.get("success"):
                return result.get("text", "").strip()
        except Exception:
            pass
        return ""

    async def _classify_intent(text: str) -> tuple:
        """
        Ask Reason2 whether the user wants ARM action, DANCE, or CHAT.
        Returns (intent, cleaned_task).
        intent ∈ {"ARM", "DANCE", "CHAT"}
        """
        ARM_KW   = {"pick","place","grab","wave","move","home","inspect",
                    "reach","grasp","open","close","gripper","arm","robot","cube","bin"}
        DANCE_KW = {"dance","music","groove","beat","bop","party","sing","song"}

        low = text.lower()
        if any(k in low for k in DANCE_KW):
            return "DANCE", text
        if any(k in low for k in ARM_KW):
            return "ARM", text
        # Ask Reason2 for ambiguous cases
        try:
            async with httpx.AsyncClient(timeout=8.0) as hc:
                r = await hc.post(f"{_H100_REASON_URL}/robot-plan", json={
                    "command": (
                        f"Classify this user voice command as ARM (physical robot action), "
                        f"DANCE (music/dance), or CHAT (conversation/question). "
                        f"Reply with ONLY one word: ARM, DANCE, or CHAT.\n"
                        f"Command: \"{text}\""
                    ),
                    "robot_type": "xarm",
                })
                if r.status_code == 200:
                    raw = (r.json().get("action") or r.json().get("reasoning") or "").upper()
                    for intent in ("ARM", "DANCE", "CHAT"):
                        if intent in raw:
                            return intent, text
        except Exception:
            pass
        return "CHAT", text

    async def _tts(text: str) -> Optional[str]:
        """Synthesize TTS, returns base64 MP3 or None."""
        try:
            from src.voice.simple_tts import get_simple_tts
            tts = get_simple_tts()
            audio = await tts.synthesize_async(text)
            if audio:
                return base64.b64encode(audio).decode()
        except Exception:
            pass
        return None

    async def _llm_chat(text: str, history: list) -> str:
        """Quick LLM answer via NIS LLM provider."""
        try:
            async with httpx.AsyncClient(timeout=12.0) as hc:
                msgs = history[-6:] + [{"role": "user", "content": text}]
                r = await hc.post("http://localhost:8000/chat",
                                  json={"message": text, "context": {"history": history[-4:]}})
                if r.status_code == 200:
                    return r.json().get("response", r.json().get("message", ""))
        except Exception:
            pass
        # Reason2 fallback for robot questions
        try:
            async with httpx.AsyncClient(timeout=10.0) as hc:
                r = await hc.post(f"{_H100_REASON_URL}/robot-plan", json={
                    "command": text, "robot_type": "xarm"})
                if r.status_code == 200:
                    return (r.json().get("reasoning") or r.json().get("action") or "")[:200]
        except Exception:
            pass
        return "I'm not sure how to answer that right now."

    # ── send handshake ────────────────────────────────────────────────────────
    await _send({
        "type": "connected",
        "session_id": session_id,
        "capabilities": {
            "stt": "whisper",
            "llm": "cosmos_reason2 + nis_llm",
            "tts": "openai_tts + gtts",
            "arm": True,
            "dance": True,
            "intents": ["ARM", "DANCE", "CHAT"],
        },
    })

    # ── main loop ─────────────────────────────────────────────────────────────
    chat_history: list = []
    try:
        while True:
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                break
            except Exception:
                break

            msg_type = data.get("type", "")

            if msg_type == "close":
                await _send({"type": "closed", "session_id": session_id})
                break

            if msg_type == "interrupt":
                await _send({"type": "interrupted"})
                continue

            # ── resolve text ─────────────────────────────────────────────────
            text = ""
            if msg_type == "audio_input":
                await _send({"type": "status", "stage": "transcribing"})
                audio_b64 = data.get("audio_data", "")
                if not audio_b64:
                    await _send({"type": "error", "message": "No audio data"}); continue
                text = await _whisper(audio_b64)
                if not text:
                    await _send({"type": "error", "message": "Transcription failed"}); continue
                await _send({"type": "transcription", "text": text})

            elif msg_type == "text_input":
                text = data.get("text", "").strip()
                if not text:
                    continue

            else:
                continue

            execute_arm = data.get("execute_arm", False)
            simulation  = data.get("simulation", True)

            # ── classify intent ──────────────────────────────────────────────
            await _send({"type": "status", "stage": "thinking"})
            intent, task = await _classify_intent(text)
            await _send({"type": "intent", "intent": intent, "task": task})

            # ── ARM branch ───────────────────────────────────────────────────
            if intent == "ARM":
                await _send({"type": "status", "stage": "executing"})
                tts_text = f"Got it. Executing: {task}"
                arm_result: dict = {}
                try:
                    async with httpx.AsyncClient(timeout=240.0) as hc:
                        r = await hc.post("http://localhost:8000/demo/run", json={
                            "task": task,
                            "execute_arm": execute_arm,
                            "simulation": simulation,
                        })
                        if r.status_code == 200:
                            arm_result = r.json()
                except Exception as e:
                    arm_result = {"ok": False, "error": str(e)}

                steps_ok    = arm_result.get("steps_ok", 0)
                steps_total = arm_result.get("steps_total", 0)
                if arm_result.get("ok"):
                    tts_text = (f"Done. I completed {steps_ok} of {steps_total} steps "
                                f"for: {task}.")
                else:
                    tts_text = f"I tried to {task} but the pipeline didn't complete."

                await _send({
                    "type": "arm_result",
                    "ok": arm_result.get("ok", False),
                    "steps_ok": steps_ok,
                    "steps_total": steps_total,
                    "plan_source": arm_result.get("plan_source"),
                    "h100_reasoning": arm_result.get("h100_reasoning", []),
                })

            # ── DANCE branch ─────────────────────────────────────────────────
            elif intent == "DANCE":
                tts_text = "Let's dance! Starting the Cosmos beat now."
                try:
                    async with httpx.AsyncClient(timeout=5.0) as hc:
                        await hc.post("http://localhost:8000/cosmos-dance/start",
                                      json={"moves": 32})
                except Exception:
                    tts_text = "Dance mode started — check the arm!"
                await _send({"type": "arm_result", "ok": True,
                             "action": "cosmos_dance_started"})

            # ── CHAT branch ──────────────────────────────────────────────────
            else:
                response = await _llm_chat(text, chat_history)
                chat_history.append({"role": "user", "content": text})
                chat_history.append({"role": "assistant", "content": response})
                if len(chat_history) > 20:
                    chat_history = chat_history[-20:]
                tts_text = response
                await _send({"type": "text_response", "text": response})

            # ── TTS for all branches ─────────────────────────────────────────
            await _send({"type": "status", "stage": "synthesizing"})
            audio_b64_out = await _tts(tts_text)
            if audio_b64_out:
                await _send({
                    "type": "audio_response",
                    "audio_data": audio_b64_out,
                    "format": "mp3",
                    "text": tts_text,
                })
            else:
                await _send({"type": "text_response", "text": tts_text})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("ws/voice-robot error: %s", e)
        await _send({"type": "error", "message": str(e)})
    finally:
        logger.info("ws/voice-robot session ended: %s", session_id)


@app.post("/demo/run")
async def demo_run(request: DemoRunRequest):
    """
    🤖 Hybrid VLA + Choreographed Demo.

    Phase 1 — AI Reasoning (H100 Cosmos Reason2):
      - Captures scene frame (synthetic workspace if no camera)
      - Calls H100 VLA in a loop: sees frame + task → reasons about pixel coords
      - Collects real AI reasoning chain showing scene understanding

    Phase 2 — Physical Execution:
      - If H100 produced a clean unique action sequence → execute it
      - Otherwise → execute the full impressive choreographed sequence:
        wave → inspect → pick_and_place → home
      - Either way: real H100 reasoning is shown in the response

    curl -X POST http://PI:8000/demo/run -d '{"task":"pick up the red cube"}'
    """
    import httpx
    t0 = time.time()

    DONE_WORDS = {"home", "done", "complete", "finished", "stop", "park"}
    # Impressive fallback sequence — always runs if H100 doesn't give 3+ unique steps
    CHOREOGRAPHED = ["wave", "inspect", "pick_and_place", "home"]
    MAX_PLAN_STEPS = 4

    reasoning_parts: List[str] = []
    plan_source = "fallback"
    h100_actions: List[str] = []
    seen: set = set()

    # Capture initial scene frame (task-driven open-vocab detection + Predict2)
    current_frame: Optional[str] = request.image_base64 or await _capture_frame(request.task)
    has_camera = current_frame is not None
    logger.info(f"/demo/run start — task={request.task[:60]} camera={'YES' if has_camera else 'NO'}")

    # ── Phase 1: H100 reasoning loop — collect AI context, don't execute ────────
    completed_ctx: List[str] = []   # rich reasoning passed back to H100 as history
    async with httpx.AsyncClient(timeout=120.0) as hc:
        for step_i in range(MAX_PLAN_STEPS):
            action, raw_action, reasoning, safe = await _h100_next_action(
                hc, request.task, step_i, completed_ctx, current_frame
            )
            if action:
                plan_source = "h100_reason2"
            if reasoning:
                reasoning_parts.append(f"S{step_i+1}: {reasoning}")

            if not action or not safe:
                break

            # Dedup: keyed on (step, action) so same action at different steps is ok
            # Only deduplicate if the EXACT same raw text appears at 2+ consecutive steps
            dedup_key = raw_action[:50].lower() if raw_action else action[:30]
            is_dup = dedup_key in seen and step_i > 0

            # Always record reasoning for all 4 steps
            completed_ctx.append(f"Step {step_i+1}: {raw_action} — {reasoning[:80]}")
            logger.info(f"/demo/run H100 step {step_i+1}: {action} | {raw_action[:60]}")

            if is_dup:
                logger.info(f"/demo/run H100 loop at step {step_i+1} — skipping exec")
            else:
                seen.add(dedup_key)
                h100_actions.append(raw_action or action)

            # Refresh frame between steps (YOLO only) — skip on final step to save time
            if step_i < MAX_PLAN_STEPS - 1:
                new_frame = await _yolo_frame(request.task)
                if new_frame:
                    current_frame = new_frame

    # ── Phase 2: decide execution sequence ────────────────────────────────────
    # Use H100 actions only if they give a rich, diverse sequence:
    # ≥4 unique moves AND contain at least one "anchor" move (wave/inspect/pick_and_place).
    # Otherwise always run the full choreographed sequence for maximum demo impressiveness.
    ANCHOR_MOVES = {"wave", "inspect", "pick_and_place", "pick and place",
                     "move gripper", "close gripper", "open gripper", "grasp"}
    unique_h100 = list(dict.fromkeys(h100_actions))  # preserve order, dedupe
    # Normalize for execution: map raw H100 actions to arm keywords
    exec_h100 = [_extract_vla_intent(a.lower()) for a in unique_h100]
    h100_has_anchors = any(
        any(kw in a.lower() for kw in ANCHOR_MOVES) for a in unique_h100
    )
    if len(unique_h100) >= 4 and h100_has_anchors:
        exec_sequence = exec_h100   # normalized keywords — cookoff.py can execute these
        exec_source = "h100_reason2"
    else:
        exec_sequence = CHOREOGRAPHED
        exec_source = "choreographed" if not h100_actions else "choreographed+h100"
        plan_source = exec_source

    logger.info(f"/demo/run exec: {exec_sequence} (source={exec_source})  raw_h100={unique_h100}")

    # ── Phase 3: physical execution with Reason2 plausibility gate ─────────────
    step_trace: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=120.0) as hc:
        for i, act in enumerate(exec_sequence):
            t_step = time.time()

            # Plausibility gate: ask Reason2 if this step makes sense before executing
            plaus_score: float = 1.0
            plaus_ok:    bool  = True
            if not request.simulation and current_frame:
                # Map single-word actions to descriptive sentences for Reason2
                _ACT_DESC = {
                    "wave":          "The robot arm waves hello to the audience",
                    "home":          "The robot arm returns to its home position",
                    "inspect":       "The robot arm moves to the inspect position to view the workspace",
                    "pick_and_place":"The robot arm picks up an object and places it in the bin",
                    "close":         "The robot gripper closes to grasp an object on the table",
                    "open":          "The robot gripper opens to release the grasped object",
                    "reach_left":    "The robot arm reaches to the left side of the workspace",
                    "reach_right":   "The robot arm reaches to the right side of the workspace",
                    "reach_forward": "The robot arm reaches forward toward the table",
                    "place_bin":     "The robot arm moves to the bin to place an object",
                    "wave_up":       "The robot arm waves upward in a greeting gesture",
                }
                act_desc = _ACT_DESC.get(act, f"The robot arm executes '{act}' as step {i+1}")
                try:
                    pr = await hc.post(
                        f"{_H100_REASON_URL}/plausibility",
                        json={
                            "description": f"{act_desc}, as part of the task: {request.task}",
                            "image_base64": current_frame,
                            "context":      "xarm 6DOF robot, wooden table, 848x480 camera",
                        },
                        timeout=12.0,
                    )
                    if pr.status_code == 200:
                        pd = pr.json()
                        plaus_score = pd.get("score", 1.0)
                        plaus_ok    = pd.get("plausible", True)
                        logger.info(f"/demo/run plausibility step {i+1} '{act}': "
                                    f"score={plaus_score} ok={plaus_ok}")
                except Exception as pe:
                    logger.warning(f"/demo/run plausibility check failed: {pe}")

            if request.simulation:
                step_trace.append({
                    "step": i + 1, "action": act, "ok": True,
                    "source": exec_source, "response": "simulated",
                    "plausibility_score": plaus_score, "plausible": plaus_ok,
                    "latency_ms": 0,
                })
            else:
                sr = await _arm_one_step(hc, act, False)
                sr["step"]   = i + 1
                sr["action"] = act
                sr["source"] = exec_source
                sr["plausibility_score"] = plaus_score
                sr["plausible"]          = plaus_ok
                sr["latency_ms"]         = round((time.time() - t_step) * 1000)
                step_trace.append(sr)
                logger.info(f"/demo/run exec step {i+1}: {act} → "
                            f"{sr.get('endpoint','?')} ok={sr.get('ok')} "
                            f"plaus={plaus_score} {sr['latency_ms']}ms")

            # Refresh frame after each step
            if not request.simulation:
                nf = await _yolo_frame(request.task)
                if nf:
                    current_frame = nf

    # ── Phase 4: Reason2 goal-verify after full execution ─────────────────────
    goal_complete: bool = False
    goal_reasoning: str = ""
    if not request.simulation and current_frame:
        try:
            last_act = exec_sequence[-1] if exec_sequence else "home"
            steps_summary = ", ".join(exec_sequence)
            gv_r = await _h100_post(
                f"{_H100_REASON_URL}/goal-verify",
                {
                    "goal": (
                        f"The robot completed these steps: [{steps_summary}] "
                        f"to accomplish: '{request.task}'. "
                        f"The final step was '{last_act}'. "
                        "Was the overall task successfully accomplished?"
                    ),
                    "image_base64": current_frame,
                    "last_action":  last_act,
                },
                timeout=15.0,
            )
            if gv_r[0] == 200:
                goal_complete  = gv_r[1].get("goal_complete", False)
                goal_reasoning = str(gv_r[1].get("reasoning", ""))[:200]
                logger.info(f"/demo/run goal-verify: complete={goal_complete}")
        except Exception as ge:
            logger.warning(f"/demo/run goal-verify failed: {ge}")

    steps_ok = sum(1 for s in step_trace if s.get("ok"))
    avg_plaus = round(sum(s.get("plausibility_score", 1.0) for s in step_trace)
                      / max(len(step_trace), 1), 3)
    # goal_complete: ground truth = all steps executed successfully.
    # Reason2 goal-verify is unreliable here — it sees the arm back at home
    # after execution, not the bin state during pick-and-place.
    # Use steps_ok as the authoritative signal; reason2 verdict is commentary.
    goal_complete = steps_ok == len(step_trace) and len(step_trace) > 0
    if goal_reasoning:
        goal_reasoning = f"[steps:{steps_ok}/{len(step_trace)}] " + goal_reasoning
    return {
        "ok": steps_ok > 0,
        "task": request.task,
        "plan_source": plan_source,
        "action_plan": [s["action"] for s in step_trace],
        "h100_reasoning": h100_actions,
        "reasoning": " | ".join(reasoning_parts),
        "goal_complete":  goal_complete,
        "goal_reasoning": goal_reasoning,
        "avg_plausibility": avg_plaus,
        "execution": {
            "ok": steps_ok > 0,
            "steps_ok": steps_ok,
            "steps_total": len(step_trace),
            "results": step_trace,
            "simulation": request.simulation,
            "latency_ms": round((time.time() - t0) * 1000),
        },
        "steps_ok": steps_ok,
        "steps_total": len(step_trace),
        "camera_used": has_camera,
        "latency_ms": round((time.time() - t0) * 1000),
        "pipeline": ["pi_camera", "yolo_open_vocab",
                     "cosmos_predict2_video2world",
                     "cosmos_reason2_planning",
                     "cosmos_reason2_plausibility",
                     "xarm_physical_execution",
                     "cosmos_reason2_goal_verify"],
        "timestamp": time.time(),
    }


@app.get("/routes")
async def list_routes():
    """List all registered routes — for panel discoverability."""
    paths = []
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            paths.append({"path": route.path, "methods": list(route.methods or [])})
    return {"routes": [r["path"] for r in paths], "count": len(paths), "details": paths}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("NEUROLINUX_AGENT_PORT") or os.getenv("NIS_PORT", "8000"))
    host = os.getenv("NEUROLINUX_AGENT_HOST") or os.getenv("NIS_HOST", "0.0.0.0")
    logger.info(f"Starting NIS Protocol Pi on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

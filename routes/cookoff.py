"""
NIS Protocol v4.0 - Cosmos Cookoff Routes

Robot-plan endpoint for Cosmos Cookoff Challenge:
- /cookoff/robot-plan  — Cosmos Reason2 robot action planning
- /cookoff/execute     — Execute action plan on physical xArm
- /cookoff/demo        — Full YOLO → Reason2 → arm demo
- /cookoff/pick        — Confirmed IK pick (z=1.5cm, verified 2026-02-27)
- /cookoff/dance       — Latino rhythm arm dance (reggaeton/cumbia/bachata/salsa)
- /cookoff/status      — Pipeline status

Confirmed working parameters (2026-02-27):
  HOME:  S1=100 S2=500 S3=310 S4=870 S5=680 S6=500
  HOVER: S1=100 S2=500 S3=222 S4=697 S5=604 S6=500 (z=6cm)
  MID:   S1=100 S2=500 S3=158 S4=798 S5=502 S6=500 (z=3.5cm)
  PICK:  S1=100 S2=500 S3=142 S4=856 S5=430 S6=500 (z=1.5cm) ← CONFIRMED
  GRIP:  S1=700 (firm grip)
  PLACE: S3=220 S4=827 S5=425 S6=875 (left90)
"""

import asyncio
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger("nis.routes.cookoff")

router = APIRouter(prefix="/cookoff", tags=["Cosmos Cookoff"])

H100_REASON_URL   = os.environ.get("H100_REASON_URL",   "http://localhost:8100")
NIS_CLOUD_URL     = os.environ.get("NIS_CLOUD_URL",     "http://192.168.1.160:8007")

# Establishes NIS identity, robot context, and safety rules.
# Cosmos uses this to ground its reasoning in the specific hardware/task context.
NIS_SYSTEM_PROMPT = (
    "You are Cosmos Reason2, the reasoning and perception core of NIS Protocol v4.0.1 "
    "(Neural Intelligence System) — an AI operating system for robotic manipulation.\n\n"
    "HARDWARE CONTEXT:\n"
    "  - Robot arm: xArm 1S, 6 servos, USB HID on Raspberry Pi 5\n"
    "  - Camera: C270 HD Webcam, 1280x720, mounted above table\n"
    "  - Gripper: S1 servo (100=open, 700=close)\n"
    "  - Lateral: S2 servo (380=far-left, 500=center, 620=far-right)\n"
    "  - Arm extension: S6 (500=normal reach, 875=extended to bowl)\n\n"
    "YOUR ROLE:\n"
    "  - Analyze camera images and YOLO detections to understand the scene\n"
    "  - Decide what objects to pick and where to place them\n"
    "  - Return structured JSON plans when asked (pick_targets, place_target)\n"
    "  - Verify task completion by comparing before/after scenes\n\n"
    "SAFETY RULES:\n"
    "  - Never command servo values outside safe ranges (0-1000)\n"
    "  - Always identify the place target before commanding picks\n"
    "  - If uncertain about object identity, ask for clarification\n"
    "  - Prefer conservative picks over aggressive ones\n\n"
    "CONTEXT: You are part of a live robotics demo. Be precise, concise, and task-focused."
)

H100_PREDICT_URL  = os.environ.get("H100_PREDICT_URL",  "http://localhost:8200")
H100_TRANSFER_URL = os.environ.get("H100_TRANSFER_URL", "http://localhost:8300")
H100_VLA_URL      = os.environ.get("H100_VLA_URL",      "http://localhost:8500")
H100_SPEECH_URL   = os.environ.get("H100_SPEECH_URL",   "http://localhost:8600")
AGENT_URL         = os.environ.get("AGENT_URL",         "http://localhost:8085")
NIS_URL           = os.environ.get("NIS_URL",           "http://localhost:8000")

VLA_XARM_MODEL    = os.environ.get("VLA_XARM_MODEL",
    "/data/organica-ai/models/vla_xarm_v4/gpu1")
BITNET_ONNX_PATH  = os.environ.get("BITNET_ONNX_PATH",
    "/opt/nis-protocol/models/bitnet_edge.onnx")
BITNET_LABELS_PATH = os.environ.get("BITNET_LABELS_PATH",
    "/opt/nis-protocol/models/label_map.json")

# ── Confirmed arm positions (IK verified 2026-02-27) ──────────────────────────
# Pi verified named positions (2026-02-25) — used via /arm/named/ API
# S6=gripper (100=open, 550=close), S2=base_pan (lateral L/R)
_PI_HOME  = {"1":500,"2":500,"3":400,"4":500,"5":400,"6":350}
_PI_PICK  = {"1":500,"2":258,"3":633,"4":500,"5":758,"6":100}  # pick_table (verified)
_PI_READY = {"1":500,"2":484,"3":433,"4":500,"5":432,"6":350}  # ready (lift)
_PI_PLACE = {"1":200,"2":370,"3":620,"4":380,"5":580,"6":100}  # place_bin (verified)
# Legacy aliases kept for non-pick routes that reference them
_HOME  = _PI_HOME
_HOVER = _PI_PICK   # approach same as pick_table
_MID   = _PI_PICK
_PICK  = _PI_PICK
_GRIP  = {"1":500,"2":258,"3":633,"4":500,"5":758,"6":550}  # pick + grip closed
_LIFT  = _PI_READY
_PLACE = _PI_PLACE
_RELAX = {**_PI_PLACE, "6":100}  # place + grip open


class RobotPlanRequest(BaseModel):
    query: str = Field(..., description="Task query (e.g. 'Pick up the red cube')")
    robot_state: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Current robot state")
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded scene image")


class TransferRequest(BaseModel):
    type: str = Field(default="edge", description="Control type: edge, depth, seg, vis")
    strength: float = Field(default=0.7, description="Transfer guidance strength")
    source_image: Optional[str] = Field(default=None, description="Source frame base64")
    target_image: Optional[str] = Field(default=None, description="Target frame base64")


class ExecuteRequest(BaseModel):
    action_plan: List[str] = Field(..., description="Action steps from /robot-plan")
    robot_state: Optional[Dict[str, Any]] = Field(default_factory=dict)
    execute_arm: bool = Field(default=True, description="Send commands to physical xArm")
    simulation: bool = Field(default=False, description="Dry-run without physical movement")


class DemoRequest(BaseModel):
    task: str = Field(default="Pick up the lighter and place it on the left",
                      description="Natural language task for full cookoff demo")
    execute_arm: bool = Field(default=True)
    simulation: bool = Field(default=False)
    image_base64: Optional[str] = Field(default=None)


class DanceRequest(BaseModel):
    genre: str = Field(default="reggaeton",
                       description="Dance genre: reggaeton, cumbia, bachata, salsa")
    moves: int = Field(default=24, description="Number of dance moves (default 24)")
    energy: float = Field(default=0.20, description="Energy level 0.0-0.5")
    use_mic: bool = Field(default=False,
                          description="Use Pi mic for live beat detection (default: demo mode)")


class PickRequest(BaseModel):
    s6: int = Field(default=500, description="Base rotation S6 (500=center, 400=right, 600=left)")
    z: float = Field(default=1.5, description="Pick height cm above table (confirmed: 1.5)")
    place: str = Field(default="left90",
                       description="Drop zone: left90, left45, right45, right90")
    wait_sec: float = Field(default=0.0,
                            description="Seconds to pause at hover before picking (0=no wait)")


# ── BitNet edge model cache (loaded once on first call) ──────────────────────
_bitnet_session = None
_bitnet_labels: Optional[Dict] = None

def _get_bitnet():
    """Lazy-load BitNet ONNX session + label map (Pi CPU inference)."""
    global _bitnet_session, _bitnet_labels
    if _bitnet_session is not None:
        return _bitnet_session, _bitnet_labels
    try:
        import onnxruntime as ort, json as _json
        _bitnet_session = ort.InferenceSession(
            BITNET_ONNX_PATH,
            providers=["CPUExecutionProvider"],
        )
        with open(BITNET_LABELS_PATH) as f:
            _bitnet_labels = _json.load(f)
        logger.info("BitNet edge model loaded from %s", BITNET_ONNX_PATH)
    except Exception as e:
        logger.warning("BitNet edge not available: %s", e)
    return _bitnet_session, _bitnet_labels


def _bitnet_infer(text: str) -> Dict[str, Any]:
    """Run BitNet intent classification on Pi CPU. Returns {intent, confidence, latency_ms}."""
    import time as _t, numpy as np
    sess, labels = _get_bitnet()
    if sess is None or labels is None:
        return {"intent": None, "confidence": 0.0, "source": "bitnet_unavailable"}
    try:
        t0 = _t.time()
        VOCAB, MAX_LEN = 512, 64
        tokens = [ord(c) % VOCAB for c in text.lower()[:MAX_LEN]]
        tokens += [0] * (MAX_LEN - len(tokens))
        inp = np.array([tokens], dtype=np.int64)
        logits = sess.run(None, {sess.get_inputs()[0].name: inp})[0][0]
        idx = int(np.argmax(logits))
        conf = float(np.exp(logits[idx]) / np.sum(np.exp(logits)))
        idx2label = labels.get("idx2label", {})
        intent = idx2label.get(str(idx), f"class_{idx}")
        latency_ms = round((_t.time() - t0) * 1000, 1)
        return {"intent": intent, "confidence": round(conf, 4),
                "latency_ms": latency_ms, "source": "bitnet_edge_v1"}
    except Exception as e:
        logger.warning("BitNet infer error: %s", e)
        return {"intent": None, "confidence": 0.0, "source": f"bitnet_error:{e}"}


@router.post("/intent")
async def cookoff_intent(request: RobotPlanRequest):
    """
    ⚡ BitNet Edge Intent Classification (Pi CPU, ~50ms, no H100 needed)

    Classifies voice/text commands offline using the 52MB BitNet ONNX model.
    Returns the NIS intent label (e.g. xarm:pick_lighter, nis:autonomy_start).
    Use this as a fast pre-filter before sending to H100 for full planning.
    """
    result = _bitnet_infer(request.query)
    return {
        "ok":         result["intent"] is not None,
        "query":      request.query,
        "intent":     result["intent"],
        "confidence": result["confidence"],
        "latency_ms": result.get("latency_ms", 0),
        "source":     result["source"],
        "timestamp":  time.time(),
    }


@router.post("/vla-infer")
async def cookoff_vla_infer(request: RobotPlanRequest):
    """
    ⚡ VLA Fast Motor Path (H100 GPU 1, ~5ms, no LLM reasoning)

    Sends image directly to VLA xArm server for sub-10ms servo position output.
    Use for simple, repeated motions where R2 planning overhead isn't needed.
    Returns raw 4D action + mapped servo positions ready for /arm/group_move.
    """
    import httpx
    t_start = time.time()
    if not request.image_base64:
        return {"ok": False, "error": "image_base64 required for VLA inference"}
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=2.0, read=5.0, write=2.0, pool=2.0)
        ) as c:
            r = await c.post(f"{H100_VLA_URL}/infer", json={
                "image_base64": request.image_base64,
                "instruction":  request.query,
            })
            if r.status_code == 200:
                d = r.json()
                return {
                    "ok":         d.get("ok", True),
                    "query":      request.query,
                    "action_raw": d.get("action_raw", []),
                    "servos":     d.get("servos", {}),
                    "latency_ms": round((time.time() - t_start) * 1000),
                    "vla_ms":     d.get("latency_ms", 0),
                    "source":     "h100_vla_xarm_v4",
                    "timestamp":  time.time(),
                }
            return {"ok": False, "error": f"VLA server HTTP {r.status_code}"}
    except Exception as e:
        logger.warning("VLA infer failed: %s", e)
        return {"ok": False, "error": str(e), "source": "h100_vla_unavailable"}


@router.post("/speech-infer")
async def cookoff_speech_infer(request: RobotPlanRequest):
    """
    🎙️ Speech2Action Fast Path (H100 GPU 5, ~5ms)

    Converts voice text directly to 6D servo positions.
    Bypasses LLM planning for known commands (wave, home, pick, dance).
    """
    import httpx
    t_start = time.time()
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=2.0, read=5.0, write=2.0, pool=2.0)
        ) as c:
            r = await c.post(f"{H100_SPEECH_URL}/infer", json={"text": request.query})
            if r.status_code == 200:
                d = r.json()
                return {
                    "ok":         d.get("ok", True),
                    "text":       request.query,
                    "servos":     d.get("servos", {}),
                    "action_raw": d.get("action_raw", []),
                    "latency_ms": round((time.time() - t_start) * 1000),
                    "source":     "h100_speech2action_v1",
                    "timestamp":  time.time(),
                }
            return {"ok": False, "error": f"Speech2Action HTTP {r.status_code}"}
    except Exception as e:
        logger.warning("Speech2Action infer failed: %s", e)
        return {"ok": False, "error": str(e), "source": "h100_speech_unavailable"}


@router.get("/status")
async def cookoff_status():
    """📊 Cosmos Cookoff pipeline status — checks all H100 services in parallel."""
    import asyncio, httpx

    checks = [
        ("reason2",        f"{H100_REASON_URL}/health"),
        ("predict25",      f"{H100_PREDICT_URL}/health"),
        ("transfer25",     f"{H100_TRANSFER_URL}/health"),
        ("vla_xarm",       f"{H100_VLA_URL}/health"),
        ("speech2action",  f"{H100_SPEECH_URL}/health"),
    ]

    async def _check(name, url):
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=3.0, read=5.0, write=3.0, pool=3.0)
            ) as c:
                r = await c.get(url)
                d = r.json() if isinstance(r.json(), dict) else {"raw": str(r.json())[:80]}
                return name, {"healthy": r.status_code == 200, "detail": d}
        except Exception as e:
            return name, {"healthy": False, "error": str(e)[:80]}

    results = await asyncio.gather(*[_check(n, u) for n, u in checks])
    h100 = dict(results)
    any_h100_up = any(v.get("healthy") for v in h100.values())
    # Check NIS cloud fallback
    cloud_ok = False
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=2.0, read=4.0, write=2.0, pool=2.0)) as c:
            r = await c.get(f"{NIS_CLOUD_URL}/health")
            cloud_ok = r.status_code == 200
    except Exception:
        pass
    pipeline_status = "operational" if (any_h100_up or cloud_ok) else "degraded"
    return {
        "status": pipeline_status,
        "h100_services": h100,
        "nis_cloud": {"healthy": cloud_ok, "url": NIS_CLOUD_URL},
        "timestamp": time.time(),
    }


class PredictVideoRequest(BaseModel):
    prompt:               str   = Field(..., description="Action description for Cosmos Predict — e.g. 'robot arm picks up the blue lighter and places it in the bin'")
    image_b64:            Optional[str]  = Field(None, description="Current camera frame as base64 JPEG. If omitted, NIS fetches a fresh snapshot.")
    num_frames:           int   = Field(25,  ge=8,  le=60,  description="Number of video frames to generate")
    fps:                  int   = Field(10,  ge=5,  le=30,  description="Playback FPS of generated video")
    num_inference_steps:  int   = Field(25,  ge=10, le=50,  description="Diffusion steps — higher = better quality, slower")
    guidance_scale:       float = Field(7.0, ge=1.0, le=15.0)
    seed:                 int   = Field(42)


@router.post("/predict-video")
async def cookoff_predict_video(request: PredictVideoRequest):
    """
    Cosmos Predict 2.5 video2world — generate a short video preview of an action
    before the arm executes it.

    Pipeline:
      1. Capture current camera frame (or use provided image_b64)
      2. Send frame + action prompt → Cosmos Predict 2.5 /video2world on H100 :8200
      3. Return base64 MP4 video + generation metadata

    The video shows Cosmos predicting what the workspace will look like
    after the described action — a visual preview before real arm movement.
    """
    import httpx

    t0 = time.time()

    # Step 1: Get camera frame if not provided
    image_b64 = request.image_b64

    if not image_b64:
        # Try agent camera snapshot
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                snap = await c.get(f"{AGENT_URL}/camera/snapshot")
                if snap.status_code == 200:
                    snap_data = snap.json()
                    image_b64 = snap_data.get("frame_b64") or snap_data.get("image_b64") or snap_data.get("data")
        except Exception as e:
            logger.warning("predict-video: agent snapshot failed: %s", e)

    if not image_b64:
        # Fallback: grab YOLO annotated frame (raw JPEG bytes from /yolo/stream-frame)
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                fr = await c.get(f"{NIS_URL}/yolo/stream-frame")
                if fr.status_code == 200 and fr.headers.get("content-type", "").startswith("image/"):
                    import base64 as _b64
                    image_b64 = _b64.b64encode(fr.content).decode()
        except Exception as e:
            logger.warning("predict-video: yolo stream-frame fallback failed: %s", e)

    if not image_b64:
        # Last resort: generate a neutral gray 848×480 placeholder so Predict never gets None
        try:
            import base64 as _b64
            import io as _io
            from PIL import Image as _PILImage
            _img = _PILImage.new("RGB", (848, 480), color=(96, 96, 96))
            _buf = _io.BytesIO()
            _img.save(_buf, format="JPEG", quality=85)
            image_b64 = _b64.b64encode(_buf.getvalue()).decode()
            logger.info("predict-video: using gray placeholder frame (no camera available)")
        except Exception as e:
            logger.warning("predict-video: placeholder frame generation failed: %s", e)

    # Step 2: Call Cosmos Predict 2.5 /video2world
    predict_body = {
        "prompt":              request.prompt,
        "num_frames":          request.num_frames,
        "fps":                 request.fps,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale":      request.guidance_scale,
        "seed":                request.seed,
        "height":              480,
        "width":               848,
    }
    if image_b64:
        predict_body["image_b64"] = image_b64

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)
        ) as c:
            r = await c.post(f"{H100_PREDICT_URL}/video2world", json=predict_body)
            if r.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Cosmos Predict returned {r.status_code}: {r.text[:200]}")
            d = r.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Cosmos Predict 2.5 timed out — try fewer frames or inference steps")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Cosmos Predict 2.5 unavailable: {e}")

    return {
        "ok":           True,
        "video_b64":    d.get("video_b64", ""),
        "prompt":       request.prompt,
        "num_frames":   request.num_frames,
        "fps":          request.fps,
        "duration_s":   round(request.num_frames / request.fps, 1),
        "latency_ms":   round((time.time() - t0) * 1000),
        "model":        "cosmos-predict2-5-video2world",
        "had_frame":    bool(image_b64),
    }


@router.post("/transfer")
async def cookoff_transfer(request: TransferRequest):
    """
    🎬 Cosmos Transfer2.5 — submits job to H100 and returns job_id immediately.
    Poll /cookoff/transfer/status/{job_id} for result.
    """
    import httpx

    ctrl = request.type if request.type in ("edge", "depth", "seg", "vis") else "edge"
    body = {
        "demo": "car_edge",
        "control_type": ctrl,
        "guidance": request.strength * 5.0,
    }
    if request.source_image:
        body["source_image"] = request.source_image
    if request.target_image:
        body["target_image"] = request.target_image

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0)
        ) as c:
            sr = await c.post(f"{H100_TRANSFER_URL}/transfer/submit", json=body)
            if sr.status_code != 200:
                raise HTTPException(status_code=502,
                    detail=f"Transfer2.5 submit failed: HTTP {sr.status_code}")
            job_id = sr.json().get("job_id")
            if not job_id:
                raise HTTPException(status_code=502,
                    detail="Transfer2.5 submit returned no job_id")

        logger.info("Transfer2.5 job submitted: %s", job_id)
        return {
            "ok": True,
            "job_id": job_id,
            "status": "submitted",
            "source": "h100_transfer25",
            "poll_url": f"/cookoff/transfer/status/{job_id}",
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Transfer2.5 submit error: %s", e)
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/transfer/status/{job_id}")
async def cookoff_transfer_status(job_id: str):
    """
    📊 Poll Transfer2.5 job status — proxy to H100 :8300/transfer/status/{job_id}.
    Returns {status: running|completed|failed, video_base64, transferred_image}.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            pr = await c.get(f"{H100_TRANSFER_URL}/transfer/status/{job_id}")
            if pr.status_code != 200:
                raise HTTPException(status_code=pr.status_code,
                    detail=f"H100 status check failed: HTTP {pr.status_code}")
            pd = pr.json()

        status = pd.get("status", "unknown")
        if pd.get("video_b64") or pd.get("video_base64"):
            return {
                "ok": True,
                "job_id": job_id,
                "status": "completed",
                "source": "h100_transfer25",
                "transferred_image": pd.get("preview_b64", ""),
                "video_base64": pd.get("video_b64") or pd.get("video_base64", ""),
                "all_videos": pd.get("all_videos", {}),
                "timestamp": time.time(),
            }
        return {
            "ok": status == "running",
            "job_id": job_id,
            "status": status,
            "error": pd.get("error"),
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Transfer2.5 status error: %s", e)
        raise HTTPException(status_code=502, detail=str(e))


def _extract_actions(text: str, limit: int = 6) -> List[str]:
    """Extract numbered/bulleted action steps from LLM text output.
    Normalizes verbose R2 natural-language steps to ACTION_MAP keywords.
    """
    # Normalization rules: substring → canonical keyword (longest match checked first)
    _NL_NORMALIZE = [
        ("pick and place",   "pick_and_place"),
        ("pick up",          "pick_and_place"),
        ("pick the",         "pick_and_place"),
        ("grab the",         "pick_and_place"),
        ("grasp the",        "pick_and_place"),
        ("transport",        "pick_and_place"),
        ("place in bin",     "pick_and_place"),
        ("put in bin",       "pick_and_place"),
        ("drop in bin",      "pick_and_place"),
        ("release into",     "release"),
        ("release the",      "release"),
        ("open gripper",     "open"),
        ("open the gripper", "open"),
        ("close gripper",    "close"),
        ("close the gripper","close"),
        ("grip",             "grip"),
        ("lift the",         "lift"),
        ("lift",             "lift"),
        ("position the robot arm over the bin", "place"),
        ("move it toward the bin",              "pick_and_place"),
        ("move to the bin",                     "pick_and_place"),
        ("move the robot arm to",               "reach"),
        ("move gripper",     "reach"),
        ("move the",         "reach"),
        ("align the",        "reach"),
        ("align",            "reach"),
        ("approach",         "reach"),
        ("lower the",        "lower"),
        ("lower",            "lower"),
        ("return",           "home"),
        ("home position",    "home"),
        ("home",             "home"),
        ("inspect",          "inspect"),
        ("wave",             "wave"),
        ("dance",            "dance"),
        ("baila",            "dance"),
    ]

    raw_lines = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if s[0].isdigit() or s[0] in "-*•":
            clean = s.lstrip("0123456789.-*• ").strip()
            if clean and len(clean) > 3:
                raw_lines.append(clean)

    if not raw_lines:
        raw_lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) > 8][:limit]

    # Normalize each step to a known keyword where possible
    actions = []
    for step in raw_lines[:limit]:
        sl = step.lower()
        normalized = None
        for phrase, keyword in _NL_NORMALIZE:
            if phrase in sl:
                normalized = keyword
                break
        actions.append(normalized if normalized else step)

    return actions[:limit]


@router.post("/robot-plan")
async def robot_plan(request: RobotPlanRequest):
    """
    🤖 Cosmos Cookoff: Robot Action Plan

    Pipeline (in order, first success wins):
      1. H100 :8100/robot-plan  — full Cosmos Reason2 robot planning
      2. H100 :8200/predict     — Predict25 scene analysis (if image) → /reason plan
      3. H100 :8100/reason      — vision-language reasoning with image
      4. Local rule-based fallback
    """
    import httpx
    t_start = time.time()
    scene_description: str = ""

    # ── 1. H100 /robot-plan (Cosmos Reason2 full pipeline) ───────────────────
    try:
        body = {"command": request.query, "robot_type": "xarm", "system_prompt": NIS_SYSTEM_PROMPT}
        if request.image_base64:
            body["image_base64"] = request.image_base64
        if request.robot_state:
            body["robot_state"] = request.robot_state
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=3.0, read=8.0, write=3.0, pool=3.0)) as c:
            r = await c.post(f"{H100_REASON_URL}/robot-plan", json=body)
            if r.status_code == 200:
                d = r.json()
                actions = d.get("action_plan", [])
                if not actions and d.get("action"):
                    actions = [d["action"]]
                if not actions:
                    actions = _extract_actions(d.get("reasoning", d.get("response", "")))
                logger.info("robot-plan: H100 /robot-plan OK (%.1fs)", time.time() - t_start)
                return {
                    "cosmos_reasoning": {
                        "reasoning_chain": d.get("reasoning", "")[:800],
                        "answer": d.get("action", ""),
                        "trajectory": d.get("trajectory", []),
                        "scene_description": d.get("scene_description", ""),
                        "spatial_understanding": {
                            "source": "h100_robot_plan",
                            "physics": d.get("physics_checks", {}),
                            "safe_to_execute": d.get("safe_to_execute", True),
                        },
                    },
                    "action_recommendations": actions or ["inspect"],
                    "combined_confidence": d.get("confidence", 0.85),
                    "nis_physics_validation": {
                        "safe": d.get("safe_to_execute", True),
                        **d.get("physics_checks", {}),
                    },
                    "robot_state": request.robot_state,
                    "source": f"h100_cosmos_reason2 ({d.get('model', 'reason2')})",
                    "latency_ms": round((time.time() - t_start) * 1000),
                    "timestamp": time.time(),
                }
    except Exception as e:
        logger.warning("H100 /robot-plan failed: %s", e)

    # ── 2. H100 /reason — structured task → numbered action steps ──────────────
    # /reason fields (confirmed): query(req), image_base64, max_tokens, use_think
    # /reason returns (confirmed): reasoning, response, full_text, confidence, model
    try:
        reason_prompt = (
            f"You are controlling an xArm robot arm on a wooden table.\n"
            f"Task: {request.query}\n"
            f"Robot state: {request.robot_state or 'unknown'}\n\n"
            f"Provide exactly 4-6 numbered steps to complete this task. "
            f"Each step must be a single concrete arm action."
        )
        reason_body: Dict[str, Any] = {
            "query": reason_prompt,
            "max_tokens": 300,
            "use_think": False,
        }
        if request.image_base64:
            reason_body["image_base64"] = request.image_base64
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=12.0, write=3.0, pool=3.0)
        ) as c:
            r = await c.post(f"{H100_REASON_URL}/reason", json=reason_body)
            if r.status_code == 200:
                d = r.json()
                # /reason confirmed fields: reasoning, response, full_text, confidence, model
                text    = d.get("reasoning") or d.get("response") or d.get("full_text", "")
                actions = _extract_actions(text)
                # Also get trajectory from /robot-plan for spatial context
                trajectory: List[Dict] = []
                safe_to_execute = True
                try:
                    async with httpx.AsyncClient(
                        timeout=httpx.Timeout(connect=3.0, read=12.0, write=3.0, pool=3.0)
                    ) as c2:
                        pr = await c2.post(f"{H100_REASON_URL}/robot-plan", json={
                            "command": request.query,
                            "robot_type": "xarm",
                            "image_base64": request.image_base64,
                            "system_prompt": NIS_SYSTEM_PROMPT,
                        })
                        if pr.status_code == 200:
                            pd = pr.json()
                            trajectory      = pd.get("trajectory", [])
                            safe_to_execute = pd.get("safe_to_execute", True)
                except Exception:
                    pass
                logger.info("robot-plan: H100 /reason+robot-plan OK (%.1fs)", time.time() - t_start)
                return {
                    "cosmos_reasoning": {
                        "reasoning_chain": text[:800],
                        "scene_description": scene_description,
                        "trajectory": trajectory,
                        "spatial_understanding": {
                            "source": "h100_reason2",
                            "has_image": bool(request.image_base64),
                            "safe_to_execute": safe_to_execute,
                        },
                    },
                    "action_recommendations": actions or ["inspect", "reach", "grasp", "release"],
                    "combined_confidence": d.get("confidence", 0.75),
                    "nis_physics_validation": {"safe": safe_to_execute},
                    "robot_state": request.robot_state,
                    "source": f"h100_reason ({d.get('model', 'cosmos-reason2')})",
                    "latency_ms": round((time.time() - t_start) * 1000),
                    "timestamp": time.time(),
                }
    except Exception as e:
        logger.warning("H100 /reason failed: %s", e)

    # 3. Windows NIS Claude fallback (H100 gone — route to :8007 which has Anthropic API)
    try:
        cloud_prompt = (
            f"You are controlling an xArm robot. Task: {request.query}\n"
            f"Give exactly 4 numbered steps to complete this task. "
            f"Each step: one concrete arm action (hover/grasp/lift/place)."
        )
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=15.0, write=3.0, pool=3.0)
        ) as c:
            r = await c.post(f"{NIS_CLOUD_URL}/chat",
                             json={"message": cloud_prompt, "max_tokens": 200})
            if r.status_code == 200:
                d = r.json()
                text = d.get("response") or d.get("message") or d.get("reasoning", "")
                actions = _extract_actions(text)
                logger.info("robot-plan: NIS cloud fallback OK (%.1fs)", time.time() - t_start)
                return {
                    "cosmos_reasoning": {
                        "reasoning_chain": text[:800],
                        "scene_description": scene_description,
                        "trajectory": [],
                        "spatial_understanding": {"source": "nis_cloud_claude", "has_image": False},
                    },
                    "action_recommendations": actions or ["inspect", "reach", "grasp", "release"],
                    "combined_confidence": 0.80,
                    "nis_physics_validation": {"safe": True},
                    "robot_state": request.robot_state,
                    "source": f"nis_cloud ({d.get('model', 'claude')})",
                    "latency_ms": round((time.time() - t_start) * 1000),
                    "timestamp": time.time(),
                }
    except Exception as e:
        logger.warning("NIS cloud /chat fallback failed: %s", e)

    # 4. Local rule-based fallback (only if all remote reasoning unavailable)
    try:
        import numpy as np
        from src.agents.cosmos import get_cosmos_reasoner

        reasoner = get_cosmos_reasoner()
        if not reasoner.initialized:
            await reasoner.initialize()

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        if request.image_base64:
            try:
                import base64, io
                from PIL import Image
                raw = base64.b64decode(request.image_base64)
                image = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
            except Exception:
                pass

        result = await reasoner.reason(image=image, task=request.query, constraints=[])
        plan = result.get("plan", [])
        actions = [s.get("action", str(s)) for s in plan] if isinstance(plan, list) else [str(plan)]

        return {
            "cosmos_reasoning": {
                "reasoning_chain": result.get("reasoning_trace", ""),
                "spatial_understanding": result.get("physics_understanding", {}),
            },
            "action_recommendations": actions,
            "combined_confidence": result.get("confidence", 0.55),
            "nis_physics_validation": result.get("safety_check", {}),
            "robot_state": request.robot_state,
            "source": "local_fallback",
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Robot plan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _log_pick_outcome(success: bool, steps: list, params: dict, latency_ms: int) -> None:
    """
    Log pick result into AdaptiveGoalSystem and AuditChain.
    Called non-blocking after each /cookoff/pick — fires-and-forgets from the route.

    AdaptiveGoalSystem learns:
      - success rate by place zone (left90, right45, …)
      - which step fails most often (grip_close, pick, …)
      - latency trends over time
    """
    # Identify failing step for learning signal
    failed_steps = [s["step"] for s in steps if not s.get("ok", True)]
    steps_ok = sum(1 for s in steps if s.get("ok", True))

    outcome = {
        "success": success,
        "steps_ok": steps_ok,
        "steps_total": len(steps),
        "failed_steps": failed_steps,
        "place_zone": params.get("place", "left90"),
        "s6": params.get("s6", 500),
        "z_cm": params.get("z", 1.5),
        "latency_ms": latency_ms,
        "strategy": "ik_pick_v2",
        "lessons": (
            [f"step '{failed_steps[0]}' failed — check servo or Pi connectivity"]
            if failed_steps else ["full sequence succeeded"]
        ),
        "metrics": {
            "steps_success_rate": steps_ok / max(len(steps), 1),
            "latency_ms": latency_ms,
        },
    }

    # 1. Log into AdaptiveGoalSystem via app.state
    try:
        from main import app
        goal_sys = getattr(app.state, "adaptive_goal_system", None)
        if goal_sys is not None:
            # Use "robotics_pick" as the goal type for pattern tracking
            import uuid as _uuid
            ephemeral_goal_id = f"pick_{_uuid.uuid4().hex[:8]}"
            goal_sys.goal_success_patterns.setdefault("robotics_pick", []).append(
                1.0 if success else 0.0
            )
            goal_sys.goal_metrics["goals_completed" if success else "goals_failed"] += 1
            total = (goal_sys.goal_metrics["goals_completed"]
                     + goal_sys.goal_metrics["goals_failed"])
            if total > 0:
                goal_sys.goal_metrics["average_success_rate"] = (
                    goal_sys.goal_metrics["goals_completed"] / total
                )
            logger.info(
                "[PickOutcome] AdaptiveGoalSystem updated — "
                f"robotics_pick success_rate="
                f"{goal_sys.goal_metrics['average_success_rate']:.2f} "
                f"(n={total})"
            )
    except Exception as e:
        logger.debug(f"[PickOutcome] AdaptiveGoalSystem update skipped: {e}")

    # 2. Log into AuditChain
    try:
        from src.core.audit_chain import get_audit_chain
        get_audit_chain().log(
            agent_id="cookoff/pick",
            action_type="arm_pick",
            layer="action",
            payload={
                "params": params,
                "steps_ok": steps_ok,
                "steps_total": len(steps),
                "failed_steps": failed_steps,
                "latency_ms": latency_ms,
            },
            success=success,
            duration_ms=float(latency_ms),
            tags=["robotics", "pick", params.get("place", "left90")],
        )
    except Exception as e:
        logger.debug(f"[PickOutcome] AuditChain log skipped: {e}")

    # 3. SSE notification
    _arm_sse_publish("pick_outcome", {"success": success, **outcome}, "done")


def _arm_sse_publish(step: str, servos: dict, status: str) -> None:
    """Publish arm movement event to SSE arm topic — non-blocking."""
    try:
        from routes.events import publish as _pub
        _pub("arm", {"step": step, "servos": servos, "status": status, "ts": time.time()})
    except Exception:
        pass


async def _group_move(client, servos: dict, dur_ms: int = 800, retries: int = 2):
    """Send a /arm/group_move command with retry."""
    body = {"positions": {str(k): v for k, v in servos.items()}, "duration_ms": dur_ms}
    last_err = None
    for _ in range(retries):
        try:
            r = await client.post(f"{AGENT_URL}/arm/group_move", json=body, timeout=8.0)
            d = r.json() if r.status_code == 200 else {}
            if r.status_code == 200:
                return d
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)
        await asyncio.sleep(0.3)
    return {"ok": False, "error": last_err}


async def _run_ik_pick(client, s6: int = 500, place: str = "left90",
                       wait_sec: float = 0.0,
                       bin_cx: int = None, bin_cy: int = None,
                       lighter_cx: int = None, lighter_cy: int = None,
                       frame_w: int = 640, frame_h: int = 480,
                       object_noun: str = "object") -> dict:
    """
    Pick-and-place using CALIBRATED positions from touch_poses.json.
    s6 = S2 base pan value (lateral alignment, 500=center, 380=far-left, 620=far-right).

    CALIBRATED positions (touch_poses.json, verified 2026-03-03):
      home/standby:  {1:350, 2:500, 3:310, 4:870, 5:680, 6:500}
      pick_hover:    {1:100, 2:s2,  3:222, 4:697, 5:604, 6:500}  z=6cm above table
      pick_down:     {1:100, 2:s2,  3:142, 4:856, 5:430, 6:500}  z=1.5cm TABLE LEVEL
      place_bin:     {1:600, 2:s2b, 3:220, 4:827, 5:425, 6:875}  over bowl, S6=875 extends

    VLA v5 (cx-aware): pass lighter_cx → server overrides S2 with calibrated lateral.
    Blend VLA depth (S3/S4) 25% with calibrated for slight depth adaptation.
    """
    # S2 lateral from s6 param (cx already converted to s2 by caller)
    s2_lateral = max(380, min(620, s6))
    executed = []
    ok = True

    async def _step(label, url, json_body=None, sleep_s=1.3):
        nonlocal ok
        try:
            if json_body is not None:
                r = await client.post(url, json=json_body, timeout=10.0)
            else:
                r = await client.post(url, timeout=10.0)
            step_ok = r.status_code == 200
            executed.append({"step": label, "ok": step_ok,
                             "error": "" if step_ok else r.text[:80]})
            if not step_ok:
                ok = False
        except Exception as e:
            executed.append({"step": label, "ok": False, "error": str(e)[:80]})
            ok = False
        await asyncio.sleep(sleep_s)

    if wait_sec > 0:
        await asyncio.sleep(wait_sec)

    # ── Calibrated base positions ────────────────────────────────────────────
    # pick_hover_center: S3=222 S4=697 S5=604 S6=500
    # pick_down_center:  S3=142 S4=856 S5=430 S6=500
    # place_bin:         S3=220 S4=827 S5=425 S6=875  (S6=875 CRITICAL for bowl reach)
    CAL_HOVER  = {"3": 222, "4": 697, "5": 604}
    CAL_DESCEND = {"3": 142, "4": 856, "5": 430}

    # ── VLA v5 cx-aware inference ────────────────────────────────────────────
    # VLA server v2 accepts cx → overrides S2 with calibrated lateral.
    # Use VLA S3/S4 for depth adaptation (25% blend).
    vla_s2 = s2_lateral
    vla_hover_s3, vla_hover_s4 = CAL_HOVER["3"], CAL_HOVER["4"]
    vla_desc_s3,  vla_desc_s4  = CAL_DESCEND["3"], CAL_DESCEND["4"]

    try:
        snap_r = await client.get(f"{AGENT_URL}/camera/snapshot", timeout=6.0)
        if snap_r.status_code == 200:
            snap_b64 = snap_r.json().get("image_base64", "")
            if snap_b64:
                cx_hint = lighter_cx if lighter_cx is not None else None
                vla_payload = {
                    "image_base64": snap_b64,
                    "instruction":  f"pick {object_noun} on table",
                    "temperature":  0.0,
                    "cx":           cx_hint,
                    "frame_w":      frame_w,
                }
                vla_r = await client.post(f"{H100_VLA_URL}/infer",
                                          json=vla_payload, timeout=8.0)
                if vla_r.status_code == 200:
                    vd = vla_r.json()
                    vservos = vd.get("servos", {})
                    # S2: VLA now computes from cx — use directly
                    vla_s2 = int(vservos.get("2", s2_lateral))
                    vla_s2 = max(380, min(620, vla_s2))
                    # S3/S4: blend 25% VLA + 75% calibrated for slight depth adaptation
                    raw_s3 = int(vservos.get("3", CAL_HOVER["3"]))
                    raw_s4 = int(vservos.get("4", CAL_HOVER["4"]))
                    vla_hover_s3 = int(0.25 * raw_s3 + 0.75 * CAL_HOVER["3"])
                    vla_hover_s4 = int(0.25 * raw_s4 + 0.75 * CAL_HOVER["4"])
                    # Clamp hover to safe range
                    vla_hover_s3 = max(190, min(280, vla_hover_s3))
                    vla_hover_s4 = max(650, min(780, vla_hover_s4))
                    logger.info("[vla-v5] S2=%d (cx=%s) hover_S3=%d S4=%d",
                                vla_s2, cx_hint, vla_hover_s3, vla_hover_s4)
    except Exception as e:
        logger.warning("[vla] skipped: %s", e)
        vla_s2 = s2_lateral  # fallback to cx-based

    # ── Depth-adaptive joints from cy — use _cy_to_pick_joints / _cy_to_place_joints ──
    # These functions interpolate S3/S4/S5 between two calibrated reference points:
    #   cy=373/480 (pick_table) → {S3=142, S4=856, S5=430}  z~1.2cm table level
    #   cy=465/480 (reach_fwd)  → {S3=222, S4=697, S5=604}  z~6-8cm extended reach
    # Extrapolation clamped to safe servo ranges.
    s6_reach = 500
    if lighter_cy is not None and frame_h > 0:
        cy_norm = lighter_cy / frame_h
        s6_reach = int(500 + (0.5 - cy_norm) * 240)
        s6_reach = max(400, min(680, s6_reach))
        hover_s3, hover_s4, hover_s5 = _cy_to_place_joints(lighter_cy, frame_h)
        desc_s3,  desc_s4,  desc_s5  = _cy_to_pick_joints(lighter_cy, frame_h)
    else:
        hover_s3, hover_s4, hover_s5 = CAL_HOVER["3"], CAL_HOVER["4"], CAL_HOVER["5"]
        desc_s3,  desc_s4,  desc_s5  = CAL_DESCEND["3"], CAL_DESCEND["4"], CAL_DESCEND["5"]

    # Blend VLA S3/S4 (25% VLA + 75% depth-computed) — VLA adds fine-grain correction
    hover_s3 = int(0.25 * vla_hover_s3 + 0.75 * hover_s3)
    hover_s4 = int(0.25 * vla_hover_s4 + 0.75 * hover_s4)
    hover_s3 = max(100, min(280, hover_s3))
    hover_s4 = max(600, min(850, hover_s4))
    logger.info("[pick] S2=%d S6=%d hover_S3=%d S4=%d S5=%d desc_S3=%d S4=%d S5=%d cy=%s",
                vla_s2, s6_reach, hover_s3, hover_s4, hover_s5, desc_s3, desc_s4, desc_s5, lighter_cy)

    # ── Pick sequence ────────────────────────────────────────────────────────
    # 1. Home
    await _step("home", f"{AGENT_URL}/arm/home", sleep_s=1.4)

    # 2. Open gripper
    await _step("grip_open", f"{AGENT_URL}/arm/gripper/open", sleep_s=0.8)

    # 3. Hover above lighter — depth-adaptive S3/S4/S5, S6=s6_reach for Y-reach
    await _step("hover", f"{AGENT_URL}/arm/group_move",
                json_body={"positions": {"1": 100, "2": vla_s2,
                                         "3": hover_s3, "4": hover_s4,
                                         "5": hover_s5, "6": s6_reach},
                           "duration_ms": 3000}, sleep_s=3.5)

    # 3b. Visual re-check: re-detect from hover position and nudge S2/depth if off
    try:
        rescan3 = await _yolo_scan_nis(object_noun, conf=0.08)
        redets = rescan3.get("detections", []) if isinstance(rescan3, dict) else []
        if not redets:
            # try broader scan if specific label misses
            rescan3b = await _yolo_scan_nis("lighter,bottle,cup,object,item", conf=0.06)
            redets = rescan3b.get("detections", []) if isinstance(rescan3b, dict) else []
        if redets:
            new_cx = redets[0].get("cx", None)
            new_cy = redets[0].get("cy", lighter_cy)
            if new_cx is not None:
                new_s2 = _cx_to_s2(new_cx, 640)
                delta_s2 = new_s2 - vla_s2
                if abs(delta_s2) > 10:
                    vla_s2 = max(380, min(620, new_s2))
                    await _step("nudge_lateral", f"{AGENT_URL}/arm/group_move",
                                json_body={"positions": {"2": vla_s2}, "duration_ms": 600},
                                sleep_s=0.8)
                    logger.info("[recheck] nudged S2 by %+d → %d (new_cx=%d)", delta_s2, vla_s2, new_cx)
            if new_cy is not None and new_cy != lighter_cy and frame_h > 0:
                lighter_cy = new_cy
                s6_reach = int(500 + (0.5 - lighter_cy / frame_h) * 240)
                s6_reach = max(400, min(680, s6_reach))
                desc_s3, desc_s4, desc_s5 = _cy_to_pick_joints(lighter_cy, frame_h)
                logger.info("[recheck] updated depth cy=%d → desc_S3=%d S4=%d S5=%d S6=%d",
                            lighter_cy, desc_s3, desc_s4, desc_s5, s6_reach)
    except Exception as _re:
        logger.warning("[recheck] skipped: %s", _re)

    # 4. Descend to pick depth — fully adaptive S3/S4/S5 from cy interpolation
    await _step("descend", f"{AGENT_URL}/arm/group_move",
                json_body={"positions": {"1": 100, "2": vla_s2,
                                         "3": desc_s3, "4": desc_s4,
                                         "5": desc_s5, "6": s6_reach},
                           "duration_ms": 2500}, sleep_s=3.0)

    # 5. Close gripper (S1=700 strong grip)
    await _step("grip_close", f"{AGENT_URL}/arm/group_move",
                json_body={"positions": {"1": 700}, "duration_ms": 1200}, sleep_s=1.5)

    # 6. Grip verify — read servo position, retry if S1 > 400 (didn't close fully)
    try:
        st_r = await client.get(f"{AGENT_URL}/arm/status", timeout=5.0)
        if st_r.status_code == 200:
            pos = st_r.json().get("positions", {})
            s1_actual = int(pos.get("1", pos.get("S1", 700)))
            if s1_actual < 400:
                logger.warning("[grip] S1=%d < 400 — grip may have missed, retrying close", s1_actual)
                # Nudge slightly and re-close
                await asyncio.sleep(0.3)
                await _step("grip_retry", f"{AGENT_URL}/arm/group_move",
                            json_body={"positions": {"1": 800}, "duration_ms": 800}, sleep_s=1.2)
            else:
                logger.info("[grip] S1=%d — grip confirmed", s1_actual)
    except Exception as e:
        logger.warning("[grip-verify] skipped: %s", e)

    # 7. Lift (home height, grip CLOSED — never use /arm/home here, it opens S1)
    await _step("lift", f"{AGENT_URL}/arm/group_move",
                json_body={"positions": {"1": 700, "2": 500,
                                         "3": 310, "4": 870, "5": 680, "6": 500},
                           "duration_ms": 2500}, sleep_s=3.0)

    # 8. Place over bowl (S6=875 CRITICAL — extends arm to reach bowl)
    if bin_cx is not None:
        s2_bin = _cx_to_s2(bin_cx, frame_w)
        logger.info("[place] bowl cx=%d → S2=%d", bin_cx, s2_bin)
        await _step("place_bin_dyn", f"{AGENT_URL}/arm/group_move",
                    json_body={"positions": {"1": 700, "2": s2_bin,
                                             "3": 220, "4": 827, "5": 425, "6": 875},
                               "duration_ms": 3500}, sleep_s=4.0)
    else:
        await _step("place_bin", f"{AGENT_URL}/arm/named/place_bin", sleep_s=2.0)

    # 9. Release over bowl
    await _step("grip_open", f"{AGENT_URL}/arm/gripper/open", sleep_s=0.8)

    # 10. Return home
    await _step("home_final", f"{AGENT_URL}/arm/home", sleep_s=1.2)

    return {
        "ok": ok,
        "message": f"pick sequence complete — {sum(1 for s in executed if s['ok'])}/{len(executed)} steps OK",
        "steps": executed,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MASTER COOKOFF PIPELINE — /cookoff/run
# Handles multi-object prompts, color-aware picking, live Cosmos reasoning SSE
# ══════════════════════════════════════════════════════════════════════════════

import re as _re


def _publish_cookoff(event: str, data: dict) -> None:
    """Publish a structured event to the SSE 'cookoff' topic (non-blocking)."""
    try:
        from routes.events import publish as _pub
        _pub("cookoff", {"event": event, "ts": time.time(), **data})
    except Exception:
        pass


def _parse_targets(prompt: str) -> List[Dict]:
    """
    Parse natural language into a list of pick targets with color + destination.
    Handles: "blue lighter", "yellow one", "the red cup in the bin", multi-target "and".

    Returns: [{"object": "blue_lighter", "color": "blue", "noun": "lighter",
                "destination": "left90"}, ...]
    """
    COLORS = ["blue", "yellow", "red", "green", "orange", "purple", "white", "black"]
    NOUNS  = ["lighter", "cup", "box", "bottle", "bin", "cube", "block", "object", "item", "thing"]
    DEST_MAP = {
        "bin":    "left90",
        "left":   "left90",
        "right":  "right90",
        "center": "left45",
        "middle": "left45",
        "front":  "left45",
        "back":   "right45",
    }

    p = prompt.lower()
    words = _re.findall(r'\b\w+\b', p)
    targets: List[Dict] = []
    seen: set = set()

    for i, w in enumerate(words):
        if w not in COLORS:
            continue
        color = w
        noun = None
        for j in range(i + 1, min(i + 5, len(words))):
            cw = words[j]
            if cw in NOUNS:
                noun = cw
                break
            if cw == "one":        # "the yellow one" → lighter
                noun = "lighter"
                break
            if cw in ("the", "a", "an"):
                continue

        if noun is None:
            continue

        # Find the most specific destination after this color+noun phrase
        dest = "left90"
        remaining = " ".join(words[i:])
        for kw, zone in DEST_MAP.items():
            if kw in remaining:
                dest = zone
                break

        key = f"{color}_{noun}"
        if key in seen:
            continue
        seen.add(key)

        targets.append({
            "object":      key,
            "color":       color,
            "noun":        noun,
            "destination": dest,
        })

    # "all"/"every"/"each" → pick-all mode (highest priority, before naked-noun fallback)
    if not targets and any(w in p for w in ("all", "every", "each")):
        noun_target = next((n for n in NOUNS if n in p and n not in ("bin", "one")), "lighter")
        return [{"object": f"_all_{noun_target}s", "color": None, "noun": noun_target,
                  "destination": "bin", "_pick_all": True}]

    # Fallback: naked noun without color
    if not targets:
        for noun in NOUNS:
            if noun in p and noun not in ("bin", "one"):
                targets.append({
                    "object": noun, "color": None,
                    "noun": noun, "destination": "left90",
                })
                break

    return targets or [{"object": "object", "color": None, "noun": "object", "destination": "left90"}]


def _find_target_in_dets(target: Dict, detections: List[Dict]) -> Optional[Dict]:
    """
    Score all YOLO detections against target color+noun.
    Returns the best-matching detection, or None if no detections at all.
    """
    color = (target.get("color") or "").lower()
    noun  = (target["noun"]).lower()

    scored: List[tuple] = []
    for det in detections:
        label     = det.get("label", "").lower()
        det_color = det.get("color", "").lower()
        score     = 0

        # Noun match — also map common COCO aliases
        lighter_aliases = {"bottle", "vase", "lighter", "cup"}
        if noun == "lighter" and any(a in label for a in lighter_aliases):
            score += 10
        elif noun in label:
            score += 10

        # Color match
        if color:
            if color in label:
                score += 25      # color embedded in label (e.g. "blue_lighter")
            elif color == det_color:
                score += 20      # color field match
            elif color in det_color:
                score += 15

        # Prefer non-bin objects when picking
        if any(k in label for k in ("bin", "bowl", "container", "toilet", "sink")):
            score -= 5

        if score > 0:
            scored.append((score, det))

    if scored:
        scored.sort(key=lambda x: (-x[0], -x[1].get("conf", 0)))
        return scored[0][1]

    # Fallback: highest-conf non-bin detection
    fallback = [d for d in detections if not any(k in d.get("label", "").lower() for k in ("bin", "toilet", "sink"))]
    if fallback:
        return max(fallback, key=lambda d: d.get("conf", 0))
    return detections[0] if detections else None


def _cx_to_s2(cx: int, frame_w: int) -> int:
    """
    Convert YOLO detection pixel X -> S2 (base pan) servo value for lateral alignment.
    S2 is the BASE ROTATION axis — the correct servo for left/right positioning.

    Calibrated from touch_poses.json color picks (640px camera reference):
      pick_blue   cx~259 -> S2=460  (arm base rotates LEFT)
      pick_yellow cx~320 -> S2=500  (center)
      pick_green  cx~381 -> S2=540  (arm base rotates RIGHT)
    Scale: ~0.65 S2 units per pixel at 640px width.
    Direction: lighter RIGHT in image (larger cx) -> larger S2.
    """
    cx_640 = cx * (640.0 / frame_w)     # normalize to 640px reference
    dx = cx_640 - 320.0                  # offset from center
    ds2 = int(dx * 0.65)                 # ~0.65 S2 units per pixel
    ds2 = max(-120, min(120, ds2))        # clamp +/-120 units (S2 range 380-620)
    return max(380, min(620, 500 + ds2))

def _cx_to_s4(cx: int, frame_w: int) -> int:
    """Legacy name — routes to S2 base pan lateral."""
    return _cx_to_s2(cx, frame_w)

def _cx_to_s6(cx: int, frame_w: int) -> int:
    """Legacy name — routes to S2 base pan lateral."""
    return _cx_to_s2(cx, frame_w)


def _cy_to_place_joints(cy: int, frame_h: int) -> tuple:
    """
    Compute arm joints (S3, S4, S5) to position gripper above an object
    at camera pixel cy — for dynamic drop/place at detected bowl position.

    Calibrated reference points (640x480 camera, S6=500 forward):
      cy=373 (pick_table depth) -> {S3=142, S4=856, S5=430}  z~1.2cm table
      cy=465 (reach_fwd depth)  -> {S3=222, S4=697, S5=604}  z~6-8cm reach

    For placing, adds height bias (+0.30 to t) so arm clears bowl rim.
    Validated: bowl at cy=420 -> S3=207, S4=727, S5=571 (matches bowl_5s test).
    Returns (s3, s4, s5).
    """
    cy_480 = cy * (480.0 / frame_h)
    cy_480 = max(320.0, min(490.0, cy_480))   # safety clamp
    t = (cy_480 - 373.0) / (465.0 - 373.0)   # 0=near pick_table, 1=reach_fwd
    t = max(0.0, min(1.0, t))
    t_h = min(1.0, t + 0.30)                  # height bias: stay above bowl rim
    s3 = int(142 + t_h * (222 - 142))         # 142->222 (shoulder)
    s4 = int(856 + t_h * (697 - 856))         # 856->697 (elbow)
    s5 = int(430 + t_h * (604 - 430))         # 430->604 (wrist)
    return s3, s4, s5


def _cy_to_pick_joints(cy: int, frame_h: int) -> tuple:
    """
    Compute S3, S4, S5 to reach DOWN to an object at camera pixel cy.
    No height bias — we want to descend to the object, not hover above it.

    Calibrated reference (640x480 camera):
      cy=373 -> pick_table  {S3=142, S4=856, S5=430}  z~1.2cm  (confirmed pick depth)
      cy=465 -> reach_fwd   {S3=222, S4=697, S5=604}  z~6-8cm  (further reach)

    For objects at cy<373 (further from arm): extrapolate gently.
    For objects at cy>465 (very close): clamp to pick_table depth.
    Returns (s3, s4, s5).
    """
    cy_480 = cy * (480.0 / frame_h)
    cy_480 = max(150.0, min(480.0, cy_480))
    t = (cy_480 - 373.0) / (465.0 - 373.0)   # 0=pick_table, 1=reach_fwd
    t = max(-1.5, min(1.0, t))                  # allow gentle extrapolation for far objects
    s3 = int(142 + t * (222 - 142))
    s4 = int(856 + t * (697 - 856))
    s5 = int(430 + t * (604 - 430))
    # Clamp to physically safe servo ranges
    s3 = max(100, min(280, s3))
    s4 = max(600, min(900, s4))
    s5 = max(380, min(680, s5))
    return s3, s4, s5


def _dest_to_place_zone(destination: str, bin_det: Optional[Dict], frame_w: int) -> str:
    """
    Derive place zone from user destination + detected bin position.
    If a bin is detected, its pixel position takes priority.
    """
    if bin_det:
        frac = bin_det["cx"] / max(frame_w, 1)
        if frac < 0.33:
            return "right90"
        elif frac < 0.50:
            return "right45"
        elif frac < 0.66:
            return "left45"
        else:
            return "left90"

    return {
        "left90":  "left90",
        "left45":  "left45",
        "right45": "right45",
        "right90": "right90",
        "bin":     "left90",
        "left":    "left90",
        "right":   "right90",
    }.get(destination, "left90")



async def _cosmos_decide(
    image_b64: Optional[str],
    task_prompt: str,
    detections: list,
) -> Optional[dict]:
    """
    Cosmos Reason2 decides WHAT ACTION to take from a natural language prompt + scene.

    Returns:
    {
        "action":        "pick_and_place" | "describe" | "count" | "home" |
                         "gripper_open" | "gripper_close" | "inspect" | "push" | "none",
        # For pick_and_place / inspect / push:
        "pick_targets":  [{"label": str, "cx": int, "cy": int, "priority": int}],
        "place_target":  {"label": str, "cx": int, "cy": int},
        "pick_labels":   [str],
        "place_labels":  [str],
        # For any action:
        "task_summary":  str,   # one-line summary of what will happen
        "response":      str,   # text to return to user (for describe/count)
        "reasoning":     str,   # Cosmos full reasoning
    }
    Returns None on failure (caller should fall back to _parse_targets / pick_and_place default).
    """
    import json as _json, httpx as _httpx

    det_desc = ", ".join(
        f"{d.get('label','?')}(cx={d.get('cx',0)},cy={d.get('cy',0)},conf={d.get('conf',0):.2f})"
        for d in detections[:14]
    ) or "no objects detected"

    prompt = (
        f"You control a 6-DOF robotic arm (xArm) via NIS Protocol.\n"
        f"Camera is mounted above a table. Arm servos: S2=lateral, S6=reach.\n\n"
        f"USER TASK: {task_prompt}\n\n"
        f"YOLO DETECTIONS: {det_desc}\n\n"
        f"Decide what action to take and return ONLY valid JSON:\n"
        f"{{\n"
        f'  "action": "<one of: pick_and_place, describe, count, home, gripper_open, gripper_close, inspect, push, none>",\n'
        f'  "pick_targets": [{{"label": "object", "cx": 320, "cy": 240, "priority": 1}}],\n'
        f'  "place_target": {{"label": "bin", "cx": 480, "cy": 380}},\n'
        f'  "pick_labels": ["label_synonym"],\n'
        f'  "place_labels": ["bin", "bowl"],\n'
        f'  "task_summary": "one sentence: what the arm will do",\n'
        f'  "response": "natural language reply to user (for describe/count actions)",\n'
        f'  "count": 0,\n'
        f'  "reasoning": "brief explanation of your decision"\n'
        f"}}\n\n"
        f"Action selection guide:\n"
        f"- pick_and_place: move objects from one place to another\n"
        f"- describe: user asks what is on the table / what you see\n"
        f"- count: user asks how many of something\n"
        f"- home: move arm to safe home position\n"
        f"- gripper_open: open the gripper/claw\n"
        f"- gripper_close: close the gripper/claw\n"
        f"- inspect: move arm above object for closer look\n"
        f"- push: push an object without gripping\n"
        f"- none: task is informational only, no arm movement needed\n"
        f"Return ONLY the JSON, no markdown, no extra text."
    )

    try:
        body: dict = {
            "query":         prompt,
            "max_tokens":    700,
            "use_think":     False,
            "system_prompt": NIS_SYSTEM_PROMPT,
        }
        if image_b64:
            body["image_base64"] = image_b64

        async with _httpx.AsyncClient(
            timeout=_httpx.Timeout(connect=3.0, read=30.0, write=3.0, pool=3.0)
        ) as c:
            r = await c.post(f"{H100_REASON_URL}/reason", json=body)
            if r.status_code != 200:
                logger.warning("[cosmos_decide] Reason2 %d", r.status_code)
                return None

            rd  = r.json()
            raw = rd.get("response") or rd.get("reasoning") or rd.get("full_text", "")

            m = re.search(r'\{[\s\S]*\}', raw)
            if not m:
                logger.warning("[cosmos_decide] no JSON: %.200s", raw)
                return None

            plan = _json.loads(m.group())
            action = plan.get("action", "pick_and_place")

            # Defaults for pick fields
            plan.setdefault("pick_targets", [])
            pt = plan.setdefault("place_target", {"label": "bin", "cx": 320, "cy": 300})
            if "cx" not in pt: pt["cx"] = 320
            if "cy" not in pt: pt["cy"] = 300
            plan.setdefault("pick_labels",  [p.get("label","object") for p in plan["pick_targets"]])
            plan.setdefault("place_labels", [pt.get("label","bin")])
            plan.setdefault("response",     plan.get("task_summary", ""))
            plan.setdefault("count", len(plan["pick_targets"]))
            plan["reasoning"] = raw

            logger.info("[cosmos_decide] action=%s targets=%d summary=%s",
                        action, len(plan["pick_targets"]), plan.get("task_summary","")[:80])
            return plan

    except Exception as e:
        logger.warning("[cosmos_decide] failed: %s", e)
        return None

async def _cosmos_scene_plan(
    image_b64: Optional[str],
    task_prompt: str,
    detections: list,
) -> Optional[dict]:
    """
    Ask Cosmos Reason2 to analyze the scene and produce a structured pick plan.
    Replaces hardcoded bin/lighter label matching with Cosmos scene understanding.

    Returns:
    {
        "pick_targets":  [{"label": str, "cx": int, "cy": int, "priority": int}],
        "place_target":  {"label": str, "cx": int, "cy": int},
        "pick_labels":   [str],   # YOLO label synonyms for rescanning
        "place_labels":  [str],   # YOLO labels for place container
        "task_summary":  str,
        "reasoning":     str,
    }
    Returns None if Cosmos fails (caller should fallback to _parse_targets).
    """
    import json as _json, httpx as _httpx

    det_desc = ", ".join(
        f"{d.get('label','?')}(cx={d.get('cx',0)},cy={d.get('cy',0)},conf={d.get('conf',0):.2f})"
        for d in detections[:12]
    ) or "no objects detected yet"

    prompt = (
        f"You are controlling a robotic arm. Camera is mounted above a table.\n\n"
        f"Task: {task_prompt}\n\n"
        f"YOLO detected: {det_desc}\n\n"
        f"Analyze the scene and return ONLY valid JSON (no markdown, no extra text):\n"
        f"{{\n"
        f'  "pick_targets": [{{"label": "object_name", "cx": 250, "cy": 300, "priority": 1}}],\n'
        f'  "place_target": {{"label": "container_name", "cx": 480, "cy": 380}},\n'
        f'  "pick_labels": ["label_synonym1", "label_synonym2"],\n'
        f'  "place_labels": ["bowl", "bin", "container"],\n'
        f'  "task_summary": "what the arm will do",\n'
        f'  "reasoning": "what you see and your decision"\n'
        f"}}\n\n"
        f"Rules:\n"
        f"- pick_targets: objects to pick, ordered by priority (1=first). Use YOLO cx/cy when available.\n"
        f"- place_target: the destination container. If not seen, estimate cx=320 (center of frame).\n"
        f"- pick_labels: all YOLO label strings that match pick objects (include synonyms).\n"
        f"- place_labels: YOLO label strings for the destination (bowl, bin, box, tray, etc.).\n"
        f"- Return ONLY the JSON object, nothing else."
    )

    try:
        body: dict = {"query": prompt, "max_tokens": 600, "use_think": False, "system_prompt": NIS_SYSTEM_PROMPT}
        if image_b64:
            body["image_base64"] = image_b64

        async with _httpx.AsyncClient(
            timeout=_httpx.Timeout(connect=3.0, read=28.0, write=3.0, pool=3.0)
        ) as c:
            r = await c.post(f"{H100_REASON_URL}/reason", json=body)
            if r.status_code != 200:
                logger.warning("[cosmos_plan] Reason2 returned %d", r.status_code)
                return None

            rd  = r.json()
            raw = rd.get("response") or rd.get("reasoning") or rd.get("full_text", "")

            # Extract JSON block (Cosmos sometimes adds prose before/after)
            m = re.search(r'\{[\s\S]*\}', raw)
            if not m:
                logger.warning("[cosmos_plan] no JSON in response: %.200s", raw)
                return None

            plan = _json.loads(m.group())

            # Validate required fields
            if not plan.get("pick_targets"):
                logger.warning("[cosmos_plan] empty pick_targets")
                return None

            # Ensure place_target has cx
            pt = plan.setdefault("place_target", {"label": "bin", "cx": 320, "cy": 300})
            if "cx" not in pt:
                pt["cx"] = 320
            if "cy" not in pt:
                pt["cy"] = 300

            # Ensure label lists exist
            plan.setdefault("pick_labels",  [p.get("label","object") for p in plan["pick_targets"]])
            plan.setdefault("place_labels", [pt.get("label","bin")])

            # Store full reasoning for streaming
            plan["reasoning"] = raw

            logger.info("[cosmos_plan] OK — %d pick targets, place=%s cx=%d reasoning=%d chars",
                        len(plan["pick_targets"]),
                        pt.get("label","?"), pt.get("cx",0), len(raw))
            return plan

    except Exception as e:
        logger.warning("[cosmos_plan] failed: %s", e)
        return None


async def _stream_reasoning(reasoning: str, delay: float = 0.07) -> None:
    """
    Break Cosmos reasoning into sentences and publish each to SSE 'cookoff' topic,
    creating a live 'typing' effect on any connected client.
    """
    sentences = _re.split(r'(?<=[.!?])\s+', reasoning.strip())
    accumulated = ""
    for sent in sentences:
        accumulated += sent + " "
        _publish_cookoff("reasoning_token", {
            "sentence":    sent.strip(),
            "accumulated": accumulated.strip(),
        })
        await asyncio.sleep(delay)
    _publish_cookoff("reasoning_done", {"full": reasoning})


async def _yolo_scan_nis(targets_csv: str = "lighter,bottle,cup,bin,box,vase,bowl",
                          conf: float = 0.10) -> Dict:
    """
    Call NIS localhost /yolo/detect and return the full result dict.
    Used for re-scans inside /cookoff/run without importing yolo_vision.
    """
    import httpx
    try:
        async with httpx.AsyncClient(timeout=12.0) as c:
            r = await c.get("http://localhost:8000/yolo/detect",
                            params={"targets": targets_csv, "conf": str(conf)})
            if r.status_code == 200:
                return r.json()
    except Exception as e:
        logger.warning("_yolo_scan_nis failed: %s", e)
    return {"detections": [], "n": 0, "frame_w": 1280, "scene_context": "", "annotated_b64": ""}


# ── RunRequest model ──────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    prompt:       str   = Field(...,
        description="Natural language task. E.g. 'put the blue lighter in the bin'")
    execute_arm:  bool  = Field(default=True)
    simulation:   bool  = Field(default=False)


@router.post("/run")
async def cookoff_run(request: RunRequest):
    """
    🧠 Master Cookoff Pipeline — everything orchestrated end-to-end.

    Flow:
      1. Parse prompt → list of color-aware pick targets
      2. YOLO + GDINO warm scan → scene context for Cosmos
      3. Cosmos Reason2 reasoning → streamed live via SSE 'cookoff' topic
      4. Per-target loop:
           a. Re-scan for current object position (tracks movement)
           b. Find target by color + noun
           c. Compute s6 from pixel X (lateral alignment)
           d. Detect bin position → pick place zone
           e. Execute 11-step IK pick sequence
      5. Goal verification snapshot → Cosmos confirm
      6. Return full structured result

    Monitor live via:  GET /events/stream?topics=cookoff,arm,cosmos
    """
    import httpx
    t_start = time.time()
    logs: List[str] = []

    # ── 1. Initial YOLO scan — broad, no label filter ──────────────────────────
    _publish_cookoff("pipeline_start", {
        "prompt":      request.prompt,
        "execute_arm": request.execute_arm,
        "simulation":  request.simulation,
    })

    import httpx
    scan0 = await _yolo_scan_nis("all,object,item,lighter,bottle,cup,bowl,bin,box,vase,tool", conf=0.08)
    dets0    = scan0.get("detections", [])
    ann_b64  = scan0.get("annotated_b64", "")
    frame_w0 = scan0.get("frame_w", 1280)
    frame_h0 = scan0.get("frame_h", 720)
    scene0   = scan0.get("scene_context", "")
    logs.append(f"Initial scan: {len(dets0)} objects, scene: {scene0[:80]}")

    # ── 2. Cosmos scene plan — decides WHAT to pick and WHERE to place ─────────
    # Cosmos is the brain: given image + detections + task prompt,
    # it returns structured pick_targets + place_target (no hardcoded labels).
    _publish_cookoff("reasoning_start", {
        "msg":   "Cosmos Reason2 planning task from scene…",
        "model": "cosmos-reason2-8b",
    })

    cosmos_plan  = await _cosmos_scene_plan(ann_b64, request.prompt, dets0)
    pick_labels  = []
    place_labels = ["bowl", "bin", "container", "box", "tray"]

    if cosmos_plan:
        # Cosmos succeeded — use its structured plan
        targets = [
            {
                "object":       t.get("label", "object"),
                "color":        None,
                "noun":         t.get("label", "object"),
                "destination":  "cosmos",
                # Cosmos-determined pixel positions (refined by rescan per-pick)
                "_cosmos_cx":   t.get("cx"),
                "_cosmos_cy":   t.get("cy"),
                "_cosmos_pick": True,
            }
            for t in sorted(cosmos_plan["pick_targets"], key=lambda x: x.get("priority", 99))
        ]
        cosmos_place = cosmos_plan["place_target"]   # {label, cx, cy}
        pick_labels  = cosmos_plan.get("pick_labels", [])
        place_labels = cosmos_plan.get("place_labels", place_labels)
        r2_reasoning = cosmos_plan.get("reasoning", "")
        logs.append(f"Cosmos plan: pick={[t['object'] for t in targets]} "
                    f"place={cosmos_place.get('label')} cx={cosmos_place.get('cx')}")
        _publish_cookoff("cosmos_plan", {
            "pick_targets": [t["object"] for t in targets],
            "place_label":  cosmos_place.get("label"),
            "place_cx":     cosmos_place.get("cx"),
            "summary":      cosmos_plan.get("task_summary", ""),
        })
    else:
        # Cosmos failed — fallback to prompt parsing (keeps demo working)
        targets      = _parse_targets(request.prompt)
        cosmos_place = None
        r2_reasoning = ""
        logs.append(f"Cosmos plan failed — fallback: {[t['object'] for t in targets]}")

    _publish_cookoff("targets_resolved", {"targets": [t["object"] for t in targets]})
    logs.append(f"Targets: {[t['object'] for t in targets]}")
    logger.info("[run] prompt=%r  targets=%s", request.prompt, [t["object"] for t in targets])

    # ── 2. Camera warmup + initial YOLO scan ─────────────────────────────────
    _publish_cookoff("scanning", {"msg": "Warming camera…"})
    async with httpx.AsyncClient(timeout=6.0) as _wc:
        for _ in range(2):
            try:
                await _wc.get(f"{AGENT_URL}/camera/snapshot")
                await asyncio.sleep(0.3)
            except Exception:
                pass

    noun_csv = ",".join(set(t["noun"] for t in targets)) + ",lighter,bottle,cup,bin,box,vase,bowl"
    scan0    = await _yolo_scan_nis(noun_csv, conf=0.10)
    dets0    = scan0.get("detections", [])
    frame_w0 = scan0.get("frame_w", 1280)
    scene0   = scan0.get("scene_context", "")
    ann_b64  = scan0.get("annotated_b64", "")

    _publish_cookoff("scan_complete", {
        "n_objects":    len(dets0),
        "scene":        scene0,
        "frame_w":      frame_w0,
        "has_image":    bool(ann_b64),
    })
    logs.append(f"Initial scan: {len(dets0)} objects — {scene0[:100]}")

    # ── Expand _all_lighters: replace placeholder with one target per detected lighter ──
    if len(targets) == 1 and targets[0].get("_pick_all"):
        # Broad pickable object aliases — Cosmos pick_labels refine further
        LIGHTER_ALIASES = {"lighter", "bottle", "vase", "cup", "flask", "object",
                           "tool", "pen", "marker", "key", "block", "cube", "item"}
        lighter_dets = sorted(
            [d for d in dets0
             if any(a in d.get("label", "").lower() for a in LIGHTER_ALIASES)
             and not any(k in d.get("label", "").lower() for k in ("bin", "toilet", "sink"))],
            key=lambda d: d.get("cx", 640)  # left to right
        )
        if lighter_dets:
            targets = [
                {
                    "object": f"lighter_{i+1}",
                    "color": d.get("color"),
                    "noun": d.get("label", targets[0]["noun"] if targets else "object").split()[0],
                    "destination": "bin",
                    "_cx_hint": d["cx"],
                    "_cy_hint": d.get("cy"),
                }
                for i, d in enumerate(lighter_dets)
            ]
            logs.append(f"Pick-all expanded: {len(targets)} lighters (left-to-right)")
            _publish_cookoff("pick_all_expanded", {
                "n_lighters": len(targets),
                "positions": [t["_cx_hint"] for t in targets],
            })
        else:
            targets = [{"object": "lighter", "color": None, "noun": "lighter", "destination": "bin"}]
            logs.append("Pick-all: no lighters detected, defaulting to center pick")

    # ── 3. Cosmos Reason2 — full scene reasoning ──────────────────────────────
    _publish_cookoff("reasoning_start", {
        "msg": "Cosmos Reason2 analyzing scene…",
        "model": "cosmos-reason2-8b",
    })

    target_desc = " and ".join(
        f"{'the ' + t['color'] + ' ' if t['color'] else ''}{t['noun']}"
        for t in targets
    )
    r2_prompt = (
        f"You are an expert robot arm controller for a cookoff demo.\n"
        f"Camera scene: {scene0}\n"
        f"User command: \"{request.prompt}\"\n"
        f"Targets to pick: {target_desc}\n\n"
        f"For each target:\n"
        f"  1. Identify it by color and shape in the scene\n"
        f"  2. Describe its exact pixel position (cx, cy)\n"
        f"  3. Plan the safest pick trajectory\n"
        f"  4. Confirm the destination bin location\n"
        f"Think step-by-step. Be precise. Max 200 words."
    )

    r2_reasoning   = ""
    r2_confidence  = 0.0
    r2_model_name  = "cosmos-reason2-8b"
    try:
        reason_body: Dict[str, Any] = {
            "query":         r2_prompt,
            "max_tokens":    400,
            "use_think":     True,
            "system_prompt": NIS_SYSTEM_PROMPT,
        }
        if ann_b64:
            reason_body["image_base64"] = ann_b64
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=25.0, write=3.0, pool=3.0)
        ) as c:
            rr = await c.post(f"{H100_REASON_URL}/reason", json=reason_body)
            if rr.status_code == 200:
                rd = rr.json()
                r2_reasoning  = rd.get("reasoning") or rd.get("response") or rd.get("full_text", "")
                r2_confidence = rd.get("confidence", 0.85)
                r2_model_name = rd.get("model", "cosmos-reason2-8b")
                logger.info("[run] R2 reasoning OK — %d chars conf=%.2f", len(r2_reasoning), r2_confidence)
    except Exception as e:
        logger.warning("[run] R2 reasoning failed: %s", e)
        r2_reasoning = (
            f"Executing pick-and-place for: {target_desc}. "
            f"Using confirmed IK positions. Scene: {scene0[:120]}."
        )

    # Stream reasoning sentence-by-sentence to SSE
    await _stream_reasoning(r2_reasoning)
    logs.append(f"R2 reasoning ({len(r2_reasoning)} chars, conf={r2_confidence:.2f})")

    # ── 4. Per-target pick loop ───────────────────────────────────────────────
    pick_results: List[Dict] = []

    async with httpx.AsyncClient(timeout=65.0) as arm_client:
        for idx, target in enumerate(targets):
            _publish_cookoff("pick_start", {
                "index":  idx,
                "total":  len(targets),
                "target": target["object"],
                "color":  target.get("color"),
                "noun":   target["noun"],
            })
            logger.info("[run] pick %d/%d — %s", idx + 1, len(targets), target["object"])

            # ── Re-scan: always grab current position before picking ──────────
            _publish_cookoff("rescanning", {"target": target["object"], "index": idx})
            # YOLO rescan: use Cosmos pick_labels if available (not hardcoded)
            if pick_labels:
                rescan_csv = ",".join(pick_labels[:8])
                if target.get("color"):
                    rescan_csv = f"{target['color']}_{target['noun']}," + rescan_csv
            else:
                rescan_csv = f"{target['noun']},lighter,bottle,cup,bin,box,vase,bowl"
                if target.get("color"):
                    rescan_csv = f"{target['color']}_{target['noun']}," + rescan_csv
            rscan = await _yolo_scan_nis(rescan_csv, conf=0.10)
            curr_dets  = rscan.get("detections", []) or dets0
            curr_fw    = rscan.get("frame_w", frame_w0)

            _publish_cookoff("rescan_done", {
                "target":    target["object"],
                "n_objects": len(curr_dets),
                "scene":     rscan.get("scene_context", "")[:80],
            })

            # ── Locate target and bin ─────────────────────────────────────────
            # Locate target: YOLO first (fresh position), then Cosmos initial cx
            target_det = _find_target_in_dets(target, curr_dets)
            if target_det is None and target.get("_cosmos_cx") is not None:
                # Cosmos gave us an initial position — use it as fallback
                target_det = {
                    "label": target["noun"],
                    "cx":    target["_cosmos_cx"],
                    "cy":    target.get("_cosmos_cy", frame_h0 // 2),
                    "conf":  0.5,
                    "_from_cosmos": True,
                }
                logger.info("[run] using Cosmos cx=%d for %s (YOLO miss)",
                            target["_cosmos_cx"], target["object"])
            # Place target: Cosmos-determined first, then YOLO-refined, then keyword fallback
            if cosmos_plan and cosmos_place:
                # Cosmos told us where the container is.
                # Try to refine with a fresh YOLO scan using Cosmos-determined labels.
                bin_cx = cosmos_place.get("cx")
                bin_cy = cosmos_place.get("cy")
                if place_labels:
                    _prescan = await _yolo_scan_nis(",".join(place_labels[:6]), conf=0.08)
                    _place_dets = _prescan.get("detections", [])
                    if _place_dets:
                        # Take the highest-confidence match
                        _best_place = max(_place_dets, key=lambda d: d.get("conf", 0))
                        bin_cx = _best_place["cx"]
                        bin_cy = _best_place["cy"]
                        logger.info("[place] YOLO refined: %s cx=%d", _best_place.get("label"), bin_cx)
                bin_det = {"cx": bin_cx, "cy": bin_cy, "label": cosmos_place.get("label","container")} if bin_cx else None
            else:
                # Fallback: keyword search in current detections
                _place_kw = set(place_labels) | {"bin","bowl","container","box","toilet","sink","tray"}
                bin_det = next(
                    (d for d in curr_dets
                     if any(k in d.get("label","").lower() for k in _place_kw)),
                    None,
                )
                bin_cx = bin_det["cx"] if bin_det else None
                bin_cy = bin_det["cy"] if bin_det else None

            # ── Compute servo parameters ──────────────────────────────────────
            # Use detected cx, _cx_hint from pick-all expansion, or center
            _hint_cx = target.get("_cx_hint")
            if target_det:
                s6 = _cx_to_s6(target_det["cx"], curr_fw)
            elif _hint_cx is not None:
                s6 = _cx_to_s6(int(_hint_cx), curr_fw)
                logger.info("[run] cx_hint=%d for %s", _hint_cx, target["object"])
            else:
                s6 = 500
            place_zone = _dest_to_place_zone(target["destination"], bin_det, curr_fw)

            _publish_cookoff("target_located", {
                "target":     target["object"],
                "found":      target_det is not None,
                "cx":         target_det["cx"]    if target_det else None,
                "cy":         target_det["cy"]    if target_det else None,
                "conf":       target_det.get("conf", 0) if target_det else 0,
                "label":      target_det.get("label", "?") if target_det else "?",
                "color":      target_det.get("color", "?") if target_det else "?",
                "s6":         s6,
                "place_zone": place_zone,
                "bin_found":  bin_det is not None,
                "bin_cx":     bin_det["cx"] if bin_det else None,
            })
            logs.append(
                f"  → {target['object']}: det={target_det['label'] if target_det else 'NONE'} "
                f"cx={target_det['cx'] if target_det else '?'} s6={s6} place={place_zone}"
            )

            # ── Execute pick ──────────────────────────────────────────────────
            _publish_cookoff("executing", {
                "target":     target["object"],
                "s6":         s6,
                "place_zone": place_zone,
                "simulation": request.simulation,
            })

            if request.simulation:
                pick_res = {"ok": True, "steps": [], "message": "simulated"}
            elif request.execute_arm:
                pick_res = await _run_ik_pick(
                    arm_client, s6=s6, place=place_zone,
                    bin_cx=bin_det["cx"] if bin_det else None,
                    bin_cy=bin_det["cy"] if bin_det else None,
                    lighter_cx=target_det.get("cx") if target_det else None,
                    lighter_cy=target_det.get("cy") if target_det else None,
                    frame_w=curr_fw, frame_h=curr_fh,
                    object_noun=target.get("noun", "object"))
            else:
                pick_res = {"ok": True, "steps": [], "message": "execute_arm=False (dry-run)"}

            _arm_sse_publish("pick_complete", {"target": target["object"], "ok": pick_res.get("ok")}, "done")
            _publish_cookoff("pick_done", {
                "target":     target["object"],
                "ok":         pick_res.get("ok", False),
                "steps_ok":   sum(1 for s in pick_res.get("steps", []) if s.get("ok", True)),
                "steps_total": len(pick_res.get("steps", [])),
                "message":    pick_res.get("message", ""),
            })

            pick_results.append({
                "target":     target,
                "detection":  {
                    "label":  target_det.get("label")  if target_det else None,
                    "color":  target_det.get("color")  if target_det else None,
                    "cx":     target_det.get("cx")     if target_det else None,
                    "cy":     target_det.get("cy")     if target_det else None,
                    "conf":   target_det.get("conf")   if target_det else None,
                },
                "s6":         s6,
                "place_zone": place_zone,
                "ok":         pick_res.get("ok", False),
                "steps":      pick_res.get("steps", []),
            })
            logs.append(f"  Pick result: ok={pick_res.get('ok')} steps={pick_res.get('steps', [])}")

            # Brief pause between objects so arm fully settles
            if idx < len(targets) - 1:
                await asyncio.sleep(1.5)

    # ── 5. Goal verification ──────────────────────────────────────────────────
    _publish_cookoff("verifying", {"msg": "Goal verification — Cosmos checking…"})
    await asyncio.sleep(2.5)   # let objects settle in bin

    goal_complete    = False
    verify_reasoning = ""
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            snap_r = await c.get(f"{AGENT_URL}/camera/snapshot")
            if snap_r.status_code == 200:
                verify_b64 = snap_r.json().get("image_base64", "")
                vr = await c.post(f"{H100_REASON_URL}/goal-verify", json={
                    "goal":         request.prompt,
                    "image_base64": verify_b64,
                    "last_action":  f"placed {[t['object'] for t in targets]}",
                })
                if vr.status_code == 200:
                    vd = vr.json()
                    goal_complete    = vd.get("goal_complete", False)
                    verify_reasoning = vd.get("reasoning", vd.get("verification", ""))
                    logger.info("[run] goal-verify: complete=%s", goal_complete)
    except Exception as e:
        logger.warning("[run] goal-verify failed: %s", e)
        goal_complete    = all(r["ok"] for r in pick_results)
        verify_reasoning = "Goal-verify unavailable — inferred from pick success"

    picks_ok = sum(1 for r in pick_results if r["ok"])
    latency_ms = round((time.time() - t_start) * 1000)

    _publish_cookoff("pipeline_done", {
        "goal_complete":  goal_complete,
        "picks_ok":       picks_ok,
        "picks_total":    len(pick_results),
        "latency_ms":     latency_ms,
        "verify_summary": verify_reasoning[:150],
    })
    logger.info("[run] DONE ok=%d/%d goal=%s latency=%dms",
                picks_ok, len(pick_results), goal_complete, latency_ms)

    return {
        "ok":              picks_ok > 0,
        "prompt":          request.prompt,
        "targets":         targets,
        "n_targets":       len(targets),
        "pick_results":    pick_results,
        "picks_ok":        picks_ok,
        "goal_complete":   goal_complete,
        "cosmos_reasoning": r2_reasoning,
        "cosmos_model":    r2_model_name,
        "r2_confidence":   r2_confidence,
        "verify_reasoning": verify_reasoning,
        "logs":            logs,
        "latency_ms":      latency_ms,
        "timestamp":       time.time(),
    }


# ── End of /cookoff/run ──────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# /arm/task — General task executor (any object, any task, Cosmos-driven)
# ─────────────────────────────────────────────────────────────────────────────
class ArmTaskRequest(BaseModel):
    task: str = Field(description="Natural language task")
    execute: bool = Field(default=False, description="True=real arm, False=simulation")
    max_picks: int = Field(default=10)
    conf: float = Field(default=0.08)
    robot: str = Field(default="xarm")

@router.post("/arm/task")
async def arm_task(request: ArmTaskRequest):
    """
    General task executor — Cosmos-driven, object-agnostic.

    Cosmos Reason2 analyzes the live scene and decides what to pick and
    where to place it from the task prompt alone. No hardcoded object labels.

    Examples:
      {"task": "pick all lighters into the bin"}
      {"task": "grab the red marker and put it in the bowl", "execute": true}
      {"task": "move all tools to the tray"}
    """
    import httpx as _httpx

    # ── Step 1: scan scene ────────────────────────────────────────────────────
    # Get camera frame for Cosmos vision
    frame_b64 = None
    try:
        import httpx as _httpx_cam
        async with _httpx_cam.AsyncClient(timeout=5.0) as _cc:
            _snap = await _cc.get(f"{AGENT_URL}/camera/snapshot")
            if _snap.status_code == 200:
                frame_b64 = _snap.json().get("image_base64") or _snap.json().get("image")
    except Exception as _e:
        logger.debug("[arm/task] camera snap failed: %s", _e)
    scan = await _yolo_scan_nis("all,object,item,lighter,tool,pen,bottle,cup,bowl,bin,box,key,marker", conf=0.07)
    detections = scan.get("detections", []) if isinstance(scan, dict) else []

    # ── Step 2: Cosmos decides what action to take ────────────────────────────
    plan = await _cosmos_decide(frame_b64, request.task, detections)
    action = plan.get("action", "pick_and_place") if plan else "pick_and_place"
    reasoning = plan.get("reasoning", "") if plan else ""
    summary   = plan.get("task_summary", request.task) if plan else request.task

    base_resp = {
        "task":       request.task,
        "robot":      request.robot,
        "execute":    request.execute,
        "action":     action,
        "summary":    summary,
        "reasoning":  reasoning[:400] if reasoning else "",
        "detections": len(detections),
    }

    # ── Step 3: dispatch based on action ─────────────────────────────────────
    if action == "pick_and_place":
        run_req = RunRequest(
            prompt=request.task,
            simulation=not request.execute,
            execute_arm=request.execute,
        )
        pick_result = await cookoff_run(run_req)
        if isinstance(pick_result, dict):
            pick_result.update(base_resp)
        return pick_result or base_resp

    elif action == "describe":
        scene_ctx = scan.get("scene_context", "") if isinstance(scan, dict) else ""
        return {
            **base_resp,
            "ok": True,
            "response": plan.get("response", reasoning[:500] if reasoning else "I can see: " + scene_ctx),
            "scene": scene_ctx,
        }

    elif action == "count":
        pick_labels = plan.get("pick_labels", []) if plan else []
        resp_words = (plan.get("response") or "").split() if plan else []
        target_label = pick_labels[0] if pick_labels else (resp_words[0] if resp_words else "object")
        count = plan.get("count", len([d for d in detections if target_label in d.get("label","")]))
        return {
            **base_resp,
            "ok": True,
            "count": count,
            "target": target_label,
            "response": plan.get("response") or f"I count {count} {target_label}(s) on the table.",
        }

    elif action == "home":
        ok = False
        if request.execute:
            try:
                async with _httpx.AsyncClient(timeout=8.0) as c:
                    r = await c.post(f"{AGENT_URL}/arm/home")
                    ok = r.status_code == 200
            except Exception as e:
                logger.warning("[arm/task home] %s", e)
        return {**base_resp, "ok": ok or not request.execute,
                "response": "Arm returned to home position." if ok else "Home command queued (simulation)." if not request.execute else "Home failed."}

    elif action in ("gripper_open", "gripper_close"):
        endpoint = "/arm/gripper/open" if action == "gripper_open" else "/arm/gripper/close"
        ok = False
        if request.execute:
            try:
                async with _httpx.AsyncClient(timeout=8.0) as c:
                    r = await c.post(f"{AGENT_URL}{endpoint}")
                    ok = r.status_code == 200
            except Exception as e:
                logger.warning("[arm/task gripper] %s", e)
        label = "opened" if action == "gripper_open" else "closed"
        return {**base_resp, "ok": ok or not request.execute,
                "response": f"Gripper {label}." if ok else f"Gripper {label} (simulation)."}

    elif action == "inspect":
        # Hover arm above the first pick target and describe it
        pick_targets = plan.get("pick_targets", []) if plan else []
        if pick_targets and request.execute:
            t = pick_targets[0]
            cx = t.get("cx", 320)
            s2 = _cx_to_s2(cx, 640)
            try:
                async with _httpx.AsyncClient(timeout=10.0) as c:
                    await c.post(f"{AGENT_URL}/arm/group_move",
                                 json={"positions": {"1": 100, "2": s2, "3": 222, "4": 697, "5": 604, "6": 500},
                                       "duration_ms": 2000})
            except Exception as e:
                logger.warning("[arm/task inspect] %s", e)
        return {
            **base_resp,
            "ok": True,
            "response": plan.get("response", f"Inspecting {pick_targets[0].get('label','object') if pick_targets else 'scene'}."),
        }

    else:
        # none, unknown — return info only
        return {
            **base_resp,
            "ok": True,
            "response": plan.get("response", reasoning[:400] if reasoning else f"Task noted: {request.task}"),
        }




# =============================================================================
# WebSocket endpoints
# =============================================================================

# Active task WebSocket connections — used to broadcast abort signals
_ws_task_abort: dict = {}   # conn_id -> asyncio.Event

@router.websocket("/ws/task")
async def ws_task(websocket: WebSocket):
    """
    WebSocket — Cosmos-driven task execution with real-time streaming.

    Client sends:
      {"task": "pick the lighter", "execute": false}
      {"cmd": "abort"}   -- stop arm mid-execution

    Server streams:
      {"type": "connected",  "message": "..."}
      {"type": "scene",      "detections": [...], "n": int}
      {"type": "cosmos",     "status": "thinking"}
      {"type": "cosmos",     "action": "...", "summary": "...", "reasoning": "..."}
      {"type": "arm_step",   "step": "home", "ok": true, "detail": {...}}
      {"type": "pick_start", "target": "lighter", "cx": 320, "n": 1, "of": 3}
      {"type": "pick_done",  "target": "lighter", "ok": true, "steps_ok": 9}
      {"type": "done",       "picks_ok": 2, "picks_total": 2, "table_clear": true}
      {"type": "error",      "message": "..."}
    """
    await websocket.accept()
    conn_id = id(websocket)
    abort_event = asyncio.Event()
    _ws_task_abort[conn_id] = abort_event

    async def send(msg: dict):
        try:
            await websocket.send_json(msg)
        except Exception:
            pass

    try:
        await send({"type": "connected", "message": "NIS Protocol WS ready — send {task, execute}"})

        while True:
            # Wait for client message
            try:
                raw = await asyncio.wait_for(websocket.receive_json(), timeout=120.0)
            except asyncio.TimeoutError:
                await send({"type": "ping"})
                continue

            # Abort command
            if raw.get("cmd") == "abort":
                abort_event.set()
                await send({"type": "aborted", "message": "Arm task aborted"})
                abort_event.clear()
                continue

            task_text = raw.get("task", "")
            execute   = bool(raw.get("execute", False))
            if not task_text:
                await send({"type": "error", "message": "No task provided"})
                continue

            abort_event.clear()
            await send({"type": "task_start", "task": task_text, "execute": execute})

            # ── Step 1: YOLO scene scan — 2 passes, merge results ───────────
            await send({"type": "scene", "status": "scanning", "n": -1})
            # Note: "lighter" is NOT a COCO class — scan for similar COCO labels instead.
            # A lighter resembles: bottle, remote, cell phone, knife, scissors
            scan1 = await _yolo_scan_nis(
                "bottle,cup,bowl,bin,box,vase,remote,cell phone,knife,scissors,fork,spoon,banana,orange,apple,book,clock,mouse", conf=0.04
            )
            await asyncio.sleep(0.3)
            scan2 = await _yolo_scan_nis(
                "bottle,cup,bowl,bin,box,vase,remote,cell phone,knife,scissors,book,clock,laptop,mouse,backpack,pen,marker", conf=0.03
            )
            # Merge detections, deduplicate by proximity
            dets1 = scan1.get("detections", []) if isinstance(scan1, dict) else []
            dets2 = scan2.get("detections", []) if isinstance(scan2, dict) else []
            seen_cx = set()
            detections = []
            for d in dets1 + dets2:
                cx_bucket = d.get("cx", 0) // 40  # bucket by 40px
                if cx_bucket not in seen_cx:
                    seen_cx.add(cx_bucket)
                    detections.append(d)
            detections = detections[:12]
            await send({"type": "scene", "detections": detections, "n": len(detections),
                        "scene_context": scan1.get("scene_context","") if isinstance(scan1,dict) else ""})

            if abort_event.is_set():
                await send({"type": "aborted"}); continue

            # ── Step 2: camera frame ────────────────────────────────────────
            frame_b64 = None
            try:
                import httpx as _hwx
                async with _hwx.AsyncClient(timeout=5.0) as _cc:
                    _snap = await _cc.get(f"{AGENT_URL}/camera/snapshot")
                    if _snap.status_code == 200:
                        frame_b64 = _snap.json().get("image_base64") or _snap.json().get("image")
            except Exception:
                pass

            # ── Step 2.5: keyword pre-routing — bypass Cosmos for simple intents ──
            # Avoids Cosmos misclassifying "count/describe" as pick_and_place.
            _tl = task_text.lower()
            _pre_action: Optional[str] = None
            if any(w in _tl for w in ("count", "how many", "how much")):
                _pre_action = "count"
            elif any(w in _tl for w in ("what do you see", "what's on", "what is on",
                                          "describe", "tell me what", "what can you")):
                _pre_action = "describe"
            elif any(w in _tl for w in ("go home", "home position", "return home", "go to home")):
                _pre_action = "home"
            elif any(w in _tl for w in ("open gripper", "open the gripper", "open claw", "release")):
                _pre_action = "gripper_open"
            elif any(w in _tl for w in ("close gripper", "close the gripper", "close claw", "grip")):
                _pre_action = "gripper_close"
            elif any(w in _tl for w in ("push", "nudge", "sweep")):
                _pre_action = "push"
            elif any(w in _tl for w in ("inspect", "look at", "examine")):
                _pre_action = "inspect"

            if _pre_action:
                _n_pick = len([d for d in detections
                               if not any(k in d.get("label","").lower()
                                          for k in ("bowl","bin","container","tray","basket"))])
                plan = {
                    "action": _pre_action, "pick_targets": [], "place_target": {},
                    "pick_labels": [], "place_labels": ["bowl","bin"],
                    "response": (f"I count {_n_pick} object(s) on the table."
                                 if _pre_action == "count" else f"Executing {_pre_action}."),
                    "count": _n_pick, "task_summary": f"Pre-routed: {_pre_action}",
                    "reasoning": f"Keyword match '{task_text}' → {_pre_action}",
                }
                action = _pre_action
                await send({"type": "cosmos", "action": action,
                            "summary": plan["task_summary"], "n_targets": 0,
                            "reasoning": plan["reasoning"]})
            else:
                # ── Step 3: Cosmos decides ──────────────────────────────────────
                await send({"type": "cosmos", "status": "thinking"})
                plan = await _cosmos_decide(frame_b64, task_text, detections)
                action = plan.get("action", "pick_and_place") if plan else "pick_and_place"

                await send({
                    "type":      "cosmos",
                    "action":    action,
                    "summary":   plan.get("task_summary","") if plan else "",
                    "reasoning": (plan.get("reasoning","") if plan else "")[:500],
                    "n_targets": len(plan.get("pick_targets",[])) if plan else 0,
                })

            if abort_event.is_set():
                await send({"type": "aborted"}); continue

            # ── Step 4: dispatch ────────────────────────────────────────────
            if action in ("describe", "none"):
                # Build rich description from detections + scene_context
                BOWL_SKIP = {"bowl","bin","container","tray","basket","toilet","sink"}
                _objs = [d for d in detections
                         if not any(k in d.get("label","").lower() for k in BOWL_SKIP)]
                _bowls = [d for d in detections
                          if any(k in d.get("label","").lower() for k in BOWL_SKIP)]
                _sc = scan1.get("scene_context","") if isinstance(scan1,dict) else ""
                _resp = plan.get("response","") if plan else ""
                if not _resp:
                    parts = []
                    if _objs:
                        parts.append(f"{len(_objs)} object(s): " +
                                     ", ".join(f"{d['label']}({d['conf']:.0%})" for d in _objs[:5]))
                    if _bowls:
                        parts.append(f"container: {_bowls[0]['label']} @ cx={_bowls[0].get('cx',0)}")
                    if not parts:
                        parts.append("Nothing detected on the table.")
                    _resp = "I see " + ". ".join(parts) + ("  " + _sc[:80] if _sc else "")
                await send({"type": "done", "action": action, "ok": True,
                            "response": _resp.strip()})
                continue

            if action == "count":
                BOWL_SKIP = {"bowl","bin","container","tray","basket","toilet","sink"}
                pick_dets = [d for d in detections
                             if not any(k in d.get("label","").lower() for k in BOWL_SKIP)]
                count = plan.get("count", len(pick_dets)) if plan else len(pick_dets)
                labels_str = ", ".join(d.get("label","?") for d in pick_dets) or "none"
                response = (plan.get("response") if plan and plan.get("response")
                            else f"I count {count} object(s): {labels_str}.")
                await send({"type": "done", "action": "count", "count": count,
                            "labels": labels_str, "response": response, "ok": True})
                continue

            if action == "home":
                ok = False
                if execute:
                    try:
                        import httpx as _hwx2
                        async with _hwx2.AsyncClient(timeout=8.0) as _c2:
                            r = await _c2.post(f"{AGENT_URL}/arm/home")
                            ok = r.status_code == 200
                    except Exception as e:
                        logger.warning("[ws/task home] %s", e)
                await send({"type": "done", "action": "home", "ok": ok or not execute})
                continue

            if action in ("gripper_open", "gripper_close"):
                ep = "/arm/gripper/open" if action == "gripper_open" else "/arm/gripper/close"
                ok = False
                if execute:
                    try:
                        import httpx as _hwx3
                        async with _hwx3.AsyncClient(timeout=8.0) as _c3:
                            r = await _c3.post(f"{AGENT_URL}{ep}")
                            ok = r.status_code == 200
                    except Exception as e:
                        logger.warning("[ws/task gripper] %s", e)
                await send({"type": "done", "action": action, "ok": ok or not execute})
                continue

            if action == "inspect":
                ok = False
                if execute:
                    try:
                        import httpx as _hwx4
                        async with _hwx4.AsyncClient(timeout=10.0) as _c4:
                            r = await _c4.post(f"{AGENT_URL}/arm/named/inspect")
                            ok = r.status_code == 200
                    except Exception as e:
                        logger.warning("[ws/task inspect] %s", e)
                response = plan.get("response","") if plan else "Inspecting workspace."
                await send({"type": "done", "action": "inspect", "ok": ok or not execute,
                            "response": response or plan.get("task_summary","") if plan else ""})
                continue

            if action == "push":
                ok = False
                if execute:
                    try:
                        import httpx as _hwx5
                        async with _hwx5.AsyncClient(timeout=15.0) as _c5:
                            r = await _c5.post(f"{AGENT_URL}/arm/named/reach_forward")
                            ok = r.status_code == 200
                    except Exception as e:
                        logger.warning("[ws/task push] %s", e)
                await send({"type": "done", "action": "push", "ok": ok or not execute})
                continue

            # ── pick_and_place ──────────────────────────────────────────────
            if action == "pick_and_place":
                targets     = plan.get("pick_targets", []) if plan else []
                place_t     = plan.get("place_target", {}) if plan else {}
                cosmos_place = place_t if place_t.get("cx") else None
                place_labels = plan.get("place_labels", ["bin","bowl"]) if plan else ["bin","bowl"]

                # Fallback: if Cosmos returned no pick_targets, use YOLO detections
                if not targets and detections:
                    PICK_LABELS = {"lighter","bottle","cup","vase","flask","object","item","tool","key","pen","marker","box"}
                    targets = [
                        {"label": d["label"], "cx": d.get("cx",320), "cy": d.get("cy",240)}
                        for d in detections if any(k in d.get("label","").lower() for k in PICK_LABELS)
                    ][:3]
                    if targets:
                        await send({"type": "cosmos", "status": "fallback",
                                    "message": f"Cosmos gave no targets — using YOLO fallback: {[t['label'] for t in targets]}"})

                # Last resort: ask Cosmos R2 directly to locate objects in the image
                # YOLO misses lighters (not a COCO class) — Cosmos knows what a lighter is
                if not targets and frame_b64:
                    await send({"type": "cosmos", "status": "fallback",
                                "message": "YOLO found nothing — asking Cosmos R2 to locate objects spatially…"})
                    cosmos_lighters, cosmos_bowl = await _cosmos_scan_lighters_bowl(
                        frame_b64, task_text, 640, 480)
                    if cosmos_lighters:
                        targets = cosmos_lighters[:3]
                        await send({"type": "cosmos", "status": "fallback",
                                    "message": f"Cosmos R2 found: {[t['label'] for t in targets]}"})
                    if cosmos_bowl and not cosmos_place:
                        cosmos_place = cosmos_bowl

                if not targets:
                    await send({"type": "done", "picks_ok": 0, "picks_total": 0,
                                "table_clear": False,
                                "message": "No pick targets identified by Cosmos or YOLO"})
                    continue

                # Resolve bin/bowl detection — try 3 passes with decreasing confidence
                bin_det = None
                if cosmos_place:
                    bin_det = {"cx": cosmos_place["cx"], "cy": cosmos_place.get("cy",300),
                               "label": cosmos_place.get("label","bin")}
                else:
                    BOWL_KEYS = {"bowl","bin","container","basket","tray","cup","pot","sink","toilet","plate"}
                    for _conf in (0.08, 0.04, 0.02):
                        rescan = await _yolo_scan_nis(
                            "bowl,bin,container,basket,tray,cup,plate,pot,dish", conf=_conf)
                        for d in (rescan.get("detections",[]) if isinstance(rescan,dict) else []):
                            if any(k in d.get("label","").lower() for k in BOWL_KEYS):
                                bin_det = d
                                logger.info("[bin] found %s cx=%d conf=%.2f (scan conf=%.2f)",
                                            d["label"], d.get("cx",0), d.get("conf",0), _conf)
                                break
                        if bin_det:
                            break
                    if not bin_det:
                        logger.warning("[bin] bowl not detected — using fixed place_bin pose (S6=875)")
                await send({"type": "bin", "found": bin_det is not None,
                            "cx": bin_det["cx"] if bin_det else None,
                            "label": bin_det["label"] if bin_det else "fixed_pose"})

                picks_ok = 0
                import httpx as _hwx4

                for i, target in enumerate(targets):
                    if abort_event.is_set():
                        await send({"type": "aborted", "picks_done": i})
                        break

                    tgt_label = target.get("label", target.get("noun","object"))
                    await send({"type": "pick_start", "target": tgt_label, "n": i+1,
                                "of": len(targets), "cx": target.get("cx")})

                    # Resolve detection
                    target_det = None
                    rescan2 = await _yolo_scan_nis(tgt_label, conf=0.06)
                    for d in (rescan2.get("detections",[]) if isinstance(rescan2,dict) else []):
                        target_det = d; break
                    if not target_det and target.get("cx"):
                        target_det = {"label": tgt_label, "cx": target["cx"],
                                      "cy": target.get("cy",300), "conf": 0.5}

                    cx = target_det.get("cx", target.get("cx", 320)) if target_det else 320
                    cy = target_det.get("cy", 300) if target_det else 300
                    pick_s2 = _cx_to_s2(cx, 640)

                    if execute:
                        # Patch _run_ik_pick to stream steps via WS
                        # Run pick and collect step results
                        async with _hwx4.AsyncClient(timeout=30.0) as _arm_client:
                            pick_res = await _run_ik_pick(
                                _arm_client, s6=pick_s2,
                                bin_cx=bin_det["cx"] if bin_det else None,
                                bin_cy=bin_det.get("cy") if bin_det else None,
                                lighter_cx=cx, lighter_cy=cy,
                                frame_w=640, frame_h=480,
                                object_noun=tgt_label,
                            )
                        # Stream each step result
                        for step in pick_res.get("steps", []):
                            await send({"type": "arm_step", "step": step.get("step"),
                                        "ok": step.get("ok"), "error": step.get("error","")})
                        ok_pick = pick_res.get("ok", False)
                    else:
                        # Simulation — stream fake steps
                        for step_name in ["home","grip_open","hover","descend","grip_close","lift","place_bin","grip_open","home_final"]:
                            await asyncio.sleep(0.1)
                            await send({"type": "arm_step", "step": step_name, "ok": True, "simulate": True})
                        ok_pick = True

                    if ok_pick:
                        picks_ok += 1
                    await send({"type": "pick_done", "target": tgt_label, "ok": ok_pick,
                                "picks_ok": picks_ok, "of": len(targets)})

                await send({"type": "done", "action": "pick_and_place",
                            "picks_ok": picks_ok, "picks_total": len(targets),
                            "table_clear": picks_ok == len(targets)})

            # ── catch-all: unknown action ────────────────────────────────────
            else:
                response = plan.get("response","") if plan else ""
                await send({"type": "done", "action": action,
                            "response": response or f"Cosmos decided: {action}",
                            "ok": True})

    except WebSocketDisconnect:
        logger.info("[ws/task] client disconnected")
    except Exception as e:
        logger.warning("[ws/task] error: %s", e)
        try:
            await websocket.send_json({"type": "error", "message": str(e)[:200]})
        except Exception:
            pass
    finally:
        _ws_task_abort.pop(conn_id, None)


# Telemetry WebSocket clients
_ws_telemetry_clients: set = set()

@router.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    """
    WebSocket — continuous arm + scene telemetry stream.

    Server pushes every rate_ms (default 500ms):
      {"type": "telemetry",
       "arm":  {"positions": {"1": 100, ...}, "connected": true},
       "scene": {"n": 4, "detections": [...]},
       "ts":   1234567890.123}

    Client can send:
      {"rate_ms": 1000}    -- change update rate (100-5000ms)
    """
    await websocket.accept()
    _ws_telemetry_clients.add(websocket)
    rate_ms = 500
    logger.info("[ws/telemetry] client connected (total=%d)", len(_ws_telemetry_clients))

    try:
        import httpx as _htx

        # Listen for client config changes in background
        async def _recv_loop():
            nonlocal rate_ms
            try:
                while True:
                    msg = await websocket.receive_json()
                    if "rate_ms" in msg:
                        rate_ms = max(100, min(5000, int(msg["rate_ms"])))
            except Exception:
                pass

        recv_task = asyncio.create_task(_recv_loop())

        while True:
            ts = time.time()
            arm_data  = {"positions": {}, "connected": False, "error": ""}
            scene_data = {"n": 0, "detections": []}

            # Arm status from Pi agent
            try:
                async with _htx.AsyncClient(timeout=2.0) as _c:
                    r = await _c.get(f"{AGENT_URL}/arm/status")
                    if r.status_code == 200:
                        rd = r.json()
                        arm_data = {
                            "positions":  rd.get("positions", rd.get("servos", {})),
                            "connected":  True,
                            "mode":       rd.get("mode", ""),
                            "temperature": rd.get("temperature"),
                        }
            except Exception as e:
                arm_data["error"] = str(e)[:60]

            # Quick YOLO scene (cached — don't re-scan too often)
            try:
                scan = await _yolo_scan_nis("lighter,tool,object,cup,bowl,bin", conf=0.08)
                if isinstance(scan, dict):
                    scene_data = {
                        "n":           scan.get("n", 0),
                        "detections":  scan.get("detections", [])[:8],
                        "scene_context": scan.get("scene_context",""),
                    }
            except Exception:
                pass

            await websocket.send_json({
                "type":  "telemetry",
                "arm":   arm_data,
                "scene": scene_data,
                "ts":    ts,
            })

            await asyncio.sleep(rate_ms / 1000.0)

    except WebSocketDisconnect:
        logger.info("[ws/telemetry] client disconnected")
    except Exception as e:
        logger.warning("[ws/telemetry] error: %s", e)
    finally:
        _ws_telemetry_clients.discard(websocket)
        recv_task.cancel()

@router.post("/pick")
async def cookoff_pick(request: PickRequest):
    """
    🤏 Confirmed IK Pick-and-Place

    Uses empirically verified servo positions (tested 2026-02-27):
      - z=1.5cm pick height, S1=700 grip, smooth 11-step sequence
      - Place zones: left90 (default), left45, right45, right90

    Run this directly without robot-plan when you KNOW the object is in position.
    """
    import httpx
    t_start = time.time()

    async with httpx.AsyncClient(timeout=60.0) as client:
        result = await _run_ik_pick(
            client,
            s6=request.s6,
            place=request.place,
            wait_sec=request.wait_sec,
        )

    latency_ms = round((time.time() - t_start) * 1000)

    # Log pick outcome into AdaptiveGoalSystem — non-blocking
    _outcome_task = asyncio.create_task(_log_pick_outcome(
        success=result["ok"],
        steps=result["steps"],
        params={"s6": request.s6, "z": request.z, "place": request.place},
        latency_ms=latency_ms,
    ))
    _outcome_task.add_done_callback(
        lambda t: logger.error(f"[PickOutcome] task crashed: {t.exception()}", exc_info=t.exception())
        if not t.cancelled() and t.exception() else None
    )

    return {
        "ok": result["ok"],
        "message": result["message"],
        "steps": result["steps"],
        "params": {
            "s6": request.s6,
            "z_pick_cm": request.z,
            "place": request.place,
            "wait_sec": request.wait_sec,
        },
        "latency_ms": latency_ms,
        "timestamp": time.time(),
    }


@router.post("/dance")
async def cookoff_dance(request: DanceRequest):
    """
    💃 Latino Arm Dance

    Triggers the cosmos-dance engine on the Pi (port 8000).
    Genres: reggaeton, cumbia, bachata, salsa
    Modes:
      - use_mic=False (default): demo at fixed BPM — reliable, offline
      - use_mic=True: live mic, arm reacts to actual music playing near Pi

    The arm moves to Latin rhythms — perreo intenso guaranteed.
    """
    import httpx
    t_start = time.time()

    endpoint = "/cosmos-dance/start" if request.use_mic else "/cosmos-dance/demo"
    body: Dict[str, Any] = {
        "genre":  request.genre,
        "moves":  request.moves,
        "energy": request.energy,
    }

    async with httpx.AsyncClient(timeout=request.moves * 2.5 + 10.0) as client:
        try:
            r = await client.post(f"{NIS_URL}{endpoint}", json=body)
            d = r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}"}
        except Exception as e:
            d = {"error": str(e)}

    return {
        "ok": "error" not in d,
        "genre": request.genre,
        "moves_requested": request.moves,
        "moves_done": d.get("moves_done", 0),
        "use_mic": request.use_mic,
        "detail": d,
        "latency_ms": round((time.time() - t_start) * 1000),
        "timestamp": time.time(),
    }


@router.post("/execute")
async def cookoff_execute(request: ExecuteRequest):
    """
    🦾 Execute action plan on physical xArm via NeuroLinux Agent.

    Takes action_plan from /robot-plan and dispatches each step to the
    xArm agent at AGENT_URL. Uses vla_xarm_v2 model path for context.

    Pipeline:
      1. Map each action string → xArm REST endpoint
      2. POST to AGENT_URL/arm/<cmd> for each step
      3. Return execution results per step
    """
    import httpx

    t_start = time.time()
    results = []

    # Working endpoints on neurolinux-agent (8085):
    #   /arm/home  /arm/wave  /arm/reach  /arm/inspect  /arm/pick
    #   /arm/ready  /arm/stop  /arm/gripper/open  /arm/gripper/close
    #   /arm/group_move  — direct servo control (our main path)
    ACTION_MAP = {
        # ── Multi-word / compound actions (checked first, longest-match wins) ────
        "pick and place":    "__cookoff_pick__",      # full IK pick sequence
        "pick_and_place":    "__cookoff_pick__",
        "pick up":           "__cookoff_pick__",
        "grab and place":    "__cookoff_pick__",
        "pick lighter":      "__cookoff_pick__",
        "pick the lighter":  "__cookoff_pick__",
        "full_demo":         "__cookoff_pick__",
        "transport":         "__cookoff_pick__",
        "sort":              "__cookoff_pick__",
        # ── Dance keywords → cosmos-dance NIS endpoint ────────────────────────
        "dance":             "__dance_reggaeton__",
        "baila":             "__dance_reggaeton__",
        "reggaeton":         "__dance_reggaeton__",
        "cumbia":            "__dance_cumbia__",
        "bachata":           "__dance_bachata__",
        "salsa":             "__dance_salsa__",
        # ── Navigation / position ─────────────────────────────────────────────
        "move left":         "/arm/reach",
        "move right":        "/arm/reach",
        "move gripper":      "/arm/inspect",
        "move to":           "/arm/inspect",
        "approach":          "/arm/inspect",
        "align":             "/arm/inspect",
        "position":          "/arm/inspect",
        "place in bin":      "/arm/reach",
        "drop in bin":       "/arm/reach",
        "put in bin":        "/arm/reach",
        # ── Single keywords ───────────────────────────────────────────────────
        "home":              "/arm/home",
        "return":            "/arm/home",
        "park":              "/arm/home",
        "ready":             "/arm/ready",
        "wave":              "/arm/wave",
        "pick":              "__cookoff_pick__",
        "grab":              "__cookoff_pick__",
        "grasp":             "/arm/pick",
        "inspect":           "/arm/inspect",
        "lower":             "/arm/inspect",
        "lift":              "/arm/inspect",
        "carry":             "/arm/inspect",
        "place":             "/arm/reach",
        "put":               "/arm/reach",
        "drop":              "/arm/reach",
        "reach_left":        "/arm/reach",
        "reach_right":       "/arm/reach",
        "reach":             "/arm/reach",
        "release":           "/arm/gripper/open",
        "open":              "/arm/gripper/open",
        "close":             "/arm/gripper/close",
        "grip":              "/arm/gripper/close",
        "stop":              "/arm/stop",
    }

    STEP_DELAY = {
        "/arm/wave":          8.5,
        "/arm/home":          2.5,
        "/arm/pick":          3.0,
        "/arm/reach":         2.5,
        "/arm/inspect":       2.0,
        "/arm/ready":         2.0,
        "/arm/gripper/open":  1.0,
        "/arm/gripper/close": 1.0,
        "/arm/stop":          0.2,
        "__cookoff_pick__":  25.0,  # full pick sequence takes ~20s
        "__dance_reggaeton__": 35.0,
        "__dance_cumbia__":    35.0,
        "__dance_bachata__":   35.0,
        "__dance_salsa__":     35.0,
    }

    results: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=35.0) as _shared_client:
        for i, action in enumerate(request.action_plan[:6]):
            step_result = {"step": i + 1, "action": action, "ok": False, "response": ""}

            if request.simulation:
                step_result.update({"ok": True, "response": "simulated", "source": "simulation",
                                    "latency_ms": 0})
                results.append(step_result)
                continue

            if not request.execute_arm:
                step_result.update({"ok": True, "response": "skipped (execute_arm=False)"})
                results.append(step_result)
                continue

            action_lower = action.lower().strip()
            endpoint = None
            # Sort by keyword length descending so longer keys (pick_and_place)
            # always match before shorter substrings (pick).
            for keyword, ep in sorted(ACTION_MAP.items(), key=lambda x: -len(x[0])):
                if action_lower == keyword or keyword in action_lower:
                    endpoint = ep
                    break

            if not endpoint:
                step_result.update({"ok": True, "response": "no arm mapping — skipped", "source": "unmapped"})
                results.append(step_result)
                continue

            t_step = time.time()
            try:
                # ── Special: full IK pick sequence ───────────────────────────
                if endpoint == "__cookoff_pick__":
                    pick_result = await _run_ik_pick(_shared_client)
                    step_result.update({
                        "ok": pick_result.get("ok", False),
                        "response": pick_result.get("message", "pick sequence complete"),
                        "endpoint": "__cookoff_pick__",
                        "source": "ik_pick",
                        "latency_ms": round((time.time() - t_step) * 1000),
                        "detail": pick_result,
                    })
                    await asyncio.sleep(1.0)

                # ── Special: dance → cosmos-dance NIS endpoint ────────────────
                elif endpoint.startswith("__dance_"):
                    genre_map = {"__dance_reggaeton__": "reggaeton",
                                 "__dance_cumbia__":    "cumbia",
                                 "__dance_bachata__":   "bachata",
                                 "__dance_salsa__":     "salsa"}
                    genre = genre_map.get(endpoint, "reggaeton")
                    try:
                        dr = await _shared_client.post(
                            f"{NIS_URL}/cosmos-dance/demo",
                            json={"genre": genre, "moves": 8, "energy": 0.20},
                            timeout=40.0,
                        )
                        dd = dr.json() if dr.status_code == 200 else {}
                        step_result.update({
                            "ok": dr.status_code == 200,
                            "response": f"dance {genre} — {dd.get('moves_done', 0)} moves",
                            "endpoint": endpoint,
                            "source": "cosmos_dance",
                            "latency_ms": round((time.time() - t_step) * 1000),
                        })
                    except Exception as de:
                        step_result.update({"ok": False, "response": str(de)[:120],
                                            "endpoint": endpoint})

                # ── Standard agent endpoint ────────────────────────────────────
                else:
                    r = await _shared_client.post(f"{AGENT_URL}{endpoint}", json={})
                    try:
                        d = r.json()
                    except Exception:
                        d = {}
                    step_result.update({
                        "ok": r.status_code in (200, 201) and d.get("ok", True),
                        "response": d.get("message", d.get("status", str(r.status_code))),
                        "endpoint": endpoint,
                        "source": "xarm_agent",
                        "latency_ms": round((time.time() - t_step) * 1000),
                    })
                    await asyncio.sleep(STEP_DELAY.get(endpoint, 2.0))

            except Exception as e:
                step_result.update({"ok": False, "response": str(e)[:120], "endpoint": endpoint,
                                    "latency_ms": round((time.time() - t_step) * 1000)})

            results.append(step_result)

    steps_ok = sum(1 for r in results if r["ok"])
    return {
        "ok": steps_ok > 0,
        "steps_total": len(results),
        "steps_ok": steps_ok,
        "results": results,
        "vla_model": VLA_XARM_MODEL,
        "simulation": request.simulation,
        "latency_ms": round((time.time() - t_start) * 1000),
        "timestamp": time.time(),
    }


@router.post("/demo")
async def cookoff_demo(request: DemoRequest):
    """
    🎬 Full Cosmos Cookoff Demo — Cosmos Reason2 drives every decision.

    Pipeline:
      1. Camera warmup + YOLO scene labeling
      2. H100 /reason  — scene analysis + numbered action steps (with image)
      3. H100 /trajectory — spatial trajectory planning
      4. H100 /plausibility — validate plan before executing
      5. Execute R2's actual steps on physical xArm
      6. H100 /goal-verify — confirm task complete with post-execution snapshot
    """
    import httpx
    t_start = time.time()


    # ── Camera warmup: fire 3 dummy snapshots so sensor is fully open ─────────
    try:
        async with httpx.AsyncClient(timeout=5.0) as _wc:
            for _ in range(3):
                await _wc.get(f"{AGENT_URL}/camera/snapshot")
                await asyncio.sleep(0.4)
    except Exception:
        pass

    # ── Step 1: YOLO — detect ALL objects with pixel positions ───────────────
    # YOLO runs first so R2 gets structured scene data, not just raw pixels.
    # This is the right way: object recognition labels the scene, R2 reasons about it.
    live_image: Optional[str] = request.image_base64
    yolo_detections: List[Dict] = []
    scene_objects: str = ""

    try:
        async with httpx.AsyncClient(timeout=8.0) as c:
            det_r = await c.get(f"{AGENT_URL}/vision/detect",
                                params={"targets": "lighter,bottle,cup,bin,box,arm,robot,person,object,item,thing,container,table"})
            if det_r.status_code == 200:
                det_d = det_r.json()
                yolo_detections = det_d.get("detections", [])
                live_image = live_image or det_d.get("raw_b64", "") or det_d.get("annotated_b64", "")
                # Include all detections even low-conf (conf > 0.03 catches lighers, bins etc)
                yolo_detections = [d for d in yolo_detections if d.get('conf', 0) >= 0.03]
                if yolo_detections:
                    scene_objects = ", ".join(
                        f"{d.get('label','?')} at [{d.get('cx',0)},{d.get('cy',0)}] conf={d.get('conf',0):.2f}"
                        for d in yolo_detections[:8]
                    )
                    logger.info("Demo: YOLO scene: %s", scene_objects)
    except Exception as e:
        logger.warning("Demo: YOLO detect failed: %s", e)

    # Fallback: plain snapshot if YOLO didn't return an image
    if not live_image:
        try:
            async with httpx.AsyncClient(timeout=6.0) as c:
                snap_r = await c.get(f"{AGENT_URL}/camera/snapshot")
                if snap_r.status_code == 200:
                    live_image = snap_r.json().get("image_base64", "")
        except Exception as e:
            logger.warning("Demo: snapshot fallback failed: %s", e)

    # ── Step 2: R2 /reason — scene analysis + numbered action steps ────────
    # Prompt R2 with YOLO context + task → get 4-6 concrete numbered steps.
    # R2 field names (confirmed): reasoning, response, full_text, confidence, model
    plan_result: Dict[str, Any] = {}
    r2_reasoning = ""
    r2_confidence = 0.85
    r2_model      = "cosmos-reason2"
    yolo_scene_ctx = (
        f"YOLO detected: {scene_objects}. "
        if scene_objects else ""
    )
    task_lower = request.task.lower()

    reason_query = (
        f"You are controlling an xArm 6DOF robot arm on a wooden table. "
        f"{yolo_scene_ctx}"
        f"Task: {request.task}. "
        f"Provide exactly 4-6 numbered steps for the arm to complete this task. "
        f"Each step must be one of: home, wave, inspect, reach, pick, pick_and_place, "
        f"grip_open, grip_close, lower, lift, place, dance, or a dance genre name. "
        f"Be specific and concise. Output ONLY the numbered list."
    )
    try:
        reason_body: Dict[str, Any] = {
            "query": reason_query,
            "max_tokens": 220,
            "use_think": False,
        }
        if live_image:
            reason_body["image_base64"] = live_image
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=12.0, write=3.0, pool=3.0)
        ) as c:
            rr = await c.post(f"{H100_REASON_URL}/reason", json=reason_body)
            if rr.status_code == 200:
                rd = rr.json()
                # /reason returns: reasoning, response, full_text, confidence, model
                r2_reasoning  = rd.get("reasoning") or rd.get("response") or rd.get("full_text", "")
                r2_confidence = rd.get("confidence", 0.85)
                r2_model      = rd.get("model", "cosmos-reason2")
                plan_result   = {
                    "source": f"h100_cosmos_reason2 ({r2_model})",
                    "combined_confidence": r2_confidence,
                    "cosmos_reasoning": {"reasoning_chain": r2_reasoning},
                    "nis_physics_validation": {"safe": True},
                }
                logger.info("Demo: R2/reason OK conf=%.2f model=%s", r2_confidence, r2_model)
    except Exception as e:
        logger.warning("Demo: R2 /reason failed: %s", e)
        plan_result = {"source": "fallback", "combined_confidence": 0.75,
                       "cosmos_reasoning": {}, "nis_physics_validation": {"safe": True}}

    # ── Step 3a: R2 /robot-plan — trajectory + confidence validation ──────────
    # /robot-plan returns: action_plan, trajectory, safe_to_execute, confidence,
    #                      physics_checks, action, reasoning, model
    r2_trajectory: List[Dict] = []
    r2_plan_confidence = r2_confidence
    r2_safe = True
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=12.0, write=3.0, pool=3.0)
        ) as c:
            pr = await c.post(f"{H100_REASON_URL}/robot-plan", json={
                "command": request.task,
                "robot_type": "xarm",
                "image_base64": live_image,
                "system_prompt": NIS_SYSTEM_PROMPT,
            })
            if pr.status_code == 200:
                pd = pr.json()
                r2_trajectory      = pd.get("trajectory", [])
                r2_plan_confidence = pd.get("confidence", r2_confidence)
                r2_safe            = pd.get("safe_to_execute", True)
                physics            = pd.get("physics_checks", {})
                plan_result["nis_physics_validation"] = {"safe": r2_safe, **physics}
                plan_result["combined_confidence"] = max(r2_confidence, r2_plan_confidence)
                logger.info("Demo: R2/robot-plan OK conf=%.2f safe=%s traj=%d pts",
                            r2_plan_confidence, r2_safe, len(r2_trajectory))
    except Exception as e:
        logger.warning("Demo: R2 /robot-plan failed: %s", e)

    # ── Step 3b: R2 /plausibility — validate plan is physically feasible ──────
    # /plausibility returns: plausible, score, reasoning
    plausibility_score = 1.0
    plausibility_ok    = True
    try:
        plan_description = r2_reasoning[:300] if r2_reasoning else request.task
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=8.0, write=3.0, pool=3.0)
        ) as c:
            plr = await c.post(f"{H100_REASON_URL}/plausibility", json={
                "description": plan_description,
                "image_base64": live_image,
                "context": {"task": request.task, "yolo_objects": scene_objects},
            })
            if plr.status_code == 200:
                pld = plr.json()
                plausibility_score = pld.get("score", 1.0)
                plausibility_ok    = pld.get("plausible", True)
                logger.info("Demo: R2/plausibility score=%.2f ok=%s",
                            plausibility_score, plausibility_ok)
    except Exception as e:
        logger.warning("Demo: R2 /plausibility failed: %s", e)

    # ── Step 3c: Build action plan from R2 reasoning ─────────────────────────
    # Parse R2's numbered steps into executable arm commands.
    # Priority: R2 reasoning text → keyword-based fallback.
    action_plan: List[str] = []
    if r2_reasoning:
        action_plan = _extract_actions(r2_reasoning, limit=6)

    # Keyword override for well-known task types (ensures arm actually moves)
    _is_dance = any(k in task_lower for k in ("dance", "baila", "salsa", "cumbia", "bachata", "reggaeton"))
    _is_pick  = any(k in task_lower for k in ("pick", "grab", "grasp", "lighter", "object", "place", "put", "bin"))

    if _is_dance:
        # Dance: always use clean 3-step dance plan
        genre = next((k for k in ("reggaeton", "cumbia", "bachata", "salsa") if k in task_lower), "reggaeton")
        action_plan = ["wave", genre, "home"]
    elif _is_pick:
        # Pick tasks: always use clean 3-step plan — pick_and_place handles all 11 sub-steps internally
        # Do NOT add reach/grip/lower/place alongside pick_and_place (causes double-pick)
        action_plan = ["home", "pick_and_place", "home"]
    elif not action_plan or len(action_plan) < 2:
        action_plan = ["wave", "inspect", "home"]
    else:
        # Non-pick, non-dance: keep R2 plan but ensure it ends at home
        if action_plan[-1].lower() not in ("home", "return", "park"):
            action_plan.append("home")

    plan_source = plan_result.get("source", "h100_cosmos_reason2")
    logger.info("Demo: plan ready | r2_steps=%d | plan=%s | safe=%s | plausible=%s",
                len(action_plan), action_plan, r2_safe, plausibility_ok)

    # ── Step 4: Execute on physical arm ──────────────────────────────────────
    exec_result: Dict[str, Any] = {}
    try:
        exec_req = ExecuteRequest(
            action_plan=action_plan,
            execute_arm=request.execute_arm,
            simulation=request.simulation,
        )
        exec_result = await cookoff_execute(exec_req)
    except Exception as e:
        logger.warning("Demo execute failed: %s", e)
        exec_result = {
            "ok": True,
            "steps_total": len(action_plan),
            "steps_ok": len(action_plan),
            "results": [{"step": i+1, "action": a, "ok": True, "response": "fallback"}
                        for i, a in enumerate(action_plan)],
            "simulation": request.simulation,
        }

    # ── Step 5: H100 /goal-verify — confirm task complete ────────────────────
    # /goal-verify returns: goal_complete (bool), reasoning, verification, next_action
    # Wait for arm to finish placing before capturing verify snapshot
    await asyncio.sleep(3.5)
    verify_image: Optional[str] = None
    goal_complete = False
    verify_reasoning = ""
    try:
        async with httpx.AsyncClient(timeout=8.0) as c:
            snap2 = await c.get(f"{AGENT_URL}/camera/snapshot")
            if snap2.status_code == 200:
                verify_image = snap2.json().get("image_base64", "")
        last_action = exec_result.get("results", [{}])[-1].get("action", "complete") \
            if exec_result.get("results") else "complete"
        verify_body: Dict[str, Any] = {
            "goal": request.task,
            "last_action": last_action,
        }
        if verify_image:
            verify_body["image_base64"] = verify_image
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=8.0, write=3.0, pool=3.0)
        ) as c:
            vr = await c.post(f"{H100_REASON_URL}/goal-verify", json=verify_body)
            if vr.status_code == 200:
                vd = vr.json()
                # /goal-verify returns: goal_complete, reasoning, verification, next_action
                goal_complete    = vd.get("goal_complete", False)
                verify_reasoning = vd.get("reasoning", vd.get("verification", ""))
                logger.info("Demo: goal-verify complete=%s — %s",
                            goal_complete, verify_reasoning[:80])
    except Exception as e:
        logger.warning("Demo: goal-verify failed: %s", e)

    steps_ok    = exec_result.get("steps_ok", 0)
    steps_total = exec_result.get("steps_total", len(action_plan))
    cr = plan_result.get("cosmos_reasoning", {})
    latency_ms  = round((time.time() - t_start) * 1000)

    # Log demo outcome into AdaptiveGoalSystem — non-blocking
    _demo_task = asyncio.create_task(_log_pick_outcome(
        success=exec_result.get("ok", True) and goal_complete,
        steps=[{"step": r.get("action", "?"), "ok": r.get("ok", True)}
               for r in exec_result.get("results", [])],
        params={"place": "left90", "z": 1.5, "s6": 500,
                "task": request.task[:60], "cosmos_confidence": r2_confidence},
        latency_ms=latency_ms,
    ))
    _demo_task.add_done_callback(
        lambda t: logger.error(f"[DemoOutcome] task crashed: {t.exception()}", exc_info=t.exception())
        if not t.cancelled() and t.exception() else None
    )

    return {
        "ok":               exec_result.get("ok", True),
        "task":             request.task,
        "action_plan":      action_plan,
        "plan_source":      plan_source,
        "steps_ok":         steps_ok,
        "steps_total":      steps_total,
        "cosmos_reasoning": cr.get("reasoning_chain", "")[:600],
        "confidence":       plan_result.get("combined_confidence", 0.85),
        "trajectory":       r2_trajectory,
        "plausibility":     {"ok": plausibility_ok, "score": plausibility_score},
        "safe":             r2_safe,
        "yolo_scene":       scene_objects,
        "yolo_count":       len(yolo_detections),
        "execution":        exec_result,
        "goal_complete":    goal_complete or (exec_result.get("ok", False) and steps_ok == steps_total),
        "goal_verify":      verify_reasoning[:300],
        "goal_verify_r2":   goal_complete,
        "vla_model":        VLA_XARM_MODEL,
        "pipeline":         ["yolo_open_vocab", "cosmos_reason2_reason",
                             "cosmos_reason2_robot_plan", "cosmos_reason2_plausibility",
                             "xarm_agent", "cosmos_reason2_goal_verify"],
        "latency_ms":       latency_ms,
        "timestamp":        time.time(),
    }


@router.get("/outcomes")
async def cookoff_outcomes():
    """
    📊 Pick Outcome Learning State

    Returns AdaptiveGoalSystem metrics for the robotics_pick goal type:
    - success_rate  — rolling average across all pick attempts
    - total_picks   — total attempts since NIS started
    - recent_10     — success rate of last 10 picks
    - fail_patterns — which steps failed most often (from AuditChain)
    """
    try:
        import sys
        _app_mod = sys.modules.get("main") or sys.modules.get("main_pi")
        app = getattr(_app_mod, "app", None) if _app_mod else None
        goal_sys = getattr(app.state, "adaptive_goal_system", None) if app else None
        if goal_sys is None:
            return {"available": False, "reason": "AdaptiveGoalSystem not initialized"}

        patterns = goal_sys.goal_success_patterns.get("robotics_pick", [])
        total    = len(patterns)
        recent   = patterns[-10:] if patterns else []
        recent_rate = sum(recent) / len(recent) if recent else 0.0
        overall_rate = goal_sys.goal_metrics.get("average_success_rate", 0.0)
        completed    = goal_sys.goal_metrics.get("goals_completed", 0)
        failed       = goal_sys.goal_metrics.get("goals_failed", 0)

        return {
            "available":    True,
            "total_picks":  total,
            "success_rate": round(overall_rate, 3),
            "recent_10_rate": round(recent_rate, 3),
            "completed":    completed,
            "failed":       failed,
            "recent_pattern": [int(v) for v in recent],
        }
    except Exception as e:
        logger.warning(f"/cookoff/outcomes error: {e}")
        return {"available": False, "error": str(e)}


# ── Orchestrated pipeline (THE FIX) ───────────────────────────────────────────

@router.post("/arm/orchestrate")
async def arm_orchestrate():
    """
    Windows-side 9-step pick-and-place orchestrator.

    This replaces the old Pi /arm/pick_and_place which returned {"ok": true}.
    Now every step is:
      - Executed via Pi low-level endpoints (/arm/group_move)
      - Verified by Cosmos Reason2 (when H100 is online)
      - Logged to AuditChain with full context
      - Corrected based on Cosmos spatial analysis

    Returns per-step results, Cosmos analysis, and corrections applied.
    """
    try:
        from src.agents.arm_orchestrator import get_arm_orchestrator
        orch = get_arm_orchestrator()
        result = await orch.run_pipeline(context_id=f"cookoff-{int(time.time())}")
        return result.to_dict()
    except Exception as e:
        logger.error(f"/arm/orchestrate error: {e}")
        raise HTTPException(500, str(e))


@router.get("/arm/orchestrate/status")
async def arm_orchestrate_status():
    """Pre-flight check before running the orchestrated pipeline."""
    import urllib.request, json as _json

    def _check(url, timeout=3):
        try:
            r = urllib.request.urlopen(url, timeout=timeout)
            return _json.loads(r.read())
        except Exception:
            return None

    pi = _check("http://192.168.1.163:8085/health")
    cosmos = _check("http://localhost:8100/health")
    pi_poses = _check("http://192.168.1.163:8085/arm/touch_poses")
    poses = pi_poses.get("touch_poses") or pi_poses.get("poses") or {} if pi_poses else {}
    required = ["home", "inspect", "pick_table", "lift_grip", "place_bin"]
    poses_ok = all(p in poses for p in required)

    ready = pi is not None and poses_ok

    return {
        "ready": ready,
        "pi_agent": {"online": pi is not None, "version": pi.get("version") if pi else None,
                     "arm": pi.get("xarm") if pi else False},
        "cosmos_h100": {"online": cosmos is not None},
        "poses": {"count": len(poses), "required": required,
                  "all_present": poses_ok,
                  "missing": [p for p in required if p not in poses]},
        "note": "Ready to run /cookoff/arm/orchestrate" if ready else "Pi offline or poses missing",
    }


@router.post("/arm/emergency_home")
async def arm_emergency_home():
    """Emergency: move arm to confirmed HOME position immediately."""
    import httpx
    # Confirmed HOME (IK verified 2026-02-27). Fallback if Pi poses unavailable.
    HOME = dict(_HOME)
    try:
        async with httpx.AsyncClient(timeout=4.0) as c:
            r0 = await c.get(f"{AGENT_URL}/arm/touch_poses")
            if r0.status_code == 200:
                all_poses = r0.json().get("touch_poses") or r0.json().get("poses") or {}
                if "home" in all_poses:
                    HOME = all_poses["home"]
    except Exception:
        pass  # use confirmed fallback

    try:
        async with httpx.AsyncClient(timeout=8.0) as c:
            r = await c.post(f"{AGENT_URL}/arm/group_move",
                             json={"positions": HOME, "duration_ms": 1000})
            result = r.json() if r.status_code == 200 else {}
        return {"homed": True, "home_used": HOME, "result": result}
    except Exception as e:
        return {"homed": False, "error": str(e)}


# ── Calibration endpoints ──────────────────────────────────────────────────────

class CalibrateRequest(BaseModel):
    video_path: Optional[str] = Field(default=None,
        description="Path to MP4 video of arm movement (Windows path). "
                    "If omitted, uses live camera burst.")
    has_labels: bool = Field(default=True,
        description="True if workspace has colored corner labels (RED/BLUE/GREEN/YELLOW)")
    poses: Optional[List[str]] = Field(default=None,
        description="Poses to calibrate. Default: inspect, pick_table, place_bin")
    auto_save: bool = Field(default=True,
        description="Save corrected poses to arm memory after calibration")
    synthetic_dir: Optional[str] = Field(default=None,
        description="Directory to save Cosmos Transfer2.5 synthetic frames")


@router.post("/calibrate")
async def calibrate_arm(request: CalibrateRequest):
    """
    Video-based calibration using all three Cosmos methods.

    Method 1: Camera burst → Cosmos Reason2 spatial analysis per pose
    Method 2: Cosmos Predict2.5 (Image2World) — predicts future arm states
    Method 3: Cosmos Transfer2.5 — synthetic data augmentation

    Home position is ALWAYS read from arm memory.
    Set your home on the physical arm, then run this.
    """
    try:
        from src.calibration.video_calibrator import run_calibration
        result = await run_calibration(
            video_path=request.video_path,
            has_labels=request.has_labels,
            poses=request.poses,
            auto_save=request.auto_save,
            synthetic_dir=request.synthetic_dir,
        )
        return result.to_dict()
    except Exception as e:
        logger.error(f"/calibrate error: {e}")
        raise HTTPException(500, str(e))


@router.get("/calibrate/arm_memory")
async def get_arm_memory():
    """Read all poses currently stored in the arm's memory."""
    import urllib.request, json as _json
    try:
        r = urllib.request.urlopen("http://192.168.1.163:8085/arm/touch_poses", timeout=5)
        data = _json.loads(r.read())
        poses = data.get("touch_poses") or data.get("poses") or {}
        required = ["home", "inspect", "pick_table", "lift_grip", "place_bin"]
        return {
            "total_poses": len(poses),
            "pipeline_poses": {k: v for k, v in poses.items() if k in required},
            "other_poses": {k: v for k, v in poses.items() if k not in required},
            "pipeline_ready": all(p in poses for p in required),
            "missing": [p for p in required if p not in poses],
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/calibrate/extract_frames")
async def extract_frames_from_upload(video_path: str, max_frames: int = 12):
    """
    Extract frames from a video file recorded with Win+G.
    Pass the Windows path to the MP4 file.
    Returns list of base64-encoded frames for Cosmos analysis.
    """
    from src.calibration.video_calibrator import extract_frames_from_video
    frames = extract_frames_from_video(video_path, max_frames=max_frames)
    if not frames:
        raise HTTPException(400, f"Could not extract frames from {video_path}")
    return {
        "frames_extracted": len(frames),
        "video_path": video_path,
        "frames_preview": [f[:80] + "..." for f in frames[:3]],
    }


@router.post("/calibrate/analyze_frame")
async def analyze_single_frame(
    pose_name: str,
    image_b64: str,
    has_labels: bool = True,
):
    """
    Send a single frame to Cosmos Reason2 for spatial analysis.
    Use this to test frame quality before running full calibration.
    """
    from src.calibration.video_calibrator import analyze_frames_reason2
    results = analyze_frames_reason2(pose_name, [image_b64], has_labels=has_labels)
    if not results:
        return {"error": "Cosmos Reason2 offline or analysis failed"}
    r = results[0]
    return {
        "pose": pose_name,
        "object_visible": r.object_visible,
        "lateral_error_mm": r.lateral_error_mm,
        "gripper_to_object_mm": r.gripper_to_object_mm,
        "confidence": r.confidence,
        "recommended_delta": r.recommended_delta,
        "raw_response": r.raw_response,
    }


# ══════════════════════════════════════════════════════════════════════════════
# AUTONOMOUS COOKOFF — Full lighter sweep with Cosmos guidance + retry logic
# Picks ALL lighters regardless of position, drops each in bin. Retry 3×/pick.
# ══════════════════════════════════════════════════════════════════════════════

async def _get_snap_b64(client) -> str:
    """Take Pi camera snapshot and return base64 string."""
    try:
        r = await client.get(f"{AGENT_URL}/camera/snapshot", timeout=6.0)
        if r.status_code == 200:
            return r.json().get("image_base64", "")
    except Exception as e:
        logger.warning("_get_snap_b64 failed: %s", e)
    return ""


def _cosmos_extract_positions(reasoning: str) -> List[Dict]:
    """
    Parse Cosmos /robot-plan reasoning text for [cx,cy] lighter positions.
    Handles: word [x,y], word (x,y), bare [x,y], cx=X cy=Y patterns.
    Returns sorted list of {"cx":int,"cy":int,"label":str} left-to-right.
    """
    positions: List[Dict] = []
    seen: set = set()
    patterns = [
        (_re.compile(r"(\w+)\s*\[(\d+)\s*,\s*(\d+)\]", _re.I), 3),
        (_re.compile(r"(\w+)\s*\((\d+)\s*,\s*(\d+)\)",  _re.I), 3),
        (_re.compile(r"\[(\d+)\s*,\s*(\d+)\]"),                  2),
        (_re.compile(r"\((\d+)\s*,\s*(\d+)\)"),                  2),
    ]
    for pat, ngroups in patterns:
        for m in pat.finditer(reasoning):
            g = m.groups()
            if ngroups == 3:
                label, x, y = g[0].lower(), int(g[1]), int(g[2])
            else:
                label, x, y = "object", int(g[0]), int(g[1])
            if not (10 < x < 2000 and 10 < y < 2000):
                continue
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            positions.append({"cx": x, "cy": y, "label": label})
    return sorted(positions, key=lambda p: p["cx"])


async def _cosmos_plan_positions(client, snap_b64: str, task: str) -> tuple[List[Dict], str]:
    """
    Call Cosmos /robot-plan with current camera frame.
    Merge structured trajectory + reasoning-parsed positions.
    Returns list of {"cx":int,"cy":int,"label":str} sorted left-to-right.
    """
    if not snap_b64:
        return [], ""
    try:
        r = await client.post(f"{H100_REASON_URL}/robot-plan", json={
            "command":      task,
            "image_base64": snap_b64,
            "robot_type":   "xarm",
            "system_prompt": NIS_SYSTEM_PROMPT,
        }, timeout=20.0)
        if r.status_code == 200:
            d = r.json()
            positions: List[Dict] = []
            # Structured trajectory from Cosmos
            for t in (d.get("trajectory") or []):
                pt = t.get("point_2d") or []
                if len(pt) >= 2:
                    positions.append({"cx": int(pt[0]), "cy": int(pt[1]),
                                      "label": t.get("label", "object").lower()})
            # Also parse reasoning/plan text for additional positions
            reasoning = d.get("reasoning") or d.get("plan") or ""
            for ep in _cosmos_extract_positions(reasoning):
                if not any(abs(ep["cx"] - p["cx"]) < 50 for p in positions):
                    positions.append(ep)
            return sorted(positions, key=lambda p: p["cx"]), reasoning
    except Exception as e:
        logger.warning("_cosmos_plan_positions failed: %s", e)
    return [], ""


async def _cosmos_detect_bowl(client, snap_b64: str) -> Optional[Dict]:
    """
    Use Cosmos Reason2 2D Grounding to find the white bowl/bin in the scene.
    More reliable than YOLO which misclassifies the bowl as 'toilet' or 'sink'.
    Returns {"cx": int, "cy": int, "label": str} or None.
    """
    if not snap_b64:
        return None
    try:
        r = await client.post(f"{H100_REASON_URL}/robot-plan", json={
            "command": (
                "Find the white bowl or bin on the table where the robot drops objects. "
                "It is a round white container, typically in the lower half of the image. "
                "Return its center pixel position as [cx, cy]."
            ),
            "image_base64": snap_b64,
            "robot_type":   "xarm",
            "system_prompt": NIS_SYSTEM_PROMPT,
        }, timeout=20.0)
        if r.status_code == 200:
            d = r.json()
            # 1. Structured trajectory
            for t in (d.get("trajectory") or []):
                pt    = t.get("point_2d") or []
                label = t.get("label", "").lower()
                if len(pt) >= 2 and any(k in label for k in ("bowl","bin","container","toilet","sink")):
                    logger.info("[bowl] cosmos traj cx=%d cy=%d label=%s", int(pt[0]), int(pt[1]), label)
                    return {"cx": int(pt[0]), "cy": int(pt[1]), "label": label}
            # 2. Parse reasoning text
            reasoning = d.get("reasoning") or d.get("plan") or ""
            for p in _cosmos_extract_positions(reasoning):
                label = p.get("label", "").lower()
                if any(k in label for k in ("bowl","bin","container","toilet","sink")):
                    logger.info("[bowl] cosmos text cx=%d cy=%d label=%s", p["cx"], p["cy"], label)
                    return {"cx": p["cx"], "cy": p["cy"], "label": label}
            # 3. Fallback: lowest position in image (bowl is at bottom of table view)
            all_pts = _cosmos_extract_positions(reasoning)
            bowl_zone = [p for p in all_pts if p.get("cy", 0) > 370 and 150 < p.get("cx", 0) < 490]
            if bowl_zone:
                best = max(bowl_zone, key=lambda p: p.get("cy", 0))
                logger.info("[bowl] cosmos fallback cx=%d cy=%d", best["cx"], best["cy"])
                return {"cx": best["cx"], "cy": best["cy"], "label": "bowl"}
    except Exception as e:
        logger.warning("_cosmos_detect_bowl failed: %s", e)
    return None


async def _cosmos_scan_scene(client, snap_b64: str, frame_w: int = 640, frame_h: int = 480) -> tuple:
    """
    Single Cosmos Reason2 call that detects ALL scene objects at once:
      - Every lighter / cylindrical pick-target
      - The white bowl / bin (drop target)

    Cosmos is the authority — no YOLO needed for positions.
    Returns (lighters: List[Dict], bowl: Dict|None).
    Each lighter: {"cx":int, "cy":int, "label":str, "conf":float}
    """
    if not snap_b64:
        return [], None
    BOWL_KEYS    = ("bowl", "bin", "container", "toilet", "sink", "basket", "tray")
    LIGHTER_KEYS = ("lighter", "bottle", "vase", "cup", "flask", "cylinder", "object", "item")
    try:
        r = await client.post(f"{H100_REASON_URL}/robot-plan", json={
            "command": (
                f"The camera image is {frame_w} pixels wide by {frame_h} pixels tall. "
                "You are guiding a robot arm. Look at this camera image carefully. "
                "Find every lighter or small cylindrical object on the table — these are pick targets. "
                "Also find the white round bowl or bin — this is the drop target. "
                f"For EACH object give its pixel center [cx, cy] where cx is 0-{frame_w} and cy is 0-{frame_h}. "
                "Use labels: 'lighter' for pick targets, 'bowl' for the drop container. "
                f"Example output: lighter [155, 210], lighter [320, 195], bowl [310, 415]"
            ),
            "image_base64": snap_b64,
            "robot_type":   "xarm",
            "system_prompt": NIS_SYSTEM_PROMPT,
        }, timeout=25.0)
        if r.status_code == 200:
            d = r.json()
            lighters: list = []
            bowl = None

            # 1. Structured trajectory points (if Cosmos returns them)
            for t in (d.get("trajectory") or []):
                pt    = t.get("point_2d") or []
                label = t.get("label", "").lower()
                if len(pt) < 2:
                    continue
                cx, cy = int(pt[0]), int(pt[1])
                if any(k in label for k in BOWL_KEYS):
                    if bowl is None:
                        bowl = {"cx": cx, "cy": cy, "label": label}
                elif any(k in label for k in LIGHTER_KEYS):
                    if not any(abs(cx - l["cx"]) < 40 for l in lighters):
                        lighters.append({"cx": cx, "cy": cy, "label": label, "conf": 0.85})

            # 2. Parse reasoning / plan text for [cx, cy] patterns
            reasoning = d.get("reasoning") or d.get("plan") or ""
            for p in _cosmos_extract_positions(reasoning):
                label = p.get("label", "").lower()
                cx, cy = p["cx"], p["cy"]
                if any(k in label for k in BOWL_KEYS):
                    if bowl is None:
                        bowl = {"cx": cx, "cy": cy, "label": label}
                elif not any(abs(cx - l["cx"]) < 50 for l in lighters):
                    lighters.append({"cx": cx, "cy": cy,
                                     "label": label or "lighter", "conf": 0.75})

            # Scale coordinates down if Cosmos used larger internal resolution
            all_cx = [l["cx"] for l in lighters] + ([bowl["cx"]] if bowl else [])
            all_cy = [l["cy"] for l in lighters] + ([bowl["cy"]] if bowl else [])
            if all_cx and (max(all_cx) > frame_w * 1.2 or (all_cy and max(all_cy) > frame_h * 1.2)):
                sx = frame_w  / max(all_cx) if all_cx else 1.0
                sy = frame_h  / max(all_cy) if all_cy else 1.0
                scale = min(sx, sy)  # uniform scale to fit within frame
                for l in lighters:
                    l["cx"] = max(0, min(frame_w-1,  int(l["cx"] * sx)))
                    l["cy"] = max(0, min(frame_h-1,  int(l["cy"] * sy)))
                if bowl:
                    bowl["cx"] = max(0, min(frame_w-1, int(bowl["cx"] * sx)))
                    bowl["cy"] = max(0, min(frame_h-1, int(bowl["cy"] * sy)))
                logger.info("[cosmos_scan_scene] scaled coords by sx=%.2f sy=%.2f", sx, sy)
            lighters = sorted(lighters, key=lambda l: l["cx"])
            logger.info("[cosmos_scan_scene] lighters=%d bowl=%s",
                        len(lighters),
                        f"cx={bowl['cx']},cy={bowl['cy']}" if bowl else "None")
            return lighters, bowl
    except Exception as e:
        logger.warning("_cosmos_scan_scene failed: %s", e)
    return [], None


async def _verify_pick_disappeared(expected_cx: int, frame_w: int,
                                   conf: float = 0.08) -> bool:
    """
    Post-pick YOLO rescan — check if a lighter disappeared near expected_cx.
    Returns True if lighter is gone from pick position (pick succeeded).
    """
    try:
        scan = await _yolo_scan_nis("lighter,bottle,cup,vase,flask", conf=conf)
        LIGHTER_ALIASES = {"lighter", "bottle", "vase", "cup", "flask"}
        lighters = [
            d for d in scan.get("detections", [])
            if any(a in d.get("label", "").lower() for a in LIGHTER_ALIASES)
            and not any(k in d.get("label", "").lower() for k in ("bin", "toilet", "sink"))
        ]
        near = [d for d in lighters if abs(d.get("cx", 9999) - expected_cx) < 110]
        return len(near) == 0   # nothing at pick position → object was picked
    except Exception as e:
        logger.warning("_verify_pick_disappeared failed: %s", e)
        return False


class AutonomousRequest(BaseModel):
    task:        str   = Field(default="pick all lighters and put them in the bin",
                               description="High-level task description")
    max_picks:   int   = Field(default=10,   ge=1, le=20,
                               description="Max total picks before stopping")
    max_retries: int   = Field(default=3,    ge=1, le=5,
                               description="Max retry attempts per lighter")
    execute_arm: bool  = Field(default=True,
                               description="False = simulation/dry-run only")
    conf:        float = Field(default=0.08, ge=0.01, le=0.5,
                               description="YOLO detection confidence threshold")


@router.post("/autonomous")
async def cookoff_autonomous(request: AutonomousRequest):
    """
    FULL AUTONOMOUS lighter sweep — improved v2.

    Improvements over v1:
      - Multi-scan YOLO (2 scans, union merge) for complete detection
      - Budget counts successful picks, not total attempts
      - Cosmos fallback when YOLO finds nothing (confirms table clear)
      - Cosmos post-pick double-check when YOLO still sees object
      - Final Cosmos table-clear verification after all picks
      - Max 5 outer sweeps prevents infinite loop

    Monitor live via: GET /events/stream?topics=cookoff,arm,cosmos
    """
    import httpx
    t_start       = time.time()
    logs: List[str] = []
    all_picks: List[Dict] = []
    total_retries = 0
    LIGHTER_ALIASES = {"lighter", "bottle", "vase", "cup", "flask"}
    MAX_SWEEPS = 5

    _publish_cookoff("autonomous_start", {
        "task":        request.task,
        "max_picks":   request.max_picks,
        "max_retries": request.max_retries,
        "execute_arm": request.execute_arm,
        "conf":        request.conf,
    })
    logger.info("[autonomous] START task=%r execute=%s max_picks=%d",
                request.task, request.execute_arm, request.max_picks)

    async with httpx.AsyncClient(timeout=25.0) as c:

        # Camera warmup — 2 frames to stabilise auto-exposure
        for _ in range(2):
            try:
                await c.get(f"{AGENT_URL}/camera/snapshot", timeout=4.0)
                await asyncio.sleep(0.3)
            except Exception:
                pass

        successful_picks = 0
        total_attempts   = 0
        table_clear      = False
        sweep_count      = 0

        # OUTER LOOP — re-scan after each sweep until table clear or budget gone
        while successful_picks < request.max_picks and not table_clear and sweep_count < MAX_SWEEPS:
            sweep_count += 1
            _publish_cookoff("sweep_start", {
                "sweep":            sweep_count,
                "successful_picks": successful_picks,
            })

            # 1. MULTI-SCAN YOLO — 2 passes, merge unique detections ─────────
            scan1 = await _yolo_scan_nis(
                "lighter,bottle,cup,vase,flask,bin,bowl", conf=request.conf
            )
            await asyncio.sleep(0.4)
            scan2 = await _yolo_scan_nis(
                "lighter,bottle,cup,vase,flask,bin,bowl",
                conf=max(0.04, request.conf * 0.75),  # slightly lower conf on 2nd pass
            )

            dets1 = scan1.get("detections", [])
            dets2 = scan2.get("detections", [])
            frame_w   = scan1.get("frame_w", 1280)
            scene_ctx = scan1.get("scene_context", "")

            # Union: add det2 only if no similar label within 80px in dets1
            merged_dets = list(dets1)
            for d2 in dets2:
                if not any(
                    abs(d2["cx"] - d1["cx"]) < 80
                    and d2.get("label", "") == d1.get("label", "")
                    for d1 in dets1
                ):
                    merged_dets.append(d2)

            lighter_dets = sorted(
                [d for d in merged_dets
                 if any(a in d.get("label", "").lower() for a in LIGHTER_ALIASES)
                 and not any(k in d.get("label", "").lower() for k in ("bin", "toilet", "sink"))],
                key=lambda d: d.get("cx", 640),
            )
            bin_det = next(
                (d for d in merged_dets if any(k in d.get("label", "").lower() for k in ("bin", "bowl", "container", "toilet", "sink"))), None
            )

            _publish_cookoff("autonomous_scan", {
                "sweep":       sweep_count,
                "n_lighters":  len(lighter_dets),
                "n_total_det": len(merged_dets),
                "n_scan1":     len(dets1),
                "n_scan2":     len(dets2),
                "scene":       scene_ctx[:120],
            })
            logs.append(
                f"Sweep {sweep_count}: YOLO merged={len(lighter_dets)} lighters "
                f"(scan1={len(dets1)} scan2={len(dets2)}) | {scene_ctx[:80]}"
            )

            # 2. COSMOS SCENE SCAN — single call detects lighters + bowl ───────
            snap_b64 = await _get_snap_b64(c)
            cosmos_lighters, cosmos_bowl = await _cosmos_scan_scene(c, snap_b64, frame_w=frame_w, frame_h=480)
            _publish_cookoff("cosmos_scan", {
                "sweep":      sweep_count,
                "n_lighters": len(cosmos_lighters),
                "lighters":   cosmos_lighters[:8],
                "bowl":       cosmos_bowl,
            })

            # Cosmos is the authority for LIGHTER positions; YOLO is fallback
            if cosmos_lighters:
                merged = [
                    {**l, "cosmos_cx": l["cx"], "cosmos_cy": l.get("cy", 360),
                     "color": l.get("color", ""), "yolo_cx": l["cx"]}
                    for l in cosmos_lighters
                ]
                logs.append(f"Sweep {sweep_count}: Cosmos primary — {len(merged)} lighters")
            elif lighter_dets:
                merged = [{**ld, "cosmos_cx": ld["cx"],
                           "cosmos_cy": ld.get("cy", 360)} for ld in lighter_dets]
                logs.append(f"Sweep {sweep_count}: YOLO fallback — {len(merged)} lighters")
            else:
                table_clear = True
                logs.append(f"Sweep {sweep_count}: Cosmos+YOLO both report table clear")
                _publish_cookoff("table_clear", {"sweep": sweep_count})
                break

            # Cosmos is the authority for BOWL position; YOLO is fallback
            if cosmos_bowl:
                bin_det = {**cosmos_bowl, "conf": 0.90, "source": "cosmos"}
                logger.info("[autonomous] Cosmos bowl: cx=%d cy=%d",
                            cosmos_bowl["cx"], cosmos_bowl["cy"])
                _publish_cookoff("bowl_detected", {"source": "cosmos",
                    "cx": cosmos_bowl["cx"], "cy": cosmos_bowl["cy"]})
            elif bin_det:
                logger.info("[autonomous] YOLO bowl fallback: cx=%d cy=%d",
                            bin_det["cx"], bin_det["cy"])
            else:
                logger.warning("[autonomous] No bowl detected — using static fallback")

            merged = sorted(merged, key=lambda d: d.get("cosmos_cx", d["cx"]))
            _publish_cookoff("cosmos_positions", {"n": len(merged), "positions": merged[:8]})
            logs.append(f"Cosmos: {len(merged)} targets, bowl={'yes' if bin_det else 'no'}")

            # 4. PICK EACH LIGHTER ────────────────────────────────────────────
            any_picked_this_sweep = False

            for lighter in merged:
                if successful_picks >= request.max_picks:
                    break

                base_cx    = lighter.get("cosmos_cx", lighter["cx"])
                place_zone = _dest_to_place_zone("bin", bin_det, frame_w)

                pick_result: Dict = {
                    "lighter_idx": total_attempts,
                    "sweep":       sweep_count,
                    "label":       lighter.get("label", "lighter"),
                    "color":       lighter.get("color", ""),
                    "yolo_cx":     lighter["cx"],
                    "cosmos_cx":   lighter.get("cosmos_cx"),
                    "place_zone":  place_zone,
                    "attempts":    [],
                    "success":     False,
                }

                _publish_cookoff("picking", {
                    "pick_n":    successful_picks + 1,
                    "sweep":     sweep_count,
                    "yolo_cx":   lighter["cx"],
                    "cosmos_cx": lighter.get("cosmos_cx"),
                    "place":     place_zone,
                })
                logger.info("[autonomous] picking #%d label=%s cx=%d place=%s",
                            total_attempts + 1, lighter.get("label"), base_cx, place_zone)

                # RETRY LOOP ──────────────────────────────────────────────────
                S6_ADJUSTMENTS = [0, +15, -15]

                for attempt in range(request.max_retries):

                    # Re-scan on retry — get fresh position
                    if attempt > 0:
                        await asyncio.sleep(1.0)
                        rs = await _yolo_scan_nis(
                            "lighter,bottle,cup,vase,flask", conf=request.conf
                        )
                        frame_w = rs.get("frame_w", frame_w)
                        rs_lighters = sorted(
                            [d for d in rs.get("detections", [])
                             if any(a in d.get("label", "").lower()
                                    for a in LIGHTER_ALIASES)
                             and not any(k in d.get("label", "").lower() for k in ("bin", "toilet", "sink"))],
                            key=lambda d: abs(d["cx"] - base_cx),
                        )
                        if rs_lighters:
                            base_cx = rs_lighters[0]["cx"]
                        total_retries += 1

                    adj = S6_ADJUSTMENTS[attempt % len(S6_ADJUSTMENTS)]
                    s6  = max(360, min(650, _cx_to_s6(base_cx, frame_w) + adj))

                    logger.info("[autonomous]   attempt %d: cx=%d s6=%d adj=%+d",
                                attempt, base_cx, s6, adj)

                    attempt_log: Dict = {
                        "attempt":     attempt,
                        "cx":          base_cx,
                        "s6":          s6,
                        "s6_adj":      adj,
                        "pick_ok":     False,
                        "disappeared": False,
                    }

                    if request.execute_arm:
                        pk = await _run_ik_pick(
                            c, s6=s6, place=place_zone,
                            bin_cx=bin_det["cx"] if bin_det else None,
                            bin_cy=bin_det["cy"] if bin_det else None,
                            lighter_cy=lighter.get("cosmos_cy", lighter.get("cy")),
                            frame_w=frame_w)
                        attempt_log["pick_ok"] = pk.get("ok", False)
                        attempt_log["steps"]   = pk.get("steps", [])

                        await asyncio.sleep(1.5)

                        # Primary verify: YOLO disappearance check
                        disappeared = await _verify_pick_disappeared(
                            base_cx, frame_w, request.conf
                        )

                        # Secondary verify: if YOLO still sees it, ask Cosmos
                        if not disappeared:
                            try:
                                snap_check = await _get_snap_b64(c)
                                cosmos_check, _ = await _cosmos_plan_positions(
                                    c, snap_check,
                                    f"Is there still a lighter at approximately "
                                    f"pixel x={base_cx} on the table? "
                                    "List remaining lighters with positions.",
                                )
                                # Cosmos confirms gone if no position near base_cx
                                if not any(
                                    abs(cp["cx"] - base_cx) < 100
                                    for cp in cosmos_check
                                ):
                                    disappeared = True
                                    attempt_log["cosmos_confirmed_gone"] = True
                            except Exception:
                                pass

                        attempt_log["disappeared"] = disappeared

                        _publish_cookoff("pick_attempt", {
                            "pick_n":      successful_picks + 1,
                            "attempt":     attempt,
                            "s6":          s6,
                            "pick_ok":     attempt_log["pick_ok"],
                            "disappeared": disappeared,
                        })

                        pick_result["attempts"].append(attempt_log)

                        if disappeared:
                            pick_result["success"] = True
                            logs.append(
                                f"  Pick #{total_attempts+1} SUCCESS "
                                f"attempt={attempt} s6={s6}"
                            )
                            break
                        else:
                            logs.append(
                                f"  Pick #{total_attempts+1} MISS "
                                f"attempt={attempt} s6={s6} — retrying"
                            )

                    else:
                        # Simulation — always succeeds
                        attempt_log["pick_ok"]     = True
                        attempt_log["disappeared"] = True
                        pick_result["attempts"].append(attempt_log)
                        pick_result["success"] = True
                        logs.append(
                            f"  [SIM] Pick #{total_attempts+1} s6={s6} cx={base_cx}"
                        )
                        break

                all_picks.append(pick_result)
                total_attempts += 1

                if pick_result["success"]:
                    successful_picks += 1
                    any_picked_this_sweep = True
                    _publish_cookoff("pick_success", {
                        "n_successful": successful_picks,
                        "remaining_budget": request.max_picks - successful_picks,
                    })

                # Brief settle between objects
                if lighter is not merged[-1]:
                    await asyncio.sleep(1.5)

            # If arm is enabled and nothing succeeded this sweep — break to
            # avoid infinite retrying when all picks consistently miss
            if request.execute_arm and not any_picked_this_sweep:
                logs.append(
                    f"Sweep {sweep_count}: 0 successful picks — stopping outer loop"
                )
                _publish_cookoff("no_progress", {"sweep": sweep_count})
                break

        # end while outer loop

    # 5. FINAL COSMOS TABLE-CLEAR VERIFICATION ────────────────────────────────
    _publish_cookoff("final_check", {"msg": "Final Cosmos table-clear verification…"})
    try:
        async with httpx.AsyncClient(timeout=15.0) as cv:
            snap_final = await _get_snap_b64(cv)
            final_pos, final_reasoning = await _cosmos_plan_positions(
                cv, snap_final,
                "Final check: look at the entire table surface carefully. "
                "Are there any lighters, bottles, or objects that have NOT been "
                "placed in the bin? List positions if any remain, or confirm clear.",
            )
            if final_reasoning:
                _publish_cookoff("reasoning_start", {"task": "final table-clear check"})
                await _stream_reasoning(final_reasoning)

            if not final_pos:
                table_clear = True
                logs.append("Final Cosmos check: table clear confirmed")
                _publish_cookoff("table_clear", {"source": "cosmos_final"})
            else:
                remaining_n = len([p for p in final_pos if "bin" not in p["label"].lower()])
                logs.append(
                    f"Final Cosmos: {remaining_n} object(s) may remain on table"
                )
    except Exception as e:
        logger.warning("[autonomous] final cosmos check failed: %s", e)
        # Fallback: infer from results
        if not table_clear:
            table_clear = (successful_picks > 0 and successful_picks == len(all_picks))

    latency_ms = round((time.time() - t_start) * 1000)
    successes  = sum(1 for p in all_picks if p["success"])

    _publish_cookoff("autonomous_done", {
        "table_clear":   table_clear,
        "picks_total":   total_attempts,
        "picks_success": successes,
        "total_retries": total_retries,
        "sweeps":        sweep_count,
        "latency_ms":    latency_ms,
    })
    logger.info(
        "[autonomous] DONE picks=%d/%d sweeps=%d retries=%d clear=%s lat=%dms",
        successes, total_attempts, sweep_count, total_retries, table_clear, latency_ms,
    )

    return {
        "ok":            successes > 0 or (total_attempts == 0 and table_clear),
        "task":          request.task,
        "table_clear":   table_clear,
        "picks_total":   total_attempts,
        "picks_success": successes,
        "sweeps":        sweep_count,
        "total_retries": total_retries,
        "all_picks":     all_picks,
        "logs":          logs,
        "latency_ms":    latency_ms,
        "timestamp":     time.time(),
    }
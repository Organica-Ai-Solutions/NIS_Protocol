#!/usr/bin/env python3
"""
VLA xArm Inference Server v2 — GPU 1, port 8500
================================================
CNN encoder (ResNet-style) + MLP head.
v5: finetuned on real table/lighter/bin pick data (100 epochs).

KEY FIX: cx-aware lateral targeting.
  The model is vision-only (no language). S2 (base_pan) was biased toward center.
  Fix: accept cx (object pixel x) + frame_w → override S2 with calibrated lateral mapping.
  This makes VLA cx-aware without retraining.

Calibrated S2 lateral map (640px frame):
  cx=0   → S2=380  (far left)
  cx=200 → S2=460  (left)
  cx=320 → S2=500  (center)
  cx=440 → S2=540  (right)
  cx=640 → S2=620  (far right)
"""
import base64, io, logging, os, time
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("vla-xarm-v2")

# ── Config ─────────────────────────────────────────────────────────────────────
DEVICE    = "cuda:0"   # GPU 1 via CUDA_VISIBLE_DEVICES=1
ACTION_DIM = 4
PORT       = 8500

# Priority: v5 best → v5 final → v4 best
_CKPT_CANDIDATES = [
    os.environ.get("VLA_CHECKPOINT", ""),
    "/data/organica-ai/models/vla_xarm_v5/vla_xarm_v5_best.pt",
    "/data/organica-ai/models/vla_xarm_v5/vla_xarm_v5_final.pt",
    "/data/organica-ai/models/vla_xarm_v4/gpu1/vla_xarm_best.pt",
]
CKPT_PATH = next((p for p in _CKPT_CANDIDATES if p and Path(p).exists()), "")
MODEL_VER = "vla_xarm_v5" if "v5" in CKPT_PATH else "vla_xarm_v4"

# xArm safe servo ranges
SERVO_CENTERS = [100, 500, 310, 870]
SERVO_RANGES  = [400, 200, 200, 200]

# Calibrated S2 lateral map: cx (0-640) → S2 servo (380-620)
# Reference points: left=460@cx200, center=500@cx320, right=540@cx440
S2_CX_MAP = [
    (0,   380),
    (160, 440),
    (200, 460),
    (320, 500),
    (440, 540),
    (480, 560),
    (640, 620),
]

def cx_to_s2(cx: int, frame_w: int = 640) -> int:
    """Map object pixel x position to S2 servo value using calibrated lookup."""
    # Normalize cx to 640px reference
    cx_norm = int(cx * 640 / max(frame_w, 1))
    cx_norm = max(0, min(640, cx_norm))
    # Interpolate
    for i in range(len(S2_CX_MAP) - 1):
        x0, s0 = S2_CX_MAP[i]
        x1, s1 = S2_CX_MAP[i + 1]
        if x0 <= cx_norm <= x1:
            t = (cx_norm - x0) / max(x1 - x0, 1)
            return int(s0 + t * (s1 - s0))
    return S2_CX_MAP[-1][1]


# ── Model ──────────────────────────────────────────────────────────────────────
class VLAPolicy(nn.Module):
    def __init__(self, action_dim=ACTION_DIM):
        super().__init__()
        self.enc = nn.Sequential(
            self._block(3,   64,  stride=2),
            self._block(64,  128, stride=2),
            self._block(128, 256, stride=2),
            self._block(256, 512, stride=2),
            self._block(512, 512, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, action_dim),
        )

    def _block(self, cin, cout, stride=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.GELU(),
        )

    def forward(self, x):
        return self.head(self.enc(x))


_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

_model = None
_ready = False
_load_error = ""

app = FastAPI(title="VLA xArm Inference Server", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _load():
    global _model, _ready, _load_error
    if not CKPT_PATH:
        _load_error = "No checkpoint found"
        log.error(_load_error)
        return
    log.info("Loading %s from %s on %s...", MODEL_VER, CKPT_PATH, DEVICE)
    try:
        model = VLAPolicy(ACTION_DIM).to(DEVICE)
        ckpt  = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        missing, _ = model.load_state_dict(state, strict=False)
        if missing:
            log.warning("Missing keys: %s", missing[:3])
        model.eval()
        _model = model
        _ready = True
        log.info("%s loaded — ready on %s port %d", MODEL_VER, DEVICE, PORT)
        # Warmup
        dummy = torch.zeros(1, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            _ = _model(dummy)
        log.info("Warmup done. ckpt=%s", Path(CKPT_PATH).name)
    except Exception as e:
        _load_error = str(e)
        log.error("VLA load failed: %s", e)


def _decode_image(b64: str) -> torch.Tensor:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return _transform(img).unsqueeze(0).to(DEVICE)


def _action_to_servos(action: np.ndarray, cx: Optional[int] = None,
                      frame_w: int = 640) -> dict:
    """
    Convert normalized 4D action → absolute servo positions.
    If cx is provided, override S2 with calibrated lateral mapping.
    VLA's S2 is biased toward center (training data was mostly center picks).
    """
    servos = {}
    for i, (center, rng, delta) in enumerate(zip(SERVO_CENTERS, SERVO_RANGES, action)):
        raw = int(center + float(delta) * rng)
        raw = max(0, min(1000, raw))
        servos[str(i + 1)] = raw

    # Override S2 with cx-based lateral if provided (VLA S2 is unreliable)
    if cx is not None:
        s2_cx = cx_to_s2(cx, frame_w)
        log.info("[cx-override] cx=%d frame_w=%d → S2=%d (was %d)",
                 cx, frame_w, s2_cx, servos.get("2", 500))
        servos["2"] = s2_cx

    # S5, S6 stay at safe defaults
    servos["5"] = 680
    servos["6"] = 500
    return servos


# ── Request/Response models ────────────────────────────────────────────────────
class InferRequest(BaseModel):
    image_base64: str
    instruction:  str = "pick and place object"
    temperature:  float = 0.0
    # NEW: cx hint for lateral targeting
    cx:           Optional[int] = None    # object pixel x from YOLO bbox
    frame_w:      Optional[int] = 640    # camera frame width


class InferResponse(BaseModel):
    ok:          bool
    action_raw:  List[float]
    servos:      dict
    instruction: str
    latency_ms:  float
    model:       str = MODEL_VER
    cx_used:     Optional[int] = None


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":    "healthy" if _ready else "loading",
        "ready":     _ready,
        "model":     MODEL_VER,
        "checkpoint": Path(CKPT_PATH).name if CKPT_PATH else "none",
        "device":    DEVICE,
        "error":     _load_error or None,
        "cx_aware":  True,
    }


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    if not _ready:
        return JSONResponse(status_code=503, content={"ok": False, "error": "Model not ready"})

    t0 = time.time()
    try:
        img_tensor = _decode_image(req.image_base64)
        with torch.no_grad():
            raw = _model(img_tensor)[0].cpu().numpy()

        if req.temperature > 0:
            raw = raw + np.random.randn(*raw.shape) * req.temperature

        servos  = _action_to_servos(raw, cx=req.cx, frame_w=req.frame_w or 640)
        latency = round((time.time() - t0) * 1000, 2)

        log.info("VLA infer: inst='%s' cx=%s action=%s S2=%d lat=%.1fms",
                 req.instruction[:40], req.cx, raw.round(3).tolist(),
                 servos.get("2", 0), latency)

        return InferResponse(
            ok=True,
            action_raw=raw.tolist(),
            servos=servos,
            instruction=req.instruction,
            latency_ms=latency,
            model=MODEL_VER,
            cx_used=req.cx,
        )
    except Exception as e:
        log.error("Infer error: %s", e)
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.post("/infer/batch")
def infer_batch(images: List[str], instruction: str = "pick and place"):
    if not _ready:
        return JSONResponse(status_code=503, content={"ok": False})
    t0 = time.time()
    tensors = torch.stack([_decode_image(b)[0] for b in images])
    with torch.no_grad():
        actions = _model(tensors).cpu().numpy()
    return {
        "ok":        True,
        "actions":   actions.tolist(),
        "count":     len(images),
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }


@app.get("/model/info")
def model_info():
    if not _model:
        return {"ready": False}
    params = sum(p.numel() for p in _model.parameters())
    return {
        "ready":         _ready,
        "model":         MODEL_VER,
        "checkpoint":    CKPT_PATH,
        "device":        DEVICE,
        "action_dim":    ACTION_DIM,
        "parameters":    params,
        "servo_centers": SERVO_CENTERS,
        "servo_ranges":  SERVO_RANGES,
        "cx_aware":      True,
        "description":   "CNN+MLP. Real xArm pick-and-place. cx-aware S2 lateral override.",
    }


@app.get("/cx-test")
def cx_test():
    """Test the cx → S2 lateral mapping."""
    return {
        "calibration": [{"cx": cx, "s2": cx_to_s2(cx)} for cx in range(0, 641, 80)],
        "note": "cx=0=far_left s2=380, cx=320=center s2=500, cx=640=far_right s2=620"
    }


if __name__ == "__main__":
    _load()
    uvicorn.run(app, host="0.0.0.0", port=PORT)

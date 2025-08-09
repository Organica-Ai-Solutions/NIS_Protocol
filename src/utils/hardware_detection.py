"""
Lightweight hardware detection utilities for runtime provider selection.
"""

from __future__ import annotations

import os
import shutil
import multiprocessing
from typing import Dict, Any


def is_gpu_available() -> bool:
    try:
        # Prefer torch if available
        import torch  # type: ignore
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    # Fallback: nvidia-smi on PATH
    return shutil.which("nvidia-smi") is not None


def detect_compute_environment() -> Dict[str, Any]:
    cpu_count = multiprocessing.cpu_count()
    try:
        import psutil  # optional
        total_ram_bytes = psutil.virtual_memory().total
    except Exception:
        # Fallback: approximate from cgroup limits or omit
        total_ram_bytes = 0

    gpu = is_gpu_available()

    # Decide default mode
    llm_mode = os.getenv("LLM_MODE", "auto").lower()  # auto|local|cloud

    return {
        "gpu_available": gpu,
        "cpu_cores": cpu_count,
        "total_ram_bytes": int(total_ram_bytes),
        "llm_mode": llm_mode,
    }



# ========================================================================
# Production Multi-stage Dockerfile for NIS Protocol v3.2.1 (GPU Edition)
# Enhanced for NVIDIA NIM, KAN/PINN workloads, and physics-informed AI
# CUDA 12.1.1 — Optimized for AWS g4dn.xlarge (NVIDIA T4)
# ========================================================================

# ---------- Stage 1: Builder -------------------------------------------------
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# Prevent interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip build-essential git gcc g++ curl wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies inside builder
RUN python3.11 -m pip install --no-cache-dir --upgrade pip && \
    python3.11 -m pip install --no-cache-dir --user -r requirements.txt

# Optional: Add core voice & TTS/STT modules
RUN python3.11 -m pip install --no-cache-dir --user \
    openai-whisper soundfile librosa ffmpeg-python \
    gtts suno-bark transformers scipy encodec nltk einops boto3

# ---------- Stage 2: Runtime -------------------------------------------------
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies (audio + system libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip ffmpeg libsndfile1 libasound2-dev portaudio19-dev \
    libgomp1 libblas3 liblapack3 curl wget git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd -m -u 1000 nisuser

# Copy Python user site-packages from builder
COPY --from=builder /root/.local /home/nisuser/.local

# Set environment paths
USER nisuser
ENV PATH="/home/nisuser/.local/bin:${PATH}"
ENV PYTHONPATH="/home/nisuser/app:${PYTHONPATH}"
WORKDIR /home/nisuser/app

# Copy project source
COPY --chown=nisuser:nisuser . .

# Prepare directories
RUN mkdir -p logs static cache models data/chat_memory && \
    chmod +x *.sh 2>/dev/null || true

# ---------------- Environment Variables ----------------
# (Used by ECS and Docker Compose)
ENV CUDA_HOME=/usr/local/cuda
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONUNBUFFERED=1

# ---------------- Healthcheck ----------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ---------------- Port Exposure ----------------
EXPOSE 8000

# ---------------- Entrypoint ----------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ---------------- GPU Validation ----------------
# Test locally before ECS:
# docker build -t nis-protocol:gpu .
# docker run --rm --gpus all nis-protocol:gpu python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# Expected: True 12.1 
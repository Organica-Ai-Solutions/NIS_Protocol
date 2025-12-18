# ========================================================================
# Production Multi-stage Dockerfile for NIS Protocol v4.0 (GPU Edition)
# Enhanced for NVIDIA NIM, KAN/PINN workloads, and physics-informed AI
# CUDA 12.1.1 — Optimized for AWS g4dn.xlarge (NVIDIA T4)
# ========================================================================

# ---------- Stage 1: Builder -------------------------------------------------
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# Prevent interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    build-essential git gcc g++ curl wget \
    libffi-dev libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python \
    && ln -s /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Copy requirements and constraints first for caching
COPY requirements.txt constraints.txt ./

# Upgrade pip and install build tools first
RUN python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies with constraints (security fixes)
RUN python3.11 -m pip install --no-cache-dir --user \
    -r requirements.txt -c constraints.txt \
    || python3.11 -m pip install --no-cache-dir --user -r requirements.txt

# Optional: Add core voice & TTS/STT modules (skip problematic packages)
RUN python3.11 -m pip install --no-cache-dir --user \
    soundfile librosa ffmpeg-python \
    gtts scipy nltk einops boto3 gdown \
    || echo 'Some optional packages failed, continuing...'

# ---------- Stage 2: Runtime -------------------------------------------------
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies (audio + system libs)
# Ubuntu 22.04 includes SQLite 3.37.2 which satisfies ChromaDB >= 3.35.0 requirement
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip ffmpeg libsndfile1 libasound2-dev portaudio19-dev \
    libgomp1 libblas3 liblapack3 curl wget git \
    libsqlite3-0 sqlite3 \
    && rm -rf /var/lib/apt/lists/* \
    && sqlite3 --version \
    && ln -s /usr/bin/python3.11 /usr/bin/python \
    && ln -s /usr/bin/python3.11 /usr/bin/python3

# Create a non-root user for security
RUN useradd -m -u 1000 nisuser

# Copy Python user site-packages from builder
COPY --from=builder /root/.local /home/nisuser/.local

# Set environment paths
USER nisuser
ENV PATH="/home/nisuser/.local/bin:${PATH}"
ENV PYTHONPATH="/home/nisuser/app"
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
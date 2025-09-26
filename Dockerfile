# Production Multi-stage Dockerfile for NIS Protocol v3.2.1
# Enhanced for NVIDIA NIM integration and real mathematical implementations
# Supports both CPU and GPU deployment modes

# Stage 1: Build stage (for compiling dependencies)
FROM python:3.11-slim as builder

# Install only essential build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt


# Stage 2: Runtime stage (lightweight)
FROM python:3.11-slim

# Install runtime dependencies including audio processing and NVIDIA tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    libasound2-dev \
    portaudio19-dev \
    libgomp1 \
    libblas3 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 nisuser

# Copy Python packages from builder
COPY --from=builder /root/.local /home/nisuser/.local

# Set up environment
USER nisuser
ENV PATH="/home/nisuser/.local/bin:${PATH}"
ENV PYTHONPATH="/home/nisuser/app:${PYTHONPATH}"
WORKDIR /home/nisuser/app

# Copy only essential application files (thanks to optimized .dockerignore)
COPY --chown=nisuser:nisuser . .

# Create directories and set permissions
RUN mkdir -p logs static cache models data/chat_memory && \
    chmod +x *.sh 2>/dev/null || true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Use uvicorn for production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"] 
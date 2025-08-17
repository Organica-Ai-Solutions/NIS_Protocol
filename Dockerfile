# Stage 1: Build stage
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc gfortran libopenblas-dev liblapack-dev cython3 && rm -rf /var/lib/apt/lists/*

# Install python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip

# Create core requirements without hnswlib
RUN grep -v "^#.*hnswlib" requirements.txt > core_requirements.txt
# Try to wheel all packages, continue if some fail (like hnswlib with GLIBCXX issues)
RUN pip wheel --no-cache-dir --wheel-dir=/app/wheels -r core_requirements.txt || \
    (echo "Some packages failed to build wheels, installing individually..." && \
     pip wheel --no-cache-dir --wheel-dir=/app/wheels numpy scipy scikit-learn fastapi uvicorn pydantic)


# Stage 2: Final stage
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install runtime dependencies - reduced build tools since hnswlib is now optional
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libopenblas0 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user
RUN useradd -m nisuser
USER nisuser
ENV PATH="/home/nisuser/.local/bin:${PATH}"
WORKDIR /home/nisuser/app

# Install dependencies from wheels (gracefully handle failures)
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/core_requirements.txt .

# Install from wheels first
RUN pip install --no-cache-dir /wheels/* && echo "✅ Installed from wheels successfully" || \
    echo "⚠️  Some wheel installations failed, continuing with individual installations"

# Install core packages directly if wheels failed
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    scikit-learn \
    fastapi \
    uvicorn \
    pydantic \
    httpx \
    requests \
    pandas || echo "⚠️  Some core packages failed to install"

# Copy application code
COPY --chown=nisuser:nisuser . .

# Create static directory and ensure permissions
RUN mkdir -p static && chown -R nisuser:nisuser static

# Expose port and set entrypoint
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 
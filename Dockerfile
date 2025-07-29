# Stage 1: Build stage
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc gfortran libopenblas-dev liblapack-dev cython3 && rm -rf /var/lib/apt/lists/*

# Install python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip wheel --no-cache-dir --wheel-dir=/app/wheels -r requirements.txt


# Stage 2: Final stage
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip libopenblas0 liblapack3 && rm -rf /var/lib/apt/lists/*

# Set up non-root user
RUN useradd -m nisuser
USER nisuser
ENV PATH="/home/nisuser/.local/bin:${PATH}"
WORKDIR /home/nisuser/app

# Install dependencies from wheels
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir /wheels/*

# Copy application code
COPY --chown=nisuser:nisuser . .

# Expose port and set entrypoint
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 
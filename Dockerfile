# NIS Protocol v3 - Main Application Container
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set work directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install system and python dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc python3-pip && \
    pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy only the necessary application code
COPY src ./src
COPY main.py .

# Create a non-root user for security
RUN useradd -m nisuser
USER nisuser
WORKDIR /home/nisuser/app

# Set correct ownership
COPY --chown=nisuser:nisuser . .

# Create necessary directories
RUN mkdir -p /home/nisuser/app/logs /home/nisuser/app/data /home/nisuser/app/models /home/nisuser/app/cache

# Expose ports
EXPOSE 8000

# Health check using Python instead of curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import http.client; conn = http.client.HTTPConnection('localhost', 8000); conn.request('GET', '/health'); exit(0) if conn.getresponse().status == 200 else exit(1)"

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 
# NIS Protocol v3 - Main Application Container
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    make \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements_enhanced_infrastructure.txt requirements_tech_stack.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_enhanced_infrastructure.txt && \
    pip install --no-cache-dir -r requirements_tech_stack.txt && \
    pip install --no-cache-dir fastapi uvicorn[standard] gunicorn

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/cache

# Set permissions
RUN chmod +x /app/start.sh || true
RUN chmod +x /app/stop.sh || true
RUN chmod +x /app/reset.sh || true

# Expose ports
EXPOSE 8000 5000

# Health check using Python instead of curl
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 
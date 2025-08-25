FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for shell execution
RUN apt-get update && apt-get install -y \
    bash \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create non-root user for security
RUN useradd -m runuser && chown -R runuser:runuser /app
USER runuser

# Default command
EXPOSE 8001
CMD ["python", "server.py"]

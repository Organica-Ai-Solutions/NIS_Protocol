# NIS Protocol Secure Code Runner
# Sandboxed environment for safe code execution
FROM python:3.11-slim

# Create non-root user for security
RUN useradd -m -u 1001 -s /bin/bash runner

# Install security and runtime dependencies
RUN apt-get update && apt-get install -y \
    # Runtime dependencies
    curl \
    jq \
    # Browser support (headless)
    chromium \
    chromium-driver \
    # Browser dependencies
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    xvfb \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for sandboxed execution
RUN pip install --no-cache-dir \
    fastapi==0.116.1 \
    uvicorn[standard]==0.34.3 \
    aiofiles==24.1.0 \
    python-multipart==0.0.12 \
    # Sandboxing and security
    RestrictedPython==7.2 \
    # Scientific computing (restricted)
    numpy==1.26.4 \
    scipy==1.16.1 \
    matplotlib==3.10.5 \
    pandas==2.3.2 \
    # Browser automation (secure)
    selenium==4.25.0 \
    beautifulsoup4==4.13.5 \
    requests==2.32.5 \
    # Safe execution environment
    psutil==7.0.0

# Set up secure workspace
WORKDIR /app
RUN mkdir -p workspace temp logs browser_cache

# Configure chromium to work without additional downloads
RUN ln -s /usr/bin/chromium /usr/bin/google-chrome || true

# Configure browser security
ENV DISPLAY=:99
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROME_ARGS="--headless --no-sandbox --disable-dev-shm-usage --disable-gpu --disable-web-security --disable-features=NetworkService --disable-background-timer-throttling --disable-backgrounding-occluded-windows --disable-renderer-backgrounding --disable-field-trial-config --disable-ipc-flooding-protection"

# Copy runner application files
COPY runner_app.py security_config.py security_hardening.py /app/
COPY browser_security.py /app/

# Set ownership and permissions
RUN chown -R runner:runner /app
RUN chmod -R 755 /app
RUN chmod 700 /app/workspace

# Switch to non-root user
USER runner

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose runner port
EXPOSE 8001

# Start the secure runner service
CMD ["uvicorn", "runner_app:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]

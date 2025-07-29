#!/bin/bash

# ==============================================================================
# NIS Protocol v3.1 - Enhanced Start Script
# Based on proven patterns from the Archaeological Discovery Platform
# ==============================================================================

# --- Configuration ---
PROJECT_NAME="nis-protocol-v3"
COMPOSE_FILE="docker-compose.yml"
REQUIRED_DIRS=("logs" "data" "models" "cache")
ENV_FILE=".env"
ENV_TEMPLATE="environment-template.txt"
BITNET_MODEL_DIR="models/bitnet/models/bitnet"
BITNET_MODEL_MARKER="${BITNET_MODEL_DIR}/config.json"

# --- ANSI Color Codes ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Helper Functions ---
function print_info {
    echo -e "${BLUE}[NIS-V3] $1${NC}"
}

function print_success {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

function print_warning {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

function print_error {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

function spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# --- Main Script ---

print_info "Starting NIS Protocol v3 Complete System..."
echo ""

# 1. Check Docker Availability
print_info "Checking Docker availability..."
if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
    print_error "Docker and Docker Compose are required. Please install them and try again."
fi
print_success "Docker and Docker Compose are available"

# 2. Create Required Directories
print_info "Creating required directories..."
for dir in "${REQUIRED_DIRS[@]}"; do
    mkdir -p "$dir"
done
print_success "All required directories are ready"

# 3. Check for BitNet Model Files
print_info "Checking for BitNet model files..."
if [ ! -f "$BITNET_MODEL_MARKER" ]; then
    print_warning "BitNet model files not found. Initiating download..."
    
    # Check if Python is available
    if command -v python &> /dev/null || command -v python3 &> /dev/null; then
        PYTHON_CMD="python"
        if ! command -v python &> /dev/null; then
            PYTHON_CMD="python3"
        fi
        
        # Check if download script exists
        if [ -f "scripts/download_bitnet_models.py" ]; then
            print_info "Running BitNet model download script..."
            $PYTHON_CMD scripts/download_bitnet_models.py
            
            if [ $? -ne 0 ]; then
                print_warning "BitNet model download failed. The system will use fallback mechanisms."
            else
                print_success "BitNet model files downloaded successfully!"
            fi
        else
            print_warning "BitNet download script not found at scripts/download_bitnet_models.py"
            print_warning "The system will use fallback mechanisms."
        fi
    else
        print_warning "Python not found. Cannot download BitNet models."
        print_warning "The system will use fallback mechanisms."
    fi
else
    print_success "BitNet model files found!"
fi

# 4. Validate Environment and API Keys
print_info "Validating environment and API keys..."
if [ ! -f "$ENV_FILE" ]; then
    print_warning "Environment file '$ENV_FILE' not found. Copying from template."
    if [ -f "$ENV_TEMPLATE" ]; then
        cp "$ENV_TEMPLATE" "$ENV_FILE"
        print_warning "Please edit '$ENV_FILE' and add your API keys."
    else
        print_error "Template '$ENV_TEMPLATE' not found. Cannot create .env file."
    fi
else
    # Check for placeholder keys
    if grep -q "your_key_here" "$ENV_FILE"; then
        print_warning "Your '$ENV_FILE' contains placeholder keys. The system may not function correctly without real API keys."
    else
        print_success "API keys are present."
    fi
fi
echo ""

# 5. Build Docker Images
print_info "Building Docker images (this may take several minutes on the first run)..."
docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" build --progress=plain
if [ $? -ne 0 ]; then
    print_error "Docker build failed. Please check the output above for errors."
fi
print_success "Docker images built successfully."

# 6. Start Docker Compose
print_info "Starting NIS Protocol v3 services in detached mode..."
docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d --force-recreate --remove-orphans

if [ $? -ne 0 ]; then
    print_error "Docker Compose failed to start. Please check the logs."
fi
print_success "Services are starting..."

# 7. Stream Backend Logs for Insight
print_info "Streaming logs from the backend service to show startup progress..."
echo -e "${YELLOW}(Press Ctrl+C to stop streaming logs and continue to health monitoring)${NC}"
(docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs --follow --tail="1" backend) &
LOGS_PID=$!

# Show spinner while logs are streaming for a bit
sleep 15
kill $LOGS_PID > /dev/null 2>&1
wait $LOGS_PID > /dev/null 2>&1
echo ""


# 8. Monitor Health of Services
print_info "Monitoring service health..."
SECONDS=0
TIMEOUT=300 # 5 minutes

while [ $SECONDS -lt $TIMEOUT ]; do
    unhealthy_services=$(docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps | grep -E "unhealthy|exited")
    
    if [ -z "$unhealthy_services" ]; then
        print_success "All services are healthy and running!"
        echo ""
        print_info "NIS Protocol v3 is now accessible at http://localhost:80"
        exit 0
    fi
    
    echo -ne "Waiting for services to become healthy... ($SECONDS/$TIMEOUT seconds)\r"
    sleep 5
    SECONDS=$((SECONDS+5))
done

print_error "One or more services failed to become healthy within the timeout period."
echo -e "${YELLOW}Please check the container logs for more details:${NC}"
docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs --tail=100 
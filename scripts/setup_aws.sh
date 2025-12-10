#!/bin/bash

# ==============================================================================
# NIS Protocol - AWS Deployment Setup Script
# Run this BEFORE docker compose to ensure all prerequisites are met
# ==============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[SETUP]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     NIS Protocol - AWS Deployment Prerequisites         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# 1. Check Docker
print_info "Checking Docker..."
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
fi
print_success "Docker available"

# 2. Create required directories
print_info "Creating required directories..."
mkdir -p logs data models cache configs
print_success "Directories created"

# 3. Setup .env file
print_info "Checking .env file..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_warning ".env created from template - EDIT WITH YOUR API KEYS"
    else
        print_error ".env.example not found"
    fi
else
    print_success ".env exists"
fi

# 4. Setup Google credentials placeholder
print_info "Checking Google credentials..."
if [ ! -f "configs/google-service-account.json" ]; then
    if [ -f "configs/google-service-account.json.example" ]; then
        cp configs/google-service-account.json.example configs/google-service-account.json
        print_warning "Google credentials placeholder created - UPDATE WITH REAL CREDENTIALS"
    else
        # Create minimal placeholder
        echo '{"type": "service_account", "project_id": "placeholder"}' > configs/google-service-account.json
        print_warning "Minimal Google credentials placeholder created"
    fi
else
    print_success "Google credentials file exists"
fi

# 5. Verify critical files exist
print_info "Verifying critical files..."
CRITICAL_FILES=(
    "Dockerfile"
    "docker-compose.yml"
    "main.py"
    "requirements.txt"
    "runner/Dockerfile"
    "runner/runner_app.py"
    "runner/security_config.py"
    "runner/browser_security.py"
    "system/config/Dockerfile.nginx"
    "system/config/nginx.conf"
)

MISSING=0
for file in "${CRITICAL_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        print_warning "Missing: $file"
        MISSING=$((MISSING+1))
    fi
done

if [ $MISSING -gt 0 ]; then
    print_error "$MISSING critical files missing. Run 'git pull origin main' first."
fi
print_success "All critical files present"

# 6. Check for port conflicts
print_info "Checking for port conflicts..."
PORTS=(80 8000 8001 6379 9092 2181)
CONFLICTS=0
for port in "${PORTS[@]}"; do
    if lsof -i :$port > /dev/null 2>&1; then
        print_warning "Port $port is in use"
        CONFLICTS=$((CONFLICTS+1))
    fi
done

if [ $CONFLICTS -gt 0 ]; then
    print_warning "$CONFLICTS ports in use - may cause conflicts"
else
    print_success "No port conflicts detected"
fi

# 7. Summary
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Setup Complete!                             ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Edit .env with your API keys"
echo "  2. (Optional) Update configs/google-service-account.json"
echo "  3. Build and run:"
echo ""
echo "     # GPU mode (requires NVIDIA GPU)"
echo "     docker compose up -d --build"
echo ""
echo "     # CPU mode (for testing)"
echo "     docker compose -f docker-compose.cpu.yml up -d --build"
echo ""
echo "  4. Check health:"
echo "     curl http://localhost:8000/health"
echo ""

#!/bin/bash

# ==============================================================================
# NIS Protocol v3.2.5 - CPU-Only Quick Start Script (Mac/Linux)
# Fast startup without GPU requirements
# ==============================================================================

# --- Configuration ---
PROJECT_NAME="nis-protocol-v3"
COMPOSE_FILE="docker-compose.cpu.yml"
ENV_FILE=".env"

# --- Colors ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Functions
print_info() {
    echo -e "${BLUE}[CPU-MODE]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

clear
echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  NIS Protocol v3.2.5 - CPU Mode (Mac/Linux Testing)     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# 1. Check Docker
print_info "Checking Docker availability..."
if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
    print_error "Docker and Docker Compose are required"
fi
print_success "Docker available"

# 2. Check .env file
print_info "Checking API keys..."
if [ ! -f "$ENV_FILE" ]; then
    print_warning ".env file not found. Creating from template..."
    cat > .env << 'EOF'
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
GCP_PROJECT_ID=organicaaisolutions
EOF
    print_warning "Please edit .env and add your API keys!"
    exit 1
fi
print_success "API keys configured"

# 3. Start services
print_info "Starting CPU-only stack..."
docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d --force-recreate

if [ $? -ne 0 ]; then
    print_error "Failed to start services"
fi

print_success "Services starting..."

# 4. Wait for health
print_info "Waiting for backend to be healthy (30s)..."
sleep 30

# 5. Test health
HEALTH=$(curl -s http://localhost:8000/health)
if echo "$HEALTH" | grep -q "healthy"; then
    print_success "✅ Backend is healthy!"
else
    print_warning "Backend may still be initializing..."
fi

# 6. Show status
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║             🎉 NIS Protocol CPU Mode Ready! 🎉           ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📋 Available Endpoints:${NC}"
echo "  • Health:  http://localhost:8000/health"
echo "  • Chat:    http://localhost:8000/chat"
echo "  • Docs:    http://localhost:8000/docs"
echo "  • Console: http://localhost:8000/chat (browser)"
echo ""
echo -e "${BLUE}🧪 Quick Test:${NC}"
echo "  curl http://localhost:8000/health | jq"
echo ""
echo -e "${BLUE}📊 View Logs:${NC}"
echo "  docker-compose -f $COMPOSE_FILE logs -f backend"
echo ""
echo -e "${BLUE}🛑 Stop System:${NC}"
echo "  docker-compose -f $COMPOSE_FILE down"
echo ""
print_success "Ready for testing!"

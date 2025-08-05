#!/bin/bash

# ==============================================================================
# NIS Protocol v3 - Fresh Rebuild and Endpoint Testing Script
# ==============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

function print_info {
    echo -e "${BLUE}[INFO] $1${NC}"
}

function print_success {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

function print_warning {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

function print_error {
    echo -e "${RED}[ERROR] $1${NC}"
}

# Step 1: Clean everything
print_info "🧹 Cleaning up old containers and images..."
docker-compose -p nis-protocol-v3 down -v --remove-orphans
docker system prune -f
docker builder prune -f

# Step 2: Check if .env exists
if [ ! -f ".env" ]; then
    print_warning "⚠️  No .env file found. Creating from template..."
    if [ -f "environment-template.txt" ]; then
        cp environment-template.txt .env
        print_warning "Please edit .env with your API keys!"
    elif [ -f "dev/environment-template.txt" ]; then
        cp dev/environment-template.txt .env
        print_warning "Please edit .env with your API keys!"
    else
        print_error "No environment template found!"
        exit 1
    fi
fi

# Step 3: Force rebuild everything
print_info "🔨 Force rebuilding all Docker images (this will take several minutes)..."
docker-compose build --no-cache --progress=plain

if [ $? -ne 0 ]; then
    print_error "❌ Docker build failed!"
    exit 1
fi

print_success "✅ Docker images rebuilt successfully"

# Step 4: Start services
print_info "🚀 Starting all services..."
docker-compose -p nis-protocol-v3 up -d --force-recreate

# Step 5: Wait for services to be ready
print_info "⏳ Waiting for services to start..."
sleep 30

# Step 6: Check service health
print_info "🔍 Checking service health..."
docker-compose -p nis-protocol-v3 ps

# Step 7: Test basic connectivity
print_info "🌐 Testing basic connectivity..."

# Test if backend is responding
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        print_success "✅ Backend is responding!"
        break
    else
        print_warning "⏳ Waiting for backend... (attempt $i/10)"
        sleep 5
    fi
    
    if [ $i -eq 10 ]; then
        print_error "❌ Backend failed to start after 50 seconds"
        print_info "📋 Showing logs..."
        docker-compose -p nis-protocol-v3 logs backend
        exit 1
    fi
done

print_success "🎉 System is ready for endpoint testing!"
print_info "📍 System accessible at: http://localhost:80"
print_info "📍 Direct backend: http://localhost:8000"
print_info "📋 Run 'python test_endpoints.py' to test all endpoints"
#!/bin/bash

# NIS Protocol v3 - Complete System Startup Script
# This script starts all required services in the proper order

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="nis-protocol-v3"
TIMEOUT=300  # 5 minutes timeout for services to start

# Function to print colored output
print_status() {
    echo -e "${BLUE}[NIS-V3]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    print_status "Checking Docker availability..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    print_success "Docker and Docker Compose are available"
}

# Function to create required directories
create_directories() {
    print_status "Creating required directories..."
    
    directories=(
        "logs"
        "data"
        "models"
        "cache"
        "config/postgres_init"
        "monitoring/grafana/dashboards"
        "monitoring/grafana/datasources"
        "static"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        fi
    done
    
    print_success "All required directories are ready"
}

# Function to validate API keys and create .env file
validate_and_setup_env() {
    print_status "Validating environment and API keys..."

    if [ ! -f ".env" ]; then
        if [ -f ".env~" ]; then
            print_status "Creating .env file from template .env~..."
            cp .env~ .env
            print_success ".env file created from template."
        else
            print_error "Neither .env nor .env~ found. Please create a .env file with your API keys."
            exit 1
        fi
    fi

    source .env
    
    if [[ -z "$OPENAI_API_KEY" || -z "$ANTHROPIC_API_KEY" || -z "$DEEPSEEK_API_KEY" || -z "$GOOGLE_API_KEY" ]]; then
        print_error "One or more LLM provider API keys are missing in .env file!"
        exit 1
    fi

    print_success "API keys are present."
}


# Function to create Redis configuration
create_redis_config() {
    if [ ! -f "config/redis.conf" ]; then
        print_status "Creating Redis configuration..."
        mkdir -p config
        cat > config/redis.conf << EOF
# NIS Protocol v3 Redis Configuration
maxmemory 512mb
maxmemory-policy allkeys-lru
appendonly yes
appendfsync everysec

# Security
protected-mode yes
bind 0.0.0.0

# Performance
tcp-keepalive 300
timeout 0
tcp-backlog 511

# Logging
loglevel notice
EOF
        print_success "Redis configuration created"
    fi
}

# Function to create database initialization script
create_db_init() {
    if [ ! -f "config/postgres_init/init.sql" ]; then
        print_status "Creating database initialization script..."
        mkdir -p config/postgres_init
        cat > config/postgres_init/init.sql << EOF
-- NIS Protocol v3 Database Initialization

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS nis_agents;
CREATE SCHEMA IF NOT EXISTS nis_monitoring;
CREATE SCHEMA IF NOT EXISTS nis_infrastructure;

-- Create basic tables for agents
CREATE TABLE IF NOT EXISTS nis_agents.agent_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_type VARCHAR(50) NOT NULL,
    session_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create tables for monitoring
CREATE TABLE IF NOT EXISTS nis_monitoring.performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL,
    metric_unit VARCHAR(20),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_agent_sessions_type ON nis_agents.agent_sessions(agent_type);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_name ON nis_monitoring.performance_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON nis_monitoring.performance_metrics(timestamp);

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA nis_agents TO nis_user;
GRANT ALL PRIVILEGES ON SCHEMA nis_monitoring TO nis_user;
GRANT ALL PRIVILEGES ON SCHEMA nis_infrastructure TO nis_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA nis_agents TO nis_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA nis_monitoring TO nis_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA nis_infrastructure TO nis_user;
EOF
        print_success "Database initialization script created"
    fi
}

# Function to check service health
check_service_health() {
    local service_name=$1
    local container_name=$2
    local health_check_command=$3
    local max_attempts=30
    local attempt=0
    
    print_status "Checking health of $service_name..."
    
    while [ $attempt -lt $max_attempts ]; do
        if docker-compose -p "$PROJECT_NAME" exec -T "$container_name" sh -c "$health_check_command" &> /dev/null; then
            print_success "$service_name is healthy"
            return 0
        fi
        
        attempt=$((attempt + 1))
        print_status "Waiting for $service_name... (attempt $attempt/$max_attempts)"
        sleep 10
    done
    
    print_error "$service_name failed to become healthy"
    return 1
}

# Function to start services
start_services() {
    print_status "Starting NIS Protocol v3 services..."
    docker-compose -p "$PROJECT_NAME" up -d --build
    
    print_status "Waiting for services to become healthy..."
    
    check_service_health "Redis" "redis" "redis-cli ping"
    check_service_health "Postgres" "postgres" "pg_isready -U nis_user -d nis_protocol_v3"
    check_service_health "Kafka" "kafka" "kafka-topics --bootstrap-server localhost:9092 --list"
    check_service_health "NIS Backend" "backend" "curl -f http://localhost:8000/health"
}

# Function to display service URLs
display_urls() {
    print_success "NIS Protocol v3 is now running!"
    echo ""
    echo "🌐 Service URLs:"
    echo "  • Main API:          http://localhost/"
    echo "  • API Documentation: http://localhost/docs"
    echo "  • Health Check:      http://localhost/health"
    echo ""
    echo "📊 Quick Status Check:"
    echo "  • docker-compose -p $PROJECT_NAME ps"
    echo "  • curl http://localhost/health"
    echo ""
    echo "🛑 To stop the system:"
    echo "  • ./stop.sh"
}

# Main execution
main() {
    print_status "Starting NIS Protocol v3 Complete System..."
    echo ""
    
    # Pre-flight checks
    check_docker
    create_directories
    validate_and_setup_env
    create_redis_config
    create_db_init
    
    # Start services
    start_services
    
    # Display information
    display_urls
    
    print_success "NIS Protocol v3 startup completed successfully!"
}

# Run main function
main "$@" 
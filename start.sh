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

# Function to validate API keys
validate_api_keys() {
    print_status "Validating LLM provider API keys..."
    
    if [ ! -f ".env" ]; then
        print_error "No .env file found! Please create one with your API keys."
        print_status "Run the following commands to create .env file:"
        echo -e "  ${YELLOW}cat > .env << EOF"
        echo -e "  OPENAI_API_KEY=your_openai_api_key_here"
        echo -e "  ANTHROPIC_API_KEY=your_anthropic_api_key_here"
        echo -e "  DEEPSEEK_API_KEY=your_deepseek_api_key_here"
        echo -e "  GOOGLE_API_KEY=your_google_api_key_here"
        echo -e "  EOF${NC}"
        exit 1
    fi
    
    # Source the .env file to check variables
    source .env
    
    # Check if at least one API key is provided
    if [[ -z "$OPENAI_API_KEY" && -z "$ANTHROPIC_API_KEY" && -z "$DEEPSEEK_API_KEY" && -z "$GOOGLE_API_KEY" ]]; then
        print_error "No LLM provider API keys found in .env file!"
        print_status "You need at least one of the following API keys:"
        echo -e "  • ${YELLOW}OPENAI_API_KEY${NC} - Get from: https://platform.openai.com/api-keys"
        echo -e "  • ${YELLOW}ANTHROPIC_API_KEY${NC} - Get from: https://console.anthropic.com/"
        echo -e "  • ${YELLOW}DEEPSEEK_API_KEY${NC} - Get from: https://platform.deepseek.com/"
        echo -e "  • ${YELLOW}GOOGLE_API_KEY${NC} - Get from: https://makersuite.google.com/app/apikey"
        exit 1
    fi
    
    # Check for placeholder values
    placeholder_found=false
    if [[ "$OPENAI_API_KEY" == "your_openai_api_key_here" ]]; then
        print_warning "OPENAI_API_KEY appears to be a placeholder"
        placeholder_found=true
    fi
    if [[ "$ANTHROPIC_API_KEY" == "your_anthropic_api_key_here" ]]; then
        print_warning "ANTHROPIC_API_KEY appears to be a placeholder"
        placeholder_found=true
    fi
    if [[ "$DEEPSEEK_API_KEY" == "your_deepseek_api_key_here" ]]; then
        print_warning "DEEPSEEK_API_KEY appears to be a placeholder"
        placeholder_found=true
    fi
    if [[ "$GOOGLE_API_KEY" == "your_google_api_key_here" ]]; then
        print_warning "GOOGLE_API_KEY appears to be a placeholder"
        placeholder_found=true
    fi
    
    if [ "$placeholder_found" = true ]; then
        print_error "Placeholder API keys detected! Please replace with actual keys."
        exit 1
    fi
    
    # Show which providers are configured
    print_status "Configured LLM providers:"
    if [[ -n "$OPENAI_API_KEY" && "$OPENAI_API_KEY" != "your_openai_api_key_here" ]]; then
        print_success "  ✓ OpenAI"
    fi
    if [[ -n "$ANTHROPIC_API_KEY" && "$ANTHROPIC_API_KEY" != "your_anthropic_api_key_here" ]]; then
        print_success "  ✓ Anthropic"
    fi
    if [[ -n "$DEEPSEEK_API_KEY" && "$DEEPSEEK_API_KEY" != "your_deepseek_api_key_here" ]]; then
        print_success "  ✓ DeepSeek"
    fi
    if [[ -n "$GOOGLE_API_KEY" && "$GOOGLE_API_KEY" != "your_google_api_key_here" ]]; then
        print_success "  ✓ Google"
    fi
}

# Function to create environment file if it doesn't exist
create_env_file() {
    if [ ! -f ".env" ]; then
        print_status "Creating .env file template..."
        cat > .env << EOF
# 🔑 NIS Protocol v3 - LLM Provider API Keys (REQUIRED)
# Get your API keys from the respective provider websites:
# • OpenAI: https://platform.openai.com/api-keys
# • Anthropic: https://console.anthropic.com/
# • DeepSeek: https://platform.deepseek.com/
# • Google: https://makersuite.google.com/app/apikey

OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Infrastructure Configuration (Docker defaults)
COMPOSE_PROJECT_NAME=${PROJECT_NAME}
DATABASE_URL=postgresql://nis_user:nis_password_2025@postgres:5432/nis_protocol_v3
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# Application Configuration
NIS_ENV=production
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
DASHBOARD_PORT=5000

# Monitoring Configuration
GRAFANA_ADMIN_PASSWORD=nis_admin_2025
EOF
        print_warning "Environment template created at .env"
        print_error "⚠️  IMPORTANT: Edit .env file and add your actual API keys before continuing!"
        print_status "Edit the file with: nano .env (or your preferred editor)"
        exit 1
    fi
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
    local health_endpoint=$2
    local max_attempts=30
    local attempt=0
    
    print_status "Checking health of $service_name..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -f -s "$health_endpoint" > /dev/null 2>&1; then
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

# Function to start core infrastructure services
start_infrastructure() {
    print_status "Starting core infrastructure services..."
    
    # Start PostgreSQL, Zookeeper, Kafka, and Redis first
    docker-compose -p "$PROJECT_NAME" up -d postgres zookeeper kafka redis
    
    print_status "Waiting for infrastructure services to be ready..."
    sleep 30
    
    # Check if services are healthy
    print_status "Verifying infrastructure health..."
    
    # Wait for PostgreSQL
    local postgres_ready=false
    for i in {1..30}; do
        if docker-compose -p "$PROJECT_NAME" exec -T postgres pg_isready -U nis_user -d nis_protocol_v3 &> /dev/null; then
            postgres_ready=true
            break
        fi
        print_status "Waiting for PostgreSQL... ($i/30)"
        sleep 5
    done
    
    if [ "$postgres_ready" = true ]; then
        print_success "PostgreSQL is ready"
    else
        print_error "PostgreSQL failed to start"
        return 1
    fi
    
    # Wait for Redis
    local redis_ready=false
    for i in {1..20}; do
        if docker-compose -p "$PROJECT_NAME" exec -T redis redis-cli ping &> /dev/null; then
            redis_ready=true
            break
        fi
        print_status "Waiting for Redis... ($i/20)"
        sleep 3
    done
    
    if [ "$redis_ready" = true ]; then
        print_success "Redis is ready"
    else
        print_error "Redis failed to start"
        return 1
    fi
    
    print_success "Core infrastructure is ready"
}

# Function to start application services
start_application() {
    print_status "Starting NIS Protocol v3 application..."
    
    # Start the main application
    docker-compose -p "$PROJECT_NAME" up -d nis-app
    
    print_status "Waiting for application to start..."
    sleep 45
    
    # Check application health
    if check_service_health "NIS Application" "http://localhost:8000/health"; then
        print_success "NIS Protocol v3 application started successfully"
    else
        print_error "NIS Protocol v3 application failed to start"
        return 1
    fi
}

# Function to start reverse proxy
start_proxy() {
    print_status "Starting reverse proxy..."
    
    docker-compose -p "$PROJECT_NAME" up -d nginx
    
    sleep 10
    
    if check_service_health "Nginx Proxy" "http://localhost/health"; then
        print_success "Reverse proxy started successfully"
    else
        print_warning "Reverse proxy may not be fully ready"
    fi
}

# Function to start monitoring services (optional)
start_monitoring() {
    if [ "$1" = "--with-monitoring" ]; then
        print_status "Starting monitoring services..."
        docker-compose -p "$PROJECT_NAME" --profile monitoring up -d
        print_success "Monitoring services started"
    fi
}

# Function to display service URLs
display_urls() {
    print_success "NIS Protocol v3 is now running!"
    echo ""
    echo "🌐 Service URLs:"
    echo "  • Main API:          http://localhost/              (via nginx)"
    echo "  • Direct API:        http://localhost:8000/         (direct)"
    echo "  • API Documentation: http://localhost/docs"
    echo "  • Health Check:      http://localhost/health"
    echo "  • Monitoring:        http://localhost/dashboard/"
    echo ""
    echo "🛠️  Development URLs:"
    echo "  • Kafka UI:          http://localhost:8080/         (if --with-monitoring)"
    echo "  • Redis Commander:   http://localhost:8081/         (if --with-monitoring)"
    echo "  • Grafana:           http://localhost:3000/         (if --with-monitoring)"
    echo "  • Prometheus:        http://localhost:9090/         (if --with-monitoring)"
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
    create_env_file
    validate_api_keys
    
    # Run comprehensive environment validation (skip if --no-validation flag is provided)
    if [ "$1" != "--no-validation" ]; then
        print_status "Running comprehensive environment validation..."
        if command -v python3 &> /dev/null; then
            if python3 validate_environment.py; then
                print_success "Environment validation passed"
            else
                print_error "Environment validation failed"
                print_warning "You can skip validation with: $0 --no-validation"
                exit 1
            fi
        else
            print_warning "Python3 not available for environment validation, proceeding..."
        fi
    else
        print_warning "Skipping environment validation (--no-validation flag provided)"
    fi
    
    create_redis_config
    create_db_init
    
    # Start services in order
    start_infrastructure
    start_application
    start_proxy
    start_monitoring "$1"
    
    # Display information
    display_urls
    
    print_success "NIS Protocol v3 startup completed successfully!"
}

# Handle script arguments
if [ "$1" = "--help" ]; then
    echo "NIS Protocol v3 Startup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help              Show this help message"
    echo "  --with-monitoring   Start with monitoring services (Kafka UI, Redis Commander, Grafana)"
    echo "  --no-validation     Skip Python environment validation (useful when using Docker)"
    echo ""
    echo "Examples:"
    echo "  $0                  Start core system only"
    echo "  $0 --with-monitoring Start with full monitoring stack"
    echo "  $0 --no-validation  Skip validation and start directly (Docker deployments)"
    exit 0
fi

# Run main function
main "$1" 
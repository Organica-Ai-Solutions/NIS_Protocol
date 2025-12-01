#!/bin/bash

# NIS Protocol v3 - Complete System Reset Script
# This script completely resets the system, removing all data and rebuilding from scratch

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="nis-protocol-v3"

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

# Function to show warning and get confirmation
show_warning() {
    print_warning "⚠️  COMPLETE SYSTEM RESET WARNING ⚠️"
    echo ""
    echo "This will PERMANENTLY DELETE:"
    echo "  🗄️  All database data (PostgreSQL)"
    echo "  📧 All message queues (Kafka)"
    echo "  🧠 All cached data (Redis)"
    echo "  📊 All monitoring data (Grafana, Prometheus)"
    echo "  📝 All logs and temporary files"
    echo "  🐳 All containers and images"
    echo "  🔧 All configuration files (recreated from defaults)"
    echo ""
    echo "💾 Data that will be preserved:"
    echo "  • Source code in src/"
    echo "  • Documentation in docs/"
    echo "  • Configuration templates"
    echo ""
    print_warning "This action cannot be undone!"
    echo ""
    
    if [ "$1" != "--force" ]; then
        read -p "Are you absolutely sure you want to reset everything? (type 'RESET' to continue): " -r
        if [ "$REPLY" != "RESET" ]; then
            print_status "Reset cancelled"
            exit 0
        fi
    fi
}

# Function to stop all services gracefully
stop_all_services() {
    print_status "Stopping all NIS Protocol v3 services gracefully..."
    
    # Use the stop.sh script for a graceful shutdown
    if [ -f "./stop.sh" ]; then
        ./stop.sh --remove-containers
    else
        # Fallback to more aggressive stop if stop.sh is not available
        print_warning "stop.sh not found, using aggressive shutdown..."
        docker-compose -p "$PROJECT_NAME" down --remove-orphans --timeout 30 2>/dev/null || true
    fi
    
    print_success "All services stopped and removed"
}

# Function to remove all volumes and data
remove_all_data() {
    print_status "Removing all data volumes..."
    
    # Remove all volumes
    docker-compose -p "$PROJECT_NAME" down -v --remove-orphans 2>/dev/null || true
    docker-compose -p "$PROJECT_NAME" --profile monitoring down -v --remove-orphans 2>/dev/null || true
    
    # Clean up local data directories
    print_status "Cleaning up local data directories..."
    
    directories_to_clean=(
        "logs"
        "data"
        "cache"
        "models/checkpoints"
        "models/trained"
    )
    
    for dir in "${directories_to_clean[@]}"; do
        if [ -d "$dir" ]; then
            rm -rf "$dir"
            print_status "Removed: $dir"
        fi
    done
    
    print_success "All data removed"
}

# Function to remove Docker images
remove_images() {
    print_status "Removing Docker images..."
    
    # Remove project-specific images
    docker-compose -p "$PROJECT_NAME" down --rmi all 2>/dev/null || true
    
    # Remove dangling images
    docker image prune -f 2>/dev/null || true
    
    # Remove build cache
    docker builder prune -f 2>/dev/null || true
    
    print_success "Docker images cleaned up"
}

# Function to clean configuration files
clean_configuration() {
    print_status "Cleaning configuration files..."
    
    config_files=(
        ".env"
        "config/redis.conf"
        "config/postgres_init/init.sql"
    )
    
    for file in "${config_files[@]}"; do
        if [ -f "$file" ]; then
            rm -f "$file"
            print_status "Removed: $file"
        fi
    done
    
    print_success "Configuration files cleaned"
}

# Function to rebuild system
rebuild_system() {
    print_status "Rebuilding NIS Protocol v3 system from scratch..."
    
    # Rebuild Docker images
    print_status "Building fresh Docker images..."
    docker-compose -p "$PROJECT_NAME" build --no-cache --pull
    
    print_success "System rebuilt successfully"
}

# Function to create fresh directories
create_fresh_directories() {
    print_status "Creating fresh directory structure..."
    
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
        mkdir -p "$dir"
        print_status "Created: $dir"
    done
    
    print_success "Fresh directory structure created"
}

# Function to perform comprehensive Docker cleanup
comprehensive_docker_cleanup() {
    print_status "Performing comprehensive Docker cleanup..."
    
    # Stop all containers
    print_status "Stopping all containers..."
    docker stop $(docker ps -a -q) 2>/dev/null || true
    
    # Remove all containers
    print_status "Removing all containers..."
    docker rm $(docker ps -a -q) 2>/dev/null || true
    
    # Remove all images
    print_status "Removing all images..."
    docker rmi $(docker images -q) -f 2>/dev/null || true
    
    # Remove all volumes
    print_status "Removing all volumes..."
    docker volume rm $(docker volume ls -q) 2>/dev/null || true
    
    # Remove all networks
    print_status "Removing all networks..."
    docker network rm $(docker network ls -q) 2>/dev/null || true
    
    # Clean system
    docker system prune -a -f --volumes 2>/dev/null || true
    
    print_success "Comprehensive Docker cleanup completed"
}

# Function to verify reset
verify_reset() {
    print_status "Verifying reset completion..."
    
    # Check for remaining containers
    local containers=$(docker ps -a --filter "name=${PROJECT_NAME}" -q | wc -l)
    print_status "Remaining project containers: $containers"
    
    # Check for remaining volumes
    local volumes=$(docker volume ls --filter "name=${PROJECT_NAME}" -q | wc -l)
    print_status "Remaining project volumes: $volumes"
    
    # Check directory sizes
    if [ -d "logs" ]; then
        local log_size=$(du -sh logs 2>/dev/null | cut -f1 || echo "0")
        print_status "Logs directory size: $log_size"
    fi
    
    print_success "Reset verification completed"
}

# Function to start fresh system
start_fresh() {
    if [ "$1" = "--start" ]; then
        print_status "Starting fresh NIS Protocol v3 system..."
        ./start.sh
    else
        echo ""
        print_success "🎉 NIS Protocol v3 reset completed successfully!"
        echo ""
        echo "🚀 To start the fresh system:"
        echo "  • ./start.sh"
        echo "  • ./start.sh --with-monitoring"
        echo ""
        echo "📋 What was reset:"
        echo "  ✅ All containers removed"
        echo "  ✅ All volumes deleted"
        echo "  ✅ All data cleared"
        echo "  ✅ All caches emptied"
        echo "  ✅ Fresh directory structure created"
        echo "  ✅ Configuration files reset to defaults"
        echo ""
        echo "🔧 System is ready for fresh installation"
    fi
}

# Main execution
main() {
    print_status "Initiating NIS Protocol v3 Complete System Reset..."
    echo ""
    
    # Show warning and get confirmation
    show_warning "$1"
    
    print_status "Starting reset process..."
    echo ""
    
    # Reset process
    stop_all_services
    remove_all_data
    
    if [ "$1" = "--deep" ] || [ "$2" = "--deep" ]; then
        print_warning "Performing deep cleanup (removing all Docker resources)..."
        comprehensive_docker_cleanup
    else
        remove_images
    fi
    
    clean_configuration
    create_fresh_directories
    rebuild_system
    verify_reset
    
    # Start if requested
    start_fresh "$2" "$3"
    
    print_success "NIS Protocol v3 reset process completed!"
}

# Handle script arguments
if [ "$1" = "--help" ]; then
    echo "NIS Protocol v3 Complete System Reset Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help     Show this help message"
    echo "  --force    Skip confirmation prompt"
    echo "  --deep     Perform deep cleanup (remove ALL Docker resources)"
    echo "  --start    Automatically start the system after reset"
    echo ""
    echo "Examples:"
    echo "  $0                    Interactive reset with confirmation"
    echo "  $0 --force            Reset without confirmation"
    echo "  $0 --deep             Deep reset removing all Docker resources"
    echo "  $0 --force --start    Reset and immediately start fresh system"
    echo ""
    echo "⚠️  WARNING: This will permanently delete all data!"
    echo ""
    echo "🔄 Alternative commands:"
    echo "  • ./stop.sh --remove-volumes    (less destructive option)"
    echo "  • ./start.sh                    (start existing system)"
    exit 0
fi

# Additional safety check for deep cleanup
if [ "$1" = "--deep" ] || [ "$2" = "--deep" ]; then
    print_warning "DEEP CLEANUP WARNING"
    echo "This will remove ALL Docker resources on your system, not just NIS Protocol v3!"
    echo "This includes containers, images, and volumes from other projects."
    echo ""
    if [ "$1" != "--force" ]; then
        read -p "Are you sure you want to perform deep cleanup? (type 'DEEP' to continue): " -r
        if [ "$REPLY" != "DEEP" ]; then
            print_status "Deep cleanup cancelled"
            exit 0
        fi
    fi
fi

# Run main function
main "$@" 
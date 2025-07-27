#!/bin/bash

# NIS Protocol v3 - Complete System Shutdown Script
# This script stops all services gracefully in the proper order

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

# Function to check if containers are running
check_containers() {
    local running_containers=$(docker-compose -p "$PROJECT_NAME" ps -q)
    if [ -z "$running_containers" ]; then
        return 1
    else
        return 0
    fi
}

# Function to stop services gracefully
stop_services() {
    print_status "Stopping NIS Protocol v3 services..."
    
    if ! check_containers; then
        print_warning "No running containers found for project: $PROJECT_NAME"
        return 0
    fi
    
    # Stop services in reverse order of dependency
    print_status "Stopping reverse proxy..."
    docker-compose -p "$PROJECT_NAME" stop nginx 2>/dev/null || true
    
    print_status "Stopping monitoring services..."
    docker-compose -p "$PROJECT_NAME" --profile monitoring stop 2>/dev/null || true
    
    print_status "Stopping main application..."
    docker-compose -p "$PROJECT_NAME" stop nis-app 2>/dev/null || true
    
    print_status "Stopping infrastructure services..."
    docker-compose -p "$PROJECT_NAME" stop kafka redis postgres zookeeper 2>/dev/null || true
    
    print_success "All services stopped"
}

# Function to remove containers
remove_containers() {
    if [ "$1" = "--remove-containers" ]; then
        print_status "Removing containers..."
        docker-compose -p "$PROJECT_NAME" down --remove-orphans 2>/dev/null || true
        docker-compose -p "$PROJECT_NAME" --profile monitoring down --remove-orphans 2>/dev/null || true
        print_success "Containers removed"
    fi
}

# Function to remove volumes
remove_volumes() {
    if [ "$1" = "--remove-volumes" ]; then
        print_warning "Removing all data volumes..."
        read -p "This will delete all data. Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose -p "$PROJECT_NAME" down -v --remove-orphans 2>/dev/null || true
            docker-compose -p "$PROJECT_NAME" --profile monitoring down -v --remove-orphans 2>/dev/null || true
            print_success "Volumes removed"
        else
            print_status "Volume removal cancelled"
        fi
    fi
}

# Function to clean up Docker resources
cleanup_docker() {
    if [ "$1" = "--cleanup" ]; then
        print_status "Cleaning up Docker resources..."
        
        # Remove dangling images
        docker image prune -f 2>/dev/null || true
        
        # Remove unused networks
        docker network prune -f 2>/dev/null || true
        
        print_success "Docker cleanup completed"
    fi
}

# Function to show status
show_status() {
    print_status "Current container status:"
    docker-compose -p "$PROJECT_NAME" ps 2>/dev/null || echo "No containers found"
    
    echo ""
    print_status "Docker system information:"
    echo "  • Running containers: $(docker ps -q | wc -l)"
    echo "  • Total containers: $(docker ps -a -q | wc -l)"
    echo "  • Images: $(docker images -q | wc -l)"
    echo "  • Volumes: $(docker volume ls -q | wc -l)"
    echo "  • Networks: $(docker network ls -q | wc -l)"
}

# Function to force stop everything
force_stop() {
    if [ "$1" = "--force" ]; then
        print_warning "Force stopping all NIS Protocol v3 containers..."
        
        # Kill all containers
        docker-compose -p "$PROJECT_NAME" kill 2>/dev/null || true
        docker-compose -p "$PROJECT_NAME" --profile monitoring kill 2>/dev/null || true
        
        # Remove containers
        docker-compose -p "$PROJECT_NAME" rm -f 2>/dev/null || true
        docker-compose -p "$PROJECT_NAME" --profile monitoring rm -f 2>/dev/null || true
        
        print_success "Force stop completed"
    fi
}

# Function to save logs before shutdown
save_logs() {
    if [ "$1" = "--save-logs" ]; then
        print_status "Saving container logs..."
        
        local log_dir="logs/shutdown-$(date +%Y%m%d-%H%M%S)"
        mkdir -p "$log_dir"
        
        # Save logs for each service
        services=("postgres" "redis" "kafka" "zookeeper" "nis-app" "nginx")
        for service in "${services[@]}"; do
            if docker-compose -p "$PROJECT_NAME" ps "$service" 2>/dev/null | grep -q "Up"; then
                print_status "Saving logs for $service..."
                docker-compose -p "$PROJECT_NAME" logs "$service" > "$log_dir/$service.log" 2>/dev/null || true
            fi
        done
        
        print_success "Logs saved to: $log_dir"
    fi
}

# Main execution
main() {
    print_status "Shutting down NIS Protocol v3 Complete System..."
    echo ""
    
    # Save logs if requested
    save_logs "$@"
    
    # Stop services
    stop_services
    
    # Handle additional options
    for arg in "$@"; do
        case $arg in
            --remove-containers)
                remove_containers "$arg"
                ;;
            --remove-volumes)
                remove_volumes "$arg"
                ;;
            --cleanup)
                cleanup_docker "$arg"
                ;;
            --force)
                force_stop "$arg"
                ;;
        esac
    done
    
    # Show final status
    show_status
    
    print_success "NIS Protocol v3 shutdown completed!"
    echo ""
    echo "🔄 To restart the system:"
    echo "  • ./start.sh"
    echo ""
    echo "🧹 Additional cleanup options:"
    echo "  • ./stop.sh --remove-containers  (remove containers)"
    echo "  • ./stop.sh --remove-volumes     (remove all data)"
    echo "  • ./stop.sh --cleanup            (cleanup Docker resources)"
    echo "  • ./stop.sh --force              (force kill all containers)"
    echo "  • ./stop.sh --save-logs          (save logs before shutdown)"
}

# Handle script arguments
if [ "$1" = "--help" ]; then
    echo "NIS Protocol v3 Shutdown Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help                Show this help message"
    echo "  --remove-containers   Remove containers after stopping"
    echo "  --remove-volumes      Remove all data volumes (DESTRUCTIVE)"
    echo "  --cleanup             Clean up unused Docker resources"
    echo "  --force               Force kill all containers"
    echo "  --save-logs           Save container logs before shutdown"
    echo ""
    echo "Examples:"
    echo "  $0                           Stop services gracefully"
    echo "  $0 --remove-containers       Stop and remove containers"
    echo "  $0 --remove-volumes          Stop and remove everything including data"
    echo "  $0 --force --save-logs       Save logs and force stop"
    echo ""
    echo "⚠️  WARNING: --remove-volumes will delete all data permanently!"
    exit 0
fi

# Warn about destructive operations
for arg in "$@"; do
    if [ "$arg" = "--remove-volumes" ]; then
        print_warning "WARNING: This operation will delete all data!"
        echo "  • Database data will be lost"
        echo "  • Kafka messages will be lost"
        echo "  • Redis cache will be cleared"
        echo "  • All logs and models will be deleted"
        echo ""
        break
    fi
done

# Run main function
main "$@" 
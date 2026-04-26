#!/bin/bash

# NIS Protocol v3.2.5 - CPU Mode Stop Script

COMPOSE_FILE="docker-compose.cpu.yml"
PROJECT_NAME="nis-protocol-v3"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}[CPU-MODE] Stopping NIS Protocol CPU stack...${NC}"

# Save logs if requested
if [ "$1" = "--save-logs" ]; then
    LOG_DIR="logs/shutdown-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$LOG_DIR"
    echo -e "${BLUE}[CPU-MODE] Saving logs to $LOG_DIR${NC}"
    docker compose -f "$COMPOSE_FILE" logs > "$LOG_DIR/all-services.log" 2>&1
fi

# Stop services
docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down

if [ "$1" = "--volumes" ]; then
    echo -e "${BLUE}[CPU-MODE] Removing volumes...${NC}"
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down -v
fi

echo -e "${GREEN}[SUCCESS] CPU stack stopped${NC}"
echo ""
echo "To restart: ./start-cpu.sh"

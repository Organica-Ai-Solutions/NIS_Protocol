#!/bin/bash
# 🚨 EMERGENCY SHUTDOWN SCRIPT
# Stops all NIS Protocol services immediately to prevent billing

echo "🚨 EMERGENCY SHUTDOWN INITIATED"
echo "Stopping all Docker containers..."

# Stop all NIS containers
docker stop $(docker ps -q --filter "name=nis-*") 2>/dev/null || echo "No NIS containers running"

# Stop docker-compose services
docker-compose down 2>/dev/null || echo "Docker-compose already down"

# Kill any remaining Python processes
pkill -f "nis" 2>/dev/null || echo "No NIS processes found"
pkill -f "uvicorn.*main" 2>/dev/null || echo "No uvicorn processes found"

echo "✅ EMERGENCY SHUTDOWN COMPLETE"
echo "💰 API billing should now be stopped"
echo ""
echo "To restart safely:"
echo "  1. Check your .env file has API keys commented out"
echo "  2. Use: ./start.sh"
echo "  3. Monitor billing at: https://console.cloud.google.com/billing"

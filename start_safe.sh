#!/bin/bash
# 🛡️ SAFE START SCRIPT - Prevents Billing Disasters
# This script starts NIS Protocol with billing protection enabled

echo "🛡️ Starting NIS Protocol in SAFE MODE..."
echo "💰 Billing Protection: ENABLED"
echo "🤖 API Calls: MOCK ONLY"
echo ""

# Check if .env.safe exists, if not create it
if [ ! -f ".env.safe" ]; then
    echo "⚠️ Creating .env.safe configuration..."
    cp .env.example .env.safe 2>/dev/null || echo "# Safe configuration" > .env.safe
fi

# Use safe configuration
echo "🔒 Using safe configuration (.env.safe)"
cp .env.safe .env

# Add safety environment variables
echo "" >> .env
echo "# 🛡️ BILLING PROTECTION" >> .env
echo "FORCE_MOCK_MODE=true" >> .env
echo "DISABLE_REAL_API_CALLS=true" >> .env

# Start services
echo "🚀 Starting Docker services..."
docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 10

# Check if services are running
if docker ps | grep -q "nis-backend"; then
    echo "✅ NIS Protocol started successfully in SAFE MODE"
    echo ""
    echo "🌐 Access points:"
    echo "  • Chat Console: http://localhost/console"
    echo "  • API Docs: http://localhost/docs"
    echo "  • Health Check: http://localhost/health"
    echo ""
    echo "🛡️ SAFETY STATUS:"
    echo "  • Real API calls: DISABLED"
    echo "  • Billing risk: MINIMAL"
    echo "  • Mock responses: ENABLED"
    echo ""
    echo "⚠️ To enable real APIs (BILLING RISK):"
    echo "  1. Edit .env file manually"
    echo "  2. Set FORCE_MOCK_MODE=false"
    echo "  3. Add real API keys"
    echo "  4. Restart with ./start.sh"
    echo ""
    echo "🚨 Emergency shutdown: ./scripts/emergency/emergency_shutdown.sh"
else
    echo "❌ Failed to start services"
    exit 1
fi

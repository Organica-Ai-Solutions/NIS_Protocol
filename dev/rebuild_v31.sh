#!/bin/bash
echo "🔄 Rebuilding NIS Protocol v3.1 Application..."
docker-compose build --no-cache nis-app
echo "✅ Rebuild complete!"
echo "🚀 Starting system..."
docker-compose up -d
echo "✅ v3.1 system started!" 
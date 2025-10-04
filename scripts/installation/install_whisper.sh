#!/bin/bash

# 🎙️ Install Whisper STT for GPT-Like Voice Chat
# Enables real-time voice conversation like ChatGPT

set -e

echo "🎙️ Installing Whisper STT for Voice Chat"
echo "=========================================="
echo ""

# Check if backend container is running
if ! docker ps | grep -q nis-backend; then
    echo "❌ Backend container not running"
    echo "   Run: ./start.sh"
    exit 1
fi

echo "📦 Step 1: Installing Python packages..."
docker exec nis-backend pip install --no-cache-dir openai-whisper soundfile librosa ffmpeg-python || {
    echo "❌ Failed to install Python packages"
    exit 1
}

echo ""
echo "🔧 Step 2: Installing ffmpeg..."
docker exec nis-backend bash -c "apt-get update && apt-get install -y ffmpeg" || {
    echo "❌ Failed to install ffmpeg"
    exit 1
}

echo ""
echo "✅ Step 3: Verifying installation..."
docker exec nis-backend python -c "import whisper; print('✅ Whisper installed successfully!')" || {
    echo "❌ Whisper verification failed"
    exit 1
}

echo ""
echo "🔄 Step 4: Restarting backend..."
docker restart nis-backend
echo "   Waiting for backend to be ready..."
sleep 5

# Wait for health check
for i in {1..30}; do
    if curl -s http://localhost/health > /dev/null 2>&1; then
        echo "✅ Backend is healthy!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  Backend health check timeout"
        echo "   Check: docker logs nis-backend"
    fi
    sleep 1
done

echo ""
echo "=========================================="
echo "✅ Whisper STT Installation Complete!"
echo "=========================================="
echo ""
echo "🎯 Test Voice Chat:"
echo "   1. Open http://localhost/console"
echo "   2. Hard refresh (Cmd+Shift+R)"
echo "   3. Click 🎙️ Voice (enable output)"
echo "   4. Click 🎤 Mic"
echo "   5. Speak your question"
echo "   6. Click ⏹️ Stop"
echo ""
echo "✨ You should see:"
echo "   - ✅ Whisper (95% confident)"
echo "   - Your actual words transcribed"
echo "   - AI responds to what you said!"
echo ""
echo "🎉 GPT-like voice chat is ready!"


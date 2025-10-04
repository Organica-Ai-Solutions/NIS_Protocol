#!/bin/bash

echo "🎙️ INSTALLING BARK TTS FOR CONVERSATIONAL VOICE"
echo "==============================================="
echo ""
echo "This will give you ChatGPT/Grok-quality voice output!"
echo ""

# Check if backend container is running
if ! docker ps | grep -q nis-backend; then
    echo "❌ NIS backend container is not running"
    echo "   Start it with: ./start.sh"
    exit 1
fi

echo "📦 Step 1: Installing Bark and dependencies..."
docker exec nis-backend pip install --no-cache-dir git+https://github.com/suno-ai/bark.git transformers scipy encodec nltk || {
    echo "❌ Failed to install Bark"
    exit 1
}

echo ""
echo "📥 Step 2: Downloading NLTK data..."
docker exec nis-backend python -c "import nltk; nltk.download('punkt')" || {
    echo "⚠️ NLTK download failed (non-critical)"
}

echo ""
echo "✅ Step 3: Verifying Bark installation..."
docker exec nis-backend python -c "
from bark import SAMPLE_RATE, generate_audio, preload_models
print('✅ Bark installed successfully!')
print(f'   Sample rate: {SAMPLE_RATE} Hz')
" || {
    echo "❌ Bark verification failed"
    exit 1
}

echo ""
echo "🔄 Step 4: Restarting backend..."
docker restart nis-backend
echo "   Waiting for backend to be ready..."

# Wait for backend
sleep 15

# Health check
max_attempts=12
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if docker exec nis-backend curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend is healthy!"
        break
    fi
    attempt=$((attempt + 1))
    if [ $attempt -eq $max_attempts ]; then
        echo "⚠️ Backend health check timeout (but may still be working)"
        break
    fi
    sleep 2
done

echo ""
echo "🎉 BARK TTS INSTALLED SUCCESSFULLY!"
echo "===================================="
echo ""
echo "🎤 Voice Quality Comparison:"
echo "   gTTS:  ⭐⭐   (robotic, basic)"
echo "   Bark:  ⭐⭐⭐⭐⭐ (natural, expressive, human-like!)"
echo ""
echo "🚀 Available Voices:"
echo "   • friendly     - Warm, conversational (default)"
echo "   • professional - Clear, authoritative"
echo "   • energetic    - Upbeat, enthusiastic"
echo "   • default      - Neutral, balanced"
echo ""
echo "📝 Next Steps:"
echo "   1. Hard refresh browser (Cmd+Shift+R)"
echo "   2. Open Modern Chat: http://localhost/modern"
echo "   3. Try voice chat - hear the difference!"
echo ""
echo "💡 The AI will now sound like a real person, not a robot!"



# VibeVoice Configuration for NIS Protocol
MODEL_NAME = "microsoft/VibeVoice-1.5B"
LOCAL_MODEL_PATH = "models/vibevoice/VibeVoice-1.5B"
SAMPLE_RATE = 24000
MAX_SPEAKERS = 4
MAX_DURATION_MINUTES = 90
CHUNK_SIZE_MS = 50
STREAMING_ENABLED = True

# Speaker voice profiles
SPEAKER_PROFILES = {
    "consciousness": {"voice_id": 0, "pitch": 0.8, "speed": 0.95},
    "physics": {"voice_id": 1, "pitch": 1.0, "speed": 1.0},
    "research": {"voice_id": 2, "pitch": 1.1, "speed": 1.05},
    "coordination": {"voice_id": 3, "pitch": 1.05, "speed": 1.0}
}

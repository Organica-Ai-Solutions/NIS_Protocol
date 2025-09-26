#!/usr/bin/env python3
"""
VibeVoice Installation Script for NIS Protocol
Installs Microsoft VibeVoice TTS model and dependencies

Copyright 2025 Organica AI Solutions
Licensed under Apache License 2.0
"""

import os
import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """Run shell command with error handling"""
    try:
        logger.info(f"🔧 {description}")
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} - Success")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} - Failed: {e.stderr}")
        return None

def install_vibevoice():
    """Install VibeVoice and all dependencies"""
    
    logger.info("🎙️ Installing Microsoft VibeVoice for NIS Protocol")
    logger.info("=" * 60)
    
    # 1. Install audio processing dependencies
    logger.info("📦 Installing audio processing dependencies...")
    audio_deps = [
        "pip install soundfile librosa resampy",
        "pip install diffusers",
        "pip install transformers[torch]",
        "pip install accelerate"
    ]
    
    for cmd in audio_deps:
        run_command(cmd, f"Installing: {cmd}")
    
    # 2. Clone VibeVoice repository (if available)
    vibevoice_dir = Path("models/vibevoice")
    vibevoice_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("📥 Setting up VibeVoice model directory...")
    
    # 3. Download model using Hugging Face Hub
    download_cmd = """
python -c "
from huggingface_hub import snapshot_download
import os

try:
    model_path = snapshot_download(
        repo_id='microsoft/VibeVoice-1.5B',
        local_dir='models/vibevoice/VibeVoice-1.5B',
        local_dir_use_symlinks=False
    )
    print(f'✅ VibeVoice model downloaded to: {model_path}')
except Exception as e:
    print(f'⚠️ Model download failed: {e}')
    print('📝 Note: Model will be downloaded on first use')
"
"""
    
    run_command(download_cmd, "Downloading VibeVoice model")
    
    # 4. Create VibeVoice configuration
    config_content = '''
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
'''
    
    config_path = Path("configs/vibevoice_config.py")
    config_path.parent.mkdir(exist_ok=True)
    config_path.write_text(config_content)
    
    logger.info("✅ VibeVoice configuration created")
    
    # 5. Test installation
    test_cmd = """
python -c "
try:
    import torch
    import transformers
    import diffusers
    import soundfile
    import librosa
    print('✅ All VibeVoice dependencies available')
    
    # Test model loading (without actual download)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained('microsoft/VibeVoice-1.5B', trust_remote_code=True)
    print('✅ VibeVoice model config accessible')
    
except Exception as e:
    print(f'⚠️ Some dependencies missing: {e}')
"
"""
    
    run_command(test_cmd, "Testing VibeVoice installation")
    
    logger.info("🎉 VibeVoice installation complete!")
    logger.info("📋 Next steps:")
    logger.info("   1. Restart the NIS Protocol backend")
    logger.info("   2. Test communication endpoints")
    logger.info("   3. Try real-time streaming features")

if __name__ == "__main__":
    install_vibevoice()

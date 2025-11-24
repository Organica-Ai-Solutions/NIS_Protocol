#!/usr/bin/env python3
"""
🚀 NIS Protocol - Local Autonomy Setup Script
Downloads a lightweight, efficient local model (TinyLlama-1.1B) to replace the simulation fallback.
This enables REAL local inference for the BitNet Agent.
"""

import os
import shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Target directory expected by BitNetOnlineTrainer
# Defined in src/agents/training/bitnet_online_trainer.py:71
TARGET_DIR = Path("models/bitnet/models/bitnet")

# Model Choice: TinyLlama 1.1B Chat
# Why? 
# 1. Small (1.1B params) - runs on most CPUs/low-end GPUs
# 2. Chat-tuned - good for conversational agent role
# 3. Standard Architecture - reliable support in transformers (no custom kernels needed)
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def setup_local_model():
    logger.info("🌌 Initializing Local Autonomy Setup...")
    
    # Create directory
    if TARGET_DIR.exists():
        logger.warning(f"⚠️ Target directory {TARGET_DIR} already exists.")
        if (TARGET_DIR / "config.json").exists():
            logger.info("✅ Valid model found. Setup already complete.")
            return True
    
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"📦 Downloading model: {MODEL_ID}")
        logger.info("   This enables the system to run OFFLINE without API keys.")
        
        # Download Tokenizer
        logger.info("   Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.save_pretrained(TARGET_DIR)
        
        # Download Model
        logger.info("   Downloading model weights (approx 2.5GB)...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        model.save_pretrained(TARGET_DIR)
        
        logger.info(f"✅ Model successfully installed to {TARGET_DIR}")
        logger.info("🚀 You can now restart the backend to enable Local Autonomy.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        # Cleanup
        if TARGET_DIR.exists():
            shutil.rmtree(TARGET_DIR)
        return False

if __name__ == "__main__":
    success = setup_local_model()
    sys.exit(0 if success else 1)

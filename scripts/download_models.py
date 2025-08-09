#!/usr/bin/env python3
"""
NIS Protocol Model Downloader
Downloads BitNet and Kimi K2 models for fine-tuning
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, login
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configurations
MODELS_CONFIG = {
    "bitnet": {
        # BitNet models from Microsoft Research
        "microsoft/bitnet_b1_58-3b": "models/bitnet/bitnet_3b",
        "microsoft/bitnet_b1_58-7b": "models/bitnet/bitnet_7b",
        # Alternative: Use quantized versions of other models
        "microsoft/DialoGPT-medium": "models/base-models/dialogpt_medium",
    },
    "kimi": {
        # Kimi K2 models (if available publicly)
        "moonshot-ai/kimi-chat": "models/kimi-k2/kimi_chat",
        # Alternative: Use similar long-context models
        "gradientai/Llama-3-8B-Instruct-Gradient-1048k": "models/base-models/llama3_long_context",
        "microsoft/Phi-3-medium-128k-instruct": "models/base-models/phi3_medium_128k",
    },
    "alternatives": {
        # Alternative models for fine-tuning
        "microsoft/phi-2": "models/base-models/phi2",
        "Qwen/Qwen2-7B-Instruct": "models/base-models/qwen2_7b",
        "meta-llama/Llama-2-7b-hf": "models/base-models/llama2_7b",
    }
}

def create_directories():
    """Create necessary directories for models"""
    base_path = Path(".")
    
    directories = [
        "models/bitnet",
        "models/kimi-k2", 
        "models/base-models",
        "models/quantized",
        "models/fine-tuned"
    ]
    
    for dir_path in directories:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {full_path}")

def download_model(model_id: str, local_path: str, token: str = None):
    """Download a model from Hugging Face Hub"""
    try:
        logger.info(f"Downloading {model_id} to {local_path}")
        
        # Create local directory
        Path(local_path).mkdir(parents=True, exist_ok=True)
        
        # Download model
        snapshot_download(
            repo_id=model_id,
            local_dir=local_path,
            token=token,
            ignore_patterns=["*.git*", "README.md", "*.msgpack", "*.safetensors.index.json"]
        )
        
        logger.info(f"✅ Successfully downloaded {model_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {model_id}: {str(e)}")
        return False

def download_bitnet_models(token: str = None):
    """Download BitNet models"""
    logger.info("🔥 Downloading BitNet models...")
    
    success_count = 0
    total_models = len(MODELS_CONFIG["bitnet"])
    
    for model_id, local_path in MODELS_CONFIG["bitnet"].items():
        if download_model(model_id, local_path, token):
            success_count += 1
    
    logger.info(f"BitNet download complete: {success_count}/{total_models} successful")
    return success_count

def download_kimi_models(token: str = None):
    """Download Kimi K2 models"""
    logger.info("🌙 Downloading Kimi/Long-context models...")
    
    success_count = 0
    total_models = len(MODELS_CONFIG["kimi"])
    
    for model_id, local_path in MODELS_CONFIG["kimi"].items():
        if download_model(model_id, local_path, token):
            success_count += 1
    
    logger.info(f"Kimi model download complete: {success_count}/{total_models} successful")
    return success_count

def download_alternative_models(token: str = None):
    """Download alternative models for comparison"""
    logger.info("🔄 Downloading alternative baseline models...")
    
    success_count = 0
    total_models = len(MODELS_CONFIG["alternatives"])
    
    for model_id, local_path in MODELS_CONFIG["alternatives"].items():
        if download_model(model_id, local_path, token):
            success_count += 1
    
    logger.info(f"Alternative models download complete: {success_count}/{total_models} successful")
    return success_count

def main():
    parser = argparse.ArgumentParser(description="Download models for NIS Protocol fine-tuning")
    parser.add_argument("--token", type=str, help="Hugging Face authentication token")
    parser.add_argument("--models", choices=["bitnet", "kimi", "alternatives", "all"], 
                       default="all", help="Which models to download")
    parser.add_argument("--login", action="store_true", help="Login to Hugging Face Hub")
    
    args = parser.parse_args()
    
    # Setup authentication
    if args.login:
        login()
        logger.info("Logged in to Hugging Face Hub")
    
    # Create directories
    create_directories()
    
    # Download models based on selection
    total_success = 0
    
    if args.models in ["bitnet", "all"]:
        total_success += download_bitnet_models(args.token)
    
    if args.models in ["kimi", "all"]:
        total_success += download_kimi_models(args.token)
    
    if args.models in ["alternatives", "all"]:
        total_success += download_alternative_models(args.token)
    
    logger.info(f"🎉 Download complete! {total_success} models downloaded successfully")
    
    # Create model inventory
    create_model_inventory()

def create_model_inventory():
    """Create an inventory of downloaded models"""
    inventory_path = Path("models/model_inventory.txt")
    
    with open(inventory_path, "w") as f:
        f.write("# NIS Protocol Model Inventory\n\n")
        
        for category, models in MODELS_CONFIG.items():
            f.write(f"## {category.upper()} Models\n")
            for model_id, local_path in models.items():
                model_path = Path(local_path)
                exists = "✅" if model_path.exists() else "❌"
                f.write(f"- {exists} {model_id} → {local_path}\n")
            f.write("\n")
    
    logger.info(f"Model inventory created: {inventory_path}")

if __name__ == "__main__":
    main() 
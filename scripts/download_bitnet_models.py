#!/usr/bin/env python
"""
Script to download BitNet models from Hugging Face Hub
"""

import os
import argparse
from transformers import AutoModel, AutoTokenizer

def download_model(model_name, output_dir):
    """Download model and tokenizer from Hugging Face Hub"""
    print(f"Downloading {model_name} to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=output_dir)
    
    # Download model
    print("Downloading model (this may take a while)...")
    model = AutoModel.from_pretrained(model_name, cache_dir=output_dir)
    
    print(f"Model and tokenizer downloaded successfully to {output_dir}")
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Download BitNet models from Hugging Face Hub")
    parser.add_argument("--model", type=str, default="microsoft/BitNet", help="Model name on Hugging Face Hub")
    parser.add_argument("--output", type=str, default="models/bitnet/models/bitnet", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        model, tokenizer = download_model(args.model, args.output)
        print("Success! Model ready for use.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
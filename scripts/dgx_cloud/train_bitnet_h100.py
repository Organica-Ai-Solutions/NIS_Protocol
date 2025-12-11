#!/usr/bin/env python3
"""
BitNet Fine-tuning on NVIDIA DGX Cloud H100

Primary training script for $100K DGX Cloud grant.
Optimized for H100 GPU with mixed precision and efficient data loading.

Usage:
    python train_bitnet_h100.py --epochs 30 --batch-size 32
    python train_bitnet_h100.py --resume --checkpoint latest
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Training configuration
DEFAULT_CONFIG = {
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Base model
    "output_dir": "models/bitnet/dgx_trained",
    "data_dir": "data/dgx_training",
    
    # H100 optimized settings
    "batch_size": 32,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "num_epochs": 30,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_seq_length": 512,
    
    # Mixed precision for H100
    "fp16": False,
    "bf16": True,  # H100 excels at BF16
    
    # LoRA configuration
    "use_lora": True,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # Checkpointing
    "save_steps": 500,
    "eval_steps": 250,
    "logging_steps": 50,
    "save_total_limit": 5,
    
    # Performance
    "dataloader_num_workers": 8,
    "gradient_checkpointing": True,
}


def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and specs"""
    gpu_info = {
        "available": False,
        "device_count": 0,
        "devices": [],
        "is_h100": False
    }
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device_count"] = torch.cuda.device_count()
            
            for i in range(gpu_info["device_count"]):
                props = torch.cuda.get_device_properties(i)
                device_info = {
                    "index": i,
                    "name": props.name,
                    "memory_gb": props.total_memory / (1024**3),
                    "compute_capability": f"{props.major}.{props.minor}"
                }
                gpu_info["devices"].append(device_info)
                
                # Check for H100
                if "H100" in props.name or props.major >= 9:
                    gpu_info["is_h100"] = True
            
            logger.info(f"🎮 Found {gpu_info['device_count']} GPU(s)")
            for dev in gpu_info["devices"]:
                logger.info(f"   {dev['name']}: {dev['memory_gb']:.1f}GB")
        else:
            logger.warning("⚠️ No GPU available - training will be slow")
            
    except ImportError:
        logger.error("❌ PyTorch not installed")
    
    return gpu_info


def load_training_data(data_dir: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load training data from JSONL files"""
    data_path = Path(data_dir)
    all_data = []
    
    # Load all JSONL files
    for jsonl_file in data_path.rglob("*.jsonl"):
        logger.info(f"📂 Loading {jsonl_file}")
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))
    
    logger.info(f"📊 Loaded {len(all_data)} training samples")
    
    if max_samples and len(all_data) > max_samples:
        import random
        all_data = random.sample(all_data, max_samples)
        logger.info(f"📊 Sampled {max_samples} for training")
    
    return all_data


def create_training_dataset(data: List[Dict], tokenizer, max_length: int = 512):
    """Create HuggingFace dataset from training data"""
    try:
        from datasets import Dataset
        import torch
        
        def format_example(example):
            instruction = example.get("instruction", "")
            output = example.get("output", "")
            return {"text": f"### Instruction:\n{instruction}\n\n### Response:\n{output}"}
        
        formatted_data = [format_example(ex) for ex in data]
        dataset = Dataset.from_list(formatted_data)
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            num_proc=4
        )
        
        return tokenized_dataset
        
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        raise


def train(config: Dict[str, Any], resume_from: Optional[str] = None):
    """Main training function"""
    try:
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            TrainingArguments, Trainer, DataCollatorForLanguageModeling
        )
        from peft import LoraConfig, get_peft_model, TaskType
        
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Install with: pip install transformers peft datasets accelerate")
        return None
    
    # Check GPU
    gpu_info = check_gpu_availability()
    if not gpu_info["available"]:
        logger.warning("⚠️ No GPU - this will be very slow!")
    
    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    logger.info(f"📥 Loading tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info(f"📥 Loading model: {config['model_name']}")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16 if config["bf16"] else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Apply LoRA
    if config["use_lora"]:
        logger.info("🔧 Applying LoRA configuration")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["lora_target_modules"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Enable gradient checkpointing for memory efficiency
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    
    # Load training data
    logger.info(f"📂 Loading training data from {config['data_dir']}")
    training_data = load_training_data(config["data_dir"])
    
    if not training_data:
        logger.error("❌ No training data found!")
        logger.error(f"   Run prepare_training_data.py first")
        return None
    
    # Create dataset
    train_dataset = create_training_dataset(
        training_data, 
        tokenizer, 
        config["max_seq_length"]
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        fp16=config["fp16"],
        bf16=config["bf16"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        dataloader_num_workers=config["dataloader_num_workers"],
        report_to="none",  # Disable wandb etc
        optim="adamw_torch_fused" if gpu_info["available"] else "adamw_torch",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Resume from checkpoint if specified
    if resume_from:
        logger.info(f"📂 Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        logger.info("🚀 Starting training...")
        trainer.train()
    
    # Save final model
    logger.info("💾 Saving final model...")
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    
    # Save training config
    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ Training complete! Model saved to {output_dir / 'final'}")
    
    return {
        "status": "complete",
        "output_dir": str(output_dir / "final"),
        "samples_trained": len(training_data),
        "epochs": config["num_epochs"]
    }


def main():
    parser = argparse.ArgumentParser(description="BitNet H100 Training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--dry-run", action="store_true", help="Check setup without training")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 NIS Protocol - BitNet H100 Training")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Update config with CLI args
    config = DEFAULT_CONFIG.copy()
    config["num_epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.lr
    
    # Check GPU
    gpu_info = check_gpu_availability()
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Checking setup...")
        print(f"   GPU available: {gpu_info['available']}")
        print(f"   H100 detected: {gpu_info['is_h100']}")
        print(f"   Config: {json.dumps(config, indent=2)}")
        
        # Check data
        data = load_training_data(config["data_dir"])
        print(f"   Training samples: {len(data)}")
        return
    
    # Run training
    resume_from = args.checkpoint if args.resume else None
    result = train(config, resume_from)
    
    if result:
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE")
        print("=" * 60)
        print(f"   Output: {result['output_dir']}")
        print(f"   Samples: {result['samples_trained']}")
        print(f"   Epochs: {result['epochs']}")


if __name__ == "__main__":
    main()

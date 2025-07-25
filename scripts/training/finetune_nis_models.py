#!/usr/bin/env python3
"""
NIS Protocol Model Fine-tuning Script
Fine-tunes BitNet and Kimi K2 models for NIS Protocol specific tasks
"""

import os
import sys
import json
import torch
import wandb
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
    Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NISTrainingConfig:
    """Configuration for NIS Protocol fine-tuning"""
    model_name: str
    model_type: str  # "bitnet" or "kimi" or "base"
    output_dir: str = "models/fine-tuned"
    dataset_path: str = "datasets/nis-training"
    
    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # NIS-specific settings
    max_length: int = 2048
    use_physics_validation: bool = True
    agent_coordination_weight: float = 1.0
    consciousness_monitoring_weight: float = 0.8

class NISDatasetProcessor:
    """Process NIS Protocol training data"""
    
    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def create_nis_training_data(self):
        """Create NIS-specific training datasets"""
        
        # Agent coordination examples
        agent_coordination_data = [
            {
                "input": "Laplace transform of signal shows frequency spike at 10Hz. What should KAN reasoning layer conclude?",
                "output": "Based on the 10Hz frequency spike, the KAN layer should extract a periodic function f(x) = A*sin(2π*10*t + φ) and validate this pattern matches known behavioral cycles. Recommend increasing agent attention to temporal patterns at this frequency."
            },
            {
                "input": "PINN physics validation failed for proposed robot movement. Error: violates conservation of momentum.",
                "output": "Physics constraint violation detected. Adjusting movement trajectory to satisfy momentum conservation: p_initial = p_final. Recalculating agent coordination parameters with physics-informed constraints."
            },
            {
                "input": "Consciousness monitoring shows elevated uncertainty in decision agent. Confidence: 0.3",
                "output": "Low confidence detected. Initiating meta-cognitive review protocol. Requesting additional sensory input and consulting memory agent for similar past experiences. Temporarily reducing decision authority until confidence exceeds 0.7 threshold."
            }
        ]
        
        # Physics-informed reasoning examples
        physics_reasoning_data = [
            {
                "input": "System energy before action: 100J. Proposed action requires 150J. Available energy: 120J.",
                "output": "Energy constraint violation detected. Available energy (120J) insufficient for proposed action (150J). Suggesting alternative approach: break action into two phases of 60J each, allowing energy regeneration between phases."
            },
            {
                "input": "Multi-agent system coordination: 3 agents need to reach target simultaneously. Current positions: A(0,0), B(5,3), C(2,7). Target: (10,10).",
                "output": "Calculating synchronized trajectories using physics-informed path planning. Agent A: direct path (10,10), time: 14.14s. Agent B: path via (7,7), time: 14.14s. Agent C: path via (6,8.5), time: 14.14s. All agents arrive simultaneously respecting momentum constraints."
            }
        ]
        
        # Temporal intelligence examples  
        temporal_intelligence_data = [
            {
                "input": "Time series shows pattern: t=0:5, t=1:8, t=2:11, t=3:14. Predict t=4 and explain reasoning.",
                "output": "Pattern analysis: linear growth with slope=3, y-intercept=5. Function: f(t) = 3t + 5. Prediction: t=4 → f(4) = 3(4) + 5 = 17. Confidence: 0.95 based on perfect linear correlation (R²=1.0). Recommend continued monitoring for pattern stability."
            }
        ]
        
        # Combine all datasets
        all_data = agent_coordination_data + physics_reasoning_data + temporal_intelligence_data
        
        return Dataset.from_list(all_data)
    
    def tokenize_function(self, examples):
        """Tokenize the training examples"""
        # Format as instruction-following pairs
        texts = []
        for input_text, output_text in zip(examples["input"], examples["output"]):
            formatted_text = f"### Instruction:\n{input_text}\n\n### Response:\n{output_text}<|endoftext|>"
            texts.append(formatted_text)
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized

class NISModelTrainer:
    """Fine-tune models for NIS Protocol"""
    
    def __init__(self, config: NISTrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
    
    def setup_model_and_tokenizer(self):
        """Load model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Apply LoRA if specified
        if self.config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # Common attention modules
            )
            self.model = get_peft_model(self.model, lora_config)
            logger.info("Applied LoRA configuration")
    
    def prepare_datasets(self):
        """Prepare training and validation datasets"""
        processor = NISDatasetProcessor(self.tokenizer, self.config.max_length)
        
        # Create NIS-specific dataset
        dataset = processor.create_nis_training_data()
        
        # Split into train/validation
        train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
        self.train_dataset = train_test_split["train"]
        self.eval_dataset = train_test_split["test"]
        
        # Tokenize datasets
        self.train_dataset = self.train_dataset.map(
            processor.tokenize_function, 
            batched=True,
            remove_columns=["input", "output"]
        )
        self.eval_dataset = self.eval_dataset.map(
            processor.tokenize_function, 
            batched=True,
            remove_columns=["input", "output"]
        )
        
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.eval_dataset)}")
    
    def setup_training(self):
        """Setup training arguments and trainer"""
        
        # Create output directory
        output_dir = Path(self.config.output_dir) / f"{self.config.model_type}_nis_finetuned"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if wandb.api.api_key else None,
            run_name=f"nis-{self.config.model_type}-finetune",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8 if torch.cuda.is_available() else None,
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
    
    def train(self):
        """Execute the training process"""
        logger.info("Starting training...")
        
        # Initialize wandb if available
        if wandb.api.api_key:
            wandb.init(
                project="nis-protocol-finetuning",
                name=f"nis-{self.config.model_type}-finetune",
                config=self.config.__dict__
            )
        
        # Train the model
        train_result = self.trainer.train()
        
        # Save the final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.trainer.args.output_dir)
        
        # Log training results
        logger.info(f"Training completed!")
        logger.info(f"Final training loss: {train_result.training_loss}")
        
        # Save training metrics
        metrics_path = Path(self.trainer.args.output_dir) / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(train_result.metrics, f, indent=2)
        
        return train_result

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune models for NIS Protocol")
    parser.add_argument("--model_name", required=True, help="Model to fine-tune")
    parser.add_argument("--model_type", choices=["bitnet", "kimi", "base"], required=True)
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA fine-tuning")
    
    args = parser.parse_args()
    
    # Create training configuration
    config = NISTrainingConfig(
        model_name=args.model_name,
        model_type=args.model_type,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_lora=args.use_lora
    )
    
    # Initialize trainer
    trainer = NISModelTrainer(config)
    
    # Setup and train
    trainer.setup_model_and_tokenizer()
    trainer.prepare_datasets()
    trainer.setup_training()
    trainer.train()
    
    logger.info("Fine-tuning complete!")

if __name__ == "__main__":
    main() 
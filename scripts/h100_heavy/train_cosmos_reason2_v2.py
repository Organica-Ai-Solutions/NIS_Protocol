#!/usr/bin/env python3
"""
Cosmos Reason2 8B — Fine-Tune v2 (QLoRA SFT)
=============================================
Continues from the v1 LoRA adapter with improved data:
  - Our robot episodes (xArm/ALOHA/PushT) converted to LLaVA format
  - LLaVA-Instruct-150K for general vision-language grounding
  - Physics reasoning chain-of-thought data

GPU: 2 (CUDA_VISIBLE_DEVICES=2)
VRAM: ~45GB with QLoRA 4-bit
ETA: ~8-12 hours for 3 epochs

Based on official Cosmos Reason2 TRL SFT example:
  https://github.com/nvidia-cosmos/cosmos-reason2/blob/main/examples/notebooks/trl_sft.ipynb
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from datasets import Dataset as HFDataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Qwen3VLForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig

# ─── Config ───
MODEL_PATH = "/data/organica-ai/models/cosmos/cosmos-reason2-8b"
PREV_ADAPTER = "/data/organica-ai/models/cosmos/nis-cosmos-reason2-finetuned/final"
OUTPUT_DIR = "/data/organica-ai/models/cosmos/nis-cosmos-reason2-v2"
LOG_DIR = "/data/organica-ai/logs"

# Dataset paths
ROBOT_DATA = "/data/organica-ai/datasets/reason2_sft/train_annotations.json"
ROBOT_VAL = "/data/organica-ai/datasets/reason2_sft/val_annotations.json"
ROBOT_IMAGES = "/data/organica-ai/datasets/reason2_sft"
LLAVA_DATA = "/data/organica-ai/datasets/llava_instruct/detail_23k.json"
LLAVA_IMAGES = "/data/organica-ai/datasets/llava_instruct/coco_images/train2017"

# Training hyperparams
BATCH_SIZE = 2
GRAD_ACCUM = 8  # effective batch = 16
NUM_EPOCHS = 3
LR = 5e-5  # lower LR for continued training
WARMUP_RATIO = 0.05
MAX_SEQ_LEN = 2048
SAVE_STEPS = 500
EVAL_STEPS = 500
LOGGING_STEPS = 10

# ─── Logging ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "cosmos_reason2_v2.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_combined_dataset(robot_path, llava_path=None, processor=None):
    """Load robot + LLaVA data into a HuggingFace Dataset with pre-formatted text."""
    texts = []

    # Load robot data
    if robot_path and os.path.exists(robot_path):
        with open(robot_path) as f:
            robot_data = json.load(f)
        for item in robot_data:
            convs = item.get("conversations", [])
            if len(convs) >= 2:
                q = convs[0]["value"].replace("<image>\n", "")
                a = convs[1]["value"]
                texts.append(format_text(q, a, processor))
        logger.info(f"Loaded {len(texts)} robot samples from {robot_path}")

    # Load LLaVA data
    prev = len(texts)
    if llava_path and os.path.exists(llava_path):
        with open(llava_path) as f:
            llava_data = json.load(f)
        for item in llava_data:
            convs = item.get("conversations", [])
            if len(convs) >= 2:
                q = convs[0]["value"].replace("<image>\n", "")
                a = convs[1]["value"]
                texts.append(format_text(q, a, processor))
        logger.info(f"Loaded {len(texts) - prev} LLaVA samples from {llava_path}")

    logger.info(f"Total combined dataset: {len(texts)} samples")
    return HFDataset.from_dict({"text": texts})


def format_text(question, answer, processor=None):
    """Format a QA pair into text for SFT."""
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    if processor is not None:
        try:
            return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    return f"User: {question}\nAssistant: {answer}"


def main():
    logger.info("=" * 70)
    logger.info("Cosmos Reason2 8B — Fine-Tune v2 (QLoRA SFT)")
    logger.info("=" * 70)
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ─── 4-bit quantization config ───
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ─── Load model ───
    logger.info(f"Loading base model from {MODEL_PATH}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # ─── Load processor ───
    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ─── Check if we should continue from v1 adapter ───
    if os.path.exists(PREV_ADAPTER):
        logger.info(f"Loading previous LoRA adapter from {PREV_ADAPTER}...")
        try:
            model = PeftModel.from_pretrained(model, PREV_ADAPTER, is_trainable=True)
            logger.info("Successfully loaded v1 adapter — continuing training")
        except Exception as e:
            logger.warning(f"Could not load v1 adapter ({e}), creating fresh LoRA")
            lora_config = LoraConfig(
                r=64,
                lora_alpha=128,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
    else:
        logger.info("No previous adapter found — creating fresh LoRA")
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ─── Load datasets ───
    logger.info("Loading datasets...")
    train_dataset = load_combined_dataset(
        robot_path=ROBOT_DATA,
        llava_path=LLAVA_DATA,
        processor=processor,
    )

    val_dataset = load_combined_dataset(
        robot_path=ROBOT_VAL,
        llava_path=None,
        processor=processor,
    )

    if len(train_dataset) == 0:
        logger.error("No training data found! Check dataset paths.")
        logger.info("Will wait for data conversion to complete and retry...")
        for i in range(60):
            time.sleep(30)
            if os.path.exists(ROBOT_DATA):
                train_dataset = load_combined_dataset(
                    robot_path=ROBOT_DATA,
                    llava_path=LLAVA_DATA,
                    processor=processor,
                )
                if len(train_dataset) > 0:
                    logger.info(f"Data available! {len(train_dataset)} samples")
                    break
            logger.info(f"Waiting for data... ({i + 1}/60)")

    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # ─── Training config ───
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        bf16=True,
        tf32=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=MAX_SEQ_LEN,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="none",
        remove_unused_columns=False,
    )

    # ─── Trainer ───
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        processing_class=processor.tokenizer,
    )

    # ─── Train ───
    logger.info("Starting training...")
    start_time = time.time()

    try:
        train_result = trainer.train()
        elapsed = time.time() - start_time

        logger.info(f"Training complete in {elapsed / 3600:.1f} hours")
        logger.info(f"Train loss: {train_result.training_loss:.4f}")

        # Save final model
        logger.info(f"Saving final model to {OUTPUT_DIR}/final")
        trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
        processor.save_pretrained(os.path.join(OUTPUT_DIR, "processor"))

        # Save metrics
        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", elapsed),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "total_time_hours": elapsed / 3600,
            "epochs": NUM_EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE * GRAD_ACCUM,
            "lora_r": 64,
            "continued_from_v1": os.path.exists(PREV_ADAPTER),
        }
        with open(os.path.join(OUTPUT_DIR, "training_metrics_v2.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics saved to {OUTPUT_DIR}/training_metrics_v2.json")
        logger.info("=" * 70)
        logger.info("COSMOS REASON2 V2 FINE-TUNE COMPLETE")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

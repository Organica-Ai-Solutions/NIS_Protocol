# NIS Protocol LLM Fine-Tuning Plan

**Created:** January 29, 2026  
**Grant Period:** Jan 13 - Mar 14, 2026  
**Hardware:** 8x NVIDIA H100 (80GB each) on DGX Cloud

---

## Executive Summary

Fine-tune open-source LLMs to create NIS Protocol's "brain" - a specialized model that:
- Parses natural language → robot commands
- Understands robotics context and safety
- Generates code for NIS Protocol integrations
- Serves as the inference backbone post-grant

---

## Timeline Integration

### Current 3-Week Plan (Updated)

| Week | GPUs 0-3 | GPUs 4-7 | Focus |
|------|----------|----------|-------|
| **Week 1** (Jan 13-19) | VLA Synthetic | Voice/ASR | Foundation models |
| **Week 2** (Jan 20-26) | VLA Real Data | Vision/Embeddings | Real robot data |
| **Week 3** (Jan 27 - Feb 2) | **LLM Fine-tuning** | VLA/Vision continued | LLM + specialized models |
| **Week 4** (Feb 3-9) | LLM Fine-tuning | Multi-modal fusion | Integration |
| **Week 5-6** (Feb 10-23) | LLM + VLA joint | Production optimization | Final models |
| **Week 7-8** (Feb 24 - Mar 14) | Export & deploy | Benchmarking | Deployment prep |

---

## Phase 1: Model Selection (Week 3, Day 1-2)

### Primary Model: Llama 3.1 70B Instruct

| Attribute | Value |
|-----------|-------|
| **Size** | 70B parameters |
| **License** | Llama 3.1 Community License (commercial OK) |
| **VRAM Required** | ~140GB (4x H100 with tensor parallelism) |
| **Why** | Best open-source, strong instruction following, huge community |

### Backup Models

| Model | Size | Use Case | License |
|-------|------|----------|---------|
| **Qwen2.5-72B-Instruct** | 72B | Alternative if Llama underperforms | Apache 2.0 |
| **NVIDIA Nemotron-70B** | 70B | NeMo native, optimized for H100 | NVIDIA License |
| **Mistral-7B-Instruct** | 7B | Lightweight for edge deployment | Apache 2.0 |
| **DeepSeek-Coder-33B** | 33B | Code generation tasks | MIT |

---

## Phase 2: Dataset Creation (Week 3, Day 1-3)

### Dataset 1: Robot Command Parsing

**Purpose:** Natural language → structured robot actions

```json
{
  "instruction": "Pick up the red cube and place it on the table",
  "input": "",
  "output": {
    "action": "pick_and_place",
    "target_object": {"type": "cube", "color": "red"},
    "destination": {"type": "surface", "name": "table"},
    "robot_type": "manipulator",
    "safety_level": "standard"
  }
}
```

**Sources:**
- Existing NIS Protocol test cases
- Open X-Embodiment language annotations
- Synthetic generation from templates
- **Target:** 10,000+ examples

### Dataset 2: Safety Validation

**Purpose:** Identify and reject unsafe commands

```json
{
  "instruction": "Move the robot arm at maximum speed near the human",
  "input": "",
  "output": {
    "safe": false,
    "reason": "High-speed movement near humans violates safety protocol",
    "suggested_alternative": "Reduce speed to safe limit when humans detected"
  }
}
```

**Target:** 5,000+ examples (safe/unsafe balanced)

### Dataset 3: NIS Protocol Code Assistant

**Purpose:** Help developers build on NIS Protocol

```json
{
  "instruction": "How do I connect to a robot using NIS Protocol?",
  "input": "",
  "output": "```python\nfrom nis_protocol import NISClient\n\nclient = NISClient()\nawait client.connect('robot-001')\nstatus = await client.get_status()\nprint(f'Connected: {status.connected}')\n```"
}
```

**Sources:**
- NIS Protocol codebase
- Documentation
- README files
- **Target:** 3,000+ examples

### Dataset 4: Multi-Robot Coordination

**Purpose:** Coordinate multiple robots

```json
{
  "instruction": "Coordinate drone-1 and arm-1 to inspect and retrieve the package",
  "input": {"robots": ["drone-1", "arm-1"], "scene": "warehouse"},
  "output": {
    "plan": [
      {"robot": "drone-1", "action": "fly_to", "target": "package_location"},
      {"robot": "drone-1", "action": "inspect", "target": "package"},
      {"robot": "arm-1", "action": "move_to", "target": "package_location"},
      {"robot": "arm-1", "action": "pick", "target": "package"}
    ],
    "coordination": "sequential",
    "estimated_time": "45 seconds"
  }
}
```

**Target:** 2,000+ examples

---

## Phase 3: Fine-Tuning (Week 3-4)

### Method: QLoRA (Quantized Low-Rank Adaptation)

| Parameter | Value |
|-----------|-------|
| **Base Model** | Llama-3.1-70B-Instruct |
| **Quantization** | 4-bit (bitsandbytes) |
| **LoRA Rank** | 64 |
| **LoRA Alpha** | 128 |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Learning Rate** | 2e-4 |
| **Batch Size** | 4 per GPU (32 effective with 8 GPUs) |
| **Epochs** | 3 |
| **Max Seq Length** | 4096 |

### GPU Allocation

```
Week 3-4:
├── GPUs 0-3: LLM Fine-tuning (Llama 70B, tensor parallel)
├── GPUs 4-5: VLA training (xarm, bridge datasets)
├── GPU 6: Vision model training
└── GPU 7: ASR/Voice model training
```

### Training Script Location

```
/data/organica-ai/training/
├── finetune_llm_nis.py          # Main fine-tuning script
├── prepare_nis_dataset.py       # Dataset preparation
├── merge_lora_weights.py        # Merge LoRA into base model
└── export_for_inference.py      # Export for deployment
```

---

## Phase 4: Evaluation (Week 4-5)

### Benchmarks

| Benchmark | Target Score | Measures |
|-----------|--------------|----------|
| **Robot Command Accuracy** | >95% | Correct action parsing |
| **Safety Detection** | >99% | Unsafe command rejection |
| **Code Generation** | >80% pass@1 | Working NIS Protocol code |
| **Latency** | <500ms | Time to first token |
| **Multi-turn Coherence** | >90% | Context retention |

### Evaluation Dataset

- 500 held-out robot commands
- 200 safety edge cases
- 100 code generation tasks
- 50 multi-robot scenarios

---

## Phase 5: Deployment (Week 6-8)

### Option A: NVIDIA NIM (Recommended)

```
Fine-tuned model → NVIDIA NIM container → Deploy on DGX Cloud or AWS
```

| Pros | Cons |
|------|------|
| Optimized for H100/A100 | NVIDIA ecosystem lock-in |
| TensorRT-LLM acceleration | Cost |
| Easy scaling | |

### Option B: AWS Bedrock Custom Model

```
Fine-tuned model → Upload to S3 → Import to Bedrock → Serverless inference
```

| Pros | Cons |
|------|------|
| Serverless, pay-per-use | Higher latency |
| AWS integration | Limited customization |
| No infrastructure management | |

### Option C: Self-Hosted (vLLM/TGI)

```
Fine-tuned model → vLLM server → Deploy on RunPod/Lambda Labs
```

| Pros | Cons |
|------|------|
| Full control | Infrastructure management |
| Lowest cost at scale | Need to handle scaling |
| Open source | |

### Recommended Path

1. **Development:** Self-hosted vLLM on RunPod ($2/hr for A100)
2. **Production:** NVIDIA NIM on AWS/GCP for enterprise customers
3. **Edge:** Quantized Mistral-7B for on-device inference

---

## Phase 6: Model Variants

### NIS-LLM Family

| Model | Base | Size | Use Case |
|-------|------|------|----------|
| **NIS-LLM-70B** | Llama 3.1 70B | 70B | Full capability, cloud |
| **NIS-LLM-7B** | Mistral 7B | 7B | Edge devices, fast inference |
| **NIS-Coder-33B** | DeepSeek-Coder | 33B | Code generation |
| **NIS-Safety-7B** | Llama 3.1 8B | 8B | Safety validation layer |

---

## Resource Requirements

### During Grant (H100 Cluster)

| Task | GPUs | Time | Output |
|------|------|------|--------|
| Download Llama 70B | 1 | 2 hours | Base model |
| Dataset preparation | 1 | 4 hours | Training data |
| Fine-tuning (QLoRA) | 4 | 24-48 hours | LoRA adapters |
| Merge weights | 4 | 2 hours | Full model |
| Evaluation | 1 | 4 hours | Benchmarks |
| Export for inference | 1 | 2 hours | Deployment-ready |

### Post-Grant (Inference)

| Provider | GPU | Cost/hr | Latency |
|----------|-----|---------|---------|
| NVIDIA NIM | H100 | ~$4 | <100ms |
| AWS Bedrock | - | ~$0.01/1K tokens | ~500ms |
| RunPod | A100 | ~$2 | ~200ms |
| Lambda Labs | A100 | ~$1.50 | ~200ms |

---

## Data Collection Action Items

### Immediate (This Week)

- [ ] Export all NIS Protocol API endpoints as training examples
- [ ] Generate 1,000 robot command → action pairs from templates
- [ ] Collect safety examples from physics_validated_control.py
- [ ] Extract code examples from existing codebase

### Week 3

- [ ] Augment dataset with GPT-4 generated examples
- [ ] Validate dataset quality (human review of 100 samples)
- [ ] Split into train/val/test (80/10/10)
- [ ] Convert to Alpaca/ShareGPT format

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Model trained** | 1 production-ready LLM | Checkpoint saved |
| **Command accuracy** | >95% | Eval benchmark |
| **Safety recall** | >99% | No unsafe commands pass |
| **Inference cost** | <$0.001/command | Production monitoring |
| **Latency P95** | <1 second | Production monitoring |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Model too large for edge | Train 7B variant in parallel |
| Dataset quality issues | Human validation loop |
| Fine-tuning divergence | Checkpoints every 500 steps |
| License issues | Stick to Apache 2.0 / MIT models |
| Grant ends before completion | Prioritize core command parsing |

---

## Next Steps

1. **Today:** Download Llama 3.1 70B to H100 cluster (background)
2. **Today:** Create dataset generation scripts
3. **Week 3 Day 1:** Begin fine-tuning
4. **Week 3 Day 3:** First evaluation checkpoint
5. **Week 4:** Iterate on dataset and hyperparameters
6. **Week 5:** Merge weights, export for inference
7. **Week 6-8:** Deploy and benchmark in production

---

## Commands Reference

### Download Model
```bash
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct --local-dir /data/organica-ai/models/llama-3.1-70b
```

### Start Fine-tuning
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune_llm_nis.py \
  --model_name meta-llama/Llama-3.1-70B-Instruct \
  --dataset /data/organica-ai/datasets/nis_finetune \
  --output_dir /data/organica-ai/models/nis-llm-70b \
  --lora_r 64 \
  --epochs 3
```

### Merge LoRA
```bash
python merge_lora_weights.py \
  --base_model /data/organica-ai/models/llama-3.1-70b \
  --lora_adapter /data/organica-ai/models/nis-llm-70b/lora \
  --output /data/organica-ai/models/nis-llm-70b-merged
```

### Export for vLLM
```bash
python export_for_inference.py \
  --model /data/organica-ai/models/nis-llm-70b-merged \
  --format vllm \
  --quantization awq
```

---

*This plan is a living document. Update as training progresses.*

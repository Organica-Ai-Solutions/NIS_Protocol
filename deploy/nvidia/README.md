# 🚀 NVIDIA Cloud Deployment Guide for NIS Protocol

## Overview

This guide covers deploying NIS Protocol to NVIDIA's cloud infrastructure:

| Platform | Type | Status | Best For |
|----------|------|--------|----------|
| **NVCF** | Serverless | ⏳ Pending approval | Pay-per-inference |
| **DGX Cloud Lepton** | Multi-cloud marketplace | ⏳ Pending approval | Flexible GPU access |
| **NVIDIA NIM API** | Hosted API | ✅ Active | Immediate use |

---

## 📁 Files in This Directory

```
deploy/nvidia/
├── Dockerfile.nvcf       # NVCF-optimized Dockerfile
├── nvcf-deploy.sh        # NVCF deployment script
├── lepton-deploy.sh      # Lepton deployment script
├── lepton-config.yaml    # Lepton deployment config (generated)
├── lepton-secrets.yaml   # Secrets template (generated)
└── README.md             # This file
```

---

## 🔑 Prerequisites

### Required API Keys

```bash
# NVIDIA API Key (for NIM models)
export NVIDIA_API_KEY="nvapi-xxxxx"

# NGC API Key (for container registry)
export NGC_API_KEY="your-ngc-key"
```

### Install NGC CLI

```bash
# Download from: https://ngc.nvidia.com/setup/installers/cli
# Linux/Mac:
wget -O ngc https://ngc.nvidia.com/downloads/ngccli_linux.zip
unzip ngccli_linux.zip
chmod +x ngc
./ngc config set
```

---

## 🚀 Option 1: NVIDIA Cloud Functions (NVCF)

Serverless GPU inference - pay only for what you use.

### Deploy Steps

```bash
# 1. Set environment variables
export NGC_API_KEY="your-ngc-key"
export NGC_ORG="organica-ai"
export IMAGE_TAG="v4.0"

# 2. Run deployment script
chmod +x deploy/nvidia/nvcf-deploy.sh
./deploy/nvidia/nvcf-deploy.sh

# 3. After approval, create function
ngc cloud-function function create -f /tmp/nvcf-function.json

# 4. Deploy
ngc cloud-function function deploy nis-protocol-inference
```

### Pricing (Estimated)

| GPU | $/hour | Best For |
|-----|--------|----------|
| T4 | ~$0.50 | Development |
| L40S | ~$1.50 | Production |
| A100 | ~$3.00 | High throughput |

---

## 🌐 Option 2: DGX Cloud Lepton

Multi-cloud GPU marketplace with flexible pricing.

### Deploy Steps

```bash
# 1. Run setup script
chmod +x deploy/nvidia/lepton-deploy.sh
./deploy/nvidia/lepton-deploy.sh

# 2. After approval, login to Lepton portal
# https://developer.nvidia.com/dgx-cloud/get-lepton

# 3. Create workspace and push image

# 4. Deploy using config
lepton deploy -f deploy/nvidia/lepton-config.yaml
```

### Cloud Providers on Lepton

- **CoreWeave** - US West
- **Lambda** - US East
- **Nebius** - Europe
- **Crusoe** - Sustainable compute
- **AWS/Azure** - Coming soon

---

## ✅ Option 3: NVIDIA NIM API (Available Now!)

Use NVIDIA's hosted models immediately via API.

### Quick Test

```bash
export NVIDIA_API_KEY="nvapi-xxxxx"

curl -X POST "https://integrate.api.nvidia.com/v1/chat/completions" \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.3-70b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Available Models

- `meta/llama-3.3-70b-instruct` (fastest)
- `nvidia/llama-3.1-nemotron-ultra-253b-v1`
- `nvidia/nemotron-4-340b-instruct`
- And 50+ more at build.nvidia.com

---

## 🔄 Dual-Cloud Strategy (AWS + NVIDIA)

```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   AWS (via Cloudelligent)          NVIDIA Cloud             │
│   ├── Primary production           ├── GPU inference        │
│   ├── Database (RDS)               ├── Model hosting        │
│   ├── Storage (S3)                 ├── Training workloads   │
│   └── Load balancing               └── Burst capacity       │
│                                                              │
│   Traffic Flow:                                              │
│   User → AWS ALB → NIS Backend → NVIDIA NIM API             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Cost Comparison

| Scenario | AWS Only | NVIDIA Only | Hybrid |
|----------|----------|-------------|--------|
| Light usage (100 req/day) | $150/mo | $50/mo | $100/mo |
| Medium (1K req/day) | $400/mo | $200/mo | $300/mo |
| Heavy (10K req/day) | $1,500/mo | $800/mo | $1,000/mo |

*Hybrid uses AWS for backend + NVIDIA for GPU inference*

---

## 🆘 Support

- **NVIDIA Developer**: https://developer.nvidia.com/
- **NGC Support**: https://ngc.nvidia.com/support
- **NIS Protocol Issues**: https://github.com/Organica-Ai-Solutions/NIS_Protocol/issues

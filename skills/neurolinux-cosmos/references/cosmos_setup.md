# Cosmos Stack Setup on H100

## Team — Organica AI Solutions

| Name | Role |
|------|------|
| Diego Torres | Founder / Lead Engineer — NIS Protocol, H100 stack, xArm, NeuroLinux OS |
| Camrin Neiss | Co-founder / Marketing + Frontend Dev — demo UI, pitch materials, React frontend |



## Requirements

- NVIDIA H100 (80 GB VRAM recommended)
- Docker + NVIDIA Container Toolkit
- NVIDIA API key for NIM microservices

## Stack Services

| Service            | Port | Model                       |
|--------------------|------|-----------------------------|
| Cosmos Reason 2    | 8100 | cosmos-reason2-7b-instruct  |
| Cosmos Predict 2.5 | 8200 | cosmos-predict2-7b-video    |
| Cosmos Transfer 2.5| 8300 | cosmos-transfer2-7b         |

## Quick Start (Docker Compose)

```yaml
# docker-compose.cosmos.yml
services:
  cosmos-reason:
    image: nvcr.io/nvidia/cosmos-reason2:latest
    ports: ["8100:8100"]
    environment:
      - NVIDIA_API_KEY=${NVIDIA_API_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  cosmos-predict:
    image: nvcr.io/nvidia/cosmos-predict2:latest
    ports: ["8200:8200"]
    environment:
      - NVIDIA_API_KEY=${NVIDIA_API_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  cosmos-transfer:
    image: nvcr.io/nvidia/cosmos-transfer2:latest
    ports: ["8300:8300"]
    environment:
      - NVIDIA_API_KEY=${NVIDIA_API_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

```bash
docker compose -f docker-compose.cosmos.yml up -d
```

## NeuroLinux / NIS Protocol Config

```bash
# .env or environment
COSMOS_REASON_URL=http://<h100-host>:8100
COSMOS_PREDICT_URL=http://<h100-host>:8200
COSMOS_TRANSFER_URL=http://<h100-host>:8300
COSMOS_QUANTIZATION=4bit   # or 8bit / none
PRELOAD_COSMOS=false        # set true for lowest latency
```

## Health Check

```bash
curl http://<h100-host>:8100/health
curl http://<h100-host>:8200/health
curl http://<h100-host>:8300/health
```

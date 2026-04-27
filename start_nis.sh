#!/bin/bash
cd /home/nvidia
export OPENAI_API_KEY=local-qwen35
export OPENAI_API_BASE=http://localhost:8701/v1
export OPENAI_MODEL=qwen35-nis
export DEFAULT_LLM_PROVIDER=openai
export AWS_SECRETS_ENABLED=false
export H100_REASON_URL=http://localhost:8100
export H100_PREDICT_URL=http://localhost:8200
export H100_TRANSFER_URL=http://localhost:8300
export H100_VLA_URL=http://localhost:8500
export H100_SPEECH_URL=http://localhost:8600
export AGENT_URL=http://localhost:18085
export NIS_URL=http://localhost:8090
export PORT=8090
export LOG_LEVEL=INFO
export YOLO_MODEL=yolov8x.pt
export BITNET_PRELOAD=false
export BITNET_COLLECT_TRAINING=false
export DISABLE_RATE_LIMIT=true
exec /home/nvidia/organica-ai/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8090 --log-level info --workers 1

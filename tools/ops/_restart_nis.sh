#!/bin/bash
# Restart NIS on H100
pkill -f "uvicorn main:app" 2>/dev/null
pkill -f "uvicorn.*8090" 2>/dev/null
tmux kill-session -t nis_h100 2>/dev/null
sleep 2

cd /home/nvidia
set -a
source /home/nvidia/.env.h100
set +a

export YOLO_MODEL=yolov8x.pt

tmux new-session -d -s nis_h100 \
  "/home/nvidia/organica-ai/venv/bin/uvicorn main:app \
   --host 0.0.0.0 --port 8090 \
   --log-level info --workers 1 \
   2>&1 | tee /tmp/nis_h100.log"

echo "NIS restart launched in tmux nis_h100"
sleep 5
curl -s http://localhost:8090/health | head -c 200

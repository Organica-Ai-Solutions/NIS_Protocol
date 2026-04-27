#!/bin/bash
# Start NIS Protocol on H100 using the correct venv
cd /data/organica-ai/NIS_Protocol
nohup /data/organica-ai/NIS_Protocol/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 > /tmp/nis.log 2>&1 &
echo "NIS started with PID $!"

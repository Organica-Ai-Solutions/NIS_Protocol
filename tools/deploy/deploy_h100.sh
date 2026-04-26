#!/usr/bin/env bash
# deploy_h100.sh — Deploy NIS Protocol to H100 and launch
# Run from: bash "C:/Users/DiegoTorres/Desktop/NIS_Protocol/deploy_h100.sh"
set -e

H100="awesome-gpu-name"
REMOTE_DIR="/opt/nis_protocol"
VENV="/data/organica-ai/venv"
LOCAL_DIR="C:/Users/DiegoTorres/Desktop/NIS_Protocol"
PORT=8090

echo "=== [1/5] Packaging NIS Protocol (tar + scp) ==="
cd "$LOCAL_DIR"
tar --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='.env' \
    --exclude='logfile' \
    --exclude='logs' \
    --exclude='cache' \
    --exclude='*.egg-info' \
    --exclude='node_modules' \
    --exclude='.venv' \
    --exclude='*.log' \
    --exclude='*.jpg' \
    --exclude='*.png' \
    -czf /tmp/nis_protocol.tar.gz .
echo "Archive size: $(du -sh /tmp/nis_protocol.tar.gz | cut -f1)"

echo "Copying to H100..."
scp /tmp/nis_protocol.tar.gz "$H100:/tmp/nis_protocol.tar.gz"

echo "Extracting on H100..."
ssh "$H100" "bash -s" <<ENDSSH
mkdir -p $REMOTE_DIR
cd $REMOTE_DIR
tar -xzf /tmp/nis_protocol.tar.gz
echo "Extracted to $REMOTE_DIR"
ls -la | head -10
ENDSSH

echo ""
echo "=== [2/5] Installing Python dependencies ==="
ssh "$H100" "bash -s" <<ENDSSH
$VENV/bin/pip install --quiet \
  "fastapi>=0.109.0" "uvicorn[standard]>=0.27.0" "pydantic>=2.5.0,<3.0.0" \
  "aiohttp>=3.9.2" "httpx>=0.26.0" "aiofiles>=23.0.0" \
  "python-dotenv>=1.0.0" "requests>=2.31.0" \
  "beautifulsoup4>=4.12.0" "duckduckgo-search>=4.0.0" \
  "sqlalchemy>=2.0.0" "python-dateutil>=2.8.0" \
  "colorlog>=6.0.0" "tqdm>=4.65.0" \
  "websockets>=12.0" "python-multipart>=0.0.6" 2>&1 | tail -3
echo "deps installed"
ENDSSH

echo ""
echo "=== [3/5] Verifying H100 -> Pi connectivity ==="
ssh "$H100" 'curl -s --max-time 5 http://192.168.1.163:8085/health 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(\"Pi OK:\", d.get(\"status\",\"?\"), \"| xarm:\", d.get(\"xarm\",\"?\"))" 2>/dev/null || echo "WARNING: Pi not reachable from H100 — arm routes will fail"'

echo ""
echo "=== [4/5] Killing any old NIS instance on port $PORT ==="
ssh "$H100" "bash -s" <<ENDSSH
fuser -k ${PORT}/tcp 2>/dev/null && echo "Killed old process on :$PORT" || echo "Port $PORT was clear"
sleep 1
ENDSSH

echo ""
echo "=== [5/5] Launching NIS Protocol on H100 ==="
ssh "$H100" "bash -s" <<ENDSSH
cd $REMOTE_DIR
set -a; source .env.h100; set +a
mkdir -p logs cache data models

nohup $VENV/bin/uvicorn main:app \
  --host 0.0.0.0 \
  --port $PORT \
  --log-level info \
  --workers 1 \
  > /tmp/nis_h100.log 2>&1 &

NIS_PID=\$!
echo "NIS PID: \$NIS_PID"
echo \$NIS_PID > /tmp/nis_h100.pid

sleep 8

# Health check
echo ""
echo "--- Health check ---"
curl -s --max-time 10 http://localhost:$PORT/health | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print('status:', d.get('status','?'))
    for k,v in d.items():
        if k != 'status':
            print(f'  {k}: {v}')
except Exception as e:
    raw = sys.stdin.read()
    print('Parse error:', e)
    print('Raw:', raw[:300])
" 2>/dev/null || echo "Not ready yet — check: ssh $H100 tail -f /tmp/nis_h100.log"

echo ""
echo "--- Startup log tail ---"
tail -20 /tmp/nis_h100.log
ENDSSH

echo ""
echo "============================================================"
echo "NIS Protocol deployed at http://172.16.1.83:$PORT"
echo "Docs:  http://172.16.1.83:$PORT/docs"
echo "Logs:  ssh $H100 'tail -f /tmp/nis_h100.log'"
echo "Stop:  ssh $H100 'kill \$(cat /tmp/nis_h100.pid)'"
echo "============================================================"

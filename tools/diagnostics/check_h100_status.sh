#!/bin/bash
# H100 Status Check — training, agents, Cosmos GPUs
# Run: ssh awesome-gpu-name "bash /tmp/check_h100_status.sh"

echo "╔══════════════════════════════════════════════════════╗"
echo "  H100 Status Check — $(date '+%Y-%m-%d %H:%M:%S')"
echo "╚══════════════════════════════════════════════════════╝"

echo ""
echo "── GPU UTILIZATION ──────────────────────────────────────"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits | while IFS=, read idx name util mem_used mem_total; do
    pct=$(echo "$mem_used $mem_total" | awk '{printf "%.0f", $1/$2*100}')
    bar=$(printf '█%.0s' $(seq 1 $((util/10))))
    printf "  GPU%s  %3s%% util  %5sMiB/%5sMiB  [%-10s]\n" \
        "$idx" "$util" "$mem_used" "$mem_total" "$bar"
done

echo ""
echo "── TMUX TRAINING SESSIONS ───────────────────────────────"
tmux ls 2>/dev/null | while read line; do
    session=$(echo "$line" | cut -d: -f1)
    echo "  📺 $line"
    # Get last 3 lines of output from each session
    output=$(tmux capture-pane -t "$session" -p 2>/dev/null | grep -v '^$' | tail -3)
    if [ -n "$output" ]; then
        echo "$output" | while read l; do echo "     $l"; done
    fi
done

echo ""
echo "── TRAINING PROCESSES ───────────────────────────────────"
ps aux | grep -E 'train|nemo|finetune|lora|torchrun|accelerate|deepspeed' \
    | grep -v grep | while read line; do
    pid=$(echo "$line" | awk '{print $2}')
    cpu=$(echo "$line" | awk '{print $3}')
    mem=$(echo "$line" | awk '{print $4}')
    cmd=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf $i" "; print ""}' | cut -c1-80)
    echo "  PID $pid  CPU:${cpu}%  MEM:${mem}%  $cmd"
done

echo ""
echo "── COSMOS SERVERS (DO NOT TOUCH) ────────────────────────"
for srv in cosmos_reason_server cosmos_demo_server cosmos_predict_server cosmos_transfer_server; do
    pid=$(pgrep -f "$srv" 2>/dev/null | head -1)
    if [ -n "$pid" ]; then
        gpu=$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory \
            --format=csv,noheader 2>/dev/null | grep "^$pid," | awk -F, '{print $3}')
        echo "  ✅ $srv  PID:$pid  GPU_MEM:${gpu:-unknown}"
    else
        echo "  ❌ $srv  NOT RUNNING"
    fi
done

echo ""
echo "── NIS PROTOCOL (AGENT COMMS) ───────────────────────────"
nis_pid=$(pgrep -f "uvicorn main:app" 2>/dev/null | head -1)
if [ -n "$nis_pid" ]; then
    uptime_s=$(ps -o etimes= -p "$nis_pid" 2>/dev/null | tr -d ' ')
    uptime_h=$((uptime_s / 3600))
    uptime_m=$(( (uptime_s % 3600) / 60 ))
    echo "  ✅ NIS uvicorn  PID:$nis_pid  uptime:${uptime_h}h${uptime_m}m"
    # Quick health check
    health=$(curl -s --max-time 3 http://localhost:8000/health 2>/dev/null | python3 -c \
        "import sys,json; d=json.load(sys.stdin); print(f\"v{d.get('version')} routes={d.get('modular_routes')}\")" 2>/dev/null)
    echo "  ✅ NIS health: $health"
else
    echo "  ❌ NIS uvicorn NOT RUNNING — restarting..."
    cd /data/organica-ai/NIS_Protocol && \
    nohup /home/nvidia/organica-ai/NIS_Protocol/venv/bin/uvicorn main:app \
        --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 120 \
        > /tmp/nis_restart.log 2>&1 &
    echo "  🔄 NIS restarted PID:$!"
fi

echo ""
echo "── AGENT WEBSOCKET CONNECTIONS ──────────────────────────"
ws_conns=$(ss -tnp 2>/dev/null | grep ':8000' | grep ESTAB | wc -l)
echo "  Active connections on :8000 → $ws_conns"
ss -tnp 2>/dev/null | grep ':8000' | grep ESTAB | head -5 | while read line; do
    echo "  $line"
done

echo ""
echo "── DISK & MEMORY ────────────────────────────────────────"
df -h /data /tmp 2>/dev/null | tail -n +2 | while read line; do
    echo "  $line"
done
free -h | grep Mem | awk '{printf "  RAM: used=%s free=%s total=%s\n", $3, $4, $2}'

echo ""
echo "── TRAINING LOG TAILS ───────────────────────────────────"
for log in /tmp/train*.log /data/organica-ai/logs/train*.log /var/log/train*.log 2>/dev/null; do
    [ -f "$log" ] || continue
    echo "  📄 $log (last 5 lines):"
    tail -5 "$log" | while read l; do echo "     $l"; done
done

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "  Done. Cosmos GPUs untouched."
echo "╚══════════════════════════════════════════════════════╝"

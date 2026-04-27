#!/bin/bash
# NeuroLinux Training Watchdog
# Monitors all tmux training sessions and auto-restarts if they die.
# Run: nohup bash keep_training.sh > /data/organica-ai/logs/watchdog.log 2>&1 &
#
# Sessions monitored:
#   vla_heavy          GPU1  train_vla_heavy.py           (active ~475k/500k)
#   grasp_v2           GPU2  train_grasp_v2.py            (active ~37k/200k)
#   vla_heavy_v2       GPU3  train_vla_heavy_v2.py        (active ~617k/750k)
#   vla_heavy_v3       GPU4  train_vla_heavy_v3.py        (active ~7k/750k)
#   speech2action      GPU5  train_speech2action_heavy.py (active ~400/500k)
#   cosmos_reason2_ft  GPU6  train_cosmos_reason2_finetune.py
#   cosmos_predict25   GPU7  train_cosmos_predict25_lora.py
#   codellama          GPU0  train_codellama34b_nis.py    (DONE)
#   mistral            GPU3  train_mistral7b_nis.py       (DONE)
#   safety_v4          GPU5  train_safety_v4.py           (DONE)

set -u
LOG_DIR="/data/organica-ai/logs"
TRAIN_DIR="/data/organica-ai/training/heavy"
VENV="/home/nvidia/organica-ai/venv/bin/python"
CHECK_INTERVAL=60   # seconds between checks

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/watchdog.log"; }

# ── Session definitions ───────────────────────────────────────────
# Format: "session_name|gpu|script|done_marker"
# done_marker: a file that exists when training is complete — skip restart if present
declare -A SESSION_GPU=(
    [vla_heavy]=1
    [grasp_v2]=2
    [vla_heavy_v2]=3
    [vla_heavy_v3]=4
    [speech2action]=5
    [cosmos_reason2_ft]=6
    [cosmos_predict25]=7
    [codellama]=0
    [mistral]=3
    [safety_v4]=5
)
declare -A SESSION_SCRIPT=(
    [vla_heavy]="train_vla_heavy.py"
    [grasp_v2]="train_grasp_v2.py"
    [vla_heavy_v2]="train_vla_heavy_v2.py"
    [vla_heavy_v3]="train_vla_heavy_v3.py"
    [speech2action]="train_speech2action_heavy.py"
    [cosmos_reason2_ft]="train_cosmos_reason2_finetune.py"
    [cosmos_predict25]="train_cosmos_predict25_lora.py"
    [codellama]="train_codellama34b_nis.py"
    [mistral]="train_mistral7b_nis.py"
    [safety_v4]="train_safety_v4.py"
)
declare -A SESSION_DONE=(
    [vla_heavy]=""
    [grasp_v2]=""
    [vla_heavy_v2]=""
    [vla_heavy_v3]=""
    [speech2action]=""
    [cosmos_reason2_ft]=""
    [cosmos_predict25]=""
    [codellama]="/data/organica-ai/models/codellama34b_nis/done.flag"
    [mistral]="/data/organica-ai/models/mistral7b_nis/done.flag"
    [safety_v4]="/data/organica-ai/models/safety_v4/done.flag"
)

is_session_alive() {
    local session="$1"
    tmux has-session -t "$session" 2>/dev/null && return 0 || return 1
}

is_training_process_running() {
    local script="$1"
    pgrep -f "$script" > /dev/null 2>&1
}

is_done() {
    local session="$1"
    local flag="${SESSION_DONE[$session]}"
    [ -n "$flag" ] && [ -f "$flag" ] && return 0
    # Also check log for DONE marker
    local logfile="$LOG_DIR/${session}_gpu${SESSION_GPU[$session]}.log"
    [ -f "$logfile" ] && grep -q "DONE\|Training complete\|Best val" "$logfile" 2>/dev/null && return 0
    return 1
}

restart_session() {
    local session="$1"
    local gpu="${SESSION_GPU[$session]}"
    local script="${SESSION_SCRIPT[$session]}"
    local logfile="$LOG_DIR/${session}_gpu${gpu}.log"

    log "RESTARTING session '$session' on GPU $gpu → $script"

    # Kill any zombie processes
    pkill -f "$script" 2>/dev/null || true
    sleep 2

    # Kill old tmux session if exists
    tmux kill-session -t "$session" 2>/dev/null || true
    sleep 1

    # Start new tmux session
    tmux new-session -d -s "$session" \
        "CUDA_VISIBLE_DEVICES=$gpu stdbuf -oL $VENV $TRAIN_DIR/$script >> $logfile 2>&1"

    sleep 5
    if is_session_alive "$session"; then
        log "  ✅ '$session' restarted successfully (tmux session alive)"
    else
        log "  ❌ '$session' restart FAILED — check $logfile"
    fi
}

log "════════════════════════════════════════════════"
log "  NeuroLinux Training Watchdog started"
log "  Monitoring: ${!SESSION_GPU[*]}"
log "  Check interval: ${CHECK_INTERVAL}s"
log "════════════════════════════════════════════════"

# ── Main watchdog loop ────────────────────────────────────────────
while true; do
    for session in "${!SESSION_GPU[@]}"; do
        script="${SESSION_SCRIPT[$session]}"
        gpu="${SESSION_GPU[$session]}"

        # Skip if already done
        if is_done "$session"; then
            continue
        fi

        # Check if tmux session exists AND training process is running
        session_ok=false
        process_ok=false
        is_session_alive "$session" && session_ok=true
        is_training_process_running "$script" && process_ok=true

        if $session_ok && $process_ok; then
            # Healthy — log progress line from log
            logfile="$LOG_DIR/${session}_gpu${gpu}.log"
            if [ -f "$logfile" ]; then
                last=$(tail -1 "$logfile" 2>/dev/null)
                log "OK [$session] $last"
            fi
        else
            log "⚠  [$session] DEAD — session_alive=$session_ok process_running=$process_ok"
            restart_session "$session"
        fi
    done

    # Also check GPU utilization — warn if training GPU is idle
    gpu_util=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | \
        awk -F, '{gsub(/ /,"",$0); if ($2 < 10) print "GPU"$1" idle("$2"%)"}')
    if [ -n "$gpu_util" ]; then
        log "GPU idle warning: $gpu_util"
    fi

    sleep "$CHECK_INTERVAL"
done

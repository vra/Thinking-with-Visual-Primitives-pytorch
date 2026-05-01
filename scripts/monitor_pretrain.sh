#!/bin/bash
# Monitor pretraining script - auto-restart on divergence or crash

LOG_FILE="outputs/pretrain/train_v3.log"
PID_FILE="outputs/pretrain/monitor.pid"
RESTART_COUNT_FILE="outputs/pretrain/restart_count.txt"
MAX_RESTARTS=10

mkdir -p outputs/pretrain
echo $$ > "$PID_FILE"

touch "$RESTART_COUNT_FILE"
RESTART_COUNT=$(cat "$RESTART_COUNT_FILE" 2>/dev/null || echo 0)

# Exponential backoff: wait longer after each restart
get_backoff_seconds() {
    local count=$1
    if [ "$count" -le 1 ]; then
        echo 10
    elif [ "$count" -le 3 ]; then
        echo 60
    elif [ "$count" -le 5 ]; then
        echo 300
    else
        echo 600
    fi
}

check_training() {
    # Check if python process is running
    PYTHON_PID=$(pgrep -f "train_pretrain.py.*pretrain_12g")
    if [ -z "$PYTHON_PID" ]; then
        echo "[$(date)] WARNING: Training process not found!"
        return 1
    fi
    
    # Check for divergence (nan streak >= 50)
    if tail -100 "$LOG_FILE" 2>/dev/null | grep -q "Training diverged"; then
        echo "[$(date)] CRITICAL: Training diverged detected!"
        return 2
    fi
    
    # Check for Python exception/crash
    if tail -50 "$LOG_FILE" 2>/dev/null | grep -qE "(Traceback|RuntimeError|OutOfMemoryError)"; then
        echo "[$(date)] CRITICAL: Python exception detected!"
        return 3
    fi
    
    # Show current progress
    LAST_LOG=$(tail -1 "$LOG_FILE" 2>/dev/null | grep -oE "Epoch [0-9]+:.*loss=[0-9.nan]+.*lr=[0-9.e-]+" | tail -1)
    if [ -n "$LAST_LOG" ]; then
        echo "[$(date)] OK - $LAST_LOG"
    fi
    
    return 0
}

while true; do
    check_training
    STATUS=$?
    
    if [ $STATUS -ne 0 ]; then
        # Check max restarts
        if [ "$RESTART_COUNT" -ge "$MAX_RESTARTS" ]; then
            echo "[$(date)] FATAL: Max restarts ($MAX_RESTARTS) reached. Giving up."
            exit 1
        fi
        
        # Kill any remaining python process
        pkill -f "train_pretrain.py.*pretrain_12g" 2>/dev/null
        sleep 5
        
        RESTART_COUNT=$((RESTART_COUNT + 1))
        echo "$RESTART_COUNT" > "$RESTART_COUNT_FILE"
        
        BACKOFF=$(get_backoff_seconds "$RESTART_COUNT")
        echo "[$(date)] Waiting ${BACKOFF}s before restart (attempt #$RESTART_COUNT/$MAX_RESTARTS)..."
        sleep "$BACKOFF"
        
        # Clear old diverged marker from log to avoid immediate re-detection
        if [ -f "$LOG_FILE" ]; then
            sed -i 's/Training diverged/Training diverged (resolved)/g' "$LOG_FILE"
        fi
        
        # Determine resume path
        if [ -d "outputs/pretrain/epoch_0" ]; then
            RESUME_PATH="outputs/pretrain/epoch_0"
        else
            RESUME_PATH=""
        fi
        
        echo "[$(date)] Restarting training (attempt #$RESTART_COUNT/$MAX_RESTARTS)..."
        cd /mnt/hdd/ws/project/github/Thinking-with-Visual-Primitives-pytorch
        if [ -n "$RESUME_PATH" ]; then
            nohup python pretraining/train_pretrain.py \
                --config configs/pretrain_12g.yaml \
                --resume "$RESUME_PATH" \
                --use_wandb \
                --wandb_run_name "pretrain_12g_auto_restart_${RESTART_COUNT}" \
                >> outputs/pretrain/train_v3.log 2>&1 &
        else
            nohup python pretraining/train_pretrain.py \
                --config configs/pretrain_12g.yaml \
                --use_wandb \
                --wandb_run_name "pretrain_12g_auto_restart_${RESTART_COUNT}" \
                >> outputs/pretrain/train_v3.log 2>&1 &
        fi
        
        echo "[$(date)] New PID: $!"
    fi
    
    # Check every 5 minutes
    sleep 300
done

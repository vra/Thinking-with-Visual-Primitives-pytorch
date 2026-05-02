#!/bin/bash
# Monitor SFT training progress

LOG_FILE="outputs/sft_box/train.log"
INTERVAL=30

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE"
    exit 1
fi

echo "Monitoring SFT training..."
echo "Log: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "=== SFT Training Monitor ==="
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Check if process is running
    if pgrep -f "train_sft_box.py" > /dev/null; then
        echo "Status: RUNNING"
    else
        echo "Status: NOT RUNNING"
    fi
    echo ""
    
    # Show latest progress
    echo "--- Latest Progress ---"
    grep -E "Epoch [0-9]+:" "$LOG_FILE" | tail -5
    echo ""
    
    # Show epoch summary if available
    echo "--- Epoch Summaries ---"
    grep -E "Epoch [0-9]+ avg loss:" "$LOG_FILE" | tail -3
    echo ""
    
    # Show GPU usage
    echo "--- GPU Usage ---"
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
    echo ""
    
    # Show log tail
    echo "--- Log Tail (last 10 lines) ---"
    tail -10 "$LOG_FILE"
    
    sleep $INTERVAL
done

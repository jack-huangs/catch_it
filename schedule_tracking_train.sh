#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="/home/jack/桌面/catch_it"
LOG_PATH="$PROJECT_DIR/logs/scheduled_tracking_20260415_042806.log"

sleep 5400

cd "$PROJECT_DIR"
source /home/jack/miniconda3/etc/profile.d/conda.sh
conda activate dcmm

echo "[start $(date '+%Y-%m-%d %H:%M:%S %z')] python train_DCMM.py task=Tracking" >> "$LOG_PATH"
python train_DCMM.py task=Tracking >> "$LOG_PATH" 2>&1

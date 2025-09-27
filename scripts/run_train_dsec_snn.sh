#!/usr/bin/env bash
set -euo pipefail

# Change to repo root
# cd "$(dirname "$0")/../.."

# GPU selection (optional)
# export CUDA_VISIBLE_DEVICES=0

export WANDB_MODE=disabled

# Paths
PYTHON=python
TRAIN_SCRIPT=scripts/train_dsec.py

# Output
OUTPUT_DIR=/root/autodl-tmp/runs/dsec_snn
EXP_NAME=snn_yaml_s_fasttrend

# SNN backbone config
SNN_YAML=src/dagr/cfg/snn_yolov8.yaml
SNN_SCALE=s
SNN_TEMPORAL_BINS=4

# Hyperparameters (default to config's baseline)
BATCH_SIZE=32
EPOCHS=801
LR=0.0002
WEIGHT_DECAY=0.00001

# Dataset settings (adjust DATASET_DIR if needed)
DATASET=dsec
# Experiment trend mode: fast | mid | full
EXP_TREND=full
# If FLAGS expects a dataset directory, set it via --dataset_directory
# Example: DATASET_DIR=/path/to/DSEC
DATASET_DIR=/root/autodl-tmp

# Create log file with timestamp
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUTPUT_DIR"

echo "Training log will be saved to: $LOG_FILE"
echo "Starting training..."

$PYTHON "$TRAIN_SCRIPT" \
  --config config/dagr-s-dsec.yaml \
  --dataset "$DATASET" \
  --output_directory "$OUTPUT_DIR" \
  --exp_name "$EXP_NAME" \
  --batch_size "$BATCH_SIZE" \
  --tot_num_epochs "$EPOCHS" \
  --l_r "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --exp_trend "$EXP_TREND" \
  --use_snn_backbone \
  --snn_yaml_path "$SNN_YAML" \
  --snn_scale "$SNN_SCALE" \
  --snn_temporal_bins "$SNN_TEMPORAL_BINS" \
  --dataset_directory "$DATASET_DIR" \
  2>&1 | tee "$LOG_FILE"

# If your FLAGS require --dataset_directory, add: \
#   --dataset_directory /absolute/path/to/datasets




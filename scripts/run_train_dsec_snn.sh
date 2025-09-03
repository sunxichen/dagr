#!/usr/bin/env bash
set -euo pipefail

# Change to repo root
cd "$(dirname "$0")/../.."

# GPU selection (optional)
# export CUDA_VISIBLE_DEVICES=0

# Paths
PYTHON=python
TRAIN_SCRIPT=dagr/scripts/train_dsec.py

# Output
OUTPUT_DIR=/root/autodl-tmp/runs/dsec_snn
EXP_NAME=snn_yaml_s

# SNN backbone config
SNN_YAML=dagr/src/dagr/cfg/snn_yolov8.yaml
SNN_SCALE=s

# Hyperparameters (adjust as needed)
BATCH_SIZE=8
EPOCHS=50
LR=0.001
WEIGHT_DECAY=0.0005

# Dataset settings (adjust DATASET_DIR if needed)
DATASET=dsec
# If FLAGS expects a dataset directory, set it via --dataset_directory
# Example: DATASET_DIR=/path/to/DSEC
DATASET_DIR=/root/autodl-tmp/DSEC

$PYTHON "$TRAIN_SCRIPT" \
  --dataset "$DATASET" \
  --output_directory "$OUTPUT_DIR" \
  --exp_name "$EXP_NAME" \
  --batch_size "$BATCH_SIZE" \
  --tot_num_epochs "$EPOCHS" \
  --l_r "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --use_snn_backbone true \
  --snn_yaml_path "$SNN_YAML" \
  --snn_scale "$SNN_SCALE" \
  --use_image false \
  --dataset_directory "$DATASET_DIR"

# If your FLAGS require --dataset_directory, add: \
#   --dataset_directory /absolute/path/to/datasets




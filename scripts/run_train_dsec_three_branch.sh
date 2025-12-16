#!/usr/bin/env bash
set -euo pipefail

# Change to repo root
# cd "$(dirname "$0")/../.."

# GPU selection (optional)
# export CUDA_VISIBLE_DEVICES=0

export WANDB_MODE=disabled
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DISTRIBUTED=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export NO_EVAL=${NO_EVAL:-0}

# Paths
PYTHON=python
TORCHRUN=torchrun
TRAIN_SCRIPT=scripts/train_dsec.py

# Output
OUTPUT_DIR=/root/autodl-tmp/runs/dsec_three_branch
EXP_NAME=three_branch_s_fasttrend

# SNN backbone config
SNN_YAML=src/dagr/cfg/snn_yolov8.yaml
SNN_SCALE=s
SNN_TEMPORAL_BINS=4

# Hyperparameters (per-GPU semantics)
BATCH_SIZE=1
EPOCHS=801
LR=0.0002
WEIGHT_DECAY=0.00001

# Dataset settings (adjust DATASET_DIR if needed)
DATASET=dsec
# Experiment trend mode: fast | mid | full
EXP_TREND=fast
# If FLAGS expects a dataset directory, set it via --dataset_directory
# Example: DATASET_DIR=/path/to/DSEC
DATASET_DIR=/root/autodl-tmp
MAD_FLOW_CHECKPOINT=/root/autodl-tmp/checkpoints/mad_flow.pth
# Create log file with timestamp
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUTPUT_DIR"

echo "Training log will be saved to: $LOG_FILE"
echo "Starting training..."

# Optional flags
NO_EVAL_FLAG=()
if [[ "${NO_EVAL}" -eq 1 ]]; then
  NO_EVAL_FLAG+=(--no_eval)
fi

# If DISTRIBUTED=1, run with torchrun and enable --distributed
if [[ "${DISTRIBUTED:-0}" -eq 1 ]]; then
  # Infer NUM_GPUS from CUDA_VISIBLE_DEVICES if not provided
  if [[ -z "${NUM_GPUS:-}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      IFS=',' read -r -a DEV_ARR <<< "$CUDA_VISIBLE_DEVICES"
      NUM_GPUS=${#DEV_ARR[@]}
    else
      NUM_GPUS=1
    fi
  fi
  $TORCHRUN --nproc_per_node="$NUM_GPUS" "$TRAIN_SCRIPT" \
    --distributed \
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
    --use_image \
    --snn_yaml_path "$SNN_YAML" \
    --snn_scale "$SNN_SCALE" \
    --snn_temporal_bins "$SNN_TEMPORAL_BINS" \
    --dataset_directory "$DATASET_DIR" \
    --mad_flow_checkpoint "$MAD_FLOW_CHECKPOINT" \
    --no_load_mad_flow \
    --use_checkpointing \
    --debug_unused_params \
    --print_param_index_map \
    "${NO_EVAL_FLAG[@]}" \
    2>&1 | tee "$LOG_FILE"
else
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
    --use_image \
    --img_net resnet18 \
    --snn_yaml_path "$SNN_YAML" \
    --snn_scale "$SNN_SCALE" \
    --snn_temporal_bins "$SNN_TEMPORAL_BINS" \
    --dataset_directory "$DATASET_DIR" \
    --mad_flow_checkpoint "$MAD_FLOW_CHECKPOINT" \
    --use_checkpointing \
    --no_load_mad_flow \
    "${NO_EVAL_FLAG[@]}" \
    2>&1 | tee "$LOG_FILE"
fi

# If your FLAGS require --dataset_directory, add: \
#   --dataset_directory /absolute/path/to/datasets




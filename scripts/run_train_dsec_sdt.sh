#!/usr/bin/env bash
set -euo pipefail

# GPU selection (optional)
# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DISTRIBUTED=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export WANDB_MODE=disabled
export NO_EVAL=${NO_EVAL:-0}

# Paths
PYTHON=python
TORCHRUN=torchrun
TRAIN_SCRIPT=scripts/train_dsec.py

# Output
OUTPUT_DIR=/root/autodl-tmp/runs/dsec_sdtv3
EXP_NAME=sdtv3_s_fasttrend

# SDT-V3 backbone config (event-only)
BACKBONE_TYPE=sdtv3
SDT_T=4
SDT_IN_CHANNELS=2          # polarity events
# SDT_EMBED_DIMS="128 256 512 640"
# SDT_DEPTHS="2 2 6 2"
# Reduce model size to fit in 24GB VRAM
SDT_EMBED_DIMS="64 128 256 512"
SDT_DEPTHS="2 2 2 2"       # Reduced depths
SDT_NUM_HEADS=8
SDT_MLP_RATIO=4.0
SDT_NORM=4.0
SDT_CHECKPOINT=1

# Hyperparameters (per-GPU semantics)
BATCH_SIZE=1
EPOCHS=801
LR=0.0002
WEIGHT_DECAY=0.00001

# Dataset settings (adjust DATASET_DIR if needed)
DATASET=dsec
EXP_TREND=fast           # fast | mid | full
DATASET_DIR=/root/autodl-tmp

# (Optional) MAD flow checkpoint not used in SDT-only run

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

COMMON_ARGS=(
  --config config/dagr-s-dsec.yaml
  --dataset "$DATASET"
  --output_directory "$OUTPUT_DIR"
  --exp_name "$EXP_NAME"
  --batch_size "$BATCH_SIZE"
  --tot_num_epochs "$EPOCHS"
  --l_r "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --exp_trend "$EXP_TREND"
  --use_snn_backbone
  --backbone_type "$BACKBONE_TYPE"
  --sdt_T "$SDT_T"
  --sdt_in_channels "$SDT_IN_CHANNELS"
  --sdt_embed_dim $SDT_EMBED_DIMS
  --sdt_depths $SDT_DEPTHS
  --sdt_num_heads "$SDT_NUM_HEADS"
  --sdt_mlp_ratio "$SDT_MLP_RATIO"
  --sdt_norm "$SDT_NORM"
  --dataset_directory "$DATASET_DIR"
  --use_checkpointing
  "${NO_EVAL_FLAG[@]}"
)

# Add SDT checkpoint flag if enabled
if [[ "${SDT_CHECKPOINT:-0}" -eq 1 ]]; then
  COMMON_ARGS+=(--sdt_checkpoint)
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
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
else
  $PYTHON "$TRAIN_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
fi

# If your FLAGS require --dataset_directory, ensure DATASET_DIR is set correctly.

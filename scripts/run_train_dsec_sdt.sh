#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# 1. 基础环境与路径配置
# ------------------------------------------------------------------------------

# GPU 选择 (可选)
# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DISTRIBUTED=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export WANDB_MODE=disabled
export NO_EVAL=${NO_EVAL:-0}
# 是否开启mad第三分支训练，1不开启，0开启
export NO_MAD=${NO_MAD:-1}

# Python 与 启动命令
PYTHON=python
TORCHRUN=torchrun
TRAIN_SCRIPT=scripts/train_dsec.py

# 输出路径配置
OUTPUT_DIR=/root/autodl-tmp/runs/dsec_sdtv3
EXP_NAME=sdtv3_s_fasttrend

# ------------------------------------------------------------------------------
# 2. 模型配置切换区 (根据需求取消注释其中一个板块)
# ------------------------------------------------------------------------------

# SDT-V3 基础配置 (所有模式通用)
BACKBONE_TYPE=sdtv3
SDT_T=4
SDT_IN_CHANNELS=2          # 事件极性通道数
SDT_MLP_RATIO=4.0
SDT_NORM=4.0
SDT_CHECKPOINT=1           # 开启 Checkpointing 以节省显存

# --- [选项 1: 加载 19M 预训练权重] (默认不开启) ---
# 说明: 必须严格匹配权重的特殊结构 (SR=4, 最后一层Dim=360)
# ----------------------------------------------------------
# PRETRAINED_WEIGHT="/root/sdtv3_ckpts/V3_19.0M_1x4.pth"
# SDT_EMBED_DIMS="64 128 256 360"  # 注意: 预训练权重的最后一层是特殊的 360
# SDT_DEPTHS="2 2 6 2"            
# SDT_NUM_HEADS=8
# SDT_SR_RATIO=4                  # 预训练权重使用了 4 倍膨胀

# --- [选项 2: 从头训练 (标准结构)] ---
# 说明: 使用标准的 512 维度，硬件效率更高；不加载权重
# ----------------------------------------------------------
PRETRAINED_WEIGHT=""            # 留空表示不加载预训练权重
SDT_EMBED_DIMS="64 128 256 512" # 标准配置，回归 2 的幂次方，更适合硬件
SDT_DEPTHS="2 2 6 2"
SDT_NUM_HEADS=8
SDT_SR_RATIO=4                  # 保持 4 以获得较好的表征能力

# ------------------------------------------------------------------------------
# 3. 训练超参数与数据集
# ------------------------------------------------------------------------------

# 训练参数 (per-GPU)
BATCH_SIZE=16
EPOCHS=801
# 注意：实际学习率 = LR * sqrt(BATCH_SIZE) / sqrt(64)
LR=0.0002
WEIGHT_DECAY=0.00001

# 数据集设置
DATASET=dsec
EXP_TREND=fast           # fast | mid | full
DATASET_DIR=/root/autodl-tmp

# 日志文件
mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "Training log will be saved to: $LOG_FILE"
echo "Starting training..."
echo "Mode: PRETRAINED_WEIGHT='${PRETRAINED_WEIGHT}'"
echo "Dims: ${SDT_EMBED_DIMS}"
if [[ "${NO_MAD}" -eq 1 ]]; then
  echo "MAD branch: DISABLED (--no_mad). Using 2-branch mode (Fused + RGB only)."
else
  echo "MAD branch: ENABLED. Using 3-branch mode (Fused + RGB + MAD)."
fi

# ------------------------------------------------------------------------------
# 4. 参数组装与启动
# ------------------------------------------------------------------------------

# 可选标志处理
NO_EVAL_FLAG=()
if [[ "${NO_EVAL}" -eq 1 ]]; then
  NO_EVAL_FLAG+=(--no_eval)
fi

NO_MAD_FLAG=()
if [[ "${NO_MAD}" -eq 1 ]]; then
  NO_MAD_FLAG+=(--no_mad)
fi

# 组装通用参数
# 去掉--use_image只打开sdtv3训练
# 加上--use_image，根据NO_MAD变量的情况，判断是否打开第三分支
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
  --use_image
  --img_net resnet18
  --backbone_type "$BACKBONE_TYPE"
  --sdt_T "$SDT_T"
  --sdt_in_channels "$SDT_IN_CHANNELS"
  --sdt_embed_dim $SDT_EMBED_DIMS
  --sdt_depths $SDT_DEPTHS
  --sdt_num_heads "$SDT_NUM_HEADS"
  --sdt_mlp_ratio "$SDT_MLP_RATIO"
  --sdt_norm "$SDT_NORM"
  --sdt_sr_ratio "$SDT_SR_RATIO"
  --dataset_directory "$DATASET_DIR"
  "${NO_EVAL_FLAG[@]}"
  "${NO_MAD_FLAG[@]}"
)

# 添加 Checkpoint 标志
if [[ "${SDT_CHECKPOINT:-0}" -eq 1 ]]; then
  COMMON_ARGS+=(--use_checkpointing) # 注意: dagr 代码通常使用 --use_checkpointing 来控制
fi

# 仅当 PRETRAINED_WEIGHT 非空时，才添加加载参数
if [[ -n "${PRETRAINED_WEIGHT}" ]]; then
  COMMON_ARGS+=(--load_pretrained_weight "$PRETRAINED_WEIGHT")
fi

# ------------------------------------------------------------------------------
# 5. 执行训练
# ------------------------------------------------------------------------------

# 分布式启动 (DISTRIBUTED=1)
if [[ "${DISTRIBUTED:-0}" -eq 1 ]]; then
  # 自动推断 GPU 数量
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

# 单卡启动
else
  $PYTHON "$TRAIN_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
fi
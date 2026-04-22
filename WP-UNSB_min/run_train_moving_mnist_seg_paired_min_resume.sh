#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

#
# Moving MNIST Paired Dataset用 UNSB 再開訓練スクリプト
#
# ドメインA: オリジナルMoving MNIST
# ドメインB: B-spline変換後のMoving MNIST

# スクリプトのディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# 日付の取得（ログ用）
DATE=$(date +"%Y%m%d_%H%M%S")

# ===== Resume settings =====
# Resume from an existing run directory by date stamp.
RESUME_DATE=${RESUME_DATE:-20260416_172616}
TARGET_TOTAL_EPOCHS=${TARGET_TOTAL_EPOCHS:-5000}

# 保存先ディレクトリ（絶対パス）
# Resume: write into the same dated directory as the previous run.
RESULT_DIR="${WORKSPACE_DIR}/data/experiment_result/WP-UNSB_min/moving-mnist/${RESUME_DATE}"

# ディレクトリ作成
mkdir -p "${RESULT_DIR}"

# データセットパス（絶対パス）
DATAROOT="${WORKSPACE_DIR}/data/org_data/moving_mnist"
DATAROOT_B="${WORKSPACE_DIR}/data/preprocessed/bspline_transformed"

# 実験名（保存先の最後のフォルダ名として使用）
NAME="moving_mnist_seg_paired_sb_wo_GL"
RUN_DIR="${RESULT_DIR}/${NAME}"

# GPU設定
GPU_IDS=0

# ===== W&B logging (optional) =====
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-WP-UNSB_min}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_GROUP=${WANDB_GROUP:-moving-mnist}
WANDB_TAGS=${WANDB_TAGS:-moving-mnist,WP-UNSB_ver2}
WANDB_IMAGE_FREQ=${WANDB_IMAGE_FREQ:-2000}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${NAME}_${DATE}}
WANDB_RUN_ID=${WANDB_RUN_ID:-5h882iyt}
WANDB_RESUME=${WANDB_RESUME:-must}

# モデル設定
MODEL="wpsb"
MODE="sb"

# ネットワーク構造（グレースケール用）
INPUT_NC=1
OUTPUT_NC=1
NGF=64
NDF=64

# データセット設定
DATASET_MODE="moving_mnist_paired"
LOAD_SIZE=64
CROP_SIZE=64
BATCH_SIZE=1

# 損失の重み
LAMBDA_GAN=0
LAMBDA_GAN_SEQ=0
LAMBDA_SB=1.0
LAMBDA_NCE=0

# シーケンス設定
SEQ_LEN=20       # num_frames_per_seq と一致させること

# タイムステップ
NUM_TIMESTEPS=5

# 訓練設定
# Global target: train until around epoch 5000.
N_EPOCHS=100000
N_EPOCHS_DECAY=0
LR=0.00001
SAVE_EPOCH_FREQ=10
PRINT_FREQ=${PRINT_FREQ:-1}

# Resume start epoch is auto-detected from existing checkpoint files.
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "[ERROR] Resume run directory not found: ${RUN_DIR}"
  exit 1
fi

LAST_EPOCH=$(find "${RUN_DIR}" -maxdepth 1 -type f -name '*_net_G.pth' \
  | sed -n 's#.*/\([0-9][0-9]*\)_net_G\.pth#\1#p' \
  | sort -n \
  | tail -1)

if [[ -z "${LAST_EPOCH}" ]]; then
  echo "[ERROR] No numeric checkpoint like <epoch>_net_G.pth found in: ${RUN_DIR}"
  exit 1
fi

EPOCH_COUNT=$((LAST_EPOCH + 1))

if (( EPOCH_COUNT > TARGET_TOTAL_EPOCHS )); then
  echo "[ERROR] Detected next epoch (${EPOCH_COUNT}) exceeds target (${TARGET_TOTAL_EPOCHS})."
  exit 1
fi

echo "======================================"
echo "Moving MNIST Paired UNSB Resume Training"
echo "======================================"
echo "Domain A: ${DATAROOT}/mnist_test_seq.npy"
echo "Domain B: ${DATAROOT_B}/transformed_global.npy"
echo "Experiment: ${NAME}"
echo "Date: ${DATE}"
echo "Resume Date: ${RESUME_DATE}"
echo "Resume from epoch: ${EPOCH_COUNT} (last saved: ${LAST_EPOCH})"
echo "Target total epochs: ${TARGET_TOTAL_EPOCHS}"
echo "Wandb Resume: ${WANDB_RESUME}"
echo "Wandb Run ID: ${WANDB_RUN_ID}"
echo "Results will be saved to: ${RESULT_DIR}"
echo "GPU: ${GPU_IDS}"
echo "======================================"

# 設定をログに保存
cat > "${RESULT_DIR}/config_resume.txt" << EOF
Resume Experiment Configuration
==============================
Date: ${DATE}
Resume Date: ${RESUME_DATE}
Name: ${NAME}

Data:
  Domain A: ${DATAROOT}/mnist_test_seq.npy
  Domain B: ${DATAROOT_B}/transformed_aligned.npy

Model:
  Model: ${MODEL}
  Mode: ${MODE}
  Input NC: ${INPUT_NC}
  Output NC: ${OUTPUT_NC}
  NGF: ${NGF}
  NDF: ${NDF}
  Num Timesteps: ${NUM_TIMESTEPS}

Training:
  Batch Size: ${BATCH_SIZE}
  Load Size: ${LOAD_SIZE}
  Crop Size: ${CROP_SIZE}
  Print Freq: ${PRINT_FREQ}
  Learning Rate: ${LR}
  Resume Epoch Count: ${EPOCH_COUNT}
  Last Saved Epoch: ${LAST_EPOCH}
  Target Total Epochs: ${TARGET_TOTAL_EPOCHS}
  Epochs: ${N_EPOCHS}
  Decay Epochs: ${N_EPOCHS_DECAY}

Loss Weights:
  Lambda GAN: ${LAMBDA_GAN}
  Lambda SB: ${LAMBDA_SB}
  Lambda NCE: ${LAMBDA_NCE}

Wandb:
  Use: ${USE_WANDB}
  Project: ${WANDB_PROJECT}
  Entity: ${WANDB_ENTITY}
  Mode: ${WANDB_MODE}
  Group: ${WANDB_GROUP}
  Tags: ${WANDB_TAGS}
  Run Name: ${WANDB_RUN_NAME}
  Run ID: ${WANDB_RUN_ID}
  Resume: ${WANDB_RESUME}
  Image Freq: ${WANDB_IMAGE_FREQ}
EOF

# WP-UNSB_minディレクトリに移動して実行
cd "${SCRIPT_DIR}"

echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "PWD=$(pwd)"
echo "train.py path=$(realpath train.py)"

cmd=(python3 train.py
  --dataroot ${DATAROOT}
  --dataroot_B ${DATAROOT_B}
  --data_file_A mnist_test_seq.npy
  --data_file_B transformed_global.npy
  --name ${NAME}
  --model ${MODEL}
  --mode ${MODE}
  --dataset_mode ${DATASET_MODE}
  --input_nc ${INPUT_NC}
  --output_nc ${OUTPUT_NC}
  --ngf ${NGF}
  --ndf ${NDF}
  --load_size ${LOAD_SIZE}
  --crop_size ${CROP_SIZE}
  --batch_size ${BATCH_SIZE}
  --num_timesteps ${NUM_TIMESTEPS}
  --lambda_GAN ${LAMBDA_GAN}
  --lambda_SB ${LAMBDA_SB}
  --lambda_NCE ${LAMBDA_NCE}
  --save_ot_details
  --ot_details_max_samples 10
  --lmda 0.5
  --sb_mode seq_ot
  --seq_ot_normalize "mean"
  --n_epochs ${N_EPOCHS}
  --n_epochs_decay ${N_EPOCHS_DECAY}
  --continue_train
  --epoch latest
  --epoch_count ${EPOCH_COUNT}
  --lr ${LR}
  --print_freq ${PRINT_FREQ}
  --seq_ot_monotone_penalty 1.0
  --save_epoch_freq ${SAVE_EPOCH_FREQ}
  --checkpoints_dir ${RESULT_DIR}
  --gpu_ids ${GPU_IDS}
  --preprocess none
  --no_flip
)

if [[ "${USE_WANDB}" == "1" ]]; then
  export WANDB_RUN_ID
  export WANDB_RESUME
  cmd+=(--use_wandb --wandb_project "${WANDB_PROJECT}" --wandb_mode "${WANDB_MODE}" --wandb_run_name "${WANDB_RUN_NAME}" --wandb_group "${WANDB_GROUP}" --wandb_tags "${WANDB_TAGS}" --wandb_image_freq "${WANDB_IMAGE_FREQ}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    cmd+=(--wandb_entity "${WANDB_ENTITY}")
  fi
fi

"${cmd[@]}"

echo "======================================"
echo "Resume training completed!"
echo "Results saved to: ${RESULT_DIR}"
echo "======================================"

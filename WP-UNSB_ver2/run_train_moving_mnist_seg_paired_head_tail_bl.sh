#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

#
# Moving MNIST Paired Dataset用 UNSB 訓練スクリプト
# 
# ドメインA: オリジナルMoving MNIST
# ドメインB: B-spline変換後のMoving MNIST

# スクリプトのディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# 日付の取得
DATE=$(date +"%Y%m%d_%H%M%S")

# 保存先ディレクトリ（絶対パス）
RESULT_DIR="${WORKSPACE_DIR}/data/experiment_result/WP-UNSB_ver2/moving-mnist/${DATE}"

# ディレクトリ作成
mkdir -p "${RESULT_DIR}"

# データセットパス（絶対パス）
DATAROOT="${WORKSPACE_DIR}/data/org_data/moving_mnist"
DATAROOT_B="${WORKSPACE_DIR}/data/preprocessed/bspline_transformed"
DATAFILE_B="transformed_cut_head_tail_global.npy"

# 実験名（保存先の最後のフォルダ名として使用）
NAME="moving_mnist_seg_paired_sb_head_tail_bl"

# GPU設定
GPU_IDS=0

# ===== W&B logging (optional) =====
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-WP-UNSB_ver2}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_GROUP=${WANDB_GROUP:-moving-mnist}
WANDB_TAGS=${WANDB_TAGS:-moving-mnist,WP-UNSB_ver2}
WANDB_IMAGE_FREQ=${WANDB_IMAGE_FREQ:-2000}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${NAME}_${DATE}}

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
BATCH_SIZE=8

# 損失の重み
LAMBDA_GAN=1.0
LAMBDA_SB=1.0
LAMBDA_NCE=1.0

# タイムステップ
NUM_TIMESTEPS=5

# 訓練設定
N_EPOCHS=200
N_EPOCHS_DECAY=200
LR=0.00001
SAVE_EPOCH_FREQ=10

echo "======================================"
echo "Moving MNIST Paired UNSB Training"
echo "======================================"
echo "Domain A: ${DATAROOT}/mnist_test_seq.npy"
echo "Domain B: ${DATAROOT_B}/${DATAFILE_B}"
echo "Experiment: ${NAME}"
echo "Date: ${DATE}"
echo "Results will be saved to: ${RESULT_DIR}"
echo "GPU: ${GPU_IDS}"
echo "======================================"

# 設定をログに保存
cat > "${RESULT_DIR}/config.txt" << EOF
Experiment Configuration
========================
Date: ${DATE}
Name: ${NAME}

Data:
  Domain A: ${DATAROOT}/mnist_test_seq.npy
  Domain B: ${DATAROOT_B}/${DATAFILE_B}

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
  Learning Rate: ${LR}
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
  Image Freq: ${WANDB_IMAGE_FREQ}
EOF

# UNSB-mainディレクトリに移動して実行
cd "${SCRIPT_DIR}"

cmd=(python3 train.py
  --dataroot ${DATAROOT}
  --dataroot_B ${DATAROOT_B}
  --data_file_A mnist_test_seq.npy
  --data_file_B ${DATAFILE_B}
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
  --sb_mode both
  --lambda_SB_original 0.5
  --lambda_SB_seq 0.5
  --n_epochs ${N_EPOCHS}
  --n_epochs_decay ${N_EPOCHS_DECAY}
  --lr ${LR}
  --save_epoch_freq ${SAVE_EPOCH_FREQ}
  --checkpoints_dir ${RESULT_DIR}
  --gpu_ids ${GPU_IDS}
  --preprocess none
  --no_flip
)

if [[ "${USE_WANDB}" == "1" ]]; then
  cmd+=(--use_wandb --wandb_project "${WANDB_PROJECT}" --wandb_mode "${WANDB_MODE}" --wandb_run_name "${WANDB_RUN_NAME}" --wandb_group "${WANDB_GROUP}" --wandb_tags "${WANDB_TAGS}" --wandb_image_freq "${WANDB_IMAGE_FREQ}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    cmd+=(--wandb_entity "${WANDB_ENTITY}")
  fi
fi

"${cmd[@]}"

echo "======================================"
echo "Training completed!"
echo "Results saved to: ${RESULT_DIR}"
echo "======================================"

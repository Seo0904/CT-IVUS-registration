#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

# Moving MNIST Paired Dataset用 UNSB 訓練スクリプト（再開）

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# 再開する学習のDATEを環境変数で受け取る（必須）
# 例: DATE=20260423_165007 bash run_train_moving_mnist_seg_paired_min_ot_div_cloze_resume.sh
if [[ -z "${DATE:-}" ]]; then
  echo "ERROR: DATE is not set. Please specify the experiment date to resume."
  echo "Example: DATE=20260423_165007 bash $(basename "$0")"
  exit 1
fi

# 既存の学習フォルダを指定
RESULT_DIR="${WORKSPACE_DIR}/data/experiment_result/WP-UNSB_min_ver2/moving-mnist/${DATE}"

if [[ ! -d "${RESULT_DIR}" ]]; then
  echo "ERROR: Result directory does not exist: ${RESULT_DIR}"
  exit 1
fi

# データセットパス（絶対パス）
DATAROOT="${WORKSPACE_DIR}/data/org_data/moving_mnist"
DATAROOT_B="${WORKSPACE_DIR}/data/preprocessed/bspline_transformed"

# 実験名（保存先の最後のフォルダ名として使用）
NAME="moving_mnist_seg_paired_sb_wo_GL_w_otdiv_015_cloze"

# GPU設定
GPU_IDS=0

# ===== W&B logging (optional) =====
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-WP-UNSB_min_ver2}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_GROUP=${WANDB_GROUP:-moving-mnist}
WANDB_TAGS=${WANDB_TAGS:-moving-mnist,WP-UNSB_min_ver2}
WANDB_IMAGE_FREQ=${WANDB_IMAGE_FREQ:-1000}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${NAME}_${DATE}_resume}
# 再開するW&B Run IDを環境変数で受け取る（必須）
# 例: WANDB_RUN_ID=xxxxxxxx DATE=20260423_165007 bash run_...resume.sh
WANDB_RUN_ID=${WANDB_RUN_ID:-}
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
BATCH_SIZE=4

# 損失の重み
LAMBDA_GAN=0
LAMBDA_GAN_SEQ=0
LAMBDA_SB=1.0
LAMBDA_NCE=0

# sequence OT (P entropy)
SEQ_OT_P_ENTROPY=${SEQ_OT_P_ENTROPY:-0}
SEQ_OT_P_ENTROPY_PENALTY=${SEQ_OT_P_ENTROPY_PENALTY:-0.0}

# sequence OT (divergence)
SEQ_OT_DIVERGENCE=${SEQ_OT_DIVERGENCE:-1}
SEQ_OT_DIVERGENCE_PENALTY=${SEQ_OT_DIVERGENCE_PENALTY:--0.5}

# タイムステップ
NUM_TIMESTEPS=5

# 訓練設定（400→800epochまで延長）
N_EPOCHS=800
N_EPOCHS_DECAY=0
LR=0.0002
SAVE_EPOCH_FREQ=10
PRINT_FREQ=${PRINT_FREQ:-1}
DISPLAY_FREQ=${DISPLAY_FREQ:-1000}

cd "${SCRIPT_DIR}"

echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "PWD=$(pwd)"
echo "train.py path=$(realpath train.py)"
echo "Resuming from: ${RESULT_DIR}"
echo "Target epochs: ${N_EPOCHS}"

a_cmd=(python3 train.py
  --dataroot ${DATAROOT}
  --dataroot_B ${DATAROOT_B}
  --data_file_A mnist_test_seq.npy
  --data_file_B transformed_cloze_global.npy
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
  --lmda 0.10
  --seq_ot_iters 150
  --sinkhorn_type "sinkhorn_log"
  --sb_mode seq_ot
  --seq_ot_normalize "none"
  --seq_ot_p_entropy ${SEQ_OT_P_ENTROPY}
  --seq_ot_p_entropy_penalty ${SEQ_OT_P_ENTROPY_PENALTY}
  --seq_ot_divergence ${SEQ_OT_DIVERGENCE}
  --seq_ot_divergence_penalty ${SEQ_OT_DIVERGENCE_PENALTY}
  --n_epochs ${N_EPOCHS}
  --n_epochs_decay ${N_EPOCHS_DECAY}
  --lr ${LR}
  --print_freq ${PRINT_FREQ}
  --display_freq ${DISPLAY_FREQ}
  --seq_ot_monotone_penalty 1.0
  --save_epoch_freq ${SAVE_EPOCH_FREQ}
  --checkpoints_dir ${RESULT_DIR}
  --gpu_ids ${GPU_IDS}
  --preprocess none
  --no_flip
  --display_id 0
  --num_threads 0
  --continue_train
  --epoch latest
  --epoch_count 401
  --iter_count 400000
)

if [[ "${USE_WANDB}" == "1" ]]; then
  if [[ -n "${WANDB_RUN_ID}" ]]; then
    export WANDB_RUN_ID
    export WANDB_RESUME
  else
    echo "WARNING: WANDB_RUN_ID is not set. W&B run will not be resumed (a new run will be created)."
  fi
  a_cmd+=(--use_wandb --wandb_project "${WANDB_PROJECT}" --wandb_mode "${WANDB_MODE}" --wandb_run_name "${WANDB_RUN_NAME}" --wandb_group "${WANDB_GROUP}" --wandb_tags "${WANDB_TAGS}" --wandb_image_freq "${WANDB_IMAGE_FREQ}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    a_cmd+=(--wandb_entity "${WANDB_ENTITY}")
  fi
fi

"${a_cmd[@]}"

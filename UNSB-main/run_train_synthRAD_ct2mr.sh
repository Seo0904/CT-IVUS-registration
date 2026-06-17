#!/bin/bash
#
# synthRAD CT -> MR 用 UNSB(本家) 訓練スクリプト
#
# ドメインA: CT  (input)
# ドメインB: MR  (target)
# データ: <DATAROOT>/<region>/<region>.h5  +  split.json
#         h5 構造: CT/<case>, MR/<case> (, MASK/<case>)  各 (X,Y,Z) float32 [-1,1]
#
# WP-UNSB_ct_mri を参考に、UNSB-main のフレーム(スライス)単位パラダイムへ移植した版。
# 1 サンプル = 1 スライス。train は unpaired (B はランダム case/slice)、
# val/test は paired (同 case/同スライス = GT 比較可能)。

# スクリプトのディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# 日付の取得
DATE=$(date +"%Y%m%d_%H%M%S")

# データセット (h5 のあるルート) と region
DATAROOT="${DATAROOT:-/workspace/data/preprocessed/synthRAD_maskfree_full}"
REGION="${REGION:-brain}"

# 保存先ディレクトリ（絶対パス）
RESULT_DIR="${WORKSPACE_DIR}/data/experiment_result/UNSB/synthRAD_${REGION}_ct2mr/${DATE}"
mkdir -p "${RESULT_DIR}"

# 実験名（保存先の最後のフォルダ名として使用）
NAME="${NAME:-synthRAD_${REGION}_ct2mr_sb}"

# GPU設定
GPU_IDS="${GPU_IDS:-0}"

# モデル設定
MODEL="sb"
MODE="sb"

# ネットワーク構造（グレースケール用）
INPUT_NC=1
OUTPUT_NC=1
NGF=64
NDF=64

# データセット設定
DATASET_MODE="ct_mri"
SLICE_AXIS="${SLICE_AXIS:-2}"          # 0=sagittal 1=coronal 2=axial
NUM_SLICES_PER_SEQ="${NUM_SLICES_PER_SEQ:-0}"   # 0=全有効スライス
# 画像サイズ (synthRAD brain は 128x128)。preprocess none なので h5 のサイズをそのまま使う。
LOAD_SIZE="${LOAD_SIZE:-128}"
CROP_SIZE="${CROP_SIZE:-128}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# 損失の重み
LAMBDA_GAN=1.0
LAMBDA_SB=1.0
LAMBDA_NCE=1.0

# タイムステップ
NUM_TIMESTEPS=5

# 訓練設定
N_EPOCHS="${N_EPOCHS:-400}"
N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-0}"
LR="${LR:-0.0002}"
SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-10}"
VAL_EPOCH_FREQ="${VAL_EPOCH_FREQ:-${SAVE_EPOCH_FREQ}}"   # val(SSIM/L2/PSNR/FID)を測る間隔

# W&B 設定 (val 指標も含めて記録される)
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-UNSB-main}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-synthRAD-${REGION}}"
WANDB_TAGS="${WANDB_TAGS:-synthRAD,${REGION},ct2mr,UNSB-main}"
WANDB_IMAGE_FREQ="${WANDB_IMAGE_FREQ:-1000}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${NAME}_${DATE}}"

echo "======================================"
echo "synthRAD CT->MR UNSB Training"
echo "======================================"
echo "Data:       ${DATAROOT}/${REGION}/${REGION}.h5"
echo "Experiment: ${NAME}"
echo "Date:       ${DATE}"
echo "Results:    ${RESULT_DIR}"
echo "GPU:        ${GPU_IDS}"
echo "======================================"

# UNSB-mainディレクトリに移動して実行
cd "${SCRIPT_DIR}"

cmd=(python3 train.py
  --dataroot ${DATAROOT}
  --region ${REGION}
  --name ${NAME}
  --model ${MODEL}
  --mode ${MODE}
  --dataset_mode ${DATASET_MODE}
  --slice_axis ${SLICE_AXIS}
  --num_slices_per_seq ${NUM_SLICES_PER_SEQ}
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
  --n_epochs ${N_EPOCHS}
  --n_epochs_decay ${N_EPOCHS_DECAY}
  --lr ${LR}
  --save_epoch_freq ${SAVE_EPOCH_FREQ}
  --val_epoch_freq ${VAL_EPOCH_FREQ}
  --checkpoints_dir ${RESULT_DIR}
  --gpu_ids ${GPU_IDS}
  --preprocess none
  --no_flip
  --display_id -1
)

# 途中再開: RESUME=1 で latest から再開する。
#   RESUME_DIR  : 既存の checkpoints_dir (RESULT_DIR と同じ構造) を指定 (必須)
#   EPOCH_COUNT : 開始 epoch (例: 64)
#   ITER_COUNT  : 開始 total_iters (W&B step 整合用。例: 700000)
if [[ "${RESUME:-0}" == "1" ]]; then
  RESULT_DIR="${RESUME_DIR:?RESUME=1 のときは RESUME_DIR を指定してください}"
  # checkpoints_dir を既存ディレクトリへ差し替え
  for i in "${!cmd[@]}"; do
    if [[ "${cmd[$i]}" == "${WORKSPACE_DIR}/data/experiment_result/"* ]]; then
      cmd[$i]="${RESULT_DIR}"
    fi
  done
  cmd+=(--continue_train --epoch latest --epoch_count "${EPOCH_COUNT:-64}" --iter_count "${ITER_COUNT:-700000}")
  # 同一 W&B run に追記する場合は WANDB_ID を指定 (例: 8uknok68)
  if [[ -n "${WANDB_ID:-}" ]]; then
    cmd+=(--wandb_id "${WANDB_ID}")
  fi
fi

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

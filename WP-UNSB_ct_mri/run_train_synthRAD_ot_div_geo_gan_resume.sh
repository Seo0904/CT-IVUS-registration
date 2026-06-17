#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

#
# synthRAD brain CT->MR 用 UNSB(seq-OT) 訓練スクリプト（再開）
#
# run_train_synthRAD_ot_div_geo_gan.sh の続きを学習する。
# 400 epoch まで学習済みの実験フォルダから 800 epoch まで延長する。
#
# 使い方:
#   DATE=20260601_120000 bash run_train_synthRAD_ot_div_geo_gan_resume.sh
#   （DATE = 再開したい既存実験フォルダの日付）
#   W&B も再開したい場合:
#     WANDB_RUN_ID=xxxxxxxx DATE=20260601_120000 bash run_train_synthRAD_ot_div_geo_gan_resume.sh

# スクリプトのディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# データ領域 (brain / pelvis)
REGION=${REGION:-brain}

# 再開する学習のDATE（W&B run m42gww73 に対応する実験フォルダ）
# 環境変数 DATE で上書き可能
DATE=${DATE:-20260614_151149}

# 既存の学習フォルダを指定
RESULT_DIR="${WORKSPACE_DIR}/data/experiment_result/WP-UNSB_ct_mri/synthRAD-${REGION}/${DATE}"

if [[ ! -d "${RESULT_DIR}" ]]; then
  echo "ERROR: Result directory does not exist: ${RESULT_DIR}"
  exit 1
fi

# データセットパス（絶対パス）: <DATAROOT>/<REGION>/<REGION>.h5 を読む
DATAROOT="${WORKSPACE_DIR}/data/preprocessed/synthRAD_maskfree"

# 実験名（保存先の最後のフォルダ名として使用）
NAME="synthRAD_${REGION}_ct2mr_sb_w_otdiv_015_geo_gan"

# GPU設定
GPU_IDS=0

# ===== W&B logging (optional) =====
USE_WANDB=${USE_WANDB:-1}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_PROJECT=${WANDB_PROJECT:-WP-UNSB_ct_mri}
WANDB_GROUP=${WANDB_GROUP:-synthRAD-${REGION}}
WANDB_TAGS=${WANDB_TAGS:-synthRAD,${REGION},ct2mr,WP-UNSB_ct_mri}
WANDB_IMAGE_FREQ=${WANDB_IMAGE_FREQ:-1000}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${NAME}_${DATE}_resume}
# 再開するW&B Run ID（同じ run に追記する）。環境変数で上書き可能
WANDB_RUN_ID=${WANDB_RUN_ID:-m42gww73}
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
DATASET_MODE="ct_mri"
SLICE_AXIS=${SLICE_AXIS:-2}            # 2=axial をシーケンス軸に
MIN_MASK_RATIO=${MIN_MASK_RATIO:-0.02} # マスク前景がこの比率未満のスライスは除外
NUM_SLICES_PER_SEQ=${NUM_SLICES_PER_SEQ:-0}  # 0=全スライス(~200)
NETG_CHUNK=${NETG_CHUNK:-32}           # netG forward を frame 軸でこの枚数ずつ実行(32bit overflow回避)
NETG_CHECKPOINT=${NETG_CHECKPOINT:-True} # チャンク毎に gradient checkpointing(全frame保持でのOOM回避)
AMP_NETG=${AMP_NETG:-True}             # netG forward のみ混合精度(fp16)。約2.6x高速。下流(seq-OT/D/E)はfp32
LOAD_SIZE=256                          # preprocess=none のため実リサイズには未使用
CROP_SIZE=256
BATCH_SIZE=1                           # 1患者=1 sequence。volume毎にS/H/Wが違うので1固定

# 損失の重み
LAMBDA_GAN=1.0   # >0 で adversarial loss (Discriminator) を有効化
LAMBDA_GAN_SEQ=0
LAMBDA_SB=1.0
LAMBDA_NCE=0

# sequence OT (P entropy)
SEQ_OT_P_ENTROPY=${SEQ_OT_P_ENTROPY:-0}
SEQ_OT_P_ENTROPY_PENALTY=${SEQ_OT_P_ENTROPY_PENALTY:-0.0}

# sequence OT (divergence)
SEQ_OT_DIVERGENCE=${SEQ_OT_DIVERGENCE:-1}
SEQ_OT_DIVERGENCE_PENALTY=${SEQ_OT_DIVERGENCE_PENALTY:--0.5}

# sequence OT (unbalanced): rho を渡すと unbalanced OT になる。空にすると balanced
SEQ_OT_UNBALANCED=${SEQ_OT_UNBALANCED:-}

# タイムステップ
NUM_TIMESTEPS=5

# 訓練設定（400→800epochまで延長）
N_EPOCHS=800
N_EPOCHS_DECAY=0
LR=0.0002
SAVE_EPOCH_FREQ=10
PRINT_FREQ=${PRINT_FREQ:-1}
DISPLAY_FREQ=${DISPLAY_FREQ:-1000}

# 再開設定
RESUME_EPOCH=${RESUME_EPOCH:-latest}   # ロードするチェックポイント (latest=epoch 510)
EPOCH_COUNT=${EPOCH_COUNT:-511}        # 次に開始する epoch 番号（epoch 513 でsegfault→510から再開）
# W&B step を正しく継続するための累計イテレーション数（= 510 epoch 時点の total_iters = 510 x 63）。
# 環境変数で上書き可能。
ITER_COUNT=${ITER_COUNT:-32130}

echo "======================================"
echo "synthRAD ${REGION} CT->MR UNSB(seq-OT) Training (RESUME)"
echo "======================================"
echo "Data: ${DATAROOT}/${REGION}/${REGION}.h5 (A=CT, B=MR)"
echo "Experiment: ${NAME}"
echo "Resuming from: ${RESULT_DIR}"
echo "Load checkpoint: ${RESUME_EPOCH}  (start epoch=${EPOCH_COUNT})"
echo "Target epochs: ${N_EPOCHS}"
echo "GPU: ${GPU_IDS}"
echo "======================================"

# UNSB-mainディレクトリに移動して実行
cd "${SCRIPT_DIR}"

echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "PWD=$(pwd)"
echo "train.py path=$(realpath train.py)"

cmd=(python3 train.py
  --dataroot ${DATAROOT}
  --region ${REGION}
  --slice_axis ${SLICE_AXIS}
  --min_mask_ratio ${MIN_MASK_RATIO}
  --num_slices_per_seq ${NUM_SLICES_PER_SEQ}
  --netg_chunk ${NETG_CHUNK}
  --netg_checkpoint ${NETG_CHECKPOINT}
  --amp_netg ${AMP_NETG}
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
  --lmda 10
  --seq_ot_iters 300
  --sinkhorn_type "sinkhorn_log"
  --sb_mode seq_ot
  --seq_ot_normalize "none"
  --seq_ot_solver "geo"
  --seq_ot_geo_scaling 0.99
  --seq_ot_geo_p 2
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
  --epoch ${RESUME_EPOCH}
  --epoch_count ${EPOCH_COUNT}
  --iter_count ${ITER_COUNT}
)

# unbalanced OT: SEQ_OT_UNBALANCED が空でなければ rho を渡す（空なら未指定=balanced）
if [[ -n "${SEQ_OT_UNBALANCED}" ]]; then
  cmd+=(--seq_ot_unbalanced ${SEQ_OT_UNBALANCED})
fi

if [[ "${USE_WANDB}" == "1" ]]; then
  if [[ -n "${WANDB_RUN_ID}" ]]; then
    export WANDB_RUN_ID
    export WANDB_RESUME
  else
    echo "WARNING: WANDB_RUN_ID is not set. W&B run will not be resumed (a new run will be created)."
  fi
  cmd+=(--use_wandb --wandb_project "${WANDB_PROJECT}" --wandb_mode "${WANDB_MODE}" --wandb_run_name "${WANDB_RUN_NAME}" --wandb_group "${WANDB_GROUP}" --wandb_tags "${WANDB_TAGS}" --wandb_image_freq "${WANDB_IMAGE_FREQ}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    cmd+=(--wandb_entity "${WANDB_ENTITY}")
  fi
fi

"${cmd[@]}"

echo "======================================"
echo "Training (resume) completed!"
echo "Results saved to: ${RESULT_DIR}"
echo "======================================"

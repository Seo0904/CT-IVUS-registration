#!/bin/bash
#============================================================
# Genkai 用 pjsub ジョブ: synthRAD CT->MR WP-UNSB(seq-OT)
#   コスト空間 = flatten (D特徴空間で L_ot)。★adv (GAN) も G 側に入れる lambda_GAN=1★
#   → D は real/fake 分類で学習しつつ、その特徴空間で L_ot 最小化 + G は D も騙す。
#   投入:  pjsub job_train_synthRAD_ct2mr_flatten_gan.sh
#   ログ:  train_ct2mr_flat_gan.<JOBID>.out
#
# 注意 (adv 比率):
#   lambda_GAN=1 は出発点 (UNSB既定)。GAN と SB の勾配比は学習中に共進化するので、
#   wandb の loss_NETG_GRAD_GAN_NORM / loss_NETG_GRAD_SB_NORM を見て
#   lambda_GAN_new = lambda_GAN × ρ × (SB_grad/GAN_grad), ρ~0.1-0.5 で1回調整推奨。
# 注意 (elapse): flatten ≈ 3.3s/iter。400 epoch ≈ 46h。24h では ~epoch200。
#   完走は elapse を伸ばすか continue_train 再投入。
#============================================================
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -j
#PJM -N train_ct2mr_flat_gan

set -e

# --- conda 環境 ---
source ~/.bashrc 2>/dev/null || true
conda activate ctmri

# --- このジョブスクリプトのある場所 (= WP-UNSB_ct_mri) ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# --- データ位置 ---
WORKSPACE_DIR="$(dirname "${SCRIPT_DIR}")"
export DATAROOT="${DATAROOT:-${WORKSPACE_DIR}/data/preprocessed/synthRAD_maskfree_full}"
export REGION="${REGION:-brain}"

# --- D 特徴空間 OT (flatten) + adv ---
export SEQ_OT_COST_SPACE="${SEQ_OT_COST_SPACE:-flatten}"
export LMDA="${LMDA:-0.05}"          # 勾配ベース選定 (grad~2x pixel)
export SEQ_OT_ITERS="${SEQ_OT_ITERS:-30}"  # P収束&勾配安定の最小
# D は学習 (adv 分類 + 特徴空間提供)。auto でも lambda_GAN>0 なので学習される。
export UPDATE_D="${UPDATE_D:-auto}"
# ★adv を G 側にも入れる。1 は出発点。ログで SB とのバランスを見て調整。
export LAMBDA_GAN="${LAMBDA_GAN:-1}"
export NETG_CHUNK="${NETG_CHUNK:-16}"

# --- 実験名 ---
export NAME="${NAME:-synthRAD_${REGION}_ct2mr_sb_otdiv_geo_dfeat_flatten_gan}"

# --- wandb ---
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_TAGS_EXTRA="${WANDB_TAGS_EXTRA:-dfeat,flatten,gan}"

echo "JOB on $(hostname) at $(date)"
echo "SEQ_OT_COST_SPACE=${SEQ_OT_COST_SPACE}  UPDATE_D=${UPDATE_D}  LAMBDA_GAN=${LAMBDA_GAN}  NETG_CHUNK=${NETG_CHUNK}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- 学習スクリプト本体を実行 ---
bash run_train_synthRAD_ot_div_geo_gan.sh

#!/bin/bash
#============================================================
# Genkai 用 pjsub ジョブ: synthRAD CT->MR WP-UNSB(seq-OT)
#   コスト空間 = flatten (D特徴空間で L_ot)。★Discriminator は学習しない(凍結)★
#   → 固定ランダム特徴空間での OT。D が動かないので reg スケールが drift せず安定。
#   投入:  pjsub job_train_synthRAD_ct2mr_flatten_frozenD.sh
#   ログ:  train_ct2mr_flat_frzD.<JOBID>.out
#
# 注意 (elapse): flatten ≈ 3.3s/iter (A6000実測)。400 epoch ≈ 46h。
#   24h では ~epoch 200。完走は elapse を伸ばすか latest_net_* から continue_train で再投入。
#============================================================
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -j
#PJM -N train_ct2mr_flat_frzD

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

# --- D 特徴空間 OT (flatten), D は凍結 ---
export SEQ_OT_COST_SPACE="${SEQ_OT_COST_SPACE:-flatten}"
export LMDA="${LMDA:-0.05}"          # 勾配ベース選定 (grad~2x pixel)
export SEQ_OT_ITERS="${SEQ_OT_ITERS:-30}"  # P収束&勾配安定の最小
# ★D を一切更新しない (random init のまま凍結)。scale drift しないので reg=0.05 が終始有効。
export UPDATE_D="${UPDATE_D:-never}"
# adv も使わない (D 凍結なので adv 学習は無意味)。
export LAMBDA_GAN="${LAMBDA_GAN:-0}"
export NETG_CHUNK="${NETG_CHUNK:-16}"

# --- 実験名 ---
export NAME="${NAME:-synthRAD_${REGION}_ct2mr_sb_otdiv_geo_dfeat_flatten_frozenD}"

# --- wandb ---
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_TAGS_EXTRA="${WANDB_TAGS_EXTRA:-dfeat,flatten,frozenD}"

echo "JOB on $(hostname) at $(date)"
echo "SEQ_OT_COST_SPACE=${SEQ_OT_COST_SPACE}  UPDATE_D=${UPDATE_D}  LAMBDA_GAN=${LAMBDA_GAN}  NETG_CHUNK=${NETG_CHUNK}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- 学習スクリプト本体を実行 ---
bash run_train_synthRAD_ot_div_geo_gan.sh

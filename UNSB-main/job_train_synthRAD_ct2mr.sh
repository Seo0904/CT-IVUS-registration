#!/bin/bash
#============================================================
# Genkai 用 pjsub ジョブ: synthRAD CT->MR UNSB 学習
#   投入:  pjsub job_train_synthRAD_ct2mr.sh
#   ログ:  train_ct2mr.<JOBID>.out
#============================================================
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -N train_ct2mr

set -e

# --- conda 環境 ---
source ~/.bashrc 2>/dev/null || true
conda activate ctmri

# --- このジョブスクリプトのある場所 (= UNSB-main) ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# --- データ位置 (リポジトリ直下の data/ から自動算出。WORKSPACE_DIR = UNSB-main の親) ---
WORKSPACE_DIR="$(dirname "${SCRIPT_DIR}")"
export DATAROOT="${DATAROOT:-${WORKSPACE_DIR}/data/preprocessed/synthRAD_maskfree_full}"
export REGION="${REGION:-brain}"

# --- バッチサイズ (H100 94GB。投入時に BATCH_SIZE=256 pjsub ... で上書き可) ---
export BATCH_SIZE="${BATCH_SIZE:-128}"

# --- wandb (計算ノードからオンライン送信可を確認済み) ---
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"

echo "JOB on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- 学習スクリプト本体を実行 ---
bash run_train_synthRAD_ct2mr.sh
